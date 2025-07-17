"""Detects COCO objects in image and publishes in ROS 2.

Subscribes to /camera/image_raw and publishes Detection2DArray on
/detected_objects. Optionally publishes an annotated image on /annotated_image.
Uses PyTorch FasterRCNN MobileNet model from torchvision.
"""

import collections
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)
from cv_bridge import CvBridge

import torch
from torchvision.models import detection as detection_model
from torchvision.utils import draw_bounding_boxes

Detection = collections.namedtuple("Detection", "label bbox score")


class CocoDetectorNode(Node):
    """Detects COCO objects in image and publishes results."""

    def __init__(self):
        super().__init__("coco_detector_node")

        # Declare parameters
        self.declare_parameter("device", "cpu")
        self.declare_parameter("detection_threshold", 0.9)
        self.declare_parameter("publish_annotated_image", True)

        self.device = self.get_parameter("device").value
        self.detection_threshold = self.get_parameter("detection_threshold").value
        publish_image = self.get_parameter("publish_annotated_image").value

        # Subscribe to image topic (best‑effort QoS matches driver)
        qos_best_effort = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.subscription = self.create_subscription(
            Image, "/camera/image_raw", self.listener_callback, qos_best_effort
        )

        # Publishers
        self.detected_objects_pub = self.create_publisher(
            Detection2DArray, "detected_objects", 10
        )
        self.annotated_image_pub = (
            self.create_publisher(Image, "annotated_image", 10)
            if publish_image
            else None
        )

        # Model
        self.model = detection_model.fasterrcnn_mobilenet_v3_large_320_fpn(
            weights="DEFAULT"
        ).to(self.device)
        self.model.eval()
        self.class_labels = (
            detection_model.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT.meta[
                "categories"
            ]
        )

        self.bridge = CvBridge()
        self.get_logger().info("COCO detector node ready")

    # Utility: convert one detection to ROS 2 Detection2D
    def mobilenet_to_ros2(self, det: Detection, header):
        det2d = Detection2D()
        det2d.header = header

        hyp = ObjectHypothesis()
        hyp.class_id = self.class_labels[det.label]
        hyp.score = float(det.score)

        hyp_pose = ObjectHypothesisWithPose()
        hyp_pose.hypothesis = hyp
        det2d.results.append(hyp_pose)

        bb = BoundingBox2D()
        bb.center.position.x = float((det.bbox[0] + det.bbox[2]) / 2)
        bb.center.position.y = float((det.bbox[1] + det.bbox[3]) / 2)
        bb.center.theta = 0.0
        bb.size_x = float(det.bbox[2] - det.bbox[0])
        bb.size_y = float(det.bbox[3] - det.bbox[1])
        det2d.bbox = bb
        return det2d

    # Publish annotated image (if enabled)
    def publish_annotated_image(self, detections, header, image_chw):
        if detections:
            boxes = torch.stack([d.bbox for d in detections])
            labels = [self.class_labels[d.label] for d in detections]
            annotated = draw_bounding_boxes(
                torch.from_numpy(image_chw).to(torch.uint8), boxes, labels, colors="yellow"
            )
        else:
            annotated = torch.from_numpy(image_chw).to(torch.uint8)

        img_msg = self.bridge.cv2_to_imgmsg(
            annotated.cpu().numpy().transpose(1, 2, 0), encoding="rgb8"
        )
        img_msg.header = header
        self.annotated_image_pub.publish(img_msg)

    # Image callback
    def listener_callback(self, msg: Image):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        img_chw = cv_img.transpose((2, 0, 1))  # HWC → CHW

        batch = np.expand_dims(img_chw, 0).astype(np.float32) / 255.0
        tensor = torch.from_numpy(batch).to(self.device)

        outputs = self.model(tensor)[0]  # Single‑image batch

        detections = [
            Detection(lbl, box, score)
            for lbl, box, score in zip(
                outputs["labels"], outputs["boxes"], outputs["scores"]
            )
            if score.item() >= self.detection_threshold
        ]

        det_array = Detection2DArray()
        det_array.header = msg.header
        det_array.detections = [self.mobilenet_to_ros2(d, msg.header) for d in detections]
        self.detected_objects_pub.publish(det_array)

        if self.annotated_image_pub:
            self.publish_annotated_image(detections, msg.header, img_chw)


def main():
    rclpy.init()
    node = CocoDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()