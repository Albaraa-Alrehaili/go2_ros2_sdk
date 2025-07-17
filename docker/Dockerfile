FROM ros:humble-ros-base

# Install basic requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    lsb-release \
    python3-pip \
    portaudio19-dev \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libopencv-dev \
    ffmpeg \
    ros-humble-rqt-gui \
    ros-humble-rqt-image-view \
    ros-humble-rviz2 \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-vision-msgs \
    ros-humble-image-transport \
    ros-humble-cv-bridge \
    ros-humble-image-tools \
    ros-humble-image-view \
    ros-humble-compressed-image-transport \
    ros-humble-camera-info-manager \
    ros-humble-tf-transformations \
    ros-humble-vision-opencv \
    ros-humble-diagnostic-updater \
    software-properties-common


# Create workspace
WORKDIR /ros2_ws

# Clone the repository with submodules
COPY . /ros2_ws/src

# Install Python requirements (with numpy constraint)
RUN cd src && pip install -r requirements.txt

# Install ROS dependencies
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    rosdep install --from-paths src --ignore-src -r -y

# Build the workspace
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    colcon build

# Source ROS2 and workspace in bashrc
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc

# Set environment variables
ENV ROBOT_IP=""
ENV CONN_TYPE="webrtc"
ENV WEBRTC_SERVER_HOST="0.0.0.0"
ENV WEBRTC_SERVER_PORT="9991"

# Remove apt cache
RUN rm -rf /var/lib/apt/lists/*

# add /ros2_ws to the entrypoint
RUN sed --in-place --expression \
      '$isource "/ros2_ws/install/setup.bash"' \
      /ros_entrypoint.sh

CMD ["ros2", "launch", "go2_robot_sdk", "robot.launch.py"]