FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS cuda_runtime

FROM osrf/ros:humble-desktop

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

RUN sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' \
        -e 's|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' \
        /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        can-utils \
        ethtool \
        git \
        libboost-system-dev \
        libboost-timer-dev \
        libcurl4-openssl-dev \
        libeigen3-dev \
        libjsoncpp-dev \
        libopencv-dev \
        libpcl-dev \
        locales \
        python3-colcon-common-extensions \
        python3-h5py \
        python3-opencv \
        python3-pandas \
        python3-pip \
        python3-pytest \
        python3-requests \
        python3-rosdep \
        python3-scipy \
        python3-tqdm \
        python3-vcstool \
        ros-humble-controller-manager \
        ros-humble-gripper-controllers \
        ros-humble-gz-ros2-control \
        ros-humble-joint-state-broadcaster \
        ros-humble-joint-state-publisher-gui \
        ros-humble-joint-trajectory-controller \
        ros-humble-diagnostic-updater \
        ros-humble-librealsense2 \
        ros-humble-moveit \
        ros-humble-moveit-servo \
        ros-humble-robot-state-publisher \
        ros-humble-ros2-control \
        ros-humble-ros2-controllers \
        ros-humble-rosbag2-storage-mcap \
        ros-humble-ros-gz \
        ros-humble-topic-tools \
        ros-humble-trajectory-msgs \
        ros-humble-vision-msgs \
        ros-humble-xacro \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV LC_NUMERIC=en_US.UTF-8

# Keep the official ROS 2 image as the application base while importing the
# matching NVIDIA CUDA 12.8 and cuDNN 9 runtime libraries for ONNX Runtime.
COPY --from=cuda_runtime /usr/local/cuda-12.8 /usr/local/cuda-12.8
COPY --from=cuda_runtime /usr/lib/x86_64-linux-gnu/libcudnn* /usr/lib/x86_64-linux-gnu/
RUN ln -s /usr/local/cuda-12.8 /usr/local/cuda

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu

# Official AgileX Python SDK required by agx_arm_ros.
RUN git clone --depth 1 https://github.com/agilexrobotics/pyAgxArm.git /tmp/pyAgxArm \
    && pip3 install --no-cache-dir /tmp/pyAgxArm \
    && rm -rf /tmp/pyAgxArm

# GPU ONNX Runtime performs YOLO inference; OpenCV handles preprocessing/NMS.
RUN pip3 install --no-cache-dir \
        numpy==1.24.4 \
        onnxruntime-gpu==1.23.2

RUN rosdep update --rosdistro humble

COPY docker/entrypoint.sh /piper_entrypoint.sh
RUN chmod +x /piper_entrypoint.sh

WORKDIR /workspace/ros2_ws

ENTRYPOINT ["/piper_entrypoint.sh"]
CMD ["bash"]
