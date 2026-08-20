# Piper Elevator ROS 2 基础环境

这个仓库当前只负责两件事：

1. 使用 Docker 提供 Ubuntu 22.04 + ROS 2 Humble 环境。
2. 下载 AgileX 官方 Piper 和 Pika ROS 2 源码。
3. 编译 Piper、Pika、MoveIt 2 和 RealSense ROS 2 功能包。

当前包含第一个自研业务包 `piper_elevator_app`，已实现基于 YOLO ONNX 的电梯按钮检测、跨帧跟踪和 RGB-D 三维定位；暂不包含 Gazebo。

## 目录

```text
piper_elevator/
├── docker/
├── ros2_ws/src/
│   ├── agx_arm_ros/
│   ├── pika_ros/
│   └── piper_elevator_app/
├── scripts/
└── docker-compose.yml
```

`agx_arm_ros` 和 `pika_ros` 均由安装脚本从 AgileX 官方仓库的 `ros2` 分支获取，目录默认不提交到本项目 Git。

## 一、准备 Docker

确认 Docker 和 Compose 可用：

```bash
docker --version
docker compose version
docker run --rm hello-world
```

## 二、配置环境

```bash
cp .env.example .env
```

如果不使用代理，将 `.env` 中的 `HTTP_PROXY` 和 `HTTPS_PROXY` 留空。两台 ROS 2 电脑通信时，必须使用相同的 `ROS_DOMAIN_ID`。

## 三、下载官方 Piper 和 Pika ROS 2

```bash
./scripts/install_official_ros2.sh
```

脚本会获取：

```text
ros2_ws/src/agx_arm_ros
ros2_ws/src/pika_ros
```

两个仓库都使用官方 `ros2` 分支并初始化全部子模块。Piper 使用的是官方 `agx_arm_ros`，不是旧的 `piper_ros`。

## 四、构建 Docker 镜像

```bash
docker compose build
```

镜像包含 ROS 2 Humble Desktop、MoveIt 2、RViz 2、ros2_control、CAN 工具、RealSense 依赖、官方 pyAgxArm SDK，以及 NVIDIA CUDA 12.8、cuDNN 9 和 GPU 版 ONNX Runtime。运行检测节点需要宿主机已安装 NVIDIA Container Toolkit。

## 五、编译完整 ROS 2 工作空间

```bash
./scripts/build.sh
```

基础构建会跳过当前 Humble 软件源中未发布的两个可选运行依赖：`moveit_ros_perception` 和 `warehouse_ros_mongo`。它们不影响基础 Piper 模型、RViz 显示和 MoveIt 规划演示，后续需要三维占据地图或规划数据库时再单独补充。

`build.sh` 会对 `ros2_ws/src` 执行 rosdep，并编译 Piper、Pika、Pika 数据工具和随 Pika 提供的 RealSense ROS 2 功能包。Pika 上游的 `serial` 依赖声明会被跳过，因为当前 ROS 2 CMake 文件并未启用该库；RealSense 的 `launch_pytest` 也会跳过，因为它只用于测试且当前 Humble 软件源没有发布。

编译产物会生成在：

```text
ros2_ws/build/
ros2_ws/install/
ros2_ws/log/
```

编译后可运行无硬件检查：

```bash
./scripts/check.sh
```

该检查会确认 12 个功能包、业务检测节点、Pika/RealSense 启动入口，以及 Piper 的 MoveIt、ros2_control 和模拟控制器。它不会连接真实设备。

Pika 的部分定位/遥操作启动文件还会引用 `pika_locator`。官方仓库没有提供它的源码，只在 `pika_ros/source/install.zip` 中附带了预编译版本；当前基础工程没有解压或加载这份约 360 MB 的硬件定位组件。等接入 Pika 定位基站时再单独启用，不影响夹爪、相机、数据工具和 Piper 仿真开发。

## 按钮检测业务节点

业务包位于 `ros2_ws/src/piper_elevator_app`。`button_detector` 使用专用 YOLOv10-S 电梯按钮模型，不再使用容易被反光和圆形图案干扰的 Hough 圆检测。运行时由 ONNX Runtime 的 CUDA 执行器在 NVIDIA GPU 上推理，由 OpenCV 完成图像预处理和 NMS，不依赖 PyTorch。默认参数 `inference_device: cuda` 会在 GPU 未成功加载时立即报错，防止静默退回 CPU；只有明确设成 `auto` 才允许回退。

检测器会先发布全部二维检测框，再通过类别一致性、IoU、中心位移和连续帧数锁定一个稳定目标。RGB-D 模式从目标框的中心区域采样深度，使用中位数和 MAD 排除空洞及飞点，然后利用相机内参投影到相机光学坐标系。

模型文件应位于：

```text
ros2_ws/src/piper_elevator_app/models/elevator_buttons_yolov10s.onnx
```

模型来源、类别和跨场景准确率限制记录在 `models/README.md`。模型缺失时节点会直接报出期望路径，不会退回旧的 Hough 算法。

当前已识别的相机是 `/dev/video6`，可这样启动相机和检测节点：

```bash
docker compose run --rm piper_ros2 bash -lc '
  source /workspace/ros2_ws/install/setup.bash
  ros2 launch piper_elevator_app pika_button_detector.launch.py fisheye_port:=6
'
```

主要输出：

```text
/button_pixel                 geometry_msgs/PointStamped
/button_detections            vision_msgs/Detection2DArray
/button_detection_valid       std_msgs/Bool
/button_detection_confidence  std_msgs/Float32
/button_detector/debug_image  sensor_msgs/Image
```

`/button_detections` 包含当前帧所有通过置信度和 NMS 筛选的按钮框；这是推荐给其他视觉程序使用的标准接口。`/button_pixel.point.x/y` 是稳定目标的平滑中心像素，`point.z` 是目标框平均半径近似值。`/button_detection_valid` 只有在目标连续稳定、且 RGB-D 模式的深度有效时才为 `true`。

调试图像可这样查看：

```bash
ros2 run rqt_image_view rqt_image_view
```

然后在窗口中选择 `/button_detector/debug_image`。

使用 RealSense D405 时，可直接启动 RGB-D 联合节点：

```bash
docker compose run --rm piper_ros2 bash -lc '
  source /workspace/ros2_ws/install/setup.bash
  ros2 launch piper_elevator_app realsense_button_detector.launch.py
'
```

该启动文件会开启彩色图、深度图和 `aligned_depth_to_color`，检测成功后同时发布二维 `/button_pixel` 和相机光学坐标系下的三维 `/button_pose`。

## 六、进入开发容器

```bash
./scripts/shell.sh
```

进入后检查官方功能包：

```bash
ros2 pkg list | grep agx_arm
ros2 pkg list | grep -E 'pika|data_tools|sensor_tools|realsense2'
```

## 七、运行 MoveIt + RViz 仿真

首次运行 RViz 前，在宿主机执行：

```bash
xhost +si:localuser:root
```

然后运行：

```bash
./scripts/sim.sh
```

该模式不连接真实机械臂，只验证 Piper 模型、MoveIt 规划和 RViz 显示。

由于基础环境暂未安装 `moveit_ros_perception`，启动日志可能提示无法加载 `PointCloudOctomapUpdater`。这是预期的非致命提示：MoveGroup、FakeSystem、控制器和轨迹规划仍可正常使用。等后期加入深度相机和三维场景更新时再补充该插件。

## 真机控制（后续）

连接并激活 CAN 后，可在容器中运行：

```bash
ros2 launch agx_arm_ctrl start_single_agx_arm_moveit.launch.py \
  can_port:=can0 \
  arm_type:=piper \
  effector_type:=none \
  auto_control_gate:=true
```

在运行真机命令前，应先确认 CAN 接口、机械臂型号、末端执行器和急停状态。
