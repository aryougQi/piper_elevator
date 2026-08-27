# Piper Elevator ROS 2 基础环境

这个仓库当前负责三件事：

1. 使用 Docker 提供 Ubuntu 22.04 + ROS 2 Humble 环境。
2. 下载 AgileX 官方 Piper 和 Pika ROS 2 源码。
3. 编译 Piper、Pika、MoveIt 2 和 RealSense ROS 2 功能包。

当前包含业务包 `piper_elevator_app`，以及基于 Gazebo Fortress 的
`piper_elevator_gazebo` 虚拟硬件包。后者复用官方 Piper、Pika、D405 描述和
官方 ROS/Gazebo 插件，只自定义组合、话题映射和可按压按钮场景。

## 目录

```text
piper_elevator/
├── docker/
├── ros2_ws/src/
│   ├── agx_arm_ros/
│   ├── pika_gripper_description/
│   ├── pika_ros/
│   ├── piper_elevator_app/
│   └── piper_elevator_gazebo/
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

该检查会确认 14 个功能包、业务检测节点、Pika/RealSense/Gazebo 启动入口，以及
Piper + 真实 Pika 模型的 MoveIt、ros2_control 和模拟控制器。它不会连接
真实设备。

Pika 的部分定位/遥操作启动文件还会引用 `pika_locator`。官方仓库没有提供它的源码，只在 `pika_ros/source/install.zip` 中附带了预编译版本；当前基础工程没有解压或加载这份约 360 MB 的硬件定位组件。等接入 Pika 定位基站时再单独启用，不影响夹爪、相机、数据工具和 Piper 仿真开发。

## 按钮检测业务节点

业务包位于 `ros2_ws/src/piper_elevator_app`。`button_detector` 使用专用 YOLOv10-S 电梯按钮模型，不再使用容易被反光和圆形图案干扰的 Hough 圆检测。运行时由 ONNX Runtime 的 CUDA 执行器在 NVIDIA GPU 上推理，由 OpenCV 完成图像预处理和 NMS，不依赖 PyTorch。默认参数 `inference_device: cuda` 会在 GPU 未成功加载时立即报错，防止静默退回 CPU；只有明确设成 `auto` 才允许回退。

实机调试使用的关键命令、问题原因和验证结果持续记录在项目根目录
`DEBUG_COMMANDS.md`。

检测器会先发布全部二维检测框，再通过类别一致性、IoU、中心位移和连续帧数锁定一个稳定目标。RGB-D 模式从目标框的中心区域采样深度，使用中位数和 MAD 排除空洞及飞点，再使用 RealSense 的内参和 `plumb_bob` 畸变参数反投影到相机光学坐标系。

模型文件应位于：

```text
ros2_ws/src/piper_elevator_app/models/elevator_buttons_yolov10s.onnx
```

模型来源、类别和跨场景准确率限制记录在 `models/README.md`。模型缺失时节点会直接报出期望路径，不会退回旧的 Hough 算法。

当前主相机为 RealSense D405（实机序列号 `315122272440`）。彩色图负责
YOLO 检测，对齐到彩色图的深度负责三维定位。启动命令：

```bash
docker compose run --rm piper_ros2 bash -lc '
  source /workspace/ros2_ws/install/setup.bash
  ros2 launch piper_elevator_app realsense_button_detector.launch.py
'
```

该启动文件带有跨容器单实例保护（项目容器使用 host IPC）。同一台主机上
已经有一套 RealSense 检测节点运行时，第二次启动会在打开相机前直接报错，
不会再令 D405 断开。需要查看图像或话题时，只启动 `rqt_image_view` 或
`ros2 topic echo`，不要重复执行上述 launch；需要重启时先在原启动终端按
`Ctrl+C`。

D405 彩色流和深度流均为 848×480@30 FPS，`align_depth` 将深度对齐到
彩色图。检测订阅和 RGB-D 同步队列都保持很小，来不及处理时丢弃旧帧
而不积压。节点会在订阅相机前预热两次 CUDA 推理。实机稳态检测约
21～26 FPS，平均处理约 34～39 ms，输入延迟不会持续增长。
节点每 5 秒输出一次实际 FPS、平均/最大处理时间和输入帧年龄。
没有订阅 `/button_detector/debug_image` 时会跳过调试图绘制和序列化。
调试图默认缩放为 424×240并跟随检测帧率发布，避免大尺寸 raw 图跨
Docker 容器传输时令 `rqt_image_view` 卡住；检测仍使用 848×480 原图，
按钮框和 `/button_pixel` 坐标也仍属于原图坐标系。

默认配置已设为 `use_depth: true`，输入为 `/camera/color/image_raw`、
`/camera/aligned_depth_to_color/image_raw` 和 `/camera/color/camera_info`。
Pika 鱼眼启动文件只作为旧方案保留，当前业务运行不会启动它。

主要输出：

```text
/button_pixel                 geometry_msgs/PointStamped
/button_pose                  geometry_msgs/PoseStamped
/button_detections            vision_msgs/Detection2DArray
/button_detection_valid       std_msgs/Bool
/button_detection_confidence  std_msgs/Float32
/button_detector/debug_image  sensor_msgs/Image
/button_selection             std_msgs/String  # 输入
/button_selected              std_msgs/String  # 当前选择反馈
```

`/button_detections` 始终包含当前帧所有通过置信度和 NMS 筛选的按钮框。坐标话题默认不发布；向 `/button_selection` 发送一个模型类别后，跟踪器只处理该类别，且只有它连续稳定、深度有效时才发布 `/button_pixel` 和 `/button_pose`。`/button_pixel.point.x/y` 是选中目标的平滑中心像素，`point.z` 是检测框平均半径近似值，不是深度；真实深度为 `/button_pose.pose.position.z`。

先从 `/button_detections` 的 `id` 查看类别（格式为 `类别:序号`），再选择按钮。例如选择 3 楼：

```bash
ros2 topic echo /button_detections --once
ros2 topic pub --once /button_selection std_msgs/msg/String "{data: '3'}"
ros2 topic echo /button_selected --once
ros2 topic echo /button_pose
```

功能按钮可使用模型类别 `up`、`down`、`open`、`close`。清除选择并立即停止坐标发布：

```bash
ros2 topic pub --once /button_selection std_msgs/msg/String "{data: clear}"
```

调试图像可这样查看：

```bash
ros2 run rqt_image_view rqt_image_view
```

然后在窗口中选择 `/button_detector/debug_image`。

RealSense D405 的 RGB-D 联合启动命令：

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

## 七、启动 Gazebo 虚拟硬件

镜像使用 ROS 2 Humble 对应的 Gazebo Fortress、`ros_gz` 和
`gz_ros2_control`。启动命令为：

```bash
./scripts/gazebo_hardware.sh
```

`./scripts/sim.sh` 是同一入口。无图形界面时使用：

```bash
./scripts/gazebo_hardware.sh gui:=false
```

这个入口只启动虚拟硬件：Piper、Pika、eye-in-hand D405、物理按钮、
`ros2_control` 和话题桥。它不会启动 MoveIt、RViz、YOLO、Planner、模拟
`/button_pose` 或自动动作。场景只包含一个有 4 mm 行程、弹簧回位和接触事件
的按钮，不模拟完整电梯、轿厢或门。

对外接口与实机保持一致：

```text
/camera/color/image_raw
/camera/aligned_depth_to_color/image_raw   # 32FC1，单位 m
/camera/color/camera_info
/piper_pika/joint_states
/arm_controller/follow_joint_trajectory
/pika_gripper_controller/follow_joint_trajectory
```

仿真额外提供测试状态：

```text
/elevator_button/pressed
/elevator_button/joint_states
/clock
```

运行无界面验收：

```bash
./scripts/check_gazebo.sh
```

该检查会验证三个控制器、相机尺寸和坐标系、米制深度、按钮回位，以及一条
0.1 rad 的安全轨迹。Gazebo 直接桥接 `32FC1` 深度；检测器对浮点深度按米
处理，对实机 `16UC1` 深度仍按毫米处理，不需要转换节点。

## 八、按需单独启动核心代码

虚拟硬件启动后，核心代码必须在其他终端显式启动：

```bash
ros2 launch piper_elevator_app piper_pika_moveit.launch.py \
  external_hardware:=true use_sim_time:=true

ros2 launch piper_elevator_app button_detector.launch.py use_sim_time:=true

ros2 launch piper_elevator_app button_approach_planner.launch.py \
  use_sim_time:=true simulation_mode:=true \
  camera_calibration_valid:=true allow_execution:=true
```

所有终端必须使用同一个 `ROS_DOMAIN_ID`。规划和执行仍由用户分别调用，不会
检测到按钮后自动移动：

```bash
ros2 service call /button_approach_planner/plan std_srvs/srv/Trigger "{}"
ros2 service call /button_approach_planner/execute std_srvs/srv/Trigger "{}"
ros2 service call /button_approach_planner/clear_plan std_srvs/srv/Trigger "{}"
```

旧的 `button_approach_sim.sh` 和 `button_approach_sim.launch.py` 只保留作
FakeSystem 回归诊断，不再是支持的仿真流程。实机继续使用独立的
`button_approach_real.launch.py` 和官方 Piper CAN 驱动；切换到实机不只是
换相机话题，还必须恢复真实控制代理、真实关节反馈、手眼标定、TCP、CAN、
限速和执行安全锁。

## 真机控制

虚拟硬件和真机不能在同一个 ROS 域同时运行：

```bash
# Gazebo 虚拟硬件
./scripts/gazebo_hardware.sh

# 真机（ROS_DOMAIN_ID=0）
./scripts/button_approach_real.sh
```

真机脚本当前默认处于安全联调状态：`auto_enable:=false`、
`hardware_commands_enabled:=false`、`publish_camera_tf:=false`、
`camera_calibration_valid:=false`、`allow_execution:=false`。因此可以读取真实
关节反馈并在 RViz 中规划，但不会向机械臂转发轨迹。Pika 串口反馈也是可选
的，需要时添加 `start_pika_driver:=true pika_serial_port:=/dev/ttyUSB60`；
当前任务不控制夹爪开合，驱动的控制入口保持断开。

完成 `tcp_link -> camera_link` 手眼标定、确认真实 TCP 和空载低速验证之后，
才能显式打开硬件命令和应用执行锁。即使解锁，真机仍为手动规划、手动执行，
不会使用仿真的自动执行模式。运行真机前还必须确认 CAN 接口、机械臂型号、
末端执行器、工作空间和急停状态。
