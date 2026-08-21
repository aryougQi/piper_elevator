# Piper Elevator ROS 2 基础环境

这个仓库当前负责三件事：

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
│   ├── pika_gripper_description/
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

该检查会确认 13 个功能包、业务检测节点、Pika/RealSense 启动入口，以及
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

## 八、按钮坐标转换和 MoveIt 靠近仿真

当前已实现 `/button_pose` 从 `camera_color_optical_frame` 到 `base_link` 的
TF 转换，并沿相机光轴在按钮前方生成安全靠近位姿。默认距离为 20 cm，当前
阶段不执行按压和视觉伺服。MoveIt 规划和执行采用两个独立服务，检测到坐标
后不会自动移动。

首次打开 RViz 前，在宿主机执行：

```bash
xhost +si:localuser:root
```

启动完整 FakeSystem 仿真、真实 Pika 夹爪模型、eye-in-hand 模拟相机和模拟
按钮：

```bash
cd /home/qi/Project/piper_elevator
./scripts/button_approach_sim.sh
```

仿真默认 `auto_execute:=true`：MoveIt 和模拟按钮就绪后自动规划并执行一次，
只会驱动独立 ROS 域中的 FakeSystem。需要先检查 RViz 再手动执行时使用：

```bash
./scripts/button_approach_sim.sh auto_execute:=false
```

该脚本固定使用 `ROS_DOMAIN_ID=42`，避免与正在运行的真实相机节点串话。新开
一个宿主机终端进入同一个仿真域：

```bash
cd /home/qi/Project/piper_elevator
ROS_DOMAIN_ID=42 ./scripts/shell.sh
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
```

查看相机坐标转换结果和靠近目标：

```bash
ros2 topic echo /button_pose_base --once
ros2 topic echo /button_approach_pose --once
ros2 topic echo /button_approach/status
```

先规划并在 RViz 中检查，然后明确执行虚拟轨迹：

```bash
ros2 service call /button_approach_planner/plan std_srvs/srv/Trigger {}
ros2 service call /button_approach_planner/execute std_srvs/srv/Trigger {}
```

模拟按钮固定在更远的 `base_link (0.55, 0, 0.30)m`，20 cm 靠近点会根据
当前相机光轴计算，初始约为 `(0.351, 0, 0.283)m`。仿真已使用 AgileX 官方
`agx_arm_sim` 仓库中的 Pika2 彩色 DAE 网格和尺寸，不再使用
`agx_gripper` 替代模型。模型来源固定为官方提交
`f8cd8b147c75d59e14f90fb0646770eefa268ed0`。夹爪以官方驱动使用的
`center_joint` 表示 0–98 mm 总开度，并加载独立的
`pika_gripper_controller`；D405 采用 eye-in-hand 方式固定到 `tcp_link`。
可以通过 launch 参数修改目标和 TCP：

```bash
./scripts/button_approach_sim.sh \
  button_base_x:=0.55 button_base_y:=0.0 button_base_z:=0.30 \
  pika_tcp_offset:='[0.006, 0.0, 0.189, 0.0, 0.0, 0.0]' \
  auto_execute:=true
```

真机使用独立的 `button_approach_real.launch.py`。它复用同一套 Piper+Pika
URDF、SRDF 和 MoveIt 控制器配置，并将官方 Piper CAN 驱动发布的六轴反馈与
Pika `center_joint` 合并到 `/piper_pika/joint_states`。MoveIt 的关节轨迹仍按
官方 `agx_arm_ros` 的方式，经 `/control/joint_states` 转交 CAN 驱动。

## 真机控制

仿真和真机使用两个不同脚本，不能同时运行：

```bash
# 仿真（ROS_DOMAIN_ID=42）
./scripts/button_approach_sim.sh

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
