# Piper 电梯按钮项目：AI 接手上下文

> 最后更新：2026-08-20  
> 项目路径：`/home/qi/Project/piper_elevator`  
> 当前阶段：完成 ROS 2/Docker 基础环境和 GPU 按钮视觉检测；尚未实现机械臂业务动作链。

## 1. 项目目标

使用 AgileX Piper 机械臂和 Pika 夹爪完成电梯按钮操作。当前设想的业务流程是：

1. 使用 Pika 夹爪上的相机识别电梯按钮。
2. 使用 MoveIt 规划到按钮前方的预接触位置。
3. 使用视觉伺服进行近距离对准。
4. 使用笛卡尔路径向前按压按钮，然后退回。

现阶段只考虑“检测并按下一个按钮”，暂不实现楼层选择，也暂不处理初始画面找不到按钮时的搜索策略。

## 2. 当前硬件和系统

- 宿主系统：Ubuntu 22.04.5 LTS，x86_64。
- GPU：NVIDIA GeForce RTX 3090，24 GB 显存。
- 宿主 NVIDIA 驱动：580.173.02，`nvidia-smi` 正常。
- Docker Engine：29.7.2，Docker Compose：5.5.0。
- NVIDIA Container Toolkit 已安装，Docker 容器可以访问 RTX 3090。
- ROS 2：官方 ROS 2 Humble Desktop，运行于 Docker 中。
- 机械臂：AgileX Piper。
- 夹爪/相机：AgileX Pika。相机头外观上有三个镜头；当前测试使用其中一个 UVC 鱼眼彩色流。
- 当前识别到的测试相机设备为 `/dev/video6`。重新插拔后设备编号可能变化。
- 当前相机链路按 RGB 相机使用，尚未确认它能输出与彩色图对齐的深度图。

开发期间曾手持同型号夹爪/相机采集和调试。最终安装到机械臂后，检测模型通常仍可复用，但必须重新标定相机内参，并完成相机到末端、末端到机械臂基座的外参/TF 标定。

## 3. 项目目录

```text
piper_elevator/
├── docker/
│   ├── Dockerfile
│   └── entrypoint.sh
├── docker-compose.yml
├── ros2_ws/
│   ├── build/
│   ├── install/
│   ├── log/
│   └── src/
│       ├── agx_arm_ros/          # AgileX 官方 Piper ROS 2 仓库
│       ├── pika_ros/             # AgileX 官方 Pika ROS 2 仓库
│       └── piper_elevator_app/   # 本项目业务包
├── scripts/
├── README.md
└── AI_PROJECT_CONTEXT.md
```

工作空间只使用 `ros2_ws/src`。项目根目录下多余的旧 `src` 已删除。

## 4. 官方 ROS 2 代码

`scripts/install_official_ros2.sh` 下载以下官方仓库的 `ros2` 分支，并初始化子模块：

- `ros2_ws/src/agx_arm_ros`：Piper 驱动、描述、消息、ros2_control 和 MoveIt 配置。
- `ros2_ws/src/pika_ros`：Pika 夹爪、相机、数据和传感器工具。

目前共构建并检查 12 个 ROS 2 功能包：

```text
agx_arm_ctrl
agx_arm_description
agx_arm_moveit
agx_arm_msgs
data_msgs
data_tools
piper_elevator_app
pika_remote_agx_arm
realsense2_camera
realsense2_camera_msgs
realsense2_description
sensor_tools
```

Pika 上游引用的 `pika_locator` 没有公开源码，只在官方仓库中带有预编译压缩包。当前没有启用它，不影响现阶段的相机、夹爪和按钮检测开发。

## 5. Docker 和 GPU 环境

最终应用镜像名为：

```text
piper-elevator:humble
```

镜像以 `osrf/ros:humble-desktop` 为应用基础，并从 NVIDIA 官方镜像导入 GPU 运行库：

```text
nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
```

主要视觉依赖：

- CUDA 12.8
- cuDNN 9
- `onnxruntime-gpu==1.23.2`
- `numpy==1.24.4`
- OpenCV

`docker-compose.yml` 已配置 `gpus: all`。按钮检测默认强制使用 `CUDAExecutionProvider`，GPU 加载失败时会直接报错，不会静默回退到 CPU。只有把参数 `inference_device` 明确设成 `auto` 时才允许自动回退。

宿主机 GID 110 是 `render` 组。容器启动时偶尔出现：

```text
groups: cannot find name for group ID 110
```

这只是容器中缺少 GID 110 的文字名称，不是权限错误。已经确认容器仍可访问 `/dev/dri/renderD128` 和 RTX 3090，不影响 ROS 2、相机或 CUDA。

## 6. 已实现的按钮检测

业务包位于 `ros2_ws/src/piper_elevator_app`。

主要文件：

- `piper_elevator_app/detector_core.py`：ONNX 推理、letterbox、YOLO 输出解码、NMS、跨帧跟踪、稳健深度采样和像素反投影。
- `piper_elevator_app/yolo_button_detector.py`：ROS 2 节点、参数、订阅、发布和调试图绘制。
- `config/button_detector.yaml`：相机话题、模型、阈值、ROI、深度和跟踪参数。
- `launch/pika_button_detector.launch.py`：启动 Pika 鱼眼相机和检测节点。
- `launch/realsense_button_detector.launch.py`：启动 RealSense 对齐深度和检测节点。
- `models/elevator_buttons_yolov10s.onnx`：当前唯一运行时模型。
- `test/test_button_detector.py`：模型解码、跟踪、深度过滤和像素投影单元测试。

早期的 Hough 圆检测和 YOLOv8-World 方案已废弃并从业务代码中删除，不应恢复。当前只保留专用电梯按钮 YOLOv10 模型。

### 6.1 模型

- 模型：YOLOv10-S 电梯按钮检测模型。
- 来源：`isharadilshanra/YOLOv10-Elevator-Button-Detection`。
- 运行格式：ONNX，不在运行时依赖 PyTorch 或 Ultralytics。
- 文件大小：30,192,276 字节。
- 输入：固定 `1 × 3 × 1280 × 1280`。
- 输出：`1 × 300 × 6`，每项是 `x1, y1, x2, y2, confidence, class_id`。
- 类别：模型元数据中包含 368 个电梯按钮/标识类别。
- 默认置信度阈值：`0.60`。
- `target_classes: ['*']`，当前保留模型中的全部电梯相关类别。
- 模型许可证信息需要在商业发布前进一步确认；ONNX 元数据显示 Ultralytics AGPL-3.0。

当前测试图在最终 ONNX/GPU 链路中的结果：

```text
up    confidence ≈ 0.94
down  confidence ≈ 0.65
```

注意：`models/README.md` 中曾记录较早测试的 `down ≈ 0.20`，该数字已经过时；最终当前模型的实测结果约为 0.65。

### 6.2 检测流程

1. 从 ROS 2 `sensor_msgs/Image` 接收 BGR 彩色图。
2. 可选使用归一化 ROI 裁剪按钮面板区域。
3. 保持宽高比缩放并填充到 1280×1280（letterbox）。
4. BGR 转 RGB、归一化到 0～1，并整理成 NCHW 张量。
5. 使用 ONNX Runtime `CUDAExecutionProvider` 在 RTX 3090 上执行 YOLOv10。
6. 按置信度、目标类别和 NMS 过滤检测框。
7. 使用类别一致性、IoU、中心位移、连续帧数和指数平滑进行跨帧跟踪。
8. 发布所有检测框、稳定目标中心、有效状态、置信度和调试图。
9. RGB-D 模式对框中心区域取深度中位数，用 MAD 排除空洞和飞点，再使用内参投影到相机光学坐标系。

默认要求目标连续稳定 5 帧，允许短暂漏检 2 帧。阈值和跟踪参数均可通过 YAML 调整。

## 7. ROS 2 输入和输出

### 7.1 默认输入

Pika RGB 模式：

```text
/camera_fisheye/color/image_raw       sensor_msgs/Image
/camera_fisheye/color/camera_info     sensor_msgs/CameraInfo
```

默认 `use_depth: false`，因此 Pika RGB 模式不发布三维按钮位置。

RealSense RGB-D 模式：

```text
/camera/color/image_raw                    sensor_msgs/Image
/camera/aligned_depth_to_color/image_raw   sensor_msgs/Image
/camera/color/camera_info                  sensor_msgs/CameraInfo
```

### 7.2 输出话题

```text
/button_detections             vision_msgs/Detection2DArray
/button_pixel                  geometry_msgs/PointStamped
/button_pose                   geometry_msgs/PoseStamped
/button_detection_valid        std_msgs/Bool
/button_detection_confidence   std_msgs/Float32
/button_detector/debug_image   sensor_msgs/Image
```

- `/button_detections`：当前帧所有通过阈值和 NMS 的按钮框，推荐给其他业务节点使用。
- `/button_pixel`：稳定目标中心；`point.x/y` 是像素坐标，`point.z` 是检测框平均半径近似值，不是深度。
- `/button_pose`：只有 RGB-D 模式深度有效时才发布，坐标位于相机光学坐标系。
- `/button_detection_valid`：目标是否满足连续稳定要求；RGB-D 模式还要求有效深度和内参。
- `/button_detection_confidence`：当前目标置信度。
- `/button_detector/debug_image`：带检测框和状态文字的调试图像。

## 8. 坐标转换完成情况

已经完成：

- 检测框中心像素坐标。
- 使用相机内参和有效深度，将 `(u, v, depth)` 反投影成相机光学坐标系 `(x, y, z)`。
- 深度中心区域中位数和 MAD 异常值过滤。

尚未完成：

- Pika RGB 相机没有可用深度时的真实三维位置估计。
- 相机最终现场内参标定。
- 相机到夹爪/末端执行器的手眼外参。
- 相机坐标系到 `tool0`、`base_link` 的 TF。
- 按钮表面法向估计和预接触位姿生成。

所以当前 `/button_pose` 只是 RGB-D 相机坐标系位置，不能直接作为 Piper 的目标位姿。

## 9. 性能

RTX 3090、固定 1280×1280 输入：

```text
ONNX 模型纯推理：
  平均约 15.99 ms
  约 62.5 FPS

完整检测链路（实拍图）：
  包含 letterbox、张量转换、GPU 推理、解码和 NMS
  平均约 43.6 ms
  中位数约 43.0 ms
  约 22.9 FPS
```

GPU 切换前 CPU 推理平均约 854 ms、约 1.17 FPS，模型推理部分约提升 53 倍。

22 FPS 左右足够完成按钮检测、MoveIt 粗定位和低速视觉伺服。最终视觉伺服建议先将末端速度限制为 2～5 cm/s，并优先保证处理最新帧而不是积压旧帧。后续可把订阅队列设为 1、允许关闭调试图、使用 ROI 或继续优化预处理。

## 10. 常用命令

### 10.1 进入容器

```bash
cd /home/qi/Project/piper_elevator
./scripts/shell.sh
```

Docker 镜像和 ROS 2 包目前已经构建完成。

### 10.2 启动 Pika 相机和检测

容器内：

```bash
ros2 launch piper_elevator_app pika_button_detector.launch.py fisheye_port:=6
```

如果 `/dev/video6` 变化，应先确认实际设备，再调整 `fisheye_port`。

### 10.3 查看调试图

宿主机首次允许 root 容器访问 X11：

```bash
xhost +si:localuser:root
```

另开一个容器终端：

```bash
cd /home/qi/Project/piper_elevator
./scripts/shell.sh
ros2 run rqt_image_view rqt_image_view
```

选择 `/button_detector/debug_image`。Qt 已配置软件渲染，以绕过 Nouveau/GLX 不匹配。`rqt_image_view` 本身可能卡顿，不能只凭窗口流畅度判断检测帧率。

### 10.4 查看话题和帧率

```bash
ros2 topic list
ros2 topic echo /button_detections
ros2 topic echo /button_pixel
ros2 topic echo /button_detection_valid
ros2 topic hz /button_detector/debug_image
ros2 topic hz /button_detections
```

### 10.5 何时重新构建

- 只修改已有 Python 源文件：当前使用 `--symlink-install`，通常重启节点即可。
- 修改 launch、YAML、`setup.py`、接口或新增文件：运行 `./scripts/build.sh`。
- 修改 Dockerfile、系统包或 Python 依赖：运行 `docker compose build`，然后退出旧容器并重新进入。

### 10.6 完整无硬件检查

宿主机：

```bash
cd /home/qi/Project/piper_elevator
./scripts/check.sh
```

当前检查结果：

```text
6 个 piper_elevator_app 单元测试全部通过
12 个 ROS 2 包可用
Pika 和 RealSense launch 文件可加载
YOLO ONNX 模型实际使用 CUDAExecutionProvider 完成推理
Piper MoveIt、ros2_control 和模拟控制器运行正常
```

### 10.7 无真机 MoveIt 演示

```bash
cd /home/qi/Project/piper_elevator
xhost +si:localuser:root
./scripts/sim.sh
```

这不是 Gazebo 物理仿真，而是 MoveIt + RViz + mock ros2_control，用于验证模型、规划和控制接口。

## 11. 已完成与未完成

### 已完成

- Docker、ROS 2 Humble 开发容器和官方 Piper/Pika ROS 2 源码。
- Piper MoveIt + RViz + mock controller 无真机运行。
- NVIDIA Container Toolkit、RTX 3090 透传、CUDA 12.8、cuDNN 9 和 GPU ONNX Runtime。
- 专用 YOLOv10 电梯按钮模型。
- RGB 检测、所有检测框发布、调试图和跨帧稳定目标。
- RGB-D 深度过滤与相机坐标反投影代码。
- ROS 2 launch、YAML 参数和单元测试。
- GPU 性能及当前实拍图识别结果验证。

### 未完成

- 真机 Piper/Pika 控制测试。
- 最终安装状态的相机内参和手眼标定。
- 相机坐标到机械臂基座坐标的 TF。
- 从多个检测结果中选择具体业务按钮。
- 预接触位姿生成和 MoveIt 业务规划节点。
- 视觉伺服控制节点。
- 笛卡尔按压和退回。
- 力/接触检测、限速、碰撞、超时和撤回保护。
- 完整状态机和“找不到按钮”恢复策略。
- Gazebo/Isaac 等物理仿真；目前不是必需项。

## 12. 推荐下一步

1. 在最终机械臂安装位置固定相机，确认稳定话题、分辨率和实际帧率。
2. 标定内参；若没有深度，决定使用已知按钮尺寸单目估距、额外深度相机或近距离视觉伺服。
3. 完成 eye-in-hand 手眼标定并发布相机到末端的 TF。
4. 新建目标选择节点，从 `/button_detections` 选择按钮。
5. 生成机械臂 `base_link` 下的预接触位姿。
6. 使用 MoveIt 到达按钮前方约 10～15 cm。
7. 使用 MoveIt Servo 或自定义图像视觉伺服低速对准。
8. 实现带限速、距离、超时和撤回保护的笛卡尔按压。

没有深度和手眼标定前，不应把二维像素直接传给机械臂执行三维运动。

## 13. 给后续 AI 的约束

- 默认不连接或驱动真实机械臂，除非用户明确要求并确认安全条件。
- 不要恢复已废弃的 Hough 圆检测、YOLOv8-World 或旧模型。
- 当前运行时模型只使用 `elevator_buttons_yolov10s.onnx`。
- 保持默认置信度阈值 `0.60`，除非有实测数据支持修改。
- 保持 GPU 为默认推理设备，不要静默回退 CPU。
- 不要假设 `/dev/video6` 永远不变，应先枚举设备和 ROS 话题。
- 不要把 `/button_pixel.point.z` 当作真实深度。
- 不要把相机坐标系的 `/button_pose` 直接当作 `base_link` 目标。
- 真机运动前先完成坐标系、限速、碰撞、超时、急停和撤回设计。
- 优先使用 ROS 2 标准消息和 MoveIt 2 接口；自研工具包后期再添加。
- 修改代码后运行对应单元测试；修改 Docker/GPU 后运行 `./scripts/check.sh`。
