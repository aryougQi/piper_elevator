# Piper Elevator 项目交接记录

## 当前版本与目标

当前代码来自 GitHub `main` 分支，版本提交为 `af44657 (v2_version)`。项目目标是在 Piper 机械臂和 Pika 夹爪上完成电梯按钮任务：从相机识别目标按钮，先进行安全的粗定位，再用视觉 Servo 精定位，执行按压，最后回到安全初始姿态。Gazebo 仿真使用与真实链路相同的业务节点和话题，区别只在底层控制、相机和接触反馈由 Gazebo 提供。

主要目录：

- `ros2_ws/src/piper_elevator_app`：按钮检测、目标选择、粗定位、视觉 Servo、按压和任务状态机。
- `ros2_ws/src/piper_elevator_gazebo`：Gazebo Fortress 世界、Piper/Pika 模型、相机、控制器和 ROS-Gazebo 桥接。
- `scripts`：Docker、构建、仿真和任务启动脚本。
- `config`：检测、规划、Servo、按压和任务状态机参数。
- `ros2_ws/diagnostics`：调试脚本、验收记录和诊断数据。

## 一键仿真入口

```bash
ROS_DOMAIN_ID=42 ./scripts/elevator_task.sh \
  simulation_mode:=true gazebo_gui:=true use_rviz:=true
```

`elevator_task.launch.py` 会组合启动：

1. `piper_elevator_gazebo/gazebo_hardware.launch.py`：Gazebo、机器人模型、相机、控制器和桥接。
2. `button_detector.launch.py`：YOLO 按钮检测和 RGB-D 目标发布。
3. `piper_pika_moveit.launch.py`：MoveIt、ros2_control、MoveIt Servo、仿真 Servo 适配器和 RViZ。
4. `button_approach_planner.launch.py`：粗定位规划和执行服务。
5. `button_visual_servo.launch.py`：视觉精定位闭环。
6. `button_press.launch.py`：接触检测、按压轨迹和 Servo/按压交接。
7. `elevator_task_manager`：把上述模块编排成一次完整任务。

RQt 图像查看需要另开容器终端执行 `ros2 run rqt_image_view rqt_image_view`，常用话题是 `/camera/color/image_raw` 和 `/button_detector/debug_image`。

## 模块逻辑

### Gazebo 与硬件抽象

Gazebo 世界 `button_press.sdf` 提供电梯面板、按钮碰撞/接触传感器和机器人。`piper_pika_gazebo.urdf.xacro` 定义 Piper、Pika 和夹爪相机。`hardware_bridge.yaml` 将 Gazebo 图像、深度、相机内参、关节状态和按钮接触消息桥接到 ROS 2；`gazebo_controllers.yaml` 提供关节状态广播、机械臂控制器和夹爪控制器。

这层的输出包括：

- `/camera/color/image_raw`
- `/camera/aligned_depth_to_color/image_raw`
- `/camera/color/camera_info`
- `/piper_pika/joint_states`
- `/elevator_button/button_<name>/contacts`
- `/clock`

### 按钮检测与选择

`button_detector` 使用 ONNX YOLO 模型产生 `/button_detections`，并根据 RGB-D 深度采样、表面拟合和稳定帧规则发布：

- `/button_pose`：相机坐标系下的三维目标位姿。
- `/button_surface_pose`：用于接近和 Servo 的表面位姿。
- `/button_pixel`：二维目标中心。
- `/button_detection_valid`、`/button_detection_confidence`：检测质量。
- `/button_detector/debug_image`：可视化调试图。

选择输入是 `/button_selection`，确认输出是瞬态本地的 `/button_selected`。可选值为模型类别，例如 `3`、`up`、`down`。检测器收到选择后只跟踪该类别，并在目标稳定、深度有效时发布坐标。

`button_select` 是可靠选择客户端：等待订阅者发现、周期重发并等待 `/button_selected` 确认。不要把一次性的 `ros2 topic pub --once` 当作可靠选择机制，因为 CLI 发布器可能在 DDS 订阅匹配完成前就退出。

### 粗定位

`button_approach_planner` 订阅 `/button_surface_pose` 和机器人状态，进行约束的 MoveIt 粗定位。它暴露：

- `/button_approach_planner/plan`
- `/button_approach_planner/execute`
- `/button_approach_planner/return_home`
- `/button_approach_planner/clear_plan`
- `/button_approach/status`

粗定位只负责把末端移动到适合视觉 Servo 的安全预接近姿态，不负责最后按压。执行后任务管理器会等待新的、运动完成后的表面观测，避免把旧的运动中目标交给 Servo。

### 视觉 Servo

`button_visual_servo` 使用 `/button_surface_pose`、TF、关节状态和 Servo 状态计算笛卡尔误差，通过 MoveIt Servo 输出平滑速度/轨迹。仿真中 `simulation_servo_adapter` 把 Servo 输出转成 Gazebo 可执行的关节轨迹。主要接口：

- `/button_visual_servo/start`
- `/button_visual_servo/stop`
- `/button_visual_servo/status`
- `/button_visual_servo/completed`

Servo 会校验目标身份、观测新鲜度、表面法向和控制权；按钮选择改变时会清空旧观测并停止当前闭环。

### 按压执行器

`button_press_executor` 在 Servo 完成交接后接管末端，执行受限的按压轨迹，并同时检查仿真接触传感器、按钮关节状态、力/力矩和反馈新鲜度。接口：

- `/button_press_executor/start`
- `/button_press_executor/stop`
- `/button_press/status`
- `/button_press/completed`

按压完成后释放控制权，任务状态机负责最终回零。

### 任务状态机

`elevator_task_manager` 监听 `/elevator_task/command`，接受 `press 3`、`press up` 或单独的 `3`。一次任务的阶段为：

```text
WAITING_FOR_NODES
→ HOMING_INITIAL
→ SELECTING_BUTTON
→ COARSE_PLANNING
→ COARSE_EXECUTING
→ WAITING_FOR_VISUAL_TARGET
→ VISUAL_SERVO
→ PRESSING
→ HOMING_FINAL
```

任意阶段失败都会发布恢复状态，尝试停止当前控制并回零；任务结束默认清除选择。关键输出：

- `/elevator_task/status`
- `/elevator_task/result`
- `/elevator_task/completed`
- `/elevator_task/active_button`

状态机还会检查关键节点唯一性，防止旧的检测器、规划器、Servo 或按压执行器残留导致服务请求落到错误实例。

## 关键调试话题和服务

| 模块 | 观察 | 控制 |
| --- | --- | --- |
| 检测 | `/button_detections`, `/button_pose`, `/button_selected` | `/button_selection` |
| 粗定位 | `/button_approach/status` | `plan`, `execute`, `return_home` |
| Servo | `/button_visual_servo/status`, `/button_visual_servo/completed` | `start`, `stop` |
| 按压 | `/button_press/status`, `/button_press/completed` | `start`, `stop` |
| 任务 | `/elevator_task/status`, `/elevator_task/result`, `/elevator_task/completed` | `/elevator_task/command`, `stop`, `reset` |

## 已知问题与处理方式

1. **按钮选择偶发丢失**：一次性 CLI 发布可能在发现订阅者前退出。使用 `ros2 run piper_elevator_app button_select <button> --timeout 12`；该客户端会重发并等待 `/button_selected`。手工 CLI 则使用可靠、瞬态本地 QoS 并连续发布数次。
2. **粗定位目标过期**：仿真中 `plan` 服务返回后，目标可能刚好超过规划器的一秒新鲜窗口。任务状态机现在只在收到 `Target is stale` 时重新获取目标并重新规划一次，绝不复用旧轨迹；第二次仍失败会进入原有安全回零流程。
3. **重复节点**：不要同时运行多个完整 `elevator_task.sh` 或重复启动单个业务节点。先用 `ros2 node list` 检查，再停止旧容器。
4. **仿真 GUI**：Gazebo 和 RViZ 需要宿主机的 `DISPLAY`、`XAUTHORITY` 和 Docker 图形映射；无桌面时使用 `gazebo_gui:=false use_rviz:=false`。
5. **真实硬件保护**：真实模式默认不允许运动。只有完成相机标定、硬件控制和 `allow_execution` 等显式配置后才可进入真实链路。

## 推荐交接顺序

1. `./scripts/build.sh`
2. 使用完整仿真命令启动 Gazebo、RViZ 和全部业务节点。
3. 另开终端运行 RQt 查看 `/camera/color/image_raw`。
4. 用 `button_select` 选择按钮并确认 `/button_selected`。
5. 先单独调试时按“粗定位 → Servo → 按压”顺序调用；验收整链路时直接发送 `/elevator_task/command`。
6. 任务结束查看 `/elevator_task/result`，确认 `COMPLETE` 和 `home_reached=true`。
