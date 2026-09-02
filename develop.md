# Gazebo 虚拟硬件

## 完整任务状态机

项目的统一执行入口是 `elevator_task.launch.py`。它会带起仿真硬件、
检测、MoveIt/Servo、粗定位、精定位、按压和上层状态机：

```bash
ROS_DOMAIN_ID=42 ./scripts/elevator_task.sh
```

状态机订阅 `/elevator_task/command` (`std_msgs/String`)，接受
`press <button>` 或单个按钮名。它会先通过 `/button_selection` 选择指定
类别，并且只在收到本次选择之后的新鲜 RGB-D 表面位姿时才进入规划。
视觉伺服和按压的完成标志同样按消息序号隔离，不会误用上一轮锁存的
`completed=true`。

对外接口：

| 接口 | 类型 | 说明 |
| --- | --- | --- |
| `/elevator_task/command` | `std_msgs/String` | 输入 `press 3`、`press up` 等 |
| `/elevator_task/status` | `std_msgs/String` | 当前状态机阶段 |
| `/elevator_task/result` | `std_msgs/String` | 最终结果及失败原因 |
| `/elevator_task/completed` | `std_msgs/Bool` | 按压成功且已回 home |
| `/elevator_task/active_button` | `std_msgs/String` | 正在执行的按钮 |
| `/elevator_task_manager/stop` | `std_srvs/Trigger` | 中止动作并进入安全恢复 |
| `/elevator_task_manager/reset` | `std_srvs/Trigger` | 空闲时清除上次结果 |

正常状态序列为 `HOMING_INITIAL` → `SELECTING_BUTTON` →
`COARSE_PLANNING` → `COARSE_EXECUTING` → `WAITING_FOR_VISUAL_TARGET` → `VISUAL_SERVO` →
`PRESSING` → `HOMING_FINAL` → `COMPLETE`。失败时进入 `RECOVERING`，
先停止按压和视觉速度控制，再清除存储的粗规划并尝试回 home。

## 启动

```bash
cd /home/qi/Project/piper_elevator
./scripts/gazebo_hardware.sh
```

启动脚本会把当前桌面会话的 `XAUTHORITY` 只读映射到容器。重启或重新登录后
GDM 会生成新的授权文件，脚本会自动使用新文件，不需要执行 `xhost +`。如果
当前没有图形会话，请使用下面的无界面模式。

无界面运行：

```bash
./scripts/gazebo_hardware.sh gui:=false
```

该命令只启动 Piper、Pika、Pika 内置镜头和实体按钮仿真，不启动 MoveIt、视觉节点或
Planner。`./scripts/sim.sh` 等价于该命令。

仿真电梯面板尺寸为 `270 x 320 mm`，上方是红色楼层显示屏，按钮按画面从左到右
排列为 `1 2 3 / 4 ↑ ↓ / 开门 关门 报警`。面板在机械臂 MoveIt `home` 姿态的
`848 x 480` 相机画面中完整可见。九个按钮均使用真实纹理，并分别具有 4 mm
弹簧回位行程、独立接触传感器和 `/elevator_button/joint_states` 反馈。按压节点
根据 `/button_selected` 自动选择同名关节与接触话题，不会用其他按钮的触点误判成功。

## 发布

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| `/camera/color/image_raw` | `sensor_msgs/Image` | Pika 内置镜头彩色图 |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | `32FC1` 米制深度图 |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | 相机内参 |
| `/piper_pika/joint_states` | `sensor_msgs/JointState` | Piper + Pika 反馈 |
| `/elevator_button/pressed` | `std_msgs/Bool` | 按钮接触事件 |
| `/elevator_button/button_<name>/contacts` | `ros_gz_interfaces/Contacts` | 每个按钮的独立持续接触点 |
| `/elevator_button/joint_states` | `sensor_msgs/JointState` | 九个按钮的独立行程 |

## 接收

| 接口 | 类型 | 说明 |
| --- | --- | --- |
| `/arm_controller/follow_joint_trajectory` | `control_msgs/action/FollowJointTrajectory` | 六轴轨迹 |
| `/pika_gripper_controller/follow_joint_trajectory` | `control_msgs/action/FollowJointTrajectory` | Pika 中心开度及左右指对称轨迹 |

运行检查：

```bash
./scripts/check_gazebo.sh
```

# 视觉节点

## 启动

```bash
cd /home/qi/Project/piper_elevator
./scripts/button_camera.sh
```

该脚本同时启动 RealSense D405 驱动和 `button_detector` 节点。首次运行前需执行：

```bash
./scripts/build.sh
```

## 订阅

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| `/camera/color/image_raw` | `sensor_msgs/Image` | 彩色图像 |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | 对齐到彩色图的图 |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | 相机内参 |
| `/button_selection` | `std_msgs/String` | 选择按钮类别，如 `3`、`up`、`open` |

## 发布

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| `/button_detections` | `vision_msgs/Detection2DArray` | 当前帧的全部按钮检测框 |
| `/button_pixel` | `geometry_msgs/PointStamped` | 选中按钮的像素中心 |
| `/button_pose` | `geometry_msgs/PoseStamped` | 按钮在相机坐标系中的三维位置 |
| `/button_detection_valid` | `std_msgs/Bool` | 目标是否稳定且深度有效 |
| `/button_detection_confidence` | `std_msgs/Float32` | 目标置信度 |
| `/button_detector/debug_image` | `sensor_msgs/Image` | 检测结果调试图 |
| `/button_selected` | `std_msgs/String` | 当前选中的按钮类别 |

默认不会发布 `/button_pixel` 和 `/button_pose`，需要先选择按钮：

```bash
ros2 topic pub --once /button_selection std_msgs/msg/String "{data: '3'}"
```

# Planner 节点

## 启动

使用仿真并关闭自动执行，方便手动调用 `plan` 和 `execute`：

```bash
cd /home/qi/Project/piper_elevator
./scripts/button_approach_sim.sh auto_execute:=false
```

## 订阅

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| `/button_pose` | `geometry_msgs/PoseStamped` | 按钮在相机坐标系中的三维位置 |

## 发布

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| `/button_pose_base` | `geometry_msgs/PoseStamped` | 按钮在 `base_link` 坐标系中的位置 |
| `/button_approach_pose` | `geometry_msgs/PoseStamped` | 按钮前方的安全靠近位姿 |
| `/button_approach/status` | `std_msgs/String` | 规划和执行状态 |

## Plan 和 Execute

打开另一个终端并进入仿真 ROS 域：

```bash
cd /home/qi/Project/piper_elevator
ROS_DOMAIN_ID=42 ./scripts/shell.sh
source /workspace/ros2_ws/install/setup.bash
```

生成轨迹：

```bash
ros2 service call /button_approach_planner/plan std_srvs/srv/Trigger "{}"
```

第一次粗略规划会先把 `center_joint` 闭合到 `0.0 m`，再以
`pika_fingertip_center_link`（两指闭合中心的最前端）作为目标点移动到按钮前
14 cm。位置是该阶段的精确目标；相机指向使用
`/button_surface_pose` 的深度平面法向，用约 `±8°` 的可视包络修正
上/下倾斜。滚转以当前可达姿态为基准，避免强制画面正立时将腕部
推到限位。垂直于按钮表面的精确姿态调整仍由后续视觉伺服完成。目标关节状态还要求
`joint4/5/6` 远离机械限位，为视觉伺服保留腕部调整余量。

执行仿真轨迹：

```bash
ros2 service call /button_approach_planner/execute std_srvs/srv/Trigger "{}"
```

清除已有轨迹：

```bash
ros2 service call /button_approach_planner/clear_plan std_srvs/srv/Trigger "{}"
```

以上 `execute` 命令用于仿真；真机模式默认禁止执行。

## 视觉伺服精定位

粗靠近完成后，RGB-D 检测器会在按钮框内拟合局部深度平面，并将按钮中心和
表面法向发布到 `/button_surface_pose`。节点启动 MoveIt Servo 后先保持
14 cm 粗定位位置不变，只调整相机光轴与画面滚转。粗定位前锁定的按钮世界
坐标会投影到当前图像，在 60 px 局部窗口内重新获取同一个物理按钮；连续收到
3 帧新的 RGB-D 表面位姿后才启动伺服。连续 2 帧姿态满足约束后，
不会停车或重启 Servo，而是直接开始修正横向位置并靠近按钮前 27.5 mm。
整个视觉阶段使用同一个 50 Hz 速度闭环，并包含低通滤波、线速度/角速度限制
和加速度限制；闭环线速度上限为 80 mm/s。

接近过程中会锁定已经对正的表面法向和末端姿态，避免贴近面板后再次大幅转腕。
若姿态偏差增大到 2--3°，轴向靠近速度会逐渐降为零，但横向和姿态校正仍继续；
只有姿态恢复后才继续靠近。到达按压预备位后才平滑减速到零，并保持同一个
Servo 会话等待按压节点接管。最终结果使用机器人 TF 验证，不要求此时 YOLO
仍能看到完整按钮。

仿真组合启动已经包含 `button_visual_servo`。开始精定位：

```bash
ros2 service call /button_visual_servo/start std_srvs/srv/Trigger "{}"
ros2 topic echo /button_visual_servo/status
```

视觉阶段连续 2 帧满足最终条件后，状态变为 `COMPLETE`：

- 指尖中心到按钮表面的法向距离为 `27.5 ± 2.5 mm`，且最近不小于 `25 mm`；
- 横向偏差不超过 `3 mm`；
- `camera_color_optical_frame` 的 +Z 光轴与表面法向夹角不超过 `3°`。

姿态计算会根据 TF 自动补偿相机与夹爪之间的安装外参，不再把
夹爪 +Z 误当作真实相机光轴。相机 optical frame 的 -Y 视为画面向上，并以
`base_link +Z` 为竖直参考，以 `0°` 滚转为控制目标，并允许在 `±3°` 内进入
最终靠近；回正角速度单独限制为 `0.30 rad/s`。表面法向垂直度始终优先于画面水平，水平参考退化或受可达性
限制时会保留可达姿态。`/button_visual_servo/status` 中的 `roll=...deg` 可用于
检查实际滚转误差。

停止命令：

```bash
ros2 service call /button_visual_servo/stop std_srvs/srv/Trigger "{}"
```

视觉伺服完成并发布 `/button_visual_servo/completed: true` 后，可以启动按压：

```bash
ros2 service call /button_press_executor/start std_srvs/srv/Trigger "{}"
ros2 topic echo /button_press/status
```

按压节点沿相机光轴以低速直线运动，以静止基线之后的六关节力矩增量判断
接触，额外推进 2.5 mm 后原路撤回。撤回前段使用 25 mm/s，最后 3 mm 降到
8 mm/s；仿真使用 0.8 mm 到位容差，真机保持 0.5 mm。真机必须先在
`config/button_press.yaml` 填入经过实验得到的六关节增量阈值和绝对力矩上限，
并设置 `torque_thresholds_calibrated: true`；未标定时节点会拒绝执行。

粗定位将 `joint5` 保持在远离零位的负腕弯分支，并检查轨迹终点，避免视觉
Servo 从腕部奇异姿态启动。视觉伺服收到 MoveIt Servo 的奇异点、碰撞或
关节边界停机状态时会立即失败，由重复测试流程回到 home 后重新规划。

真机只有在 RGB-D 深度有效、手眼外参已经标定，并显式设置
`camera_calibration_valid:=true allow_execution:=true` 后才允许启动。Pika
单目鱼眼相机不能单独提供精确的 3 cm 距离和表面法向。
