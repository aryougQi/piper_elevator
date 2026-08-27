# Gazebo 虚拟硬件

## 启动

```bash
cd /home/q/project/piper_elevator/piper_elevator
./scripts/gazebo_hardware.sh
```

无界面运行：

```bash
./scripts/gazebo_hardware.sh gui:=false
```

该命令只启动 Piper、Pika、D405 和实体按钮仿真，不启动 MoveIt、视觉节点或
Planner。`./scripts/sim.sh` 等价于该命令。

## 发布

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| `/camera/color/image_raw` | `sensor_msgs/Image` | D405 彩色图 |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | `32FC1` 米制深度图 |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | 相机内参 |
| `/piper_pika/joint_states` | `sensor_msgs/JointState` | Piper + Pika 反馈 |
| `/elevator_button/pressed` | `std_msgs/Bool` | 按钮接触事件 |
| `/elevator_button/joint_states` | `sensor_msgs/JointState` | 按钮行程 |

## 接收

| 接口 | 类型 | 说明 |
| --- | --- | --- |
| `/arm_controller/follow_joint_trajectory` | `control_msgs/action/FollowJointTrajectory` | 六轴轨迹 |
| `/pika_gripper_controller/follow_joint_trajectory` | `control_msgs/action/FollowJointTrajectory` | Pika 开度轨迹 |

运行检查：

```bash
./scripts/check_gazebo.sh
```

# 视觉节点

## 启动

```bash
cd /home/q/project/piper_elevator/piper_elevator
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
cd /home/q/project/piper_elevator/piper_elevator
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
cd /home/q/project/piper_elevator/piper_elevator
ROS_DOMAIN_ID=42 ./scripts/shell.sh
source /workspace/ros2_ws/install/setup.bash
```

生成轨迹：

```bash
ros2 service call /button_approach_planner/plan std_srvs/srv/Trigger "{}"
```

执行仿真轨迹：

```bash
ros2 service call /button_approach_planner/execute std_srvs/srv/Trigger "{}"
```

清除已有轨迹：

```bash
ros2 service call /button_approach_planner/clear_plan std_srvs/srv/Trigger "{}"
```

以上 `execute` 命令用于仿真；真机模式默认禁止执行。
