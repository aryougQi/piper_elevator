# 运行命令

以下命令以当前电脑的项目目录为准：

```bash
cd /home/qi/Project/piper_elevator
```

所有并行终端必须使用相同的 `ROS_DOMAIN_ID`。本文统一使用 `42`。

## 1. 编译与检查

```bash
./scripts/build.sh
./scripts/check.sh
./scripts/check_gazebo.sh
```

## 2. 启动完整仿真（推荐）

一个终端只启动一次完整状态机，避免产生同名检测器、Planner 或 Servo 节点：

```bash
cd /home/qi/Project/piper_elevator
ROS_DOMAIN_ID=42 ./scripts/elevator_task.sh use_rviz:=false
```

该命令会同时启动 Gazebo GUI、虚拟硬件、按钮检测、MoveIt、粗定位、视觉
Servo、按压节点和任务管理器。Gazebo 世界文件已经配置为面板与机械臂的最佳
观察视角。不要再单独启动 `button_detector`、`button_approach_planner` 或
`button_visual_servo`，否则会出现重复节点和服务请求随机落到旧实例的问题。

没有图形桌面时使用：

```bash
ROS_DOMAIN_ID=42 ./scripts/elevator_task.sh gazebo_gui:=false use_rviz:=false
```

## 3. 进入调试终端

另开一个终端：

```bash
cd /home/qi/Project/piper_elevator
ROS_DOMAIN_ID=42 ./scripts/shell.sh
```

容器入口会自动加载 ROS 2 和工作区环境。以下 ROS 2 命令都在这个调试终端
内执行，不需要再次 `source`。

检查关键节点是否只有一个实例：

```bash
ros2 node list | sort
```

应当各只有一个：

```text
/button_detector
/button_approach_planner
/button_visual_servo
/button_press_executor
/elevator_task_manager
```

## 4. 执行单个按钮

支持的按钮为 `1 2 3 4 up down open close alarm`：

```bash
ros2 topic pub --once /elevator_task/command \
  std_msgs/msg/String "{data: 'press up'}"
```

查看状态和最终结果：

```bash
ros2 topic echo --once /elevator_task/status
ros2 topic echo --once /elevator_task/result
ros2 topic echo --once /elevator_task/completed
```

安全停止当前任务并自动尝试回 Home：

```bash
ros2 service call /elevator_task_manager/stop \
  std_srvs/srv/Trigger "{}"
```

## 5. 完整九按钮测试

在宿主机的新终端执行，不需要进入 `scripts/shell.sh`：

```bash
cd /home/qi/Project/piper_elevator
ROS_DOMAIN_ID=42 ./scripts/test_elevator_task.sh \
  --execute \
  --continue-on-failure \
  --settle-seconds 1.0
```

只测试指定按钮：

```bash
ROS_DOMAIN_ID=42 ./scripts/test_elevator_task.sh \
  --execute \
  --buttons 4 up open \
  --continue-on-failure \
  --settle-seconds 1.0
```

连续测试 `up` 三轮：

```bash
ROS_DOMAIN_ID=42 ./scripts/test_elevator_task.sh \
  --execute \
  --buttons up \
  --cycles 3 \
  --continue-on-failure \
  --settle-seconds 1.0
```

结果保存在：

```text
test_logs/elevator_task_*.csv
test_logs/endpoint_trace_*.csv
```

第二个文件以约 50 Hz 记录末端位置、姿态、六个关节和任务阶段，可用于检查
回退、连续摆动以及 Servo 和按压交接是否连贯。

## 6. 视觉与末端位置排查

选择目标但不启动任务：

```bash
ros2 topic pub --once /button_selection \
  std_msgs/msg/String "{data: 'up'}"
```

查看检测、规划器和视觉 Servo 是否分别就绪：

```bash
ros2 topic echo --once /button_selected
ros2 topic echo --once /button_detection_valid
ros2 topic echo --once /button_pose
ros2 topic echo --once /button_surface_pose
ros2 topic echo --once /button_approach/status
ros2 topic echo --once /button_visual_servo/status
```

查看实时末端 TF：

```bash
ros2 run tf2_ros tf2_echo base_link pika_fingertip_center_link
```

查看检测画面：

```bash
ros2 run rqt_image_view rqt_image_view
```

在窗口内选择 `/button_detector/debug_image`。清除手动选择：

```bash
ros2 topic pub --once /button_selection \
  std_msgs/msg/String "{data: clear}"
```

## 7. 单阶段服务（仅在完整状态机已启动时调试）

粗定位规划与执行：

```bash
ros2 service call /button_approach_planner/plan \
  std_srvs/srv/Trigger "{}"
ros2 service call /button_approach_planner/execute \
  std_srvs/srv/Trigger "{}"
```

视觉 Servo：

```bash
ros2 service call /button_visual_servo/start \
  std_srvs/srv/Trigger "{}"
ros2 service call /button_visual_servo/stop \
  std_srvs/srv/Trigger "{}"
```

按压节点：

```bash
ros2 service call /button_press_executor/start \
  std_srvs/srv/Trigger "{}"
ros2 service call /button_press_executor/stop \
  std_srvs/srv/Trigger "{}"
```

调用单阶段服务前必须先发布 `/button_selection`，并等待对应状态为
`TARGET_READY` 或 `READY`。正常完整测试应优先使用 `/elevator_task/command`，
由任务管理器处理这些握手。

## 8. 关闭仿真与容器

先在启动完整仿真的终端按 `Ctrl+C`。然后在项目根目录检查：

```bash
docker ps --filter name=piper_elevator
```

如果仍有本项目的残留容器，再执行：

```bash
docker compose down --remove-orphans
```
