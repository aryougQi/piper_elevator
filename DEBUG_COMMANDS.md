# 运行命令

## 编译

```bash
cd /home/q/project/piper_elevator/piper_elevator
./scripts/build.sh
```

## 启动仿真

```bash
cd /home/q/project/piper_elevator/piper_elevator
ROS_DOMAIN_ID=42 ./scripts/gazebo_hardware.sh
```

## 启动视觉

```bash
cd /home/q/project/piper_elevator/piper_elevator
ROS_DOMAIN_ID=42 ./scripts/shell.sh
```
```bash
source /workspace/ros2_ws/install/setup.bash

ros2 launch piper_elevator_app button_detector.launch.py \
  use_sim_time:=true
```

```bash
ros2 topic pub --once /button_selection \
  std_msgs/msg/String "{data: 'up'}"
```
data可以改down，1,2,3等

```bash
ros2 run rqt_image_view rqt_image_view
```

## 启动moveit及坐标转换，plan execute


```bash
cd /home/q/project/piper_elevator/piper_elevator
ROS_DOMAIN_ID=42 ./scripts/shell.sh
```

```bash
source /workspace/ros2_ws/install/setup.bash

ros2 launch piper_elevator_app piper_pika_moveit.launch.py \
  external_hardware:=true \
  use_sim_time:=true
```
```bash
source /workspace/ros2_ws/install/setup.bash

ros2 launch piper_elevator_app button_approach_planner.launch.py \
  use_sim_time:=true \
  simulation_mode:=true \
  camera_calibration_valid:=true \
  allow_execution:=true
```

```bash
ros2 service call /button_approach_planner/plan \
  std_srvs/srv/Trigger "{}"
```

```bash
ros2 service call /button_approach_planner/execute \
  std_srvs/srv/Trigger "{}"
```

## 启动视觉伺服精定位

```bash
ros2 launch piper_elevator_app button_visual_servo.launch.py \
  use_sim_time:=true \
  simulation_mode:=true \
  camera_calibration_valid:=true \
  allow_execution:=true
```

开始闭环精定位，并在另一个终端查看误差：

```bash
ros2 service call /button_visual_servo/start \
  std_srvs/srv/Trigger "{}"

ros2 topic echo /button_visual_servo/status
```

紧急停止当前伺服轨迹：

```bash
ros2 service call /button_visual_servo/stop \
  std_srvs/srv/Trigger "{}"
```

## 一键重复性测试

先保证 Gazebo、检测器、MoveIt、粗定位节点和视觉伺服节点都已经启动。
单轮测试会驱动机械臂，必须显式传入 `--execute`：

```bash
cd /home/q/project/piper_elevator/piper_elevator
ROS_DOMAIN_ID=42 ./scripts/test_visual_servo.sh --execute
```

连续测试 5 轮，失败时仍继续后续轮次：

```bash
ROS_DOMAIN_ID=42 ./scripts/test_visual_servo.sh \
  --execute \
  --runs 5 \
  --continue-on-failure
```

如果机械臂已经在粗定位的 15 cm 姿态，可仅测试视觉伺服：

```bash
ROS_DOMAIN_ID=42 ./scripts/test_visual_servo.sh \
  --execute \
  --skip-coarse
```

## 按压按钮

```bash

source /workspace/ros2_ws/install/setup.bash

ros2 launch piper_elevator_app button_press.launch.py \
  use_sim_time:=true \
  simulation_mode:=true \
  allow_execution:=true


ros2 service call /button_press_executor/start \
  std_srvs/srv/Trigger "{}"

```
## 状态机

```bash

ROS_DOMAIN_ID=42 ./scripts/elevator_task.sh

ROS_DOMAIN_ID=42 ./scripts/shell.sh
source /workspace/ros2_ws/install/setup.bash

ros2 topic pub --once /elevator_task/command \
  std_msgs/msg/String "{data: 'press up'}"


ros2 topic echo /elevator_task/status
ros2 topic echo /elevator_task/result
ros2 topic echo /elevator_task/completed


ros2 service call /elevator_task_manager/stop \
  std_srvs/srv/Trigger "{}"

```