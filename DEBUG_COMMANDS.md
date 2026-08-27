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