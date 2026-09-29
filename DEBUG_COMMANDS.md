# 按钮任务调试命令

以下命令在仓库根目录执行。容器内的 ROS 命令请在另一个终端运行；两个终端使用相同的 `ROS_DOMAIN_ID`。仿真默认使用 `42`，实机使用 `0`。

## 编译

```bash
cd /home/qi/Project/piper_elevator
./scripts/build.sh
```

## 启动完整仿真

终端 A：

```bash
cd /home/qi/Project/piper_elevator
ROS_DOMAIN_ID=42 ./scripts/elevator_task.sh gazebo_gui:=false use_rviz:=false
```

有桌面显示时可去掉 `gazebo_gui:=false use_rviz:=false`。完整任务启动脚本已经包含 Gazebo、检测器、MoveIt、粗定位、Servo 和按压节点，不需要再分别启动它们。

终端 B：

```bash
cd /home/qi/Project/piper_elevator
ROS_DOMAIN_ID=42 ./scripts/shell.sh
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
```

后续仿真命令都在终端 B 的容器 shell 中执行。

## 观察首次识别

先选按钮（把 `up` 换成 `down`、`1`、`2`、`3`、`open` 或 `close`）：

```bash
ros2 run piper_elevator_app button_select up --timeout 12
```

选择客户端会等待订阅者并确认 `/button_selected`。如果只启动了检测器，可加 `--min-subscribers 1`。

分别在新终端查看二维检测、相机系 RGB-D 位姿、粗定位稳定后的 `base_link` 位姿和拒收原因。下面每行单独运行：

```bash
ros2 topic echo /button_detections
ros2 topic echo /button_surface_pose
ros2 topic echo /button_pose_base
ros2 topic echo /button_approach_planner/observation_status
```

每条 `ros2 topic echo` 会持续占用一个终端。仿真完整任务默认开启已知布局重标，故 `/button_detections` 中的类别可能经过重标，不能用它评估模型原始分类。要临时检查模型原始分类，在另一个终端执行：

```bash
ros2 param set /button_detector simulation_layout_relabel false
```

查看完毕后，可用 `ros2 param set /button_detector simulation_layout_relabel true` 恢复仿真布局重标。`/button_surface_pose` 还要求深度和局部平面拟合成功；`/button_pose_base` 是粗定位使用的稳定三维坐标。比较真实按钮坐标时，使用同一 `base_link` 坐标系。

图像调试：

```bash
ros2 run rqt_image_view rqt_image_view
```

在窗口中选择 `/button_detector/debug_image`。无桌面显示时可检查话题和帧率：

```bash
ros2 topic list
ros2 topic hz /camera/color/image_raw
ros2 topic hz /camera/aligned_depth_to_color/image_raw
```

## 单步粗定位、Servo 与按压

仅在没有运行中的 `/elevator_task/command` 任务时使用。先发布 `/button_selection`，并确认 `/button_pose_base` 有输出，再依次执行下面的命令。一次只运行一行，确认服务响应后再运行下一行：

```bash
ros2 service call /button_approach_planner/plan std_srvs/srv/Trigger "{}"
ros2 service call /button_approach_planner/execute std_srvs/srv/Trigger "{}"
ros2 service call /button_visual_servo/start std_srvs/srv/Trigger "{}"
ros2 service call /button_press_executor/start std_srvs/srv/Trigger "{}"
ros2 service call /button_approach_planner/return_home std_srvs/srv/Trigger "{}"
```

每一步都检查服务响应中的 `success` 和相应状态，再执行下一步。停止当前 Servo 或整个任务：

```bash
ros2 service call /button_visual_servo/stop std_srvs/srv/Trigger "{}"
ros2 service call /elevator_task_manager/stop std_srvs/srv/Trigger "{}"
```

## 一条命令运行完整按钮任务

```bash
ros2 topic pub --rate 2 --times 3 /elevator_task/command std_msgs/msg/String "{data: 'press up'}"
```

需要观察过程或结果时，在另外的容器 shell 中分别运行：

```bash
ros2 topic echo /elevator_task/status
ros2 topic echo /elevator_task/result
```

短时间重复发布可避免 ROS 发现连接建立前丢失首条消息。`press up` 可改成 `press down`、`press 1`、`press 2`、`press 3`、`press open` 或 `press close`。

## 重复性测试

下面的脚本会自行启动和清理一套无界面仿真，使用独立的 ROS domain，避免与终端 A 的仿真混用：

```bash
cd /home/qi/Project/piper_elevator
ROS_DOMAIN_ID=78 ./scripts/test_button_stability.sh --execute
```

如果要测试已经启动的完整任务栈，在同一个 `ROS_DOMAIN_ID` 下使用 `--attach`：

```bash
ROS_DOMAIN_ID=42 ./scripts/test_button_stability.sh --execute --attach
```

仅测视觉伺服重复性时，先启动完整仿真，再运行：

```bash
ROS_DOMAIN_ID=42 ./scripts/test_visual_servo.sh --execute --runs 5 --continue-on-failure
```


### 远程版本的稳定性回归脚本

合并的 `v1.1` 还提供两种完整链路回归模式。此脚本会自动启动无界面仿真，并把 CSV 和启动日志存入 `ros2_ws/diagnostics/data/stability/`：

```bash
cd /home/qi/Project/piper_elevator
./scripts/stability_test.sh --mode both --runs 5 --button 3
```

`--mode stable` 使用固定的合成目标验证运动链路；`--mode yolo` 使用相机、YOLO 和深度检测链路；`both` 依次运行两种模式。

## 实机只读调试

当前仓库没有 `scripts/start_real.sh`。实机入口是 `scripts/elevator_task.sh simulation_mode:=false`；默认禁止硬件命令和任务执行。仅观察相机、检测与状态时：

```bash
cd /home/qi/Project/piper_elevator
ROS_DOMAIN_ID=0 ./scripts/elevator_task.sh simulation_mode:=false use_rviz:=false
```

另开终端进入相同 domain：

```bash
ROS_DOMAIN_ID=0 ./scripts/shell.sh
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
```

随后查看 `/button_detections`、`/button_surface_pose` 等话题。`/button_pose_base` 还需要可用的相机到 `base_link` 变换；此处默认未发布相机外参。实机执行需要在启动时显式开启相应的硬件与任务执行参数。
