# 仿真与调试命令

本文档以项目根目录 `/home/q/project/piper_elevator/piper_elevator` 为准。所有终端都使用同一个 `ROS_DOMAIN_ID`；下面统一使用 `42`。

## 1. 编译

第一次运行或代码更新后执行：

```bash
cd /home/q/project/piper_elevator/piper_elevator
./scripts/build.sh
```

`build.sh` 会在 Docker 容器中安装 rosdep 并编译完整工作空间。编译完成后，重新启动的容器会自动使用 `ros2_ws/install/setup.bash`。

## 2. 一次启动完整仿真

这一条会启动完整链路：Gazebo Fortress、Gazebo 控制器和相机桥、按钮检测器、MoveIt、RViZ、粗定位规划器、视觉 Servo、按压执行器和电梯任务状态机。

```bash
cd /home/q/project/piper_elevator/piper_elevator
ROS_DOMAIN_ID=42 ./scripts/elevator_task.sh \
  simulation_mode:=true \
  gazebo_gui:=true \
  use_rviz:=true
```

这个终端保持运行，不要关闭。启动参数的含义：

- `gazebo_gui:=true`：显示 Gazebo；无桌面或没有 X11 权限时改成 `false`。
- `use_rviz:=true`：启动 MoveIt 的 RViZ 配置；不需要时改成 `false`。
- `simulation_mode:=true`：使用 Gazebo 控制器，不连接真实 Piper/Pika。

如果出现 `DISPLAY` 或 `Xauthority` 错误，先确认宿主机在图形桌面中；只想后台跑仿真时使用：

```bash
ROS_DOMAIN_ID=42 ./scripts/elevator_task.sh \
  simulation_mode:=true gazebo_gui:=false use_rviz:=false
```

检查所有节点：

```bash
ROS_DOMAIN_ID=42 ./scripts/shell.sh
source /workspace/ros2_ws/install/setup.bash
ros2 node list
ros2 topic list | sort
```

完整仿真至少应看到：`/gazebo`、`/button_detector`、`/move_group`、`/servo_node`、`/button_approach_planner`、`/button_visual_servo`、`/button_press_executor`、`/elevator_task_manager`。

## 3. 用 RQt 查看摄像头

另开一个终端进入同一个 ROS 容器：

```bash
cd /home/q/project/piper_elevator/piper_elevator
ROS_DOMAIN_ID=42 ./scripts/shell.sh
```

容器内执行：

```bash
source /workspace/ros2_ws/install/setup.bash
ros2 run rqt_image_view rqt_image_view
```

在 RQt 的图像话题下拉框中选择：

- `/camera/color/image_raw`：Gazebo 相机原图。
- `/button_detector/debug_image`：带检测框的调试图；选择该话题后检测器才会发布调试图。
- `/camera/aligned_depth_to_color/image_raw`：对齐深度图。

也可以先确认相机话题确实存在：

```bash
ros2 topic hz /camera/color/image_raw
ros2 topic echo /button_detections --once
```

## 4. 只做按钮选择

完整仿真启动后，推荐使用带确认和重发机制的选择客户端。它会等待订阅者发现，按 0.5 秒重发，并等待检测器在 `/button_selected` 上确认：

```bash
source /workspace/ros2_ws/install/setup.bash
ros2 run piper_elevator_app button_select 3 --timeout 12
```

按钮名称可以是 `1`、`2`、`3`、`4`、`up`、`down`、`open`、`close` 或 `alarm`。只启动检测器时可降低订阅者要求：

```bash
ros2 run piper_elevator_app button_select 3 --min-subscribers 1
```

确认结果：

```bash
ros2 topic echo /button_selected --once
ros2 topic echo /button_detection_valid
ros2 topic echo /button_pose
```

清除当前选择：

```bash
ros2 run piper_elevator_app button_select clear --timeout 8 --min-subscribers 1
```

### 选择偶发收不到时

不要只依赖一次 `ros2 topic pub --once`：CLI 发布器刚建立发现连接时，第一条消息可能在订阅者匹配前发出。优先使用上面的 `button_select`。如果必须使用 CLI，改为短时间重复发布，并使用可靠、瞬态本地 QoS：

```bash
ros2 topic pub \
  --rate 2 --times 6 \
  --qos-reliability reliable \
  --qos-durability transient_local \
  /button_selection std_msgs/msg/String "{data: '3'}"
```

然后检查检测器确认：

```bash
ros2 topic echo /button_selected --once
```

如果仍无确认，检查是否有重复检测器或旧容器：

```bash
ros2 node list | grep button_detector
ros2 topic info /button_selection -v
```

同一时间只能运行一个 `/button_detector`；完整任务管理器也会拒绝重复的关键节点。

## 5. 手动执行粗定位

完整任务启动时粗定位节点已经存在。需要单独调试时，先启动 Gazebo、MoveIt 和检测器，再启动：

```bash
ros2 launch piper_elevator_app button_approach_planner.launch.py \
  use_sim_time:=true \
  simulation_mode:=true \
  camera_calibration_valid:=true \
  allow_execution:=true
```

先规划，再执行：

```bash
ros2 service call /button_approach_planner/plan std_srvs/srv/Trigger "{}"
ros2 topic echo /button_approach/status
ros2 service call /button_approach_planner/execute std_srvs/srv/Trigger "{}"
```

只规划不移动时不要调用 `execute`。粗定位完成后应等待新的表面位姿，再启动 Servo。

## 6. 手动执行视觉 Servo

```bash
ros2 launch piper_elevator_app button_visual_servo.launch.py \
  use_sim_time:=true \
  simulation_mode:=true \
  camera_calibration_valid:=true \
  allow_execution:=true
```

选择按钮后启动闭环：

```bash
ros2 service call /button_visual_servo/start std_srvs/srv/Trigger "{}"
ros2 topic echo /button_visual_servo/status
```

停止 Servo：

```bash
ros2 service call /button_visual_servo/stop std_srvs/srv/Trigger "{}"
```

## 7. 手动执行按压

```bash
ros2 launch piper_elevator_app button_press.launch.py \
  use_sim_time:=true \
  simulation_mode:=true \
  allow_execution:=true
```

Servo 已经把机械臂交接给按压执行器后，启动按压：

```bash
ros2 service call /button_press_executor/start std_srvs/srv/Trigger "{}"
ros2 topic echo /button_press/status
ros2 topic echo /button_press/completed
```

停止按压：

```bash
ros2 service call /button_press_executor/stop std_srvs/srv/Trigger "{}"
```

## 8. 一键执行“选按钮 → 粗规划 → Servo → 按压 → 回零”

完整仿真已经启动时，只需向任务状态机发送一条命令。建议重复发送 3 次，避免 CLI 发布器尚未完成发现：

```bash
source /workspace/ros2_ws/install/setup.bash
ros2 topic pub --rate 2 --times 3 \
  /elevator_task/command std_msgs/msg/String "{data: 'press 3'}"
```

任务状态和结果：

```bash
ros2 topic echo /elevator_task/status
ros2 topic echo /elevator_task/result
ros2 topic echo /elevator_task/completed
```

任务管理器内部顺序是：等待节点 → 初始回零 → 选择按钮并等待视觉目标 → 粗规划 → 粗定位执行 → 等待新的 RGB-D 目标 → 视觉 Servo → 按压 → 最终回零。也可以发送 `press up`、`press down`、`press open` 等。

停止并让状态机恢复：

```bash
ros2 service call /elevator_task_manager/stop std_srvs/srv/Trigger "{}"
```

重置空闲状态：

```bash
ros2 service call /elevator_task_manager/reset std_srvs/srv/Trigger "{}"
```

## 9. 结束仿真

在启动完整仿真的终端按 `Ctrl+C`。如果有遗留容器，可检查并清理：

```bash
docker ps --format '{{.Names}}'
docker compose down --remove-orphans
```

## 10. 稳定性回归测试

脚本会自动启动一套无 GUI 仿真，重复发送任务命令，等待完整终态，并将每轮结果写入 CSV。默认每种模式运行 5 轮：

```bash
cd /home/q/project/piper_elevator/piper_elevator
./scripts/stability_test.sh --mode both --runs 5 --button 3
```

两种模式含义：

- `--mode stable`：关闭 YOLO，使用固定的合成按钮目标，只验证粗定位、MoveIt 执行、Servo、按压和回零链路。
- `--mode yolo`：使用当前 ONNX YOLO、Gazebo 相机和真实检测/深度链路，验证完整系统。
- `--mode both`：先跑 stable，再跑 yolo。

常用命令：

```bash
./scripts/stability_test.sh --mode stable --runs 10 --button 3
./scripts/stability_test.sh --mode yolo --runs 10 --button 3 --timeout 300
```

输出会显示每轮 `COMPLETE` 或具体失败阶段，并在 `ros2_ws/diagnostics/data/stability/` 保存：

- `elevator_stability_<mode>_<timestamp>.csv`：成功率、耗时、重规划次数和失败原因。
- `launch_<mode>_<timestamp>.log`：该轮仿真所有节点日志。

稳定目标模式通过 `stable_target_mode:=true` 启动 `mock_button_pose`，不会使用 YOLO；它的通过只能证明运动控制链路可用。只有 `yolo` 模式多轮通过，才能说明当前模型和 RGB-D 跟踪也稳定。
