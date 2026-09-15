# 整套实机启动脚本

```bash
cd /home/q/project/piper_elevator/piper_elevator
./scripts/start_real.sh
```

脚本启动原有实机识别、驱动、MoveIt、粗定位、Servo、按压和任务管理器。
独立视觉实验模块不会接入。默认 ROS_DOMAIN_ID=0，相机 315122272433，
CAN can0 / 1000000 bps，速度参数 10%，不自动使能、不允许执行运动。

启动前需要已经运行过 `./scripts/build.sh`，Docker 可用、机械臂和相机连接。
脚本仅在 CAN 接口未 UP 时使用 sudo 配置速率并启用；已 UP 且速率匹配时保留。
已 UP 但速率不同、BUS-OFF、接口缺失或初始化失败时退出，不启动节点。
脚本不会自动 reset 机械臂、解除急停、修改力矩阈值或发送按压任务。
同一项目同一 CAN 的重复脚本实例由文件锁阻止，相机 launch 另有独占检查；
文件锁不能检测绕过此脚本启动的其他机械臂驱动。先关闭旧的整套启动实例。

常用选项：

```bash
./scripts/start_real.sh --help
./scripts/start_real.sh --dry-run
./scripts/start_real.sh --no-rviz
./scripts/start_real.sh --camera-serial 315122272440
./scripts/start_real.sh --can-interface can1
```

`--dry-run` 只打印命令，不访问 Docker/CAN。普通启动的 sudo 提示仅用于 CAN
初始化。不要用 sudo 运行整个脚本，否则可能改变显示权限和 Docker 环境。

## 执行模式

2026-09-14 用户明确选用 2026-09-02 线性拟合外参，已保存到
`config/real_handeye.json`，对应相机 315122272433、tcp_link → camera_link。
脚本自动加载并发布这份 TF。保留来源及后续复核未通过的记录，不代表完成了新验证。

现在直接执行：

```bash
./scripts/start_real.sh --execute
```

无需再次填写 camera_calibration_valid 或六个外参。--execute 会自动使能并开放运动，
不会自动发送按压任务。默认不带 --execute 时仍禁止使能和运动，但发布相机 TF。
命令行提供的 camera_x/y/z/roll/pitch/yaw 可以覆盖保存值；更换相机执行时必须提供
完整对应外参，避免跨序列号误用。若已有其他相机 TF 发布者，可传
publish_camera_tf:=false 避免重复发布。

力矩标定和 button_press.yaml 没有修改，按压仍受该节点自身的力矩标定检查约束。

另一个宿主终端：

```bash
cd /home/q/project/piper_elevator/piper_elevator
ROS_DOMAIN_ID=0 ./scripts/shell.sh
source /workspace/ros2_ws/install/setup.bash
ros2 topic echo /feedback/arm_status --once
ros2 topic echo /feedback/joint_states --once
```

确认系统可执行后发任务：

```bash
ros2 topic pub --once /elevator_task/command std_msgs/msg/String "{data: 'press up'}"
ros2 topic echo /elevator_task/status
```

启动终端 Ctrl+C 退出节点。任务 stop 服务当前不保证中断粗定位和回原点，
不作为硬件急停使用。本轮仅通过模拟 Docker/CAN 的 11 项脚本测试，未实际启用设备。
