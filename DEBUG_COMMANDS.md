# 运行命令

## 修改代码后构建

```bash
cd /home/qi/Project/piper_elevator
./scripts/shell.sh

source /opt/ros/humble/setup.bash
cd /workspace/ros2_ws
colcon build --packages-select \
  pika_gripper_description piper_elevator_app \
  --symlink-install
source install/setup.bash
exit
```

## 只启动相机和按钮检测

```bash
cd /home/qi/Project/piper_elevator
./scripts/button_camera.sh
```

## 启动仿真

```bash
cd /home/qi/Project/piper_elevator
xhost +si:localuser:root
./scripts/button_approach_sim.sh
```

## 启动真机安全模式

```bash
cd /home/qi/Project/piper_elevator
xhost +si:localuser:root
./scripts/button_approach_real.sh
```

当前真机命令默认不使能机械臂、不转发硬件命令、不发布相机外参，也不允许
执行轨迹。完成手眼标定后再补入外参并解除对应安全开关。

## 仿真和真机切换

先在当前启动终端按 `Ctrl+C`，然后运行另一种模式：

```bash
# 切到仿真
cd /home/qi/Project/piper_elevator
./scripts/button_approach_sim.sh
```

```bash
# 切到真机安全模式
cd /home/qi/Project/piper_elevator
./scripts/button_approach_real.sh
```
