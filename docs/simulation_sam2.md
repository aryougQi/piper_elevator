# 仿真：全轨迹视野检查与 SAM2 严格交接

## 启动

在项目目录执行：

```bash
./scripts/build.sh
./scripts/elevator_task.sh
```

无界面启动：

```bash
./scripts/elevator_task.sh gazebo_gui:=false use_rviz:=false
```

入口和默认 ROS_DOMAIN_ID=42 保持一致。Docker 镜像须包含 SAM2 和 tiny 权重；
本次修复复用 `piper-elevator:humble`，ROS 配置与代码通过挂载工作空间更新。
构建脚本使用 Humble 的构建工具，避免 PyTorch 安装的新 setuptools 与 colcon 冲突。
SAM2 launch 为该进程优先设置 PyTorch 自带 CUDA/cuDNN 库，YOLO 仍使用原 ONNX 库。

## 两项逻辑

1. 粗定位对候选轨迹以 0.03 秒间隔检查相机投影，保留 60 像素边距。
   使用控制器对应的线性、三次或五次插值，未来末端 FK 加固定相机外参得到未来相机姿态。
   当前 Humble 的 VisibilityConstraint 采样器在该场景崩溃，因此用强制轨迹验收和
   最多 5 次重规划实现约束；未通过的轨迹不会执行。
2. SAM2 随仿真启动并加载模型；粗定位期间待机。收到验证完成状态后等待 0.3 秒，
   用选中按钮的最新 YOLO 框初始化。连续 5 个不同时间戳的有效掩膜确认 tracking_valid；独立 geometry worker 获得有效深度与平面后才发布 tracker_ready。
   仿真任务管理器和 Servo 强制 `require_sam2_tracking=true`，YOLO 不能替代 SAM2 放行或更新 Servo 位姿。

按钮在画面边缘被裁切时，掩膜质心并不等于物理按钮中心。SAM2 初始化时登记所选中心，
每帧用采集时刻的相机 TF、当前掩膜和深度平面重新验证该物理点；可见区域必须仍支持
同一按钮位置。深度平面提供每帧法向与法向位置修正，切向中心保持登记值。
这一过程要求新的分割与深度证据，不使用 YOLO 回退或失视盲走。

短暂观测间隔超限时发送零速度。单帧深度或平面失败只发布 geometry_invalid，不触发 SAM2 LOST，也不刷新最后有效 3D target 的时间戳。
跟踪丢失、图像或有效几何状态超时、持续深度无效时，Servo 停止，本次任务失败并进入已有恢复流程，不继续按压。换目标、停止、结束后清理跟踪状态。

现有实机入口默认 `require_sam2_tracking=false`，本次仅验证 Gazebo。
仿真适配器将固定 0.1 位置增益的补偿调至 10，仍保留 0.08 rad 单次补偿上限与关节限位。

## 验证

另一个项目终端执行：

```bash
./scripts/test_elevator_task.sh --execute --buttons up --cycles 3
./scripts/test_elevator_task.sh --execute --cycles 1
```

测试和启动须使用同一 ROS_DOMAIN_ID。诊断采样器及丢失注入工具位于
`ros2_ws/diagnostics/scripts/`，丢失注入工具仅允许本次验收使用的隔离域 73。
验收日志、图像、任务 CSV 和逐帧投影记录位于 `ros2_ws/diagnostics/data/sam2_fov_validation/`。
最终结果以该目录及诊断报告为准，不能只以节点启动成功判断通过。

## SAM2 低延迟调度

RGB 回调只替换最新帧并唤醒推理 worker；推理不在相机回调中执行。
mask 就绪后立即发布 center 和 tracking_valid，随后分别覆盖 geometry/debug 的单槽任务。
geometry 默认最高 15 Hz，debug 默认最高 3 Hz 且无订阅时不渲染。
geometry 和 debug 均不持有推理锁做 TF 等待、平面拟合或绘图；换目标后的在途几何结果被丢弃。
编译仍使用已验证的 torch.compile 设置，禁用 CUDA Graph 和激进 autotune。

粗定位验证完成、SAM2 稳定且 Servo 成功领取交接结果并启动后，YOLO 跳过模型推理；
手动等待 Servo 启动期间仍保留 YOLO，供粗定位交接执行新鲜近距离观测检查。LOST、换目标、任务退出、
重新粗定位或 1 秒收不到新鲜 SAM2 状态均恢复 YOLO。粗定位规划与交接门槛不变。
Servo 的 SAM2 位姿队列深度为 1，循环同时检查接收间隔和采集时间戳年龄；
tracking_valid 不会把旧几何伪装成新 target。几何超时仍通过已有停机和恢复路径处理。

性能证据见 `ros2_ws/diagnostics/data/sam2_decoupled_20260915/`；
其中 `synchronous_baseline/` 是诊断专用对照代码，不参与默认启动。
