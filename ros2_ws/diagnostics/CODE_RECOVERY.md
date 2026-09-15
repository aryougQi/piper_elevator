# 2026-09-13 代码恢复与接口验证

## 最终范围

用户确认“视觉”指按钮识别，不是 Servo。

- 按钮识别以本地 Git `ffe3646` 的 `detector_core.py` 为基线：YOLO ONNX、严格类别匹配、原有时序框跟踪、中位数/MAD 深度与局部深度平面法向。
- 粗定位、Servo、按压、控制门和任务调度恢复 9 月 2 日之后的修改。原先 `3fc3351` 是对 `0bd4c5e` 的整批回退，不仅撤回了识别，也撤回了运动控制。
- Git 中没有保存后续未提交的主程序，恢复依据为本地任务记录中的文件补丁和已检查的文本替换。重建先在 `/tmp/piper_code_recovery` 完成，然后用保留的回归测试验证。
- 粗定位保留稳定观测、可见性 IK 候选、全关节限位、轨迹与实际到位验收、运动质量优化、一次性 Servo 交接；同时保留 9 月 11 日相机安装变换与图像使用同一时间戳的修复。
- Servo 保持新版连续控制和与按压的握手；没有采用旧版 Servo/LIN 交接流程。

## 识别接口适配

识别算法的新增变化仅用于与当前下游契约衔接：

1. 增加 `/button_tracking_state`（`std_msgs/String` JSON），携带图像 `frame_id`、`stamp`、当前类别、稳定性、深度/表面是否有效以及已观测类别。漏检或尚未稳定不能发布正向语义证据。
2. 将相机系三维位置和法向滤波系数设为 1.0；滤波由粗定位和 Servo 在 `base_link` 下完成，避免把机械臂运动前后的相机坐标混合后标成当前帧。原二维框跟踪仍采用旧流程。
3. RGB-D 同步容差为 20 ms；保留当前 RealSense 启动参数、图像模式和相机编号接口。

旧识别没有新版自适应 ROI、投影重识别、加权深度、全局面板几何推断或投影冲突分类。Servo 的相关保护代码仍保留，但旧检测器不会虚构这些额外观测证据；丢检、目标跳变、观测过期、粗定位凭证和控制权保护继续生效。

| 连接 | 契约 |
| --- | --- |
| 选择 → 识别/Servo | `/button_selection`，`std_msgs/String`；检测器以 `/button_selected` 确认 |
| 识别 → 粗定位/Servo | `/button_surface_pose`，`geometry_msgs/PoseStamped`；同帧按钮中心与表面接近方向，+Z 指向面板 |
| 识别 → 任务管理 | `/button_detection_valid`，`std_msgs/Bool`；同时等待新表面观测和节点 READY 状态 |
| 识别 → Servo | `/button_tracking_state`，带帧与时间戳的 JSON |
| 粗定位 → Servo | `/button_approach_planner/claim_servo`，`std_srvs/Trigger`；schema 1 JSON，一次性领取，检查选择、时间和位姿 |
| Servo → 按压 | `/button_visual_servo/completed` + `/button_visual_servo/claim_for_press`；完成边沿与控制权握手；`continuous_servo_handoff: true` |
| 按压 → 任务管理 | `/button_press/completed` 与 `/button_press/status` |

真机组合入口与独立粗定位入口统一保持 `publish_camera_tf`、`camera_calibration_valid`、`allow_execution` 默认关闭；历史手眼参数并不等于已通过当前验证。未运行任何真机动作。

## 验证

- 应用与 Gazebo 现行完整测试：**973 passed**，见 [final_tests.txt](data/code_recovery/final_tests.txt)。
- 新增识别接口行为测试覆盖同帧位置/法向/状态、深度失效、漏检、错误类别，以及新版 Servo 的语义恢复门槛。
- 编译安装：`piper_elevator_app`、`piper_elevator_gazebo` 两包成功，见 [build.txt](data/code_recovery/build.txt)。
- 安装入口与启动参数检查见 [entrypoints.txt](data/code_recovery/entrypoints.txt)。
- 测试容器禁用网络、不挂载设备；使用实际 ROS Humble 消息和现有依赖。未完成实际相机识别效果、Gazebo 动态任务或机械臂运动验证。

已撤回的新版识别实验及其测试原样保存到 [superseded_detector](data/code_recovery/superseded_detector/README.md)。归档测试使用 `.py.txt` 扩展名避免被默认测试发现，原路径及 SHA256 见该目录 manifest.json。粗定位、Servo、按压和安全测试均保留在现行测试集。

## 使用

编译安装已完成。重新启动所需 ROS 节点后可使用带确认的选择命令：

```bash
source /workspace/ros2_ws/install/setup.bash
ros2 run piper_elevator_app button_select 3
ros2 service call /button_approach_planner/plan std_srvs/srv/Trigger "{}"
```

选择与规划不会自动执行轨迹；现行执行开关与标定要求仍适用。完整系统运行方式参见项目 `develop.md` 和 launch 参数。
