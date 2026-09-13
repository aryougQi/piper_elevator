# 电梯按钮任务栈：接口 / 逻辑链 / 约束手册

面向 vibe-coding 的速查文档。每个模块给出：角色、入口、发布/订阅/服务、逻辑链函数、约束与不变量。
参数默认值以 `config/*.yaml` 为准；本文只收录安全相关与容易踩坑的那些。

---

## 0. 节点总览

| 节点 | 可执行文件 | 常驻 | 角色 |
|---|---|---|---|
| `button_detector` | `yolo_button_detector.py` | 是 | RGB-D 检测 → 相机系按钮位姿/法向/跟踪状态 |
| `button_approach_planner` | `button_approach_planner.py` | 是 | IK 候选 → MoveIt 规划/执行粗定位 → 一次性交接令牌 |
| `button_visual_servo` | `button_visual_servo.py` | 是 | 近景视觉伺服对准到安全站位 → 交接给按压 |
| `button_press_executor` | `button_press_executor.py` | 是 | 接触搜索 → 触发行程 → 保持 → 撤回 |
| `elevator_task_manager` | `elevator_task_manager.py` | 是 | 任务状态机（默认入口：`press <button>`） |
| `piper_pika_control_gate` | `control_gate.py` | 仅真机 | 硬件使能门控（轨迹 / Servo 互斥 + 心跳） |
| `piper_pika_joint_state_mux` | `joint_state_mux.py` | 仅真机 | 机械臂 + 夹爪关节状态合成 `/control/joint_states` |
| `simulation_servo_adapter` | `simulation_servo_adapter.py` | 仅仿真 | Servo 位置目标超前补偿（Gazebo 增益固定 0.1） |
| 工具 | `button_select.py` / `mock_button_pose.py` / `pika_fisheye_camera.py` | 否 | 选择按钮 / 假位姿 / 鱼眼相机桥 |

纯逻辑核（不跑 ROS，可单测）：`detector_core`（检测/跟踪/深度/法向几何）、`motion_core`（伺服误差/捕捉检查/四元数工具）、`coarse_approach_core`（候选视角合法性）、`handover_core`（交接令牌编解码/新鲜度）、`press_core`（力矩检测、堵转检测、相位计时）、`joint_limits_core`（限位合并/归一化）、`control_gate_core`（门控策略）、`servo_adapter_core`（Lead 补偿）、`approach_quality`、`task_core`、`launch_mode`。

---

## 1. 端到端时序（`elevator_task_manager._run_task`，elevator_task_manager.py:344）

```
命令 /elevator_task/command ("press 1")
 └─ _command_callback           解析按钮 → 起线程跑 _run_task
    ├─ WAITING_FOR_NODES        _wait_for_required_services + _ensure_unique_nodes
    ├─ HOMING_INITIAL           planner ~/return_home（return_home_before_task）
    ├─ SELECTING_BUTTON         manager 发 /button_selection；_select_and_wait_for_target
    │                           等检测器 stable + planner 状态 TARGET_READY
    ├─ COARSE_PLANNING          planner ~/plan      → 存 _planned_*，status=PLAN_READY
    ├─ COARSE_EXECUTING         planner ~/execute   → MoveIt 执行 + 近景复核
    │                           status=APPROACH_REACHED_VERIFIED（并生成一次性交接令牌）
    ├─ WAITING_FOR_VISUAL_TARGET _wait_for_post_motion_target（新帧 + 连续表面位姿）
    ├─ VISUAL_SERVO             /button_visual_servo/start（阻塞到对准结束）
    ├─ PRESSING                 /button_press_executor/start
    └─ HOMING_FINAL             planner ~/return_home
 失败 → _recover(at_home)：停 press/视觉 → clear_plan → 回原点（busy 时重试 recovery_retry_seconds）
 结束 → /elevator_task/result + /elevator_task/completed(success and at_home，锁存)
```

---

## 1.5 三个交接契约（改代码时最容易踩的地方）

| 交接 | 提供方 → 消费方 | 介质 | 一次性 | 时效 | 失败表现 |
|---|---|---|---|---|---|
| 粗定位 → 视觉伺服 | planner `~/claim_servo` → servo `_claim_coarse_handover` | `Trigger` 返回 JSON 令牌（schema 1） | 是（claim 后清空 `_verified_handover`） | `handover_max_age_seconds: 120`、令牌内观测新鲜度、关节/TCP 复核 | `No verified coarse handover` / `re-run coarse approach` |
| 视觉伺服 → 按压 | servo `/button_visual_servo/completed`（锁存 `Bool`）+ `~/claim_for_press`（`Trigger`） | 锁存话题 + 服务 | 锁存值会被撤销 | `press_claim_timeout_seconds`（仿真 60 s） | press 报 `Visual servo has not completed the safe alignment` |
| 按压接管确认 | press `/button_press/servo_claimed`（锁存 `Bool`） | 锁存话题 | 每次按压一个脉冲 | 伺服侧 `handover_claim_timeout_seconds` 内确认 | `Visual command release was not confirmed` |

共同不变量：**同一时刻只有一个节点拥有 Servo 会话**（`_owns_servo` / `_press_claim_event`），交接必须是"先释放再接管"，禁止两端同时发 twist。

---

## 2. `button_detector`（yolo_button_detector.py）

**订阅**

| 话题 | 类型 | 说明 |
|---|---|---|
| `color_topic` `/camera/color/image_raw` | `Image` | 彩色图（`reliable_input`/sensor QoS 由仿真或相机决定） |
| `depth_topic` `/camera/aligned_depth_to_color/image_raw` | `Image` | 与彩色对齐的深度；`use_depth: true` 时与彩色做 `ApproximateTimeSynchronizer`（`sync_slop_seconds: 0.02`） |
| `camera_info_topic` `/camera/color/camera_info` | `CameraInfo` | 内参（投影/法向拟合必需） |
| `button_selection_topic` `/button_selection` | `String`（锁存） | 操作者/管理器的目标选择 |

**发布**

| 话题 | 类型 | 说明 |
|---|---|---|
| `button_pose_topic` `/button_pose` | `PoseStamped` | 相机系位置（仅位置） |
| `button_surface_pose_topic` `/button_surface_pose` | `PoseStamped` | 相机系位置 + 由深度拟合的表面法向 → **planner 与 visual servo 的唯一近景观测源** |
| `button_pixel_topic` `/button_pixel` | `PointStamped` | 像素中心 + 尺度 |
| `button_detections_topic` `/button_detections` | `Detection2DArray` | 全部检测 |
| `button_valid_topic` `/button_detection_valid` | `Bool` | 当前帧有效 |
| `button_confidence_topic` `/button_detection_confidence` | `Float32` | 选中类置信度 |
| `button_selected_topic` `/button_selected` | `String`（锁存） | 当前选中类（`<none>` 表示无） |
| `tracking_state_topic` `/button_tracking_state` | `String` JSON（锁存） | `reason` + `stable_detection/depth_valid/surface_valid` → **伺服用语义冲突来源** |
| `debug_image_topic` `/button_detector/debug_image` | `Image` | 调试画面（仅被订阅时发布） |

**逻辑链**：`_rgbd_callback`(423) → `_process_frame`(501) → `_detect`(637, ONNX + ROI) → `filter_detections_by_class` → `_tracker.update`（稳定/漏检） → `robust_box_depth` → `project_pixel` → `estimate_surface_normal` → `_smooth_*` → `_publish_pose` / `_publish_surface_pose`(804) / `_publish_tracking_state`(745) / `_publish_detections` / `_publish_pixel` / `_publish_state` / `_publish_debug`。

**约束**

- 只有 `selected & stable & depth_valid & surface_valid` 才发 `/button_surface_pose`（消费方按"缺帧=丢失"处理）。
- `required_stable_frames: 5`、`max_missed_frames: 2`、`tracking_minimum_iou: 0.15`、`max_center_jump_ratio: 0.10`。
- `min_depth_m: 0.10`、`max_depth_m: 2.00`、`surface_minimum_samples: 30`、`surface_max_residual_m: 0.004`、`surface_max_tilt_degrees: 60`。
- 仿真：`simulation_layout_relabel: true` + 低阈值（launch 传 `confidence_threshold: 0.05`），与真机模型行为不同。

---

## 3. `button_approach_planner`（button_approach_planner.py）

**订阅**：`/button_surface_pose`、`/button_selected`（String）、`/joint_states`（`joint_state_topic`）、`/camera_info`。

**发布**：`/button_base`、`/button_approach_pose`（PoseStamped）、`/button_approach/status`（String 锁存）、`~/observation_status`、`/display_planned_path`（RViz）。

**服务（server，全部 `Trigger`）**：`~/plan`、`~/execute`、`~/return_home`、`~/clear_plan`、`~/claim_servo`。

**客户端**：`/compute_ik`（`GetPositionIK`）、`/compute_fk`（`GetPositionFK`）、`/plan_kinematic_path`（`GetMotionPlan`）、`/move_group/get_parameters`（取 URDF 限位）。

**逻辑链**

```
~/plan  _plan_callback(1020)
  ├─ (sim) _close_gripper(2698)
  ├─ _wait_for_planning_observation(1133)        稳定近景观测
  ├─ _candidate_poses(1323)                      tilt/roll/offset 网格；tilt 采样留 candidate_tilt_margin_rad
  ├─ _ik_search(...)                             多种子调用 /compute_ik
  │    └─ _joint_configuration_is_safe(1544)     joint_limit_margin_rad + 腕部奇异带过滤
  ├─ _plan_constraints(2529) → MoveIt 规划
  └─ _validate_planned_candidate(1554)           端点/可见性/倾角/滚转/时长
     成功 → 存 _planned_trajectory / _planned_target / _planned_button / _planned_observation

~/execute  _execute_coarse_target
  ├─ 校验 plan 新鲜度 + 目标漂移（max_target_drift_m）
  ├─ _execute_trajectory(2760)                   MoveIt FollowJointTrajectory
  └─ _verify_approach_reached(2301)              近景新帧 + TCP/姿态复核 → APPROACH_REACHED_VERIFIED

~/claim_servo  _claim_servo_handover_callback(2174)
  ├─ 校验 token 时效（handover_max_age_seconds）、按钮一致、关节未移动、TCP TF 新鲜
  └─ 返回一次性 JSON 令牌（schema 1：button/normal/tcp/joint_positions/observation_stamp_ns）
     并清空 _verified_handover（只能用一次）
```

**约束**

- `joint_limit_margin_rad: 0.15` 必须 ≥ `servo_joint_limit_margin_rad: 0.10`，否则 `_load_arm_joint_limits` 报错。
- `candidate_tilt_rad(5°) < maximum_camera_tilt_rad(10°) < handover_maximum_camera_tilt_rad(15°)`；滚转同理（candidate 10° < 15°）。候选采样在限值内侧留 `candidate_tilt_margin_rad: 0.5°`，避免 IK 残差把自己判越界。
- `handover_minimum_standoff_m: 0.08`、`maximum_start_error_m`、`execution_joint_tolerance_rad`、`plan_max_age_seconds`。
- `_busy` 单飞：规划/执行/回原点互斥；执行被中止时 `_busy` 会在回调 finally 释放（恢复动作需要重试，见 manager）。

---

## 4. `button_visual_servo`（button_visual_servo.py）

**订阅**：`/button_surface_pose`、`/button_selection`、`/button_tracking_state`（语义冲突）、`/servo_node/status`（`Int8`）、`/joint_states`（腕部限位保护）、`/button_press/servo_claimed`（`Bool` 锁存）。

**发布**：`/button_visual_servo/status`（String 锁存）、`/button_visual_servo/target_pose`（PoseStamped）、`/button_visual_servo/completed`（Bool 锁存）、`/servo_node/delta_twist_cmds`（TwistStamped）。

**服务（server）**：`~/start`、`~/stop`、`~/claim_for_press`。**客户端**：`/servo_node/start_servo`、`pause_servo`、`unpause_servo`、`/piper_pika_control_gate/servo_enable`（SetBool 心跳）、planner `~/claim_servo`。

**逻辑链**

```
~/start  _start_callback(860)   允许执行/标定/站位/调平参数检查 → 非忙
  ├─ _claim_coarse_handover(837) 调 planner ~/claim_servo → decode_coarse_handover
  ├─ 用令牌的 button/normal 作为锚点与初始观测（不再信任旧帧）
  └─ 起线程 _run_servo(1200) 并阻塞等待对准终态（_await_alignment_result(2751)）

_run_servo(1200)
  ├─ 启动 MoveIt Servo（start_servo + unpause）→ _set_hardware_servo_gate(2434) 心跳
  └─ _track_visually(1424)               50 Hz 相位机
       ├─ 观测：_surface_pose_callback(542) 缓存 / 丢失时按锁定目标继续
       ├─ 当前位姿：_current_servo_pose(2644)（TF 指尖 + 相机，含时间戳年龄/超前检查）
       ├─ 误差：visual_servo_errors(axial/lateral/angular) + level roll
       ├─ 相位：ORIENTING → REACQUIRING → FINAL_APPROACH
       ├─ 指令：P 控制 / 丢失时恒速盲走；平滑 + 加速度/工作空间/速度上限
       └─ 完成 → publish completed=True → _signal_alignment_finished(2737)
            → _hold_for_press_claim(1379) 等 press 接管
~/claim_for_press  _claim_for_press_callback(1064) 释放 twist 发布权 → press 拥有会话
```

**约束**

- 相位门槛：`required_alignment_observations`、`required_locked_alignment_cycles`、`required_post_orientation_observations`、`required_stable_observations`、`required_locked_target_stable_cycles`。
- 验收容差：`distance_tolerance_m: 2.5mm`、`lateral_tolerance_m: 3.0mm`、`perpendicular_tolerance_rad: 3°`、`level_roll_tolerance_rad: 3°`（腕部贴近限位时按 `wrist_limit_guard_*` 放宽到 6° 并冻结滚转）、`workspace_min/max`。
- 观测/丢失：`target_max_age_seconds`、`expected_observation_gap_seconds`、`observation_timeout_seconds`、`vision_loss_continuation_seconds` + `vision_loss_max_travel_m` + `vision_loss_stall_seconds`（恒速盲走，进度/停滞驱动）。
- TF：`tf_timeout_seconds`、`maximum_tf_fallback_age_seconds`、仿真另有 `simulation_future_stamp_tolerance_seconds`。
- Servo 安全：`/servo_node/status` 5（关节限位停机）/7（碰撞）立即失败；`_servo_safety_failure(1160)` 还检查"命令已开始但 Servo 未确认"。
- 语义冲突：`/button_tracking_state` 的 `reason` 直接进 `SEMANTIC_CONFLICT_HOLD`，重复帧不能解除。
- 交接窗口：`press_claim_timeout_seconds: 3.0`（仿真 `simulation_press_claim_timeout_seconds: 60`）；超时会释放会话并把 `completed` 置回 false。
- `~/start` 现在**阻塞到对准终态**才返回（`success` 即"已对准"）；上限 `start_response_timeout_seconds`。

---

## 5. `button_press_executor`（button_press_executor.py）

**订阅**：`/button_visual_servo/completed`（Bool 锁存）、`/feedback/joint_states`（`effort_topic`，力矩模式用）、`/servo_node/status`、`/button_selected`、仿真按钮触点话题列表（`Contacts`）。

**发布**：`/button_press/status`（锁存）、`/button_press/completed`（Bool 锁存）、`/button_press/timing`（锁存）、`/button_press/servo_claimed`（Bool 锁存）、`/servo_node/delta_twist_cmds`。

**服务（server）**：`~/start`、`~/stop`；**客户端**：`/button_visual_servo/claim_for_press`、Servo start/pause/unpause、硬件门控。

**逻辑链**

```
~/start  _start_callback(547) 前置：allow_execution、仿真按钮合法性、visual completed 且按钮一致、
                             接触模式所需的力矩标定、几何行程 ≤ maximum_approach_travel_m
_run_press(801)
  ├─ claim 视觉交接（连续会话，不 pause/unpause）
  ├─ 发布 /button_press/servo_claimed=True，接管零速度会话
  ├─ _settle_servo_origin           确认稳定原点 + 按压力向
  ├─ _approach_until_contact(1010)  仿真=触点；真机=力矩/堵转；每周期 _guard_motion(1361)
  │    到达上限：几何按压模式视为按到位（GEOMETRY_PRESS_REACHED），否则报 contact not detected
  ├─ 真机：_advance_to_travel(1178) 追加 press_extension_m（堵转则 PRESS_BOTTOMED_OUT 提前停）
  ├─ _hold_with_torque_monitor(1318) 保持 hold_seconds（力矩模式监控急停）
  ├─ _retract_to_start(1228)        快退 + 末端减速
  └─ 仿真再 _wait_for_simulated_release(1280)
成功 → /button_press/completed=True + status 'COMPLETE: button=<x> pressed and retracted'
```

**接触判据模式（`contact_detection_mode`）**

| 模式 | 需要力矩标定 | 判据 |
|---|---|---|
| `torque`（默认） | 是 | 六关节 `joint_torque_delta_thresholds_nm` + 基线 + 连续样本 |
| `stall` | 否 | 命令前进但指尖不再前进（`StallContactDetector`，press_core.py） |
| `stall_or_torque` | 是 | 两者任一 |

**约束**：`maximum_lateral_drift_m: 3mm`、`maximum_direction_change_rad: 4°`、`maximum_approach_travel_m: 38mm`、`maximum_lateral_correction_speed_mps: 6mm/s`、`motion_timeout_seconds: 15s`、`press_extension_m: 2.5mm`、`hold_seconds: 0.3s`、`emergency_threshold_multiplier: 2.5`；几何按压另需 `geometry_press_surface_travel_m + press_extension_m ≤ maximum_approach_travel_m`。

---

## 6. `elevator_task_manager`（elevator_task_manager.py）

**订阅**：`/elevator_task/command`（String）、`/button_selected`、`/button_detection_valid`、`/button_surface_pose`、`/button_approach/status`、`/button_visual_servo/completed|status`、`/button_press/completed|status`。

**发布**：`/elevator_task/status`、`/elevator_task/result`、`/elevator_task/completed`（锁存）、`/elevator_task/active_button`、`/button_selection`（它自己发选择）。

**服务**：`~/stop`、`~/reset`；**客户端**：plan / execute / home / clear_plan / visual start&stop / press start&stop。

**逻辑链**：`_command_callback(274)` → `_run_task(344)`（相位见 §1）→ `_select_and_wait_for_target(462)` → `_wait_for_post_motion_target(497)` → `_start_and_wait_for_completion(550)` → `_call_trigger(595)`；失败 `_recover(620)`；输出 `_publish_status/result/completion`。

**约束**：`required_unique_nodes`（防止旧 launch 残留导致请求落到别的实例）、各相位超时（见 §8）、`return_home_before/after_failure`、`clear_selection_after_task`、所有输出锁存 + `task_sequence` 防串扰；`_recover` 对 `Planner is busy` 在 `recovery_retry_seconds` 内重试。

---

## 7. 真机专有 / 仿真专有

### `piper_pika_control_gate`（真机）

- 订阅：`/arm_controller/follow_joint_trajectory/_action/status`、`/control/joint_states`（命令）、`/feedback/joint_states`（实测）。
- 服务：`/piper_pika_control_gate/servo_enable`（`SetBool`，带心跳）；客户端：驱动 `/control_enable`、`/servo_control_enable`。
- 逻辑：`ControlGatePolicy` 保证 **轨迹 / Servo 二选一**；轨迹门保持到动作终态且实测收敛（`trajectory_settle_tolerance_rad: 0.010`）或硬超时 `maximum_trajectory_gate_seconds: 45`；Servo 需要 `servo_heartbeat_timeout_seconds: 0.75` 的心跳，Servo 授权期间拒绝轨迹请求。

### `piper_pika_joint_state_mux`（真机）

- `/feedback/joint_states`（机械臂）+ `/gripper/joint_state` → 合成 `/control/joint_states`（追加 `center_joint`）。

### `simulation_servo_adapter`（仿真）

- `/servo_node/raw_joint_trajectory` → `compensated_positions`（`lead = gain×(target-current)`，限幅 `maximum_lead_rad`，再夹到关节限位）→ `/arm_controller/joint_trajectory`。
- Gazebo 的 `position_proportional_gain` 固定 0.1，仿真里所有 Servo 运动都靠它补偿；真机没有这个节点。

### 仿真专有参数（真机忽略或为 1.0）

`simulation_mode`、`simulation_*_multiplier`、`simulation_*_timeout_seconds`、`simulation_future_stamp_tolerance_seconds`、`simulation_vision_loss_*`、`simulation_press_claim_timeout_seconds`、`simulation_contacts_topics`、`simulation_pressed_depth_m`、`simulation_layout_relabel`、`gazebo_controllers.yaml` 的 `constraints`（真机控制器文件没有这段）。

---

## 8. 超时与窗口速查

| 名称 | 默认 | 作用 |
|---|---|---|
| `press_claim_timeout_seconds` | 3.0（仿真 60） | 视觉对准完成后等 press 接管的窗口 |
| `handover_claim_timeout_seconds` | 2.0 | press 接管后伺服侧释放确认 |
| `handover_max_age_seconds` | 120 | 交接令牌最长可用年龄 |
| `target_max_age_seconds` | 0.75 | 观测新鲜度 |
| `observation_timeout_seconds` | 8.25 | 丢失后保持/重取上限 |
| `vision_loss_continuation_seconds` | 8.0（仿真 ≥30） | 盲走绝对上限（另有进度/行程约束） |
| `tf_timeout_seconds` | 0.25 | TF 查询与年龄上限 |
| `servo_timeout_seconds` | 90 | 视觉阶段总预算 |
| `start_response_timeout_seconds` | 120 | `~/start` 阻塞返回上限 |
| `motion_timeout_seconds` | 15 | press 单段运动预算 |
| `hold_seconds` | 0.3 | 按压保持 |
| `planning/execution/visual/press/home_timeout_seconds` | 45/60/120/120/60 | manager 相位预算 |
| `recovery_retry_seconds` | 5.0 | 恢复回原点对 `Planner is busy` 的重试窗口 |

---

## 9. 失败信息 → 根因速查

| 信息（status / log） | 含义 | 处理方向 |
|---|---|---|
| `Visual servo has not completed the safe alignment` | press 看到的 `completed` 不是当前 true | 是否错过 3 s 交接窗口；用 manager 或调大窗口 |
| `MoveIt Servo halted at a joint bound (status=5)` | 关节贴限位被 Servo 停机 | 抬高目标/加大 `joint_limit_margin_rad`（注意 IK 可解性）/腕部限位保护 |
| `RGB-D loss exceeded bounded Servo continuation` | 视觉丢失超过盲走预算 | 检查可见性（站位/FOV）、盲走速度与进度判据 |
| `plan rejected: … Camera tilt … exceeds limit` | 候选视角越界（IK 残差） | 采样余量 `candidate_tilt_margin_rad`；或放宽 `maximum_camera_tilt_rad` |
| `No acceptable coarse IK candidate` | 限位/奇异/可见性过滤后无解 | 看 `_ik_search_failure_detail` 的拒绝原因分布 |
| `contact not detected before travel limit` | 行程走完没检测到接触 | 换 `stall`/几何按压模式，或校准 standoff |
| `home rejected: Planner is busy` | 执行刚结束、planner 未释放 | 已有 `recovery_retry_seconds` 重试 |
| `Real press requires calibrated six-joint torque limits` | 真机力矩模式未标定 | 改 `contact_detection_mode: stall`（+ 可选 `geometry_press_enabled`） |
| `EXECUTION_FAILED: MoveIt execution error -4` | 控制器中止（容差/碰撞/负载） | 看 Gazebo/控制器容差、轨迹速度 |

---

## 10. 常用调试命令

```bash
# 全流程（推荐）
ros2 topic pub --once /elevator_task/command std_msgs/msg/String "{data: 'press 1'}"
# 命令语法（task_core.parse_task_command）：'press 1' / 'press:open' / 直接 'open' / 'stop'
# 单步
ros2 service call /button_approach_planner/plan    std_srvs/srv/Trigger "{}"
ros2 service call /button_approach_planner/execute std_srvs/srv/Trigger "{}"
ros2 service call /button_visual_servo/start       std_srvs/srv/Trigger "{}"   # 阻塞到对准结束
ros2 service call /button_press_executor/start     std_srvs/srv/Trigger "{}"
ros2 service call /button_approach_planner/return_home std_srvs/srv/Trigger "{}"
# 状态（都锁存）
ros2 topic echo --once /elevator_task/status
ros2 topic echo --once /button_visual_servo/status
ros2 topic echo --once /button_press/status
ros2 topic echo --once /button_approach/status
ros2 topic echo --once /button_tracking_state
# 手动调参（无需重编译）
ros2 param set /button_visual_servo press_claim_timeout_seconds 60.0
```
