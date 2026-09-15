"""Closed-loop position-based visual servo for the selected button."""

import json
import math
import threading
import time

from geometry_msgs.msg import PoseStamped, TwistStamped
import numpy as np
from piper_elevator_app.handover_core import decode_coarse_handover
from piper_elevator_app.motion_core import camera_level_roll_error
from piper_elevator_app.motion_core import check_servo_capture
from piper_elevator_app.motion_core import (
    orientation_prioritized_linear_command,
)
from piper_elevator_app.motion_core import position_in_workspace
from piper_elevator_app.motion_core import quaternion_error_rotation_vector
from piper_elevator_app.motion_core import quaternion_to_matrix
from piper_elevator_app.motion_core import tangential_spiral_offset
from piper_elevator_app.motion_core import (
    tool_orientation_for_camera_direction,
)
from piper_elevator_app.motion_core import visual_servo_errors
import rclpy
from sensor_msgs.msg import JointState
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Bool
from std_msgs.msg import Int8
from std_msgs.msg import String
from std_srvs.srv import SetBool
from std_srvs.srv import Trigger
from tf2_ros import Buffer
from tf2_ros import ConnectivityException
from tf2_ros import ExtrapolationException
from tf2_ros import LookupException
from tf2_ros import TransformListener


class ButtonVisualServo(Node):
    """Align the fingertip and stop at a configured safe standoff."""

    def __init__(self):
        super().__init__('button_visual_servo')
        self._declare_parameters()
        self._callback_group = ReentrantCallbackGroup()
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._running = False
        self._starting = False
        self._stop_pending = False
        self._owns_servo = False
        self._cleanup_confirmed = True
        # The start service must not report success while the alignment is
        # still running: callers (and operators) treat it as "aligned".
        self._alignment_finished = threading.Event()
        self._alignment_result = (False, 'visual servo has not run')
        self._alignment_thread = None
        self._handoff_ready = False
        self._handoff_pending = False
        self._handoff_abort = False
        self._handoff_deadline = 0.0
        self._handoff_released = threading.Event()
        self._press_release_requested = threading.Event()
        self._selected_button = self._string_parameter('initial_selected_button').strip()
        self._selection_changed_stamp_ns = 0
        self._selection_generation = 0
        self._semantic_status_stamp_ns = 0
        self._semantic_conflict_stamp_ns = 0
        self._semantic_positive_stamp_ns = 0
        self._semantic_conflict_reason = ''
        self._semantic_conflict_started_at = 0.0
        self._observation_stamp_ns = 0
        self._observation = None
        self._observation_anchor = None
        self._filtered_world_position = None
        self._filtered_world_normal = None
        self._observation_sequence = 0
        self._press_claim_event = threading.Event()
        self._servo_started = False
        self._servo_status_code = None
        self._servo_status_received_at = 0.0
        self._wrist_guard_position = None
        self._wrist_guard_state_received_at = 0.0
        self._wrist_guard_reported = False
        self._servo_command_started_at = 0.0
        self._last_linear_command = np.zeros(3)
        self._last_angular_command = np.zeros(3)
        self._last_command_at = time.monotonic()
        self._last_gate_heartbeat = 0.0

        self._base_frame = self._string_parameter('base_frame')
        self._camera_frame = self._string_parameter('camera_frame')
        self._end_effector_link = self._string_parameter(
            'end_effector_link'
        )
        self._workspace_min = self._vector_parameter('workspace_min')
        self._workspace_max = self._vector_parameter('workspace_max')
        self._level_reference_axis = self._vector_parameter(
            'level_reference_axis'
        )
        level_axis_norm = float(np.linalg.norm(self._level_reference_axis))
        if level_axis_norm < 1.0e-9:
            raise ValueError('level_reference_axis must be non-zero')
        self._level_reference_axis /= level_axis_norm
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(
            self._tf_buffer,
            self,
            spin_thread=False,
        )

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._status_publisher = self.create_publisher(
            String,
            self._string_parameter('status_topic'),
            latched_qos,
        )
        self._target_publisher = self.create_publisher(
            PoseStamped,
            self._string_parameter('target_pose_topic'),
            latched_qos,
        )
        self._completion_publisher = self.create_publisher(
            Bool,
            self._string_parameter('completion_topic'),
            latched_qos,
        )
        self._twist_publisher = self.create_publisher(
            TwistStamped,
            self._string_parameter('servo_twist_topic'),
            10,
        )
        self.create_subscription(
            PoseStamped,
            self._string_parameter('surface_pose_topic'),
            self._yolo_surface_callback,
            10,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            PoseStamped,
            self._string_parameter('sam2_surface_pose_topic'),
            self._sam2_surface_callback,
            1,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            String,
            self._string_parameter('button_selection_topic'),
            self._button_selection_callback,
            latched_qos,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            String,
            self._string_parameter('tracking_state_topic'),
            self._tracking_state_callback,
            10,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            Int8,
            self._string_parameter('servo_status_topic'),
            self._servo_status_callback,
            10,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            JointState,
            self._string_parameter('joint_state_topic'),
            self._joint_state_callback,
            20,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            Bool,
            self._string_parameter('press_servo_claim_topic'),
            self._press_servo_claim_callback,
            latched_qos,
            callback_group=self._callback_group,
        )
        self._servo_start_client = self.create_client(
            Trigger,
            self._string_parameter('servo_start_service'),
            callback_group=self._callback_group,
        )
        self._servo_pause_client = self.create_client(
            Trigger,
            self._string_parameter('servo_pause_service'),
            callback_group=self._callback_group,
        )
        self._servo_unpause_client = self.create_client(
            Trigger,
            self._string_parameter('servo_unpause_service'),
            callback_group=self._callback_group,
        )
        self._hardware_gate_client = self.create_client(
            SetBool,
            self._string_parameter('hardware_gate_service'),
            callback_group=self._callback_group,
        )
        self._coarse_handover_client = self.create_client(
            Trigger, self._string_parameter('coarse_handover_service'),
            callback_group=self._callback_group,
        )
        self.create_service(
            Trigger,
            '~/start',
            self._start_callback,
            callback_group=self._callback_group,
        )
        self.create_service(
            Trigger, '~/claim_for_press', self._claim_for_press_callback,
            callback_group=self._callback_group,
        )
        self.create_timer(0.1, self._check_pending_handoff, callback_group=self._callback_group)
        self.create_service(
            Trigger,
            '~/stop',
            self._stop_callback,
            callback_group=self._callback_group,
        )

        self._publish_completion(False)
        self._publish_status('WAITING_FOR_SURFACE_POSE')
        self.get_logger().info(
            'Button visual servo ready: '
            f'tip={self._end_effector_link}, '
            f'standoff={self._standoff_distance():.3f} m, '
            f'execution={self.get_parameter("allow_execution").value}'
        )

    def _declare_parameters(self):
        self.declare_parameter('coarse_handover_service', '/button_approach_planner/claim_servo')
        self.declare_parameter('handover_claim_timeout_seconds', 2.0)
        self.declare_parameter('stop_timeout_seconds', 8.0)
        self.declare_parameter('require_sam2_tracking', False)
        self._require_sam2 = bool(self.get_parameter('require_sam2_tracking').value)
        self._sam2_ready = False
        self._sam2_received_at = 0.0
        self.declare_parameter('surface_pose_topic', '/button_surface_pose')
        self.declare_parameter(
            'sam2_surface_pose_topic', '/sam2_button_tracker/surface_pose'
        )
        self.declare_parameter('tracking_state_topic', '/button_tracking_state')
        self.declare_parameter('initial_selected_button', '')
        self.declare_parameter('button_selection_topic', '/button_selection')
        self.declare_parameter(
            'target_pose_topic',
            '/button_visual_servo/target_pose',
        )
        self.declare_parameter('status_topic', '/button_visual_servo/status')
        self.declare_parameter(
            'completion_topic',
            '/button_visual_servo/completed',
        )
        self.declare_parameter(
            'press_servo_claim_topic',
            '/button_press/servo_claimed',
        )
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter(
            'end_effector_link',
            'pika_fingertip_center_link',
        )
        self.declare_parameter(
            'servo_twist_topic',
            '/servo_node/delta_twist_cmds',
        )
        self.declare_parameter('servo_status_topic', '/servo_node/status')
        self.declare_parameter(
            'servo_start_service',
            '/servo_node/start_servo',
        )
        self.declare_parameter(
            'servo_pause_service',
            '/servo_node/pause_servo',
        )
        self.declare_parameter(
            'servo_unpause_service',
            '/servo_node/unpause_servo',
        )
        self.declare_parameter(
            'hardware_gate_service',
            '/piper_pika_control_gate/servo_enable',
        )
        self.declare_parameter('hardware_gate_required', False)
        self.declare_parameter('hardware_gate_heartbeat_seconds', 0.20)

        self.declare_parameter('standoff_distance_m', 0.030)
        self.declare_parameter('minimum_standoff_m', 0.025)
        self.declare_parameter('distance_tolerance_m', 0.0025)
        self.declare_parameter('lateral_tolerance_m', 0.0030)
        self.declare_parameter(
            'perpendicular_tolerance_rad',
            math.radians(3.0),
        )
        self.declare_parameter('required_alignment_observations', 2)
        self.declare_parameter('required_locked_alignment_cycles', 5)
        self.declare_parameter(
            'required_post_orientation_observations',
            3,
        )
        self.declare_parameter('reacquisition_search_enabled', True)
        self.declare_parameter(
            'reacquisition_initial_hold_seconds',
            0.50,
        )
        self.declare_parameter('reacquisition_search_radius_m', 0.012)
        self.declare_parameter(
            'reacquisition_search_radial_speed_mps',
            0.003,
        )
        self.declare_parameter(
            'reacquisition_search_angular_speed_radps',
            1.50,
        )
        self.declare_parameter('reacquisition_search_speed_mps', 0.012)
        self.declare_parameter(
            'reacquisition_maximum_axial_drift_m',
            0.002,
        )
        self.declare_parameter('required_stable_observations', 2)
        self.declare_parameter('maximum_normal_change_rad', math.radians(10.0))
        self.declare_parameter('required_conflicting_normal_observations', 3)
        self.declare_parameter('maximum_start_error_m', 0.20)
        self.declare_parameter(
            'handover_maximum_camera_tilt_rad', math.radians(15.0),
        )
        self.declare_parameter(
            'handover_maximum_camera_roll_rad', math.radians(15.0),
        )
        self.declare_parameter('handover_minimum_standoff_m', 0.08)
        self.declare_parameter('maximum_target_jump_m', 0.015)
        self.declare_parameter('world_position_smoothing_alpha', 0.25)
        self.declare_parameter('world_normal_smoothing_alpha', 0.20)
        self.declare_parameter('servo_timeout_seconds', 90.0)
        # start 服务在对准真正结束（成功或失败）后才返回；该上限只需略大于
        # servo_timeout_seconds，正常路径永远不会用到。
        self.declare_parameter('start_response_timeout_seconds', 120.0)
        self.declare_parameter('target_max_age_seconds', 0.75)
        self.declare_parameter('observation_timeout_seconds', 8.25)
        self.declare_parameter('expected_observation_gap_seconds', 0.20)
        self.declare_parameter('vision_loss_continuation_seconds', 8.0)
        self.declare_parameter('vision_loss_speed_scale', 0.50)
        self.declare_parameter(
            'vision_loss_continuation_max_distance_m',
            0.10,
        )
        self.declare_parameter('vision_loss_max_travel_m', 0.080)
        # 盲走（RGB-D 丢失但目标已锁定）阶段使用恒速逼近：比例控制在误差
        # 变小时速度趋近于 0，仿真里 Servo 又只跟得上指令的一小部分，最后
        # 几毫米永远走不完。接近目标时按 ramp_gain 线性减速。
        self.declare_parameter('vision_loss_approach_speed_mps', 0.020)
        self.declare_parameter('vision_loss_approach_ramp_gain', 6.0)
        # 只要仍在朝锁定目标推进就继续盲走；连续停顿超过该时间才判定卡住。
        self.declare_parameter('vision_loss_stall_seconds', 3.0)
        # 仿真专用：Gazebo 实时率约 0.4、Servo 实际速度只有指令的约 10%，
        # 需要更快指令与更长绝对上限；真机（simulation_mode=false）不读取。
        self.declare_parameter(
            'simulation_vision_loss_approach_speed_multiplier',
            3.0,
        )
        self.declare_parameter(
            'simulation_vision_loss_continuation_seconds',
            30.0,
        )
        self.declare_parameter('required_locked_target_stable_cycles', 5)
        self.declare_parameter('servo_control_rate_hz', 50.0)
        self.declare_parameter('linear_proportional_gain', 1.8)
        self.declare_parameter('orientation_control_enabled', True)
        self.declare_parameter('angular_proportional_gain', 2.4)
        self.declare_parameter(
            'axial_approach_full_speed_angle_rad',
            math.radians(2.0),
        )
        self.declare_parameter(
            'axial_approach_stop_angle_rad',
            math.radians(3.0),
        )
        self.declare_parameter('level_roll_enabled', True)
        self.declare_parameter('level_reference_axis', [0.0, 0.0, 1.0])
        self.declare_parameter('target_level_roll_rad', 0.0)
        self.declare_parameter(
            'level_roll_tolerance_rad',
            math.radians(3.0),
        )
        # 腕部接近限位时（面板角落的 open/close 这类按钮），继续强行把滚转
        # 调到 3 度会把腕关节推过 MoveIt Servo 的限位裕度并触发 HALT。
        # 此时放宽调平容差，并在更贴近限位时冻结继续加压的滚转指令；
        # 正常情况仍按 level_roll_tolerance_rad = 3 度验收。
        self.declare_parameter('wrist_limit_guard_enabled', True)
        self.declare_parameter('wrist_limit_guard_joint', 'joint5')
        self.declare_parameter('wrist_limit_guard_lower_rad', -1.2217304)
        self.declare_parameter('wrist_limit_guard_upper_rad', 1.2217304)
        # 进入 relax 余量 → 放宽调平容差；进入 hold 余量 → 冻结滚转指令。
        self.declare_parameter('wrist_limit_guard_relax_margin_rad', 0.22)
        self.declare_parameter('wrist_limit_guard_hold_margin_rad', 0.14)
        self.declare_parameter(
            'level_roll_relaxed_tolerance_rad',
            math.radians(6.0),
        )
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter(
            'wrist_limit_guard_state_timeout_seconds',
            1.0,
        )
        self.declare_parameter(
            'maximum_level_roll_speed_radps',
            0.30,
        )
        self.declare_parameter('maximum_linear_speed_mps', 0.080)
        self.declare_parameter('maximum_angular_speed_radps', 0.35)
        self.declare_parameter('maximum_linear_acceleration_mps2', 0.30)
        self.declare_parameter(
            'maximum_angular_acceleration_radps2',
            1.20,
        )
        self.declare_parameter('command_smoothing_alpha', 0.50)
        self.declare_parameter('servo_deceleration_seconds', 0.25)
        self.declare_parameter('press_claim_timeout_seconds', 3.0)
        # 仅仿真生效：手动单步调试时，对准结束到 press 接管之间有更长的
        # 交接窗口（Servo 只是保持零速度）。真机保持 3 s 不变。
        self.declare_parameter(
            'simulation_press_claim_timeout_seconds',
            60.0,
        )
        self.declare_parameter('tf_timeout_seconds', 0.25)
        self.declare_parameter('exact_tf_wait_seconds', 0.03)
        self.declare_parameter('allow_latest_tf_fallback', False)
        self.declare_parameter('maximum_tf_fallback_age_seconds', 0.02)
        # 仿真时钟按物理步长推进，/clock 与 TF 时间戳可能相差一个步长，
        # 表现为 stamp 比节点当前时刻超前约 1 ms。真机
        # （simulation_mode=false）下该容差恒为 0，判定与原来完全一致。
        self.declare_parameter(
            'simulation_future_stamp_tolerance_seconds',
            0.02,
        )
        self.declare_parameter('action_timeout_seconds', 20.0)
        self.declare_parameter(
            'workspace_min',
            [-0.65, -0.65, 0.02],
        )
        self.declare_parameter(
            'workspace_max',
            [0.65, 0.65, 0.75],
        )
        self.declare_parameter('simulation_mode', False)
        self.declare_parameter('simulation_linear_speed_multiplier', 5.0)
        self.declare_parameter('camera_calibration_valid', False)
        self.declare_parameter('allow_execution', False)

    def _string_parameter(self, name):
        return str(self.get_parameter(name).value)

    def _vector_parameter(self, name):
        values = np.asarray(self.get_parameter(name).value, dtype=np.float64)
        if values.shape != (3,):
            raise ValueError(f'{name} must contain exactly three values')
        return values

    def _standoff_distance(self):
        return float(self.get_parameter('standoff_distance_m').value)

    def _future_stamp_tolerance(self):
        """Return how far a stamp may lead the node clock.

        Gazebo advances sim time in one-millisecond physics steps, so the
        /clock sample this node holds can be one step older than the stamp
        carried by TF and sensor messages.  Only the simulation stack gets
        that slack: real hardware keeps the strict rule that a stamp ahead
        of the node clock is invalid.
        """
        if not bool(self.get_parameter('simulation_mode').value):
            return 0.0
        return max(
            0.0,
            float(
                self.get_parameter(
                    'simulation_future_stamp_tolerance_seconds'
                ).value
            ),
        )

    def _lookup_surface_transform(self, source_frame, stamp):
        """Return the transform at image capture time.

        Pairing an old eye-in-hand image with the latest robot pose makes a
        static button appear to move with the camera.  Real visual Servo must
        therefore fail closed when the exact transform is unavailable.  A
        tightly bounded latest-TF fallback remains opt-in for diagnostics.
        """
        timeout = float(self.get_parameter('tf_timeout_seconds').value)
        if stamp.nanoseconds == 0:
            return self._tf_buffer.lookup_transform(
                self._base_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=timeout),
            )
        try:
            return self._tf_buffer.lookup_transform(
                self._base_frame,
                source_frame,
                stamp,
                timeout=Duration(
                    seconds=min(
                        timeout,
                        float(
                            self.get_parameter(
                                'exact_tf_wait_seconds'
                            ).value
                        ),
                    )
                ),
            )
        except (
            LookupException,
            ConnectivityException,
            ExtrapolationException,
        ):
            if not bool(
                self.get_parameter('allow_latest_tf_fallback').value
            ):
                raise
            transform = self._tf_buffer.lookup_transform(
                self._base_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=timeout),
            )
            transform_stamp = Time.from_msg(transform.header.stamp)
            age = abs(stamp.nanoseconds - transform_stamp.nanoseconds) / 1e9
            maximum_age = float(
                self.get_parameter('maximum_tf_fallback_age_seconds').value
            )
            if transform_stamp.nanoseconds > 0 and age > maximum_age:
                raise ExtrapolationException(
                    'latest transform is too far from the RGB-D frame: '
                    f'age={age:.3f}s limit={maximum_age:.3f}s'
                )
            return transform

    def _yolo_surface_callback(self, message):
        if not self._require_sam2:
            self._surface_pose_callback(message)

    def _sam2_surface_callback(self, message):
        if self._require_sam2:
            self._surface_pose_callback(message)

    def _sam2_failure(self):
        if getattr(self, '_require_sam2', False):
            if not self._sam2_ready or time.monotonic() - self._sam2_received_at > 1.0:
                return 'SAM2 tracking unavailable or stale'
        return None

    def _surface_pose_callback(self, message):
        if message.header.frame_id != self._camera_frame:
            self._publish_status('REJECTED: surface pose frame does not match camera_frame')
            return

        message_stamp_ns = Time.from_msg(message.header.stamp).nanoseconds
        with self._condition:
            selection_generation = self._selection_generation
            if (
                message_stamp_ns < self._selection_changed_stamp_ns
                or message_stamp_ns <= self._observation_stamp_ns
                or not self._surface_stamp_is_fresh(message_stamp_ns)
            ):
                # DDS may deliver a queued pose from the previous button just
                # after a new selection. Never let that stale pose establish
                # the world-space identity for the new task.
                return
        button_camera = np.asarray([
            message.pose.position.x,
            message.pose.position.y,
            message.pose.position.z,
        ])
        surface_quaternion = np.asarray([
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ])
        if (
            not np.all(np.isfinite(button_camera))
            or button_camera[2] <= 0.0
            or not np.all(np.isfinite(surface_quaternion))
            or np.linalg.norm(surface_quaternion) < 1.0e-6
        ):
            self._publish_status('REJECTED: invalid surface pose')
            return

        stamp = Time.from_msg(message.header.stamp)
        if message.header.stamp.sec == 0 and message.header.stamp.nanosec == 0:
            stamp = Time()
        try:
            transform = self._lookup_surface_transform(
                message.header.frame_id,
                stamp,
            )
        except (
            LookupException,
            ConnectivityException,
            ExtrapolationException,
        ) as error:
            self.get_logger().warning(
                f'Cannot transform surface pose to {self._base_frame}: '
                f'{error}',
                throttle_duration_sec=2.0,
            )
            self._publish_status('WAITING_FOR_SURFACE_TO_BASE_TF')
            return

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        base_from_camera = quaternion_to_matrix([
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        ])
        button_base = np.asarray([
            translation.x,
            translation.y,
            translation.z,
        ]) + base_from_camera @ button_camera
        normal_camera = quaternion_to_matrix(surface_quaternion)[:, 2]
        normal_base = base_from_camera @ normal_camera
        normal_base /= np.linalg.norm(normal_base)
        if np.dot(normal_camera, button_camera) < 0.0:
            normal_base = -normal_base

        received_at = time.monotonic()
        with self._condition:
            # A selection or newer callback can complete while TF lookup is
            # waiting. Commit only to the selection and frame we started on.
            if (
                selection_generation != self._selection_generation
                or message_stamp_ns <= self._observation_stamp_ns
                or not self._surface_stamp_is_fresh(message_stamp_ns)
            ):
                return
            if self._observation_anchor is not None:
                jump = float(np.linalg.norm(
                    button_base - self._observation_anchor
                ))
                maximum_jump = float(
                    self.get_parameter('maximum_target_jump_m').value
                )
                if jump > maximum_jump:
                    # This is an observation-level outlier, not a terminal
                    # task rejection. Keep the locked world-space target and
                    # let the next RGB-D frame recover tracking.
                    self._publish_status(
                        f'IGNORED_TARGET_JUMP: {jump:.3f} m'
                    )
                    return
            else:
                self._observation_anchor = button_base.copy()
            position_alpha = float(np.clip(
                self.get_parameter(
                    'world_position_smoothing_alpha'
                ).value,
                0.0,
                1.0,
            ))
            normal_alpha = float(np.clip(
                self.get_parameter('world_normal_smoothing_alpha').value,
                0.0,
                1.0,
            ))
            if self._filtered_world_position is None:
                self._filtered_world_position = button_base.copy()
            else:
                self._filtered_world_position = (
                    position_alpha * button_base
                    + (1.0 - position_alpha)
                    * self._filtered_world_position
                )
            if self._filtered_world_normal is None:
                self._filtered_world_normal = normal_base.copy()
            else:
                if np.dot(
                    normal_base,
                    self._filtered_world_normal,
                ) < 0.0:
                    normal_base = -normal_base
                self._filtered_world_normal = (
                    normal_alpha * normal_base
                    + (1.0 - normal_alpha)
                    * self._filtered_world_normal
                )
                self._filtered_world_normal /= np.linalg.norm(
                    self._filtered_world_normal
                )
            self._observation_sequence += 1
            self._observation_stamp_ns = message_stamp_ns
            capture_age = max(
                0.0,
                (self.get_clock().now().nanoseconds - message_stamp_ns) / 1e9,
            )
            self._observation = (
                self._filtered_world_position.copy(),
                self._filtered_world_normal.copy(),
                received_at - capture_age,
                self._observation_sequence,
            )
            self._condition.notify_all()
        if not self._running:
            self._publish_status('READY')

    def _tracking_state_callback(self, message):
        try:
            payload = json.loads(message.data)
            if not isinstance(payload, dict):
                return
            selected = payload.get('selected')
            if getattr(self, '_require_sam2', False) and payload.get('source') != 'sam2_button_tracker':
                return
            stamp = payload.get('stamp')
            if not isinstance(selected, dict) or not isinstance(stamp, dict):
                return
            if payload.get('source') == 'sam2_button_tracker':
                if not getattr(self, '_require_sam2', False):
                    return
                if selected.get('class_name') != self._selected_button:
                    return
                stamp_ns = int(stamp.get('sec', 0)) * 1000000000 + int(stamp.get('nanosec', 0))
                if stamp_ns and stamp_ns < self._selection_changed_stamp_ns:
                    return
                if payload.get('state') == 'TRACKING' and payload.get('reason') in ('tracking_valid', 'geometry_invalid'):
                    return  # Keep the last geometry timestamp; existing freshness checks expire it.
                ready = payload.get('state') == 'TRACKING' and payload.get('reason') == 'tracker_ready'
                fresh = self._surface_stamp_is_fresh(stamp_ns)
                self._sam2_ready = ready and fresh
                if self._sam2_ready:
                    self._sam2_received_at = time.monotonic()
                elif self._running or self._starting:
                    self._stop_event.set()
                    self._publish_zero_twist()
                    with self._condition:
                        self._condition.notify_all()
                    self._publish_status('STOPPED: SAM2 tracking lost')
                if not self._sam2_ready:
                    return
            label = selected.get('class_name')
            sec, nanosec = stamp.get('sec'), stamp.get('nanosec')
            if (
                not isinstance(label, str)
                or type(sec) is not int or type(nanosec) is not int
                or sec < 0 or not 0 <= nanosec < 1_000_000_000
                or payload.get('frame_id') != self._camera_frame
            ):
                return
            stamp_ns = sec * 1_000_000_000 + nanosec
            reason = payload.get('reason')
            negative = reason in ('direction_conflict', 'projection_conflict')
            measured = selected.get('measured')
            sam2_source = payload.get('source') == 'sam2_button_tracker'
            positive = (
                (reason == '' or (sam2_source and reason in ('tracker_ready', 'initialized')))
                and selected.get('stable_detection') is True
                and selected.get('depth_valid') is True
                and isinstance(measured, dict)
                and measured.get('class_name') == label
            )
            if not negative and not positive:
                return
        except (TypeError, ValueError, RecursionError):
            return
        with self._condition:
            if (
                not self._selected_button
                or label.strip().casefold() != self._selected_button.strip().casefold()
                or stamp_ns <= self._selection_changed_stamp_ns
                or not self._surface_stamp_is_fresh(stamp_ns)
                or stamp_ns < getattr(self, '_semantic_status_stamp_ns', 0)
            ):
                return
            if stamp_ns == getattr(self, '_semantic_status_stamp_ns', 0):
                # Negative evidence wins a same-capture disagreement. Repeated
                # positive messages cannot release that capture's hold.
                if not negative or stamp_ns == getattr(self, '_semantic_conflict_stamp_ns', 0):
                    return
            ButtonVisualServo._semantic_conflict_locked(self)
            self._semantic_status_stamp_ns = stamp_ns
            if negative:
                if not getattr(self, '_semantic_conflict_reason', ''):
                    self._semantic_conflict_started_at = time.monotonic()
                self._semantic_conflict_stamp_ns = stamp_ns
                self._semantic_conflict_reason = reason
                self._semantic_positive_stamp_ns = 0
            elif (
                not getattr(self, '_semantic_conflict_reason', '')
                or not ButtonVisualServo._fresh_semantic_positive_stamp_locked(self)
            ):
                # Retain the first fresh recovery capture until its surface
                # arrives. Moving this stamp on every status can starve a
                # surface callback that consistently runs one frame behind.
                self._semantic_positive_stamp_ns = stamp_ns
            hold = ButtonVisualServo._semantic_conflict_locked(self)
            self._condition.notify_all()
            if hold and self._running:
                self._publish_zero_twist()
                self._publish_status('SEMANTIC_CONFLICT_HOLD: ' + hold)

    def _fresh_semantic_positive_stamp_locked(self):
        stamp_ns = getattr(self, '_semantic_positive_stamp_ns', 0)
        if (
            stamp_ns > getattr(self, '_semantic_conflict_stamp_ns', 0)
            and self._surface_stamp_is_fresh(stamp_ns)
        ):
            return stamp_ns
        return 0

    def _semantic_conflict_locked(self):
        reason = getattr(self, '_semantic_conflict_reason', '')
        if not reason:
            return ''
        positive_stamp = ButtonVisualServo._fresh_semantic_positive_stamp_locked(self)
        if (
            positive_stamp > 0
            and self._observation_stamp_ns >= positive_stamp
            and self._observation_is_fresh_locked()
        ):
            self._semantic_conflict_reason = ''
            self._semantic_conflict_started_at = 0.0
            return ''
        return reason

    def _button_selection_callback(self, message):
        selected = str(message.data).strip()
        if selected.casefold() in {'clear', 'none'}:
            selected = ''
        with self._condition:
            if selected == self._selected_button:
                return
            self._selected_button = selected
            self._sam2_ready = False
            self._sam2_received_at = 0.0
            self._selection_generation += 1
            self._selection_changed_stamp_ns = (
                self.get_clock().now().nanoseconds
            )
            self._semantic_status_stamp_ns = 0
            self._semantic_conflict_stamp_ns = 0
            self._semantic_positive_stamp_ns = 0
            self._semantic_conflict_reason = ''
            self._semantic_conflict_started_at = 0.0
            # A button is static in the base frame.  Clear the world-space
            # identity only when the operator changes the requested button;
            # during coarse motion, reject any detector jump to another
            # same-name icon even if the old observation is no longer fresh.
            self._observation = None
            self._observation_anchor = None
            self._filtered_world_position = None
            self._filtered_world_normal = None
            if self._running:
                self._stop_event.set()
            self._condition.notify_all()
        self._publish_completion(False)

    def _press_servo_claim_callback(self, message):
        with self._condition:
            if message.data:
                self._press_claim_event.set()
                if self._handoff_pending and not self._stop_pending:
                    self._handoff_pending = False
                    self._owns_servo = False
                    self._cleanup_confirmed = True
                elif self._running:
                    self._stop_event.set()
            else:
                self._press_claim_event.clear()
                if self._handoff_pending:
                    self._handoff_abort = True
            self._condition.notify_all()

    def _claim_coarse_handover(self):
        timeout = float(self.get_parameter('handover_claim_timeout_seconds').value)
        client = self._coarse_handover_client
        started = time.monotonic()
        if not client.wait_for_service(timeout_sec=timeout):
            raise ValueError('Coarse handover service unavailable; start the approach planner')
        while True:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0.0 or self._stop_event.is_set():
                raise ValueError('Coarse handover request stopped or timed out')
            future = client.call_async(Trigger.Request())
            result = self._wait_for_future(future, remaining)
            if result is None:
                client.remove_pending_request(future)
                raise ValueError('Coarse handover request stopped or timed out')
            if result.success:
                break
            if result.message != 'Near-view observation changed during handover; retry':
                raise ValueError(result.message)
            # The planner retained its token and requested a new atomic check.
            # Never retry errors that indicate motion, identity, or stale data.
            self._stop_event.wait(min(0.05, max(0., remaining)))
        return decode_coarse_handover(
            result.message, selected_button=self._selected_button,
            frame_id=self._base_frame, now_ns=self.get_clock().now().nanoseconds,
            maximum_age_seconds=float(self.get_parameter('target_max_age_seconds').value),
            future_tolerance_seconds=float(self.get_parameter('maximum_tf_fallback_age_seconds').value),
        )

    def _start_callback(self, request, response):
        failure = ButtonVisualServo._sam2_failure(self)
        if failure:
            response.success = False
            response.message = failure
            return response
        del request
        if not bool(self.get_parameter('allow_execution').value):
            response.success = False
            response.message = 'Execution is disabled by allow_execution'
            return response
        if (
            not bool(self.get_parameter('simulation_mode').value)
            and not bool(
                self.get_parameter('camera_calibration_valid').value
            )
        ):
            response.success = False
            response.message = 'Real servo requires calibrated camera TF'
            return response
        if self._standoff_distance() < float(
            self.get_parameter('minimum_standoff_m').value
        ):
            response.success = False
            response.message = 'Configured standoff is below the safety limit'
            return response
        if bool(self.get_parameter('level_roll_enabled').value):
            target_roll = float(
                self.get_parameter('target_level_roll_rad').value
            )
            roll_tolerance = float(
                self.get_parameter('level_roll_tolerance_rad').value
            )
            if (
                not math.isfinite(target_roll)
                or not math.isfinite(roll_tolerance)
                or target_roll < 0.0
                or target_roll >= roll_tolerance
            ):
                response.success = False
                response.message = (
                    'Level-roll target must be non-negative and strictly '
                    'inside its acceptance tolerance'
                )
                return response

        with self._condition:
            if (self._running or self._starting or self._stop_pending or self._owns_servo
                    or self._handoff_pending or self._press_claim_event.is_set()):
                response.success = False
                response.message = 'Servo is busy, stopping, or owned by the press executor'
                return response
            conflict = ButtonVisualServo._semantic_conflict_locked(self)
            recovering_conflict = bool(conflict)
            recovery_positive_stamp = ButtonVisualServo._fresh_semantic_positive_stamp_locked(self)
            if conflict and not recovery_positive_stamp:
                response.success = False
                response.message = 'Fresh positive RGB-D evidence required after ' + conflict
                return response
            conflict_stamp_before_claim = getattr(self, '_semantic_conflict_stamp_ns', 0)
            self._starting = True
            self._stop_event.clear()
            generation = self._selection_generation
        try:
            try:
                handover = self._claim_coarse_handover()
            except (ValueError, RuntimeError) as error:
                if recovering_conflict:
                    raise ValueError(
                        'Fresh positive RGB-D evidence recovered, but no verified '
                        'coarse handover is available; re-run coarse approach: '
                        + str(error)
                    ) from error
                raise
            with self._condition:
                conflict = ButtonVisualServo._semantic_conflict_locked(self)
                if (
                    getattr(self, '_semantic_conflict_stamp_ns', 0)
                    > conflict_stamp_before_claim
                ):
                    raise ValueError(
                        'New semantic conflict during coarse handover: '
                        + (conflict or 'restart required')
                    )
                if recovering_conflict:
                    positive_stamp = recovery_positive_stamp
                    if not self._surface_stamp_is_fresh(positive_stamp):
                        positive_stamp = ButtonVisualServo._fresh_semantic_positive_stamp_locked(self)
                    if not positive_stamp:
                        raise ValueError(
                            'Fresh positive RGB-D evidence expired during coarse handover'
                        )
                    if handover['observation_stamp_ns'] < positive_stamp:
                        raise ValueError(
                            'Verified coarse observation predates positive RGB-D '
                            'recovery evidence; re-run coarse approach'
                        )
                elif conflict:
                    raise ValueError('Coarse handover blocked by ' + conflict)
                if (self._stop_event.is_set() or self._stop_pending
                        or generation != self._selection_generation
                        or self._press_claim_event.is_set()):
                    raise ValueError('Servo start invalidated while claiming the coarse handover')
                if not self._surface_stamp_is_fresh(handover['observation_stamp_ns']):
                    raise ValueError('Claimed coarse observation expired before Servo start')
                # Replace the far-view anchor only with this verified near-view
                # target. In-flight callbacks from the old filter are obsolete.
                self._selection_generation += 1
                self._observation_anchor = handover['button'].copy()
                self._filtered_world_position = handover['button'].copy()
                self._filtered_world_normal = handover['normal'].copy()
                self._observation_stamp_ns = handover['observation_stamp_ns']
                self._observation_sequence += 1
                age = max(0.0, (self.get_clock().now().nanoseconds - self._observation_stamp_ns) / 1e9)
                self._observation = (handover['button'].copy(), handover['normal'].copy(),
                                     time.monotonic() - age, self._observation_sequence)
                ButtonVisualServo._semantic_conflict_locked(self)
                self._active_handover_id = handover['handover_id']
                self._running = True
                self._cleanup_confirmed = True
                self._handoff_ready = False
                self._handoff_abort = False
                self._handoff_released.clear()
                self._press_release_requested.clear()
                self._servo_status_code = None
                self._servo_status_received_at = 0.0
                self._wrist_guard_reported = False
                self._servo_command_started_at = 0.0
            self._publish_completion(False)
            with self._condition:
                self._alignment_finished.clear()
                self._alignment_result = (
                    False,
                    'visual alignment did not report a result',
                )
            self._alignment_thread = threading.Thread(
                target=self._run_servo,
                daemon=True,
            )
            self._alignment_thread.start()
            self._await_alignment_result()
            completed, message = self._alignment_result
            response.success = bool(completed)
            response.message = message
        except (ValueError, RuntimeError) as error:
            with self._condition:
                self._running = False
            response.success = False
            response.message = str(error)
        finally:
            with self._condition:
                self._starting = False
                self._condition.notify_all()
        return response

    def _stop_callback(self, request, response):
        del request
        deadline = time.monotonic() + float(self.get_parameter('stop_timeout_seconds').value)
        with self._condition:
            if self._stop_pending:
                response.success = False
                response.message = 'Visual Servo stop confirmation is already pending'
                return response
            self._stop_pending = True
            had_worker = self._running or self._starting
            self._stop_event.set()
            self._handoff_abort = True
            self._condition.notify_all()
        try:
            with self._condition:
                while self._running or self._starting:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        response.success = False
                        response.message = 'Timed out waiting for the visual worker to stop'
                        return response
                    self._condition.wait(min(remaining, 0.1))
            if had_worker:
                stopped = self._cleanup_confirmed and not self._owns_servo
                message = ('Visual worker stopped and cleanup confirmed' if stopped else
                           'Visual worker stopped but cleanup is unconfirmed; retry stop')
            elif self._owns_servo:
                stopped, message = self._cleanup_servo_session()
            else:
                stopped, message = True, 'Visual Servo already stopped; no shared controls changed'
            response.success = stopped
            response.message = message
        finally:
            with self._condition:
                self._stop_pending = False
                self._condition.notify_all()
        return response

    def _cleanup_servo_session(self):
        if not self._owns_servo:
            return True, 'Visual Servo does not own the shared session'
        self._publish_zero_twist()
        paused, pause_message = self._pause_moveit_servo(wait=True)
        closed, gate_message = self._set_hardware_servo_gate(False, wait=True)
        confirmed = paused and closed
        with self._condition:
            self._cleanup_confirmed = confirmed
            self._owns_servo = not confirmed
            self._handoff_pending = False
            self._handoff_ready = False
            self._condition.notify_all()
        return confirmed, ('Visual Servo stopped and shared session closed' if confirmed else
                           f'Visual Servo cleanup unconfirmed: {pause_message}; {gate_message}')

    def _claim_for_press_callback(self, request, response):
        del request
        deadline = time.monotonic() + min(
            1.0, float(self.get_parameter('handover_claim_timeout_seconds').value),
        )
        with self._condition:
            conflict = ButtonVisualServo._sam2_failure(self) or ButtonVisualServo._semantic_conflict_locked(self)
            if conflict:
                response.success = False
                response.message = 'Press handover blocked by ' + conflict
                return response
            if (not self._running or not self._handoff_ready or self._stop_event.is_set()
                    or self._press_release_requested.is_set()):
                response.success = False
                response.message = 'No completed visual alignment is awaiting press handover'
                return response
            self._press_release_requested.set()
            self._condition.notify_all()
            while not self._handoff_released.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0.0 or self._stop_event.is_set():
                    self._handoff_abort = True
                    self._stop_event.set()
                    response.success = False
                    response.message = 'Visual command release was not confirmed'
                    return response
                self._condition.wait(min(remaining, 0.05))
            conflict = ButtonVisualServo._semantic_conflict_locked(self)
            response.success = (
                self._handoff_pending and not self._stop_event.is_set() and not conflict
            )
            response.message = ('Visual command publisher released for press'
                                if response.success else
                                'Press handover blocked by ' + conflict if conflict else
                                'Visual handover was revoked')
        return response

    def _check_pending_handoff(self):
        with self._condition:
            if (not self._handoff_pending or self._stop_pending or
                    (not self._handoff_abort and time.monotonic() < self._handoff_deadline)):
                return
            self._stop_pending = True
        try:
            confirmed, message = self._cleanup_servo_session()
            self._publish_completion(False)
            self._publish_status('PRESS_HANDOVER_UNCONFIRMED: ' + message)
            if not confirmed:
                self.get_logger().error(message)
        finally:
            with self._condition:
                self._stop_pending = False
                self._condition.notify_all()

    def _observation_is_fresh_locked(self):
        return (
            self._observation is not None
            and self._surface_stamp_is_fresh(self._observation_stamp_ns)
            and time.monotonic() - self._observation[2]
            <= float(self.get_parameter('target_max_age_seconds').value)
        )

    def _surface_stamp_is_fresh(self, stamp_ns):
        if stamp_ns <= 0:
            return False
        age = (self.get_clock().now().nanoseconds - stamp_ns) / 1e9
        return (
            -float(
                self.get_parameter('maximum_tf_fallback_age_seconds').value
            ) <= age
            <= float(self.get_parameter('target_max_age_seconds').value)
        )

    def _servo_status_callback(self, message):
        with self._condition:
            self._servo_status_code = int(message.data)
            self._servo_status_received_at = time.monotonic()
            self._condition.notify_all()

    def _joint_state_callback(self, message):
        joint = str(
            self.get_parameter('wrist_limit_guard_joint').value
        ).strip()
        if not joint or joint not in message.name:
            return
        try:
            index = list(message.name).index(joint)
            position = float(message.position[index])
        except (IndexError, TypeError, ValueError):
            return
        if not math.isfinite(position):
            return
        with self._condition:
            self._wrist_guard_position = position
            self._wrist_guard_state_received_at = time.monotonic()

    def _servo_safety_failure(self):
        failure = ButtonVisualServo._sam2_failure(self)
        if failure:
            return failure
        with self._condition:
            code = self._servo_status_code
            received_at = self._servo_status_received_at
            command_started_at = self._servo_command_started_at
        if (
            command_started_at <= 0.0
            or received_at < command_started_at
        ):
            return None
        failures = {
            2: (
                'MoveIt Servo halted at a singularity (status=2); '
                'return home and replan the coarse approach'
            ),
            4: 'MoveIt Servo halted for collision (status=4)',
            5: 'MoveIt Servo halted at a joint bound (status=5)',
        }
        return failures.get(code)

    def _wait_for_observation(self, after_sequence, timeout):
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._stop_event.is_set():
                if (
                    self._observation_is_fresh_locked()
                    and self._observation[3] > after_sequence
                ):
                    return (
                        self._observation[0].copy(),
                        self._observation[1].copy(),
                        self._observation[2],
                        self._observation[3],
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(min(remaining, 0.2))
        return None

    def _run_servo(self):
        self._publish_status('STARTING_MOVEIT_SERVO')
        started = time.monotonic()
        completed = False
        handed_off = False
        final_status = 'FAILED: unknown error'
        try:
            initial = self._wait_for_observation(
                -1,
                float(
                    self.get_parameter('observation_timeout_seconds').value
                ),
            )
            if initial is None:
                final_status = 'FAILED: no initial RGB-D surface pose'
                return
            button, normal, _, _ = initial
            current = self._current_servo_pose()
            if current is None:
                final_status = 'FAILED: fingertip or camera TF unavailable'
                return
            (
                current_position,
                current_orientation,
                camera_orientation,
            ) = current
            captured, capture_message = check_servo_capture(
                button, normal, current_position, camera_orientation,
                maximum_tilt_rad=float(self.get_parameter(
                    'handover_maximum_camera_tilt_rad'
                ).value),
                maximum_roll_rad=float(self.get_parameter(
                    'handover_maximum_camera_roll_rad'
                ).value),
                minimum_standoff_m=float(self.get_parameter(
                    'handover_minimum_standoff_m'
                ).value),
                target_standoff_m=self._standoff_distance(),
                maximum_start_error_m=float(self.get_parameter(
                    'maximum_start_error_m'
                ).value),
                level_reference_axis=self.get_parameter(
                    'level_reference_axis'
                ).value,
            )
            if not captured:
                final_status = f'FAILED: unsafe Servo handover; {capture_message}'
                return
            final_position, final_orientation = self._servo_target(
                button,
                normal,
                current_orientation,
                camera_orientation,
                self._standoff_distance(),
            )
            if not position_in_workspace(
                final_position,
                self._workspace_min,
                self._workspace_max,
            ):
                final_status = 'FAILED: target is outside workspace'
                return
            if self._stop_event.is_set():
                final_status = 'STOPPED'
                return
            self._owns_servo = True
            self._cleanup_confirmed = False
            resumed, message = self._resume_moveit_servo()
            if not resumed:
                final_status = f'FAILED: {message}'
                return
            if self._stop_event.is_set():
                final_status = 'STOPPED'
                return
            gate_ready, message = self._set_hardware_servo_gate(True)
            if not gate_ready:
                final_status = f'FAILED: {message}'
                return
            deadline = started + float(
                self.get_parameter('servo_timeout_seconds').value
            )
            locked, _, message = self._track_visually(
                deadline,
                initial,
            )
            if locked is None:
                final_status = f'FAILED: {message}'
                return
            if self._stop_event.is_set():
                final_status = 'STOPPED'
                return

            held, hold_message = self._decelerate_servo_to_hold()
            if not held:
                final_status = f'FAILED: {hold_message}'
                return

            button, normal = locked
            current = self._current_servo_pose()
            if current is None:
                final_status = (
                    'FAILED: final fingertip or camera TF unavailable'
                )
                return
            axial, lateral, angular = visual_servo_errors(
                current[0],
                current[2],
                button,
                normal,
                self._standoff_distance(),
            )
            angular = self._controlled_angular_error(angular)
            level_roll = self._level_roll_error(current[2], normal)
            if (
                not self._within_tolerance(axial, lateral, angular)
                or not self._roll_within_tolerance(level_roll)
            ):
                final_status = (
                    'FAILED: final TF verification '
                    f'axial={axial * 1000.0:.1f}mm '
                    f'lateral={lateral * 1000.0:.1f}mm '
                    f'angle={math.degrees(angular):.2f}deg '
                    f'roll={self._roll_status(level_roll)}'
                )
                return
            completed = True
            with self._condition:
                conflict = ButtonVisualServo._semantic_conflict_locked(self)
                if conflict:
                    completed = False
                    final_status = 'FAILED: semantic conflict before alignment completion: ' + conflict
                    return
                if self._stop_event.is_set():
                    final_status = 'STOPPED'
                    completed = False
                    return
                self._handoff_ready = True
            final_status = (
                'COMPLETE: continuous Servo alignment; '
                f'standoff={self._standoff_distance() * 100.0:.1f}cm '
                f'lateral={lateral * 1000.0:.1f}mm '
                f'angle={math.degrees(angular):.2f}deg '
                f'roll={self._roll_status(level_roll)}'
            )
            self._publish_completion(True)
            self._publish_status(final_status)
            # Report the alignment before holding the session for the press:
            # the start service waits for exactly this signal.
            self._signal_alignment_finished(True, final_status)
            handed_off = self._hold_for_press_claim()
        except Exception as error:
            final_status = f'FAILED: unexpected error: {error}'
            self.get_logger().error(final_status)
        finally:
            handed_off = handed_off and not self._stop_event.is_set()
            if not handed_off:
                confirmed, cleanup_message = self._cleanup_servo_session()
                if not confirmed:
                    final_status = 'FAILED: ' + cleanup_message
                    completed = False
                self._publish_completion(False)
                self._publish_status(final_status)
            if completed:
                self.get_logger().info(final_status)
            elif final_status not in {'STOPPED', 'FAILED: unknown error'}:
                self.get_logger().error(final_status)
            with self._condition:
                self._handoff_ready = False
                if handed_off:
                    self._handoff_pending = True
                    self._handoff_deadline = time.monotonic() + float(
                        self.get_parameter('handover_claim_timeout_seconds').value)
                    self._handoff_released.set()
                self._running = False
                self._condition.notify_all()
            # Failure, stop, and any early return still have to release the
            # start service with a truthful result.
            self._signal_alignment_finished(completed, final_status)

    def _hold_for_press_claim(self):
        timeout = max(
            0.0,
            float(self.get_parameter('press_claim_timeout_seconds').value),
        )
        if bool(self.get_parameter('simulation_mode').value):
            timeout = max(
                timeout,
                float(
                    self.get_parameter(
                        'simulation_press_claim_timeout_seconds'
                    ).value
                ),
            )
        period = 1.0 / max(
            1.0,
            float(self.get_parameter('servo_control_rate_hz').value),
        )
        deadline = time.monotonic() + timeout
        self._publish_status('SERVO_HANDOFF_READY: waiting for press claim')
        while not self._stop_event.is_set():
            with self._condition:
                conflict = ButtonVisualServo._semantic_conflict_locked(self)
            if conflict:
                self._publish_zero_twist()
                self._publish_completion(False)
                self._publish_status('SEMANTIC_CONFLICT_HOLD: ' + conflict)
                return False
            if self._press_release_requested.is_set():
                self._publish_status('SERVO_HANDOFF_CLAIMED')
                return True
            if time.monotonic() >= deadline:
                self._publish_status(
                    'SERVO_HANDOFF_TIMEOUT: pausing standalone session'
                )
                return False
            self._publish_zero_twist()
            gate_ready, gate_message = self._set_hardware_servo_gate(True)
            if not gate_ready:
                self.get_logger().error(gate_message)
                return False
            if self._stop_event.wait(period):
                break
        return False

    def _track_visually(self, deadline, initial_observation=None):
        period = 1.0 / max(
            1.0,
            float(self.get_parameter('servo_control_rate_hz').value),
        )
        tracking_distance = self._standoff_distance()
        normal_change_limit = float(
            self.get_parameter('maximum_normal_change_rad').value
        )
        required_conflicting_normals = int(
            self.get_parameter('required_conflicting_normal_observations').value
        )
        if (
            not math.isfinite(normal_change_limit)
            or not 0.0 < normal_change_limit < math.pi
            or required_conflicting_normals < 1
        ):
            return None, '', 'invalid final-approach normal consistency limits'
        if initial_observation is None:
            locked = None
            locked_at = 0.0
            locked_sequence = -1
        else:
            locked = (
                initial_observation[0].copy(),
                initial_observation[1].copy(),
            )
            locked_at = float(initial_observation[2])
            locked_sequence = int(initial_observation[3])
        stable_observations = 0
        alignment_stable_observations = 0
        locked_alignment_stable_cycles = 0
        reacquisition_stable_observations = 0
        locked_target_stable_cycles = 0
        counted_sequence = -1
        alignment_counted_sequence = -1
        reacquisition_counted_sequence = -1
        normal_observation_sequence = -1
        conflicting_normal_observations = 0
        normal_change = 0.0
        # The coarse MoveIt trajectory has already put the fingertip at a
        # safe standoff.  Correct the camera orientation at that exact pose
        # before allowing any translation toward the panel.  If RGB-D drops
        # while the wrist rotates, the locked static target may still drive
        # angular correction, but never translation.  Once aligned, hold the
        # pose until fresh RGB-D observations reacquire the same button, then
        # transition into the final visual approach.
        servo_phase = 'ORIENTING'
        aligned_normal = None
        aligned_orientation = None
        reacquisition_started_at = None
        reacquisition_origin = None
        reacquisition_reference_orientation = None
        reacquisition_hold_position = None
        vision_loss_start_position = None
        vision_loss_best_remaining = None
        vision_loss_progress_at = None
        self._last_linear_command = np.zeros(3)
        self._last_angular_command = np.zeros(3)
        self._last_command_at = time.monotonic()
        with self._condition:
            self._servo_command_started_at = 0.0

        while not self._stop_event.is_set():
            now = time.monotonic()
            if now >= deadline:
                return None, '', 'visual tracking timeout'
            safety_failure = self._servo_safety_failure()
            if safety_failure is not None:
                self._publish_zero_twist()
                return None, '', safety_failure
            with self._condition:
                conflict = ButtonVisualServo._semantic_conflict_locked(self)
                conflict_started = getattr(self, '_semantic_conflict_started_at', now)
            if conflict:
                self._publish_zero_twist()
                stable_observations = 0
                alignment_stable_observations = 0
                locked_alignment_stable_cycles = 0
                reacquisition_stable_observations = 0
                locked_target_stable_cycles = 0
                if now - conflict_started > float(
                    self.get_parameter('observation_timeout_seconds').value
                ):
                    return None, '', 'unresolved semantic conflict: ' + conflict
                self._publish_status('SEMANTIC_CONFLICT_HOLD: ' + conflict)
                if self._stop_event.wait(period):
                    break
                continue
            gate_ready, gate_message = self._set_hardware_servo_gate(True)
            if not gate_ready:
                return None, '', gate_message
            with self._condition:
                observation = None
                if self._observation is not None:
                    observation_age = now - self._observation[2]
                    if getattr(self, '_require_sam2', False):
                        observation_age = max(observation_age, (
                            self.get_clock().now().nanoseconds-self._observation_stamp_ns)/1e9)
                    expected_gap = float(
                        self.get_parameter(
                            'expected_observation_gap_seconds'
                        ).value
                    )
                else:
                    observation_age = math.inf
                    expected_gap = 0.0
                if observation_age <= expected_gap:
                    observation = (
                        self._observation[0].copy(),
                        self._observation[1].copy(),
                        self._observation[2],
                        self._observation[3],
                    )
            if getattr(self, '_require_sam2', False) and observation is None:
                self._publish_zero_twist()
                if observation_age > float(self.get_parameter('observation_timeout_seconds').value):
                    return None, '', 'SAM2 pose timeout; blind continuation disabled'
                if self._stop_event.wait(period):
                    break
                continue
            current = self._current_servo_pose()
            if current is None:
                self._publish_zero_twist()
                return (
                    None,
                    '',
                    'fingertip or camera TF unavailable during tracking',
                )
            (
                current_position,
                current_orientation,
                camera_orientation,
            ) = current

            using_locked_observation = False
            orientation_only_locked = False
            loss_speed_scale = 1.0
            if observation is None:
                if locked is not None:
                    button, normal = locked
                    axial, lateral, angular = visual_servo_errors(
                        current_position,
                        camera_orientation,
                        button,
                        normal,
                        tracking_distance,
                    )
                    angular = self._controlled_angular_error(angular)
                    observation_age = now - locked_at
                    loss_age = max(0.0, observation_age - expected_gap)
                    remaining_distance = math.hypot(axial, lateral)
                    observation_timeout = float(
                        self.get_parameter(
                            'observation_timeout_seconds'
                        ).value
                    )
                    if (
                        servo_phase == 'ORIENTING'
                        and loss_age <= observation_timeout
                    ):
                        # At the coarse 14 cm standoff, finish only angular
                        # correction from the locked static surface pose.  No
                        # translation is permitted until RGB-D is reacquired.
                        observation = (
                            button,
                            normal,
                            locked_at,
                            locked_sequence,
                        )
                        using_locked_observation = True
                        orientation_only_locked = True
                    elif servo_phase == 'ORIENTING':
                        self._publish_zero_twist()
                        return (
                            None,
                            '',
                            'RGB-D loss while correcting camera orientation: '
                            f'loss={loss_age:.2f}s '
                            f'angle={math.degrees(angular):.2f}deg',
                        )
                    elif servo_phase == 'REACQUIRING':
                        reacquisition_age = (
                            math.inf
                            if reacquisition_started_at is None
                            else now - reacquisition_started_at
                        )
                        if reacquisition_age > observation_timeout:
                            self._publish_zero_twist()
                            return (
                                None,
                                '',
                                'no fresh RGB-D target after orientation '
                                f'correction: waited={reacquisition_age:.2f}s',
                            )
                        observation = (
                            button,
                            normal,
                            locked_at,
                            locked_sequence,
                        )
                        using_locked_observation = True
                        orientation_only_locked = True
                    else:
                        if vision_loss_start_position is None:
                            vision_loss_start_position = (
                                current_position.copy()
                            )
                        blind_travel = float(np.linalg.norm(
                            current_position - vision_loss_start_position
                        ))
                        # A blind approach is bounded by how far it may
                        # travel and by whether it is still making progress
                        # toward the locked target.  A wall-clock budget alone
                        # aborts a slow but healthy approach: the simulation
                        # reaches only a fraction of the commanded speed, and
                        # its clock runs at a fraction of real time.
                        if (
                            vision_loss_best_remaining is None
                            or remaining_distance
                            < vision_loss_best_remaining - 0.0005
                        ):
                            vision_loss_best_remaining = remaining_distance
                            vision_loss_progress_at = now
                        stalled = (
                            vision_loss_progress_at is None
                            or now - vision_loss_progress_at
                            > float(
                                self.get_parameter(
                                    'vision_loss_stall_seconds'
                                ).value
                            )
                        )
                        if (
                            loss_age <= self._vision_loss_continuation_seconds()
                            and not stalled
                            and remaining_distance <= float(
                                self.get_parameter(
                                    'vision_loss_continuation_max_distance_m'
                                ).value
                            )
                            and blind_travel <= float(
                                self.get_parameter(
                                    'vision_loss_max_travel_m'
                                ).value
                            )
                        ):
                            observation = (
                                button,
                                normal,
                                locked_at,
                                locked_sequence,
                            )
                            using_locked_observation = True
                            loss_speed_scale = float(np.clip(
                                self.get_parameter(
                                    'vision_loss_speed_scale'
                                ).value,
                                0.0,
                                1.0,
                            ))
                        elif loss_age <= observation_timeout:
                            self._publish_zero_twist()
                            # Outside the explicitly bounded blind-motion
                            # region, hold still for projected reacquisition.
                            self._publish_status(
                                'VISION_LOSS_HOLDING '
                                f'loss={loss_age:.2f}s '
                                f'remaining={remaining_distance * 1000.0:.1f}mm '
                                f'angle={math.degrees(angular):.2f}deg'
                            )
                            if self._stop_event.wait(period):
                                break
                            continue
                        else:
                            return (
                                None,
                                '',
                                'RGB-D loss exceeded bounded Servo '
                                'continuation: '
                                f'loss={loss_age:.2f}s '
                                f'remaining={remaining_distance * 1000.0:.1f}mm '
                                f'travel={blind_travel * 1000.0:.1f}mm '
                                f'axial={axial * 1000.0:.1f}mm '
                                f'lateral={lateral * 1000.0:.1f}mm '
                                f'angle={math.degrees(angular):.2f}deg',
                            )
                else:
                    self._publish_zero_twist()
                if observation is None:
                    if self._stop_event.wait(period):
                        break
                    continue

            if servo_phase == 'FINAL_APPROACH' and aligned_normal is not None:
                if (
                    not using_locked_observation
                    and observation[3] > normal_observation_sequence
                ):
                    normal_observation_sequence = observation[3]
                    fresh_normal = observation[1]
                    normal_change = math.acos(float(np.clip(
                        np.dot(fresh_normal, aligned_normal), -1.0, 1.0,
                    )))
                    if normal_change > normal_change_limit:
                        conflicting_normal_observations += 1
                    else:
                        conflicting_normal_observations = 0
                if conflicting_normal_observations:
                    # Keep the control orientation locked, but do not move or
                    # complete while fresh geometry contradicts that lock.
                    # Repeated frames and vision loss cannot clear this hold.
                    self._publish_zero_twist()
                    stable_observations = 0
                    locked_target_stable_cycles = 0
                    detail = (
                        f'change={math.degrees(normal_change):.2f}deg '
                        f'limit={math.degrees(normal_change_limit):.2f}deg '
                        f'observations={conflicting_normal_observations}/'
                        f'{required_conflicting_normals}'
                    )
                    if conflicting_normal_observations >= required_conflicting_normals:
                        return (
                            None, '',
                            'fresh surface normal disagrees with locked '
                            'approach direction: ' + detail,
                        )
                    self._publish_status(
                        'FINAL_APPROACH_NORMAL_CONFLICT ' + detail
                    )
                    if self._stop_event.wait(period):
                        break
                    continue

            button, normal, locked_at, sequence = observation
            if servo_phase == 'FINAL_APPROACH' and aligned_normal is not None:
                normal = aligned_normal.copy()
            if not using_locked_observation:
                locked = (button, normal)
                locked_sequence = sequence
                vision_loss_start_position = None
                vision_loss_best_remaining = None
                vision_loss_progress_at = None
                locked_target_stable_cycles = 0

            target_distance = tracking_distance
            if servo_phase == 'ORIENTING':
                target_position = current_position.copy()
                _, target_orientation = self._servo_target(
                    button,
                    normal,
                    current_orientation,
                    camera_orientation,
                    target_distance,
                )
            elif servo_phase == 'REACQUIRING':
                _, target_orientation = self._servo_target(
                    button,
                    normal,
                    current_orientation,
                    camera_orientation,
                    target_distance,
                )
                if reacquisition_hold_position is not None:
                    target_position = reacquisition_hold_position.copy()
                elif (
                    bool(
                        self.get_parameter(
                            'reacquisition_search_enabled'
                        ).value
                    )
                    and reacquisition_origin is not None
                    and reacquisition_reference_orientation is not None
                    and reacquisition_started_at is not None
                ):
                    search_offset = tangential_spiral_offset(
                        aligned_normal,
                        reacquisition_reference_orientation,
                        now - reacquisition_started_at,
                        float(
                            self.get_parameter(
                                'reacquisition_initial_hold_seconds'
                            ).value
                        ),
                        float(
                            self.get_parameter(
                                'reacquisition_search_radial_speed_mps'
                            ).value
                        ),
                        float(
                            self.get_parameter(
                                'reacquisition_search_angular_speed_radps'
                            ).value
                        ),
                        float(
                            self.get_parameter(
                                'reacquisition_search_radius_m'
                            ).value
                        ),
                    )
                    target_position = reacquisition_origin + search_offset
                else:
                    target_position = current_position.copy()
            else:
                target_distance = tracking_distance
                target_position = button - target_distance * normal
                target_orientation = aligned_orientation.copy()
            self._target_publisher.publish(
                self._make_pose(target_position, target_orientation)
            )
            axial, lateral, angular = visual_servo_errors(
                current_position,
                camera_orientation,
                button,
                normal,
                target_distance,
            )
            angular = self._controlled_angular_error(angular)
            level_roll = self._level_roll_error(
                camera_orientation,
                normal,
            )
            measured_distance = axial + target_distance
            if orientation_only_locked:
                tracking_state = f'{servo_phase}_WITH_LOCKED_TARGET'
            elif using_locked_observation:
                tracking_state = 'VISION_LOSS_CONTINUING'
            else:
                tracking_state = f'VISUAL_{servo_phase}'
            self._publish_status(
                f'{tracking_state} '
                f'distance={measured_distance * 1000.0:.1f}mm '
                f'lateral={lateral * 1000.0:.1f}mm '
                f'angle={math.degrees(angular):.2f}deg '
                f'roll={self._roll_status(level_roll)}'
                + self._wrist_limit_guard_status_text()
            )
            if (
                servo_phase == 'REACQUIRING'
                and reacquisition_origin is not None
                and aligned_normal is not None
            ):
                search_displacement = (
                    current_position - reacquisition_origin
                )
                axial_drift = abs(float(np.dot(
                    search_displacement,
                    aligned_normal,
                )))
                tangent_displacement = (
                    search_displacement
                    - np.dot(search_displacement, aligned_normal)
                    * aligned_normal
                )
                tangent_distance = float(np.linalg.norm(
                    tangent_displacement
                ))
                maximum_axial_drift = float(
                    self.get_parameter(
                        'reacquisition_maximum_axial_drift_m'
                    ).value
                )
                maximum_search_radius = float(
                    self.get_parameter(
                        'reacquisition_search_radius_m'
                    ).value
                )
                if axial_drift > maximum_axial_drift:
                    self._publish_zero_twist()
                    return (
                        None,
                        '',
                        'reacquisition search exceeded axial drift limit: '
                        f'drift={axial_drift * 1000.0:.1f}mm',
                    )
                if tangent_distance > maximum_search_radius + 0.003:
                    self._publish_zero_twist()
                    return (
                        None,
                        '',
                        'reacquisition search exceeded tangent boundary: '
                        f'distance={tangent_distance * 1000.0:.1f}mm',
                    )
            phase_aligned = (
                angular <= float(
                    self.get_parameter(
                        'perpendicular_tolerance_rad'
                    ).value
                )
                and self._roll_within_tolerance(level_roll)
            )
            if servo_phase == 'ORIENTING':
                if using_locked_observation:
                    locked_alignment_stable_cycles = (
                        locked_alignment_stable_cycles + 1
                        if phase_aligned
                        else 0
                    )
                elif sequence != alignment_counted_sequence:
                    alignment_counted_sequence = sequence
                    alignment_stable_observations = (
                        alignment_stable_observations + 1
                        if phase_aligned
                        else 0
                    )
                    locked_alignment_stable_cycles = 0

            required_alignment = int(
                self.get_parameter(
                    'required_alignment_observations'
                ).value
            )
            locked_alignment_required = int(
                self.get_parameter(
                    'required_locked_alignment_cycles'
                ).value
            )
            if (
                servo_phase == 'ORIENTING'
                and (
                    alignment_stable_observations >= required_alignment
                    or locked_alignment_stable_cycles
                    >= locked_alignment_required
                )
            ):
                aligned_normal = normal.copy()
                aligned_orientation = target_orientation.copy()
                locked = (button, aligned_normal)
                servo_phase = 'REACQUIRING'
                reacquisition_started_at = now
                reacquisition_origin = current_position.copy()
                reacquisition_reference_orientation = (
                    camera_orientation.copy()
                )
                reacquisition_hold_position = None
                reacquisition_counted_sequence = sequence
                reacquisition_stable_observations = 0
                self._publish_status(
                    'PHASE_COMPLETE: ORIENTING; '
                    'holding level pose for fresh RGB-D reacquisition'
                )
                self._publish_zero_twist()
                if self._stop_event.wait(period):
                    break
                continue

            if (
                servo_phase == 'REACQUIRING'
                and not using_locked_observation
                and sequence != reacquisition_counted_sequence
            ):
                reacquisition_counted_sequence = sequence
                if phase_aligned:
                    if reacquisition_hold_position is None:
                        reacquisition_hold_position = (
                            current_position.copy()
                        )
                        target_position = (
                            reacquisition_hold_position.copy()
                        )
                    reacquisition_stable_observations += 1
                else:
                    # The new fitted normal disagrees with the locked one.
                    # Correct orientation again without translating, then
                    # require another set of fresh observations.
                    servo_phase = 'ORIENTING'
                    alignment_stable_observations = 0
                    locked_alignment_stable_cycles = 0
                    reacquisition_stable_observations = 0
                    reacquisition_started_at = None
                    reacquisition_origin = None
                    reacquisition_reference_orientation = None
                    reacquisition_hold_position = None
                    self._publish_status(
                        'ORIENTATION_RECHECK: fresh surface normal changed'
                    )
                required_reacquisition = int(
                    self.get_parameter(
                        'required_post_orientation_observations'
                    ).value
                )
                if (
                    servo_phase == 'REACQUIRING'
                    and reacquisition_stable_observations
                    >= required_reacquisition
                ):
                    aligned_normal = normal.copy()
                    aligned_orientation = target_orientation.copy()
                    locked = (button, aligned_normal)
                    servo_phase = 'FINAL_APPROACH'
                    normal_observation_sequence = sequence
                    target_position = current_position.copy()
                    counted_sequence = -1
                    stable_observations = 0
                    self._publish_status(
                        'PHASE_COMPLETE: REACQUIRING; '
                        'continuing to final approach'
                    )

            visual_aligned = (
                servo_phase == 'FINAL_APPROACH'
                and abs(axial) <= float(
                    self.get_parameter('distance_tolerance_m').value
                )
                and lateral <= float(
                    self.get_parameter('lateral_tolerance_m').value
                )
                and angular <= float(
                    self.get_parameter(
                        'perpendicular_tolerance_rad'
                    ).value
                )
                and self._roll_within_tolerance(level_roll)
            )
            if servo_phase == 'FINAL_APPROACH':
                if using_locked_observation:
                    if visual_aligned:
                        locked_target_stable_cycles += 1
                    else:
                        locked_target_stable_cycles = 0
                elif sequence != counted_sequence:
                    counted_sequence = sequence
                    if visual_aligned:
                        stable_observations += 1
                    else:
                        stable_observations = 0
                if stable_observations >= int(
                    self.get_parameter(
                        'required_stable_observations'
                    ).value
                ):
                    return locked, 'visual_target', ''
                if locked_target_stable_cycles >= int(
                    self.get_parameter(
                        'required_locked_target_stable_cycles'
                    ).value
                ):
                    return locked, 'locked_visual_target', ''

            if servo_phase == 'ORIENTING':
                desired_linear = np.zeros(3)
            elif servo_phase == 'REACQUIRING':
                desired_linear = (
                    target_position - current_position
                ) * float(
                    self.get_parameter('linear_proportional_gain').value
                ) * self._linear_speed_multiplier()
                # Search strictly in the fitted panel tangent plane.  Any
                # axial drift is handled by the hard guard above, never by an
                # inward correction command.
                desired_linear -= np.dot(
                    desired_linear,
                    aligned_normal,
                ) * aligned_normal
                desired_linear = self._limit_vector(
                    desired_linear,
                    float(
                        self.get_parameter(
                            'reacquisition_search_speed_mps'
                        ).value
                    ) * self._linear_speed_multiplier(),
                )
            else:
                desired_linear = (
                    target_position - current_position
                ) * float(
                    self.get_parameter('linear_proportional_gain').value
                ) * self._linear_speed_multiplier() * loss_speed_scale
            if using_locked_observation and servo_phase == 'FINAL_APPROACH':
                # Finish the approach at an advertised constant speed instead
                # of a proportional command that decays with the error.  The
                # travel limits, the stall guard, and the reacquisition budget
                # still bound this blind motion.
                blind_error = target_position - current_position
                blind_distance = float(np.linalg.norm(blind_error))
                if blind_distance > 1e-6:
                    desired_linear = (
                        blind_error / blind_distance
                        * min(
                            self._blind_approach_speed() * loss_speed_scale,
                            float(
                                self.get_parameter(
                                    'vision_loss_approach_ramp_gain'
                                ).value
                            ) * blind_distance,
                        )
                    )
                else:
                    desired_linear = np.zeros(3)
            if servo_phase == 'FINAL_APPROACH':
                approach_alignment_error = angular
                if bool(self.get_parameter('level_roll_enabled').value):
                    approach_alignment_error = max(
                        angular,
                        abs(level_roll) if math.isfinite(level_roll) else math.pi,
                    )
                desired_linear, axial_speed_scale = (
                    orientation_prioritized_linear_command(
                        desired_linear,
                        normal,
                        approach_alignment_error,
                        float(
                            self.get_parameter(
                                'axial_approach_full_speed_angle_rad'
                            ).value
                        ),
                        float(
                            self.get_parameter(
                                'axial_approach_stop_angle_rad'
                            ).value
                        ),
                    )
                )
                if axial_speed_scale < 1.0:
                    self._publish_status(
                        'FINAL_APPROACH_ORIENTATION_GUARD '
                        f'angle={math.degrees(angular):.2f}deg '
                        f'roll={self._roll_status(level_roll)} '
                        f'axial_scale={axial_speed_scale:.2f}'
                    )
            desired_angular = quaternion_error_rotation_vector(
                current_orientation,
                target_orientation,
            ) * float(
                self.get_parameter('angular_proportional_gain').value
            )
            desired_angular = self._limit_level_roll_speed(
                desired_angular,
                normal,
            ) * loss_speed_scale
            linear, angular_command = self._smooth_servo_command(
                desired_linear,
                desired_angular,
            )
            # Smoothing retains commands from the previous phase. Enforce
            # motion constraints on the published command and its memory.
            if servo_phase == 'ORIENTING':
                linear = np.zeros(3)
            elif servo_phase == 'REACQUIRING':
                linear -= np.dot(linear, aligned_normal) * aligned_normal
                linear = self._limit_vector(
                    linear,
                    float(self.get_parameter(
                        'reacquisition_search_speed_mps'
                    ).value) * self._linear_speed_multiplier(),
                )
            elif axial_speed_scale < 1.0:
                inward_speed = float(np.dot(linear, normal))
                allowed_inward_speed = max(
                    0.0, float(np.dot(desired_linear, normal)),
                )
                if inward_speed > allowed_inward_speed:
                    linear -= (inward_speed - allowed_inward_speed) * normal
            if using_locked_observation:
                # The smoother contains the previous full-speed command.  Cap
                # its output as well as its input so a camera dropout reduces
                # the very next command to the advertised blind-motion bound.
                linear = self._limit_vector(
                    linear,
                    float(
                        self.get_parameter(
                            'maximum_linear_speed_mps'
                        ).value
                    ) * self._linear_speed_multiplier() * loss_speed_scale,
                )
                angular_command = self._limit_vector(
                    angular_command,
                    float(
                        self.get_parameter(
                            'maximum_angular_speed_radps'
                        ).value
                    ) * loss_speed_scale,
                )
            self._last_linear_command = linear.copy()
            self._last_angular_command = angular_command.copy()
            self._publish_twist(linear, angular_command)
            if self._stop_event.wait(period):
                break
        return None, '', 'visual servo stopped'

    @staticmethod
    def _limit_vector(vector, maximum_norm):
        values = np.asarray(vector, dtype=np.float64)
        if not math.isfinite(maximum_norm) or maximum_norm < 0.0:
            raise ValueError('Vector norm limit must be finite and non-negative')
        norm = float(np.linalg.norm(values))
        if not math.isfinite(norm):
            raise ValueError('Command vector must be finite')
        if maximum_norm == 0.0:
            return np.zeros_like(values)
        if norm > maximum_norm:
            values = values * (float(maximum_norm) / norm)
        return values

    def _linear_speed_multiplier(self):
        if not bool(self.get_parameter('simulation_mode').value):
            return 1.0
        return max(
            1.0,
            float(
                self.get_parameter(
                    'simulation_linear_speed_multiplier'
                ).value
            ),
        )

    def _blind_approach_speed(self):
        """Constant speed used while completing the approach blind.

        Only the simulation stack gets the extra multiplier: Gazebo Servo
        tracking reaches roughly a tenth of the commanded speed, so a
        proportional command would decay to nothing before the last few
        millimetres are covered.
        """
        speed = float(
            self.get_parameter('vision_loss_approach_speed_mps').value
        )
        if bool(self.get_parameter('simulation_mode').value):
            speed *= max(
                1.0,
                float(
                    self.get_parameter(
                        'simulation_vision_loss_approach_speed_multiplier'
                    ).value
                ),
            )
        return max(0.0, speed)

    def _vision_loss_continuation_seconds(self):
        seconds = float(
            self.get_parameter('vision_loss_continuation_seconds').value
        )
        if bool(self.get_parameter('simulation_mode').value):
            seconds = max(
                seconds,
                float(
                    self.get_parameter(
                        'simulation_vision_loss_continuation_seconds'
                    ).value
                ),
            )
        return seconds

    def _smooth_servo_command(self, desired_linear, desired_angular):
        now = time.monotonic()
        elapsed = float(np.clip(now - self._last_command_at, 0.001, 0.10))
        self._last_command_at = now
        linear_speed_multiplier = self._linear_speed_multiplier()
        desired_linear = self._limit_vector(
            desired_linear,
            float(
                self.get_parameter('maximum_linear_speed_mps').value
            ) * linear_speed_multiplier,
        )
        desired_angular = self._limit_vector(
            desired_angular,
            float(
                self.get_parameter('maximum_angular_speed_radps').value
            ),
        )
        alpha = float(np.clip(
            self.get_parameter('command_smoothing_alpha').value,
            0.0,
            1.0,
        ))
        filtered_linear = (
            alpha * desired_linear
            + (1.0 - alpha) * self._last_linear_command
        )
        filtered_angular = (
            alpha * desired_angular
            + (1.0 - alpha) * self._last_angular_command
        )
        linear_delta = self._limit_vector(
            filtered_linear - self._last_linear_command,
            float(
                self.get_parameter(
                    'maximum_linear_acceleration_mps2'
                ).value
            ) * linear_speed_multiplier * elapsed,
        )
        angular_delta = self._limit_vector(
            filtered_angular - self._last_angular_command,
            float(
                self.get_parameter(
                    'maximum_angular_acceleration_radps2'
                ).value
            ) * elapsed,
        )
        self._last_linear_command += linear_delta
        self._last_angular_command += angular_delta
        return (
            self._last_linear_command.copy(),
            self._last_angular_command.copy(),
        )

    def _publish_twist(self, linear, angular):
        with self._condition:
            # Serialize this final gate with semantic callbacks so a conflict
            # arriving after the control-loop check cannot leak a motion command.
            if ButtonVisualServo._sam2_failure(self) or ButtonVisualServo._semantic_conflict_locked(self):
                linear, angular = np.zeros(3), np.zeros(3)
                self._last_linear_command = np.zeros(3)
                self._last_angular_command = np.zeros(3)
            if (
                np.linalg.norm(linear) > 1.0e-6
                or np.linalg.norm(angular) > 1.0e-6
            ):
                if self._servo_command_started_at <= 0.0:
                    self._servo_command_started_at = time.monotonic()
            command = TwistStamped()
            command.header.frame_id = self._base_frame
            command.header.stamp = self.get_clock().now().to_msg()
            command.twist.linear.x = float(linear[0])
            command.twist.linear.y = float(linear[1])
            command.twist.linear.z = float(linear[2])
            command.twist.angular.x = float(angular[0])
            command.twist.angular.y = float(angular[1])
            command.twist.angular.z = float(angular[2])
            self._twist_publisher.publish(command)

    def _publish_zero_twist(self):
        self._last_linear_command = np.zeros(3)
        self._last_angular_command = np.zeros(3)
        self._last_command_at = time.monotonic()
        self._publish_twist(np.zeros(3), np.zeros(3))

    def _decelerate_servo_to_hold(self):
        duration = max(
            0.0,
            float(
                self.get_parameter('servo_deceleration_seconds').value
            ),
        )
        rate = max(
            1.0,
            float(self.get_parameter('servo_control_rate_hz').value),
        )
        period = 1.0 / rate
        deadline = time.monotonic() + duration
        # Tracking has already reached the final clearance. Do not carry an
        # old inward command past that boundary while decaying a local filter.
        self._publish_zero_twist()
        while time.monotonic() < deadline and not self._stop_event.is_set():
            failure = self._servo_safety_failure()
            if failure is not None:
                return False, failure
            if self._current_servo_pose() is None:
                return False, 'Fresh TCP/camera TF lost while settling'
            ready, message = self._set_hardware_servo_gate(True)
            if not ready:
                return False, message
            self._publish_zero_twist()
            self._stop_event.wait(period)
        self._publish_zero_twist()
        return (False, 'Visual Servo stopped while settling') if self._stop_event.is_set() else (True, '')

    def _resume_moveit_servo(self):
        if self._stop_event.is_set():
            return False, 'Visual Servo stop is pending'
        if self._servo_started:
            result = self._call_servo_service(
                self._servo_unpause_client,
                'MoveIt Servo unpause service',
            )
            return (False, 'Visual Servo stopped while unpausing') if self._stop_event.is_set() else result
        started, start_message = self._call_servo_service(
            self._servo_start_client,
            'MoveIt Servo start service',
        )
        if not started:
            return False, start_message
        self._servo_started = True
        if self._stop_event.is_set():
            return False, 'Visual Servo stopped after start acknowledgement'

        # MoveIt Servo can survive a visual-node restart in its paused state.
        # Calling start_servo again reports success but does not necessarily
        # clear that pause, so always request an explicit unpause as well.
        unpaused, unpause_message = self._call_servo_service(
            self._servo_unpause_client,
            'MoveIt Servo unpause service',
        )
        if self._stop_event.is_set():
            return False, 'Visual Servo stopped while unpausing'
        if unpaused:
            return True, unpause_message or start_message
        return False, unpause_message or 'MoveIt Servo unpause was not confirmed'

    def _pause_moveit_servo(self, wait):
        if not self._servo_started:
            return True, ''
        if wait:
            return self._call_servo_service(
                self._servo_pause_client,
                'MoveIt Servo pause service',
            )
        if self._servo_pause_client.service_is_ready():
            self._servo_pause_client.call_async(Trigger.Request())
        return True, ''

    def _call_servo_service(self, client, label):
        timeout = min(
            5.0,
            float(self.get_parameter('action_timeout_seconds').value),
        )
        if not client.wait_for_service(timeout_sec=timeout):
            return False, f'{label} is unavailable'
        future = client.call_async(Trigger.Request())
        result = self._wait_for_future(
            future,
            timeout,
            stop_sensitive=False,
        )
        if result is None:
            return False, f'{label} timed out'
        if not result.success:
            return False, result.message or f'{label} rejected the request'
        return True, result.message

    def _set_hardware_servo_gate(self, enabled, wait=True):
        if enabled and self._stop_event.is_set():
            return False, 'Visual Servo stop is pending'
        if not bool(self.get_parameter('hardware_gate_required').value):
            return True, ''
        now = time.monotonic()
        if enabled and now - self._last_gate_heartbeat < float(
            self.get_parameter('hardware_gate_heartbeat_seconds').value
        ):
            return True, ''
        if not self._hardware_gate_client.wait_for_service(
            timeout_sec=(2.0 if wait else 0.0)
        ):
            return False, 'hardware Servo authorization service unavailable'
        request = SetBool.Request()
        request.data = bool(enabled)
        with self._condition:
            if enabled and self._stop_event.is_set():
                return False, 'Visual Servo stopped while waiting for hardware authorization'
            future = self._hardware_gate_client.call_async(request)
        if not wait:
            self._last_gate_heartbeat = 0.0
            return True, ''
        result = self._wait_for_future(
            future,
            2.0,
            stop_sensitive=False,
        )
        if result is None or not result.success:
            return False, 'hardware Servo authorization rejected'
        if enabled and self._stop_event.is_set():
            return False, 'Visual Servo stopped while awaiting hardware acknowledgement'
        self._last_gate_heartbeat = now if enabled else 0.0
        return True, result.message

    def _within_tolerance(self, axial, lateral, angular):
        return (
            abs(axial) <= float(
                self.get_parameter('distance_tolerance_m').value
            )
            and lateral <= float(
                self.get_parameter('lateral_tolerance_m').value
            )
            and angular <= float(
                self.get_parameter('perpendicular_tolerance_rad').value
            )
        )

    def _controlled_angular_error(self, measured_angular):
        if not bool(
            self.get_parameter('orientation_control_enabled').value
        ):
            return 0.0
        return float(measured_angular)

    def _servo_target(
        self,
        button,
        normal,
        current_tool_orientation,
        current_camera_orientation,
        standoff_distance,
    ):
        direction = np.asarray(normal, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        position = (
            np.asarray(button, dtype=np.float64)
            - float(standoff_distance) * direction
        )
        if bool(self.get_parameter('orientation_control_enabled').value):
            orientation = tool_orientation_for_camera_direction(
                direction,
                current_tool_orientation,
                current_camera_orientation,
                (
                    self._level_reference_axis
                    if bool(self.get_parameter('level_roll_enabled').value)
                    else None
                ),
                (
                    float(
                        self.get_parameter('target_level_roll_rad').value
                    )
                    if bool(self.get_parameter('level_roll_enabled').value)
                    else None
                ),
            )
        else:
            orientation = np.asarray(
                current_tool_orientation,
                dtype=np.float64,
            ).copy()
        return position, orientation

    def _level_roll_error(self, camera_orientation, normal):
        if not bool(self.get_parameter('level_roll_enabled').value):
            return math.nan
        return camera_level_roll_error(
            camera_orientation,
            normal,
            self._level_reference_axis,
        )

    def _roll_within_tolerance(self, roll):
        if not bool(self.get_parameter('level_roll_enabled').value):
            return True
        if not math.isfinite(roll):
            return True
        if self._wrist_limit_guard_freezes_roll():
            # 限位在即：接受当前倾角，绝不再继续朝限位加压。
            return True
        return abs(roll) <= self._level_roll_tolerance()

    def _level_roll_tolerance(self):
        tolerance = float(
            self.get_parameter('level_roll_tolerance_rad').value
        )
        margin = self._wrist_limit_guard_margin()
        if margin is None:
            return tolerance
        if margin <= float(
            self.get_parameter('wrist_limit_guard_relax_margin_rad').value
        ):
            return max(
                tolerance,
                float(
                    self.get_parameter(
                        'level_roll_relaxed_tolerance_rad'
                    ).value
                ),
            )
        return tolerance

    def _wrist_limit_guard_margin(self):
        """Distance from the guarded wrist joint to its nearest limit."""
        if not bool(
            self.get_parameter('wrist_limit_guard_enabled').value
        ):
            return None
        with self._condition:
            position = self._wrist_guard_position
            received_at = self._wrist_guard_state_received_at
        if position is None or not math.isfinite(position):
            return None
        if (
            received_at <= 0.0
            or time.monotonic() - received_at
            > float(
                self.get_parameter(
                    'wrist_limit_guard_state_timeout_seconds'
                ).value
            )
        ):
            # Stale feedback must fall back to the strict acceptance.
            return None
        lower = float(
            self.get_parameter('wrist_limit_guard_lower_rad').value
        )
        upper = float(
            self.get_parameter('wrist_limit_guard_upper_rad').value
        )
        return min(position - lower, upper - position)

    def _wrist_limit_guard_freezes_roll(self):
        margin = self._wrist_limit_guard_margin()
        if margin is None:
            return False
        return margin <= float(
            self.get_parameter('wrist_limit_guard_hold_margin_rad').value
        )

    def _wrist_limit_guard_status_text(self):
        """Suffix for the tracking status once the guard is armed."""
        margin = self._wrist_limit_guard_margin()
        if margin is None or margin > float(
            self.get_parameter('wrist_limit_guard_relax_margin_rad').value
        ):
            return ''
        text = (
            f' wrist_guard={margin:.3f}rad'
            f' roll_limit={math.degrees(self._level_roll_tolerance()):.1f}deg'
        )
        if self._wrist_limit_guard_freezes_roll():
            text += ' wrist_roll_frozen'
        if not self._wrist_guard_reported:
            self._wrist_guard_reported = True
            self._publish_status('WRIST_LIMIT_GUARD_ARMED' + text)
        return text

    @staticmethod
    def _roll_status(roll):
        if not math.isfinite(roll):
            return 'unavailable'
        return f'{math.degrees(roll):.2f}deg'

    def _limit_level_roll_speed(self, angular, normal):
        if not bool(self.get_parameter('level_roll_enabled').value):
            return angular
        direction = np.asarray(normal, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        command = np.asarray(angular, dtype=np.float64).copy()
        roll_speed = float(np.dot(command, direction))
        if self._wrist_limit_guard_freezes_roll():
            return command - roll_speed * direction
        maximum = float(
            self.get_parameter('maximum_level_roll_speed_radps').value
        )
        limited = float(np.clip(roll_speed, -maximum, maximum))
        return command + (limited - roll_speed) * direction

    def _current_servo_pose(self):
        try:
            tool_transform = self._tf_buffer.lookup_transform(
                self._base_frame,
                self._end_effector_link,
                Time(),
                timeout=Duration(
                    seconds=float(
                        self.get_parameter('tf_timeout_seconds').value
                    )
                ),
            )
            camera_transform = self._tf_buffer.lookup_transform(
                self._base_frame,
                self._camera_frame,
                Time(),
                timeout=Duration(
                    seconds=float(
                        self.get_parameter('tf_timeout_seconds').value
                    )
                ),
            )
            now_ns = self.get_clock().now().nanoseconds
            maximum_age = float(self.get_parameter('tf_timeout_seconds').value)
            future_tolerance = self._future_stamp_tolerance()
            for transform in (tool_transform, camera_transform):
                stamp_ns = Time.from_msg(transform.header.stamp).nanoseconds
                age = (now_ns - stamp_ns) / 1e9
                if stamp_ns <= 0 or not (
                    -future_tolerance <= age <= maximum_age
                ):
                    raise ExtrapolationException(
                        f'Current {transform.child_frame_id} TF is stale or '
                        f'future-dated: age={age:.3f}s limit={maximum_age:.3f}s'
                    )
        except (
            LookupException,
            ConnectivityException,
            ExtrapolationException,
        ) as error:
            self.get_logger().warning(
                f'Cannot read fingertip/camera TF: {error}',
                throttle_duration_sec=2.0,
            )
            return None
        translation = tool_transform.transform.translation
        rotation = tool_transform.transform.rotation
        camera_rotation = camera_transform.transform.rotation
        return (
            np.asarray([translation.x, translation.y, translation.z]),
            np.asarray([rotation.x, rotation.y, rotation.z, rotation.w]),
            np.asarray([
                camera_rotation.x,
                camera_rotation.y,
                camera_rotation.z,
                camera_rotation.w,
            ]),
        )

    def _make_pose(self, position, orientation):
        pose = PoseStamped()
        pose.header.frame_id = self._base_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(position[0])
        pose.pose.position.y = float(position[1])
        pose.pose.position.z = float(position[2])
        pose.pose.orientation.x = float(orientation[0])
        pose.pose.orientation.y = float(orientation[1])
        pose.pose.orientation.z = float(orientation[2])
        pose.pose.orientation.w = float(orientation[3])
        return pose

    def _wait_for_future(self, future, timeout, stop_sensitive=True):
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        deadline = time.monotonic() + timeout
        while not event.wait(0.1):
            if (
                (stop_sensitive and self._stop_event.is_set())
                or time.monotonic() >= deadline
            ):
                return None
        try:
            return future.result()
        except Exception:
            return None

    def _publish_status(self, text):
        self._status_publisher.publish(String(data=str(text)))

    def _publish_completion(self, completed):
        self._completion_publisher.publish(Bool(data=bool(completed)))

    def _signal_alignment_finished(self, completed, message):
        """Record the alignment outcome exactly once.

        The first terminal result wins: the start service returns as soon as
        the alignment is decided, while the worker may keep holding the Servo
        session for the press handover afterwards.
        """
        with self._condition:
            if self._alignment_finished.is_set():
                return
            self._alignment_result = (bool(completed), str(message))
            self._alignment_finished.set()
            self._condition.notify_all()

    def _await_alignment_result(self):
        """Block until the visual phase reaches a terminal state.

        Returning while the alignment is still running is what let callers
        start the press against a Servo that had not finished yet.
        """
        timeout = max(
            1.0,
            float(
                self.get_parameter('start_response_timeout_seconds').value
            ),
        )
        deadline = time.monotonic() + timeout
        thread = self._alignment_thread
        alive = (
            thread.is_alive if thread is not None
            else (lambda: False)
        )
        while not self._alignment_finished.is_set():
            with self._condition:
                if self._alignment_finished.is_set():
                    break
                if not alive() and not self._running:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    self._stop_event.set()
                    self._condition.notify_all()
                    break
                self._condition.wait(min(remaining, 0.05))


def main(args=None):
    rclpy.init(args=args)
    node = ButtonVisualServo()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_event.set()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
