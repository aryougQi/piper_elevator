"""Closed-loop position-based visual servo for the selected button."""

import math
import threading
import time

from geometry_msgs.msg import PoseStamped, TwistStamped
import numpy as np
from piper_elevator_app.motion_core import camera_level_roll_error
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
        self._selected_button = ''
        self._selection_changed_stamp_ns = 0
        self._observation = None
        self._observation_anchor = None
        self._observation_sequence = 0
        self._press_claim_event = threading.Event()
        self._servo_started = False
        self._servo_status_code = None
        self._servo_status_received_at = 0.0
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
            self._surface_pose_callback,
            10,
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
            Int8,
            self._string_parameter('servo_status_topic'),
            self._servo_status_callback,
            10,
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
        self.create_service(
            Trigger,
            '~/start',
            self._start_callback,
            callback_group=self._callback_group,
        )
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
        self.declare_parameter('surface_pose_topic', '/button_surface_pose')
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
        self.declare_parameter('maximum_start_error_m', 0.20)
        self.declare_parameter('maximum_target_jump_m', 0.015)
        self.declare_parameter('servo_timeout_seconds', 90.0)
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
        self.declare_parameter('tf_timeout_seconds', 0.25)
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

    def _surface_pose_callback(self, message):
        if not message.header.frame_id:
            self._publish_status('REJECTED: surface pose has no frame_id')
            return

        message_stamp_ns = Time.from_msg(message.header.stamp).nanoseconds
        with self._condition:
            if (
                message_stamp_ns > 0
                and message_stamp_ns < self._selection_changed_stamp_ns
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
            or not np.all(np.isfinite(surface_quaternion))
            or np.linalg.norm(surface_quaternion) < 1.0e-6
        ):
            self._publish_status('REJECTED: invalid surface pose')
            return

        stamp = Time.from_msg(message.header.stamp)
        if message.header.stamp.sec == 0 and message.header.stamp.nanosec == 0:
            stamp = Time()
        try:
            transform = self._tf_buffer.lookup_transform(
                self._base_frame,
                message.header.frame_id,
                stamp,
                timeout=Duration(
                    seconds=float(
                        self.get_parameter('tf_timeout_seconds').value
                    )
                ),
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

        received_at = time.monotonic()
        with self._condition:
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
            self._observation_sequence += 1
            self._observation = (
                button_base,
                normal_base,
                received_at,
                self._observation_sequence,
            )
            self._condition.notify_all()
        if not self._running:
            self._publish_status('READY')

    def _button_selection_callback(self, message):
        selected = str(message.data).strip()
        if selected.casefold() in {'clear', 'none'}:
            selected = ''
        with self._condition:
            if selected == self._selected_button:
                return
            self._selected_button = selected
            self._selection_changed_stamp_ns = (
                self.get_clock().now().nanoseconds
            )
            # A button is static in the base frame.  Clear the world-space
            # identity only when the operator changes the requested button;
            # during coarse motion, reject any detector jump to another
            # same-name icon even if the old observation is no longer fresh.
            self._observation = None
            self._observation_anchor = None
            self._condition.notify_all()

    def _press_servo_claim_callback(self, message):
        if message.data:
            self._press_claim_event.set()
        else:
            self._press_claim_event.clear()

    def _start_callback(self, request, response):
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
            if self._running:
                response.success = False
                response.message = 'Visual servo is already running'
                return response
            if not self._observation_is_fresh_locked():
                response.success = False
                response.message = 'No fresh RGB-D surface pose'
                return response
            self._running = True
            self._stop_event.clear()
            self._press_claim_event.clear()
            self._servo_status_code = None
            self._servo_status_received_at = 0.0
            self._servo_command_started_at = 0.0
        self._publish_completion(False)
        threading.Thread(target=self._run_servo, daemon=True).start()
        response.success = True
        response.message = 'Visual servo started; call ~/stop to abort'
        return response

    def _stop_callback(self, request, response):
        del request
        with self._condition:
            was_running = self._running
            self._stop_event.set()
            self._condition.notify_all()
        self._publish_zero_twist()
        self._pause_moveit_servo(wait=False)
        self._set_hardware_servo_gate(False, wait=False)
        self._publish_completion(False)
        response.success = was_running
        response.message = (
            'Stop requested' if was_running else 'Visual servo is not running'
        )
        if was_running:
            self._publish_status('STOPPING')
        return response

    def _observation_is_fresh_locked(self):
        return (
            self._observation is not None
            and time.monotonic() - self._observation[2]
            <= float(self.get_parameter('target_max_age_seconds').value)
        )

    def _servo_status_callback(self, message):
        with self._condition:
            self._servo_status_code = int(message.data)
            self._servo_status_received_at = time.monotonic()
            self._condition.notify_all()

    def _servo_safety_failure(self):
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
            final_position, final_orientation = self._servo_target(
                button,
                normal,
                current_orientation,
                camera_orientation,
                self._standoff_distance(),
            )
            if np.linalg.norm(final_position - current_position) > float(
                self.get_parameter('maximum_start_error_m').value
            ):
                final_status = (
                    'FAILED: run the coarse approach before visual servo'
                )
                return
            if not position_in_workspace(
                final_position,
                self._workspace_min,
                self._workspace_max,
            ):
                final_status = 'FAILED: target is outside workspace'
                return

            resumed, message = self._resume_moveit_servo()
            if not resumed:
                final_status = f'FAILED: {message}'
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

            self._decelerate_servo_to_hold()

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
            final_status = (
                'COMPLETE: continuous Servo alignment; '
                f'standoff={self._standoff_distance() * 100.0:.1f}cm '
                f'lateral={lateral * 1000.0:.1f}mm '
                f'angle={math.degrees(angular):.2f}deg '
                f'roll={self._roll_status(level_roll)}'
            )
            self._publish_completion(True)
            self._publish_status(final_status)
            handed_off = self._hold_for_press_claim()
        except Exception as error:
            final_status = f'FAILED: unexpected error: {error}'
            self.get_logger().error(final_status)
        finally:
            self._publish_zero_twist()
            if not handed_off:
                self._pause_moveit_servo(wait=False)
                self._set_hardware_servo_gate(False, wait=False)
                if completed:
                    # Do not leave a stale latched completion after the live
                    # Servo session is no longer available to the press node.
                    self._publish_completion(False)
            with self._condition:
                self._running = False
            if not completed:
                self._publish_completion(False)
                self._publish_status(final_status)
            if completed:
                self.get_logger().info(final_status)
            elif final_status not in {'STOPPED', 'FAILED: unknown error'}:
                self.get_logger().error(final_status)

    def _hold_for_press_claim(self):
        timeout = max(
            0.0,
            float(self.get_parameter('press_claim_timeout_seconds').value),
        )
        period = 1.0 / max(
            1.0,
            float(self.get_parameter('servo_control_rate_hz').value),
        )
        deadline = time.monotonic() + timeout
        self._publish_status('SERVO_HANDOFF_READY: waiting for press claim')
        while not self._stop_event.is_set():
            if self._press_claim_event.is_set():
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
            gate_ready, gate_message = self._set_hardware_servo_gate(True)
            if not gate_ready:
                return None, '', gate_message
            with self._condition:
                observation = None
                if self._observation is not None:
                    observation_age = now - self._observation[2]
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
                        if (
                            loss_age <= float(
                                self.get_parameter(
                                    'vision_loss_continuation_seconds'
                                ).value
                            )
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

            button, normal, locked_at, sequence = observation
            if servo_phase == 'FINAL_APPROACH' and aligned_normal is not None:
                normal = aligned_normal.copy()
            if not using_locked_observation:
                locked = (button, normal)
                locked_sequence = sequence
                vision_loss_start_position = None
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
            if servo_phase == 'FINAL_APPROACH':
                desired_linear, axial_speed_scale = (
                    orientation_prioritized_linear_command(
                        desired_linear,
                        normal,
                        angular,
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
        norm = float(np.linalg.norm(values))
        if norm > maximum_norm > 0.0:
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
        if (
            np.linalg.norm(linear) > 1.0e-6
            or np.linalg.norm(angular) > 1.0e-6
        ):
            with self._condition:
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
        while time.monotonic() < deadline and not self._stop_event.is_set():
            linear, angular = self._smooth_servo_command(
                np.zeros(3),
                np.zeros(3),
            )
            self._publish_twist(linear, angular)
            self._stop_event.wait(period)
        self._publish_zero_twist()
        return True, ''

    def _resume_moveit_servo(self):
        if self._servo_started:
            return self._call_servo_service(
                self._servo_unpause_client,
                'MoveIt Servo unpause service',
            )
        started, start_message = self._call_servo_service(
            self._servo_start_client,
            'MoveIt Servo start service',
        )
        if not started:
            return False, start_message
        self._servo_started = True

        # MoveIt Servo can survive a visual-node restart in its paused state.
        # Calling start_servo again reports success but does not necessarily
        # clear that pause, so always request an explicit unpause as well.
        unpaused, unpause_message = self._call_servo_service(
            self._servo_unpause_client,
            'MoveIt Servo unpause service',
        )
        if unpaused:
            return True, unpause_message or start_message
        # A newly started Servo may report that it was not paused.  In that
        # case start_servo has already made it ready to consume commands.
        self.get_logger().info(
            f'MoveIt Servo started without unpause: {unpause_message}'
        )
        return True, start_message

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
        return abs(roll) <= float(
            self.get_parameter('level_roll_tolerance_rad').value
        )

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
        except (
            LookupException,
            ConnectivityException,
            ExtrapolationException,
        ) as error:
            self.get_logger().warning(
                f'Cannot read fingertip/camera TF: {error}'
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
