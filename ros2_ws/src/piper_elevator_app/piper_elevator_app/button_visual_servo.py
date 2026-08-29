"""Closed-loop position-based visual servo for the selected button."""

import copy
import math
import threading
import time

from geometry_msgs.msg import PoseStamped, TwistStamped
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.msg import OrientationConstraint
from moveit_msgs.msg import PositionConstraint
import numpy as np
from piper_elevator_app.motion_core import camera_level_roll_error
from piper_elevator_app.motion_core import position_in_workspace
from piper_elevator_app.motion_core import quaternion_error_rotation_vector
from piper_elevator_app.motion_core import quaternion_to_matrix
from piper_elevator_app.motion_core import (
    tool_orientation_for_camera_direction,
)
from piper_elevator_app.motion_core import visual_servo_errors
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.time import Time
from shape_msgs.msg import SolidPrimitive
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
    """Iteratively align the fingertip and stop 3 cm before the button."""

    def __init__(self):
        super().__init__('button_visual_servo')
        self._declare_parameters()
        self._callback_group = ReentrantCallbackGroup()
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._running = False
        self._observation = None
        self._observation_sequence = 0
        self._active_plan_goal = None
        self._active_execute_goal = None
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
            Int8,
            self._string_parameter('servo_status_topic'),
            self._servo_status_callback,
            10,
            callback_group=self._callback_group,
        )
        self._move_group_client = ActionClient(
            self,
            MoveGroup,
            self._string_parameter('move_group_action'),
            callback_group=self._callback_group,
        )
        self._execute_client = ActionClient(
            self,
            ExecuteTrajectory,
            self._string_parameter('execute_action'),
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
        self.declare_parameter(
            'target_pose_topic',
            '/button_visual_servo/target_pose',
        )
        self.declare_parameter('status_topic', '/button_visual_servo/status')
        self.declare_parameter(
            'completion_topic',
            '/button_visual_servo/completed',
        )
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter(
            'end_effector_link',
            'pika_fingertip_center_link',
        )
        self.declare_parameter('planning_group', 'arm')
        self.declare_parameter('move_group_action', '/move_action')
        self.declare_parameter('execute_action', '/execute_trajectory')
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
        self.declare_parameter('visual_handoff_distance_m', 0.075)
        self.declare_parameter('visual_distance_tolerance_m', 0.005)
        self.declare_parameter('visual_lateral_tolerance_m', 0.004)
        self.declare_parameter(
            'visual_perpendicular_tolerance_rad',
            math.radians(4.0),
        )
        self.declare_parameter('required_stable_observations', 2)
        self.declare_parameter(
            'vision_loss_handoff_max_distance_m',
            0.090,
        )
        self.declare_parameter('maximum_final_cartesian_distance_m', 0.065)
        self.declare_parameter('maximum_start_error_m', 0.20)
        self.declare_parameter('maximum_target_jump_m', 0.025)
        self.declare_parameter('servo_timeout_seconds', 90.0)
        self.declare_parameter('target_max_age_seconds', 0.75)
        self.declare_parameter('observation_timeout_seconds', 2.0)
        self.declare_parameter('servo_control_rate_hz', 50.0)
        self.declare_parameter('linear_proportional_gain', 1.4)
        self.declare_parameter('angular_proportional_gain', 2.0)
        self.declare_parameter('level_roll_enabled', True)
        self.declare_parameter('level_reference_axis', [0.0, 0.0, 1.0])
        self.declare_parameter(
            'maximum_level_roll_rad',
            math.radians(10.0),
        )
        self.declare_parameter(
            'maximum_level_roll_speed_radps',
            0.15,
        )
        self.declare_parameter('maximum_linear_speed_mps', 0.080)
        self.declare_parameter('maximum_angular_speed_radps', 0.45)
        self.declare_parameter('maximum_linear_acceleration_mps2', 0.30)
        self.declare_parameter(
            'maximum_angular_acceleration_radps2',
            1.40,
        )
        self.declare_parameter('command_smoothing_alpha', 0.50)
        self.declare_parameter('servo_deceleration_seconds', 0.25)

        self.declare_parameter('position_tolerance_m', 0.002)
        self.declare_parameter('orientation_tolerance_rad', 0.025)
        self.declare_parameter('planning_time_seconds', 3.0)
        self.declare_parameter('planning_attempts', 3)
        self.declare_parameter(
            'final_planning_pipeline',
            'pilz_industrial_motion_planner',
        )
        self.declare_parameter('final_planner_id', 'LIN')
        self.declare_parameter('velocity_scaling', 0.12)
        self.declare_parameter('acceleration_scaling', 0.10)
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
            if self._running and self._observation_is_fresh_locked():
                previous_button = self._observation[0]
                jump = float(np.linalg.norm(button_base - previous_button))
                maximum_jump = float(
                    self.get_parameter('maximum_target_jump_m').value
                )
                if jump > maximum_jump:
                    self._publish_status(
                        f'REJECTED: target jumped {jump:.3f} m'
                    )
                    return
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
            plan_goal = self._active_plan_goal
            execute_goal = self._active_execute_goal
            self._condition.notify_all()
        for goal in (plan_goal, execute_goal):
            if goal is not None:
                goal.cancel_goal_async()
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
            locked, handoff_reason, message = self._track_visually(deadline)
            paused, pause_message = self._decelerate_and_pause_servo()
            self._set_hardware_servo_gate(False, wait=False)
            if not paused:
                final_status = f'FAILED: {pause_message}'
                return
            if locked is None:
                final_status = f'FAILED: {message}'
                return
            if self._stop_event.is_set():
                final_status = 'STOPPED'
                return

            button, normal = locked
            current = self._current_servo_pose()
            if current is None:
                final_status = (
                    'FAILED: fingertip or camera TF unavailable at handoff'
                )
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
            final_travel = float(
                np.linalg.norm(final_position - current_position)
            )
            maximum_final_travel = float(
                self.get_parameter(
                    'maximum_final_cartesian_distance_m'
                ).value
            )
            if final_travel > maximum_final_travel:
                final_status = (
                    'FAILED: visual handoff is too far from final target '
                    f'({final_travel:.3f} m)'
                )
                return
            target = self._make_pose(final_position, final_orientation)
            self._target_publisher.publish(target)
            self._publish_status(
                f'CARTESIAN_HANDOFF reason={handoff_reason} '
                f'distance={final_travel * 1000.0:.1f}mm'
            )
            result, message = self._plan_pose(target)
            if result is None:
                final_status = f'FAILED: {message}'
                return
            self._publish_status('CARTESIAN_EXECUTING')
            success, message = self._execute_trajectory(
                result.planned_trajectory
            )
            if not success:
                final_status = f'FAILED: {message}'
                return
            if self._stop_event.is_set():
                final_status = 'STOPPED'
                return

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
            level_roll = self._level_roll_error(current[2], normal)
            if not self._within_tolerance(axial, lateral, angular):
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
                'COMPLETE: smooth visual/LIN handoff; '
                f'standoff={self._standoff_distance() * 100.0:.1f}cm '
                f'lateral={lateral * 1000.0:.1f}mm '
                f'angle={math.degrees(angular):.2f}deg '
                f'roll={self._roll_status(level_roll)}'
            )
        except Exception as error:
            final_status = f'FAILED: unexpected error: {error}'
            self.get_logger().error(final_status)
        finally:
            self._publish_zero_twist()
            self._pause_moveit_servo(wait=False)
            self._set_hardware_servo_gate(False, wait=False)
            with self._condition:
                self._running = False
                self._active_plan_goal = None
                self._active_execute_goal = None
            self._publish_completion(completed)
            self._publish_status(final_status)
            if completed:
                self.get_logger().info(final_status)
            elif final_status not in {'STOPPED', 'FAILED: unknown error'}:
                self.get_logger().error(final_status)

    def _track_visually(self, deadline):
        period = 1.0 / max(
            1.0,
            float(self.get_parameter('servo_control_rate_hz').value),
        )
        handoff_distance = float(
            self.get_parameter('visual_handoff_distance_m').value
        )
        locked = None
        locked_at = 0.0
        stable_observations = 0
        counted_sequence = -1
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
                if self._observation_is_fresh_locked():
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

            if observation is None:
                self._publish_zero_twist()
                if locked is not None:
                    button, normal = locked
                    axial, lateral, angular = visual_servo_errors(
                        current_position,
                        camera_orientation,
                        button,
                        normal,
                        handoff_distance,
                    )
                    measured_distance = axial + handoff_distance
                    if (
                        measured_distance
                        <= float(
                            self.get_parameter(
                                'vision_loss_handoff_max_distance_m'
                            ).value
                        )
                        and measured_distance
                        >= float(
                            self.get_parameter('minimum_standoff_m').value
                        )
                        and lateral <= 0.020
                        and angular <= math.radians(12.0)
                    ):
                        return locked, 'vision_limit', ''
                    if now - locked_at > float(
                        self.get_parameter(
                            'observation_timeout_seconds'
                        ).value
                    ):
                        return None, '', 'RGB-D surface pose timed out'
                if self._stop_event.wait(period):
                    break
                continue

            button, normal, locked_at, sequence = observation
            locked = (button, normal)
            handoff_position, handoff_orientation = self._servo_target(
                button,
                normal,
                current_orientation,
                camera_orientation,
                handoff_distance,
            )
            final_position, final_orientation = self._servo_target(
                button,
                normal,
                current_orientation,
                camera_orientation,
                self._standoff_distance(),
            )
            self._target_publisher.publish(
                self._make_pose(final_position, final_orientation)
            )
            axial, lateral, angular = visual_servo_errors(
                current_position,
                camera_orientation,
                button,
                normal,
                handoff_distance,
            )
            level_roll = self._level_roll_error(
                camera_orientation,
                normal,
            )
            measured_distance = axial + handoff_distance
            self._publish_status(
                'VISUAL_TRACKING '
                f'distance={measured_distance * 1000.0:.1f}mm '
                f'lateral={lateral * 1000.0:.1f}mm '
                f'angle={math.degrees(angular):.2f}deg '
                f'roll={self._roll_status(level_roll)}'
            )
            visual_aligned = (
                abs(axial) <= float(
                    self.get_parameter(
                        'visual_distance_tolerance_m'
                    ).value
                )
                and lateral <= float(
                    self.get_parameter(
                        'visual_lateral_tolerance_m'
                    ).value
                )
                and angular <= float(
                    self.get_parameter(
                        'visual_perpendicular_tolerance_rad'
                    ).value
                )
            )
            if sequence != counted_sequence:
                counted_sequence = sequence
                if visual_aligned:
                    stable_observations += 1
                else:
                    stable_observations = 0
            if stable_observations >= int(
                self.get_parameter('required_stable_observations').value
            ):
                return locked, 'visual_target', ''

            desired_linear = (
                handoff_position - current_position
            ) * float(
                self.get_parameter('linear_proportional_gain').value
            )
            desired_angular = quaternion_error_rotation_vector(
                current_orientation,
                handoff_orientation,
            ) * float(
                self.get_parameter('angular_proportional_gain').value
            )
            desired_angular = self._limit_level_roll_speed(
                desired_angular,
                normal,
            )
            linear, angular_command = self._smooth_servo_command(
                desired_linear,
                desired_angular,
            )
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

    def _smooth_servo_command(self, desired_linear, desired_angular):
        now = time.monotonic()
        elapsed = float(np.clip(now - self._last_command_at, 0.001, 0.10))
        self._last_command_at = now
        desired_linear = self._limit_vector(
            desired_linear,
            float(
                self.get_parameter('maximum_linear_speed_mps').value
            ),
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
            ) * elapsed,
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

    def _decelerate_and_pause_servo(self):
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
        return self._pause_moveit_servo(wait=True)

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
                    self.get_parameter('maximum_level_roll_rad').value
                )
                if bool(self.get_parameter('level_roll_enabled').value)
                else None
            ),
        )
        return position, orientation

    def _level_roll_error(self, camera_orientation, normal):
        if not bool(self.get_parameter('level_roll_enabled').value):
            return math.nan
        return camera_level_roll_error(
            camera_orientation,
            normal,
            self._level_reference_axis,
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

    def _plan_pose(self, target):
        timeout = float(self.get_parameter('action_timeout_seconds').value)
        if not self._move_group_client.wait_for_server(timeout_sec=timeout):
            return None, 'MoveGroup action server is unavailable'

        goal = MoveGroup.Goal()
        goal.request.group_name = self._string_parameter('planning_group')
        goal.request.pipeline_id = self._string_parameter(
            'final_planning_pipeline'
        )
        goal.request.planner_id = self._string_parameter('final_planner_id')
        goal.request.num_planning_attempts = int(
            self.get_parameter('planning_attempts').value
        )
        goal.request.allowed_planning_time = float(
            self.get_parameter('planning_time_seconds').value
        )
        goal.request.max_velocity_scaling_factor = float(
            self.get_parameter('velocity_scaling').value
        )
        goal.request.max_acceleration_scaling_factor = float(
            self.get_parameter('acceleration_scaling').value
        )
        goal.request.start_state.is_diff = True
        goal.request.workspace_parameters.header.frame_id = self._base_frame
        minimum = goal.request.workspace_parameters.min_corner
        maximum = goal.request.workspace_parameters.max_corner
        minimum.x, minimum.y, minimum.z = self._workspace_min.tolist()
        maximum.x, maximum.y, maximum.z = self._workspace_max.tolist()
        goal.request.goal_constraints = [self._pose_constraints(target)]
        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        future = self._move_group_client.send_goal_async(goal)
        goal_handle = self._wait_for_future(
            future,
            timeout,
            stop_sensitive=False,
        )
        if goal_handle is None:
            return None, 'Timed out while sending final Cartesian plan'
        if not goal_handle.accepted:
            return None, 'MoveGroup rejected final Cartesian plan'
        with self._condition:
            self._active_plan_goal = goal_handle
        if self._stop_event.is_set():
            goal_handle.cancel_goal_async()
            return None, 'Visual-servo planning was stopped'
        wrapped = self._wait_for_future(
            goal_handle.get_result_async(),
            timeout + float(
                self.get_parameter('planning_time_seconds').value
            ),
        )
        with self._condition:
            self._active_plan_goal = None
        if wrapped is None:
            return None, 'Final Cartesian planning timed out or was stopped'
        result = wrapped.result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            return None, f'MoveIt planning error {result.error_code.val}'
        if not result.planned_trajectory.joint_trajectory.points:
            return None, 'MoveIt returned an empty Cartesian trajectory'
        return result, ''

    def _pose_constraints(self, target):
        constraints = Constraints()
        constraints.name = 'button_final_cartesian_pose'
        position = PositionConstraint()
        position.header = target.header
        position.link_name = self._end_effector_link
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [float(
            self.get_parameter('position_tolerance_m').value
        )]
        position.constraint_region.primitives = [sphere]
        position.constraint_region.primitive_poses = [target.pose]
        position.weight = 1.0

        orientation = OrientationConstraint()
        orientation.header = target.header
        orientation.link_name = self._end_effector_link
        orientation.orientation = target.pose.orientation
        tolerance = float(
            self.get_parameter('orientation_tolerance_rad').value
        )
        orientation.absolute_x_axis_tolerance = tolerance
        orientation.absolute_y_axis_tolerance = tolerance
        orientation.absolute_z_axis_tolerance = tolerance
        orientation.parameterization = OrientationConstraint.ROTATION_VECTOR
        orientation.weight = 1.0
        constraints.position_constraints = [position]
        constraints.orientation_constraints = [orientation]
        return constraints

    def _execute_trajectory(self, trajectory):
        timeout = float(self.get_parameter('action_timeout_seconds').value)
        if not self._execute_client.wait_for_server(timeout_sec=timeout):
            return False, 'ExecuteTrajectory action server is unavailable'
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = copy.deepcopy(trajectory)
        future = self._execute_client.send_goal_async(goal)
        goal_handle = self._wait_for_future(
            future,
            timeout,
            stop_sensitive=False,
        )
        if goal_handle is None:
            return False, 'Timed out while sending visual-servo trajectory'
        if not goal_handle.accepted:
            return False, 'MoveIt rejected visual-servo trajectory'
        with self._condition:
            self._active_execute_goal = goal_handle
        if self._stop_event.is_set():
            goal_handle.cancel_goal_async()
            return False, 'Visual-servo execution was stopped'
        wrapped = self._wait_for_future(
            goal_handle.get_result_async(),
            timeout,
        )
        with self._condition:
            self._active_execute_goal = None
        if wrapped is None:
            return False, 'Visual-servo execution timed out or was stopped'
        if wrapped.result.error_code.val != MoveItErrorCodes.SUCCESS:
            return (
                False,
                f'MoveIt execution error {wrapped.result.error_code.val}',
            )
        return True, 'Visual-servo step reached'

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
