"""Guarded Cartesian button press with joint-effort contact detection."""

import json
import math
import threading
import time
from functools import partial

from geometry_msgs.msg import TwistStamped
import numpy as np
from piper_elevator_app.motion_core import quaternion_to_matrix
from piper_elevator_app.press_core import JointEffortContactDetector
from piper_elevator_app.press_core import PhaseTimer
from piper_elevator_app.press_core import simulated_button_depression
from piper_elevator_app.press_core import StallContactDetector
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.time import Time
from ros_gz_interfaces.msg import Contacts
from sensor_msgs.msg import JointState
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


class PressFailure(RuntimeError):
    """A guarded press stopped before successful completion."""


class ButtonPressExecutor(Node):
    """Press after visual alignment and always retract to the start pose."""

    def __init__(self):
        super().__init__('button_press_executor')
        self._declare_parameters()
        self._callback_group = ReentrantCallbackGroup()
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._running = False
        self._stop_in_progress = False
        self._stop_generation = 0
        self._owns_servo = False
        self._cleanup_confirmed = True
        self._cleanup_message = ''
        self._visual_completed = False
        self._visual_completion_level = False
        self._visual_completion_button = ''
        self._latest_effort = None
        self._effort_sequence = 0
        self._effort_stamp_ns = 0
        self._selected_button = ''
        self._active_simulation_button = ''
        self._simulation_joint_names, self._simulation_contact_topics = (
            self._simulation_button_configuration()
        )
        self._simulation_contacts = {
            button: None for button in self._simulation_joint_names
        }
        self._simulation_true_contacts = {
            button: None for button in self._simulation_joint_names
        }
        self._simulation_contacts_sequences = {
            button: 0 for button in self._simulation_joint_names
        }
        self._latest_button_joints = {
            button: None for button in self._simulation_joint_names
        }
        self._button_joint_sequences = {
            button: 0 for button in self._simulation_joint_names
        }
        self._servo_started = False
        self._last_linear_command = np.zeros(3)
        self._last_command_at = time.monotonic()
        self._last_gate_heartbeat = 0.0
        self._phase_timer = None
        self._motion_state_stamp_ns = 0
        self._servo_status_code = None
        self._servo_status_received_at = 0.0
        self._servo_command_started_at = 0.0

        self._base_frame = self._string_parameter('base_frame')
        self._camera_frame = self._string_parameter('camera_frame')
        self._end_effector_link = self._string_parameter(
            'end_effector_link'
        )
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
        self._completion_publisher = self.create_publisher(
            Bool,
            self._string_parameter('completion_topic'),
            latched_qos,
        )
        self._timing_publisher = self.create_publisher(
            String,
            self._string_parameter('timing_topic'),
            latched_qos,
        )
        self._servo_claim_publisher = self.create_publisher(
            Bool,
            self._string_parameter('servo_claim_topic'),
            latched_qos,
        )
        self._twist_publisher = self.create_publisher(
            TwistStamped,
            self._string_parameter('servo_twist_topic'),
            10,
        )
        self.create_subscription(
            Bool,
            self._string_parameter('visual_completion_topic'),
            self._visual_completion_callback,
            latched_qos,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            JointState,
            self._string_parameter('effort_topic'),
            self._effort_callback,
            20,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            Int8,
            self._string_parameter('servo_status_topic'),
            self._servo_status_callback,
            20,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            String,
            self._string_parameter('button_selected_topic'),
            self._button_selection_callback,
            20,
            callback_group=self._callback_group,
        )
        self._simulation_contact_subscriptions = [
            self.create_subscription(
                Contacts,
                topic,
                partial(self._simulation_contacts_callback, button),
                20,
                callback_group=self._callback_group,
            )
            for button, topic in self._simulation_contact_topics.items()
        ]
        self.create_subscription(
            JointState,
            self._string_parameter('simulation_button_joint_topic'),
            self._simulation_button_joint_callback,
            20,
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
        self._visual_handoff_client = self.create_client(
            Trigger,
            self._string_parameter('visual_servo_handoff_service'),
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
        self._publish_servo_claim(False)
        self._publish_status('WAITING_FOR_VISUAL_SERVO')

    def _declare_parameters(self):
        self.declare_parameter(
            'visual_completion_topic',
            '/button_visual_servo/completed',
        )
        self.declare_parameter(
            'servo_claim_topic',
            '/button_press/servo_claimed',
        )
        self.declare_parameter('continuous_servo_handoff', True)
        self.declare_parameter(
            'visual_servo_handoff_service',
            '/button_visual_servo/claim_for_press',
        )
        self.declare_parameter('handoff_timeout_seconds', 2.0)
        self.declare_parameter('stop_timeout_seconds', 8.0)
        self.declare_parameter('status_topic', '/button_press/status')
        self.declare_parameter('completion_topic', '/button_press/completed')
        self.declare_parameter('timing_topic', '/button_press/timing')
        self.declare_parameter('effort_topic', '/feedback/joint_states')
        self.declare_parameter('button_selected_topic', '/button_selected')
        self.declare_parameter(
            'simulation_button_joint_topic',
            '/elevator_button/joint_states',
        )
        self.declare_parameter(
            'simulation_button_names',
            ['1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm'],
        )
        self.declare_parameter(
            'simulation_button_joint_names',
            [
                'button_1_press_joint',
                'button_2_press_joint',
                'button_3_press_joint',
                'button_4_press_joint',
                'button_up_press_joint',
                'button_down_press_joint',
                'button_open_press_joint',
                'button_close_press_joint',
                'button_alarm_press_joint',
            ],
        )
        self.declare_parameter(
            'simulation_contacts_topics',
            [
                '/elevator_button/button_1/contacts',
                '/elevator_button/button_2/contacts',
                '/elevator_button/button_3/contacts',
                '/elevator_button/button_4/contacts',
                '/elevator_button/button_up/contacts',
                '/elevator_button/button_down/contacts',
                '/elevator_button/button_open/contacts',
                '/elevator_button/button_close/contacts',
                '/elevator_button/button_alarm/contacts',
            ],
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

        self.declare_parameter('control_rate_hz', 50.0)
        self.declare_parameter('approach_speed_mps', 0.010)
        self.declare_parameter('press_speed_mps', 0.004)
        self.declare_parameter('retract_speed_mps', 0.025)
        self.declare_parameter('retract_slowdown_distance_m', 0.003)
        self.declare_parameter('retract_slow_speed_mps', 0.008)
        self.declare_parameter('maximum_acceleration_mps2', 0.10)
        self.declare_parameter('press_extension_m', 0.0025)
        self.declare_parameter('hold_seconds', 0.30)
        self.declare_parameter('maximum_approach_travel_m', 0.038)
        # 真机拿不到可靠力矩阈值时的替代接触判据：
        #   torque          现有六关节力矩阈值（默认，行为不变）
        #   stall           仅用"命令前进但指尖不再前进"的堵转判据
        #   stall_or_torque 两者任一触发即认为接触
        self.declare_parameter('contact_detection_mode', 'torque')
        # 堵转判据：已推进 ≥ minimum_travel_m、命令轴向速度 ≥
        # minimum_command_speed_mps，且连续 required_cycles 个控制周期的位移
        # 增量 < progress_epsilon_m，即判定顶到硬止挡。
        self.declare_parameter('stall_minimum_travel_m', 0.005)
        self.declare_parameter('stall_progress_epsilon_m', 0.00005)
        self.declare_parameter('stall_required_cycles', 5)
        self.declare_parameter('stall_minimum_command_speed_mps', 0.002)
        # 几何行程按压：交接站位到按钮表面的已知行程 + press_extension_m，
        # 不依赖任何力/力矩信号；stall 判据仍作为压过头的兜底。
        self.declare_parameter('geometry_press_enabled', False)
        self.declare_parameter('geometry_press_surface_travel_m', 0.030)
        self.declare_parameter('maximum_lateral_drift_m', 0.003)
        self.declare_parameter('lateral_correction_gain', 2.0)
        self.declare_parameter(
            'maximum_lateral_correction_speed_mps',
            0.006,
        )
        self.declare_parameter(
            'maximum_direction_change_rad',
            math.radians(4.0),
        )
        self.declare_parameter('retract_tolerance_m', 0.0005)
        self.declare_parameter('simulation_retract_tolerance_m', 0.0008)
        self.declare_parameter('motion_timeout_seconds', 15.0)
        self.declare_parameter('tf_timeout_seconds', 0.25)
        self.declare_parameter('feedback_timeout_seconds', 0.25)
        # 仿真时钟按物理步长推进，/clock 与 TF/力反馈时间戳可能相差一个
        # 步长。真机（simulation_mode=false）下该容差恒为 0，判定不变。
        self.declare_parameter(
            'simulation_future_stamp_tolerance_seconds',
            0.02,
        )
        self.declare_parameter('servo_settle_timeout_seconds', 2.0)
        self.declare_parameter('servo_settle_required_samples', 5)
        self.declare_parameter('servo_settle_position_tolerance_m', 0.0002)
        self.declare_parameter(
            'servo_settle_direction_tolerance_rad',
            math.radians(0.25),
        )

        self.declare_parameter(
            'arm_joint_names',
            ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
        )
        self.declare_parameter('torque_thresholds_calibrated', False)
        self.declare_parameter(
            'joint_torque_delta_thresholds_nm',
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        self.declare_parameter(
            'joint_torque_absolute_limits_nm',
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        self.declare_parameter('baseline_sample_count', 25)
        self.declare_parameter('baseline_timeout_seconds', 2.0)
        self.declare_parameter('contact_consecutive_samples', 4)
        self.declare_parameter('contact_minimum_joint_count', 1)
        self.declare_parameter('torque_smoothing_alpha', 0.30)
        self.declare_parameter('emergency_threshold_multiplier', 2.5)

        self.declare_parameter('simulation_mode', False)
        self.declare_parameter('simulation_speed_multiplier', 10.0)
        self.declare_parameter('simulation_pressed_depth_m', 0.0020)
        self.declare_parameter('simulation_release_tolerance_m', 0.0003)
        self.declare_parameter('simulation_release_timeout_seconds', 3.0)
        self.declare_parameter('simulation_feedback_timeout_seconds', 0.50)
        self.declare_parameter('allow_execution', False)

    def _string_parameter(self, name):
        return str(self.get_parameter(name).value)

    def _future_stamp_tolerance(self):
        """Return how far a stamp may lead the node clock.

        Gazebo advances sim time in one-millisecond physics steps, so the
        /clock sample this node holds can be one step older than the stamp
        carried by TF and effort messages.  Only the simulation stack gets
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

    def _simulation_button_configuration(self):
        buttons = [
            str(value).strip()
            for value in self.get_parameter('simulation_button_names').value
        ]
        joints = [
            str(value).strip()
            for value in self.get_parameter(
                'simulation_button_joint_names'
            ).value
        ]
        topics = [
            str(value).strip()
            for value in self.get_parameter(
                'simulation_contacts_topics'
            ).value
        ]
        if (
            not buttons
            or len(buttons) != len(joints)
            or len(buttons) != len(topics)
        ):
            raise ValueError(
                'simulation button names, joint names, and contact topics '
                'must be non-empty lists of equal length'
            )
        if len(set(buttons)) != len(buttons):
            raise ValueError('simulation button names must be unique')
        if any(not value for value in buttons + joints + topics):
            raise ValueError(
                'simulation button mapping contains an empty value'
            )
        return dict(zip(buttons, joints)), dict(zip(buttons, topics))

    def _visual_completion_callback(self, message):
        with self._condition:
            level = bool(message.data)
            if not level:
                self._visual_completed = False
                self._visual_completion_button = ''
            elif not self._visual_completion_level and not self._running:
                self._visual_completed = True
                self._visual_completion_button = self._selected_button
            self._visual_completion_level = level
            ready = self._visual_completed
            self._condition.notify_all()
        if not self._running:
            self._publish_status(
                'READY' if ready else 'WAITING_FOR_VISUAL_SERVO'
            )

    def _effort_callback(self, message):
        with self._condition:
            stamp_ns = Time.from_msg(message.header.stamp).nanoseconds
            age = (self.get_clock().now().nanoseconds - stamp_ns) / 1e9
            maximum_age = float(
                self.get_parameter('feedback_timeout_seconds').value
            )
            if (
                stamp_ns <= self._effort_stamp_ns
                or age > maximum_age
                or age < -self._future_stamp_tolerance()
            ):
                return
            self._effort_stamp_ns = stamp_ns
            self._effort_sequence += 1
            self._latest_effort = (
                list(message.name),
                list(message.effort),
                time.monotonic() - age,
                self._effort_sequence,
            )
            self._condition.notify_all()

    def _button_selection_callback(self, message):
        with self._condition:
            selected = str(message.data).strip()
            if selected != self._selected_button:
                self._selected_button = selected
                self._visual_completed = False
                self._visual_completion_button = ''
                if self._running:
                    self._stop_event.set()
            self._condition.notify_all()

    def _servo_status_callback(self, message):
        with self._condition:
            self._servo_status_code = int(message.data)
            self._servo_status_received_at = time.monotonic()
            self._condition.notify_all()

    def _simulation_contacts_callback(self, button, message):
        with self._condition:
            self._simulation_contacts_sequences[button] += 1
            sample = (
                bool(message.contacts),
                time.monotonic(),
                self._simulation_contacts_sequences[button],
            )
            self._simulation_contacts[button] = sample
            # Gazebo publishes contacts at 200 Hz while the guarded press
            # loop runs at 50 Hz.  Latch a true edge so it cannot be replaced
            # by a subsequent false sample before the press loop observes it.
            if sample[0]:
                self._simulation_true_contacts[button] = sample
            self._condition.notify_all()

    def _simulation_button_joint_callback(self, message):
        names = list(message.name)
        positions = list(message.position)
        now = time.monotonic()
        with self._condition:
            updated = False
            for button, joint_name in self._simulation_joint_names.items():
                try:
                    index = names.index(joint_name)
                    position = float(positions[index])
                except (ValueError, IndexError, TypeError):
                    continue
                if not math.isfinite(position):
                    continue
                self._button_joint_sequences[button] += 1
                self._latest_button_joints[button] = (
                    position,
                    now,
                    self._button_joint_sequences[button],
                )
                updated = True
            if updated:
                self._condition.notify_all()
            self._condition.notify_all()

    def _start_callback(self, request, response):
        del request
        if not bool(self.get_parameter('allow_execution').value):
            response.success = False
            response.message = 'Execution is disabled by allow_execution'
            return response
        with self._condition:
            if self._running or self._stop_in_progress or self._owns_servo:
                response.success = False
                response.message = 'Button press is running or awaiting cleanup'
                return response
            stop_generation = self._stop_generation
            if (
                not self._visual_completed
                or self._visual_completion_button != self._selected_button
            ):
                response.success = False
                response.message = (
                    'Visual servo has not completed the safe alignment'
                )
                return response
        if self._current_motion_state() is None:
            response.success = False
            response.message = 'Fingertip or camera TF is unavailable'
            return response
        simulation = bool(self.get_parameter('simulation_mode').value)
        if simulation:
            with self._condition:
                selected_button = self._selected_button
            if selected_button not in self._simulation_joint_names:
                response.success = False
                response.message = (
                    'No supported button is selected; publish one of: '
                    + ', '.join(self._simulation_joint_names)
                )
                return response
            if self._fresh_button_joint_sample(selected_button) is None:
                response.success = False
                response.message = (
                    'Gazebo joint feedback is unavailable or stale for '
                    f'button `{selected_button}`'
                )
                return response
        if not simulation:
            try:
                needs_torque = self._contact_mode_needs_torque()
            except ValueError as error:
                response.success = False
                response.message = str(error)
                return response
            if needs_torque and not bool(
                self.get_parameter('torque_thresholds_calibrated').value
            ):
                response.success = False
                response.message = (
                    'Real press requires calibrated six-joint torque limits '
                    'for contact_detection_mode='
                    f'{self._contact_mode()}; use mode=stall (optionally with '
                    'geometry_press_enabled) when torque feedback is unusable'
                )
                return response
            try:
                self._make_contact_detector()
                if (
                    self._geometry_press_enabled()
                    and self._geometry_press_travel()
                    > float(
                        self.get_parameter(
                            'maximum_approach_travel_m'
                        ).value
                    )
                ):
                    raise ValueError(
                        'geometry press travel exceeds maximum_approach_travel_m'
                    )
            except ValueError as error:
                response.success = False
                response.message = f'Invalid press configuration: {error}'
                return response
        with self._condition:
            if (
                self._running
                or self._stop_in_progress
                or self._stop_generation != stop_generation
                or self._owns_servo
                or not self._visual_completed
                or self._visual_completion_button != self._selected_button
            ):
                response.success = False
                response.message = 'Visual Servo handoff is no longer available'
                return response
            if simulation and selected_button != self._selected_button:
                response.success = False
                response.message = 'Button selection changed before press start'
                return response
            # One completed alignment authorizes exactly one press, including
            # failed attempts. A new alignment must publish a false/true edge.
            self._visual_completed = False
            self._visual_completion_button = ''
            self._running = True
            self._cleanup_confirmed = False
            self._cleanup_message = ''
            self._servo_command_started_at = 0.0
            self._active_simulation_button = (
                selected_button if simulation else ''
            )
            self._stop_event.clear()
        self._publish_completion(False)
        threading.Thread(target=self._run_press, daemon=True).start()
        response.success = True
        response.message = 'Guarded button press started; call ~/stop to abort'
        return response

    def _stop_callback(self, request, response):
        del request
        deadline = time.monotonic() + float(
            self.get_parameter('stop_timeout_seconds').value
        )
        with self._condition:
            self._stop_generation += 1
            self._visual_completed = False
            self._visual_completion_button = ''
            self._stop_event.set()
            self._condition.notify_all()
            while self._stop_in_progress:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    response.success = False
                    response.message = 'Timed out waiting for press stop'
                    return response
                self._condition.wait(min(remaining, 0.05))
            self._stop_in_progress = True
            worker_was_running = self._running
        try:
            with self._condition:
                while self._running:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        response.success = False
                        response.message = 'Press worker has not stopped'
                        return response
                    self._condition.wait(min(remaining, 0.05))
            # A previous cleanup failure retains ownership, so an explicit
            # stop can retry it without touching another node's session.
            if self._owns_servo and not worker_was_running:
                self._cleanup_confirmed, self._cleanup_message = (
                    self._cleanup_servo_control()
                )
                self._publish_servo_claim(self._owns_servo)
            response.success = self._cleanup_confirmed
            response.message = (
                'Press stopped; control cleanup confirmed'
                if response.success
                else self._cleanup_message or 'Press cleanup is unconfirmed'
            )
            self._publish_completion(False)
            return response
        finally:
            with self._condition:
                self._stop_in_progress = False
                self._condition.notify_all()

    def _cleanup_servo_control(self):
        if not self._owns_servo:
            return True, ''
        self._publish_zero_twist()
        paused, pause_message = self._pause_moveit_servo(wait=True)
        closed, gate_message = self._set_hardware_servo_gate(False, wait=True)
        success = paused and closed
        if success:
            with self._condition:
                self._owns_servo = False
        return success, '; '.join(
            message for okay, message in (
                (paused, pause_message), (closed, gate_message)
            ) if not okay and message
        )

    def _make_contact_detector(self):
        if not self._contact_mode_needs_torque():
            # Stall / geometry modes need no joint-torque thresholds at all.
            return None
        return JointEffortContactDetector(
            self.get_parameter('arm_joint_names').value,
            self.get_parameter('joint_torque_delta_thresholds_nm').value,
            self.get_parameter('joint_torque_absolute_limits_nm').value,
            baseline_sample_count=int(
                self.get_parameter('baseline_sample_count').value
            ),
            consecutive_samples=int(
                self.get_parameter('contact_consecutive_samples').value
            ),
            smoothing_alpha=float(
                self.get_parameter('torque_smoothing_alpha').value
            ),
            minimum_joint_count=int(
                self.get_parameter('contact_minimum_joint_count').value
            ),
            emergency_multiplier=float(
                self.get_parameter('emergency_threshold_multiplier').value
            ),
        )

    def _contact_mode(self):
        mode = str(
            self.get_parameter('contact_detection_mode').value
        ).strip().lower()
        if mode not in ('torque', 'stall', 'stall_or_torque'):
            raise ValueError(
                'contact_detection_mode must be torque, stall, or '
                'stall_or_torque'
            )
        return mode

    def _contact_mode_needs_torque(self):
        return self._contact_mode() in ('torque', 'stall_or_torque')

    def _stall_detection_enabled(self):
        return self._contact_mode() in ('stall', 'stall_or_torque')

    def _make_stall_detector(self):
        return StallContactDetector(
            minimum_travel_m=float(
                self.get_parameter('stall_minimum_travel_m').value
            ),
            progress_epsilon_m=float(
                self.get_parameter('stall_progress_epsilon_m').value
            ),
            required_cycles=int(
                self.get_parameter('stall_required_cycles').value
            ),
            minimum_command_speed_mps=float(
                self.get_parameter(
                    'stall_minimum_command_speed_mps'
                ).value
            ),
        )

    def _geometry_press_enabled(self):
        return bool(
            self.get_parameter('geometry_press_enabled').value
        )

    def _geometry_press_travel(self):
        """Travel from the handover pose to 'surface + press depth'."""
        return (
            float(
                self.get_parameter(
                    'geometry_press_surface_travel_m'
                ).value
            )
            + float(self.get_parameter('press_extension_m').value)
        )

    def _run_press(self):
        completed = False
        final_status = 'FAILED: unknown error'
        start_position = None
        servo_motion_started = False
        simulation_button = ''
        self._phase_timer = PhaseTimer()
        self._start_timed_phase('setup')
        try:
            if self._stop_event.is_set():
                raise PressFailure('press stopped before Servo handoff')
            claimed, message = self._call_trigger(
                self._visual_handoff_client,
                'Visual Servo handoff service',
                timeout=float(
                    self.get_parameter('handoff_timeout_seconds').value
                ),
            )
            if not claimed:
                raise PressFailure(message)
            # The acknowledgment is sent only after the visual worker has
            # stopped publishing. Do not restart/unpause it in continuous mode.
            self._servo_started = True
            self._owns_servo = True
            self._publish_servo_claim(True)
            if not bool(self.get_parameter('continuous_servo_handoff').value):
                resumed, message = self._resume_moveit_servo()
                if not resumed:
                    raise PressFailure(message)
            if self._stop_event.is_set():
                raise PressFailure('press stopped during Servo handoff')
            self._publish_zero_twist()
            gate_ready, message = self._set_hardware_servo_gate(True)
            if not gate_ready:
                raise PressFailure(message)
            self._servo_command_started_at = time.monotonic()
            state = self._current_motion_state()
            if state is None:
                raise PressFailure('fingertip or camera TF unavailable')
            start_position, press_direction = state
            detector = None
            simulation = bool(self.get_parameter('simulation_mode').value)
            simulation_button = self._active_simulation_button
            button_rest_position = None
            contact_sequence = -1
            if simulation:
                button_sample = self._fresh_button_joint_sample(
                    simulation_button
                )
                if button_sample is None:
                    raise PressFailure(
                        'Gazebo joint feedback is stale for button '
                        f'`{simulation_button}`'
                    )
                button_rest_position = button_sample[0]
                with self._condition:
                    contact_sequence = (
                        self._simulation_contacts_sequences[
                            simulation_button
                        ]
                    )
            else:
                detector = self._make_contact_detector()
                if detector is not None:
                    self._collect_torque_baseline(detector)

            # Confirm that the inherited zero-command hold is stable before
            # defining the guarded press origin.
            settled_state = self._settle_servo_origin()
            start_position, press_direction = settled_state
            servo_motion_started = True

            contact_travel = self._approach_until_contact(
                start_position,
                press_direction,
                detector,
                contact_sequence,
                simulation_button,
            )
            self._publish_status(
                'CONTACT_DETECTED '
                f'button={simulation_button or "real"} '
                f'travel={contact_travel * 1000.0:.1f}mm'
            )
            if simulation:
                self._press_simulated_button(
                    start_position,
                    press_direction,
                    button_rest_position,
                    simulation_button,
                )
            elif self._geometry_press_enabled():
                # The approach already travelled to 'surface + press depth';
                # adding the extension again would press twice.
                pass
            else:
                target_travel = contact_travel + float(
                    self.get_parameter('press_extension_m').value
                )
                maximum = float(
                    self.get_parameter('maximum_approach_travel_m').value
                )
                if target_travel > maximum:
                    raise PressFailure('press extension exceeds travel limit')
                self._advance_to_travel(
                    start_position,
                    press_direction,
                    target_travel,
                    detector,
                )
            self._publish_zero_twist()
            self._start_timed_phase('hold', 'HOLDING')
            self._hold_with_torque_monitor(
                detector,
                button_rest_position,
                simulation_button,
            )
            self._retract_to_start(start_position)
            if simulation:
                self._wait_for_simulated_release(
                    button_rest_position,
                    simulation_button,
                )
            completed = True
            final_status = (
                'COMPLETE: '
                f'button={simulation_button or "real"} pressed and retracted'
            )
        except PressFailure as error:
            final_status = f'FAILED: {error}'
            if (
                servo_motion_started
                and start_position is not None
                and not self._stop_event.is_set()
            ):
                self._publish_status(
                    f'RETRACTING_AFTER_FAILURE reason={error}'
                )
                try:
                    self._retract_to_start(start_position)
                except PressFailure as retract_error:
                    final_status += f'; retract failed: {retract_error}'
        except Exception as error:
            final_status = f'FAILED: unexpected error: {error}'
            self.get_logger().error(final_status)
        finally:
            self._phase_timer.stop()
            cleaned, cleanup_message = self._cleanup_servo_control()
            self._publish_servo_claim(self._owns_servo)
            if not cleaned:
                completed = False
                final_status = (
                    f'FAILED: control cleanup failed: {cleanup_message}; '
                    f'prior outcome: {final_status}'
                )
            if completed:
                self._publish_status('SETTLING_AFTER_PRESS')
                try:
                    self._settle_servo_origin(maintain_control=False)
                except PressFailure as settle_error:
                    completed = False
                    final_status = f'FAILED: {settle_error}'
            if self._stop_event.is_set():
                completed = False
                final_status = 'STOPPED' + (
                    f'; control cleanup failed: {cleanup_message}'
                    if not cleaned else ''
                )
            self._publish_timing(
                simulation_button or 'real',
                completed,
            )
            self._publish_completion(completed)
            self._publish_status(final_status)
            if completed:
                self.get_logger().info(final_status)
            elif start_position is not None and not self._stop_event.is_set():
                self.get_logger().error(final_status)
            with self._condition:
                self._running = False
                self._cleanup_confirmed = cleaned
                self._cleanup_message = cleanup_message
                self._active_simulation_button = ''
                self._condition.notify_all()

    def _collect_torque_baseline(self, detector):
        self._start_timed_phase('baseline', 'BASELINING_TORQUE')
        deadline = time.monotonic() + float(
            self.get_parameter('baseline_timeout_seconds').value
        )
        sequence = -1
        while not detector.baseline_ready:
            if self._stop_event.is_set():
                raise PressFailure('stopped during torque baseline')
            self._publish_zero_twist()
            self._refresh_gate_or_raise()
            sample = self._fresh_effort_sample(after_sequence=sequence)
            if sample is not None:
                names, efforts, _, sequence = sample
                try:
                    detector.add_baseline_sample(names, efforts)
                except RuntimeError as error:
                    raise PressFailure(str(error)) from error
            if time.monotonic() >= deadline:
                raise PressFailure(
                    'finite joint torque feedback unavailable for baseline'
                )
            self._stop_event.wait(0.01)

    def _approach_until_contact(
        self,
        start,
        direction,
        detector,
        simulation_contacts_sequence=-1,
        simulation_button='',
    ):
        self._start_timed_phase('approach', 'APPROACHING')
        deadline = time.monotonic() + self._motion_timeout_seconds()
        sequence = -1
        last_status = 0.0
        simulation = bool(self.get_parameter('simulation_mode').value)
        geometry_press = self._geometry_press_enabled()
        travel_limit = float(
            self.get_parameter('maximum_approach_travel_m').value
        )
        if geometry_press:
            travel_limit = min(travel_limit, self._geometry_press_travel())
        stall = (
            self._make_stall_detector()
            if self._stall_detection_enabled() else None
        )
        commanded_speed = self._motion_speed('approach_speed_mps')
        while not self._stop_event.is_set():
            position, travel = self._guard_motion(start, direction)
            if travel >= travel_limit:
                if geometry_press:
                    # Geometry press: 'surface + press depth' is the intended
                    # end of the approach, not a missing contact.
                    self._publish_zero_twist()
                    self._publish_status(
                        'GEOMETRY_PRESS_REACHED '
                        f'travel={travel * 1000.0:.1f}mm '
                        f'limit={travel_limit * 1000.0:.1f}mm'
                    )
                    return travel
                raise PressFailure(
                    'contact not detected before travel limit: '
                    f'travel={travel * 1000.0:.1f}mm '
                    f'limit={travel_limit * 1000.0:.1f}mm'
                )
            if stall is not None and stall.update(travel, commanded_speed):
                self._publish_zero_twist()
                self._publish_status(
                    'CONTACT_DETECTED_BY_STALL '
                    f'travel={travel * 1000.0:.1f}mm'
                )
                return travel
            if simulation:
                contacts = self._fresh_simulation_contacts(
                    simulation_button,
                    after_sequence=simulation_contacts_sequence,
                )
                if contacts is not None and contacts[0]:
                    self._publish_zero_twist()
                    return travel
                if contacts is not None:
                    simulation_contacts_sequence = contacts[2]
                if time.monotonic() - last_status >= 0.20:
                    self._publish_status(
                        'APPROACHING '
                        f'button={simulation_button} '
                        f'travel={travel * 1000.0:.1f}mm '
                        'waiting_for_gazebo_contact'
                    )
                    last_status = time.monotonic()
            if detector is not None:
                sample = self._fresh_effort_sample(after_sequence=sequence)
                if sample is not None:
                    names, efforts, _, sequence = sample
                    try:
                        result = detector.update(names, efforts)
                    except ValueError as error:
                        raise PressFailure(str(error)) from error
                    if result.emergency:
                        raise PressFailure(result.reason)
                    if result.contact:
                        self._publish_zero_twist()
                        return travel
                    if time.monotonic() - last_status >= 0.20:
                        self._publish_status(
                            'APPROACHING '
                            f'travel={travel * 1000.0:.1f}mm '
                            f'torque_ratio={result.normalized_peak:.2f}'
                        )
                        last_status = time.monotonic()
                elif self._effort_feedback_stale():
                    raise PressFailure('joint torque feedback timed out')
            self._guard_deadline(
                deadline,
                'approach timed out before contact: '
                f'travel={travel * 1000.0:.1f}mm',
            )
            self._refresh_gate_or_raise()
            self._publish_smoothed_linear(
                self._line_tracking_command(
                    start,
                    position,
                    direction,
                    'approach_speed_mps',
                )
            )
            self._wait_period()
        raise PressFailure('press stopped')

    def _press_simulated_button(
        self,
        start,
        direction,
        rest_position,
        simulation_button,
    ):
        self._start_timed_phase(
            'press',
            f'PRESSING button={simulation_button}',
        )
        deadline = time.monotonic() + self._motion_timeout_seconds()
        required_depth = float(
            self.get_parameter('simulation_pressed_depth_m').value
        )
        last_status = 0.0
        while not self._stop_event.is_set():
            position, travel = self._guard_motion(start, direction)
            if travel >= float(
                self.get_parameter('maximum_approach_travel_m').value
            ):
                raise PressFailure(
                    'button did not depress before travel limit'
                )
            sample = self._fresh_button_joint_sample(simulation_button)
            if sample is None:
                raise PressFailure(
                    'Gazebo joint feedback timed out for button '
                    f'`{simulation_button}`'
                )
            depth = simulated_button_depression(rest_position, sample[0])
            if depth >= required_depth:
                self._publish_zero_twist()
                self._publish_status(
                    'BUTTON_DEPRESSED '
                    f'button={simulation_button} '
                    f'depth={depth * 1000.0:.1f}mm'
                )
                return depth
            if time.monotonic() - last_status >= 0.20:
                self._publish_status(
                    f'PRESSING depth={depth * 1000.0:.1f}mm'
                )
                last_status = time.monotonic()
            self._guard_deadline(
                deadline,
                'button depression timed out: '
                f'depth={depth * 1000.0:.1f}mm '
                f'travel={travel * 1000.0:.1f}mm',
            )
            self._refresh_gate_or_raise()
            self._publish_smoothed_linear(
                self._line_tracking_command(
                    start,
                    position,
                    direction,
                    'press_speed_mps',
                )
            )
            self._wait_period()
        raise PressFailure('press stopped')

    def _advance_to_travel(self, start, direction, target, detector):
        self._start_timed_phase('press', 'PRESSING')
        deadline = time.monotonic() + self._motion_timeout_seconds()
        sequence = -1
        stall = (
            self._make_stall_detector()
            if self._stall_detection_enabled() else None
        )
        commanded_speed = self._motion_speed('press_speed_mps')
        while not self._stop_event.is_set():
            position, travel = self._guard_motion(start, direction)
            if travel >= target:
                self._publish_zero_twist()
                return
            if stall is not None and stall.update(travel, commanded_speed):
                # The button bottomed out before the nominal press travel.
                self._publish_zero_twist()
                self._publish_status(
                    'PRESS_BOTTOMED_OUT '
                    f'travel={travel * 1000.0:.1f}mm '
                    f'target={target * 1000.0:.1f}mm'
                )
                return
            if detector is not None:
                sample = self._fresh_effort_sample(after_sequence=sequence)
                if sample is not None:
                    names, efforts, _, sequence = sample
                    result = detector.update(names, efforts)
                    if result.emergency:
                        raise PressFailure(result.reason)
                elif self._effort_feedback_stale():
                    raise PressFailure('joint torque feedback timed out')
            self._guard_deadline(
                deadline,
                'press extension timed out: '
                f'travel={travel * 1000.0:.1f}mm '
                f'target={target * 1000.0:.1f}mm',
            )
            self._refresh_gate_or_raise()
            self._publish_smoothed_linear(
                self._line_tracking_command(
                    start,
                    position,
                    direction,
                    'press_speed_mps',
                )
            )
            self._wait_period()
        raise PressFailure('press stopped')

    def _retract_to_start(self, start):
        self._start_timed_phase('retract', 'RETRACTING')
        # Clear any residual press-direction command before reversing.  The
        # following loop still applies the normal acceleration bound.
        self._publish_zero_twist()
        deadline = time.monotonic() + self._retract_timeout_seconds()
        tolerance = self._retract_tolerance_m()
        fast_speed = self._motion_speed('retract_speed_mps')
        slow_speed = self._motion_speed('retract_slow_speed_mps')
        slowdown_distance = float(
            self.get_parameter('retract_slowdown_distance_m').value
        )
        last_status = 0.0
        while not self._stop_event.is_set():
            state = self._current_motion_state()
            if state is None:
                raise PressFailure('fingertip TF lost during retract')
            error = start - state[0]
            distance = float(np.linalg.norm(error))
            if distance <= tolerance:
                self._publish_zero_twist()
                return
            if time.monotonic() - last_status >= 0.20:
                self._publish_status(
                    f'RETRACTING remaining={distance * 1000.0:.1f}mm'
                )
                last_status = time.monotonic()
            self._guard_deadline(
                deadline,
                'retract timed out: '
                f'remaining={distance * 1000.0:.1f}mm '
                f'tolerance={tolerance * 1000.0:.1f}mm',
            )
            self._refresh_gate_or_raise()
            # Gazebo's embedded position interface applies a fixed 0.1 gain.
            # Apply the same simulation-only velocity compensation used by
            # approach and press; real hardware retains the original gain.
            desired = (
                error
                * 2.0
                * self._simulation_motion_multiplier()
            )
            speed = float(np.linalg.norm(desired))
            maximum_speed = (
                slow_speed if distance <= slowdown_distance else fast_speed
            )
            if speed > maximum_speed:
                desired *= maximum_speed / speed
            self._publish_smoothed_linear(desired)
            self._wait_period()
        raise PressFailure('retract stopped')

    def _wait_for_simulated_release(
        self,
        rest_position,
        simulation_button,
    ):
        self._start_timed_phase('release_wait', 'WAITING_FOR_RELEASE')
        deadline = time.monotonic() + float(
            self.get_parameter(
                'simulation_release_timeout_seconds'
            ).value
        )
        tolerance = float(
            self.get_parameter('simulation_release_tolerance_m').value
        )
        while not self._stop_event.is_set():
            sample = self._fresh_button_joint_sample(simulation_button)
            if sample is None:
                raise PressFailure(
                    'Gazebo button joint feedback timed out during release'
                )
            depth = simulated_button_depression(rest_position, sample[0])
            if depth <= tolerance:
                self._publish_status(
                    'BUTTON_RELEASED '
                    f'button={simulation_button} '
                    f'depth={depth * 1000.0:.1f}mm'
                )
                return
            self._guard_deadline(
                deadline,
                'Gazebo button release timed out: '
                f'depth={depth * 1000.0:.1f}mm '
                f'tolerance={tolerance * 1000.0:.1f}mm',
            )
            self._publish_zero_twist()
            self._wait_period()
        raise PressFailure('press stopped during button release')

    def _hold_with_torque_monitor(
        self,
        detector,
        simulation_button_rest=None,
        simulation_button='',
    ):
        deadline = time.monotonic() + float(
            self.get_parameter('hold_seconds').value
        )
        sequence = -1
        while not self._stop_event.is_set() and time.monotonic() < deadline:
            if detector is not None:
                sample = self._fresh_effort_sample(after_sequence=sequence)
                if sample is not None:
                    names, efforts, _, sequence = sample
                    result = detector.update(names, efforts)
                    if result.emergency:
                        raise PressFailure(result.reason)
                elif self._effort_feedback_stale():
                    raise PressFailure('joint torque feedback timed out')
            elif simulation_button_rest is not None:
                sample = self._fresh_button_joint_sample(simulation_button)
                if sample is None:
                    raise PressFailure(
                        'Gazebo button joint feedback timed out during hold'
                    )
                depth = simulated_button_depression(
                    simulation_button_rest,
                    sample[0],
                )
                required = float(
                    self.get_parameter('simulation_pressed_depth_m').value
                )
                if depth < required * 0.75:
                    raise PressFailure(
                        'Gazebo button released before hold completed'
                    )
            self._refresh_gate_or_raise()
            self._publish_zero_twist()
            self._wait_period()
        if self._stop_event.is_set():
            raise PressFailure('press stopped')

    def _guard_motion(self, start, expected_direction):
        state = self._current_motion_state()
        if state is None:
            raise PressFailure('fingertip or camera TF lost during press')
        position, current_direction = state
        displacement = position - start
        travel = float(np.dot(displacement, expected_direction))
        lateral = float(np.linalg.norm(
            displacement - travel * expected_direction
        ))
        lateral_limit = float(
            self.get_parameter('maximum_lateral_drift_m').value
        )
        if lateral > lateral_limit:
            raise PressFailure(
                'lateral drift exceeded safety limit: '
                f'lateral={lateral * 1000.0:.1f}mm '
                f'limit={lateral_limit * 1000.0:.1f}mm '
                f'travel={travel * 1000.0:.1f}mm'
            )
        cosine = float(np.clip(
            np.dot(current_direction, expected_direction),
            -1.0,
            1.0,
        ))
        direction_error = math.acos(cosine)
        direction_limit = float(
            self.get_parameter('maximum_direction_change_rad').value
        )
        if direction_error > direction_limit:
            raise PressFailure(
                'camera lost perpendicular press direction: '
                f'error={math.degrees(direction_error):.2f}deg '
                f'limit={math.degrees(direction_limit):.2f}deg'
            )
        return position, travel

    def _line_tracking_command(
        self,
        start,
        position,
        direction,
        speed_parameter,
    ):
        """Advance along the press normal while correcting lateral drift."""
        displacement = position - start
        travel = float(np.dot(displacement, direction))
        desired_line_position = start + travel * direction
        lateral_error = desired_line_position - position
        correction = lateral_error * float(
            self.get_parameter('lateral_correction_gain').value
        ) * self._simulation_motion_multiplier()
        maximum_correction = float(
            self.get_parameter(
                'maximum_lateral_correction_speed_mps'
            ).value
        ) * self._simulation_motion_multiplier()
        correction_norm = float(np.linalg.norm(correction))
        if correction_norm > maximum_correction > 0.0:
            correction *= maximum_correction / correction_norm
        return direction * self._motion_speed(speed_parameter) + correction

    def _current_motion_state(self):
        try:
            tip = self._tf_buffer.lookup_transform(
                self._base_frame,
                self._end_effector_link,
                Time(),
                timeout=Duration(
                    seconds=float(
                        self.get_parameter('tf_timeout_seconds').value
                    )
                ),
            )
            camera = self._tf_buffer.lookup_transform(
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
            maximum_age = float(
                self.get_parameter('feedback_timeout_seconds').value
            )
            future_tolerance = self._future_stamp_tolerance()
            for transform in (tip, camera):
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
            self._motion_state_stamp_ns = 0
            self.get_logger().warning(
                f'Cannot read fingertip/camera TF: {error}',
                throttle_duration_sec=2.0,
            )
            return None
        translation = tip.transform.translation
        self._motion_state_stamp_ns = (
            int(tip.header.stamp.sec) * 1_000_000_000
            + int(tip.header.stamp.nanosec)
        )
        rotation = camera.transform.rotation
        direction = quaternion_to_matrix([
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        ])[:, 2]
        direction /= np.linalg.norm(direction)
        return (
            np.asarray([translation.x, translation.y, translation.z]),
            direction,
        )

    def _settle_servo_origin(self, maintain_control=True):
        """Wait for fresh, consecutively stable TF after a Servo transition."""
        deadline = time.monotonic() + float(
            self.get_parameter('servo_settle_timeout_seconds').value
        )
        required = max(
            2,
            int(
                self.get_parameter(
                    'servo_settle_required_samples'
                ).value
            ),
        )
        position_tolerance = float(
            self.get_parameter(
                'servo_settle_position_tolerance_m'
            ).value
        )
        direction_tolerance = float(
            self.get_parameter(
                'servo_settle_direction_tolerance_rad'
            ).value
        )
        last_stamp = None
        previous_state = None
        stable_updates = 0
        latest_state = None
        while not self._stop_event.is_set() and time.monotonic() < deadline:
            if maintain_control:
                self._publish_zero_twist()
                self._refresh_gate_or_raise()
            state = self._current_motion_state()
            stamp = self._motion_state_stamp_ns
            if state is not None and stamp > 0 and stamp != last_stamp:
                latest_state = state
                last_stamp = stamp
                if previous_state is None:
                    stable_updates = 1
                else:
                    position_delta = float(np.linalg.norm(
                        state[0] - previous_state[0]
                    ))
                    direction_delta = math.acos(float(np.clip(
                        np.dot(state[1], previous_state[1]),
                        -1.0,
                        1.0,
                    )))
                    if (
                        position_delta <= position_tolerance
                        and direction_delta <= direction_tolerance
                    ):
                        stable_updates += 1
                    else:
                        stable_updates = 1
                previous_state = state
                if stable_updates >= required:
                    return latest_state
            self._wait_period()
        if self._stop_event.is_set():
            raise PressFailure('press stopped during Servo settling')
        raise PressFailure('fingertip did not settle before guarded press')

    def _fresh_simulation_contacts(self, button, after_sequence=-1):
        with self._condition:
            timeout = float(
                self.get_parameter(
                    'simulation_feedback_timeout_seconds'
                ).value
            )
            true_sample = self._simulation_true_contacts.get(button)
            if (
                true_sample is not None
                and true_sample[2] > after_sequence
                and time.monotonic() - true_sample[1] <= timeout
            ):
                return true_sample
            sample = self._simulation_contacts.get(button)
            if sample is None or sample[2] <= after_sequence:
                return None
            if time.monotonic() - sample[1] > timeout:
                return None
            return sample

    def _fresh_button_joint_sample(self, button=None):
        selected = button or self._active_simulation_button
        with self._condition:
            sample = self._latest_button_joints.get(selected)
            if sample is None:
                return None
            timeout = float(
                self.get_parameter(
                    'simulation_feedback_timeout_seconds'
                ).value
            )
            if time.monotonic() - sample[1] > timeout:
                return None
            return sample

    def _fresh_effort_sample(self, after_sequence=-1):
        with self._condition:
            sample = self._latest_effort
            if sample is None or sample[3] <= after_sequence:
                return None
            if self._effort_feedback_stale():
                return None
            return (list(sample[0]), list(sample[1]), sample[2], sample[3])

    def _effort_feedback_stale(self):
        with self._condition:
            maximum_age = float(
                self.get_parameter('feedback_timeout_seconds').value
            )
            capture_age = (
                self.get_clock().now().nanoseconds - self._effort_stamp_ns
            ) / 1e9
            return (
                self._latest_effort is None
                or self._effort_stamp_ns <= 0
                or capture_age > maximum_age
                or capture_age < -self._future_stamp_tolerance()
                or time.monotonic() - self._latest_effort[2] > maximum_age
            )

    def _publish_smoothed_linear(self, desired):
        desired = np.asarray(desired, dtype=np.float64)
        now = time.monotonic()
        elapsed = float(np.clip(now - self._last_command_at, 0.001, 0.10))
        self._last_command_at = now
        maximum_delta = float(
            self.get_parameter('maximum_acceleration_mps2').value
        ) * elapsed
        delta = desired - self._last_linear_command
        norm = float(np.linalg.norm(delta))
        if norm > maximum_delta > 0.0:
            delta *= maximum_delta / norm
        self._last_linear_command += delta
        self._publish_twist(self._last_linear_command)

    def _motion_speed(self, parameter_name):
        speed = float(self.get_parameter(parameter_name).value)
        return speed * self._simulation_motion_multiplier()

    def _simulation_motion_multiplier(self):
        if not bool(self.get_parameter('simulation_mode').value):
            return 1.0
        return float(
            self.get_parameter('simulation_speed_multiplier').value
        )

    def _motion_timeout_seconds(self):
        return float(self.get_parameter('motion_timeout_seconds').value)

    def _retract_timeout_seconds(self):
        return float(self.get_parameter('motion_timeout_seconds').value)

    def _retract_tolerance_m(self):
        if bool(self.get_parameter('simulation_mode').value):
            return float(
                self.get_parameter(
                    'simulation_retract_tolerance_m'
                ).value
            )
        return float(self.get_parameter('retract_tolerance_m').value)

    def _publish_twist(self, linear):
        if not self._owns_servo:
            return
        if self._stop_event.is_set():
            linear = np.zeros(3)
        command = TwistStamped()
        command.header.frame_id = self._base_frame
        command.header.stamp = self.get_clock().now().to_msg()
        command.twist.linear.x = float(linear[0])
        command.twist.linear.y = float(linear[1])
        command.twist.linear.z = float(linear[2])
        self._twist_publisher.publish(command)

    def _publish_zero_twist(self):
        self._last_linear_command = np.zeros(3)
        self._last_command_at = time.monotonic()
        self._publish_twist(np.zeros(3))

    def _resume_moveit_servo(self):
        if self._stop_event.is_set():
            return False, 'Press stopped before MoveIt Servo resume'
        if self._servo_started:
            resumed, message = self._call_trigger(
                self._servo_unpause_client,
                'MoveIt Servo unpause service',
            )
            with self._condition:
                if self._stop_event.is_set():
                    return False, 'Press stopped while resuming MoveIt Servo'
                return resumed, message
        started, message = self._call_trigger(
            self._servo_start_client,
            'MoveIt Servo start service',
        )
        if not started:
            return False, message
        self._servo_started = True
        if self._stop_event.is_set():
            return False, 'Press stopped while starting MoveIt Servo'
        unpaused, unpause_message = self._call_trigger(
            self._servo_unpause_client,
            'MoveIt Servo unpause service',
        )
        if not unpaused:
            return False, unpause_message
        with self._condition:
            if self._stop_event.is_set():
                return False, 'Press stopped while resuming MoveIt Servo'
            return True, message

    def _pause_moveit_servo(self, wait):
        if not self._servo_started:
            return True, ''
        if not wait:
            if self._servo_pause_client.service_is_ready():
                self._servo_pause_client.call_async(Trigger.Request())
            return True, ''
        return self._call_trigger(
            self._servo_pause_client,
            'MoveIt Servo pause service',
        )

    def _call_trigger(self, client, label, timeout=3.0):
        deadline = time.monotonic() + timeout
        if not client.wait_for_service(timeout_sec=timeout):
            return False, f'{label} is unavailable'
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return False, f'{label} timed out before request'
        result = self._wait_for_future(
            client.call_async(Trigger.Request()),
            remaining,
        )
        if result is None:
            return False, f'{label} timed out'
        if not result.success:
            return False, result.message or f'{label} rejected request'
        return True, result.message

    def _set_hardware_servo_gate(self, enabled, wait=True):
        if enabled and (self._stop_event.is_set() or not self._owns_servo):
            return False, 'Press no longer owns active Servo control'
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
        remaining = now + 2.0 - time.monotonic()
        if wait and remaining <= 0.0:
            return False, 'hardware Servo authorization timed out'
        with self._condition:
            if enabled and (self._stop_event.is_set() or not self._owns_servo):
                return False, 'Press stopped before hardware Servo authorization'
            request = SetBool.Request()
            request.data = bool(enabled)
            future = self._hardware_gate_client.call_async(request)
        if not wait:
            self._last_gate_heartbeat = 0.0
            return True, ''
        result = self._wait_for_future(future, remaining)
        if result is None or not result.success:
            return False, 'hardware Servo authorization rejected'
        with self._condition:
            if enabled and (self._stop_event.is_set() or not self._owns_servo):
                return False, 'Press stopped while awaiting hardware Servo authorization'
            self._last_gate_heartbeat = now if enabled else 0.0
            return True, result.message

    def _refresh_gate_or_raise(self):
        success, message = self._set_hardware_servo_gate(True)
        if not success:
            raise PressFailure(message)

    @staticmethod
    def _wait_for_future(future, timeout):
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        if not event.wait(timeout):
            return None
        try:
            return future.result()
        except Exception:
            return None

    def _guard_deadline(
        self,
        deadline,
        message='Cartesian motion timed out',
    ):
        failure = self._servo_safety_failure()
        if failure is not None:
            raise PressFailure(failure)
        if time.monotonic() >= deadline:
            raise PressFailure(message)

    def _servo_safety_failure(self):
        with self._condition:
            code = self._servo_status_code
            received_at = self._servo_status_received_at
            command_started_at = self._servo_command_started_at
        if command_started_at <= 0.0 or received_at < command_started_at:
            return None
        return {
            2: 'MoveIt Servo halted at a singularity (status=2)',
            4: 'MoveIt Servo halted for collision (status=4)',
            5: 'MoveIt Servo halted at a joint bound (status=5)',
        }.get(code)

    def _wait_period(self):
        period = 1.0 / max(
            1.0,
            float(self.get_parameter('control_rate_hz').value),
        )
        self._stop_event.wait(period)

    def _publish_status(self, text):
        self._status_publisher.publish(String(data=str(text)))

    def _start_timed_phase(self, phase, status=None):
        if self._phase_timer is not None:
            self._phase_timer.start(phase)
        if status is not None:
            self._publish_status(status)

    def _publish_timing(self, button, completed):
        snapshot = (
            {'total_seconds': 0.0, 'phases': {}}
            if self._phase_timer is None
            else self._phase_timer.snapshot()
        )
        payload = {
            'button': str(button),
            'completed': bool(completed),
            'total_seconds': round(snapshot['total_seconds'], 6),
            'phases': {
                name: round(seconds, 6)
                for name, seconds in snapshot['phases'].items()
            },
        }
        self._timing_publisher.publish(
            String(data=json.dumps(payload, sort_keys=True))
        )

    def _publish_completion(self, completed):
        self._completion_publisher.publish(Bool(data=bool(completed)))

    def _publish_servo_claim(self, claimed):
        self._servo_claim_publisher.publish(Bool(data=bool(claimed)))


def main(args=None):
    rclpy.init(args=args)
    node = ButtonPressExecutor()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_event.set()
        node._publish_zero_twist()
        node._set_hardware_servo_gate(False, wait=False)
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
