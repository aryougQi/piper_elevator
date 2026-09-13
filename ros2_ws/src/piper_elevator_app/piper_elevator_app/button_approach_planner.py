import copy
from collections import Counter, deque
import json
import math
import threading
import time
import uuid

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints
from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.msg import JointConstraint
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.msg import MotionPlanRequest
from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.srv import GetPositionIK
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import JointState
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer
from tf2_ros import ConnectivityException
from tf2_ros import ExtrapolationException
from tf2_ros import LookupException
from tf2_ros import TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

from piper_elevator_app.motion_core import position_in_workspace
from piper_elevator_app.motion_core import (
    camera_centered_tool_approach_position,
)
from piper_elevator_app.motion_core import matrix_to_quaternion
from piper_elevator_app.motion_core import quaternion_to_matrix
from piper_elevator_app.motion_core import level_limited_camera_orientation
from piper_elevator_app.motion_core import check_servo_capture
from piper_elevator_app.coarse_approach_core import stable_observation_window
from piper_elevator_app.coarse_approach_core import CameraModel
from piper_elevator_app.coarse_approach_core import check_camera_view
from piper_elevator_app.coarse_approach_core import joint_configuration_cost
from piper_elevator_app.coarse_approach_core import joint_configuration_is_safe
from piper_elevator_app.coarse_approach_core import parse_arm_joint_limits
from piper_elevator_app.approach_quality import configuration_quality
from piper_elevator_app.approach_quality import trajectory_quality
from piper_elevator_app.joint_limits_core import bounded_goal_tolerances
from piper_elevator_app.joint_limits_core import merge_joint_limits
from piper_elevator_app.joint_limits_core import normalize_joint_positions
from piper_elevator_app.joint_limits_core import (
    trajectory_position_limit_violation,
)


class ButtonApproachPlanner(Node):
    """Transform a selected button and plan a safe MoveIt approach."""

    def __init__(self):
        super().__init__('button_approach_planner')
        self._declare_parameters()
        self._validate_configuration()
        self._callback_group = ReentrantCallbackGroup()
        self._lock = threading.Lock()
        self._busy = False
        self._latest_button = None
        self._latest_approach = None
        self._latest_received_at = 0.0
        self._latest_surface_normal = None
        self._surface_received_at = 0.0
        self._planned_trajectory = None
        self._planned_target = None
        self._planned_button = None
        self._plan_created_at = 0.0
        self._auto_started = False
        self._latest_joint_positions = {}
        self._latest_joint_received_at = 0.0
        self._latest_joint_stamp_ns = 0
        self._latest_tip_orientation = None
        self._camera_model = None
        self._latest_observation = None
        self._planned_observation = None
        self._execution_observation = None
        self._last_execution_diagnostic = {}
        self._last_ik_search_diagnostic = {}
        self._last_plan_quality_diagnostic = {}
        self._verified_handover = None
        self._observation_not_before_stamp_ns = 0
        self._observations = deque(maxlen=max(
            1, int(self.get_parameter('observation_stable_samples').value)
        ))
        self._observation_counts = Counter()
        self._observation_detail = 'waiting for surface poses'
        self._observation_last_rejection = ''
        self._observation_last_rejection_received_at = None
        self._observation_current_rejection = ''
        self._last_observation_wait = {}
        self._surface_input_received_at = None
        self._surface_input_stamp_ns = 0
        self._arm_joint_limits = None
        self._planning_deadline = None
        self._motion_stop_unconfirmed = False
        self._selected_button = ''
        self._selection_changed_stamp_ns = 0

        self._base_frame = self._string_parameter('base_frame')
        self._end_effector_link = self._string_parameter(
            'end_effector_link'
        )
        self._workspace_min = self._vector_parameter('workspace_min')
        self._workspace_max = self._vector_parameter('workspace_max')

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
        self._button_base_publisher = self.create_publisher(
            PoseStamped,
            self._string_parameter('button_base_topic'),
            latched_qos,
        )
        self._approach_publisher = self.create_publisher(
            PoseStamped,
            self._string_parameter('approach_pose_topic'),
            latched_qos,
        )
        self._status_publisher = self.create_publisher(
            String,
            self._string_parameter('status_topic'),
            latched_qos,
        )
        self._observation_status_publisher = self.create_publisher(
            String, '~/observation_status', latched_qos,
        )
        self.create_timer(
            1.0, self._publish_observation_diagnostics,
            callback_group=self._callback_group,
        )
        # Publish the exact trajectory stored by this node.  Otherwise RViz
        # may keep displaying an older plan made with its own MotionPlanning
        # panel while ~/execute correctly sends a different stored plan.
        self._display_trajectory_publisher = self.create_publisher(
            DisplayTrajectory,
            '/display_planned_path',
            10,
        )
        self.create_subscription(
            CameraInfo,
            self._string_parameter('camera_info_topic'),
            self._camera_info_callback,
            qos_profile_sensor_data,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            String, '/button_selected', self._selection_callback,
            latched_qos, callback_group=self._callback_group,
        )
        self.create_subscription(
            JointState,
            self._string_parameter('joint_state_topic'),
            self._joint_state_callback,
            10,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            PoseStamped,
            self._string_parameter('button_surface_pose_topic'),
            self._surface_pose_callback,
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
        self._gripper_client = ActionClient(
            self,
            FollowJointTrajectory,
            self._string_parameter('gripper_action'),
            callback_group=self._callback_group,
        )
        self._ik_client = self.create_client(
            GetPositionIK, self._string_parameter('ik_service'),
            callback_group=self._callback_group,
        )
        self._fk_client = self.create_client(
            GetPositionFK, self._string_parameter('fk_service'),
            callback_group=self._callback_group,
        )
        self._quality_plan_client = self.create_client(
            GetMotionPlan, self._string_parameter('motion_plan_service'),
            callback_group=self._callback_group,
        )
        self._description_client = self.create_client(
            GetParameters, self._string_parameter('robot_description_service'),
            callback_group=self._callback_group,
        )
        self.create_service(
            Trigger,
            '~/plan',
            self._plan_callback,
            callback_group=self._callback_group,
        )
        self._auto_timer = None
        if bool(self.get_parameter('auto_plan_execute').value):
            self._auto_timer = self.create_timer(
                1.0,
                self._auto_plan_execute_callback,
                callback_group=self._callback_group,
            )
        self.create_service(
            Trigger,
            '~/execute',
            self._execute_callback,
            callback_group=self._callback_group,
        )
        self.create_service(
            Trigger,
            '~/return_home',
            self._return_home_callback,
            callback_group=self._callback_group,
        )
        self.create_service(
            Trigger,
            '~/clear_plan',
            self._clear_plan_callback,
            callback_group=self._callback_group,
        )
        self.create_service(
            Trigger, '~/claim_servo', self._claim_servo_handover_callback,
            callback_group=self._callback_group,
        )

        self._publish_status('WAITING_FOR_BUTTON')
        approach_distance = self.get_parameter(
            'approach_distance_m'
        ).value
        coarse_vertical_offset = self.get_parameter(
            'coarse_vertical_offset_m'
        ).value
        self.get_logger().info(
            'Button approach planner ready: '
            f'base={self._base_frame}, tip={self._end_effector_link}, '
            f'approach={approach_distance:.3f} m, '
            f'execution={self.get_parameter("allow_execution").value}, '
            'policy=visible_joint_goal, '
            f'vertical_offset='
            f'{float(coarse_vertical_offset):.3f} m'
        )

    def _declare_parameters(self):
        self.declare_parameter('button_pose_topic', '/button_pose')
        self.declare_parameter(
            'button_surface_pose_topic',
            '/button_surface_pose',
        )
        self.declare_parameter('button_base_topic', '/button_pose_base')
        self.declare_parameter(
            'approach_pose_topic',
            '/button_approach_pose',
        )
        self.declare_parameter('status_topic', '/button_approach/status')
        self.declare_parameter(
            'joint_state_topic',
            '/piper_pika/joint_states',
        )
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter(
            'end_effector_link',
            'pika_fingertip_center_link',
        )
        self.declare_parameter('planning_group', 'arm')
        self.declare_parameter('move_group_action', '/move_action')
        self.declare_parameter('execute_action', '/execute_trajectory')
        self.declare_parameter(
            'gripper_action',
            '/pika_gripper_controller/follow_joint_trajectory',
        )
        self.declare_parameter('close_gripper_before_plan', True)
        self.declare_parameter('closed_gripper_position_m', 0.0)
        self.declare_parameter('gripper_motion_seconds', 1.0)
        self.declare_parameter('approach_distance_m', 0.14)
        self.declare_parameter('maximum_camera_centering_shift_m', 0.045)
        self.declare_parameter(
            'maximum_uncompensated_camera_offset_m',
            0.035,
        )
        self.declare_parameter('position_tolerance_m', 0.008)
        self.declare_parameter('pointing_tolerance_rad', math.radians(8.0))
        self.declare_parameter('roll_tolerance_rad', 0.26)
        self.declare_parameter('constrain_coarse_orientation', True)
        self.declare_parameter(
            'preserve_coarse_camera_orientation',
            False,
        )
        self.declare_parameter(
            'coarse_orientation_tolerance_rad',
            math.radians(5.0),
        )
        self.declare_parameter(
            'constrain_coarse_orientation_along_path',
            False,
        )
        self.declare_parameter('coarse_vertical_offset_m', 0.0)
        self.declare_parameter('preserve_wrist_roll_from_current', False)
        self.declare_parameter(
            'wrist_roll_guard_joints',
            ['joint4', 'joint6'],
        )
        self.declare_parameter(
            'wrist_roll_guard_tolerance_rad',
            1.0,
        )
        self.declare_parameter(
            'wrist_safe_joints',
            ['joint4', 'joint5', 'joint6'],
        )
        self.declare_parameter(
            'wrist_safe_centers_rad',
            [0.0, -0.60, 0.0],
        )
        self.declare_parameter(
            'wrist_safe_tolerances_rad',
            [1.20, 0.15, 1.50],
        )
        self.declare_parameter('wrist_singularity_joint', 'joint5')
        self.declare_parameter('minimum_abs_wrist_bend_rad', 0.40)
        self.declare_parameter('wrist_joint_lower_limit_rad', -1.2217304)
        self.declare_parameter('wrist_joint_upper_limit_rad', 1.2217304)
        self.declare_parameter('servo_joint_limit_margin_rad', 0.10)
        self.declare_parameter(
            'home_joint_names',
            ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
        )
        self.declare_parameter(
            'home_joint_positions_rad',
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        self.declare_parameter('home_joint_tolerance_rad', 0.015)
        self.declare_parameter('planning_time_seconds', 4.0)
        self.declare_parameter('planning_attempts', 3)
        self.declare_parameter('planning_request_attempts', 3)
        self.declare_parameter('planning_retry_delay_seconds', 0.15)
        self.declare_parameter('velocity_scaling', 0.10)
        self.declare_parameter('acceleration_scaling', 0.10)
        self.declare_parameter('tf_timeout_seconds', 0.25)
        self.declare_parameter('maximum_tf_fallback_age_seconds', 0.03)
        # 仿真时钟按物理步长推进，/clock 与 TF/关节时间戳可能相差一个步长。
        # 真机（simulation_mode=false）下该容差恒为 0，判定不变。
        self.declare_parameter(
            'simulation_future_stamp_tolerance_seconds',
            0.02,
        )
        self.declare_parameter('action_timeout_seconds', 30.0)
        self.declare_parameter('execution_joint_tolerance_rad', 0.010)
        self.declare_parameter('execution_settle_timeout_seconds', 20.0)
        self.declare_parameter('joint_state_max_age_seconds', 0.50)
        self.declare_parameter('home_execution_retries', 2)
        self.declare_parameter('home_retry_delay_seconds', 0.15)
        self.declare_parameter('target_max_age_seconds', 1.0)
        self.declare_parameter('surface_normal_max_age_seconds', 0.5)
        self.declare_parameter('plan_max_age_seconds', 120.0)
        self.declare_parameter('max_target_drift_m', 0.03)
        self.declare_parameter('maximum_execution_position_error_m', 0.015)
        self.declare_parameter(
            'maximum_execution_orientation_error_rad',
            0.35,
        )
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
        self.declare_parameter('auto_plan_execute', False)
        parameters = {
            'camera_frame': 'camera_color_optical_frame',
            'camera_info_topic': '/camera/color/camera_info',
            'ik_service': '/compute_ik',
            'fk_service': '/compute_fk',
            'motion_plan_service': '/plan_kinematic_path',
            'robot_description_service': '/move_group/get_parameters',
            'approach_distance_offsets_m': [0.0, 0.03, 0.06],
            'candidate_tilt_rad': math.radians(5.0),
            'candidate_roll_rad': math.radians(10.0),
            # 采样到限界值会因为 IK 实现误差被判越界（见 _candidate_poses），
            # 因此在限值内留一点采样余量。
            'candidate_tilt_margin_rad': math.radians(0.5),
            'maximum_camera_tilt_rad': math.radians(10.0),
            'maximum_camera_roll_rad': math.radians(15.0),
            'handover_maximum_camera_tilt_rad': math.radians(15.0),
            'handover_maximum_camera_roll_rad': math.radians(15.0),
            'handover_minimum_standoff_m': 0.08,
            'servo_standoff_distance_m': 0.03,
            'servo_maximum_start_error_m': 0.20,
            'servo_maximum_target_jump_m': 0.015,
            'handover_max_age_seconds': 120.0,
            'visibility_button_radius_m': 0.020,
            'visibility_position_uncertainty_m': 0.015,
            'visibility_image_margin_ratio': 0.12,
            'visibility_minimum_depth_m': 0.10,
            'visibility_maximum_depth_m': 2.0,
            'observation_stable_samples': 20,
            'observation_minimum_samples': 8,
            'observation_window_max_seconds': 3.0,
            'observation_position_tolerance_m': 0.008,
            'observation_normal_tolerance_rad': math.radians(5.0),
            'joint_goal_tolerance_rad': 0.005,
            'joint_limit_margin_rad': 0.15,
            'joint_state_boundary_tolerance_rad': 0.0001,
            'trajectory_boundary_tolerance_rad': 1.0e-9,
            'ik_timeout_seconds': 0.05,
            'ik_search_budget_seconds': 8.0,
            'maximum_ik_solutions': 8,
            'maximum_candidate_plans': 3,
            'planning_budget_seconds': 30.0,
            'planning_observation_wait_seconds': 6.0,
            'moveit_service_timeout_seconds': 1.0,
            'optimize_coarse_motion': True,
            'motion_quality_budget_seconds': 3.0,
            'motion_quality_maximum_ik_calls': 48,
            'motion_quality_maximum_plans': 2,
            'execution_timeout_margin_seconds': 5.0,
            'cancellation_timeout_seconds': 3.0,
            'execution_start_tolerance_rad': 0.02,
            'execution_stable_samples': 3,
            'execution_stable_velocity_rad_s': 0.03,
            'execution_stop_hold_seconds': 0.3,
            'post_execution_observation_timeout_seconds': 6.0,
        }
        for name, value in parameters.items():
            self.declare_parameter(name, value)

    def _string_parameter(self, name):
        return str(self.get_parameter(name).value)

    def _future_stamp_tolerance(self):
        """Return how far a stamp may lead the node clock.

        Gazebo advances sim time in one-millisecond physics steps, so the
        /clock sample this node holds can be one step older than the stamp
        carried by TF, joint state, and observation messages.  Only the
        simulation stack gets that slack: real hardware keeps the strict
        rule that a stamp ahead of the node clock is invalid.
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

    def _validate_configuration(self):
        positive = (
            'approach_distance_m', 'position_tolerance_m',
            'maximum_camera_tilt_rad', 'maximum_camera_roll_rad',
            'visibility_button_radius_m', 'visibility_minimum_depth_m',
            'visibility_maximum_depth_m', 'joint_goal_tolerance_rad',
            'joint_limit_margin_rad', 'ik_timeout_seconds',
            'joint_state_boundary_tolerance_rad',
            'trajectory_boundary_tolerance_rad',
            'ik_search_budget_seconds', 'planning_budget_seconds',
            'planning_observation_wait_seconds', 'observation_window_max_seconds',
            'planning_time_seconds', 'moveit_service_timeout_seconds',
            'execution_joint_tolerance_rad', 'execution_start_tolerance_rad',
            'execution_stable_velocity_rad_s', 'cancellation_timeout_seconds',
            'execution_stop_hold_seconds',
            'execution_timeout_margin_seconds', 'action_timeout_seconds',
            'execution_settle_timeout_seconds', 'joint_state_max_age_seconds',
            'surface_normal_max_age_seconds', 'target_max_age_seconds',
            'post_execution_observation_timeout_seconds',
            'observation_position_tolerance_m', 'observation_normal_tolerance_rad',
            'maximum_execution_position_error_m',
            'maximum_execution_orientation_error_rad',
            'handover_maximum_camera_tilt_rad', 'handover_maximum_camera_roll_rad',
            'handover_minimum_standoff_m', 'servo_standoff_distance_m',
            'servo_maximum_start_error_m',
            'motion_quality_budget_seconds',
            'servo_maximum_target_jump_m', 'handover_max_age_seconds',
        )
        for name in positive:
            value = float(self.get_parameter(name).value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('candidate_tilt_rad', 'candidate_roll_rad',
                     'candidate_tilt_margin_rad',
                     'visibility_position_uncertainty_m'):
            value = float(self.get_parameter(name).value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('maximum_ik_solutions', 'maximum_candidate_plans',
                     'motion_quality_maximum_ik_calls', 'motion_quality_maximum_plans',
                     'observation_stable_samples', 'execution_stable_samples'):
            if int(self.get_parameter(name).value) < 1:
                raise ValueError(f'{name} must be positive')
        if int(self.get_parameter('observation_stable_samples').value) < 3:
            raise ValueError('observation_stable_samples must be at least 3')
        minimum = int(self.get_parameter('observation_minimum_samples').value)
        if not 3 <= minimum <= int(self.get_parameter('observation_stable_samples').value):
            raise ValueError('observation_minimum_samples must be between 3 and observation_stable_samples')
        offsets = self.get_parameter('approach_distance_offsets_m').value
        if not offsets or any(not math.isfinite(x) or x < 0 for x in offsets):
            raise ValueError('Approach distance offsets must be nonnegative')
        margin = float(self.get_parameter(
            'visibility_image_margin_ratio'
        ).value)
        if not 0.0 <= margin < 0.5:
            raise ValueError('Image margin must be in [0, 0.5)')
        for candidate, maximum in (
            ('candidate_tilt_rad', 'maximum_camera_tilt_rad'),
            ('candidate_roll_rad', 'maximum_camera_roll_rad'),
            ('maximum_camera_tilt_rad', 'handover_maximum_camera_tilt_rad'),
            ('maximum_camera_roll_rad', 'handover_maximum_camera_roll_rad'),
        ):
            if float(self.get_parameter(candidate).value) > float(
                self.get_parameter(maximum).value
            ):
                raise ValueError(f'{candidate} exceeds {maximum}')
        if not float(self.get_parameter('servo_standoff_distance_m').value) < float(
            self.get_parameter('handover_minimum_standoff_m').value
        ):
            raise ValueError('Handover standoff must exceed the final Servo standoff')
        state_tolerance = float(self.get_parameter(
            'joint_state_boundary_tolerance_rad'
        ).value)
        trajectory_tolerance = float(self.get_parameter(
            'trajectory_boundary_tolerance_rad'
        ).value)
        if not trajectory_tolerance <= state_tolerance < min(
            float(self.get_parameter('joint_goal_tolerance_rad').value),
            float(self.get_parameter('execution_joint_tolerance_rad').value),
            float(self.get_parameter('execution_start_tolerance_rad').value),
        ):
            raise ValueError(
                'Joint boundary tolerances must be smaller than goal and '
                'execution tolerances'
            )
        if float(self.get_parameter('joint_limit_margin_rad').value) < float(
            self.get_parameter('servo_joint_limit_margin_rad').value
        ):
            raise ValueError(
                'Handover margin is smaller than Servo limit margin'
            )

    def _selection_callback(self, message):
        selected = str(message.data)
        with self._lock:
            if selected == self._selected_button:
                return
            self._selected_button = selected
            self._selection_changed_stamp_ns = (
                self.get_clock().now().nanoseconds
            )
            self._observations.clear()
            self._observation_detail = 'selection changed; acquiring new window'
            self._observation_reset_reason = ''
            self._latest_observation = None
            self._latest_button = None
            self._latest_approach = None
            self._latest_received_at = 0.0
            self._clear_stored_plan_locked()

    def _lookup_message_transform(self, target_frame, source_frame, stamp):
        """Use an exact TF when available, otherwise a bounded latest TF.

        Camera frames can arrive a few milliseconds ahead of the most recent
        robot-state TF.  Waiting for every exact transform inside a high-rate
        subscription can occupy all executor threads and prevent the TF
        listener from draining its own queue.  Probe the exact timestamp
        without blocking, then accept only a recent latest transform.
        """
        timeout = float(self.get_parameter('tf_timeout_seconds').value)
        if stamp.nanoseconds == 0:
            return self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=timeout),
            )
        try:
            return self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                stamp,
                timeout=Duration(seconds=0.0),
            )
        except (
            LookupException,
            ConnectivityException,
            ExtrapolationException,
        ):
            transform = self._tf_buffer.lookup_transform(
                target_frame,
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
                    'latest transform is too far from the camera frame: '
                    f'age={age:.3f}s limit={maximum_age:.3f}s'
                )
            return transform

    def _vector_parameter(self, name):
        values = np.asarray(self.get_parameter(name).value, dtype=np.float64)
        if values.shape != (3,):
            raise ValueError(f'{name} must contain exactly three values')
        return values

    def _camera_info_callback(self, message):
        if message.header.frame_id != self._string_parameter('camera_frame'):
            self.get_logger().warning(
                'CameraInfo frame must match camera_frame',
                throttle_duration_sec=2.0,
            )
            return
        try:
            model = CameraModel(
                message.width, message.height, message.k, message.d,
                message.distortion_model,
            )
        except ValueError as error:
            self.get_logger().warning(str(error), throttle_duration_sec=2.0)
            return
        with self._lock:
            self._camera_model = model

    def _joint_state_callback(self, message):
        if len(message.name) != len(message.position):
            return
        positions = dict(zip(message.name, map(float, message.position)))
        if not positions or not all(map(math.isfinite, positions.values())):
            return
        stamp_ns = Time.from_msg(message.header.stamp).nanoseconds
        if stamp_ns and (
            abs(self.get_clock().now().nanoseconds - stamp_ns) / 1e9
            > float(self.get_parameter('joint_state_max_age_seconds').value)
        ):
            return
        with self._lock:
            if stamp_ns and stamp_ns <= self._latest_joint_stamp_ns:
                return
            self._latest_joint_positions = positions
            self._latest_joint_received_at = time.monotonic()
            self._latest_joint_stamp_ns = stamp_ns
        self._monitor_active_joint_limits()
        self._update_physical_stop_guards()

    def _surface_pose_callback(self, message):
        # Surface PoseStamped already carries a same-frame point and normal.
        # Consuming it atomically avoids mixing two independent topic caches.
        stamp = Time.from_msg(message.header.stamp)
        stamp_ns = stamp.nanoseconds
        with self._lock:
            self._observation_counts['received'] += 1
            self._surface_input_received_at = time.monotonic()
            self._surface_input_stamp_ns = stamp_ns
        if message.header.frame_id != self._string_parameter('camera_frame'):
            self._reject_surface_observation('frame_mismatch')
            return
        age = (self.get_clock().now().nanoseconds - stamp_ns) / 1e9
        if stamp_ns <= 0 or age < -0.03 or age > float(
            self.get_parameter('surface_normal_max_age_seconds').value
        ):
            self._reject_surface_observation('invalid_stamp_or_age')
            return
        with self._lock:
            model = self._camera_model
            selected = self._selected_button
            selection_stamp = self._selection_changed_stamp_ns
        if stamp_ns < selection_stamp:
            self._reject_surface_observation('before_selection')
            return
        if model is None:
            self._reject_surface_observation('missing_camera_info')
            self._publish_status('WAITING_FOR_CAMERA_INFO')
            return
        point = np.array([
            message.pose.position.x,
            message.pose.position.y,
            message.pose.position.z,
        ])
        if not np.all(np.isfinite(point)) or point[2] <= 0.0:
            self._reject_surface_observation('invalid_position')
            return
        try:
            local_normal = quaternion_to_matrix([
                message.pose.orientation.x, message.pose.orientation.y,
                message.pose.orientation.z, message.pose.orientation.w,
            ])[:, 2]
            camera_tf = self._lookup_message_transform(
                self._base_frame, message.header.frame_id, stamp,
            )
            # The camera pose is evaluated at the image timestamp above.  Use
            # the same timestamp for the tip-to-camera mount transform too;
            # querying the latest transform here mixed a moving TCP state
            # with the captured image and introduced viewpoint-dependent
            # target errors during motion.  The helper keeps the existing
            # bounded fallback policy when an exact sample is unavailable.
            mount_tf = self._lookup_message_transform(
                self._end_effector_link, message.header.frame_id, stamp,
            )
            camera_position, camera_orientation = self._transform_arrays(
                camera_tf
            )
            mount_position, mount_orientation = self._transform_arrays(
                mount_tf
            )
            camera_matrix = quaternion_to_matrix(camera_orientation)
            normal = camera_matrix @ local_normal
            button = camera_position + camera_matrix @ point
            if np.dot(normal, camera_matrix @ point) < 0.0:
                normal = -normal
            observation = {
                'button': button,
                'normal': normal,
                'tip_to_camera_translation': mount_position,
                'tip_to_camera_quaternion': mount_orientation,
                'camera_orientation': camera_orientation,
                'camera_model': model,
                'camera_frame': message.header.frame_id,
                'stamp_ns': stamp_ns,
                'received_at': time.monotonic(),
                'selected_button': selected,
            }
        except (ValueError, LookupException, ConnectivityException,
                ExtrapolationException) as error:
            self._reject_surface_observation('geometry_or_tf', str(error))
            self.get_logger().warning(str(error), throttle_duration_sec=2.0)
            return

        with self._lock:
            if (selected != self._selected_button
                    or selection_stamp != self._selection_changed_stamp_ns):
                self._reject_surface_observation_locked('selection_changed')
                return
            if stamp_ns <= getattr(self, '_observation_not_before_stamp_ns', 0):
                self._reject_surface_observation_locked('before_post_motion_boundary')
                return
            if not self._observation_is_fresh(observation):
                self._reject_surface_observation_locked('expired_during_tf')
                return
            if self._observations:
                previous = self._observations[-1]
                if stamp_ns <= previous['stamp_ns']:
                    self._reject_surface_observation_locked('duplicate_or_out_of_order')
                    return
                drift = np.linalg.norm(button - previous['button'])
                source_gap = (stamp_ns - previous['stamp_ns']) / 1e9
                receipt_gap = observation['received_at'] - previous['received_at']
                if (
                    drift > float(self.get_parameter(
                        'max_target_drift_m'
                    ).value)
                    or max(source_gap, receipt_gap)
                    > float(self.get_parameter(
                        'surface_normal_max_age_seconds'
                    ).value)
                ):
                    self._observations.clear()
                    self._observation_counts['window_resets'] += 1
                    self._observation_reset_reason = (
                        f'window reset: drift={drift:.4f}m, '
                        f'source_gap={source_gap:.3f}s, receipt_gap={receipt_gap:.3f}s'
                    )
                    self._latest_observation = None
                    self._latest_approach = None
                    self._latest_button = None
                    self._latest_received_at = 0.0
            self._observations.append(observation)
            self._observation_counts['accepted_frames'] += 1
            self._observation_current_rejection = ''
            # Bound the acquisition-time span; delayed callbacks must not
            # turn old measurements into a fresh filter window.
            maximum_span = float(self.get_parameter(
                'observation_window_max_seconds'
            ).value)
            while (
                (stamp_ns - self._observations[0]['stamp_ns']) / 1e9
                > maximum_span
            ):
                self._observations.popleft()
            self._latest_observation = None
            self._latest_approach = None
            self._latest_button = None
            self._latest_received_at = 0.0
            samples = list(self._observations)
            minimum = int(self.get_parameter('observation_minimum_samples').value)
            # All points and normals are in base_link at their image times.
            # Keep the raw window on isolated noise; never retain its old output.
            mean_button, mean_normal, detail = stable_observation_window(
                samples, minimum,
                float(self.get_parameter('observation_position_tolerance_m').value),
                float(self.get_parameter('observation_normal_tolerance_rad').value),
            )
            if mean_normal is None:
                self._observation_counts['unstable_windows'] += 1
                self._observation_detail = (
                    detail + '; ' + getattr(self, '_observation_reset_reason', '')
                ).rstrip('; ')
                return
            self._observation_detail = detail
            self._observation_counts['stable_windows'] += 1
            observation = dict(
                observation, button=mean_button, normal=mean_normal
            )
            self._latest_observation = observation
            self._latest_received_at = observation['received_at']
            self._latest_surface_normal = mean_normal.copy()
            self._surface_received_at = observation['received_at']

        target = self._candidate_pose(
            observation,
            float(self.get_parameter('approach_distance_m').value),
            mean_normal, None,
        )
        button_pose = self._make_pose(
            mean_button, [0.0, 0.0, 0.0, 1.0], message.header.stamp,
        )
        with self._lock:
            if (
                selected != self._selected_button
                or self._latest_observation is None
                or observation['stamp_ns']
                != self._latest_observation['stamp_ns']
            ):
                return
            if (
                self._planned_button is not None
                and np.linalg.norm(mean_button - self._planned_button)
                > float(self.get_parameter('max_target_drift_m').value)
            ):
                self._clear_stored_plan_locked()
            self._latest_button = button_pose
            self._latest_approach = target
            busy = self._busy
        self._button_base_publisher.publish(button_pose)
        if not busy:
            self._approach_publisher.publish(target)
            self._publish_status('TARGET_READY')

    def _reject_surface_observation(self, reason, detail=''):
        with self._lock:
            self._reject_surface_observation_locked(reason, detail)

    def _reject_surface_observation_locked(self, reason, detail=''):
        self._observation_counts['rejected_' + reason] += 1
        self._observation_last_rejection = reason + (': ' + detail if detail else '')
        self._observation_last_rejection_received_at = time.monotonic()
        self._observation_current_rejection = self._observation_last_rejection

    def _observation_is_fresh(self, observation):
        if observation is None:
            return False
        limit = min(float(self.get_parameter(name).value) for name in (
            'target_max_age_seconds', 'surface_normal_max_age_seconds',
        ))
        stamp_ns = observation['stamp_ns']
        source_age = (self.get_clock().now().nanoseconds - stamp_ns) / 1e9
        receipt_age = time.monotonic() - observation['received_at']
        return stamp_ns > 0 and -0.03 <= source_age <= limit and 0.0 <= receipt_age <= limit

    def _observation_diagnostics(self):
        with self._lock:
            samples = list(self._observations)
            span = ((samples[-1]['stamp_ns'] - samples[0]['stamp_ns']) / 1e9
                    if len(samples) > 1 else 0.0)
            now = time.monotonic()
            ready = self._observation_is_fresh(self._latest_observation)
            input_receipt_age = (
                now - self._surface_input_received_at
                if self._surface_input_received_at is not None else None
            )
            input_source_age = (
                (self.get_clock().now().nanoseconds - self._surface_input_stamp_ns) / 1e9
                if self._surface_input_stamp_ns else None
            )
            current_rejection = getattr(self, '_observation_current_rejection', '')
            last_rejection_at = getattr(
                self, '_observation_last_rejection_received_at', None,
            )
            freshness_limit = min(float(self.get_parameter(name).value) for name in (
                'target_max_age_seconds', 'surface_normal_max_age_seconds',
            ))
            minimum_samples = int(self.get_parameter('observation_minimum_samples').value)
            if ready:
                blocking_reason = ''
            elif input_receipt_age is None:
                blocking_reason = 'no_surface_input'
            elif input_receipt_age > freshness_limit:
                blocking_reason = 'surface_input_stale'
            elif current_rejection:
                blocking_reason = 'surface_input_rejected'
            elif input_source_age is None or not -0.03 <= input_source_age <= freshness_limit:
                blocking_reason = 'surface_capture_stale_or_invalid'
            elif self._latest_observation is not None:
                blocking_reason = 'stable_observation_stale'
            elif len(samples) < minimum_samples:
                blocking_reason = 'acquiring_window'
            else:
                blocking_reason = 'unstable_window'
            return {
                'selected_button': self._selected_button,
                'ready': ready,
                'blocking_reason': blocking_reason,
                'window_samples': len(samples),
                'minimum_samples': minimum_samples,
                'maximum_samples': int(self.get_parameter('observation_stable_samples').value),
                'window_span_seconds': span,
                'window_rate_hz': (len(samples) - 1) / span if span > 0.0 else None,
                'input_receipt_age_seconds': input_receipt_age,
                'input_source_age_seconds': input_source_age,
                'window_detail': self._observation_detail,
                'current_rejection': current_rejection,
                'last_rejection': self._observation_last_rejection,
                'last_rejection_age_seconds': (
                    now - last_rejection_at if last_rejection_at is not None else None
                ),
                'counts_scope': 'node_lifetime',
                'counts': dict(self._observation_counts),
                'last_observation_wait': copy.deepcopy(
                    getattr(self, '_last_observation_wait', {})),
                'last_execution': copy.deepcopy(
                    getattr(self, '_last_execution_diagnostic', {})),
                'last_ik_search': copy.deepcopy(
                    getattr(self, '_last_ik_search_diagnostic', {})),
                'last_plan_quality': copy.deepcopy(
                    getattr(self, '_last_plan_quality_diagnostic', {})),
                'handover_available': bool(getattr(self, '_verified_handover', None)),
            }

    def _publish_observation_diagnostics(self):
        message = String()
        message.data = json.dumps(self._observation_diagnostics(), ensure_ascii=True)
        self._observation_status_publisher.publish(message)

    def _record_execution_diagnostic(self, **fields):
        with self._lock:
            if not hasattr(self, '_last_execution_diagnostic'):
                self._last_execution_diagnostic = {}
            self._last_execution_diagnostic.update(copy.deepcopy(fields))

    @staticmethod
    def _pose_diagnostic(pose):
        return {
            'frame_id': pose.header.frame_id,
            'position': [pose.pose.position.x, pose.pose.position.y,
                         pose.pose.position.z],
            'orientation': [pose.pose.orientation.x, pose.pose.orientation.y,
                            pose.pose.orientation.z, pose.pose.orientation.w],
        }

    @staticmethod
    def _transform_arrays(transform):
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return (
            np.array([translation.x, translation.y, translation.z]),
            np.array([rotation.x, rotation.y, rotation.z, rotation.w]),
        )

    def _make_pose(self, position, orientation, stamp):
        pose = PoseStamped()
        pose.header.frame_id = self._base_frame
        pose.header.stamp = stamp
        pose.pose.position.x = float(position[0])
        pose.pose.position.y = float(position[1])
        pose.pose.position.z = float(position[2])
        pose.pose.orientation.x = float(orientation[0])
        pose.pose.orientation.y = float(orientation[1])
        pose.pose.orientation.z = float(orientation[2])
        pose.pose.orientation.w = float(orientation[3])
        return pose

    def _plan_callback(self, request, response):
        del request
        with self._lock:
            if self._busy or self._motion_stop_unconfirmed:
                response.success = False
                response.message = (
                    'Planner busy or previous action has not stopped'
                )
                return response
            self._clear_stored_plan_locked()
            self._busy = True
            self._last_plan_quality_diagnostic = {}
        self._planning_deadline = time.monotonic() + float(
            self.get_parameter('planning_budget_seconds').value
        )
        self._publish_status('WAITING_FOR_STABLE_TARGET')
        try:
            observation = self._wait_for_planning_observation()
            self._publish_status('PLANNING')
            if (
                bool(self.get_parameter('simulation_mode').value)
                and bool(self.get_parameter('close_gripper_before_plan').value)
            ):
                closed, message = self._close_gripper()
                if not closed:
                    raise ValueError(message)
            self._load_arm_joint_limits()
            candidates = self._solve_visible_candidates(observation)
            if not candidates:
                raise ValueError(
                    'No acceptable coarse IK candidate; ' + self._ik_search_failure_detail()
                )
            message = 'No candidate trajectory passed endpoint validation'
            maximum = min(len(candidates), int(
                self.get_parameter('maximum_candidate_plans').value
            ))
            for index, (_, joints, target) in enumerate(candidates[:maximum]):
                if time.monotonic() >= self._planning_deadline:
                    message = 'Coarse planning budget exhausted'
                    break
                self._publish_status(
                    f'PLANNING candidate={index + 1}/{maximum}'
                )
                result, message = self._plan_constraints(
                    self._joint_goal_constraints(joints)
                )
                if result is None:
                    if not self._retryable_planning_failure(message):
                        break
                    continue
                safe, message = self._validate_planned_candidate(result, target, observation)
                if not safe:
                    continue
                result, target, message = self._improve_coarse_plan(
                    result, target, observation, candidates, message,
                )
                trajectory = result.planned_trajectory
                # A noisy window at the endpoint is not proof of target motion.
                # Wait within the existing total budget, then recheck identity,
                # position and normal against the original planning snapshot.
                self._wait_for_planning_observation()
                with self._lock:
                    latest = self._latest_observation
                    if latest is None:
                        raise ValueError(
                            'Stable target lost during planning; '
                            + getattr(self, '_observation_detail', 'reacquire')
                        )
                    if not self._observation_is_fresh(latest):
                        raise ValueError(
                            'Target observation expired during planning; reacquire'
                        )
                    if (
                        latest.get('selected_button')
                        != observation.get('selected_button')
                        or np.linalg.norm(
                            latest['button'] - observation['button']
                        )
                        > float(self.get_parameter('max_target_drift_m').value)
                        or math.acos(float(np.clip(
                            np.dot(latest['normal'], observation['normal']),
                            -1.0, 1.0,
                        ))) > float(self.get_parameter(
                            'observation_normal_tolerance_rad'
                        ).value)
                    ):
                        raise ValueError(
                            'Target changed during planning; reacquire'
                        )
                    self._planned_trajectory = trajectory
                    self._planned_target = copy.deepcopy(target)
                    self._planned_button = observation['button'].copy()
                    self._planned_observation = copy.deepcopy(observation)
                    self._plan_created_at = time.monotonic()
                self._approach_publisher.publish(target)
                self._publish_display_trajectory(result)
                response.success = True
                response.message = (
                    f'Plan ready: candidate {index + 1}/{maximum}; {message}'
                )
                self._publish_status('PLAN_READY')
                return response
            raise ValueError(message)
        except (ValueError, RuntimeError) as error:
            response.success = False
            response.message = str(error)
            self._publish_status(f'PLAN_FAILED: {error}')
            return response
        finally:
            self._planning_deadline = None
            with self._lock:
                self._busy = False

    def _wait_for_planning_observation(self):
        started = time.monotonic()
        with self._lock:
            counts_before = self._observation_counts.copy()
        deadline = min(
            self._planning_deadline,
            started + float(self.get_parameter(
                'planning_observation_wait_seconds'
            ).value),
        )
        accepted = None
        while time.monotonic() < deadline:
            with self._lock:
                observation = self._latest_observation
                if self._motion_stop_unconfirmed:
                    raise ValueError('Previous motion stop is unconfirmed')
                if self._observation_is_fresh(observation):
                    accepted = copy.deepcopy(observation)
                    break
            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
        with self._lock:
            wait_diagnostic = {
                'outcome': 'ready' if accepted is not None else 'timed_out',
                'elapsed_seconds': time.monotonic() - started,
                'counts': dict(self._observation_counts - counts_before),
            }
            self._last_observation_wait = wait_diagnostic
        if accepted is not None:
            return accepted
        diagnostic = self._observation_diagnostics()
        raise ValueError(
            'Timed out waiting for stable, fresh RGB-D observation; '
            + f'reason={diagnostic["blocking_reason"]}; '
            + diagnostic['window_detail']
            + f'; window={diagnostic["window_samples"]}/{diagnostic["maximum_samples"]}'
            + f'; input_age={diagnostic["input_source_age_seconds"]}'
            + f'; current_rejection={diagnostic["current_rejection"] or "none"}'
            + f'; waited_seconds={wait_diagnostic["elapsed_seconds"]:.3f}'
            + f'; wait_counts={wait_diagnostic["counts"]}'
        )

    def _call_moveit_service(self, client, request, deadline=None):
        timeout = float(self.get_parameter(
            'moveit_service_timeout_seconds'
        ).value)
        deadlines = [limit for limit in (deadline, self._planning_deadline)
                     if limit is not None]
        if deadlines:
            timeout = min(timeout, min(deadlines) - time.monotonic())
        if timeout <= 0.0:
            raise ValueError('MoveIt service budget exhausted')
        started = time.monotonic()
        if not client.wait_for_service(timeout_sec=timeout):
            raise ValueError(f'MoveIt service unavailable: {client.srv_name}')
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0.0:
            raise ValueError('MoveIt service budget exhausted')
        future = client.call_async(request)
        result = self._wait_for_future(future, remaining)
        if result is None:
            client.remove_pending_request(future)
            raise ValueError(f'MoveIt service timed out: {client.srv_name}')
        return result

    def _load_arm_joint_limits(self):
        if self._arm_joint_limits is not None:
            return
        names = list(self.get_parameter('home_joint_names').value)
        request = GetParameters.Request()
        fields = ('has_position_limits', 'min_position', 'max_position')
        request.names = ['robot_description'] + [
            f'robot_description_planning.joint_limits.{name}.{field}'
            for name in names for field in fields
        ]
        result = self._call_moveit_service(self._description_client, request)
        if (
            len(result.values) != len(request.names)
            or not result.values[0].string_value
        ):
            raise ValueError('MoveIt did not provide robot_description')
        urdf_limits = parse_arm_joint_limits(
            result.values[0].string_value, names,
        )
        overrides = {}
        for index, name in enumerate(names):
            enabled, lower, upper = result.values[1 + 3 * index:4 + 3 * index]
            if enabled.type == 1 and enabled.bool_value:
                if lower.type not in (2, 3) or upper.type not in (2, 3):
                    raise ValueError(
                        f'Incomplete MoveIt position bounds: {name}'
                    )
                overrides[name] = {
                    'has_position_limits': True,
                    'min_position': (lower.double_value if lower.type == 3
                                     else lower.integer_value),
                    'max_position': (upper.double_value if upper.type == 3
                                     else upper.integer_value),
                }
        limits = merge_joint_limits(urdf_limits, overrides)
        reserve = (
            float(self.get_parameter('joint_goal_tolerance_rad').value)
            + float(self.get_parameter('execution_joint_tolerance_rad').value)
        )
        margin = float(self.get_parameter('joint_limit_margin_rad').value)
        for name, (lower, upper) in limits.items():
            if lower + margin + reserve >= upper - margin - reserve:
                raise ValueError(
                    f'{name} limits cannot accommodate handover and '
                    'goal/tracking error margins'
                )
        self._arm_joint_limits = limits

    def _normalized_joint_positions(self, context):
        self._load_arm_joint_limits()
        positions = self._fresh_joint_positions()
        normalized, corrections = normalize_joint_positions(
            positions, self._arm_joint_limits,
            float(self.get_parameter(
                'joint_state_boundary_tolerance_rad'
            ).value),
            context=context,
        )
        if corrections:
            details = ', '.join(
                f'{name}: {before:+.9g} -> {after:+.9g} rad'
                for name, (before, after) in corrections.items()
            )
            self.get_logger().info(
                f'{context}: normalized boundary roundoff ({details})'
            )
        return normalized

    def _fresh_joint_positions(self):
        with self._lock:
            positions = self._latest_joint_positions.copy()
            age = time.monotonic() - self._latest_joint_received_at
        names = list(self.get_parameter('home_joint_names').value)
        if (
            age > float(self.get_parameter(
                'joint_state_max_age_seconds'
            ).value)
            or any(name not in positions for name in names)
            or not all(map(math.isfinite, positions.values()))
        ):
            raise ValueError('Fresh, complete arm joint feedback is required')
        return positions

    def _candidate_pose(self, observation, distance, direction, roll):
        camera_orientation = level_limited_camera_orientation(
            direction.copy(), observation['camera_orientation'].copy(),
            np.array([0.0, 0.0, 1.0]),
            float(self.get_parameter('candidate_roll_rad').value),
        )
        if roll is not None:
            camera_matrix = quaternion_to_matrix(
                level_limited_camera_orientation(
                    direction.copy(), observation['camera_orientation'].copy(),
                    np.array([0.0, 0.0, 1.0]), 0.0,
                )
            )
            c, s = math.cos(roll), math.sin(roll)
            camera_orientation = matrix_to_quaternion(
                camera_matrix @ np.array([
                    [c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]
                ])
            )
        mount_rotation = quaternion_to_matrix(
            observation['tip_to_camera_quaternion']
        )
        tool_orientation = matrix_to_quaternion(
            quaternion_to_matrix(camera_orientation) @ mount_rotation.T
        )
        position = camera_centered_tool_approach_position(
            observation['button'].copy(), observation['normal'].copy(),
            tool_orientation, observation['tip_to_camera_translation'],
            distance,
            float(self.get_parameter(
                'maximum_camera_centering_shift_m'
            ).value),
            float(self.get_parameter(
                'maximum_uncompensated_camera_offset_m'
            ).value),
        )
        position[2] += float(self.get_parameter(
            'coarse_vertical_offset_m'
        ).value)
        return self._make_pose(
            position, tool_orientation, self.get_clock().now().to_msg()
        )

    def _candidate_poses(self, observation):
        self._candidate_view_diagnostic = {
            'generated': 0, 'visible': 0, 'duplicates': 0, 'view_rejections': {},
        }
        seen = set()
        normal = observation['normal']
        level = quaternion_to_matrix(level_limited_camera_orientation(
            normal.copy(), observation['camera_orientation'].copy(),
            np.array([0.0, 0.0, 1.0]), 0.0,
        ))
        tilt = float(self.get_parameter('candidate_tilt_rad').value)
        roll = float(self.get_parameter('candidate_roll_rad').value)
        orientations = [(normal, None), (normal, 0.0),
                        (normal, -roll), (normal, roll)]
        for tangent in (level[:, 0], level[:, 1]):
            for sign in (-1.0, 1.0):
                direction = (
                    math.cos(tilt) * normal + sign * math.sin(tilt) * tangent
                )
                orientations.append((direction, None))
        # Preserve the original preferences, then cover tilt/roll combinations
        # up to the existing planning limit. A six-axis arm cannot generally
        # choose wrist bend independently of the requested camera orientation.
        maximum_tilt = float(self.get_parameter('maximum_camera_tilt_rad').value)
        # Sampling exactly at the limit is self-defeating: the IK realises the
        # requested direction only up to a small residual, which then reads as
        # ``tilt > maximum_camera_tilt_rad`` and rejects the candidate.  Leave
        # a sampling margin so the achieved tilt stays inside the limit.
        sampled_tilt = max(
            0.0,
            maximum_tilt - float(
                self.get_parameter('candidate_tilt_margin_rad').value
            ),
        )
        for angle in sorted(set((tilt, sampled_tilt))):
            for tangent in (level[:, 0], level[:, 1]):
                for sign in (-1.0, 1.0):
                    direction = math.cos(angle) * normal + sign * math.sin(angle) * tangent
                    for roll_angle in (0.0, -roll, roll):
                        orientations.append((direction, roll_angle))
        # Include the closest permitted optical direction to the current
        # view, which need not lie on one of the sampled tangent axes.
        current_axis = quaternion_to_matrix(observation['camera_orientation'])[:, 2]
        tangent = current_axis - np.dot(current_axis, normal) * normal
        tangent_length = float(np.linalg.norm(tangent))
        if tangent_length > 1e-9:
            angle = min(
                sampled_tilt,
                math.acos(float(np.clip(current_axis @ normal, -1, 1))),
            )
            orientations.append((
                math.cos(angle) * normal + math.sin(angle) * tangent / tangent_length,
                None,
            ))
        distance = float(self.get_parameter('approach_distance_m').value)
        for direction, roll_angle in orientations:
            for offset in self.get_parameter(
                'approach_distance_offsets_m'
            ).value:
                target = self._candidate_pose(
                    observation, distance + float(offset),
                    direction, roll_angle
                )
                self._candidate_view_diagnostic['generated'] += 1
                pose = self._pose_diagnostic(target)
                q = np.asarray(pose['orientation'])
                if q[3] < 0.0:
                    q = -q
                key = tuple(np.round([*pose['position'], *q], 8))
                if key in seen:
                    self._candidate_view_diagnostic['duplicates'] += 1
                    continue
                seen.add(key)
                safe, reason = self._view_is_safe(target, observation)
                if not safe:
                    rejections = self._candidate_view_diagnostic['view_rejections']
                    rejections[reason] = rejections.get(reason, 0) + 1
                    continue
                self._candidate_view_diagnostic['visible'] += 1
                yield target

    def _ik_search_failure_detail(self):
        with self._lock:
            report = copy.deepcopy(getattr(self, '_last_ik_search_diagnostic', {}))
        if not report:
            return 'no candidate search diagnostics available'
        return (
            f'selected={report["selected_button"]}; '
            f'visible={report["visible_candidates"]}; '
            f'tried={report["candidates_attempted"]}; calls={report["ik_calls"]}; '
            f'IK_success={report["ik_successes"]}; '
            f'IK_codes={report["ik_return_codes"]}; '
            f'joint_rejections={report["joint_rejections"]}, '
            f'examples={[item["reason"] for item in report["joint_rejection_examples"][:3]]}; '
            f'view_rejections={report["candidate_filter"].get("view_rejections", {})}; '
            f'stop={report["stop_reason"]}; elapsed={report["elapsed_seconds"]:.2f}s'
        )

    def _solve_visible_candidates(self, observation):
        current = self._normalized_joint_positions('IK start')
        names = list(self.get_parameter('home_joint_names').value)
        deadline = min(
            self._planning_deadline,
            time.monotonic() + float(self.get_parameter(
                'ik_search_budget_seconds'
            ).value),
        )
        # Nearby seeds plus both bent-wrist branches make IK exploration
        # repeatable without fixing one wrist branch as a planning constraint.
        seeds = [current.copy()]
        for bend, roll in ((-0.60, 0.0), (0.60, 0.0),
                           (-0.60, 0.5), (-0.60, -0.5),
                           (0.60, 0.5), (0.60, -0.5)):
            seed = current.copy()
            seed['joint5'] = bend
            seed['joint4'] = current['joint4'] + roll
            seed['joint6'] = current['joint6'] - roll
            for name, (lower, upper) in self._arm_joint_limits.items():
                seed[name] = float(np.clip(
                    seed[name], lower + 0.01, upper - 0.01
                ))
            seeds.append(seed)
        solutions = []
        maximum = int(self.get_parameter('maximum_ik_solutions').value)
        started = time.monotonic()
        targets = list(self._candidate_poses(observation))
        report = {
            'selected_button': observation.get('selected_button'),
            'observation_stamp_ns': observation.get('stamp_ns'),
            'button': observation['button'].tolist(),
            'normal': observation['normal'].tolist(),
            'current_joints': dict(current),
            'candidate_filter': copy.deepcopy(getattr(self, '_candidate_view_diagnostic', {})),
            'visible_candidates': len(targets), 'candidates_attempted': 0,
            'ik_calls': 0, 'ik_successes': 0, 'ik_return_codes': {},
            'joint_rejections': 0, 'joint_rejection_examples': [],
            'duplicate_solutions': 0, 'accepted_solutions': 0,
            'stop_reason': 'complete', 'elapsed_seconds': 0.0,
        }
        attempted = set()

        def finish(reason):
            report.update(
                stop_reason=reason, elapsed_seconds=time.monotonic() - started,
                candidates_attempted=len(attempted), accepted_solutions=len(solutions),
            )
            with self._lock:
                self._last_ik_search_diagnostic = copy.deepcopy(report)
            return sorted(solutions, key=lambda item: item[0])

        # Give every visible pose a first attempt before spending the bounded
        # budget on alternate wrist seeds at any one pose.
        for seed in seeds:
            for index, target in enumerate(targets):
                if time.monotonic() >= deadline or len(solutions) >= maximum:
                    return finish('solution_limit' if len(solutions) >= maximum else 'time_budget')
                attempted.add(index)
                report['ik_calls'] += 1
                request = GetPositionIK.Request()
                request.ik_request.group_name = self._string_parameter(
                    'planning_group'
                )
                request.ik_request.ik_link_name = self._end_effector_link
                request.ik_request.pose_stamped = target
                request.ik_request.avoid_collisions = True
                request.ik_request.robot_state.is_diff = True
                request.ik_request.robot_state.joint_state.name = list(seed)
                request.ik_request.robot_state.joint_state.position = list(
                    seed.values()
                )
                request.ik_request.timeout = Duration(seconds=min(
                    float(self.get_parameter('ik_timeout_seconds').value),
                    max(0.001, deadline - time.monotonic()),
                )).to_msg()
                try:
                    result = self._call_moveit_service(
                        self._ik_client, request, deadline
                    )
                except ValueError as error:
                    if time.monotonic() >= deadline:
                        return finish('time_budget')
                    finish('service_error: ' + str(error))
                    raise
                code = str(result.error_code.val)
                report['ik_return_codes'][code] = report['ik_return_codes'].get(code, 0) + 1
                if result.error_code.val != MoveItErrorCodes.SUCCESS:
                    continue
                report['ik_successes'] += 1
                joints = dict(zip(
                    result.solution.joint_state.name,
                    result.solution.joint_state.position,
                ))
                reserve = (
                    float(self.get_parameter('joint_goal_tolerance_rad').value)
                    + float(self.get_parameter(
                        'execution_joint_tolerance_rad'
                    ).value)
                )
                safe, reason = self._joint_configuration_is_safe(
                    joints, reserve_rad=reserve
                )
                if not safe:
                    report['joint_rejections'] += 1
                    if len(report['joint_rejection_examples']) < 5:
                        report['joint_rejection_examples'].append({
                            'candidate': index, 'reason': reason, 'joints': joints,
                        })
                    continue
                joints = {name: joints[name] for name in names}
                if any(
                    max(abs(joints[name] - old[name]) for name in names) < 0.01
                    for _, old, _ in solutions
                ):
                    report['duplicate_solutions'] += 1
                    continue
                score = joint_configuration_cost(
                    joints, current, self._arm_joint_limits
                )
                solutions.append((score, joints, target))
        return finish('complete')

    def _joint_configuration_is_safe(self, positions, reserve_rad=0.0):
        return joint_configuration_is_safe(
            positions, self._arm_joint_limits,
            float(self.get_parameter('joint_limit_margin_rad').value)
            + reserve_rad,
            self._string_parameter('wrist_singularity_joint'),
            float(self.get_parameter('minimum_abs_wrist_bend_rad').value)
            + reserve_rad,
        )

    def _validate_planned_candidate(self, result, target, observation, deadline=None):
        trajectory = result.planned_trajectory
        safe, message = self._trajectory_wrist_is_safe(trajectory)
        if not safe:
            return False, message
        endpoint_time = trajectory.joint_trajectory.points[-1].time_from_start
        duration = endpoint_time.sec + endpoint_time.nanosec * 1e-9
        if not 0.0 < duration <= float(self.get_parameter('action_timeout_seconds').value):
            return False, 'Candidate trajectory exceeds execution budget'
        endpoint = dict(zip(
            trajectory.joint_trajectory.joint_names,
            trajectory.joint_trajectory.points[-1].positions,
        ))
        actual = (self._fk_pose(endpoint) if deadline is None
                  else self._fk_pose(endpoint, deadline=deadline))
        return self._validate_endpoint(
            actual, target, observation,
            position_tolerance=float(self.get_parameter('position_tolerance_m').value),
        )

    def _quality_metrics(self, result, start):
        trajectory = result.planned_trajectory
        joints = trajectory.joint_trajectory
        endpoint = dict(zip(joints.joint_names, joints.points[-1].positions))
        configuration = configuration_quality(
            endpoint, start, self._arm_joint_limits,
            wrist_joint=self._string_parameter('wrist_singularity_joint'),
            minimum_abs_wrist_bend=float(self.get_parameter('minimum_abs_wrist_bend_rad').value),
        )
        path = trajectory_quality(trajectory, start_positions=start)
        return {
            'score': configuration['cost'] + path['cost'],
            'configuration': configuration, 'path': path,
        }

    def _quality_candidate_pool(self, observation, candidates, start, deadline):
        pool = [(joints, target) for _, joints, target in candidates]
        targets = list(self._candidate_poses(observation))
        attempted = getattr(self, '_last_ik_search_diagnostic', {}).get('candidates_attempted', 0)
        targets = targets[attempted:] + targets[:attempted]
        seed = dict(start)
        seed.update(candidates[0][1])
        names = list(self.get_parameter('home_joint_names').value)
        maximum = int(self.get_parameter('motion_quality_maximum_ik_calls').value)
        reserve = sum(float(self.get_parameter(name).value) for name in (
            'joint_goal_tolerance_rad', 'execution_joint_tolerance_rad',
        ))
        # This exploration starts only after the original policy has a valid
        # trajectory. It cannot spend the original candidate-search budget.
        for target in targets[:maximum]:
            if time.monotonic() >= deadline:
                break
            request = GetPositionIK.Request()
            ik = request.ik_request
            ik.group_name = self._string_parameter('planning_group')
            ik.ik_link_name = self._end_effector_link
            ik.pose_stamped = target
            ik.avoid_collisions = True
            ik.robot_state.is_diff = True
            ik.robot_state.joint_state.name = list(seed)
            ik.robot_state.joint_state.position = list(seed.values())
            ik.timeout = Duration(seconds=min(
                float(self.get_parameter('ik_timeout_seconds').value),
                max(0.001, deadline - time.monotonic()),
            )).to_msg()
            try:
                result = self._call_moveit_service(self._ik_client, request, deadline)
            except (ValueError, RuntimeError):
                break
            if result.error_code.val != MoveItErrorCodes.SUCCESS:
                continue
            joints = dict(zip(result.solution.joint_state.name, result.solution.joint_state.position))
            if not self._joint_configuration_is_safe(joints, reserve_rad=reserve)[0]:
                continue
            joints = {name: joints[name] for name in names}
            if any(max(abs(joints[name] - old[name]) for name in names) < 0.01
                   for old, _ in pool):
                continue
            pool.append((joints, target))
        return sorted(pool, key=lambda item: configuration_quality(
            item[0], start, self._arm_joint_limits,
            wrist_joint=self._string_parameter('wrist_singularity_joint'),
            minimum_abs_wrist_bend=float(self.get_parameter('minimum_abs_wrist_bend_rad').value),
        )['cost'])

    def _plan_quality_candidate(self, joints, start, deadline):
        available = deadline - time.monotonic()
        if available <= 0.1:
            raise ValueError('Motion quality budget exhausted')
        request = GetMotionPlan.Request()
        request.motion_plan_request = self._joint_motion_plan_request(
            self._joint_goal_constraints(joints), start,
            min(0.75, available - 0.1),
            pipeline='pilz_industrial_motion_planner', planner='PTP',
        )
        # A computation-only service keeps optional planning failures separate
        # from the action cancellation and physical-stop guards.
        response = self._call_moveit_service(self._quality_plan_client, request, deadline)
        plan = response.motion_plan_response
        if plan.error_code.val != MoveItErrorCodes.SUCCESS:
            raise ValueError(f'Preferred PTP planning error {plan.error_code.val}')
        result = MoveGroup.Result()
        result.planned_trajectory = plan.trajectory
        result.trajectory_start = plan.trajectory_start
        result.error_code = plan.error_code
        return result

    def _improve_coarse_plan(self, baseline, target, observation, candidates, message):
        best_result, best_target, best_message = baseline, target, message
        report = {'used_fallback': True, 'selected_pipeline': 'ompl', 'attempts': []}
        started = time.monotonic()
        try:
            if not bool(self.get_parameter('optimize_coarse_motion').value):
                report['reason'] = 'disabled'
                return best_result, best_target, best_message
            client = getattr(self, '_quality_plan_client', None)
            if client is None or not client.service_is_ready():
                report['reason'] = 'optional planning service unavailable'
                return best_result, best_target, best_message
            # Leave the full original target revalidation allowance intact.
            deadline = min(
                started + float(self.get_parameter('motion_quality_budget_seconds').value),
                self._planning_deadline - float(self.get_parameter('planning_observation_wait_seconds').value),
            )
            if deadline - started < 0.5:
                report['reason'] = 'preserving target revalidation budget'
                return best_result, best_target, best_message
            trajectory = baseline.planned_trajectory.joint_trajectory
            start = dict(zip(trajectory.joint_names, trajectory.points[0].positions))
            baseline_metrics = self._quality_metrics(baseline, start)
            best_metrics = baseline_metrics
            report.update(baseline=baseline_metrics, selected=baseline_metrics)
            pool = self._quality_candidate_pool(
                observation, candidates, start, min(started + 1.0, deadline - 0.5),
            )
            report['candidate_count'] = len(pool)
            maximum = int(self.get_parameter('motion_quality_maximum_plans').value)
            for joints, alternative in pool[:maximum]:
                if deadline - time.monotonic() < 0.15:
                    break
                attempt = {'joints': dict(joints), 'pipeline': 'pilz_ptp'}
                report['attempts'].append(attempt)
                try:
                    proposal = self._plan_quality_candidate(joints, start, deadline)
                    safe, detail = self._validate_planned_candidate(
                        proposal, alternative, observation, deadline=deadline,
                    )
                    attempt.update(safe=safe, message=detail)
                    if not safe:
                        continue
                    metrics = self._quality_metrics(proposal, start)
                    attempt['quality'] = metrics
                    if metrics['score'] + 1e-6 < best_metrics['score']:
                        best_result, best_target = proposal, alternative
                        best_message = 'Preferred synchronized joint path; ' + detail
                        best_metrics = metrics
                        report.update(used_fallback=False, selected_pipeline='pilz_ptp', selected=metrics)
                except (ValueError, RuntimeError) as error:
                    attempt.update(safe=False, message=str(error))
            report['reason'] = 'selected lower soft cost' if not report['used_fallback'] else 'retained original valid plan'
        except (ValueError, RuntimeError) as error:
            report['reason'] = 'optional quality search unavailable: ' + str(error)
        finally:
            report['elapsed_seconds'] = time.monotonic() - started
            with self._lock:
                self._last_plan_quality_diagnostic = copy.deepcopy(report)
        return best_result, best_target, best_message

    def _joint_goal_constraints(self, joints):
        constraints = Constraints()
        constraints.name = 'visible_coarse_joint_goal'
        tolerance = float(self.get_parameter('joint_goal_tolerance_rad').value)
        for name in self.get_parameter('home_joint_names').value:
            joint = JointConstraint()
            joint.joint_name = str(name)
            joint.position = float(joints[name])
            lower, upper = self._arm_joint_limits[name]
            margin = (
                float(self.get_parameter('joint_limit_margin_rad').value)
                + float(self.get_parameter(
                    'execution_joint_tolerance_rad'
                ).value)
            )
            below, above = bounded_goal_tolerances(
                joint.position, lower, upper, tolerance, margin=margin,
            )
            joint.tolerance_above = above
            joint.tolerance_below = below
            joint.weight = 1.0
            constraints.joint_constraints.append(joint)
        return constraints

    def _fk_pose(self, joints, deadline=None):
        state = self._fresh_joint_positions()
        state.update(joints)
        request = GetPositionFK.Request()
        request.header.frame_id = self._base_frame
        request.fk_link_names = [self._end_effector_link]
        request.robot_state.is_diff = True
        request.robot_state.joint_state.name = list(state)
        request.robot_state.joint_state.position = [
            float(value) for value in state.values()
        ]
        result = (self._call_moveit_service(self._fk_client, request)
                  if deadline is None else self._call_moveit_service(self._fk_client, request, deadline))
        if (
            result.error_code.val != MoveItErrorCodes.SUCCESS
            or len(result.pose_stamped) != 1
        ):
            raise ValueError('MoveIt FK failed for planned endpoint')
        if result.pose_stamped[0].header.frame_id != self._base_frame:
            raise ValueError('MoveIt FK returned an unexpected frame')
        return result.pose_stamped[0]

    def _view_is_safe(self, target, observation, *, handover=False):
        position = np.array([
            target.pose.position.x, target.pose.position.y,
            target.pose.position.z,
        ])
        orientation = np.array([
            target.pose.orientation.x, target.pose.orientation.y,
            target.pose.orientation.z, target.pose.orientation.w,
        ])
        if not position_in_workspace(
            position, self._workspace_min, self._workspace_max
        ):
            return False, 'TCP outside workspace'
        return check_camera_view(
            observation['button'], observation['normal'],
            position, orientation,
            observation['tip_to_camera_translation'],
            observation['tip_to_camera_quaternion'],
            observation['camera_model'],
            button_radius_m=float(self.get_parameter(
                'visibility_button_radius_m'
            ).value),
            position_uncertainty_m=float(self.get_parameter(
                'visibility_position_uncertainty_m'
            ).value),
            image_margin_ratio=float(self.get_parameter(
                'visibility_image_margin_ratio'
            ).value),
            minimum_depth_m=float(self.get_parameter(
                'visibility_minimum_depth_m'
            ).value),
            maximum_depth_m=float(self.get_parameter(
                'visibility_maximum_depth_m'
            ).value),
            maximum_tilt_rad=float(self.get_parameter(
                'handover_maximum_camera_tilt_rad' if handover else 'maximum_camera_tilt_rad'
            ).value),
            maximum_roll_rad=float(self.get_parameter(
                'handover_maximum_camera_roll_rad' if handover else 'maximum_camera_roll_rad'
            ).value),
        )

    def _validate_endpoint(
        self, actual, target, observation, position_tolerance=None, *, handover=False,
    ):
        position_error = np.linalg.norm(np.array([
            actual.pose.position.x - target.pose.position.x,
            actual.pose.position.y - target.pose.position.y,
            actual.pose.position.z - target.pose.position.z,
        ]))
        if position_tolerance is None:
            position_tolerance = float(self.get_parameter(
                'maximum_execution_position_error_m'
            ).value)
        if (
            not math.isfinite(position_error)
            or position_error > position_tolerance
        ):
            return False, (
                f'TCP endpoint error {position_error * 1000.0:.1f} mm'
            )
        safe, message = self._view_is_safe(actual, observation, handover=handover)
        detail = f'TCP error={position_error * 1000.0:.1f}mm; {message}'
        if not safe:
            return False, detail
        try:
            actual_rotation = quaternion_to_matrix([
                actual.pose.orientation.x, actual.pose.orientation.y,
                actual.pose.orientation.z, actual.pose.orientation.w,
            ])
            target_rotation = quaternion_to_matrix([
                target.pose.orientation.x, target.pose.orientation.y,
                target.pose.orientation.z, target.pose.orientation.w,
            ])
            orientation_error = math.acos(float(np.clip(
                (np.trace(target_rotation.T @ actual_rotation) - 1.0) / 2.0,
                -1.0, 1.0,
            )))
        except ValueError as error:
            return False, f'Invalid TCP endpoint orientation: {error}'
        detail += f'; TCP orientation error={math.degrees(orientation_error):.1f}deg'
        return orientation_error <= float(self.get_parameter(
            'maximum_execution_orientation_error_rad'
        ).value), detail

    def _execute_callback(self, request, response):
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
            response.message = 'Real execution requires calibrated camera TF'
            return response

        with self._lock:
            if self._busy or getattr(self, '_motion_stop_unconfirmed', False):
                response.success = False
                response.message = (
                    'Planner is busy or motion stop is unconfirmed'
                )
                return response
            age = time.monotonic() - self._plan_created_at
            if (
                self._planned_trajectory is None
                or age > float(
                    self.get_parameter('plan_max_age_seconds').value
                )
            ):
                self._clear_stored_plan_locked()
                response.success = False
                response.message = 'No valid plan; call ~/plan again'
                return response
            trajectory = copy.deepcopy(self._planned_trajectory)
            target = copy.deepcopy(self._planned_target)
            planned_button = copy.deepcopy(self._planned_button)
            self._execution_button = copy.deepcopy(planned_button)
            self._execution_observation = copy.deepcopy(
                getattr(self, '_planned_observation', None)
            )
            self._clear_stored_plan_locked()
            if self._execution_observation is None:
                response.success = False
                response.message = 'Stored plan has no frozen camera observation; replan required'
                return response
            target_age = time.monotonic() - self._latest_received_at
            current_button = self._latest_button
            if (
                current_button is None
                or planned_button is None
                or target_age > float(
                    self.get_parameter('target_max_age_seconds').value
                )
            ):
                response.success = False
                response.message = (
                    'Target is stale; acquire a fresh target and replan'
                )
                return response
            current_position = np.asarray([
                current_button.pose.position.x,
                current_button.pose.position.y,
                current_button.pose.position.z,
            ])
            if (
                not np.all(np.isfinite(current_position))
                or np.linalg.norm(current_position - planned_button) > float(
                    self.get_parameter('max_target_drift_m').value
                )
            ):
                response.success = False
                response.message = (
                    'Target moved after planning; replan required'
                )
                return response
            self._busy = True
            self._executing_coarse_target = True
            self._last_execution_diagnostic = {
                'phase': 'executing',
                'trajectory_reached': False,
                'handover_verified': False,
                'target_tcp': self._pose_diagnostic(target),
                'selected_button': self._execution_observation.get('selected_button'),
                'planned_button': self._execution_observation['button'].tolist(),
                'planned_normal': self._execution_observation['normal'].tolist(),
            }

        self._publish_status('EXECUTING')
        try:
            reached, message = self._execute_trajectory(trajectory)
            success = reached
            if reached:
                self._record_execution_diagnostic(
                    phase='verifying', trajectory_reached=True,
                )
                self._publish_status('VERIFYING_APPROACH')
                success, message = self._verify_approach_reached(target)
                if not success:
                    message = 'Trajectory reached; post-motion verification failed: ' + message
            phase = ('verified' if success else
                     'verification_failed' if reached else 'motion_failed')
            self._record_execution_diagnostic(
                phase=phase, handover_verified=success, message=message,
            )
            response.success = success
            response.message = message
            if success:
                self._publish_status('APPROACH_REACHED_VERIFIED')
            elif reached:
                self._publish_status(f'APPROACH_VERIFICATION_FAILED: {message}')
            else:
                self._publish_status(f'EXECUTION_FAILED: {message}')
            return response
        finally:
            with self._lock:
                self._busy = False
                self._executing_coarse_target = False

    def _clear_plan_callback(self, request, response):
        del request
        with self._lock:
            if self._busy:
                response.success = False
                response.message = 'Planner is busy'
                return response
            self._clear_stored_plan_locked()
        self._publish_status('PLAN_CLEARED')
        response.success = True
        response.message = 'Stored plan cleared'
        return response

    def _return_home_callback(self, request, response):
        del request
        if not bool(self.get_parameter('allow_execution').value):
            response.success = False
            response.message = 'Execution is disabled by allow_execution'
            return response
        with self._lock:
            if self._busy or self._motion_stop_unconfirmed:
                response.success = False
                response.message = 'Planner is busy'
                return response
            self._busy = True
            self._clear_stored_plan_locked()
        self._planning_deadline = None

        self._publish_status('HOME_PLANNING')
        try:
            try:
                constraints = self._home_constraints()
                acceptance = self._home_constraints(acceptance=True)
                current = self._normalized_joint_positions('Home verification')
            except ValueError as error:
                response.success = False
                response.message = str(error)
                self._publish_status(f'HOME_FAILED: {error}')
                return response
            if self._home_region_contains(acceptance, current):
                self._publish_status('HOME_VERIFYING_EXISTING_POSE')
                success, message = self._wait_for_home_region(acceptance)
                response.success = success
                response.message = (
                    f'Home already reached and stable; {message}'
                    if success else message
                )
                self._publish_status(
                    'HOME_COMPLETE' if success else f'HOME_FAILED: {message}'
                )
                return response
            attempts = max(
                1,
                int(
                    self.get_parameter('home_execution_retries').value
                ),
            )
            success = False
            message = 'Home execution did not run'
            for attempt in range(1, attempts + 1):
                result, message = self._plan_constraints(constraints)
                if result is None:
                    break
                self._publish_status(
                    f'HOME_EXECUTING attempt={attempt}/{attempts}'
                )
                success, message = self._execute_trajectory(
                    result.planned_trajectory
                )
                if success:
                    success, message = self._wait_for_home_region(acceptance)
                    break
                # -4 is MoveIt CONTROL_FAILED.  Immediately after Servo it
                # commonly represents a stale planned start state.  Re-read
                # the current state and replan; never replay the old path.
                if (
                    message != 'MoveIt execution error -4'
                    or attempt >= attempts
                ):
                    break
                self._publish_status(
                    f'HOME_REPLANNING_AFTER_START_RACE '
                    f'attempt={attempt}/{attempts}'
                )
                time.sleep(max(
                    0.0,
                    float(
                        self.get_parameter(
                            'home_retry_delay_seconds'
                        ).value
                    ),
                ))
            response.success = success
            response.message = (
                'MoveIt home pose reached' if success else message
            )
            self._publish_status(
                'HOME_COMPLETE'
                if success
                else f'HOME_FAILED: {message}'
            )
            return response
        finally:
            with self._lock:
                self._busy = False

    @staticmethod
    def _home_region_contains(constraints, positions):
        goals = constraints.joint_constraints
        return bool(goals) and all(
            goal.joint_name in positions
            and goal.position - goal.tolerance_below
            <= positions[goal.joint_name]
            <= goal.position + goal.tolerance_above
            for goal in goals
        )

    def _wait_for_home_region(self, constraints):
        names = [goal.joint_name for goal in constraints.joint_constraints]
        deadline = time.monotonic() + float(self.get_parameter(
            'execution_settle_timeout_seconds'
        ).value)
        baseline = time.monotonic()
        previous = None
        previous_at = None
        steady_since = None
        samples = 0
        required = max(3, int(self.get_parameter(
            'execution_stable_samples'
        ).value))
        maximum_speed = float(self.get_parameter(
            'execution_stable_velocity_rad_s'
        ).value)
        hold = float(self.get_parameter('execution_stop_hold_seconds').value)
        maximum_age = float(self.get_parameter(
            'joint_state_max_age_seconds'
        ).value)
        while time.monotonic() < deadline:
            with self._lock:
                raw = self._latest_joint_positions.copy()
                received_at = self._latest_joint_received_at
            if time.monotonic() - received_at > maximum_age:
                return (
                    False,
                    'Home verification requires fresh real joint feedback',
                )
            try:
                positions, _ = normalize_joint_positions(
                    raw, self._arm_joint_limits,
                    float(self.get_parameter(
                        'joint_state_boundary_tolerance_rad'
                    ).value),
                    context='home verification feedback',
                )
            except ValueError as error:
                return False, str(error)
            if not self._home_region_contains(constraints, positions):
                return (
                    False,
                    'Actual joints are outside the configured home region',
                )
            if received_at > baseline and (
                previous_at is None or received_at > previous_at
            ):
                if previous is None or received_at - previous_at > maximum_age:
                    steady_since = received_at
                    samples = 1
                else:
                    speed = max(
                        abs(raw[name] - previous[name])
                        / (received_at - previous_at)
                        for name in names
                    )
                    if speed > maximum_speed:
                        steady_since = received_at
                        samples = 1
                    else:
                        samples += 1
                previous = raw
                previous_at = received_at
                if samples >= required and received_at - steady_since >= hold:
                    error = max(
                        abs(positions[goal.joint_name] - goal.position)
                        for goal in constraints.joint_constraints
                    )
                    return (
                        True, f'Home region verified: max_error={error:.4f}rad'
                    )
            time.sleep(0.05)
        return (
            False,
            'Actual joints did not remain stationary inside the home region',
        )

    def _clear_stored_plan_locked(self):
        self._planned_trajectory = None
        self._planned_target = None
        self._planned_button = None
        self._planned_observation = None
        self._plan_created_at = 0.0
        self._verified_handover = None

    def _claim_servo_handover_callback(self, request, response):
        del request
        with self._lock:
            if self._busy or self._motion_stop_unconfirmed:
                response.success = False
                response.message = 'Coarse planner busy or physical stop unconfirmed'
                return response
            token = copy.deepcopy(getattr(self, '_verified_handover', None))
            if token is None:
                response.success = False
                response.message = 'No verified coarse handover; plan and execute the approach first'
                return response
            self._busy = True
        try:
            age = (self.get_clock().now().nanoseconds - token['verified_at_ns']) / 1e9
            if (
                age > float(
                    self.get_parameter('handover_max_age_seconds').value
                )
                or age < -self._future_stamp_tolerance()
            ):
                raise ValueError('Verified coarse handover expired; replan required')
            with self._lock:
                latest = copy.deepcopy(self._latest_observation)
                selected = self._selected_button
            if selected != token['selected_button']:
                raise ValueError('Selected button changed after coarse verification')
            if not self._observation_is_fresh(latest):
                raise ValueError('Fresh stable near-view observation is required for Servo handover')
            if latest.get('selected_button') != token['selected_button']:
                raise ValueError('Stable observation belongs to a different button')
            drift = float(np.linalg.norm(latest['button'] - np.asarray(token['button'])))
            if drift > float(self.get_parameter('servo_maximum_target_jump_m').value):
                raise ValueError(f'Button moved since coarse verification: {drift * 1000:.1f}mm')
            joints = self._fresh_joint_positions()
            if any(abs(joints[name] - value) > float(self.get_parameter(
                'execution_start_tolerance_rad').value)
                   for name, value in token['joint_positions'].items()):
                raise ValueError('Robot joints moved after coarse verification; replan required')
            safe, detail = self._joint_configuration_is_safe(joints)
            if not safe:
                raise ValueError(detail)
            transform = self._tf_buffer.lookup_transform(
                self._base_frame, self._end_effector_link, Time(),
                timeout=Duration(seconds=float(self.get_parameter('tf_timeout_seconds').value)),
            )
            stamp_ns = Time.from_msg(transform.header.stamp).nanoseconds
            age = (self.get_clock().now().nanoseconds - stamp_ns) / 1e9
            if (
                stamp_ns <= 0
                or age > float(
                    self.get_parameter('joint_state_max_age_seconds').value
                )
                or age < -self._future_stamp_tolerance()
            ):
                raise ValueError('Fresh actual TCP TF is required for Servo handover')
            position, orientation = self._transform_arrays(transform)
            actual = self._make_pose(position, orientation, transform.header.stamp)
            verified = self._make_pose(token['tcp_position'], token['tcp_orientation'], transform.header.stamp)
            safe, detail = self._validate_endpoint(actual, verified, latest, handover=True)
            if not safe:
                raise ValueError(detail)
            camera_q = matrix_to_quaternion(quaternion_to_matrix(orientation) @
                                            quaternion_to_matrix(latest['tip_to_camera_quaternion']))
            safe, detail = check_servo_capture(
                latest['button'], latest['normal'], position, camera_q,
                maximum_tilt_rad=float(self.get_parameter('handover_maximum_camera_tilt_rad').value),
                maximum_roll_rad=float(self.get_parameter('handover_maximum_camera_roll_rad').value),
                minimum_standoff_m=float(self.get_parameter('handover_minimum_standoff_m').value),
                target_standoff_m=float(self.get_parameter('servo_standoff_distance_m').value),
                maximum_start_error_m=float(self.get_parameter('servo_maximum_start_error_m').value),
            )
            if not safe:
                raise ValueError(detail)
            with self._lock:
                pending = self._verified_handover
                current = self._latest_observation
                if (pending is None or pending['handover_id'] != token['handover_id']
                        or self._selected_button != token['selected_button']
                        or self._motion_stop_unconfirmed):
                    raise ValueError('Coarse handover was invalidated while checking it')
                if (current is None or current['stamp_ns'] != latest['stamp_ns']
                        or not self._observation_is_fresh(latest)):
                    raise ValueError('Near-view observation changed during handover; retry')
                maximum_age = float(self.get_parameter('joint_state_max_age_seconds').value)
                now_ns = self.get_clock().now().nanoseconds
                joint_stamp = self._latest_joint_stamp_ns
                current_joints = self._latest_joint_positions
                future_tolerance = self._future_stamp_tolerance()
                joint_age = (now_ns - joint_stamp) / 1e9
                observation_age = (now_ns - stamp_ns) / 1e9
                if (time.monotonic() - self._latest_joint_received_at > maximum_age
                        or joint_stamp <= 0
                        or joint_age > maximum_age or joint_age < -future_tolerance
                        or observation_age > maximum_age
                        or observation_age < -future_tolerance):
                    raise ValueError('Robot feedback or TCP expired during handover')
                if any(name not in current_joints or abs(current_joints[name] - value) > float(
                    self.get_parameter('execution_start_tolerance_rad').value)
                       for name, value in token['joint_positions'].items()):
                    raise ValueError('Robot joints moved while checking the coarse handover')
                safe, detail = self._joint_configuration_is_safe(current_joints)
                if not safe:
                    raise ValueError(detail)
                token.update(
                    observation_stamp_ns=latest['stamp_ns'], button=latest['button'].tolist(),
                    normal=latest['normal'].tolist(), tcp_position=position.tolist(),
                    tcp_orientation=orientation.tolist(),
                )
                self._verified_handover = None
            response.success = True
            response.message = json.dumps(token, allow_nan=False)
            self._record_execution_diagnostic(handover_claimed=True, handover_id=token['handover_id'])
        except (ValueError, LookupException, ConnectivityException, ExtrapolationException) as error:
            response.success = False
            response.message = str(error)
        finally:
            with self._lock:
                self._busy = False
        return response

    def _publish_display_trajectory(self, result):
        display = DisplayTrajectory()
        display.trajectory_start = result.trajectory_start
        display.trajectory = [result.planned_trajectory]
        self._display_trajectory_publisher.publish(display)

    def _verify_approach_reached(self, target):
        observation = self._execution_observation
        if target is None or observation is None:
            return False, 'Executed plan has no frozen camera observation'
        baseline = self.get_clock().now().nanoseconds
        # The near view is a new measurement regime. Start a new window at
        # physical arrival; queued pre-arrival callbacks cannot refill it.
        with self._lock:
            self._observation_not_before_stamp_ns = baseline
            self._observations.clear()
            self._latest_observation = None
            self._latest_button = None
            self._latest_approach = None
            self._latest_received_at = 0.0
            self._latest_surface_normal = None
            self._surface_received_at = 0.0
            self._observation_reset_reason = 'post-motion window'
            self._observation_detail = 'acquiring post-motion samples'
        self._record_execution_diagnostic(post_motion_baseline_stamp_ns=baseline)
        deadline = time.monotonic() + float(self.get_parameter(
            'post_execution_observation_timeout_seconds'
        ).value)
        message = 'Waiting for fresh post-motion RGB-D observations'
        while time.monotonic() < deadline:
            with self._lock:
                latest = copy.deepcopy(self._latest_observation)
                samples = list(self._observations)
                selected = self._selected_button
            if selected != observation.get('selected_button'):
                return False, 'Selected button changed during execution'
            fresh = (
                latest is not None
                and len(samples) >= int(self.get_parameter(
                    'observation_minimum_samples'
                ).value)
                and all(item['stamp_ns'] > baseline for item in samples)
                and self._observation_is_fresh(latest)
            )
            if not fresh:
                with self._lock:
                    message = (
                        'Waiting for fresh post-motion RGB-D observations; '
                        + self._observation_detail
                    )
                time.sleep(0.05)
                continue
            if latest.get('selected_button') != observation.get(
                'selected_button'
            ):
                return False, 'Selected button changed during execution'
            position_drift = float(np.linalg.norm(
                latest['button'] - observation['button']
            ))
            if position_drift > float(
                self.get_parameter('max_target_drift_m').value
            ):
                return False, f'Observed button moved after execution: drift={position_drift * 1000:.1f}mm'
            normal_error = math.acos(float(np.clip(
                np.dot(latest['normal'], observation['normal']), -1.0, 1.0,
            )))
            # Window stability does not bound viewpoint-dependent bias. The
            # current camera view below decides handover readiness; retain
            # the far/near normal difference for calibration diagnostics.
            normal_detail = f'normal change across views={math.degrees(normal_error):.1f}deg'
            self._record_execution_diagnostic(
                observed_button=latest['button'].tolist(),
                observed_normal=latest['normal'].tolist(),
                observation_stamp_ns=latest['stamp_ns'],
                normal_change_degrees=math.degrees(normal_error),
                button_drift_m=position_drift,
                post_motion_samples=len(samples),
            )
            try:
                transform = self._tf_buffer.lookup_transform(
                    self._base_frame, self._end_effector_link, Time(),
                    timeout=Duration(seconds=float(
                        self.get_parameter('tf_timeout_seconds').value
                    )),
                )
                stamp_ns = Time.from_msg(transform.header.stamp).nanoseconds
                if stamp_ns <= 0 or abs(
                    self.get_clock().now().nanoseconds - stamp_ns
                ) / 1e9 > float(self.get_parameter(
                    'joint_state_max_age_seconds'
                ).value):
                    message = 'Actual TCP transform is stale'
                    time.sleep(0.05)
                    continue
                position, orientation = self._transform_arrays(transform)
                actual = self._make_pose(
                    position, orientation, transform.header.stamp
                )
                self._record_execution_diagnostic(
                    actual_tcp=self._pose_diagnostic(actual),
                    actual_tcp_stamp_ns=stamp_ns,
                )
                safe, message = self._validate_endpoint(actual, target, latest, handover=True)
                if safe:
                    camera_orientation = matrix_to_quaternion(
                        quaternion_to_matrix(orientation)
                        @ quaternion_to_matrix(latest['tip_to_camera_quaternion'])
                    )
                    safe, capture_message = check_servo_capture(
                        latest['button'], latest['normal'], position, camera_orientation,
                        maximum_tilt_rad=float(self.get_parameter(
                            'handover_maximum_camera_tilt_rad').value),
                        maximum_roll_rad=float(self.get_parameter(
                            'handover_maximum_camera_roll_rad').value),
                        minimum_standoff_m=float(self.get_parameter(
                            'handover_minimum_standoff_m').value),
                        target_standoff_m=float(self.get_parameter(
                            'servo_standoff_distance_m').value),
                        maximum_start_error_m=float(self.get_parameter(
                            'servo_maximum_start_error_m').value),
                    )
                    message += '; ' + capture_message
                message += '; ' + normal_detail
                if not safe:
                    self._record_execution_diagnostic(last_check=message)
                    time.sleep(0.05)
                    continue
                joints = self._fresh_joint_positions()
                safe, wrist_message = self._joint_configuration_is_safe(joints)
                if not safe:
                    return False, wrist_message
                with self._lock:
                    current = self._latest_observation
                    if self._selected_button != observation.get('selected_button'):
                        return False, 'Selected button changed during execution'
                    current_snapshot = not (
                        current is None
                        or current['stamp_ns'] != latest['stamp_ns']
                        or not self._observation_is_fresh(latest)
                        or (self.get_clock().now().nanoseconds - stamp_ns) / 1e9
                        > float(self.get_parameter('joint_state_max_age_seconds').value)
                    )
                    if current_snapshot:
                        self._verified_handover = {
                            'schema_version': 1, 'handover_id': uuid.uuid4().hex,
                            'selected_button': latest.get('selected_button'),
                            'frame_id': self._base_frame,
                            'verified_at_ns': self.get_clock().now().nanoseconds,
                            'observation_stamp_ns': latest['stamp_ns'],
                            'button': latest['button'].tolist(), 'normal': latest['normal'].tolist(),
                            'tcp_position': position.tolist(), 'tcp_orientation': orientation.tolist(),
                            'joint_positions': {name: float(joints[name]) for name in self._arm_joint_limits},
                        }
                if not current_snapshot:
                    message = 'Observation or TCP changed/expired during post-motion verification'
                    time.sleep(0.05)
                    continue
                self._record_execution_diagnostic(last_check=message)
                return True, f'Approach reached and observed: {message}'
            except (ValueError, LookupException, ConnectivityException,
                    ExtrapolationException) as error:
                message = str(error)
                time.sleep(0.05)
        return False, message

    def _auto_plan_execute_callback(self):
        with self._lock:
            target_is_fresh = (
                self._latest_approach is not None
                and time.monotonic() - self._latest_received_at
                <= float(
                    self.get_parameter('target_max_age_seconds').value
                )
            )
            if (self._auto_started or not target_is_fresh
                    or self._motion_stop_unconfirmed):
                return
            self._auto_started = True
        if self._auto_timer is not None:
            self._auto_timer.cancel()

        self.get_logger().info('Automatic simulation planning started')
        plan_response = self._plan_callback(
            Trigger.Request(),
            Trigger.Response(),
        )
        if not plan_response.success:
            self.get_logger().error(
                f'Automatic planning failed: {plan_response.message}'
            )
            return
        self.get_logger().info(plan_response.message)

        execute_response = self._execute_callback(
            Trigger.Request(),
            Trigger.Response(),
        )
        if not execute_response.success:
            self.get_logger().error(
                f'Automatic execution failed: {execute_response.message}'
            )
            return
        self.get_logger().info(execute_response.message)

    @staticmethod
    def _retryable_planning_failure(message):
        """Retry only completed MoveIt requests that found no valid plan."""
        return str(message).startswith('MoveIt planning error ')

    def _joint_motion_plan_request(
        self, constraints, start, allowed_time, *,
        pipeline='ompl', planner='RRTConnectkConfigDefault',
    ):
        request = MotionPlanRequest()
        request.group_name = self._string_parameter('planning_group')
        request.pipeline_id = pipeline
        request.planner_id = planner
        request.num_planning_attempts = (
            int(self.get_parameter('planning_attempts').value) if pipeline == 'ompl' else 1
        )
        request.allowed_planning_time = float(allowed_time)
        request.max_velocity_scaling_factor = float(self.get_parameter('velocity_scaling').value)
        request.max_acceleration_scaling_factor = float(self.get_parameter('acceleration_scaling').value)
        request.start_state.is_diff = True
        request.start_state.joint_state.name = list(start)
        request.start_state.joint_state.position = list(start.values())
        request.workspace_parameters.header.frame_id = self._base_frame
        minimum = request.workspace_parameters.min_corner
        maximum = request.workspace_parameters.max_corner
        minimum.x, minimum.y, minimum.z = self._workspace_min.tolist()
        maximum.x, maximum.y, maximum.z = self._workspace_max.tolist()
        request.goal_constraints = [constraints]
        return request

    def _plan_constraints(self, constraints, path_constraints=None):
        if getattr(self, '_motion_stop_unconfirmed', False):
            return None, 'Previous action stop is unconfirmed'
        action_timeout = float(
            self.get_parameter('action_timeout_seconds').value
        )
        transport_timeout = min(
            action_timeout,
            float(self.get_parameter('cancellation_timeout_seconds').value),
        )
        deadline = getattr(self, '_planning_deadline', None)

        def remaining(maximum):
            if deadline is None:
                return maximum
            return max(0.0, min(maximum, deadline - time.monotonic()))

        if remaining(transport_timeout) <= 0.0:
            return None, 'Coarse planning budget exhausted'
        if not self._move_group_client.wait_for_server(
            timeout_sec=remaining(transport_timeout)
        ):
            return None, 'MoveGroup action server is unavailable'
        try:
            feedback = self._normalized_joint_positions(
                'MoveIt planning start'
            )
        except (ValueError, RuntimeError) as error:
            return None, str(error)

        goal = MoveGroup.Goal()
        goal.request = self._joint_motion_plan_request(
            constraints, feedback,
            remaining(float(self.get_parameter('planning_time_seconds').value)),
        )
        if goal.request.allowed_planning_time <= 0.0:
            return None, 'Coarse planning budget exhausted'
        if path_constraints is not None:
            goal.request.path_constraints = path_constraints
        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        goal_handle, message = self._send_action_goal(
            self._move_group_client,
            goal,
            'MoveGroup planning',
            remaining(transport_timeout),
        )
        if goal_handle is None:
            return None, message

        wrapped, message = self._wait_for_action_result(
            goal_handle,
            remaining(min(
                action_timeout,
                goal.request.allowed_planning_time + 2.0,
            )),
            'MoveGroup planning',
        )
        if wrapped is None:
            return None, message
        result = wrapped.result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            return None, f'MoveIt planning error {result.error_code.val}'
        if wrapped.status != 4:
            return None, 'MoveGroup planning did not finish successfully'
        if not result.planned_trajectory.joint_trajectory.points:
            return None, 'MoveIt returned an empty trajectory'
        return result, ''

    def _home_constraints(self, acceptance=False):
        names = list(self.get_parameter('home_joint_names').value)
        positions = list(
            self.get_parameter('home_joint_positions_rad').value
        )
        if len(names) != 6 or len(positions) != 6:
            raise ValueError('MoveIt home must contain six arm joints')
        if len(set(names)) != 6 or not np.all(np.isfinite(positions)):
            raise ValueError('MoveIt home joint configuration is invalid')
        home_tolerance = float(
            self.get_parameter('home_joint_tolerance_rad').value
        )
        tracking_tolerance = float(
            self.get_parameter('execution_joint_tolerance_rad').value
        )
        if (
            not math.isfinite(home_tolerance)
            or home_tolerance <= tracking_tolerance
        ):
            raise ValueError(
                'Home tolerance must exceed execution tracking tolerance'
            )
        tolerance = (
            home_tolerance if acceptance else min(
                float(self.get_parameter('joint_goal_tolerance_rad').value),
                home_tolerance - tracking_tolerance,
            )
        )
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError('MoveIt home goal tolerance must be positive')
        self._load_arm_joint_limits()
        constraints = Constraints()
        constraints.name = 'home'
        for name, position in zip(names, positions):
            if name not in self._arm_joint_limits:
                raise ValueError(
                    f'MoveIt home joint {name} has no effective limits'
                )
            lower, upper = self._arm_joint_limits[name]
            try:
                below, above = bounded_goal_tolerances(
                    position, lower, upper, tolerance,
                )
            except ValueError as error:
                raise ValueError(f'MoveIt home {name}: {error}') from error
            joint = JointConstraint()
            joint.joint_name = str(name)
            joint.position = float(position)
            joint.tolerance_above = above
            joint.tolerance_below = below
            joint.weight = 1.0
            constraints.joint_constraints.append(joint)
        return constraints

    def _motion_trajectory_is_safe(self, trajectory):
        try:
            self._load_arm_joint_limits()
        except ValueError as error:
            return False, str(error)
        joint_trajectory = trajectory.joint_trajectory
        if joint_trajectory.points:
            first_time = joint_trajectory.points[0].time_from_start
            # A delayed first point adds a controller-generated segment from
            # current position/velocity that is absent from the checked path.
            if first_time.sec != 0 or first_time.nanosec != 0:
                return False, (
                    'Trajectory must start at time zero; '
                    'controller initial interpolation would be unchecked'
                )
        violation = trajectory_position_limit_violation(
            joint_trajectory.joint_names, joint_trajectory.points,
            self._arm_joint_limits,
            float(self.get_parameter(
                'trajectory_boundary_tolerance_rad'
            ).value),
        )
        return (
            (False, violation) if violation is not None
            else (True, 'Trajectory samples and spline extrema within bounds')
        )

    def _trajectory_wrist_is_safe(self, trajectory):
        valid, message = self._motion_trajectory_is_safe(trajectory)
        if not valid:
            return False, message
        joints = trajectory.joint_trajectory
        endpoint = dict(zip(joints.joint_names, joints.points[-1].positions))
        # Reserve tracking error here; IK also reserves the goal-region width.
        # Home uses the motion check alone because it is not a Servo handover.
        return self._joint_configuration_is_safe(
            endpoint,
            reserve_rad=float(
                self.get_parameter('execution_joint_tolerance_rad').value
            ),
        )

    def _close_gripper(self):
        if getattr(self, '_motion_stop_unconfirmed', False):
            return False, 'Previous action stop is unconfirmed'
        action_timeout = float(self.get_parameter(
            'action_timeout_seconds'
        ).value)
        transport_timeout = min(
            action_timeout,
            float(self.get_parameter('cancellation_timeout_seconds').value),
        )
        if not self._gripper_client.wait_for_server(
            timeout_sec=transport_timeout
        ):
            return False, 'Pika gripper action server is unavailable'

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [
            'center_joint',
            'pika_left_finger_joint',
            'pika_right_finger_joint',
        ]
        point = JointTrajectoryPoint()
        opening = float(
            self.get_parameter('closed_gripper_position_m').value
        )
        point.positions = [opening, 0.5 * opening, -0.5 * opening]
        point.time_from_start = Duration(
            seconds=float(
                self.get_parameter('gripper_motion_seconds').value
            )
        ).to_msg()
        goal.trajectory.points = [point]

        goal_handle, message = self._send_action_goal(
            self._gripper_client, goal, 'Pika close command', transport_timeout
        )
        if goal_handle is None:
            return False, message
        wrapped, message = self._wait_for_action_result(
            goal_handle,
            min(action_timeout, float(
                self.get_parameter('gripper_motion_seconds').value
            ) + float(self.get_parameter(
                'execution_timeout_margin_seconds'
            ).value)),
            'Pika close command',
        )
        if wrapped is None:
            return False, message
        if (
            wrapped.result.error_code
            != FollowJointTrajectory.Result.SUCCESSFUL
        ):
            return (
                False,
                'Pika close command failed with error '
                f'{wrapped.result.error_code}',
            )
        if wrapped.status != 4:
            return False, 'Pika close command did not finish successfully'
        return True, 'Pika gripper closed'

    def _execute_trajectory(self, trajectory):
        if getattr(self, '_motion_stop_unconfirmed', False):
            return False, 'Previous action stop is unconfirmed'
        valid, message = self._trajectory_start_is_current(trajectory)
        if not valid:
            return False, message
        validator = (
            self._trajectory_wrist_is_safe
            if getattr(self, '_executing_coarse_target', False)
            else self._motion_trajectory_is_safe
        )
        valid, message = validator(trajectory)
        if not valid:
            return False, f'Trajectory rejected before execution: {message}'
        action_timeout = float(self.get_parameter(
            'action_timeout_seconds'
        ).value)
        endpoint_time = trajectory.joint_trajectory.points[-1].time_from_start
        duration = (
            float(endpoint_time.sec) + float(endpoint_time.nanosec) * 1e-9
        )
        if (
            not math.isfinite(duration)
            or duration <= 0.0 or duration > action_timeout
        ):
            return False, (
                'Trajectory duration is invalid or exceeds execution budget'
            )
        transport_timeout = min(
            action_timeout,
            float(self.get_parameter('cancellation_timeout_seconds').value),
        )
        if not self._execute_client.wait_for_server(
            timeout_sec=transport_timeout
        ):
            return False, 'ExecuteTrajectory action server is unavailable'
        valid, message = self._trajectory_start_is_current(trajectory)
        if not valid:
            return False, message
        if getattr(self, '_executing_coarse_target', False):
            valid, message = self._execution_target_is_current()
            if not valid:
                return False, message
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        self._execution_started_at = time.monotonic()
        physical_stop_token = self._start_physical_stop_guard()
        goal_handle, message = self._send_action_goal(
            self._execute_client, goal, 'Trajectory execution',
            transport_timeout,
            physical_stop_token=physical_stop_token,
        )
        if goal_handle is None:
            return False, message
        with self._lock:
            self._active_arm_goal_handle = goal_handle
            self._active_arm_result_future = None
            self._execution_limit_violation = None
        try:
            wrapped, message = self._wait_for_action_result(
                goal_handle,
                min(action_timeout, duration + float(
                    self.get_parameter(
                        'execution_timeout_margin_seconds'
                    ).value
                )),
                'Trajectory execution',
                physical_stop_token=physical_stop_token,
            )
            with self._lock:
                violation = self._execution_limit_violation
            if violation:
                return False, self._physical_stop_failure_message(
                    violation, physical_stop_token
                )
            if wrapped is None:
                return False, self._physical_stop_failure_message(
                    message, physical_stop_token
                )
            if wrapped.result.error_code.val != MoveItErrorCodes.SUCCESS:
                return (
                    False,
                    self._physical_stop_failure_message(
                        f'MoveIt execution error '
                        f'{wrapped.result.error_code.val}',
                        physical_stop_token,
                    ),
                )
            if wrapped.status != 4:
                return False, self._physical_stop_failure_message(
                    'Trajectory execution did not finish successfully',
                    physical_stop_token
                )
            success, message = self._wait_for_real_joint_endpoint(trajectory)
            with self._lock:
                violation = self._execution_limit_violation
            if violation:
                success, message = False, violation
            if success:
                self._release_physical_stop_guard(physical_stop_token)
                return True, message
            return False, self._physical_stop_failure_message(
                message, physical_stop_token
            )
        finally:
            with self._lock:
                if self._active_arm_goal_handle is goal_handle:
                    self._active_arm_goal_handle = None
                    self._active_arm_result_future = None

    def _monitor_active_joint_limits(self):
        with self._lock:
            handle = getattr(self, '_active_arm_goal_handle', None)
            if handle is None or getattr(
                self, '_execution_limit_violation', None
            ):
                return
            limits = getattr(self, '_arm_joint_limits', None)
            if not limits:
                return
            future = getattr(self, '_active_arm_result_future', None)
            positions = self._latest_joint_positions.copy()
            age = time.monotonic() - self._latest_joint_received_at
        if future is not None and future.done():
            try:
                if future.result().status in (4, 5, 6):
                    return
            except Exception:
                pass
        if age > float(self.get_parameter(
            'joint_state_max_age_seconds'
        ).value):
            return
        if any(name not in positions for name in limits):
            return
        try:
            normalize_joint_positions(
                positions, limits,
                float(self.get_parameter(
                    'joint_state_boundary_tolerance_rad'
                ).value),
                context='active arm feedback',
            )
            return
        except ValueError as error:
            violation = str(error)
        with self._lock:
            if (
                self._active_arm_goal_handle is not handle
                or self._execution_limit_violation is not None
            ):
                return
            if future is not None and future.done():
                try:
                    if future.result().status in (4, 5, 6):
                        return
                except Exception:
                    pass
            self._execution_limit_violation = violation
        try:
            handle.cancel_goal_async()
        except Exception as error:
            with self._lock:
                if self._active_arm_goal_handle is handle:
                    self._execution_limit_violation += (
                        f'; cancellation request failed: {error}'
                    )
        self._publish_status(f'EXECUTION_JOINT_LIMIT_VIOLATION: {violation}')

    def _execution_target_is_current(self):
        with self._lock:
            current = copy.deepcopy(self._latest_button)
            expected = copy.deepcopy(self._execution_button)
            age = time.monotonic() - self._latest_received_at
            latest_observation = copy.deepcopy(self._latest_observation)
            planned_observation = copy.deepcopy(self._execution_observation)
        if current is None or expected is None or age > float(
            self.get_parameter('target_max_age_seconds').value
        ):
            return False, 'Target is stale before execution; replan required'
        if latest_observation is None or planned_observation is None:
            return False, (
                'Stable paired observation unavailable before execution'
            )
        if not self._observation_is_fresh(latest_observation):
            return False, (
                'Paired observation is stale before execution; replan required'
            )
        if latest_observation.get(
            'selected_button'
        ) != planned_observation.get(
            'selected_button'
        ):
            return False, (
                'Selected button changed before execution; replan required'
            )
        normal_error = math.acos(float(np.clip(
            np.dot(
                latest_observation['normal'], planned_observation['normal']
            ),
            -1.0, 1.0,
        )))
        if not math.isfinite(normal_error) or normal_error > float(
            self.get_parameter('observation_normal_tolerance_rad').value
        ):
            return False, (
                'Panel normal changed before execution; replan required'
            )
        position = np.asarray([
            current.pose.position.x,
            current.pose.position.y,
            current.pose.position.z,
        ])
        if not np.all(np.isfinite(position)) or np.linalg.norm(
            position - expected
        ) > float(self.get_parameter('max_target_drift_m').value):
            return False, 'Target moved before execution; replan required'
        return True, ''

    def _trajectory_start_is_current(self, trajectory):
        joints = trajectory.joint_trajectory
        if not joints.points or not joints.joint_names:
            return False, 'Trajectory has no joint start state'
        names = list(joints.joint_names)
        start = list(joints.points[0].positions)
        if (
            len(start) != len(names)
            or len(set(names)) != len(names)
            or not all(math.isfinite(value) for value in start)
        ):
            return False, 'Trajectory joint start state is invalid'
        try:
            self._normalized_joint_positions('Execution start')
        except (ValueError, RuntimeError) as error:
            return False, str(error)
        with self._lock:
            feedback = self._latest_joint_positions.copy()
            age = time.monotonic() - self._latest_joint_received_at
        if (
            any(name not in feedback for name in names)
            or age > float(self.get_parameter(
                'joint_state_max_age_seconds'
            ).value)
            or not all(
                math.isfinite(feedback.get(name, math.nan)) for name in names
            )
        ):
            return False, (
                'Execution requires fresh complete real joint feedback'
            )
        error = max(
            abs(feedback[name] - value) for name, value in zip(names, start)
        )
        tolerance = float(self.get_parameter(
            'execution_start_tolerance_rad'
        ).value)
        if error > tolerance:
            return False, (
                f'Real joints moved from planned start: '
                f'max_error={error:.4f}rad; '
                'replan required'
            )
        return True, ''

    def _track_unconfirmed_action(self):
        token = object()
        with self._lock:
            if not hasattr(self, '_unconfirmed_actions'):
                self._unconfirmed_actions = set()
            self._unconfirmed_actions.add(token)
            self._motion_stop_unconfirmed = True
        return token

    def _start_physical_stop_guard(self):
        token = self._track_unconfirmed_action()
        with self._lock:
            if not hasattr(self, '_physical_stop_guards'):
                self._physical_stop_guards = {}
            self._physical_stop_guards[token] = {
                'terminal_at': None,
                'previous': None,
                'previous_at': None,
                'steady_since': None,
                'samples': 0,
            }
        return token

    def _release_physical_stop_guard(self, token):
        if token is None:
            return
        with self._lock:
            self._physical_stop_guards.pop(token, None)
        self._confirm_action_stopped(token)

    def _watch_physical_action_terminal(self, future, token):
        if token is None:
            return

        def terminal_result(done):
            try:
                wrapped = done.result()
            except Exception:
                return
            if wrapped is None or wrapped.status not in (4, 5, 6):
                return
            with self._lock:
                guard = self._physical_stop_guards.get(token)
                if guard is not None and guard['terminal_at'] is None:
                    guard['terminal_at'] = time.monotonic()

        future.add_done_callback(terminal_result)

    def _update_physical_stop_guards(self):
        """An action terminal state can precede the real proxy arm stopping."""
        with self._lock:
            if not getattr(self, '_physical_stop_guards', {}):
                return
        names = list(self.get_parameter('home_joint_names').value)
        maximum_age = float(self.get_parameter(
            'joint_state_max_age_seconds'
        ).value)
        maximum_speed = float(self.get_parameter(
            'execution_stable_velocity_rad_s'
        ).value)
        minimum_hold = float(self.get_parameter(
            'execution_stop_hold_seconds'
        ).value)
        now = time.monotonic()
        confirmed = []
        with self._lock:
            guards = getattr(self, '_physical_stop_guards', {})
            feedback = self._latest_joint_positions.copy()
            received_at = self._latest_joint_received_at
            complete = (
                bool(names)
                and all(
                    name in feedback and math.isfinite(feedback[name])
                    for name in names
                )
                and 0.0 <= now - received_at <= maximum_age
            )
            limits = getattr(self, '_arm_joint_limits', None)
            if complete and limits:
                try:
                    normalize_joint_positions(
                        feedback, limits,
                        float(self.get_parameter(
                            'joint_state_boundary_tolerance_rad'
                        ).value),
                        context='physical stop feedback',
                    )
                except ValueError:
                    complete = False
            for token, guard in guards.items():
                if guard['terminal_at'] is None:
                    continue
                if not complete:
                    guard.update(
                        previous=None, previous_at=None,
                        steady_since=None, samples=0
                    )
                    continue
                if received_at <= guard['terminal_at']:
                    continue
                previous_at = guard['previous_at']
                if previous_at is not None and received_at <= previous_at:
                    continue
                if (
                    previous_at is None
                    or received_at - previous_at > maximum_age
                ):
                    guard['steady_since'] = received_at
                    guard['samples'] = 1
                else:
                    speed = max(
                        abs(feedback[name] - guard['previous'][name])
                        / (received_at - previous_at)
                        for name in names
                    )
                    if speed > maximum_speed:
                        guard['steady_since'] = received_at
                        guard['samples'] = 1
                    else:
                        guard['samples'] += 1
                guard['previous'] = feedback
                guard['previous_at'] = received_at
                if (
                    guard['samples'] >= 3
                    and received_at - guard['steady_since'] >= minimum_hold
                ):
                    confirmed.append(token)
            for token in confirmed:
                guards.pop(token, None)
                self._unconfirmed_actions.discard(token)
            if confirmed:
                self._motion_stop_unconfirmed = bool(self._unconfirmed_actions)
        if confirmed:
            self._publish_status('REAL_ARM_STOP_CONFIRMED')

    def _physical_stop_failure_message(self, message, token):
        with self._lock:
            pending = token in getattr(self, '_physical_stop_guards', {})
        if pending:
            return f'{message}; waiting for stationary real joint feedback'
        return message

    def _confirm_action_stopped(self, token):
        with self._lock:
            self._unconfirmed_actions.discard(token)
            self._motion_stop_unconfirmed = bool(self._unconfirmed_actions)

    def _watch_terminal_action_result(self, future, token):
        def terminal_result(done):
            try:
                wrapped = done.result()
            except Exception:
                return
            # ROS action terminal states: SUCCEEDED, CANCELED, ABORTED.
            if wrapped is not None and wrapped.status in (4, 5, 6):
                self._confirm_action_stopped(token)

        future.add_done_callback(terminal_result)

    def _send_action_goal(
        self, client, goal, label, timeout, physical_stop_token=None,
    ):
        if timeout <= 0.0:
            self._release_physical_stop_guard(physical_stop_token)
            return None, f'{label} budget exhausted before goal send'
        try:
            send_future = client.send_goal_async(goal)
        except Exception as error:
            self._release_physical_stop_guard(physical_stop_token)
            return None, f'{label} send failed: {error}'
        handle = self._wait_for_future(send_future, timeout)
        if handle is not None:
            if not handle.accepted:
                self._release_physical_stop_guard(physical_stop_token)
                return None, f'{label} goal was rejected'
            return handle, ''

        # A send timeout does not prove rejection: retain the operation until
        # late acceptance can be canceled and its terminal result confirmed.
        token = self._track_unconfirmed_action()

        def cancel_late_goal(done):
            try:
                late_handle = done.result()
                if late_handle is None:
                    return
                if not late_handle.accepted:
                    self._confirm_action_stopped(token)
                    self._release_physical_stop_guard(physical_stop_token)
                    return
            except Exception:
                return
            try:
                result_future = late_handle.get_result_async()
                self._watch_terminal_action_result(result_future, token)
                self._watch_physical_action_terminal(
                    result_future, physical_stop_token
                )
            except Exception:
                pass
            try:
                late_handle.cancel_goal_async()
            except Exception:
                pass

        send_future.add_done_callback(cancel_late_goal)
        return None, (
            f'{label} goal response timed out; awaiting cancellation/stop '
            'confirmation before further motion'
        )

    def _cancel_action_and_confirm(self, handle, result_future):
        token = self._track_unconfirmed_action()
        self._watch_terminal_action_result(result_future, token)
        with self._lock:
            limit_cancel_requested = (
                getattr(self, '_active_arm_goal_handle', None) is handle
                and getattr(
                    self, '_execution_limit_violation', None
                ) is not None
            )
        if not limit_cancel_requested:
            try:
                handle.cancel_goal_async()
            except Exception:
                pass
        wrapped = self._wait_for_future(
            result_future,
            float(self.get_parameter('cancellation_timeout_seconds').value),
        )
        if wrapped is not None and wrapped.status in (4, 5, 6):
            self._confirm_action_stopped(token)
            return True
        return False

    def _wait_for_action_result(
        self, handle, timeout, label, physical_stop_token=None,
    ):
        try:
            result_future = handle.get_result_async()
        except Exception as error:
            self._track_unconfirmed_action()
            try:
                handle.cancel_goal_async()
            except Exception:
                pass
            return None, (
                f'{label} result unavailable: {error}; '
                'stop unconfirmed; further motion blocked'
            )
        if physical_stop_token is not None:
            with self._lock:
                if getattr(self, '_active_arm_goal_handle', None) is handle:
                    self._active_arm_result_future = result_future
            self._monitor_active_joint_limits()
        self._watch_physical_action_terminal(
            result_future, physical_stop_token
        )
        wrapped = self._wait_for_future(result_future, timeout)
        if wrapped is None:
            stopped = self._cancel_action_and_confirm(handle, result_future)
            return None, self._action_timeout_message(label, stopped)
        if wrapped.status not in (4, 5, 6):
            stopped = self._cancel_action_and_confirm(handle, result_future)
            return None, self._action_timeout_message(label, stopped)
        return wrapped, ''

    @staticmethod
    def _action_timeout_message(label, stopped):
        if stopped:
            return (
                f'{label} timed out; action termination confirmed; '
                'replan required'
            )
        return f'{label} timed out; stop unconfirmed; further motion blocked'

    def _wait_for_real_joint_endpoint(self, trajectory):
        """Wait until real feedback reaches the proxy trajectory endpoint."""
        joint_trajectory = trajectory.joint_trajectory
        if not joint_trajectory.points:
            return False, 'executed trajectory has no joint endpoint'
        names = list(joint_trajectory.joint_names)
        positions = list(joint_trajectory.points[-1].positions)
        if (
            not names
            or len(positions) != len(names)
            or not all(math.isfinite(value) for value in positions)
        ):
            return False, 'executed trajectory joint endpoint is incomplete'
        target = {
            str(name): float(positions[index])
            for index, name in enumerate(names)
        }
        tolerance = float(
            self.get_parameter('execution_joint_tolerance_rad').value
        )
        maximum_age = float(
            self.get_parameter('joint_state_max_age_seconds').value
        )
        required_samples = max(3, int(
            self.get_parameter('execution_stable_samples').value
        ))
        maximum_velocity = float(
            self.get_parameter('execution_stable_velocity_rad_s').value
        )
        deadline = time.monotonic() + float(
            self.get_parameter('execution_settle_timeout_seconds').value
        )
        last_error = math.inf
        last_received_at = time.monotonic()
        previous_feedback = None
        previous_received_at = None
        stable_samples = 0
        missing = names
        while time.monotonic() < deadline:
            now = time.monotonic()
            with self._lock:
                feedback = self._latest_joint_positions.copy()
                received_at = self._latest_joint_received_at
                feedback_age = now - received_at
                violation = getattr(self, '_execution_limit_violation', None)
            if violation:
                return False, violation
            missing = [name for name in names if name not in feedback]
            limits = getattr(self, '_arm_joint_limits', None)
            if not missing and feedback_age <= maximum_age and limits:
                try:
                    normalize_joint_positions(
                        feedback, limits,
                        float(self.get_parameter(
                            'joint_state_boundary_tolerance_rad'
                        ).value),
                        context='executed joint feedback',
                    )
                except ValueError as error:
                    return False, str(error)
            if (
                not missing
                and feedback_age <= maximum_age
                and received_at > last_received_at
                and all(math.isfinite(feedback[name]) for name in names)
            ):
                last_received_at = received_at
                last_error = max(
                    abs(target[name] - feedback[name])
                    for name in names
                )
                velocity = 0.0
                if previous_feedback is not None:
                    interval = received_at - previous_received_at
                    velocity = max(
                        abs(feedback[name] - previous_feedback[name])
                        / interval
                        for name in names
                    )
                previous_feedback = feedback
                previous_received_at = received_at
                if last_error <= tolerance and velocity <= maximum_velocity:
                    stable_samples += 1
                else:
                    stable_samples = 0
                if stable_samples >= required_samples:
                    return (
                        True,
                        'Real joints reached trajectory endpoint: '
                        f'max_error={last_error:.4f}rad '
                        f'stable_samples={stable_samples}',
                    )
            elif missing or feedback_age > maximum_age:
                stable_samples = 0
                previous_feedback = None
                previous_received_at = None
            time.sleep(0.05)
        if missing:
            return (
                False,
                'real joint feedback did not contain trajectory joints: '
                + ', '.join(missing),
            )
        return (
            False,
            'real joints did not settle at trajectory endpoint: '
            f'max_error={last_error:.4f}rad tolerance={tolerance:.4f}rad '
            f'stable_samples={stable_samples}/{required_samples}',
        )

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

    def _publish_status(self, text):
        self._status_publisher.publish(String(data=str(text)))


def main(args=None):
    rclpy.init(args=args)
    node = ButtonApproachPlanner()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
