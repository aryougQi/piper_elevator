import copy
import threading
import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.msg import OrientationConstraint
from moveit_msgs.msg import PositionConstraint
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
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer
from tf2_ros import ConnectivityException
from tf2_ros import ExtrapolationException
from tf2_ros import LookupException
from tf2_ros import TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

from piper_elevator_app.motion_core import position_in_workspace
from piper_elevator_app.motion_core import transform_button_to_approach


class ButtonApproachPlanner(Node):
    """Transform a selected button and plan a safe MoveIt approach."""

    def __init__(self):
        super().__init__('button_approach_planner')
        self._declare_parameters()
        self._callback_group = ReentrantCallbackGroup()
        self._lock = threading.Lock()
        self._busy = False
        self._latest_button = None
        self._latest_approach = None
        self._latest_received_at = 0.0
        self._planned_trajectory = None
        self._planned_button = None
        self._plan_created_at = 0.0
        self._auto_started = False

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
        self.create_subscription(
            PoseStamped,
            self._string_parameter('button_pose_topic'),
            self._button_pose_callback,
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
            '~/clear_plan',
            self._clear_plan_callback,
            callback_group=self._callback_group,
        )

        self._publish_status('WAITING_FOR_BUTTON')
        approach_distance = self.get_parameter(
            'approach_distance_m'
        ).value
        self.get_logger().info(
            'Button approach planner ready: '
            f'base={self._base_frame}, tip={self._end_effector_link}, '
            f'approach={approach_distance:.3f} m, '
            f'execution={self.get_parameter("allow_execution").value}'
        )

    def _declare_parameters(self):
        self.declare_parameter('button_pose_topic', '/button_pose')
        self.declare_parameter('button_base_topic', '/button_pose_base')
        self.declare_parameter(
            'approach_pose_topic',
            '/button_approach_pose',
        )
        self.declare_parameter('status_topic', '/button_approach/status')
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
        self.declare_parameter('approach_distance_m', 0.08)
        self.declare_parameter('position_tolerance_m', 0.008)
        self.declare_parameter('orientation_tolerance_rad', 0.02)
        self.declare_parameter('planning_time_seconds', 5.0)
        self.declare_parameter('planning_attempts', 5)
        self.declare_parameter('velocity_scaling', 0.10)
        self.declare_parameter('acceleration_scaling', 0.10)
        self.declare_parameter('tf_timeout_seconds', 0.25)
        self.declare_parameter('action_timeout_seconds', 30.0)
        self.declare_parameter('target_max_age_seconds', 1.0)
        self.declare_parameter('plan_max_age_seconds', 120.0)
        self.declare_parameter('max_target_drift_m', 0.03)
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

    def _string_parameter(self, name):
        return str(self.get_parameter(name).value)

    def _vector_parameter(self, name):
        values = np.asarray(self.get_parameter(name).value, dtype=np.float64)
        if values.shape != (3,):
            raise ValueError(f'{name} must contain exactly three values')
        return values

    def _button_pose_callback(self, message):
        if not message.header.frame_id:
            self._publish_status('REJECTED: button pose has no frame_id')
            return
        button_camera = np.array([
            message.pose.position.x,
            message.pose.position.y,
            message.pose.position.z,
        ])
        if not np.all(np.isfinite(button_camera)) or button_camera[2] <= 0.0:
            self._publish_status('REJECTED: invalid camera button position')
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
                f'Cannot transform {message.header.frame_id} to '
                f'{self._base_frame}: {error}',
                throttle_duration_sec=2.0,
            )
            self._publish_status('WAITING_FOR_CAMERA_TO_BASE_TF')
            return

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        try:
            button_base, approach, _, _ = (
                transform_button_to_approach(
                    button_camera,
                    [translation.x, translation.y, translation.z],
                    [rotation.x, rotation.y, rotation.z, rotation.w],
                    float(
                        self.get_parameter('approach_distance_m').value
                    ),
                )
            )
        except ValueError as error:
            self._publish_status(f'REJECTED: {error}')
            return

        # A coarse approach only translates the closed fingertip centre.  The
        # camera image axes must not define tool roll: doing so can make joint6
        # spin even though the tool's forward axis already faces the panel.
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
        except (
            LookupException,
            ConnectivityException,
            ExtrapolationException,
        ) as error:
            self.get_logger().warning(
                f'Cannot read current {self._end_effector_link} pose: '
                f'{error}',
                throttle_duration_sec=2.0,
            )
            self._publish_status('WAITING_FOR_FINGERTIP_TF')
            return
        tool_rotation = tool_transform.transform.rotation
        orientation = np.array([
            tool_rotation.x,
            tool_rotation.y,
            tool_rotation.z,
            tool_rotation.w,
        ])

        if not position_in_workspace(
            approach,
            self._workspace_min,
            self._workspace_max,
        ):
            self._publish_status(
                'REJECTED: approach pose is outside configured workspace'
            )
            return

        header_stamp = self.get_clock().now().to_msg()
        button_pose = self._make_pose(
            button_base,
            orientation,
            header_stamp,
        )
        approach_pose = self._make_pose(
            approach,
            orientation,
            header_stamp,
        )
        self._button_base_publisher.publish(button_pose)
        self._approach_publisher.publish(approach_pose)

        with self._lock:
            if (
                self._planned_button is not None
                and np.linalg.norm(button_base - self._planned_button)
                > float(self.get_parameter('max_target_drift_m').value)
            ):
                self._clear_stored_plan_locked()
                self._publish_status('PLAN_INVALIDATED: button moved')
            self._latest_button = button_pose
            self._latest_approach = approach_pose
            self._latest_received_at = time.monotonic()

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
            if self._busy:
                response.success = False
                response.message = 'Planner is busy'
                return response
            age = time.monotonic() - self._latest_received_at
            if (
                self._latest_approach is None
                or age > float(
                    self.get_parameter('target_max_age_seconds').value
                )
            ):
                response.success = False
                response.message = 'No fresh transformed button target'
                return response
            target = copy.deepcopy(self._latest_approach)
            button = copy.deepcopy(self._latest_button)
            self._busy = True

        self._publish_status('PLANNING')
        try:
            if (
                bool(self.get_parameter('simulation_mode').value)
                and bool(
                    self.get_parameter('close_gripper_before_plan').value
                )
            ):
                self._publish_status('CLOSING_GRIPPER')
                closed, close_message = self._close_gripper()
                if not closed:
                    response.success = False
                    response.message = close_message
                    self._publish_status(
                        f'PLAN_FAILED: {close_message}'
                    )
                    return response
            result, message = self._plan_pose(target)
            if result is None:
                response.success = False
                response.message = message
                self._publish_status(f'PLAN_FAILED: {message}')
                return response
            with self._lock:
                self._planned_trajectory = result.planned_trajectory
                self._planned_button = np.array([
                    button.pose.position.x,
                    button.pose.position.y,
                    button.pose.position.z,
                ])
                self._plan_created_at = time.monotonic()
            response.success = True
            response.message = (
                f'Plan ready in {result.planning_time:.3f} s; '
                'call ~/execute to move the simulated arm'
            )
            self._publish_status('PLAN_READY')
            return response
        finally:
            with self._lock:
                self._busy = False

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
            if self._busy:
                response.success = False
                response.message = 'Planner is busy'
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
            self._busy = True

        self._publish_status('EXECUTING')
        try:
            success, message = self._execute_trajectory(trajectory)
            response.success = success
            response.message = message
            if success:
                self._publish_status('APPROACH_REACHED')
                with self._lock:
                    self._clear_stored_plan_locked()
            else:
                self._publish_status(f'EXECUTION_FAILED: {message}')
            return response
        finally:
            with self._lock:
                self._busy = False

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

    def _clear_stored_plan_locked(self):
        self._planned_trajectory = None
        self._planned_button = None
        self._plan_created_at = 0.0

    def _auto_plan_execute_callback(self):
        with self._lock:
            target_is_fresh = (
                self._latest_approach is not None
                and time.monotonic() - self._latest_received_at
                <= float(
                    self.get_parameter('target_max_age_seconds').value
                )
            )
            if self._auto_started or not target_is_fresh:
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

    def _plan_pose(self, target):
        timeout = float(self.get_parameter('action_timeout_seconds').value)
        if not self._move_group_client.wait_for_server(timeout_sec=timeout):
            return None, 'MoveGroup action server is unavailable'

        goal = MoveGroup.Goal()
        goal.request.group_name = self._string_parameter('planning_group')
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

        send_future = self._move_group_client.send_goal_async(goal)
        goal_handle = self._wait_for_future(send_future, timeout)
        if goal_handle is None:
            return None, 'Timed out while sending planning goal'
        if not goal_handle.accepted:
            return None, 'MoveGroup rejected planning goal'

        wrapped = self._wait_for_future(
            goal_handle.get_result_async(),
            timeout + float(
                self.get_parameter('planning_time_seconds').value
            ),
        )
        if wrapped is None:
            return None, 'MoveGroup planning timed out'
        result = wrapped.result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            return None, f'MoveIt planning error {result.error_code.val}'
        if not result.planned_trajectory.joint_trajectory.points:
            return None, 'MoveIt returned an empty trajectory'
        return result, ''

    def _pose_constraints(self, target):
        constraints = Constraints()
        constraints.name = 'button_approach_pose'

        position = PositionConstraint()
        position.header = target.header
        position.link_name = self._end_effector_link
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [
            float(self.get_parameter('position_tolerance_m').value)
        ]
        position.constraint_region.primitives = [sphere]
        position.constraint_region.primitive_poses = [target.pose]
        position.weight = 1.0

        orientation = self._orientation_constraint(target)

        constraints.position_constraints = [position]
        constraints.orientation_constraints = [orientation]
        return constraints

    def _orientation_constraint(self, target):
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
        return orientation

    def _close_gripper(self):
        timeout = float(self.get_parameter('action_timeout_seconds').value)
        if not self._gripper_client.wait_for_server(timeout_sec=timeout):
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

        send_future = self._gripper_client.send_goal_async(goal)
        goal_handle = self._wait_for_future(send_future, timeout)
        if goal_handle is None:
            return False, 'Timed out while sending Pika close command'
        if not goal_handle.accepted:
            return False, 'Pika controller rejected close command'
        wrapped = self._wait_for_future(
            goal_handle.get_result_async(),
            timeout,
        )
        if wrapped is None:
            return False, 'Pika close command timed out'
        if (
            wrapped.result.error_code
            != FollowJointTrajectory.Result.SUCCESSFUL
        ):
            return (
                False,
                'Pika close command failed with error '
                f'{wrapped.result.error_code}',
            )
        return True, 'Pika gripper closed'

    def _execute_trajectory(self, trajectory):
        timeout = float(self.get_parameter('action_timeout_seconds').value)
        if not self._execute_client.wait_for_server(timeout_sec=timeout):
            return False, 'ExecuteTrajectory action server is unavailable'
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        send_future = self._execute_client.send_goal_async(goal)
        goal_handle = self._wait_for_future(send_future, timeout)
        if goal_handle is None:
            return False, 'Timed out while sending trajectory'
        if not goal_handle.accepted:
            return False, 'MoveIt rejected trajectory execution'
        wrapped = self._wait_for_future(
            goal_handle.get_result_async(),
            timeout,
        )
        if wrapped is None:
            return False, 'Trajectory execution timed out'
        if wrapped.result.error_code.val != MoveItErrorCodes.SUCCESS:
            return (
                False,
                f'MoveIt execution error {wrapped.result.error_code.val}',
            )
        return True, 'Button approach pose reached'

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
