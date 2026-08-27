#!/usr/bin/env python3
"""Probe the running virtual hardware contract without core application nodes."""

import argparse
import math
import sys
import time

from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, JointState
from trajectory_msgs.msg import JointTrajectoryPoint


ARM_JOINTS = [f'joint{index}' for index in range(1, 7)]
TARGET = [0.10, 0.0, 0.0, 0.0, 0.0, 0.0]
EXPECTED_CONTROLLERS = {
    'joint_state_broadcaster',
    'arm_controller',
    'pika_gripper_controller',
}
FORBIDDEN_NODES = {
    '/move_group',
    '/button_detector',
    '/button_approach_planner',
    '/mock_button_pose',
}


def stamp_seconds(message):
    """Convert a ROS message header stamp to floating seconds."""
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class VirtualHardwareProbe(Node):
    """Collect hardware topics and send one bounded trajectory."""

    def __init__(self):
        super().__init__('virtual_hardware_probe')
        self.robot_joints = None
        self.button_joints = None
        self.color = None
        self.depth = None
        self.camera_info = None
        self.clock_seen = False
        self.create_subscription(
            JointState,
            '/piper_pika/joint_states',
            self._robot_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            JointState,
            '/elevator_button/joint_states',
            self._button_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self._color_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            '/camera/aligned_depth_to_color/image_raw',
            self._depth_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self._camera_info_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Clock,
            '/clock',
            lambda _: setattr(self, 'clock_seen', True),
            10,
        )
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory',
        )
        self.controller_client = self.create_client(
            ListControllers,
            '/controller_manager/list_controllers',
        )

    def _robot_callback(self, message):
        self.robot_joints = message

    def _button_callback(self, message):
        self.button_joints = message

    def _color_callback(self, message):
        self.color = message

    def _depth_callback(self, message):
        self.depth = message

    def _camera_info_callback(self, message):
        self.camera_info = message

    def missing_streams(self):
        """Return names of hardware streams not observed yet."""
        observed = {
            'robot_joint_states': self.robot_joints,
            'button_joint_states': self.button_joints,
            'color': self.color,
            'depth': self.depth,
            'camera_info': self.camera_info,
            'clock': self.clock_seen,
        }
        return [name for name, value in observed.items() if not value]

    def validate_streams(self):
        """Raise with a precise error when a topic violates the contract."""
        for image in (self.color, self.depth):
            if (image.width, image.height) != (848, 480):
                raise RuntimeError(
                    f'Unexpected image size: {image.width}x{image.height}'
                )
            if image.header.frame_id != 'camera_color_optical_frame':
                raise RuntimeError(
                    f'Unexpected camera frame: {image.header.frame_id}'
                )
        if self.depth.encoding != '32FC1':
            raise RuntimeError(
                f'Expected Gazebo 32FC1 depth, got {self.depth.encoding}'
            )
        depth_values = np.frombuffer(self.depth.data, dtype=np.float32)
        finite = depth_values[np.isfinite(depth_values)]
        valid = finite[(finite >= 0.1) & (finite <= 2.0)]
        if not valid.size:
            raise RuntimeError('Depth image has no values in the 0.1-2.0 m range')
        if self.camera_info.header.frame_id != 'camera_color_optical_frame':
            raise RuntimeError(
                'Unexpected CameraInfo frame: '
                f'{self.camera_info.header.frame_id}'
            )
        timestamps = [
            stamp_seconds(self.color),
            stamp_seconds(self.depth),
            stamp_seconds(self.camera_info),
        ]
        if max(timestamps) - min(timestamps) > 0.08:
            raise RuntimeError(f'RGB-D timestamps exceed 80 ms: {timestamps}')
        if 'button_press_joint' not in self.button_joints.name:
            raise RuntimeError('button_press_joint is absent from fixture state')
        button_index = self.button_joints.name.index('button_press_joint')
        button_position = self.button_joints.position[button_index]
        if abs(button_position) > 0.0005:
            raise RuntimeError(
                f'Button does not rest at zero: {button_position:.6f} m'
            )
        return float(np.median(valid)), button_position

    def wait_for_active_controllers(self, timeout):
        """Verify controller state through ROS service without ros2cli daemon."""
        if not self.controller_client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError('Controller manager service is unavailable')
        deadline = time.monotonic() + timeout
        missing = set(EXPECTED_CONTROLLERS)
        while time.monotonic() < deadline:
            future = self.controller_client.call_async(
                ListControllers.Request()
            )
            rclpy.spin_until_future_complete(
                self,
                future,
                timeout_sec=min(2.0, max(0.1, deadline - time.monotonic())),
            )
            if future.done() and future.result() is not None:
                active = {
                    controller.name
                    for controller in future.result().controller
                    if controller.state == 'active'
                }
                missing = EXPECTED_CONTROLLERS - active
                if not missing:
                    return
            rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError(
            'Controllers not active: ' + ', '.join(sorted(missing))
        )

    def validate_node_boundary(self):
        """Reject application-core nodes in the hardware-only launch."""
        names = {
            f'{namespace.rstrip("/")}/{name}'
            if namespace != '/'
            else f'/{name}'
            for name, namespace in self.get_node_names_and_namespaces()
        }
        unexpected = FORBIDDEN_NODES & names
        if unexpected:
            raise RuntimeError(
                'Unexpected core nodes: ' + ', '.join(sorted(unexpected))
            )

    def send_trajectory(self, timeout):
        """Send one safe six-joint position trajectory and return max error."""
        if not self.trajectory_client.wait_for_server(timeout_sec=timeout):
            raise RuntimeError('Arm trajectory action is unavailable')
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ARM_JOINTS
        point = JointTrajectoryPoint()
        point.positions = TARGET
        point.time_from_start.sec = 3
        goal.trajectory.points = [point]
        goal_future = self.trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, goal_future, timeout_sec=timeout)
        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError('Arm trajectory goal was rejected')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self,
            result_future,
            timeout_sec=max(timeout, 8.0),
        )
        wrapped_result = result_future.result()
        if wrapped_result is None:
            raise RuntimeError('Arm trajectory result timed out')
        if wrapped_result.status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(
                f'Arm trajectory failed with status {wrapped_result.status}'
            )
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        positions = dict(
            zip(self.robot_joints.name, self.robot_joints.position)
        )
        errors = [
            abs(positions.get(name, math.inf) - target)
            for name, target in zip(ARM_JOINTS, TARGET)
        ]
        maximum = max(errors)
        if maximum >= 0.03:
            raise RuntimeError(f'Arm final joint error too large: {errors}')
        return maximum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', type=float, default=45.0)
    args = parser.parse_args()
    rclpy.init()
    node = VirtualHardwareProbe()
    try:
        node.wait_for_active_controllers(args.timeout)
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline and node.missing_streams():
            rclpy.spin_once(node, timeout_sec=0.1)
        missing = node.missing_streams()
        if missing:
            raise RuntimeError('Missing hardware streams: ' + ', '.join(missing))
        depth_m, button_position = node.validate_streams()
        node.validate_node_boundary()
        maximum_error = node.send_trajectory(args.timeout)
        print(
            'OK: virtual hardware '
            f'depth={depth_m:.3f}m '
            f'button_rest={button_position:.6f}m '
            f'max_joint_error={maximum_error:.5f}rad'
        )
        return 0
    except RuntimeError as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
