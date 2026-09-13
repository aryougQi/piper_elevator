"""Bounded simulation-only adapter for MoveIt Servo position commands."""

from copy import deepcopy
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

from piper_elevator_app.servo_adapter_core import compensated_positions


class SimulationServoAdapter(Node):
    """Compensate Gazebo position-interface lag without changing real Servo."""

    def __init__(self):
        super().__init__('simulation_servo_adapter')
        self.declare_parameter(
            'input_topic', '/servo_node/raw_joint_trajectory'
        )
        self.declare_parameter(
            'output_topic', '/arm_controller/joint_trajectory'
        )
        self.declare_parameter(
            'joint_state_topic', '/piper_pika/joint_states'
        )
        self.declare_parameter(
            'joint_names',
            ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
        )
        self.declare_parameter(
            'lower_limits_rad',
            [-2.6179938, 0.0, -2.9670597, -1.7453292, -1.2217304,
             -2.0943951],
        )
        self.declare_parameter(
            'upper_limits_rad',
            [2.6179938, 3.1415926, 0.0, 1.7453292, 1.2217304,
             2.0943951],
        )
        self.declare_parameter('position_gain', 2.0)
        self.declare_parameter('maximum_lead_rad', 0.08)
        self.declare_parameter('joint_state_timeout_seconds', 0.20)

        self._joint_names = list(
            self.get_parameter('joint_names').value
        )
        lower = list(self.get_parameter('lower_limits_rad').value)
        upper = list(self.get_parameter('upper_limits_rad').value)
        if not self._joint_names or not (
            len(self._joint_names) == len(lower) == len(upper)
        ):
            raise ValueError('joint names and limit arrays must match')
        self._lower_by_name = dict(zip(self._joint_names, lower))
        self._upper_by_name = dict(zip(self._joint_names, upper))
        self._position_gain = float(
            self.get_parameter('position_gain').value
        )
        self._maximum_lead = float(
            self.get_parameter('maximum_lead_rad').value
        )
        self._state_timeout = float(
            self.get_parameter('joint_state_timeout_seconds').value
        )
        if self._state_timeout <= 0.0:
            raise ValueError('joint_state_timeout_seconds must be positive')

        self._current_by_name = {}
        self._state_received_at = None
        self._last_warning_at = 0.0
        self._publisher = self.create_publisher(
            JointTrajectory,
            str(self.get_parameter('output_topic').value),
            10,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter('joint_state_topic').value),
            self._joint_state_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            JointTrajectory,
            str(self.get_parameter('input_topic').value),
            self._trajectory_callback,
            10,
        )
        self.get_logger().info(
            'Simulation Servo tracking compensation enabled: '
            f'gain={self._position_gain:.2f}, '
            f'max_lead={self._maximum_lead:.3f} rad'
        )

    def _joint_state_callback(self, message):
        if len(message.name) != len(message.position):
            self._warn_throttled('Ignoring malformed joint state')
            return
        positions = dict(zip(message.name, message.position))
        if not all(name in positions for name in self._joint_names):
            return
        self._current_by_name = {
            name: float(positions[name]) for name in self._joint_names
        }
        self._state_received_at = time.monotonic()

    def _trajectory_callback(self, message):
        now = time.monotonic()
        if (
            self._state_received_at is None
            or now - self._state_received_at > self._state_timeout
        ):
            self._warn_throttled(
                'Dropping Servo command because joint feedback is stale'
            )
            return
        if not message.points:
            self._warn_throttled('Ignoring Servo trajectory without points')
            return

        output = deepcopy(message)
        try:
            for point in output.points:
                point.positions = compensated_positions(
                    output.joint_names,
                    point.positions,
                    self._current_by_name,
                    self._position_gain,
                    self._maximum_lead,
                    self._lower_by_name,
                    self._upper_by_name,
                )
        except (TypeError, ValueError) as error:
            self._warn_throttled(f'Ignoring unsafe Servo command: {error}')
            return
        self._publisher.publish(output)

    def _warn_throttled(self, text):
        now = time.monotonic()
        if now - self._last_warning_at >= 2.0:
            self._last_warning_at = now
            self.get_logger().warning(text)


def main(args=None):
    rclpy.init(args=args)
    node = SimulationServoAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
