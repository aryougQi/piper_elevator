#!/usr/bin/env python3

from sensor_msgs.msg import JointState

import rclpy
from rclpy.node import Node


class PiperPikaJointStateMux(Node):
    """Merge Piper feedback and the Pika opening into one MoveIt state."""

    def __init__(self):
        super().__init__('piper_pika_joint_state_mux')
        self.declare_parameter('arm_topic', '/feedback/joint_states')
        self.declare_parameter('gripper_topic', '/gripper/joint_state')
        self.declare_parameter('output_topic', '/piper_pika/joint_states')
        self.declare_parameter('gripper_joint_name', 'center_joint')
        self.declare_parameter('default_gripper_position', 0.0)

        self._gripper_joint_name = str(
            self.get_parameter('gripper_joint_name').value
        )
        self._gripper_position = float(
            self.get_parameter('default_gripper_position').value
        )
        self._gripper_velocity = 0.0
        self._gripper_effort = 0.0

        self._publisher = self.create_publisher(
            JointState,
            str(self.get_parameter('output_topic').value),
            10,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter('arm_topic').value),
            self._arm_callback,
            10,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter('gripper_topic').value),
            self._gripper_callback,
            10,
        )

    @staticmethod
    def _sample(values, index, default=0.0):
        return float(values[index]) if index < len(values) else default

    def _gripper_callback(self, message):
        if not message.position:
            return
        try:
            index = list(message.name).index(self._gripper_joint_name)
        except ValueError:
            if len(message.position) != 1:
                self.get_logger().warn(
                    'Ignoring Pika state without center_joint'
                )
                return
            index = 0
        self._gripper_position = self._sample(message.position, index)
        self._gripper_velocity = self._sample(message.velocity, index)
        self._gripper_effort = self._sample(message.effort, index)

    def _arm_callback(self, message):
        output = JointState()
        output.header = message.header

        for index, name in enumerate(message.name):
            if name == self._gripper_joint_name:
                continue
            output.name.append(name)
            output.position.append(self._sample(message.position, index))
            output.velocity.append(self._sample(message.velocity, index))
            output.effort.append(self._sample(message.effort, index))

        output.name.append(self._gripper_joint_name)
        output.position.append(self._gripper_position)
        output.velocity.append(self._gripper_velocity)
        output.effort.append(self._gripper_effort)
        self._publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = PiperPikaJointStateMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
