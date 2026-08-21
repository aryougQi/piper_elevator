import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer
from tf2_ros import ConnectivityException
from tf2_ros import ExtrapolationException
from tf2_ros import LookupException
from tf2_ros import TransformListener

from piper_elevator_app.motion_core import quaternion_to_matrix


class MockButtonPose(Node):
    """Publish a fixed camera-frame button for MoveIt simulation."""

    def __init__(self):
        super().__init__('mock_button_pose')
        self.declare_parameter('topic', '/button_pose')
        self.declare_parameter('frame_id', 'camera_color_optical_frame')
        self.declare_parameter('fixed_frame_id', '')
        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('z', 0.45)
        self.declare_parameter('publish_rate_hz', 5.0)
        self._output_frame = str(self.get_parameter('frame_id').value)
        self._fixed_frame = str(
            self.get_parameter('fixed_frame_id').value
        )
        self._tf_buffer = None
        self._tf_listener = None
        if self._fixed_frame:
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
        self._publisher = self.create_publisher(
            PoseStamped,
            str(self.get_parameter('topic').value),
            10,
        )
        rate = max(0.2, float(self.get_parameter('publish_rate_hz').value))
        self.create_timer(1.0 / rate, self._publish)
        self.get_logger().info(
            'Publishing simulated button '
            f'({self.get_parameter("x").value:.3f}, '
            f'{self.get_parameter("y").value:.3f}, '
            f'{self.get_parameter("z").value:.3f}) m in '
            f'{self._fixed_frame or self._output_frame}'
        )

    def _publish(self):
        position = np.array([
            float(self.get_parameter('x').value),
            float(self.get_parameter('y').value),
            float(self.get_parameter('z').value),
        ])
        if self._fixed_frame:
            try:
                transform = self._tf_buffer.lookup_transform(
                    self._output_frame,
                    self._fixed_frame,
                    Time(),
                    timeout=Duration(seconds=0.10),
                )
            except (
                LookupException,
                ConnectivityException,
                ExtrapolationException,
            ) as error:
                self.get_logger().warning(
                    f'Waiting for {self._output_frame} <- '
                    f'{self._fixed_frame}: {error}',
                    throttle_duration_sec=2.0,
                )
                return
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            matrix = quaternion_to_matrix([
                rotation.x,
                rotation.y,
                rotation.z,
                rotation.w,
            ])
            position = np.array([
                translation.x,
                translation.y,
                translation.z,
            ]) + matrix @ position

        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._output_frame
        message.pose.position.x = float(position[0])
        message.pose.position.y = float(position[1])
        message.pose.position.z = float(position[2])
        message.pose.orientation.w = 1.0
        self._publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = MockButtonPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
