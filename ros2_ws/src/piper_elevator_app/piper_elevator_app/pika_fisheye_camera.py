"""Low-latency ROS 2 camera node for the Pika UVC fisheye camera."""

from threading import Event, Lock, Thread
from time import monotonic
from typing import Optional

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster


class PikaFisheyeCamera(Node):
    """Publish the newest UVC frame without software double-throttling."""

    def __init__(self) -> None:
        super().__init__('camera_fisheye')
        self.declare_parameter('camera_port', 6)
        self.declare_parameter('camera_fps', 30)
        self.declare_parameter('camera_height', 480)
        self.declare_parameter('camera_width', 640)
        self.declare_parameter('camera_frame_id', 'camera_rgb')

        self._camera_port = int(self.get_parameter('camera_port').value)
        self._camera_fps = max(
            1,
            int(self.get_parameter('camera_fps').value),
        )
        self._camera_height = max(
            1,
            int(self.get_parameter('camera_height').value),
        )
        self._camera_width = max(
            1,
            int(self.get_parameter('camera_width').value),
        )
        self._camera_frame_id = str(
            self.get_parameter('camera_frame_id').value
        ).lstrip('/')
        self._color_frame_id = f'{self._camera_frame_id}_color'
        self._bridge = CvBridge()
        self._failed_reads = 0
        self._stop_event = Event()
        self._frame_lock = Lock()
        self._latest_frame = None
        self._latest_stamp = None
        self._latest_sequence = 0
        self._published_sequence = 0
        self._capture_window_started = monotonic()
        self._capture_frame_count = 0
        self._capture_max_gap_seconds = 0.0
        self._last_capture_time: Optional[float] = None
        self._publish_window_started = monotonic()
        self._publish_frame_count = 0
        self._publish_max_gap_seconds = 0.0
        self._publish_total_seconds = 0.0
        self._publish_max_seconds = 0.0
        self._last_publish_time: Optional[float] = None

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            # Large raw images are fragmented by DDS. Reliable delivery
            # avoids losing a whole frame when any one fragment is dropped;
            # depth=1 still prevents a stale-image backlog.
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._image_publisher = self.create_publisher(
            Image,
            '/camera_fisheye/color/image_raw',
            qos,
        )
        self._camera_info_publisher = self.create_publisher(
            CameraInfo,
            '/camera_fisheye/color/camera_info',
            qos,
        )

        self._capture = cv2.VideoCapture(
            self._camera_port,
            cv2.CAP_V4L2,
        )
        self._configure_capture()
        self._publish_static_transform()
        self._capture_thread = Thread(
            target=self._capture_loop,
            name='pika-fisheye-capture',
            daemon=True,
        )
        self._capture_thread.start()
        self._publish_timer = self.create_timer(
            1.0 / float(self._camera_fps),
            self._publish_latest_frame,
        )
        actual_width = int(
            self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        actual_height = int(
            self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        actual_fps = self._capture.get(cv2.CAP_PROP_FPS)
        self.get_logger().info(
            'Pika fisheye camera ready: '
            f'/dev/video{self._camera_port}, '
            f'{actual_width}x{actual_height}, '
            f'requested_fps={self._camera_fps}, '
            f'device_fps={actual_fps:.1f}'
        )

    def _configure_capture(self) -> None:
        if not self._capture.isOpened():
            raise RuntimeError(
                f'Unable to open /dev/video{self._camera_port}'
            )
        self._capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*'MJPG'),
        )
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._camera_width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._camera_height)
        self._capture.set(cv2.CAP_PROP_FPS, self._camera_fps)

    def _publish_static_transform(self) -> None:
        broadcaster = StaticTransformBroadcaster(self)
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self._camera_frame_id
        transform.child_frame_id = self._color_frame_id
        transform.transform.rotation.w = 1.0
        broadcaster.sendTransform(transform)
        self._static_transform_broadcaster = broadcaster

    def _capture_loop(self) -> None:
        """Read at the hardware cadence without an additional ROS sleep."""
        while not self._stop_event.is_set() and rclpy.ok():
            success, frame = self._capture.read()
            if not success:
                self._failed_reads += 1
                if self._failed_reads == 1 or self._failed_reads % 30 == 0:
                    self.get_logger().warning(
                        'Failed to read Pika fisheye frame '
                        f'({self._failed_reads} consecutive failures).'
                    )
                continue
            self._failed_reads = 0

            captured_at = monotonic()
            with self._frame_lock:
                self._latest_frame = frame
                self._latest_stamp = self.get_clock().now().to_msg()
                self._latest_sequence += 1
            self._record_capture(captured_at)

    def _record_capture(self, captured_at: float) -> None:
        if self._last_capture_time is not None:
            self._capture_max_gap_seconds = max(
                self._capture_max_gap_seconds,
                captured_at - self._last_capture_time,
            )
        self._last_capture_time = captured_at
        self._capture_frame_count += 1
        elapsed = captured_at - self._capture_window_started
        if elapsed < 5.0:
            return
        fps = self._capture_frame_count / max(elapsed, 1e-6)
        self.get_logger().info(
            'Capture performance: '
            f'{fps:.1f} FPS, '
            f'max_gap={self._capture_max_gap_seconds * 1000.0:.1f} ms'
        )
        self._capture_window_started = captured_at
        self._capture_frame_count = 0
        self._capture_max_gap_seconds = 0.0

    def _publish_latest_frame(self) -> None:
        publish_started = monotonic()
        with self._frame_lock:
            if self._latest_sequence == self._published_sequence:
                return
            frame = self._latest_frame
            stamp = self._latest_stamp
            self._published_sequence = self._latest_sequence
        if frame is None or stamp is None:
            return

        image = self._bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        image.header.stamp = stamp
        image.header.frame_id = self._color_frame_id
        self._image_publisher.publish(image)

        camera_info = CameraInfo()
        camera_info.header = image.header
        camera_info.width = int(frame.shape[1])
        camera_info.height = int(frame.shape[0])
        self._camera_info_publisher.publish(camera_info)
        published_at = monotonic()
        self._record_publish(
            published_at,
            published_at - publish_started,
        )

    def _record_publish(
        self,
        published_at: float,
        publish_seconds: float,
    ) -> None:
        """Report ROS publication cadence without adding a subscriber."""
        if self._last_publish_time is not None:
            self._publish_max_gap_seconds = max(
                self._publish_max_gap_seconds,
                published_at - self._last_publish_time,
            )
        self._last_publish_time = published_at
        self._publish_frame_count += 1
        self._publish_total_seconds += publish_seconds
        self._publish_max_seconds = max(
            self._publish_max_seconds,
            publish_seconds,
        )
        elapsed = published_at - self._publish_window_started
        if elapsed < 5.0:
            return
        fps = self._publish_frame_count / max(elapsed, 1e-6)
        average_ms = (
            self._publish_total_seconds
            / max(self._publish_frame_count, 1)
            * 1000.0
        )
        self.get_logger().info(
            'Publish performance: '
            f'{fps:.1f} FPS, '
            f'average={average_ms:.1f} ms, '
            f'max={self._publish_max_seconds * 1000.0:.1f} ms, '
            f'max_gap={self._publish_max_gap_seconds * 1000.0:.1f} ms'
        )
        self._publish_window_started = published_at
        self._publish_frame_count = 0
        self._publish_total_seconds = 0.0
        self._publish_max_seconds = 0.0
        self._publish_max_gap_seconds = 0.0

    def destroy_node(self):
        """Release the V4L2 device before destroying the ROS node."""
        self._stop_event.set()
        self._capture_thread.join(timeout=1.0)
        if self._capture.isOpened():
            self._capture.release()
        return super().destroy_node()


def main(args=None) -> None:
    """Run the low-latency Pika fisheye camera node."""
    rclpy.init(args=args)
    node = None
    try:
        node = PikaFisheyeCamera()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except (KeyboardInterrupt, RuntimeError):
                pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except RuntimeError:
                pass


if __name__ == '__main__':
    main()
