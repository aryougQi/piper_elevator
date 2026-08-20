"""Run the YOLO ONNX elevator-button ROS 2 node."""

from pathlib import Path
from typing import List, Optional, Tuple

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, PoseStamped
import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float32
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

from piper_elevator_app.detector_core import Detection
from piper_elevator_app.detector_core import project_pixel
from piper_elevator_app.detector_core import robust_box_depth
from piper_elevator_app.detector_core import TemporalButtonTracker
from piper_elevator_app.detector_core import YoloOnnxDetector


class ButtonDetector(Node):
    """Publish stable 2D and 3D results for one elevator button."""

    def __init__(self) -> None:
        super().__init__('button_detector')
        self._declare_parameters()

        self._bridge = CvBridge()
        self._camera_matrix: Optional[np.ndarray] = None
        self._camera_frame = ''
        self._filtered_position: Optional[np.ndarray] = None
        self._use_depth = bool(self.get_parameter('use_depth').value)

        model_path = self._resolve_model_path(
            self._string_parameter('model_path')
        )
        self._model = YoloOnnxDetector(
            model_path=str(model_path),
            class_names=self._string_list_parameter('class_names'),
            target_classes=self._string_list_parameter('target_classes'),
            confidence_threshold=float(
                self.get_parameter('confidence_threshold').value
            ),
            nms_iou_threshold=float(
                self.get_parameter('nms_iou_threshold').value
            ),
            input_size=int(self.get_parameter('model_input_size').value),
            inference_device=self._string_parameter('inference_device'),
        )
        self._tracker = TemporalButtonTracker(
            required_stable_frames=int(
                self.get_parameter('required_stable_frames').value
            ),
            max_missed_frames=int(
                self.get_parameter('max_missed_frames').value
            ),
            minimum_iou=float(
                self.get_parameter('tracking_minimum_iou').value
            ),
            max_center_jump_ratio=float(
                self.get_parameter('max_center_jump_ratio').value
            ),
            smoothing_alpha=float(
                self.get_parameter('position_smoothing_alpha').value
            ),
        )

        self._create_publishers()
        self._create_subscriptions()
        mode = 'RGB-D' if self._use_depth else 'RGB'
        self.get_logger().info(
            f'YOLO button detector ready in {mode} mode. '
            f'model={model_path}, '
            f'providers={self._model.active_providers}, '
            f'color={self._string_parameter("color_topic")}'
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter('use_depth', False)
        self.declare_parameter(
            'color_topic',
            '/camera_fisheye/color/image_raw',
        )
        self.declare_parameter(
            'depth_topic',
            '/camera/aligned_depth_to_color/image_raw',
        )
        self.declare_parameter(
            'camera_info_topic',
            '/camera_fisheye/color/camera_info',
        )
        self.declare_parameter('button_pixel_topic', '/button_pixel')
        self.declare_parameter('button_pose_topic', '/button_pose')
        self.declare_parameter(
            'button_detections_topic',
            '/button_detections',
        )
        self.declare_parameter(
            'button_valid_topic',
            '/button_detection_valid',
        )
        self.declare_parameter(
            'button_confidence_topic',
            '/button_detection_confidence',
        )
        self.declare_parameter(
            'debug_image_topic',
            '/button_detector/debug_image',
        )

        self.declare_parameter(
            'model_path',
            'models/elevator_buttons_yolov10s.onnx',
        )
        self.declare_parameter(
            'class_names',
            ['__model_metadata__'],
        )
        self.declare_parameter(
            'target_classes',
            ['*'],
        )
        self.declare_parameter('model_input_size', 1280)
        self.declare_parameter('inference_device', 'cuda')
        self.declare_parameter('confidence_threshold', 0.60)
        self.declare_parameter('nms_iou_threshold', 0.45)

        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('sync_slop_seconds', 0.08)
        self.declare_parameter('roi_x', 0.0)
        self.declare_parameter('roi_y', 0.0)
        self.declare_parameter('roi_width', 1.0)
        self.declare_parameter('roi_height', 1.0)

        self.declare_parameter('depth_unit_scale', 0.001)
        self.declare_parameter('depth_inner_ratio', 0.45)
        self.declare_parameter('minimum_depth_samples', 20)
        self.declare_parameter('min_depth_m', 0.10)
        self.declare_parameter('max_depth_m', 2.00)

        self.declare_parameter('required_stable_frames', 5)
        self.declare_parameter('max_missed_frames', 2)
        self.declare_parameter('tracking_minimum_iou', 0.15)
        self.declare_parameter('max_center_jump_ratio', 0.10)
        self.declare_parameter('position_smoothing_alpha', 0.35)

    def _create_publishers(self) -> None:
        self._pose_publisher = self.create_publisher(
            PoseStamped,
            self._string_parameter('button_pose_topic'),
            10,
        )
        self._pixel_publisher = self.create_publisher(
            PointStamped,
            self._string_parameter('button_pixel_topic'),
            10,
        )
        self._detections_publisher = self.create_publisher(
            Detection2DArray,
            self._string_parameter('button_detections_topic'),
            10,
        )
        self._valid_publisher = self.create_publisher(
            Bool,
            self._string_parameter('button_valid_topic'),
            10,
        )
        self._confidence_publisher = self.create_publisher(
            Float32,
            self._string_parameter('button_confidence_topic'),
            10,
        )
        self._debug_publisher = self.create_publisher(
            Image,
            self._string_parameter('debug_image_topic'),
            qos_profile_sensor_data,
        )

    def _create_subscriptions(self) -> None:
        color_topic = self._string_parameter('color_topic')
        self._camera_info_subscription = None
        self._color_subscription = None
        self._depth_subscription = None
        self._synchronizer = None
        self._color_only_subscription = None
        if self._use_depth:
            self._camera_info_subscription = self.create_subscription(
                CameraInfo,
                self._string_parameter('camera_info_topic'),
                self._camera_info_callback,
                qos_profile_sensor_data,
            )
            self._color_subscription = message_filters.Subscriber(
                self,
                Image,
                color_topic,
                qos_profile=qos_profile_sensor_data,
            )
            self._depth_subscription = message_filters.Subscriber(
                self,
                Image,
                self._string_parameter('depth_topic'),
                qos_profile=qos_profile_sensor_data,
            )
            self._synchronizer = message_filters.ApproximateTimeSynchronizer(
                [self._color_subscription, self._depth_subscription],
                queue_size=int(
                    self.get_parameter('sync_queue_size').value
                ),
                slop=float(
                    self.get_parameter('sync_slop_seconds').value
                ),
            )
            self._synchronizer.registerCallback(self._rgbd_callback)
        else:
            self._color_only_subscription = self.create_subscription(
                Image,
                color_topic,
                self._color_callback,
                qos_profile_sensor_data,
            )

    def _resolve_model_path(self, configured_path: str) -> Path:
        path = Path(configured_path).expanduser()
        if not path.is_absolute():
            package_share = Path(
                get_package_share_directory('piper_elevator_app')
            )
            path = package_share / path
        if not path.is_file():
            raise FileNotFoundError(
                'Button model is missing. Expected ONNX file at: '
                f'{path}'
            )
        return path

    def _string_parameter(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def _string_list_parameter(self, name: str) -> List[str]:
        return [str(value) for value in self.get_parameter(name).value]

    def _camera_info_callback(self, message: CameraInfo) -> None:
        matrix = np.asarray(message.k, dtype=np.float64).reshape(3, 3)
        if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
            self.get_logger().warning('Ignoring invalid camera intrinsics.')
            return
        self._camera_matrix = matrix
        self._camera_frame = message.header.frame_id

    def _color_callback(self, color_message: Image) -> None:
        try:
            color_image = self._bridge.imgmsg_to_cv2(
                color_message,
                desired_encoding='bgr8',
            )
        except CvBridgeError as error:
            self.get_logger().error(f'Color conversion failed: {error}')
            self._publish_state(None, False)
            return
        self._process_frame(color_message, color_image, None)

    def _rgbd_callback(
        self,
        color_message: Image,
        depth_message: Image,
    ) -> None:
        try:
            color_image = self._bridge.imgmsg_to_cv2(
                color_message,
                desired_encoding='bgr8',
            )
            depth_image = self._bridge.imgmsg_to_cv2(
                depth_message,
                desired_encoding='passthrough',
            )
        except CvBridgeError as error:
            self.get_logger().error(f'Image conversion failed: {error}')
            self._publish_state(None, False)
            return
        self._process_frame(color_message, color_image, depth_image)

    def _process_frame(
        self,
        color_message: Image,
        color_image: np.ndarray,
        depth_image: Optional[np.ndarray],
    ) -> None:
        detections = self._detect(color_image)
        self._publish_detections(color_message, detections)
        selected, stable = self._tracker.update(
            detections,
            color_image.shape[1],
            color_image.shape[0],
        )

        depth_m = None
        position = None
        valid = stable
        if self._use_depth:
            valid = False
            if (
                selected is not None
                and stable
                and depth_image is not None
                and self._camera_matrix is not None
                and depth_image.shape[:2] == color_image.shape[:2]
            ):
                depth_m = robust_box_depth(
                    depth_image,
                    selected,
                    unit_scale=float(
                        self.get_parameter('depth_unit_scale').value
                    ),
                    inner_ratio=float(
                        self.get_parameter('depth_inner_ratio').value
                    ),
                    min_depth_m=float(
                        self.get_parameter('min_depth_m').value
                    ),
                    max_depth_m=float(
                        self.get_parameter('max_depth_m').value
                    ),
                    min_samples=int(
                        self.get_parameter('minimum_depth_samples').value
                    ),
                )
                if depth_m is not None:
                    center_x, center_y = selected.center
                    measured = project_pixel(
                        self._camera_matrix,
                        center_x,
                        center_y,
                        depth_m,
                    )
                    position = self._smooth_position(measured)
                    valid = True

        if selected is None and self._tracker.current is None:
            self._filtered_position = None
        self._publish_state(selected, valid)
        if valid and selected is not None:
            self._publish_pixel(color_message, selected)
            if position is not None:
                self._publish_pose(color_message, position)
        self._publish_debug(
            color_message,
            color_image,
            detections,
            selected,
            depth_m,
            valid,
        )

    def _detect(self, image: np.ndarray) -> List[Detection]:
        height, width = image.shape[:2]
        x0, y0, x1, y1 = self._roi_bounds(width, height)
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            return []
        return [
            detection.translated(x0, y0)
            for detection in self._model.infer(crop)
        ]

    def _roi_bounds(
        self,
        width: int,
        height: int,
    ) -> Tuple[int, int, int, int]:
        roi_x = float(self.get_parameter('roi_x').value)
        roi_y = float(self.get_parameter('roi_y').value)
        roi_width = float(self.get_parameter('roi_width').value)
        roi_height = float(self.get_parameter('roi_height').value)
        x0 = int(np.clip(roi_x, 0.0, 1.0) * width)
        y0 = int(np.clip(roi_y, 0.0, 1.0) * height)
        x1 = int(np.clip(roi_x + roi_width, 0.0, 1.0) * width)
        y1 = int(np.clip(roi_y + roi_height, 0.0, 1.0) * height)
        return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)

    def _smooth_position(self, measured: np.ndarray) -> np.ndarray:
        alpha = float(
            np.clip(
                self.get_parameter('position_smoothing_alpha').value,
                0.0,
                1.0,
            )
        )
        if self._filtered_position is None:
            self._filtered_position = measured
        else:
            self._filtered_position = (
                alpha * measured
                + (1.0 - alpha) * self._filtered_position
            )
        return self._filtered_position

    def _publish_detections(
        self,
        source_message: Image,
        detections: List[Detection],
    ) -> None:
        message = Detection2DArray()
        message.header = source_message.header
        for index, detection in enumerate(detections):
            item = Detection2D()
            item.header = source_message.header
            item.id = f'{detection.class_name}:{index}'
            item.bbox.center.position.x = detection.center[0]
            item.bbox.center.position.y = detection.center[1]
            item.bbox.center.theta = 0.0
            item.bbox.size_x = detection.width
            item.bbox.size_y = detection.height
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = detection.class_name
            hypothesis.hypothesis.score = detection.confidence
            item.results.append(hypothesis)
            message.detections.append(item)
        self._detections_publisher.publish(message)

    def _publish_state(
        self,
        selected: Optional[Detection],
        valid: bool,
    ) -> None:
        confidence = selected.confidence if selected is not None else 0.0
        self._valid_publisher.publish(Bool(data=valid))
        self._confidence_publisher.publish(
            Float32(data=float(confidence))
        )

    def _publish_pixel(
        self,
        source_message: Image,
        selected: Detection,
    ) -> None:
        center_x, center_y = selected.center
        pixel = PointStamped()
        pixel.header = source_message.header
        pixel.point.x = center_x
        pixel.point.y = center_y
        pixel.point.z = (selected.width + selected.height) / 4.0
        self._pixel_publisher.publish(pixel)

    def _publish_pose(
        self,
        source_message: Image,
        position: np.ndarray,
    ) -> None:
        pose = PoseStamped()
        pose.header.stamp = source_message.header.stamp
        pose.header.frame_id = (
            self._camera_frame or source_message.header.frame_id
        )
        pose.pose.position.x = float(position[0])
        pose.pose.position.y = float(position[1])
        pose.pose.position.z = float(position[2])
        pose.pose.orientation.w = 1.0
        self._pose_publisher.publish(pose)

    def _publish_debug(
        self,
        source_message: Image,
        image: np.ndarray,
        detections: List[Detection],
        selected: Optional[Detection],
        depth_m: Optional[float],
        valid: bool,
    ) -> None:
        debug_image = image.copy()
        height, width = debug_image.shape[:2]
        x0, y0, x1, y1 = self._roi_bounds(width, height)
        cv2.rectangle(
            debug_image,
            (x0, y0),
            (x1 - 1, y1 - 1),
            (255, 160, 0),
            2,
        )
        for detection in detections:
            self._draw_detection(
                debug_image,
                detection,
                (160, 160, 160),
                1,
            )

        status = 'NO BUTTON'
        status_color = (0, 0, 255)
        if selected is not None:
            status = 'STABLE' if valid else 'TRACKING'
            status_color = (0, 255, 0) if valid else (0, 215, 255)
            self._draw_detection(
                debug_image,
                selected,
                status_color,
                3,
            )
            status += (
                f' {selected.class_name} {selected.confidence:.2f}'
            )
            if depth_m is not None:
                status += f' {depth_m:.3f}m'
        cv2.putText(
            debug_image,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            status_color,
            2,
            cv2.LINE_AA,
        )
        debug_message = self._bridge.cv2_to_imgmsg(
            debug_image,
            encoding='bgr8',
        )
        debug_message.header = source_message.header
        self._debug_publisher.publish(debug_message)

    @staticmethod
    def _draw_detection(
        image: np.ndarray,
        detection: Detection,
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        top_left = (int(round(detection.x1)), int(round(detection.y1)))
        bottom_right = (
            int(round(detection.x2)),
            int(round(detection.y2)),
        )
        cv2.rectangle(image, top_left, bottom_right, color, thickness)
        label = f'{detection.class_name} {detection.confidence:.2f}'
        cv2.putText(
            image,
            label,
            (top_left[0], max(15, top_left[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def main(args=None) -> None:
    """Run the ROS 2 button detector node."""
    rclpy.init(args=args)
    node = ButtonDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
