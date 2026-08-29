"""Run the YOLO ONNX elevator-button ROS 2 node."""

from pathlib import Path
from time import monotonic
from typing import List, Optional, Tuple

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, PoseStamped
import message_filters
import numpy as np
from piper_elevator_app.detector_core import Detection
from piper_elevator_app.detector_core import estimate_surface_normal
from piper_elevator_app.detector_core import filter_detections_by_class
from piper_elevator_app.detector_core import project_pixel
from piper_elevator_app.detector_core import relabel_three_by_three_panel
from piper_elevator_app.detector_core import robust_box_depth
from piper_elevator_app.detector_core import TemporalButtonTracker
from piper_elevator_app.detector_core import YoloOnnxDetector
from piper_elevator_app.motion_core import orientation_from_approach_direction
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float32, String
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


class ButtonDetector(Node):
    """Publish stable 2D and 3D results for one elevator button."""

    def __init__(self) -> None:
        super().__init__('button_detector')
        self._declare_parameters()

        self._bridge = CvBridge()
        self._camera_matrix: Optional[np.ndarray] = None
        self._distortion_coefficients = np.asarray([], dtype=np.float64)
        self._distortion_model = ''
        self._camera_frame = ''
        self._filtered_position: Optional[np.ndarray] = None
        self._filtered_surface_normal: Optional[np.ndarray] = None
        self._use_depth = bool(self.get_parameter('use_depth').value)
        self._input_qos = self._sensor_qos(
            int(self.get_parameter('input_queue_size').value),
            reliable=bool(
                self.get_parameter('reliable_input').value
            ),
        )

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
        warmup_iterations = max(
            0,
            int(self.get_parameter('warmup_iterations').value),
        )
        warmup_started = monotonic()
        self._model.warmup(warmup_iterations)
        self.get_logger().info(
            f'Model warmup completed: iterations={warmup_iterations}, '
            f'elapsed={monotonic() - warmup_started:.3f}s'
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
        self._selected_button_class = self._string_parameter(
            'selected_button_class'
        ).strip()

        self._create_publishers()
        self._create_subscriptions()
        self._publish_selection_state()
        self._performance_window_started = monotonic()
        self._performance_frame_count = 0
        self._performance_total_seconds = 0.0
        self._performance_max_seconds = 0.0
        self._last_debug_publish_time: Optional[float] = None
        self._debug_frame_count = 0
        mode = 'RGB-D' if self._use_depth else 'RGB'
        self.get_logger().info(
            f'YOLO button detector ready in {mode} mode. '
            f'model={model_path}, '
            f'providers={self._model.active_providers}, '
            f'color={self._string_parameter("color_topic")}, '
            f'input_queue={self._input_qos.depth}, '
            f'input_reliability={self._input_qos.reliability.name}, '
            f'selected_button={self._selection_log_text()}'
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
            'button_surface_pose_topic',
            '/button_surface_pose',
        )
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
        self.declare_parameter('button_selection_topic', '/button_selection')
        self.declare_parameter('button_selected_topic', '/button_selected')
        self.declare_parameter('selected_button_class', '')
        self.declare_parameter('simulation_layout_relabel', False)
        self.declare_parameter(
            'simulation_panel_layout_labels',
            ['1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm'],
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
        self.declare_parameter('warmup_iterations', 2)
        self.declare_parameter('confidence_threshold', 0.60)
        self.declare_parameter('nms_iou_threshold', 0.45)

        # Inference is slower than a typical 30 FPS camera. Keep only the
        # newest input so the robot never acts on a growing queue of old
        # images.
        self.declare_parameter('input_queue_size', 1)
        self.declare_parameter('reliable_input', False)
        self.declare_parameter('sync_queue_size', 2)
        self.declare_parameter('sync_slop_seconds', 0.08)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('debug_image_only_when_subscribed', True)
        self.declare_parameter('debug_image_scale', 0.5)
        self.declare_parameter('debug_max_fps', 0.0)
        self.declare_parameter('performance_log_interval_seconds', 5.0)
        self.declare_parameter('roi_x', 0.0)
        self.declare_parameter('roi_y', 0.0)
        self.declare_parameter('roi_width', 1.0)
        self.declare_parameter('roi_height', 1.0)

        self.declare_parameter('depth_unit_scale', 0.001)
        self.declare_parameter('depth_inner_ratio', 0.45)
        self.declare_parameter('minimum_depth_samples', 20)
        self.declare_parameter('min_depth_m', 0.10)
        self.declare_parameter('max_depth_m', 2.00)
        self.declare_parameter('surface_inner_ratio', 0.70)
        self.declare_parameter('surface_minimum_samples', 30)
        self.declare_parameter('surface_maximum_samples', 400)
        self.declare_parameter('surface_max_residual_m', 0.004)
        self.declare_parameter('surface_max_tilt_degrees', 60.0)
        self.declare_parameter('surface_normal_smoothing_alpha', 0.25)

        self.declare_parameter('required_stable_frames', 5)
        self.declare_parameter('max_missed_frames', 2)
        self.declare_parameter('tracking_minimum_iou', 0.15)
        self.declare_parameter('max_center_jump_ratio', 0.10)
        self.declare_parameter('position_smoothing_alpha', 0.35)

    def _create_publishers(self) -> None:
        self._pose_publisher = self.create_publisher(
            PoseStamped,
            self._string_parameter('button_pose_topic'),
            1,
        )
        self._surface_pose_publisher = self.create_publisher(
            PoseStamped,
            self._string_parameter('button_surface_pose_topic'),
            1,
        )
        self._pixel_publisher = self.create_publisher(
            PointStamped,
            self._string_parameter('button_pixel_topic'),
            1,
        )
        self._detections_publisher = self.create_publisher(
            Detection2DArray,
            self._string_parameter('button_detections_topic'),
            1,
        )
        self._valid_publisher = self.create_publisher(
            Bool,
            self._string_parameter('button_valid_topic'),
            1,
        )
        self._confidence_publisher = self.create_publisher(
            Float32,
            self._string_parameter('button_confidence_topic'),
            1,
        )
        self._debug_publisher = self.create_publisher(
            Image,
            self._string_parameter('debug_image_topic'),
            self._sensor_qos(1),
        )
        selection_state_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._selection_state_publisher = self.create_publisher(
            String,
            self._string_parameter('button_selected_topic'),
            selection_state_qos,
        )

    def _create_subscriptions(self) -> None:
        color_topic = self._string_parameter('color_topic')
        self._camera_info_subscription = None
        self._color_subscription = None
        self._depth_subscription = None
        self._synchronizer = None
        self._color_only_subscription = None
        self._selection_subscription = self.create_subscription(
            String,
            self._string_parameter('button_selection_topic'),
            self._selection_callback,
            10,
        )
        if self._use_depth:
            self._camera_info_subscription = self.create_subscription(
                CameraInfo,
                self._string_parameter('camera_info_topic'),
                self._camera_info_callback,
                self._input_qos,
            )
            self._color_subscription = message_filters.Subscriber(
                self,
                Image,
                color_topic,
                qos_profile=self._input_qos,
            )
            self._depth_subscription = message_filters.Subscriber(
                self,
                Image,
                self._string_parameter('depth_topic'),
                qos_profile=self._input_qos,
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
                self._input_qos,
            )

    def _selection_callback(self, message: String) -> None:
        requested = str(message.data).strip()
        if requested.casefold() in {'clear', 'none'}:
            requested = ''
        if requested == self._selected_button_class:
            self._publish_selection_state()
            return
        self._selected_button_class = requested
        self._tracker.reset()
        self._filtered_position = None
        self._filtered_surface_normal = None
        self._publish_selection_state()
        self._publish_state(None, False)
        self.get_logger().info(
            f'Button selection changed: {self._selection_log_text()}'
        )

    def _publish_selection_state(self) -> None:
        self._selection_state_publisher.publish(
            String(data=self._selected_button_class)
        )

    def _selection_log_text(self) -> str:
        return self._selected_button_class or '<none>'

    @staticmethod
    def _sensor_qos(
        depth: int,
        reliable: bool = False,
    ) -> QoSProfile:
        """Return a low-depth profile suitable for live sensor data."""
        return QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=max(1, int(depth)),
            reliability=(
                QoSReliabilityPolicy.RELIABLE
                if reliable
                else QoSReliabilityPolicy.BEST_EFFORT
            ),
            durability=QoSDurabilityPolicy.VOLATILE,
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
        self._distortion_coefficients = np.asarray(
            message.d,
            dtype=np.float64,
        )
        self._distortion_model = message.distortion_model
        self._camera_frame = message.header.frame_id

    def _color_callback(self, color_message: Image) -> None:
        started = monotonic()
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
        self._record_performance(color_message, monotonic() - started)

    def _rgbd_callback(
        self,
        color_message: Image,
        depth_message: Image,
    ) -> None:
        started = monotonic()
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
        self._record_performance(color_message, monotonic() - started)

    def _record_performance(
        self,
        source_message: Image,
        processing_seconds: float,
    ) -> None:
        """Periodically report throughput and source-frame age."""
        self._performance_frame_count += 1
        self._performance_total_seconds += processing_seconds
        self._performance_max_seconds = max(
            self._performance_max_seconds,
            processing_seconds,
        )

        interval = max(
            0.0,
            float(
                self.get_parameter(
                    'performance_log_interval_seconds'
                ).value
            ),
        )
        elapsed = monotonic() - self._performance_window_started
        if interval <= 0.0 or elapsed < interval:
            return

        count = max(1, self._performance_frame_count)
        average_ms = 1000.0 * self._performance_total_seconds / count
        maximum_ms = 1000.0 * self._performance_max_seconds
        throughput = count / max(elapsed, 1e-6)
        debug_throughput = self._debug_frame_count / max(elapsed, 1e-6)
        age_text = 'unavailable'
        stamp_ns = (
            int(source_message.header.stamp.sec) * 1_000_000_000
            + int(source_message.header.stamp.nanosec)
        )
        if stamp_ns > 0:
            age_seconds = (
                self.get_clock().now().nanoseconds - stamp_ns
            ) / 1_000_000_000.0
            if 0.0 <= age_seconds <= 60.0:
                age_text = f'{age_seconds * 1000.0:.1f} ms'

        self.get_logger().info(
            'Performance: '
            f'{throughput:.1f} FPS, '
            f'debug={debug_throughput:.1f} FPS, '
            f'average={average_ms:.1f} ms, '
            f'max={maximum_ms:.1f} ms, '
            f'input_age={age_text}'
        )
        self._performance_window_started = monotonic()
        self._performance_frame_count = 0
        self._performance_total_seconds = 0.0
        self._performance_max_seconds = 0.0
        self._debug_frame_count = 0

    def _process_frame(
        self,
        color_message: Image,
        color_image: np.ndarray,
        depth_image: Optional[np.ndarray],
    ) -> None:
        detections = self._detect(color_image)
        self._publish_detections(color_message, detections)
        selected_detections = filter_detections_by_class(
            detections,
            self._selected_button_class,
        )
        selected, stable = self._tracker.update(
            selected_detections,
            color_image.shape[1],
            color_image.shape[0],
        )

        depth_m = None
        position = None
        surface_normal = None
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
                        self._distortion_coefficients,
                        self._distortion_model,
                    )
                    position = self._smooth_position(measured)
                    valid = True
                    measured_normal = estimate_surface_normal(
                        depth_image,
                        selected,
                        self._camera_matrix,
                        unit_scale=float(
                            self.get_parameter('depth_unit_scale').value
                        ),
                        inner_ratio=float(
                            self.get_parameter('surface_inner_ratio').value
                        ),
                        min_depth_m=float(
                            self.get_parameter('min_depth_m').value
                        ),
                        max_depth_m=float(
                            self.get_parameter('max_depth_m').value
                        ),
                        min_samples=int(
                            self.get_parameter(
                                'surface_minimum_samples'
                            ).value
                        ),
                        max_samples=int(
                            self.get_parameter(
                                'surface_maximum_samples'
                            ).value
                        ),
                        max_residual_m=float(
                            self.get_parameter(
                                'surface_max_residual_m'
                            ).value
                        ),
                        max_tilt_degrees=float(
                            self.get_parameter(
                                'surface_max_tilt_degrees'
                            ).value
                        ),
                        distortion_coefficients=(
                            self._distortion_coefficients
                        ),
                        distortion_model=self._distortion_model,
                    )
                    if measured_normal is not None:
                        surface_normal = self._smooth_surface_normal(
                            measured_normal
                        )

        if selected is None and self._tracker.current is None:
            self._filtered_position = None
            self._filtered_surface_normal = None
        self._publish_state(selected, valid)
        if valid and selected is not None:
            self._publish_pixel(color_message, selected)
            if position is not None:
                self._publish_pose(color_message, position)
                if surface_normal is not None:
                    self._publish_surface_pose(
                        color_message,
                        position,
                        surface_normal,
                    )
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
        detections = [
            detection.translated(x0, y0)
            for detection in self._model.infer(crop)
        ]
        if bool(self.get_parameter('simulation_layout_relabel').value):
            detections = relabel_three_by_three_panel(
                detections,
                self._string_list_parameter(
                    'simulation_panel_layout_labels'
                ),
                self._model.class_names,
            )
        return detections

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

    def _smooth_surface_normal(self, measured: np.ndarray) -> np.ndarray:
        normal = np.asarray(measured, dtype=np.float64)
        normal /= np.linalg.norm(normal)
        alpha = float(np.clip(
            self.get_parameter('surface_normal_smoothing_alpha').value,
            0.0,
            1.0,
        ))
        if self._filtered_surface_normal is None:
            self._filtered_surface_normal = normal
        else:
            if np.dot(normal, self._filtered_surface_normal) < 0.0:
                normal = -normal
            self._filtered_surface_normal = (
                alpha * normal
                + (1.0 - alpha) * self._filtered_surface_normal
            )
            self._filtered_surface_normal /= np.linalg.norm(
                self._filtered_surface_normal
            )
        return self._filtered_surface_normal

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

    def _publish_surface_pose(
        self,
        source_message: Image,
        position: np.ndarray,
        surface_normal: np.ndarray,
    ) -> None:
        pose = PoseStamped()
        pose.header.stamp = source_message.header.stamp
        pose.header.frame_id = (
            self._camera_frame or source_message.header.frame_id
        )
        pose.pose.position.x = float(position[0])
        pose.pose.position.y = float(position[1])
        pose.pose.position.z = float(position[2])
        orientation = orientation_from_approach_direction(
            surface_normal,
            [0.0, 0.0, 0.0, 1.0],
        )
        pose.pose.orientation.x = float(orientation[0])
        pose.pose.orientation.y = float(orientation[1])
        pose.pose.orientation.z = float(orientation[2])
        pose.pose.orientation.w = float(orientation[3])
        self._surface_pose_publisher.publish(pose)

    def _publish_debug(
        self,
        source_message: Image,
        image: np.ndarray,
        detections: List[Detection],
        selected: Optional[Detection],
        depth_m: Optional[float],
        valid: bool,
    ) -> None:
        if not bool(self.get_parameter('publish_debug_image').value):
            return
        if (
            bool(
                self.get_parameter(
                    'debug_image_only_when_subscribed'
                ).value
            )
            and self._debug_publisher.get_subscription_count() == 0
        ):
            return

        now = monotonic()
        maximum_fps = max(
            0.0,
            float(self.get_parameter('debug_max_fps').value),
        )
        if (
            maximum_fps > 0.0
            and self._last_debug_publish_time is not None
            and now - self._last_debug_publish_time < 1.0 / maximum_fps
        ):
            return

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

        status = 'SELECT BUTTON'
        status_color = (0, 0, 255)
        if self._selected_button_class:
            status = f'NO {self._selected_button_class}'
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
        scale = float(
            np.clip(
                self.get_parameter('debug_image_scale').value,
                0.1,
                1.0,
            )
        )
        if scale < 1.0:
            debug_image = cv2.resize(
                debug_image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
        debug_message = self._bridge.cv2_to_imgmsg(
            debug_image,
            encoding='bgr8',
        )
        debug_message.header = source_message.header
        self._debug_publisher.publish(debug_message)
        self._last_debug_publish_time = now
        self._debug_frame_count += 1

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
