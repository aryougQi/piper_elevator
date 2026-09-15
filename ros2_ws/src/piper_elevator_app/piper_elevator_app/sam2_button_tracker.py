"""YOLO-prompted SAM2 video tracker for the post-coarse-approach phase.

The node deliberately owns no robot controls.  It publishes a fresh RGB-D
target only while a validated SAM2 mask exists; LOST publishes an invalidation
state and never republishes the last target.
"""

from collections import deque
import copy
import json
import tempfile
import threading
import time

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformListener
from piper_elevator_app.motion_core import quaternion_to_matrix
from piper_elevator_app.plane_core import fit_plane_consensus
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection2DArray


class _Sam2Backend:
    """Bounded live-frame adapter for the installed official video predictor.

    Update its image cache and frame count explicitly, preserving conditioning
    memory across frames. Never substitute a per-frame detector or CPU fallback.
    """

    def __init__(self, config, checkpoint, device, *, compile_model=False):
        if not config or not checkpoint:
            raise RuntimeError('sam2_model_cfg and sam2_checkpoint are required')
        try:
            import torch
            if device == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('SAM2 device=cuda requested but CUDA is unavailable')
            from sam2.build_sam import build_sam2_video_predictor
        except Exception as error:
            raise RuntimeError(f'official SAM2 import failed: {error}') from error
        torch.set_float32_matmul_precision('high')
        if device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        self._predictor = build_sam2_video_predictor(config, checkpoint, device=device)
        self._compiled = bool(compile_model) and self._predictor.device.type == 'cuda'
        if self._compiled:
            # Compile the official modules without CUDA Graph tensor-lifetime
            # assumptions: live tracking retains memory across invocations.
            # Keep model weights, image size, masks and temporal window intact.
            for name in ('image_encoder', 'memory_encoder', 'memory_attention',
                         'sam_prompt_encoder', 'sam_mask_decoder'):
                module = getattr(self._predictor, name)
                module.forward = torch.compile(
                    module.forward, fullgraph=True,
                    dynamic=name == 'memory_attention',
                    # Exhaustive runtime autotuning can spend minutes per
                    # memory shape. Use compiled fusion without that search.
                    options={'triton.cudagraphs': False, 'coordinate_descent_tuning': False},
                )
        # Keep normalization tensors on the inference device and avoid the
        # PIL conversion/allocation on every live frame.
        self._mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self._predictor.device,
        ).view(3, 1, 1)
        self._std = torch.tensor(
            [0.229, 0.224, 0.225], device=self._predictor.device,
        ).view(3, 1, 1)
        required = ('init_state', 'add_new_points_or_box', 'propagate_in_video')
        if not all(hasattr(self._predictor, name) for name in required):
            raise RuntimeError('official SAM2 video predictor API is incomplete')
        self._directory = tempfile.TemporaryDirectory(prefix='sam2_frames_')
        self._frame_index = -1
        self._state = None
        self._object_id = 1

    def reset(self):
        self._state = None
        self._frame_index = -1

    def warmup(self):
        """Compile before accepting live images; discard all synthetic state."""
        if not self._compiled:
            return
        # Output resolution is independent of the compiled 1024 image encoder.
        frame = np.zeros((480, 848, 3), dtype=np.uint8)
        frame[200:280, 380:460] = 180
        self.initialize(frame, np.array([380., 200., 460., 280.]))
        # Exercise memory attention with a full pointer window before control.
        for _ in range(18):
            self.track(frame)
        self.reset()

    def _tensor(self, frame):
        import torch
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(
            rgb, (self._predictor.image_size, self._predictor.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        array = np.ascontiguousarray(rgb)
        # Upload bytes first. Combining dtype/device in .to() converts on CPU
        # and sends four times the data over PCIe on the installed Torch build.
        tensor = torch.from_numpy(array).to(self._predictor.device)
        tensor = tensor.permute(2, 0, 1).float().div_(255.0)
        return tensor.sub_(self._mean).div_(self._std)

    def initialize(self, frame, box):
        import torch
        self.reset()
        self._frame_index = 0
        # init_state constructs the official per-object memory dictionaries.
        # Replace its one JPEG with the exact, uncompressed live frame tensor.
        cv2.imwrite(f'{self._directory.name}/000000.jpg', frame)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=self._predictor.device.type == 'cuda'):
            self._state = self._predictor.init_state(video_path=self._directory.name)
            self._state['images'] = {0: self._tensor(frame)}
            self._state['cached_features'].clear()
            _, object_ids, mask_logits = self._predictor.add_new_points_or_box(
                self._state, frame_idx=0, obj_id=self._object_id,
                box=np.asarray(box, dtype=np.float32),
            )
            return self._mask_from_logits(object_ids, mask_logits)

    def track(self, frame):
        import torch
        if self._state is None:
            raise RuntimeError('SAM2 has not been initialized')
        if frame.shape[:2] != (self._state['video_height'], self._state['video_width']):
            raise RuntimeError('Camera resolution changed during tracking')
        self._frame_index += 1
        index = self._frame_index
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=self._predictor.device.type == 'cuda'):
            self._state['images'] = {index: self._tensor(frame)}
            self._state['num_frames'] = index + 1
            result = None
            for frame_idx, object_ids, mask_logits in self._predictor.propagate_in_video(
                self._state, start_frame_idx=index, max_frame_num_to_track=0,
            ):
                if frame_idx == index:
                    result = self._mask_from_logits(object_ids, mask_logits)
            # Keep the initial conditioning memory and a bounded recent window.
            # This exceeds the tiny model's temporal memory and pointer window.
            for outputs in self._state['output_dict_per_obj'].values():
                for old in list(outputs['non_cond_frame_outputs']):
                    if old < index - 32:
                        del outputs['non_cond_frame_outputs'][old]
            for tracked in self._state['frames_tracked_per_obj'].values():
                for old in list(tracked):
                    if old < index - 32:
                        del tracked[old]
            if result is None:
                raise RuntimeError('SAM2 produced no mask for the latest frame')
            return result

    @staticmethod
    def _mask_from_logits(object_ids, mask_logits):
        if len(object_ids) == 0:
            return np.zeros(mask_logits.shape[-2:], dtype=bool)
        index = list(object_ids).index(1) if 1 in list(object_ids) else 0
        logits = mask_logits[index]
        return (logits > 0.0).detach().cpu().numpy().squeeze().astype(bool)


class Sam2ButtonTracker(Node):
    IDLE = 'IDLE'
    WAITING_FOR_TARGET = 'WAITING_FOR_TARGET'
    INITIALIZING = 'INITIALIZING'
    TRACKING = 'TRACKING'
    LOST = 'LOST'

    def __init__(self):
        super().__init__('sam2_button_tracker')
        self._declare_parameters()
        self._bridge = CvBridge()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._lock = threading.RLock()
        self._frame = None
        self._frame_stamp_ns = 0
        self._frame_received_at = 0.0
        self._frame_id = ''
        self._detections = None
        self._detection_stamp_ns = 0
        self._surface_pose = None
        self._camera_info = None
        self._depth = None
        self._depth_stamp_ns = 0
        self._selected_label = ''
        self._state = self.IDLE
        self._coarse_ready_at = None
        self._stable_count = 0
        self._geometry_failures = 0
        self._last_center = None
        self._backend = None
        self._generation = 0
        self._processed_stamp_ns = 0
        self._depth_frames = deque(maxlen=60)
        self._output_stamp_ns = 0
        self._tracking_stamp_ns = 0
        self._anchor_world = None
        self._seed_center = None
        self._support_radius_m = 0.04
        self._stop = threading.Event()
        self._frame_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._debug_condition = threading.Condition()
        self._debug_pending = None
        self._debug_worker = threading.Thread(target=self._debug_loop, daemon=True)
        self._geometry_condition = threading.Condition()
        self._geometry_pending = None
        self._geometry_worker = threading.Thread(target=self._geometry_loop, daemon=True)
        self._performance_started_at = time.monotonic()
        self._performance_samples = []
        self._performance_debug_count = 0
        self._geometry_samples = []

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._state_pub = self.create_publisher(String, self._string('state_topic'), latched)
        self._center_pub = self.create_publisher(PointStamped, self._string('center_topic'), 1)
        self._pose_pub = self.create_publisher(PoseStamped, self._string('surface_pose_topic'), 1)
        self._debug_pub = self.create_publisher(Image, self._string('debug_image_topic'), 1)
        self._performance_pub = self.create_publisher(String, '~/performance', 1)
        self.create_subscription(Image, self._string('image_topic'), self._image_callback, 1)
        self.create_subscription(CameraInfo, self._string('camera_info_topic'), self._camera_info_callback, 1)
        self.create_subscription(Image, self._string('depth_topic'), self._depth_callback, 1)
        self.create_subscription(Detection2DArray, self._string('detections_topic'), self._detections_callback, 1)
        self.create_subscription(String, self._string('selected_topic'), self._selected_callback, latched)
        self.create_subscription(String, self._string('approach_status_topic'), self._approach_callback, latched)
        self.create_subscription(PoseStamped, self._string('yolo_surface_pose_topic'), self._surface_callback, 1)
        self.create_service(Trigger, '~/initialize', self._initialize_service)
        self.create_subscription(String, '/elevator_task/status', self._task_callback, 10)
        self._publish_state('waiting_for_coarse_approach')
        if bool(self.get_parameter('enabled').value):
            try:
                import torch
                torch.set_num_threads(max(1, int(self.get_parameter('torch_num_threads').value)))
                cv2.setNumThreads(1)
                started = time.monotonic()
                self.get_logger().info('Loading SAM2 and warming inference kernels; waiting for model readiness')
                self._backend = _Sam2Backend(
                    self._string('model_cfg'), self._string('checkpoint'), self._string('device'),
                    compile_model=bool(self.get_parameter('compile_model').value),
                )
                self._backend.warmup()
                self.get_logger().info(
                    f'SAM2 model ready: compiled={self._backend._compiled}, '
                    f'warmup={time.monotonic()-started:.1f}s'
                )
            except Exception as error:
                self.get_logger().error(str(error))
                self._set_state(self.LOST, 'sam2_unavailable')
        self._worker.start()
        self._debug_worker.start()
        self._geometry_worker.start()

    def _declare_parameters(self):
        self.declare_parameter('enabled', True)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('model_cfg', '')
        self.declare_parameter('checkpoint', '')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('compile_model', True)
        self.declare_parameter('torch_num_threads', 1)
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('detections_topic', '/button_detections')
        self.declare_parameter('selected_topic', '/button_selected')
        self.declare_parameter('approach_status_topic', '/button_approach/status')
        self.declare_parameter('yolo_surface_pose_topic', '/button_surface_pose')
        self.declare_parameter('state_topic', '/button_tracking_state')
        self.declare_parameter('center_topic', '/sam2_button_tracker/center')
        self.declare_parameter('surface_pose_topic', '/sam2_button_tracker/surface_pose')
        self.declare_parameter('debug_image_topic', '/sam2_button_tracker/debug_image')
        self.declare_parameter('init_max_age_sec', 0.15)
        self.declare_parameter('settle_time_sec', 0.3)
        self.declare_parameter('tracking_timeout_sec', 1.0)
        self.declare_parameter('stable_frames', 5)
        self.declare_parameter('min_mask_area_px', 200)
        self.declare_parameter('max_mask_area_ratio', 0.6)
        self.declare_parameter('max_center_jump_px', 150.0)
        self.declare_parameter('use_mask_median_depth', True)
        self.declare_parameter('min_depth_m', 0.10)
        self.declare_parameter('max_depth_m', 2.0)
        self.declare_parameter('debug_image', True)
        self.declare_parameter('debug_max_fps', 3.0)
        self.declare_parameter('geometry_max_fps', 15.0)
        self.declare_parameter('debug_image_only_when_subscribed', True)
        self.declare_parameter('performance_log_interval_seconds', 5.0)
        self.declare_parameter('max_geometry_failures', 5)

    def _string(self, name):
        return str(self.get_parameter(name).value)

    def _set_state(self, state, reason=''):
        with self._lock:
            self._state = state
            if state != self.TRACKING:
                self._stable_count = 0
        self._publish_state(reason)

    def _publish_state(self, reason=''):
        with self._lock:
            state, label, stamp, frame_id = self._state, self._selected_label, self._output_stamp_ns, self._frame_id
            tracking_valid = state == self.TRACKING and self._stable_count >= int(self.get_parameter('stable_frames').value)
            if reason != 'tracker_ready':
                stamp = self._tracking_stamp_ns
            ready = state == self.TRACKING and reason == 'tracker_ready'
        payload = {
            'source': 'sam2_button_tracker', 'state': state,
            'reason': reason, 'frame_id': frame_id,
            'stamp': {'sec': stamp // 1_000_000_000, 'nanosec': stamp % 1_000_000_000},
            'selected': {
                'class_name': label, 'stable_detection': ready,
                'depth_valid': ready, 'surface_valid': ready,
                'measured': ({'class_name': label} if ready else None),
                'tracking_valid': tracking_valid,
                'geometry_valid': ready,
            },
        }
        self._state_pub.publish(String(data=json.dumps(payload)))

    def _reset_tracking(self, state, reason):
        with self._lock:
            self._generation += 1
            self._state = state
            self._stable_count = 0
            self._last_center = None
            self._coarse_ready_at = time.monotonic() if state == self.WAITING_FOR_TARGET else None
            self._processed_stamp_ns = self._frame_stamp_ns
            self._output_stamp_ns = 0
            self._tracking_stamp_ns = 0
            self._anchor_world = None
            self._seed_center = None
        self._publish_state(reason)

    def _selected_callback(self, message):
        with self._lock:
            label = str(message.data).strip()
            if label == self._selected_label:
                return
            self._selected_label = label
            self._reset_tracking(self.IDLE, 'selection_changed')

    def _task_callback(self, message):
        if str(message.data).startswith(('HOMING', 'RECOVERING', 'STOP_REQUESTED', 'COMPLETE', 'FAILED', 'IDLE')):
            self._reset_tracking(self.IDLE, 'task_inactive')

    def _approach_callback(self, message):
        if str(message.data).startswith('APPROACH_REACHED_VERIFIED'):
            self._reset_tracking(self.WAITING_FOR_TARGET, 'coarse_approach_complete')
        elif str(message.data).startswith(('PLANNING', 'EXECUTING')):
            self._reset_tracking(self.IDLE, 'coarse_approach_in_progress')

    def _initialize_service(self, request, response):
        del request
        response.success = self._backend is not None and bool(self._selected_label)
        response.message = 'SAM2 waiting for coarse completion and a fresh selected YOLO bbox'
        # Diagnostic request must not bypass the verified coarse arrival gate.
        if self._coarse_ready_at is None:
            response.success = False
        elif response.success:
            self._reset_tracking(self.WAITING_FOR_TARGET, 'manual_initialize_requested')
        return response

    def _image_callback(self, message):
        try:
            frame = self._bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        except Exception as error:
            self.get_logger().warning(f'Cannot decode tracker image: {error}', throttle_duration_sec=2.0)
            return
        with self._lock:
            self._frame = np.asarray(frame).copy()
            self._frame_stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
            self._frame_received_at = time.monotonic()
            self._frame_id = message.header.frame_id
        self._frame_event.set()

    def _camera_info_callback(self, message):
        with self._lock:
            self._camera_info = copy.deepcopy(message)

    def _depth_callback(self, message):
        try:
            depth = self._bridge.imgmsg_to_cv2(message, desired_encoding='passthrough')
        except Exception:
            return
        with self._lock:
            self._depth = np.asarray(depth).copy()
            self._depth_stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
            self._depth_frames.append((self._depth_stamp_ns, self._depth))

    def _detections_callback(self, message):
        with self._lock:
            self._detections = copy.deepcopy(message)
            self._detection_stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec

    def _surface_callback(self, message):
        with self._lock:
            self._surface_pose = copy.deepcopy(message)

    def _find_bbox(self, detections, label):
        if detections is None or not label:
            return None
        for item in detections.detections:
            labels = [r.hypothesis.class_id for r in item.results]
            if label.casefold() not in [str(value).casefold() for value in labels]:
                continue
            center = item.bbox.center.position
            return np.array([
                center.x - item.bbox.size_x / 2.0,
                center.y - item.bbox.size_y / 2.0,
                center.x + item.bbox.size_x / 2.0,
                center.y + item.bbox.size_y / 2.0,
            ], dtype=np.float32)
        return None

    def _worker_loop(self):
        backend_generation = -1
        while not self._stop.is_set():
            # A new frame wakes inference immediately. While busy, this event
            # coalesces arrivals into one latest-frame update, not a FIFO.
            self._frame_event.wait(0.05)
            self._frame_event.clear()
            if self._stop.is_set():
                break
            with self._lock:
                # Image callbacks replace their owned array; a worker can keep
                # a reference without copying a full frame on every idle poll.
                frame = self._frame
                stamp_ns, ready_at = self._frame_stamp_ns, self._coarse_ready_at
                received_at = getattr(self, '_frame_received_at', 0.0)
                detections, detection_stamp = self._detections, self._detection_stamp_ns
                label, state = self._selected_label, self._state
                generation = self._generation
            if self._backend is not None and backend_generation != generation:
                self._backend.reset()
                backend_generation = generation
            if state not in (self.WAITING_FOR_TARGET, self.TRACKING):
                continue
            age = (self.get_clock().now().nanoseconds - stamp_ns) / 1e9
            if state == self.TRACKING and age > float(self.get_parameter('tracking_timeout_sec').value):
                self._set_current_state(generation, self.LOST, 'image_timeout')
                continue
            if stamp_ns <= self._processed_stamp_ns:
                continue
            if frame is None or self._backend is None or not bool(self.get_parameter('enabled').value):
                continue
            if state == self.WAITING_FOR_TARGET and ready_at is not None:
                if time.monotonic() - ready_at < float(self.get_parameter('settle_time_sec').value):
                    continue
                if stamp_ns <= 0 or detection_stamp <= 0 or abs(stamp_ns - detection_stamp) / 1e9 > float(self.get_parameter('init_max_age_sec').value):
                    continue
                box = self._find_bbox(detections, label)
                if box is None:
                    continue
                self._processed_stamp_ns = stamp_ns
                if not self._set_current_state(generation, self.INITIALIZING, 'yolo_bbox_prompt'):
                    continue
                try:
                    mask = self._backend.initialize(frame, box)
                except Exception as error:
                    self.get_logger().error(f'SAM2 initialization failed: {error}')
                    self._set_current_state(generation, self.LOST, 'initialization_failed')
                    continue
                with self._lock:
                    if generation != self._generation:
                        continue
                    if not self._accept_mask(mask, frame.shape, None):
                        self._set_current_state(generation, self.LOST, 'invalid_initial_mask')
                        continue
                    self._stable_count = 0
                    self._seed_center = ((box[0]+box[2])/2, (box[1]+box[3])/2)
                    self._performance_started_at = time.monotonic()
                    self._performance_samples = []
                    self._geometry_samples.clear()
                    self._performance_debug_count = 0
                    self._set_state(self.TRACKING, 'initialized')
                self._publish_if_current(frame, mask, stamp_ns, generation, received_at)
            elif state == self.TRACKING:
                self._processed_stamp_ns = stamp_ns
                try:
                    inference_started = time.monotonic()
                    mask = self._backend.track(frame)
                    inference_seconds = time.monotonic() - inference_started
                except Exception as error:
                    self.get_logger().warning(f'SAM2 tracking failed: {error}', throttle_duration_sec=1.0)
                    self._set_current_state(generation, self.LOST, 'tracking_exception')
                    continue
                with self._lock:
                    if generation != self._generation:
                        continue
                    if not self._accept_mask(mask, frame.shape, self._last_center):
                        self._set_current_state(generation, self.LOST, 'mask_quality_failed')
                        continue
                self._publish_if_current(frame, mask, stamp_ns, generation, received_at)
                if hasattr(self, '_performance_pub'):
                    self._record_performance(
                        stamp_ns, received_at, inference_seconds,
                        time.monotonic() - inference_started,
                    )

    def _set_current_state(self, generation, state, reason):
        with self._lock:
            if generation != self._generation:
                return False
            self._set_state(state, reason)
            return True

    def _publish_if_current(self, frame, mask, stamp_ns, generation, received_at=None):
        with self._lock:
            if generation != self._generation:
                return
            frame_id = self._frame_id
        # Never wait for TF while holding the image callback lock: a single
        # executor must be free to drain images and the TF messages behind them.
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return
        header = Header()
        header.stamp = Time(nanoseconds=stamp_ns).to_msg()
        header.frame_id = frame_id
        # Fill the pixel center without waiting for depth or TF.
        center = PointStamped(header=header)
        center.point.x, center.point.y = float(xs.mean()), float(ys.mean())
        center.point.z = float(len(xs))
        with self._lock:
            if generation != self._generation:
                return
            if self._state != self.TRACKING:
                return
            self._center_pub.publish(center)
            self._tracking_stamp_ns = stamp_ns
            self._stable_count += 1
            if self._stable_count >= int(self.get_parameter('stable_frames').value):
                self._publish_state('tracking_valid')
        self._queue_debug(frame, mask, header, float(xs.mean()), float(ys.mean()))
        with self._geometry_condition:
            self._geometry_pending = (frame, mask, stamp_ns, generation, received_at)
            self._geometry_condition.notify()

    def _geometry_loop(self):
        last_started = 0.0
        while not self._stop.is_set():
            fps = float(self.get_parameter('geometry_max_fps').value)
            if fps > 0 and self._stop.wait(max(0., 1./fps-(time.monotonic()-last_started))):
                return
            with self._geometry_condition:
                self._geometry_condition.wait_for(
                    lambda: self._stop.is_set() or self._geometry_pending is not None,
                )
                if self._stop.is_set():
                    return
                pending = self._geometry_pending
                self._geometry_pending = None
            frame, mask, stamp_ns, generation, received_at = pending
            last_started = time.monotonic()
            with self._lock:
                if generation != self._generation:
                    continue
            try:
                transform = self._camera_transform(stamp_ns, self._frame_id)
                self._publish_output(frame, mask, stamp_ns, transform, generation, received_at)
            except Exception as error:
                with self._lock:
                    if generation == self._generation and self._state == self.TRACKING:
                        self._publish_state('geometry_invalid')
                self.get_logger().warning(
                    f'SAM2 geometry transform rejected: {error}', throttle_duration_sec=1.0,
                )

    def _accept_mask(self, mask, shape, previous_center):
        if mask.shape[:2] != shape[:2]:
            return False
        area = int(np.count_nonzero(mask))
        if area < int(self.get_parameter('min_mask_area_px').value):
            return False
        if area / float(shape[0] * shape[1]) > float(self.get_parameter('max_mask_area_ratio').value):
            return False
        ys, xs = np.nonzero(mask)
        center = np.array([float(np.mean(xs)), float(np.mean(ys))])
        if previous_center is not None and np.linalg.norm(center - previous_center) > float(self.get_parameter('max_center_jump_px').value):
            return False
        self._last_center = center
        return True

    def _camera_transform(self, stamp_ns, frame_id):
        transform = self._tf_buffer.lookup_transform(
            self._string('base_frame'), frame_id, Time(nanoseconds=stamp_ns),
            timeout=Duration(seconds=0.1),
        ).transform
        p, q = transform.translation, transform.rotation
        return np.array([p.x, p.y, p.z]), quaternion_to_matrix([q.x, q.y, q.z, q.w])

    def _registered_target(self, mask, depth_image, info, stamp_ns, camera_transform=None, generation=None):
        """Locate the registered physical point, supported by fresh visible mask depth.

        Cropping/occlusion moves a mask centroid. Retain the selected material
        point in the static panel frame, and measure its plane each frame. This
        is never a fallback: absent/mismatched live mask, depth or TF fails closed.
        """
        with self._lock:
            anchor_world = self._anchor_world
            seed_center = self._seed_center
            support_radius = self._support_radius_m
        ys, xs = np.nonzero(mask)
        scale = 0.001 if depth_image.dtype == np.uint16 else 1.0
        depths = depth_image[ys, xs].astype(float) * scale
        valid = np.isfinite(depths) & (depths >= float(self.get_parameter('min_depth_m').value)) & (depths <= float(self.get_parameter('max_depth_m').value))
        xs, ys, depths = xs[valid], ys[valid], depths[valid]
        if len(depths) < 30:
            raise ValueError('insufficient valid mask depth')
        points = np.column_stack(((xs-info.k[2])*depths/info.k[0], (ys-info.k[5])*depths/info.k[4], depths))
        # Bound plane fitting cost without changing the full mask's identity.
        support = points[::max(1, len(points)//1200)]
        plane = fit_plane_consensus(support, iterations=32, threshold_m=0.003)
        if plane is None:
            raise ValueError('mask depth does not support a surface plane')
        translation, rotation = (camera_transform if camera_transform is not None else self._camera_transform(stamp_ns, info.header.frame_id))
        if anchor_world is None:
            u, v = seed_center or (float(xs.mean()), float(ys.mean()))
            ray = np.array([(u-info.k[2])/info.k[0], (v-info.k[5])/info.k[4], 1.])
            denominator = float(plane.normal @ ray)
            if abs(denominator) < 0.1:
                raise ValueError('target ray parallel to observed surface')
            camera_point = ray * (-plane.offset / denominator)
            anchor_world = translation + rotation @ camera_point
            support_radius = max(0.015, float(np.percentile(np.linalg.norm(points-camera_point, axis=1), 95))*1.5)
        anchor = rotation.T @ (anchor_world-translation)
        displacement = float(plane.normal @ anchor + plane.offset)
        if abs(displacement) > 0.015:
            raise ValueError('tracked surface moved from the registered target')
        nearby = np.linalg.norm(points-anchor, axis=1) <= support_radius
        if np.count_nonzero(nearby) < max(30, int(.25*len(points))):
            raise ValueError('SAM2 mask no longer supports the registered button')
        # Only the normal component comes from the new plane. Tangential center
        # stays registered, so a visible fragment cannot drag it toward an edge.
        camera_point = anchor - displacement*plane.normal
        if camera_point[2] <= 0:
            raise ValueError('registered point is behind the camera')
        normal = plane.normal
        quaternion = np.array([-normal[1], normal[0], 0., 1.+normal[2]])
        quaternion /= np.linalg.norm(quaternion)
        with self._lock:
            if generation is not None and (generation != self._generation or self._state != self.TRACKING):
                raise ValueError('obsolete geometry generation')
            self._anchor_world = anchor_world
            self._support_radius_m = support_radius
        return camera_point, quaternion

    def _publish_output(self, frame, mask, stamp_ns, camera_transform=None, generation=None, received_at=None):
        started = time.monotonic()
        ys, xs = np.nonzero(mask)
        u, v = float(np.mean(xs)), float(np.mean(ys))
        center = PointStamped()
        center.header.stamp.sec = stamp_ns // 1_000_000_000
        center.header.stamp.nanosec = stamp_ns % 1_000_000_000
        with self._lock:
            info = copy.deepcopy(self._camera_info)
            depth_stamp_ns, depth_image = min(
                self._depth_frames, key=lambda item: abs(item[0] - stamp_ns),
                default=(0, None),
            )
        center.header.frame_id = info.header.frame_id if info else ''
        center.point.x, center.point.y = u, v
        center.point.z = float(np.count_nonzero(mask))
        pose_published = False
        if info is not None and info.k[0] > 0 and info.k[4] > 0 and depth_image is not None and depth_image.shape == mask.shape and bool(self.get_parameter('use_mask_median_depth').value):
            if depth_stamp_ns > 0 and abs(stamp_ns - depth_stamp_ns) / 1e9 <= float(self.get_parameter('init_max_age_sec').value):
                try:
                    point, quaternion = self._registered_target(mask, depth_image, info, stamp_ns, camera_transform, generation)
                    pose = PoseStamped()
                    pose.header = center.header
                    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, point)
                    pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = map(float, quaternion)
                    with self._lock:
                        if self._state != self.TRACKING or (generation is not None and generation != self._generation):
                            return
                        self._pose_pub.publish(pose)
                        pose_published = True
                except Exception as error:
                    self.get_logger().warning(f'SAM2 geometry rejected: {error}', throttle_duration_sec=1.0)
        with self._lock:
            if self._state != self.TRACKING or (generation is not None and generation != self._generation):
                return
            if not pose_published:
                self._geometry_failures += 1
                self._publish_state('geometry_invalid')
                return
            self._geometry_failures = 0
            self._output_stamp_ns = stamp_ns
            if hasattr(self, '_geometry_samples'):
                now = time.monotonic()
                self._geometry_samples.append((
                    (now-started)*1000,
                    (now-received_at)*1000 if received_at is not None else float('nan'),
                    (self.get_clock().now().nanoseconds-stamp_ns)/1e6,
                ))
            if self._stable_count >= int(self.get_parameter('stable_frames').value):
                self._publish_state('tracker_ready')

    def _queue_debug(self, frame, mask, header, u, v):
        if not bool(self.get_parameter('debug_image').value):
            return
        if (bool(self.get_parameter('debug_image_only_when_subscribed').value)
                and self._debug_pub.get_subscription_count() == 0):
            return
        # Display is independent of inference/geometry. Replace an unrendered
        # frame instead of blocking control or building a backlog for rqt.
        with self._debug_condition:
            self._debug_pending = (frame, mask, header, u, v, self._generation)
            self._debug_condition.notify()

    @staticmethod
    def _render_debug(frame, mask, u, v):
        debug = frame.copy()
        debug[mask] = (0.5 * debug[mask] + np.array([0, 180, 0]) * 0.5).astype(np.uint8)
        x, y, width, height = cv2.boundingRect(mask.astype(np.uint8))
        cv2.rectangle(debug, (x, y), (x + width, y + height), (255, 180, 0), 2)
        cv2.circle(debug, (int(round(u)), int(round(v))), 5, (0, 0, 255), -1)
        cv2.drawMarker(debug, (debug.shape[1] // 2, debug.shape[0] // 2), (255, 255, 255), cv2.MARKER_CROSS, 14, 1)
        cv2.putText(debug, f'TRACKING ({u:.0f},{v:.0f})', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return debug

    def _debug_loop(self):
        last_publish = 0.0
        while not self._stop.is_set():
            fps = float(self.get_parameter('debug_max_fps').value)
            if fps > 0 and self._stop.wait(max(0.0, 1.0/fps - (time.monotonic()-last_publish))):
                break
            with self._debug_condition:
                self._debug_condition.wait_for(
                    lambda: self._stop.is_set() or self._debug_pending is not None,
                )
                if self._stop.is_set():
                    break
                frame, mask, header, u, v, generation = self._debug_pending
                self._debug_pending = None
            if not bool(self.get_parameter('debug_image').value):
                continue
            debug = self._render_debug(frame, mask, u, v)
            message = self._bridge.cv2_to_imgmsg(debug, encoding='bgr8')
            message.header = header
            if generation != self._generation or self._state != self.TRACKING:
                continue
            self._debug_pub.publish(message)
            self._performance_debug_count += 1
            last_publish = time.monotonic()

    def _record_performance(self, stamp_ns, received_at, inference_seconds, total_seconds):
        now = time.monotonic()
        age = (self.get_clock().now().nanoseconds-stamp_ns)/1e9
        self._performance_samples.append((
            inference_seconds, total_seconds, now-received_at, age,
        ))
        interval = float(self.get_parameter('performance_log_interval_seconds').value)
        elapsed = now-self._performance_started_at
        if elapsed < max(1.0, interval):
            return
        values = np.asarray(self._performance_samples)
        with self._lock:
            geometry = np.asarray(self._geometry_samples)
            self._geometry_samples.clear()
        payload = dict(
            mask_fps=len(values)/elapsed, inference_fps=len(values)/elapsed,
            geometry_fps=len(geometry)/elapsed, surface_fps=len(geometry)/elapsed,
            geometry_ms=float(geometry[:, 0].mean()) if len(geometry) else None,
            target_latency_ms=float(geometry[:, 1].mean()) if len(geometry) else None,
            target_age_ros_ms=float(geometry[:, 2].mean()) if len(geometry) else None,
            debug_fps=self._performance_debug_count/elapsed,
            inference_ms=float(values[:, 0].mean()*1000),
            mask_publish_ms=float((values[:, 1]-values[:, 0]).mean()*1000),
            receipt_to_output_ms=float(values[:, 2].mean()*1000),
            capture_age_ros_ms=float(values[:, 3].mean()*1000),
            capture_stamp_ns=stamp_ns, window_seconds=elapsed,
        )
        self._performance_pub.publish(String(data=json.dumps(payload)))
        if interval > 0:
            self.get_logger().info(
                f'SAM2 performance: surface={payload["surface_fps"]:.1f} FPS, '
                f'inference={payload["inference_ms"]:.1f} ms, '
                f'mask/publish={payload["mask_publish_ms"]:.1f} ms, '
                f'receipt-to-output={payload["receipt_to_output_ms"]:.1f} ms'
            )
        self._performance_samples.clear()
        self._performance_debug_count = 0
        self._performance_started_at = now

    def destroy_node(self):
        self._stop.set()
        self._frame_event.set()
        with self._debug_condition:
            self._debug_condition.notify_all()
        with self._geometry_condition:
            self._geometry_condition.notify_all()
        if self._debug_worker.is_alive():
            self._debug_worker.join(timeout=1.0)
        if self._geometry_worker.is_alive():
            self._geometry_worker.join(timeout=1.0)
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Sam2ButtonTracker()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
