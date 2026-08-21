"""Model inference and temporal filtering for elevator button detection."""

import ast

from dataclasses import dataclass, replace
from math import hypot
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime


@dataclass(frozen=True)
class Detection:
    """One model detection in source-image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> Tuple[float, float]:
        """Return the bounding-box center."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        """Return the bounding-box width."""
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Return the bounding-box height."""
        return max(0.0, self.y2 - self.y1)

    def translated(self, dx: float, dy: float) -> 'Detection':
        """Return a detection shifted by the supplied pixel offset."""
        return replace(
            self,
            x1=self.x1 + dx,
            y1=self.y1 + dy,
            x2=self.x2 + dx,
            y2=self.y2 + dy,
        )


def filter_detections_by_class(
    detections: Sequence[Detection],
    selected_class: str,
) -> List[Detection]:
    """Return only detections matching the operator-selected class."""
    normalized = str(selected_class).strip().casefold()
    if not normalized:
        return []
    return [
        detection
        for detection in detections
        if detection.class_name.strip().casefold() == normalized
    ]


def intersection_over_union(first: Detection, second: Detection) -> float:
    """Calculate intersection-over-union for two detections."""
    left = max(first.x1, second.x1)
    top = max(first.y1, second.y1)
    right = min(first.x2, second.x2)
    bottom = min(first.y2, second.y2)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = (
        first.width * first.height
        + second.width * second.height
        - intersection
    )
    if union <= 0.0:
        return 0.0
    return intersection / union


class YoloOnnxDetector:
    """Run raw YOLOv8 or end-to-end YOLOv10 ONNX detection models."""

    def __init__(
        self,
        model_path: Optional[str],
        class_names: Sequence[str],
        target_classes: Sequence[str],
        confidence_threshold: float = 0.55,
        nms_iou_threshold: float = 0.45,
        input_size: int = 640,
        inference_device: str = 'cuda',
        session=None,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.input_size = int(input_size)
        if self.input_size <= 0:
            raise ValueError('input_size must be positive')
        if session is not None:
            self.session = session
            get_providers = getattr(session, 'get_providers', None)
            self.active_providers = (
                list(get_providers()) if get_providers else ['TestProvider']
            )
        else:
            if not model_path:
                raise ValueError('model_path must not be empty')
            path = Path(model_path)
            if not path.is_file():
                raise FileNotFoundError(f'ONNX model not found: {path}')
            options = onnxruntime.SessionOptions()
            options.graph_optimization_level = (
                onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            providers = self._select_providers(inference_device)
            self.session = onnxruntime.InferenceSession(
                str(path),
                sess_options=options,
                providers=providers,
            )
            self.active_providers = list(self.session.get_providers())
            if (
                str(inference_device).strip().casefold() == 'cuda'
                and 'CUDAExecutionProvider' not in self.active_providers
            ):
                raise RuntimeError(
                    'CUDA was requested but the ONNX session did not load '
                    f'the CUDA provider: {self.active_providers}'
                )
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(
                f'Expected one ONNX input, received {len(inputs)}'
            )
        self.input_name = inputs[0].name
        self.class_names = list(class_names)
        if self.class_names == ['__model_metadata__']:
            self.class_names = self._class_names_from_metadata()
        if not self.class_names:
            raise ValueError('class_names must not be empty')
        self.target_classes = {
            str(name).casefold() for name in target_classes
        }
        self._allow_all_classes = '*' in self.target_classes

        input_shape = getattr(inputs[0], 'shape', None)
        if (
            isinstance(input_shape, (list, tuple))
            and len(input_shape) == 4
            and isinstance(input_shape[2], int)
            and isinstance(input_shape[3], int)
            and (
                input_shape[2] != self.input_size
                or input_shape[3] != self.input_size
            )
        ):
            raise ValueError(
                'model_input_size does not match ONNX input shape: '
                f'configured={self.input_size}, model={input_shape[2:]}'
            )

    @staticmethod
    def _select_providers(inference_device: str):
        """Choose CUDA without silently falling back when required."""
        device = str(inference_device).strip().casefold()
        if device not in {'cuda', 'cpu', 'auto'}:
            raise ValueError(
                "inference_device must be one of: 'cuda', 'cpu', 'auto'"
            )
        available = set(onnxruntime.get_available_providers())
        if device == 'cuda' and 'CUDAExecutionProvider' not in available:
            raise RuntimeError(
                'CUDA inference was requested, but CUDAExecutionProvider is '
                f'not available. Installed providers: {sorted(available)}'
            )
        if device == 'cpu' or (
            device == 'auto' and 'CUDAExecutionProvider' not in available
        ):
            return ['CPUExecutionProvider']
        return [
            (
                'CUDAExecutionProvider',
                {
                    'device_id': 0,
                    'cudnn_conv_algo_search': 'HEURISTIC',
                    'do_copy_in_default_stream': '1',
                },
            ),
            'CPUExecutionProvider',
        ]

    def _class_names_from_metadata(self) -> List[str]:
        """Read the Ultralytics class-name dictionary from ONNX metadata."""
        get_modelmeta = getattr(self.session, 'get_modelmeta', None)
        if get_modelmeta is None:
            raise ValueError('ONNX session does not expose model metadata')
        metadata = get_modelmeta().custom_metadata_map
        encoded_names = metadata.get('names', '')
        try:
            mapping = ast.literal_eval(encoded_names)
        except (SyntaxError, ValueError) as error:
            raise ValueError('Invalid ONNX names metadata') from error
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError('ONNX names metadata must be a non-empty dict')
        indexed = {int(key): str(value) for key, value in mapping.items()}
        expected = list(range(max(indexed) + 1))
        if sorted(indexed) != expected:
            raise ValueError('ONNX names metadata indices are not contiguous')
        return [indexed[index] for index in expected]

    def _is_target_class(self, class_name: str) -> bool:
        return (
            self._allow_all_classes
            or not self.target_classes
            or class_name.casefold() in self.target_classes
        )

    def infer(self, image: np.ndarray) -> List[Detection]:
        """Return filtered model detections in source-image coordinates."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Expected a BGR image with three channels')

        prepared, scale, pad_x, pad_y = self._letterbox(image)
        blob = cv2.dnn.blobFromImage(
            prepared,
            scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            swapRB=True,
            crop=False,
        )
        outputs = self.session.run(None, {self.input_name: blob})
        if not outputs:
            return []
        output = outputs[0]
        return self._decode(
            np.asarray(output),
            image.shape[1],
            image.shape[0],
            scale,
            pad_x,
            pad_y,
        )

    def warmup(self, iterations: int = 2) -> None:
        """Run dummy frames before subscribing to remove first-frame stalls."""
        dummy = np.full(
            (self.input_size, self.input_size, 3),
            114,
            dtype=np.uint8,
        )
        for _ in range(max(0, int(iterations))):
            self.infer(dummy)

    def _letterbox(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, float, int, int]:
        height, width = image.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        pad_x = (self.input_size - resized_width) // 2
        pad_y = (self.input_size - resized_height) // 2
        prepared = np.full(
            (self.input_size, self.input_size, 3),
            114,
            dtype=np.uint8,
        )
        prepared[
            pad_y:pad_y + resized_height,
            pad_x:pad_x + resized_width,
        ] = resized
        return prepared, scale, pad_x, pad_y

    def _decode(
        self,
        output: np.ndarray,
        source_width: int,
        source_height: int,
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> List[Detection]:
        predictions = np.squeeze(output)
        if predictions.ndim != 2:
            raise ValueError(
                f'Unsupported YOLO output shape: {output.shape}'
            )

        expected_features = 4 + len(self.class_names)
        if predictions.shape[1] == 6 and expected_features != 6:
            return self._decode_end_to_end(
                predictions,
                source_width,
                source_height,
                scale,
                pad_x,
                pad_y,
            )
        if predictions.shape[0] in (
            expected_features,
            expected_features + 1,
        ):
            predictions = predictions.T
        if predictions.shape[1] not in (
            expected_features,
            expected_features + 1,
        ):
            raise ValueError(
                'Model class count does not match class_names: '
                f'output={output.shape}, names={len(self.class_names)}'
            )

        has_objectness = predictions.shape[1] == expected_features + 1
        boxes = []
        scores = []
        class_ids = []
        for prediction in predictions:
            class_offset = 5 if has_objectness else 4
            class_scores = prediction[class_offset:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            if has_objectness:
                score *= float(prediction[4])
            if score < self.confidence_threshold:
                continue

            class_name = self.class_names[class_id]
            if not self._is_target_class(class_name):
                continue

            center_x, center_y, width, height = map(
                float,
                prediction[:4],
            )
            x1 = (center_x - width / 2.0 - pad_x) / scale
            y1 = (center_y - height / 2.0 - pad_y) / scale
            x2 = (center_x + width / 2.0 - pad_x) / scale
            y2 = (center_y + height / 2.0 - pad_y) / scale
            x1 = float(np.clip(x1, 0.0, source_width - 1.0))
            y1 = float(np.clip(y1, 0.0, source_height - 1.0))
            x2 = float(np.clip(x2, 0.0, source_width - 1.0))
            y2 = float(np.clip(y2, 0.0, source_height - 1.0))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)
            class_ids.append(class_id)

        return self._apply_nms(boxes, scores, class_ids)

    def _decode_end_to_end(
        self,
        predictions: np.ndarray,
        source_width: int,
        source_height: int,
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> List[Detection]:
        """Decode YOLOv10 rows formatted as x1, y1, x2, y2, score, class."""
        boxes = []
        scores = []
        class_ids = []
        for prediction in predictions:
            score = float(prediction[4])
            if score < self.confidence_threshold:
                continue
            class_id = int(round(float(prediction[5])))
            if class_id < 0 or class_id >= len(self.class_names):
                continue
            class_name = self.class_names[class_id]
            if not self._is_target_class(class_name):
                continue

            x1, y1, x2, y2 = map(float, prediction[:4])
            x1 = float(
                np.clip((x1 - pad_x) / scale, 0.0, source_width - 1.0)
            )
            y1 = float(
                np.clip((y1 - pad_y) / scale, 0.0, source_height - 1.0)
            )
            x2 = float(
                np.clip((x2 - pad_x) / scale, 0.0, source_width - 1.0)
            )
            y2 = float(
                np.clip((y2 - pad_y) / scale, 0.0, source_height - 1.0)
            )
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)
            class_ids.append(class_id)
        return self._apply_nms(boxes, scores, class_ids)

    def _apply_nms(
        self,
        boxes: Sequence[Sequence[float]],
        scores: Sequence[float],
        class_ids: Sequence[int],
    ) -> List[Detection]:
        """Apply a final class-agnostic NMS pass and build detections."""
        if not boxes:
            return []
        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.confidence_threshold,
            self.nms_iou_threshold,
        )
        selected = np.asarray(indices).reshape(-1).tolist()
        detections = []
        for index in selected:
            x, y, width, height = boxes[index]
            class_id = class_ids[index]
            detections.append(
                Detection(
                    x1=x,
                    y1=y,
                    x2=x + width,
                    y2=y + height,
                    confidence=scores[index],
                    class_id=class_id,
                    class_name=self.class_names[class_id],
                )
            )
        return sorted(
            detections,
            key=lambda detection: detection.confidence,
            reverse=True,
        )


class TemporalButtonTracker:
    """Associate and smooth one button across consecutive frames."""

    def __init__(
        self,
        required_stable_frames: int = 5,
        max_missed_frames: int = 2,
        minimum_iou: float = 0.15,
        max_center_jump_ratio: float = 0.10,
        smoothing_alpha: float = 0.35,
    ) -> None:
        self.required_stable_frames = max(1, required_stable_frames)
        self.max_missed_frames = max(0, max_missed_frames)
        self.minimum_iou = float(minimum_iou)
        self.max_center_jump_ratio = float(max_center_jump_ratio)
        self.smoothing_alpha = float(
            np.clip(smoothing_alpha, 0.0, 1.0)
        )
        self.reset()

    def reset(self) -> None:
        """Discard the active track."""
        self.current: Optional[Detection] = None
        self.stable_frames = 0
        self.missed_frames = 0

    def update(
        self,
        detections: Sequence[Detection],
        image_width: int,
        image_height: int,
    ) -> Tuple[Optional[Detection], bool]:
        """Update the active track and return it with its validity state."""
        matched = self._select_match(
            detections,
            image_width,
            image_height,
        )
        if matched is None:
            self.missed_frames += 1
            if self.missed_frames > self.max_missed_frames:
                self.reset()
            return None, False

        if self.current is None:
            self.current = matched
            self.stable_frames = 1
        else:
            self.current = self._smooth(self.current, matched)
            self.stable_frames += 1
        self.missed_frames = 0
        return (
            self.current,
            self.stable_frames >= self.required_stable_frames,
        )

    def _select_match(
        self,
        detections: Sequence[Detection],
        image_width: int,
        image_height: int,
    ) -> Optional[Detection]:
        if not detections:
            return None
        if self.current is None:
            return max(detections, key=lambda item: item.confidence)

        diagonal = max(hypot(image_width, image_height), 1.0)
        matches = []
        current_x, current_y = self.current.center
        for candidate in detections:
            if candidate.class_name != self.current.class_name:
                continue
            candidate_x, candidate_y = candidate.center
            distance_ratio = hypot(
                candidate_x - current_x,
                candidate_y - current_y,
            ) / diagonal
            overlap = intersection_over_union(self.current, candidate)
            if (
                overlap >= self.minimum_iou
                or distance_ratio <= self.max_center_jump_ratio
            ):
                matches.append((overlap, candidate.confidence, candidate))
        if not matches:
            self.reset()
            return max(detections, key=lambda item: item.confidence)
        return max(matches, key=lambda item: (item[0], item[1]))[2]

    def _smooth(
        self,
        previous: Detection,
        current: Detection,
    ) -> Detection:
        alpha = self.smoothing_alpha

        def blend(old: float, new: float) -> float:
            return (1.0 - alpha) * old + alpha * new

        return Detection(
            x1=blend(previous.x1, current.x1),
            y1=blend(previous.y1, current.y1),
            x2=blend(previous.x2, current.x2),
            y2=blend(previous.y2, current.y2),
            confidence=blend(
                previous.confidence,
                current.confidence,
            ),
            class_id=current.class_id,
            class_name=current.class_name,
        )


def robust_box_depth(
    depth_image: np.ndarray,
    detection: Detection,
    unit_scale: float,
    inner_ratio: float,
    min_depth_m: float,
    max_depth_m: float,
    min_samples: int,
) -> Optional[float]:
    """Estimate depth from the central box region using median and MAD."""
    center_x, center_y = detection.center
    ratio = float(np.clip(inner_ratio, 0.05, 1.0))
    half_width = max(1.0, detection.width * ratio / 2.0)
    half_height = max(1.0, detection.height * ratio / 2.0)
    image_height, image_width = depth_image.shape[:2]
    x0 = max(0, int(round(center_x - half_width)))
    x1 = min(image_width, int(round(center_x + half_width + 1)))
    y0 = max(0, int(round(center_y - half_height)))
    y1 = min(image_height, int(round(center_y + half_height + 1)))
    samples = np.asarray(depth_image[y0:y1, x0:x1], dtype=np.float64)
    if np.issubdtype(depth_image.dtype, np.integer):
        samples *= float(unit_scale)
    valid = samples[
        np.isfinite(samples)
        & (samples >= min_depth_m)
        & (samples <= max_depth_m)
    ]
    if valid.size < max(1, min_samples):
        return None

    median = float(np.median(valid))
    absolute_deviation = np.abs(valid - median)
    mad = float(np.median(absolute_deviation))
    tolerance = max(3.0 * 1.4826 * mad, 0.01)
    inliers = valid[absolute_deviation <= tolerance]
    if inliers.size < max(1, min_samples):
        return None
    return float(np.median(inliers))


def project_pixel(
    camera_matrix: np.ndarray,
    u: float,
    v: float,
    depth_m: float,
    distortion_coefficients: Optional[np.ndarray] = None,
    distortion_model: str = '',
) -> np.ndarray:
    """Project a possibly distorted color pixel into camera coordinates."""
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    normalized_x = (u - cx) / fx
    normalized_y = (v - cy) / fy
    coefficients = np.asarray(
        distortion_coefficients
        if distortion_coefficients is not None
        else [],
        dtype=np.float64,
    ).reshape(-1)
    model = str(distortion_model).strip().casefold()
    if coefficients.size > 0 and np.any(np.abs(coefficients) > 1e-12):
        pixel = np.asarray([[[u, v]]], dtype=np.float64)
        if model in {'plumb_bob', 'rational_polynomial'}:
            normalized = cv2.undistortPoints(
                pixel,
                camera_matrix,
                coefficients,
            ).reshape(2)
        elif model in {'equidistant', 'fisheye'}:
            if coefficients.size < 4:
                raise ValueError(
                    'Fisheye projection requires four coefficients'
                )
            normalized = cv2.fisheye.undistortPoints(
                pixel,
                camera_matrix,
                coefficients[:4],
            ).reshape(2)
        else:
            raise ValueError(
                f'Unsupported camera distortion model: {distortion_model}'
            )
        normalized_x = float(normalized[0])
        normalized_y = float(normalized[1])
    return np.asarray(
        [
            normalized_x * depth_m,
            normalized_y * depth_m,
            depth_m,
        ],
        dtype=np.float64,
    )
