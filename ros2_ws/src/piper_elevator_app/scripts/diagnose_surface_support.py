#!/usr/bin/env python3
"""Read-only capture and offline comparison of RGB-D surface support regions."""

import argparse
from collections import OrderedDict
import json
import math
from pathlib import Path
import time

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_ros import TransformException
from vision_msgs.msg import Detection2DArray

from diagnose_rgbd import Diagnostics, quaternion, stats
from piper_elevator_app.coarse_approach_core import stable_surface_normal
from piper_elevator_app.detector_core import Detection, estimate_surface_normal
from piper_elevator_app.motion_core import quaternion_to_matrix


class SurfaceCapture(Diagnostics):
    def __init__(self):
        super().__init__()
        self.bridge = CvBridge()
        self.depth_cache = OrderedDict()
        self.pending_boxes = OrderedDict()
        self.paired = []
        self.color = None
        self.color_stamp_ns = None
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.depth_subscription = self.create_subscription(
            Image, '/camera/aligned_depth_to_color/image_raw', self.depth_callback, qos,
        )
        self.color_subscription = self.create_subscription(
            Image, '/camera/color/image_raw', self.color_callback, qos,
        )
        self.raw_detection_subscription = self.create_subscription(
            Detection2DArray, '/button_detections/raw',
            lambda message: self.receive('/button_detections/raw', message), qos,
        )

    def color_callback(self, message):
        if self.color is None and self.recording:
            self.color = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8').copy()
            self.color_stamp_ns = Time.from_msg(message.header.stamp).nanoseconds
            self.destroy_subscription(self.color_subscription)
            self.color_subscription = None

    def depth_callback(self, message):
        if not self.recording:
            return
        stamp = Time.from_msg(message.header.stamp).nanoseconds
        self.depth_cache[stamp] = message
        while len(self.depth_cache) > 120:
            self.depth_cache.popitem(last=False)
        self.match()

    def receive(self, topic, message):
        super().receive(topic, message)
        if not self.recording or topic != '/button_detections/raw':
            return
        stamp = Time.from_msg(message.header.stamp).nanoseconds
        all_boxes = [
            [
                detection.bbox.center.position.x - detection.bbox.size_x / 2.0,
                detection.bbox.center.position.y - detection.bbox.size_y / 2.0,
                detection.bbox.center.position.x + detection.bbox.size_x / 2.0,
                detection.bbox.center.position.y + detection.bbox.size_y / 2.0,
            ]
            for detection in message.detections
        ]
        for detection in message.detections:
            labels = [result.hypothesis.class_id for result in detection.results]
            if self.selected in labels:
                self.pending_boxes[stamp] = (detection, all_boxes)
                break
        self.match()

    def match(self):
        for stamp in list(self.pending_boxes):
            if stamp not in self.depth_cache:
                continue
            detection, all_boxes = self.pending_boxes.pop(stamp)
            depth_message = self.depth_cache.pop(stamp)
            depth = self.bridge.imgmsg_to_cv2(depth_message, desired_encoding='passthrough').copy()
            self.paired.append((stamp, depth, detection, all_boxes))
        while len(self.pending_boxes) > 120:
            self.pending_boxes.popitem(last=False)

    def collect(self, target, timeout):
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        started = time.monotonic()
        self.recording = True
        while len(self.paired) < target and time.monotonic() - started < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
        self.recording = False
        return time.monotonic() - started


def angle_degrees(first, second):
    return math.degrees(math.acos(float(np.clip(np.dot(first, second), -1, 1))))


def read_detector_parameters(node):
    names = [
        'depth_unit_scale', 'min_depth_m', 'max_depth_m',
        'surface_minimum_samples', 'surface_maximum_samples',
        'surface_max_residual_m', 'surface_max_tilt_degrees', 'surface_inner_ratio',
    ]
    for _ in range(3):
        result = node.parameters('/button_detector', names)
        if 'error' not in result and all(result.get(name) is not None for name in names):
            return result
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.1)
    raise RuntimeError('Detector parameters unavailable: ' + str(result))


def evaluate(depths, boxes, rotations, metadata):
    camera = metadata['camera_info']
    detector = metadata['detector_parameters']
    k = np.asarray(camera['k']).reshape(3, 3)
    d = np.asarray(camera['d'])
    variants = [(ratio, 400) for ratio in (0.45, 0.7, 1.0, 1.4, 2.0)] + [(0.7, 10000)]
    results = {}
    averages = {}
    series = {}
    for ratio, max_samples in variants:
        normals = []
        valid_percentages = []
        spreads = []
        roi_sizes = []
        strides = []
        for depth, box, rotation in zip(depths, boxes, rotations):
            u, v, width, height = box
            detection = Detection(
                u - width * ratio / 2, v - height * ratio / 2,
                u + width * ratio / 2, v + height * ratio / 2,
                1.0, 0, metadata['selected_button'],
            )
            normal = estimate_surface_normal(
                depth, detection, k,
                unit_scale=detector['depth_unit_scale'], inner_ratio=1.0,
                min_depth_m=detector['min_depth_m'], max_depth_m=detector['max_depth_m'],
                min_samples=detector['surface_minimum_samples'], max_samples=max_samples,
                max_residual_m=detector['surface_max_residual_m'],
                max_tilt_degrees=detector['surface_max_tilt_degrees'],
                distortion_coefficients=d, distortion_model=camera['distortion_model'],
            )
            normals.append(rotation @ normal if normal is not None else None)
            x0 = max(0, int(round(detection.x1)))
            x1 = min(depth.shape[1], int(round(detection.x2 + 1)))
            y0 = max(0, int(round(detection.y1)))
            y1 = min(depth.shape[0], int(round(detection.y2 + 1)))
            patch = depth[y0:y1, x0:x1].astype(float)
            if np.issubdtype(depth.dtype, np.integer):
                patch *= detector['depth_unit_scale']
            valid = np.isfinite(patch) & (patch >= detector['min_depth_m']) & (patch <= detector['max_depth_m'])
            valid_percentages.append(100 * np.mean(valid))
            if np.any(valid):
                spreads.append(1000 * (np.percentile(patch[valid], 95) - np.percentile(patch[valid], 5)))
            roi_sizes.append([x1 - x0, y1 - y0])
            strides.append(max(1, int(np.ceil(np.sqrt(patch.size / max_samples)))))
        key = f'roi_{ratio:g}_max_samples_{max_samples}'
        series[key] = normals
        good = np.asarray([normal for normal in normals if normal is not None])
        result = {
            'valid_fits': len(good), 'total_frames': len(depths),
            'roi_size_px_median': np.median(roi_sizes, axis=0).tolist(),
            'sampling_stride_px': stats(strides),
            'depth_valid_percentage': stats(valid_percentages),
            'depth_p95_minus_p05_mm': stats(spreads),
            'stable_windows_5deg': {},
        }
        if len(good):
            mean = good.mean(axis=0)
            mean /= np.linalg.norm(mean)
            averages[key] = mean
            result['mean_base_normal_unit'] = mean.tolist()
            result['angle_from_mean_degrees'] = stats([angle_degrees(normal, mean) for normal in good])
            result['frame_step_degrees'] = stats([
                angle_degrees(first, second) for first, second in zip(normals, normals[1:])
                if first is not None and second is not None
            ])
        for length in (8, 12, 16, 20, 30, 40, 50):
            successes = 0
            tested = 0
            missing = 0
            uncertainties = []
            trends = []
            spans = []
            for end in range(length, len(normals) + 1):
                window = normals[end - length:end]
                tested += 1
                stamps = metadata['capture_stamps_ns']
                spans.append((stamps[end - 1] - stamps[end - length]) / 1e9)
                if any(normal is None for normal in window):
                    missing += 1
                    continue
                stable, _ = stable_surface_normal(window, math.radians(5))
                successes += stable is not None
                values = np.asarray(window)
                mean = values.mean(axis=0)
                mean /= np.linalg.norm(mean)
                angles = np.arccos(np.clip(values @ mean, -1, 1))
                uncertainties.append(math.degrees(2 * np.sqrt(np.sum(angles ** 2)) / length))
                halves = [part.mean(axis=0) for part in np.array_split(values, 2)]
                halves = [part / np.linalg.norm(part) for part in halves]
                trends.append(angle_degrees(*halves))
            result['stable_windows_5deg'][str(length)] = {
                'tested': tested, 'passed': successes,
                'pass_fraction': successes / tested if tested else None,
                'missing_fit_windows': missing,
                'capture_span_seconds': stats(spans),
                'normal_uncertainty_degrees': stats(uncertainties),
                'window_change_degrees': stats(trends),
            }
        results[key] = result
    offsets = {
        first: {second: angle_degrees(mean, other) for second, other in averages.items()}
        for first, mean in averages.items()
    }
    return {
        'metadata': metadata, 'variants': results,
        'mean_normal_offsets_degrees': offsets,
        'note': 'Offline normal-only windows at fixed captured robot pose; broader regions may fit a different surface. These results do not establish ground-truth normal accuracy.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=80)
    parser.add_argument('--timeout', type=float, default=20)
    parser.add_argument(
        '--output-prefix',
        default=Path(__file__).resolve().parents[3] / 'diagnostics' / 'data' / 'surface_support_diagnostic',
    )
    parser.add_argument('--replay', help='Replay a previously captured NPZ without creating a ROS node')
    parser.add_argument('--refresh-parameters', action='store_true',
                        help='Read current detector parameters before replaying the archive')
    args = parser.parse_args()
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if args.replay:
        with np.load(args.replay, allow_pickle=False) as archive:
            saved = {name: archive[name] for name in archive.files}
        metadata = json.loads(str(saved['metadata_json']))
        if args.refresh_parameters:
            rclpy.init()
            node = Diagnostics()
            try:
                metadata['detector_parameters'] = read_detector_parameters(node)
                metadata['parameters_refreshed_after_capture'] = True
            finally:
                node.destroy_node()
                rclpy.shutdown()
            saved['metadata_json'] = json.dumps(metadata)
            np.savez_compressed(str(prefix) + '.npz', **saved)
        result = evaluate(saved['depths'], saved['boxes'], saved['rotations'], metadata)
    else:
        rclpy.init()
        node = SurfaceCapture()
        try:
            detector = read_detector_parameters(node)
            elapsed = node.collect(args.frames, args.timeout)
            camera = node.camera_info.get('/camera/color/camera_info')
            if camera is None or not node.paired:
                raise RuntimeError('No camera info or no timestamp-matched depth/detection frames')
            depths, boxes, rotations, stamps = [], [], [], []
            all_boxes, translations = [], []
            errors = []
            for stamp, depth, detection, frame_boxes in node.paired:
                try:
                    transform = node.tf_buffer.lookup_transform(
                        'base_link', camera['frame_id'], Time(nanoseconds=stamp),
                    ).transform
                except TransformException as error:
                    errors.append(str(error))
                    continue
                stamps.append(stamp)
                depths.append(depth)
                boxes.append([
                    detection.bbox.center.position.x, detection.bbox.center.position.y,
                    detection.bbox.size_x, detection.bbox.size_y,
                ])
                rotations.append(quaternion_to_matrix(quaternion(transform.rotation)))
                all_boxes.append(frame_boxes)
                translations.append([
                    transform.translation.x, transform.translation.y, transform.translation.z,
                ])
            metadata = {
                'read_only': True, 'capture_seconds': elapsed,
                'matched_frames': len(depths), 'matching': 'exact color/detection and depth stamp',
                'selected_button': node.selected,
                'detector_parameters': detector, 'camera_info': camera,
                'tf_errors': errors, 'color_stamp_ns': node.color_stamp_ns,
                'capture_stamp_interval_seconds': stats(np.diff(stamps) / 1e9),
                'capture_stamps_ns': stamps,
            }
            np.savez_compressed(
                str(prefix) + '.npz', depths=np.asarray(depths), boxes=np.asarray(boxes),
                rotations=np.asarray(rotations), stamps=np.asarray(stamps, dtype=np.int64),
                translations=np.asarray(translations, dtype=np.float64).reshape(-1, 3),
                all_boxes_json=json.dumps(all_boxes),
                metadata_json=json.dumps(metadata),
            )
            if node.color is not None:
                cv2.imwrite(str(prefix) + '_color.png', node.color)
        finally:
            node.destroy_node()
            rclpy.shutdown()
        result = evaluate(depths, boxes, rotations, metadata)
    output = json.dumps(result, indent=2)
    Path(str(prefix) + '.json').write_text(output + '\n', encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
