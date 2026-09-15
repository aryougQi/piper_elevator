#!/usr/bin/env python3
"""Compare fresh home RGB-D geometry with preserved pre/post-approach points."""

import argparse
from collections import Counter, OrderedDict
import json
from pathlib import Path
import time

from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection2DArray

from piper_elevator_app.candidate_validation import (
    deduplicate_candidates, filter_by_physical_size,
)
from piper_elevator_app.detector_core import Detection, project_pixel, robust_box_depth
from piper_elevator_app.motion_core import quaternion_to_matrix


DATA = Path(__file__).resolve().parents[1] / 'data'


def stamp(message):
    return Time.from_msg(message.header.stamp).nanoseconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=8.0)
    parser.add_argument('--label', default='up')
    parser.add_argument('--baseline', type=Path,
                        default=DATA / 'coarse_detection_loss/projection_geometry.json')
    parser.add_argument('--output', type=Path, default=DATA / 'handeye_home_return_check.json')
    args = parser.parse_args()
    if not 0.0 < args.seconds <= 30.0:
        parser.error('seconds must be within (0, 30]')
    if args.output.exists():
        parser.error('output already exists; use a new path to preserve the capture')
    baseline = json.loads(args.baseline.read_text())
    planned = np.asarray(baseline['planned_button'])
    coarse = np.median([row['measured_base'] for row in baseline['samples']], axis=0)
    rclpy.init()
    node = Node('handeye_home_return_readonly')
    buffer = Buffer(cache_time=Duration(seconds=30.0))
    listener = TransformListener(buffer, node)
    bridge = CvBridge()
    depths, colors, detections = OrderedDict(), OrderedDict(), OrderedDict()
    camera, joints, counts = {}, [], Counter()
    subscriptions = []
    qos = QoSProfile(depth=3, reliability=ReliabilityPolicy.BEST_EFFORT)

    def cache(store, key, value, maximum=240):
        store[key] = value
        while len(store) > maximum:
            store.popitem(last=False)

    def depth_callback(message):
        counts['depth_frames'] += 1
        cache(depths, stamp(message), message)

    def color_callback(message):
        counts['color_frames'] += 1
        cache(colors, stamp(message), {
            'frame_id': message.header.frame_id,
            'width': message.width, 'height': message.height,
            'encoding': message.encoding,
        })

    def detections_callback(message):
        counts['detection_frames'] += 1
        cache(detections, stamp(message), message)

    def camera_callback(message):
        camera.update(
            k=list(message.k), d=list(message.d), model=message.distortion_model,
            frame=message.header.frame_id, width=message.width, height=message.height,
        )

    def joint_callback(message):
        by_name = dict(zip(message.name, message.position))
        if all('joint' + str(index) in by_name for index in range(1, 7)):
            joints.append({
                'stamp_ns': stamp(message),
                'position': [by_name['joint' + str(index)] for index in range(1, 7)],
            })

    for kind, topic, callback in (
        (Image, '/camera/aligned_depth_to_color/image_raw', depth_callback),
        (Image, '/camera/color/image_raw', color_callback),
        (Detection2DArray, '/button_detections/raw', detections_callback),
        (CameraInfo, '/camera/color/camera_info', camera_callback),
        (JointState, '/feedback/joint_states', joint_callback),
    ):
        subscriptions.append(node.create_subscription(kind, topic, callback, qos))
    started = time.monotonic()
    try:
        while time.monotonic() - started < args.seconds:
            rclpy.spin_once(node, timeout_sec=0.02)
        if not camera:
            raise RuntimeError('No CameraInfo received')
        matrix = np.asarray(camera['k']).reshape(3, 3)
        distortion = np.asarray(camera['d'])
        rows, ambiguous, failures = [], [], []
        for capture, message in detections.items():
            if capture not in depths or capture not in colors:
                counts['missing_exact_rgbd_pair'] += 1
                continue
            color = colors[capture]
            depth_message = depths[capture]
            if (message.header.frame_id != camera['frame']
                    or depth_message.header.frame_id != camera['frame']
                    or color['frame_id'] != camera['frame']
                    or (color['height'], color['width']) != (camera['height'], camera['width'])
                    or (depth_message.height, depth_message.width) != (camera['height'], camera['width'])):
                counts['rgbd_contract_mismatch'] += 1
                continue
            depth = bridge.imgmsg_to_cv2(depth_message, 'passthrough')
            candidates = []
            for item in message.detections:
                if not item.results:
                    continue
                hypothesis = item.results[0].hypothesis
                if hypothesis.class_id.strip().casefold() != args.label.casefold():
                    continue
                u, v = item.bbox.center.position.x, item.bbox.center.position.y
                w, h = item.bbox.size_x, item.bbox.size_y
                candidates.append(Detection(
                    u - w / 2, v - h / 2, u + w / 2, v + h / 2,
                    hypothesis.score, 0, args.label,
                ))
            valid, size_diagnostics = filter_by_physical_size(
                candidates, depth, matrix, minimum_diameter_m=0.012,
                unit_scale=0.001, min_depth_m=0.1, max_depth_m=2.0,
                distortion_coefficients=distortion, distortion_model=camera['model'],
            )
            unique = deduplicate_candidates(valid)
            if len(unique) != 1:
                ambiguous.append({'stamp_ns': capture, 'candidate_count': len(unique),
                                  'candidate_geometry': size_diagnostics})
                continue
            box = unique[0]
            z = robust_box_depth(depth, box, 0.001, 0.45, 0.1, 2.0, 20, method='weighted')
            if z is None:
                counts['insufficient_weighted_depth'] += 1
                continue
            try:
                transform = buffer.lookup_transform(
                    'base_link', camera['frame'], Time(nanoseconds=capture),
                ).transform
            except TransformException as error:
                failures.append({'stamp_ns': capture, 'error': str(error)})
                continue
            q, t = transform.rotation, transform.translation
            rotation = quaternion_to_matrix([q.x, q.y, q.z, q.w])
            translation = np.array([t.x, t.y, t.z])
            point_camera = project_pixel(matrix, *box.center, z, distortion, camera['model'])
            point_base = translation + rotation @ point_camera
            rows.append({
                'stamp_ns': capture, 'pixel': list(box.center), 'confidence': box.confidence,
                'box': [box.x1, box.y1, box.x2, box.y2], 'depth_m': z,
                'point_camera': point_camera.tolist(), 'measured_base': point_base.tolist(),
                'camera_translation': translation.tolist(), 'camera_rotation': rotation.tolist(),
                'delta_to_old_plan_mm': ((point_base - planned) * 1000).tolist(),
                'delta_to_old_coarse_mm': ((point_base - coarse) * 1000).tolist(),
            })
        joint_array = np.array([sample['position'] for sample in joints])
        home_confirmed = bool(joints and np.max(np.abs(joint_array)) <= 0.001)
        summary = {'sample_count': len(rows), 'home_confirmed': home_confirmed}
        if rows:
            points = np.asarray([row['measured_base'] for row in rows])
            median = np.median(points, axis=0)
            summary.update(
                home_base_median=median.tolist(),
                home_base_axis_std_mm=(np.std(points, axis=0) * 1000).tolist(),
                home_minus_old_plan_mm=((median - planned) * 1000).tolist(),
                home_to_old_plan_distance_mm=float(np.linalg.norm(median - planned) * 1000),
                home_minus_old_coarse_mm=((median - coarse) * 1000).tolist(),
                home_to_old_coarse_distance_mm=float(np.linalg.norm(median - coarse) * 1000),
            )
        result = {
            'read_only': True, 'capture_seconds': args.seconds, 'selected': args.label,
            'baseline': str(args.baseline), 'old_planned_button': planned.tolist(),
            'old_coarse_measured_median': coarse.tolist(), 'camera_info': camera,
            'rgbd_matching': 'exact capture timestamp; exact image-time TF; no fallback',
            'selection': 'unique physical-size-valid raw class after overlapping-box deduplication',
            'counts': dict(counts), 'summary': summary, 'samples': rows,
            'ambiguous_frames': ambiguous, 'tf_errors': failures,
            'joint_feedback': {
                'sample_count': len(joints), 'first': joints[0] if joints else None,
                'last': joints[-1] if joints else None,
                'axis_maximum_abs_rad': np.max(np.abs(joint_array), axis=0).tolist() if joints else None,
                'axis_span_rad': np.ptp(joint_array, axis=0).tolist() if joints else None,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print(json.dumps({'output': str(args.output), 'summary': summary,
                          'counts': dict(counts), 'tf_errors': len(failures),
                          'ambiguous_frames': len(ambiguous)}, indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
