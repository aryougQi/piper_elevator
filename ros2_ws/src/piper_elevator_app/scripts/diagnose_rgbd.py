#!/usr/bin/env python3
"""Read-only RGB-D/TF diagnostics; never invokes planning or motion services.

Run after sourcing the ROS workspace:
  python3 src/piper_elevator_app/scripts/diagnose_rgbd.py --seconds 15
"""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from rcl_interfaces.srv import GetParameters
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection2DArray

from piper_elevator_app.coarse_approach_core import stable_observation_window
from piper_elevator_app.motion_core import quaternion_to_matrix


def stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    return {
        'minimum': float(np.min(values)),
        'median': float(np.median(values)),
        'p95': float(np.percentile(values, 95)),
        'maximum': float(np.max(values)),
    }


def quaternion(rotation):
    return [rotation.x, rotation.y, rotation.z, rotation.w]


class Diagnostics(Node):
    def __init__(self):
        super().__init__('rgbd_read_only_diagnostics')
        self.recording = False
        self.topics = defaultdict(list)
        self.surface = []
        self.boxes = []
        self.pixels = []
        self.selected = ''
        self.selection_events = []
        self.camera_info = {}
        self.valid = Counter()
        self.status = Counter()
        self.tf_buffer = Buffer(cache_time=Duration(seconds=60.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.subscriptions_owned = []
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        selection_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        for topic, message_type in [
            ('/button_surface_pose', PoseStamped),
            ('/button_pose', PoseStamped),
            ('/button_pixel', PointStamped),
            ('/button_detections', Detection2DArray),
            ('/button_detection_valid', Bool),
            ('/button_approach/status', String),
            ('/camera/color/camera_info', CameraInfo),
            ('/camera/aligned_depth_to_color/camera_info', CameraInfo),
            ('/button_selected', String),
        ]:
            self.subscriptions_owned.append(self.create_subscription(
                message_type, topic,
                lambda message, topic=topic: self.receive(topic, message),
                selection_qos if topic == '/button_selected' else qos,
            ))

    def receive(self, topic, message):
        if topic == '/button_selected':
            self.selected = message.data
            self.selection_events.append(message.data)
        if isinstance(message, CameraInfo):
            self.camera_info[topic] = {
                'frame_id': message.header.frame_id,
                'width': message.width, 'height': message.height,
                'k': list(message.k), 'd': list(message.d),
                'distortion_model': message.distortion_model,
            }
        if not self.recording:
            return
        received = time.monotonic()
        stamp = None
        age = None
        if hasattr(message, 'header'):
            stamp = Time.from_msg(message.header.stamp).nanoseconds
            age = (self.get_clock().now().nanoseconds - stamp) / 1e9
        self.topics[topic].append((received, stamp, age))
        if topic == '/button_surface_pose':
            self.surface.append((message, received, age, self.selected))
        elif topic == '/button_detection_valid':
            self.valid[str(message.data)] += 1
        elif topic == '/button_approach/status':
            self.status[message.data] += 1
        elif topic == '/button_pixel':
            self.pixels.append([message.point.x, message.point.y, message.point.z])
        elif topic == '/button_detections':
            for detection in message.detections:
                labels = [result.hypothesis.class_id for result in detection.results]
                if self.selected in labels:
                    self.boxes.append([detection.bbox.size_x, detection.bbox.size_y])

    def parameters(self, node_name, names):
        client = self.create_client(GetParameters, node_name + '/get_parameters')
        try:
            if not client.wait_for_service(timeout_sec=1.0):
                return {'error': 'parameter service unavailable'}
            future = client.call_async(GetParameters.Request(names=names))
            rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
            if not future.done() or future.result() is None:
                return {'error': 'parameter request timed out'}
            return {
                name: Parameter.from_parameter_msg(
                    self.parameter_message(name, value)
                ).value for name, value in zip(names, future.result().values)
            }
        finally:
            self.destroy_client(client)

    @staticmethod
    def parameter_message(name, value):
        from rcl_interfaces.msg import Parameter as ParameterMessage
        return ParameterMessage(name=name, value=value)

    def collect(self, seconds):
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        started = time.monotonic()
        self.recording = True
        while time.monotonic() - started < seconds:
            rclpy.spin_once(self, timeout_sec=0.05)
        self.recording = False
        return time.monotonic() - started

    def report(self, elapsed, planner, detector):
        report = {
            'read_only': True, 'seconds': elapsed,
            'selected_button': self.selected,
            'selection_events': self.selection_events,
            'parameters': {'planner': planner, 'detector': detector},
            'topics': {}, 'detection_valid': dict(self.valid),
            'planner_status': dict(self.status), 'camera_info': self.camera_info,
        }
        for topic, rows in self.topics.items():
            times = [row[0] for row in rows]
            stamps = [row[1] / 1e9 for row in rows if row[1] is not None]
            ages = [row[2] for row in rows if row[2] is not None]
            report['topics'][topic] = {
                'count': len(rows), 'rate_hz': len(rows) / elapsed,
                'arrival_interval_seconds': stats(np.diff(times)),
                'capture_interval_seconds': stats(np.diff(stamps)),
                'age_seconds': stats(ages),
                'nonincreasing_stamp_count': int(np.sum(np.diff(stamps) <= 0)),
            }
        if self.pixels:
            pixels = np.asarray(self.pixels)
            report['selected_pixel'] = {
                'u_px': stats(pixels[:, 0]), 'v_px': stats(pixels[:, 1]),
                'radius_px': stats(pixels[:, 2]),
            }
        if self.boxes:
            boxes = np.asarray(self.boxes)
            ratio = detector.get('surface_inner_ratio') or 0.7
            report['selected_detection_box'] = {
                'count': len(boxes), 'width_px': stats(boxes[:, 0]),
                'height_px': stats(boxes[:, 1]),
                'surface_roi_width_px': stats(boxes[:, 0] * ratio),
                'surface_roi_height_px': stats(boxes[:, 1] * ratio),
            }
        observations = []
        errors = Counter()
        base = planner.get('base_frame') or 'base_link'
        expected_frame = planner.get('camera_frame') or 'camera_color_optical_frame'
        max_age = planner.get('surface_normal_max_age_seconds') or 0.5
        for message, received, age, selected in self.surface:
            if message.header.frame_id != expected_frame:
                errors['wrong_camera_frame'] += 1
                continue
            if age < -0.03 or age > max_age:
                errors['stale_or_future_stamp'] += 1
                continue
            try:
                transform = self.tf_buffer.lookup_transform(
                    base, message.header.frame_id, Time.from_msg(message.header.stamp)
                ).transform
            except TransformException as error:
                errors[type(error).__name__ + ': ' + str(error)] += 1
                continue
            rotation = quaternion_to_matrix(quaternion(transform.rotation))
            translation = np.array([
                transform.translation.x, transform.translation.y, transform.translation.z,
            ])
            camera_point = np.array([
                message.pose.position.x, message.pose.position.y, message.pose.position.z,
            ])
            normal = rotation @ quaternion_to_matrix(quaternion(message.pose.orientation))[:, 2]
            if np.dot(normal, rotation @ camera_point) < 0.0:
                normal = -normal
            observations.append({
                'button': translation + rotation @ camera_point,
                'normal': normal, 'camera_point': camera_point,
                'received_at': received, 'selected_button': selected,
                'stamp_ns': Time.from_msg(message.header.stamp).nanoseconds,
            })
        report['capture_time_tf_errors'] = dict(errors)
        report['transformed_samples'] = len(observations)
        if not observations:
            return report
        points = np.array([sample['button'] for sample in observations])
        center = np.median(points, axis=0)
        normals = np.array([sample['normal'] for sample in observations])
        mean = np.mean(normals, axis=0)
        mean /= np.linalg.norm(mean)
        report['base_position'] = {
            'median_m': center.tolist(), 'axis_std_m': np.std(points, axis=0).tolist(),
            'distance_from_median_mm': stats(1000 * np.linalg.norm(points - center, axis=1)),
            'step_mm': stats(1000 * np.linalg.norm(np.diff(points, axis=0), axis=1)),
        }
        report['base_normal'] = {
            'mean_unit': mean.tolist(),
            'angle_from_mean_degrees': stats(np.degrees(np.arccos(np.clip(normals @ mean, -1, 1)))),
            'step_degrees': stats(np.degrees(np.arccos(np.clip(np.sum(normals[1:] * normals[:-1], axis=1), -1, 1)))),
        }
        report['camera_depth_m'] = stats([sample['camera_point'][2] for sample in observations])
        report['stable_windows'] = {}
        minimum = int(planner.get('observation_minimum_samples') or 8)
        position_limit = planner.get('observation_position_tolerance_m') or 0.008
        normal_limit = planner.get('observation_normal_tolerance_rad') or math.radians(5.0)
        maximum_duration = planner.get('observation_window_max_seconds') or 3.0
        for length in sorted(set([8, 12, 16, 20, int(planner.get('observation_stable_samples') or 20)])):
            outcomes = []
            reasons = Counter()
            uncertainties = []
            trends = []
            for end in range(length, len(observations) + 1):
                window = observations[end - length:end]
                if (window[-1]['stamp_ns'] - window[0]['stamp_ns']) / 1e9 > maximum_duration:
                    reasons['window_too_old'] += 1
                    outcomes.append(False)
                    continue
                position, normal, detail = stable_observation_window(
                    window, minimum, position_limit, normal_limit,
                )
                success = position is not None and normal is not None
                outcomes.append(success)
                if not success:
                    reasons[detail] += 1
                unit = np.array([sample['normal'] for sample in window])
                unit_mean = unit.mean(axis=0)
                unit_mean /= np.linalg.norm(unit_mean)
                angles = np.arccos(np.clip(unit @ unit_mean, -1, 1))
                uncertainties.append(math.degrees(2 * np.sqrt(np.sum(angles ** 2)) / length))
                halves = [part.mean(axis=0) for part in np.array_split(unit, 2)]
                halves = [part / np.linalg.norm(part) for part in halves]
                trends.append(math.degrees(math.acos(float(np.clip(halves[0] @ halves[1], -1, 1)))))
            report['stable_windows'][str(length)] = {
                'tested': len(outcomes), 'passed': sum(outcomes),
                'pass_fraction': sum(outcomes) / len(outcomes) if outcomes else None,
                'normal_uncertainty_degrees': stats(uncertainties),
                'window_change_degrees': stats(trends),
                'frequent_rejections': reasons.most_common(5),
            }
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=15.0)
    parser.add_argument(
        '--output', type=Path,
        default=Path(__file__).resolve().parents[3] / 'diagnostics' / 'data' / 'rgbd_diagnostic.json',
        help='Write JSON here as well as stdout',
    )
    args = parser.parse_args()
    if not 1.0 <= args.seconds <= 45.0:
        parser.error('--seconds must be between 1 and 45')
    rclpy.init()
    node = Diagnostics()
    try:
        planner = node.parameters('/button_approach_planner', [
            'base_frame', 'camera_frame', 'surface_normal_max_age_seconds',
            'observation_stable_samples', 'observation_minimum_samples',
            'observation_window_max_seconds', 'observation_position_tolerance_m',
            'observation_normal_tolerance_rad', 'planning_observation_wait_seconds',
            'max_target_drift_m', 'use_sim_time',
        ])
        detector = node.parameters('/button_detector', [
            'surface_inner_ratio', 'surface_minimum_samples', 'surface_maximum_samples',
            'surface_max_residual_m', 'surface_max_tilt_degrees',
            'surface_normal_smoothing_alpha', 'camera_position_smoothing_alpha',
            'position_smoothing_alpha', 'required_stable_frames', 'depth_inner_ratio',
            'minimum_depth_samples', 'selected_button_class', 'use_sim_time',
        ])
        elapsed = node.collect(args.seconds)
        result = json.dumps(node.report(elapsed, planner, detector), indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as output:
                output.write(result + '\n')
        print(result)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
