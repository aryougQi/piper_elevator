#!/usr/bin/env python3
"""Read-only coarse/Servo handoff snapshot; no motion endpoints are created.

Run inside the sourced ROS workspace. Parameter GET/LIST services and topic/TF
subscriptions are the only ROS operations. The output preserves raw samples.
"""

import argparse
from array import array
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import rclpy
from rcl_interfaces.srv import ListParameters
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.utilities import get_message

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src/piper_elevator_app/scripts'))
from diagnose_rgbd import Diagnostics, quaternion, stats
from piper_elevator_app.motion_core import check_servo_capture, quaternion_to_matrix


def json_value(value):
    if isinstance(value, (np.ndarray, array)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class HandoffDiagnostics(Diagnostics):
    def __init__(self):
        super().__init__()
        self.messages = {}
        self.extra_samples = defaultdict(list)
        self.extra_counts = defaultdict(Counter)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        sensor = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        for topic, type_name, qos in [
            ('/button_selection', 'std_msgs/msg/String', latched),
            ('/button_approach/status', 'std_msgs/msg/String', latched),
            ('/button_approach_planner/observation_status', 'std_msgs/msg/String', latched),
            ('/button_visual_servo/status', 'std_msgs/msg/String', latched),
            ('/button_visual_servo/completed', 'std_msgs/msg/Bool', latched),
            ('/button_press/status', 'std_msgs/msg/String', latched),
            ('/button_press/servo_claimed', 'std_msgs/msg/Bool', latched),
            ('/button_tracking_state', 'std_msgs/msg/String', sensor),
            ('/servo_node/status', 'std_msgs/msg/Int8', sensor),
            ('/servo_node/collision_velocity_scale', 'std_msgs/msg/Float64', sensor),
            ('/servo_node/delta_twist_cmds', 'geometry_msgs/msg/TwistStamped', sensor),
            ('/feedback/arm_status', 'agx_arm_msgs/msg/AgxArmStatus', sensor),
            ('/feedback/joint_states', 'sensor_msgs/msg/JointState', sensor),
            ('/piper_pika/joint_states', 'sensor_msgs/msg/JointState', sensor),
            ('/control/joint_states', 'sensor_msgs/msg/JointState', sensor),
            ('/arm_controller/follow_joint_trajectory/_action/status',
             'action_msgs/msg/GoalStatusArray', latched),
        ]:
            self.subscriptions_owned.append(self.create_subscription(
                get_message(type_name), topic,
                lambda message, topic=topic: self.receive_extra(topic, message), qos))

    def receive_extra(self, topic, message):
        value = message_to_ordereddict(message)
        if topic.endswith('/observation_status') or topic == '/button_tracking_state':
            try:
                value = json.loads(message.data)
            except ValueError:
                pass
        self.messages[topic] = value
        if not self.recording:
            return
        stamp_ns = (Time.from_msg(message.header.stamp).nanoseconds
                    if hasattr(message, 'header') else None)
        age = ((self.get_clock().now().nanoseconds - stamp_ns) / 1e9
               if stamp_ns is not None else None)
        self.extra_samples[topic].append((time.monotonic(), stamp_ns, age))
        if topic == '/button_tracking_state' and isinstance(value, dict):
            selected = value.get('selected') or {}
            self.extra_counts[topic][str(selected.get('state', value.get('reason', 'unknown')))] += 1
        elif hasattr(message, 'data') and topic != '/button_approach_planner/observation_status':
            self.extra_counts[topic][str(message.data)] += 1

    def all_parameters(self, name):
        client = self.create_client(ListParameters, name + '/list_parameters')
        try:
            if not client.wait_for_service(timeout_sec=1.0):
                return {'error': 'parameter listing service unavailable'}
            future = client.call_async(ListParameters.Request(depth=0))
            rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
            if not future.done() or future.result() is None:
                return {'error': 'parameter listing timed out'}
            names = [name for name in future.result().result.names
                     if not name.startswith(('robot_description', 'robot_description_semantic'))]
            return self.parameters(name, names)
        finally:
            self.destroy_client(client)

    def handoff_report(self, elapsed, parameters):
        planner = parameters['/button_approach_planner']
        servo = parameters['/button_visual_servo']
        report = self.report(elapsed, planner, parameters['/button_detector'])
        report.update({
            'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
            'parameters': parameters,
            'last_messages': self.messages,
            'extra_status_counts': {topic: dict(values) for topic, values in self.extra_counts.items()},
            'limitations': [
                'Stationary subscriptions do not exercise the Servo control loop or hardware transitions.',
                'The control gate has no public mode topic; hardware arm_status is not a Servo authorization acknowledgement.',
                'Replay EMA starts at the first captured sample and is not the running Servo internal filter state.',
                'READY is a callback status, not proof that the complete handoff capture geometry passed.',
            ],
        })
        for topic, rows in self.extra_samples.items():
            times = [row[0] for row in rows]
            stamps = [row[1] / 1e9 for row in rows if row[1] is not None]
            ages = [row[2] for row in rows if row[2] is not None]
            report['topics'][topic] = {
                'count': len(rows), 'rate_hz': len(rows) / elapsed,
                'arrival_interval_seconds': stats(np.diff(times)),
                'age_seconds': stats(ages),
                'nonincreasing_stamp_count': int(np.sum(np.diff(stamps) <= 0)),
            }
        base = servo.get('base_frame', 'base_link')
        tool = servo.get('end_effector_link', 'pika_fingertip_center_link')
        camera = servo.get('camera_frame', 'camera_color_optical_frame')
        current_transforms = {}
        for frame in (tool, camera):
            try:
                current_transforms[frame] = self.tf_buffer.lookup_transform(base, frame, Time())
            except Exception as error:
                report.setdefault('current_tf_errors', {})[frame] = str(error)
        report['current_tf'] = {frame: message_to_ordereddict(tf)
                                for frame, tf in current_transforms.items()}
        samples = []
        errors = Counter()
        position_ema = normal_ema = None
        for message, received, age, selected in self.surface:
            try:
                tf = self.tf_buffer.lookup_transform(
                    base, message.header.frame_id, Time.from_msg(message.header.stamp)).transform
                rotation = quaternion_to_matrix(quaternion(tf.rotation))
                point = rotation @ np.array([message.pose.position.x, message.pose.position.y,
                                             message.pose.position.z])
                point += [tf.translation.x, tf.translation.y, tf.translation.z]
                normal = rotation @ quaternion_to_matrix(quaternion(message.pose.orientation))[:, 2]
                if normal_ema is not None and np.dot(normal_ema, normal) < 0:
                    normal = -normal
                position_alpha = servo.get('world_position_smoothing_alpha', .25)
                normal_alpha = servo.get('world_normal_smoothing_alpha', .20)
                position_ema = point.copy() if position_ema is None else position_alpha * point + (1-position_alpha) * position_ema
                normal_ema = normal.copy() if normal_ema is None else normal_alpha * normal + (1-normal_alpha) * normal_ema
                normal_ema /= np.linalg.norm(normal_ema)
                row = {'surface_pose': message_to_ordereddict(message), 'age_seconds': age,
                       'selected_button': selected, 'base_position_m': point.tolist(),
                       'base_normal': normal.tolist(), 'position_ema_m': position_ema.tolist(),
                       'normal_ema': normal_ema.tolist()}
                if tool in current_transforms and camera in current_transforms:
                    tf_tool = current_transforms[tool].transform.translation
                    passed, detail = check_servo_capture(
                        position_ema, normal_ema, [tf_tool.x, tf_tool.y, tf_tool.z],
                        quaternion(current_transforms[camera].transform.rotation),
                        maximum_tilt_rad=servo.get('handover_maximum_camera_tilt_rad', np.pi/12),
                        maximum_roll_rad=servo.get('handover_maximum_camera_roll_rad', np.pi/12),
                        minimum_standoff_m=servo.get('handover_minimum_standoff_m', .08),
                        target_standoff_m=servo.get('standoff_distance_m', .03),
                        maximum_start_error_m=servo.get('maximum_start_error_m', .20),
                        level_reference_axis=servo.get('level_reference_axis', [0., 0., 1.]))
                    row['replay_capture_check'] = {'passed': passed, 'detail': detail}
                samples.append(row)
            except Exception as error:
                errors[type(error).__name__ + ': ' + str(error)] += 1
        report['samples'] = samples
        report['sample_transform_errors'] = dict(errors)
        checks = [row['replay_capture_check'] for row in samples if 'replay_capture_check' in row]
        report['replay_capture_checks'] = dict(Counter(str(row['passed']) for row in checks))
        report['last_replay_capture_check'] = checks[-1] if checks else None
        return report


DATA_DIR = Path(__file__).resolve().parents[1] / 'data'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=10.)
    parser.add_argument('--output', default=str(DATA_DIR / 'servo_handoff_diagnostic.json'))
    args = parser.parse_args()
    if not 1. <= args.seconds <= 30.:
        parser.error('--seconds must be between 1 and 30')
    if Path(args.output).exists():
        parser.error('--output already exists; choose another path to preserve evidence')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = HandoffDiagnostics()
    try:
        parameters = {name: node.all_parameters(name) for name in (
            '/button_approach_planner', '/button_visual_servo', '/button_detector',
            '/piper_pika_control_gate', '/servo_node', '/agx_arm_ctrl_single_node')}
        elapsed = node.collect(args.seconds)
        report = node.handoff_report(elapsed, parameters)
        Path(args.output).write_text(json.dumps(report, indent=2, default=json_value) + '\n')
        print(json.dumps({key: report.get(key) for key in (
            'read_only', 'seconds', 'selected_button', 'transformed_samples',
            'detection_valid', 'base_position', 'base_normal', 'extra_status_counts',
            'replay_capture_checks', 'last_replay_capture_check')}, indent=2))
        print('Full evidence: ' + str(Path(args.output).resolve()))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
