#!/usr/bin/env python3
"""Sweep simulated arm joint angles and record YOLO button detections.

Run inside the ROS 2 workspace container after Gazebo and button_detector start.
This measures detector output, not ground-truth mAP: an absent class may simply
be outside the camera view. Review saved images before interpreting recall.
"""

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time

import cv2
from cv_bridge import CvBridge
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from vision_msgs.msg import Detection2DArray


JOINTS = tuple(f'joint{i}' for i in range(1, 7))
# Controls of the vertical cabin panel whose labels exist in the 14-class
# detector. The bell (alarm) and handset (intercom) tiles have no class.
PANEL_CLASSES = ('1', '2', '3', 'open', 'close', 'up', 'down')


def stamp_ns(header):
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def parse_angles(value):
    angles = [float(part.strip()) for part in value.split(',')]
    if not angles or any(not math.isfinite(x) or abs(x) > 20 for x in angles):
        raise argparse.ArgumentTypeError('angles must be comma-separated degrees within ±20')
    return angles


class Collector(Node):
    def __init__(self):
        super().__init__(f'button_angle_collector_{os.getpid()}')
        self.bridge = CvBridge()
        self.joints = None
        self.sim_panel_seen = False
        self.images = {}
        self.detections = []
        self.create_subscription(JointState, '/piper_pika/joint_states', self.on_joints, 10)
        self.create_subscription(JointState, '/elevator_button/joint_states',
                                 self.on_sim_panel, 10)
        self.create_subscription(Image, '/camera/color/image_raw', self.on_image, qos_profile_sensor_data)
        self.create_subscription(Detection2DArray, '/button_detections', self.on_detections, 10)
        self.arm = ActionClient(self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')

    def on_joints(self, msg):
        values = dict(zip(msg.name, msg.position))
        if all(joint in values for joint in JOINTS):
            self.joints = {joint: float(values[joint]) for joint in JOINTS}

    def on_sim_panel(self, msg):
        if msg.name:
            self.sim_panel_seen = True

    def on_image(self, msg):
        self.images[stamp_ns(msg.header)] = msg
        if len(self.images) > 100:
            self.images.pop(next(iter(self.images)))

    def on_detections(self, msg):
        self.detections.append((time.monotonic(), msg))
        if len(self.detections) > 500:
            del self.detections[:100]

    def until(self, predicate, timeout, description):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if predicate():
                return
        raise TimeoutError(f'timed out waiting for {description}')

    def move(self, positions, duration, timeout):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(JOINTS)
        point = JointTrajectoryPoint()
        point.positions = [positions[joint] for joint in JOINTS]
        point.time_from_start = Duration(seconds=duration).to_msg()
        goal.trajectory.points = [point]
        # A zero stamp starts immediately in the controller's clock domain.
        # The collector itself does not require use_sim_time.
        sent = self.arm.send_goal_async(goal)
        self.until(sent.done, timeout, 'trajectory acceptance')
        handle = sent.result()
        if not handle.accepted:
            raise RuntimeError('arm trajectory was rejected')
        result = handle.get_result_async()
        self.until(result.done, timeout, 'trajectory result')
        response = result.result()
        if response.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(f'arm trajectory failed: {response.result.error_string}')
        self.until(
            lambda: self.joints is not None and all(
                abs(self.joints[j] - positions[j]) < 0.02 for j in JOINTS
            ), 3.0, 'joint position feedback',
        )


def boxes(message):
    output = []
    for detection in message.detections:
        if not detection.results:
            continue
        result = max(detection.results, key=lambda item: item.hypothesis.score)
        output.append({
            'class': result.hypothesis.class_id,
            'confidence': round(float(result.hypothesis.score), 5),
            'cx': float(detection.bbox.center.position.x),
            'cy': float(detection.bbox.center.position.y),
            'width': float(detection.bbox.size_x),
            'height': float(detection.bbox.size_y),
        })
    return output


def save_image(node, message, detections, path):
    image_msg = node.images.get(stamp_ns(message.header))
    if image_msg is None:
        return False
    original = node.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
    if not cv2.imwrite(str(path.with_name(path.stem + '_raw.png')), original):
        return False
    image = original.copy()
    for item in detections:
        x1 = round(item['cx'] - item['width'] / 2)
        y1 = round(item['cy'] - item['height'] / 2)
        x2 = round(item['cx'] + item['width'] / 2)
        y2 = round(item['cy'] + item['height'] / 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{item['class']} {item['confidence']:.2f}",
                    (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)
    return bool(cv2.imwrite(str(path), image))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='enable simulated arm movement')
    parser.add_argument('--joint', choices=JOINTS, default='joint1')
    parser.add_argument('--angles', type=parse_angles, default=parse_angles('-15,-10,-5,0,5,10,15'),
                        help='offsets in degrees relative to current pose; each must be within ±20')
    parser.add_argument('--samples', type=int, default=20, help='detector frames at each angle')
    parser.add_argument('--settle', type=float, default=1.0)
    parser.add_argument('--move-seconds', type=float, default=3.0)
    parser.add_argument('--timeout', type=float, default=30.0)
    parser.add_argument('--output', type=Path,
                        default=Path('/workspace/ros2_ws/test_logs/button_angle_scan'))
    parser.add_argument('--classes', default=','.join(PANEL_CLASSES),
                        help='expected classes; use only classes actually visible when interpreting recall')
    return parser.parse_args()


def main():
    args = arguments()
    if not args.execute:
        raise SystemExit('Pass --execute to move the simulated arm.')
    if args.samples < 1 or min(args.settle, args.move_seconds, args.timeout) <= 0:
        raise SystemExit('samples must be positive and times must be greater than zero')
    expected = tuple(name.strip() for name in args.classes.split(',') if name.strip())
    output = args.output / datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    rclpy.init()
    node = Collector()
    start_pose = None
    try:
        node.until(lambda: node.sim_panel_seen and node.joints is not None
                   and node.images and node.detections,
                   args.timeout, 'Gazebo panel, joint states, camera, and detector')
        if not node.arm.wait_for_server(timeout_sec=args.timeout):
            raise RuntimeError('arm trajectory action is unavailable')
        start_pose = dict(node.joints)
        with (output / 'metadata.json').open('w', encoding='utf-8') as file:
            json.dump({'joint': args.joint, 'angles_deg': args.angles,
                       'start_pose_rad': start_pose, 'classes': expected,
                       'samples_per_angle': args.samples}, file, indent=2)
        with (output / 'samples.jsonl').open('w', encoding='utf-8') as raw, \
             (output / 'summary.csv').open('w', newline='', encoding='utf-8') as summary:
            writer = csv.DictWriter(summary, fieldnames=[
                'angle_deg', 'joint', 'target_rad', 'actual_rad', 'class',
                'frames', 'frames_detected', 'detection_rate', 'mean_confidence',
                'image_saved',
            ])
            writer.writeheader()
            for angle in args.angles:
                target = dict(start_pose)
                target[args.joint] += math.radians(angle)
                print(f'Moving {args.joint} to offset {angle:+g}°', flush=True)
                node.move(target, args.move_seconds, args.timeout)
                node_spin(node, args.settle)
                since = time.monotonic()
                collected = []
                seen = set()
                deadline = since + args.timeout
                while len(collected) < args.samples and time.monotonic() < deadline:
                    rclpy.spin_once(node, timeout_sec=0.1)
                    for received, message in node.detections:
                        key = stamp_ns(message.header)
                        if received >= since and key not in seen:
                            seen.add(key)
                            collected.append(message)
                            if len(collected) == args.samples:
                                break
                if len(collected) != args.samples:
                    raise TimeoutError(f'only {len(collected)}/{args.samples} detector frames at {angle:g}°')
                per_frame = []
                for index, message in enumerate(collected):
                    found = boxes(message)
                    per_frame.append(found)
                    raw.write(json.dumps({
                        'angle_deg': angle, 'joint': args.joint, 'frame': index,
                        'stamp_ns': stamp_ns(message.header), 'actual_joints_rad': node.joints,
                        'detections': found,
                    }) + '\n')
                saved = any(save_image(node, message, per_frame[index],
                                       output / f'{args.joint}_{angle:+g}deg.png')
                            for index, message in reversed(list(enumerate(collected))))
                for name in expected:
                    scores = [max((item['confidence'] for item in frame
                                   if item['class'] == name), default=None)
                              for frame in per_frame]
                    present = [score for score in scores if score is not None]
                    writer.writerow({
                        'angle_deg': angle, 'joint': args.joint,
                        'target_rad': target[args.joint],
                        'actual_rad': node.joints[args.joint],
                        'class': name, 'frames': len(per_frame),
                        'frames_detected': len(present),
                        'detection_rate': round(len(present) / len(per_frame), 4),
                        'mean_confidence': round(sum(present) / len(present), 4) if present else '',
                        'image_saved': saved,
                    })
                summary.flush()
                print(f'{angle:+g}°: {len(collected)} frames; image={saved}', flush=True)
        print(f'Results: {output}', flush=True)
    finally:
        if start_pose is not None:
            try:
                print('Returning to initial joint pose...', flush=True)
                node.move(start_pose, args.move_seconds, args.timeout)
            except Exception as exc:
                print(f'WARNING: could not return to initial pose: {exc}', flush=True)
        node.destroy_node()
        rclpy.shutdown()


def node_spin(node, seconds):
    deadline = time.monotonic() + seconds
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=min(0.1, deadline - time.monotonic()))


if __name__ == '__main__':
    main()
