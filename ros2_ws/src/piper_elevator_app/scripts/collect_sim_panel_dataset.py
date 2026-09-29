#!/usr/bin/env python3
"""Collect labeled Gazebo panel images using project camera, TF and arm action.

The detector is deliberately unused: YOLO labels are projected from the SDF
button geometry. Run with gazebo_hardware.launch.py and --execute.
"""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import subprocess
import time
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
from control_msgs.action import FollowJointTrajectory
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint


JOINTS = tuple(f'joint{i}' for i in range(1, 7))
CLASSES = [str(i) for i in range(1, 11)] + ['up', 'down', 'open', 'close']
# Only the panel controls that the 14-class detector can label; the bell and
# handset tiles of the vertical cabin panel are deliberately excluded.
TARGETS = ('1', '2', '3', 'open', 'close', 'up', 'down')


def rotation(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def transform_matrix(message):
    p = message.transform.translation
    q = message.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    matrix = np.eye(4)
    matrix[:3, :3] = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    matrix[:3, 3] = [p.x, p.y, p.z]
    return matrix


def read_panel_geometry(model_path=None, panel_pose=None):
    """Read the installed fixed Gazebo scene and its button visual dimensions."""
    share = Path(get_package_share_directory('piper_elevator_gazebo'))
    world = ET.parse(share / 'worlds/button_press.sdf').getroot()
    panel = ET.parse(model_path or share / 'models/elevator_button/model.sdf').getroot()
    include = next(item for item in world.findall('.//include')
                   if item.findtext('name') == 'elevator_button')
    pose = (list(panel_pose) if panel_pose is not None else
            [float(value) for value in include.findtext('pose').split()])
    if len(pose) != 6:
        raise ValueError('expected a six-value elevator_button world pose')
    panel_R = rotation(*pose[3:])
    panel_t = np.asarray(pose[:3])
    faces = {}
    for name in TARGETS:
        link = panel.find(f".//link[@name='button_{name}_face']")
        if link is None:
            raise ValueError(f'button_{name}_face missing from model.sdf')
        local_pose = [float(value) for value in link.findtext('pose').split()]
        size = [float(value) for value in
                link.findtext('./visual/geometry/box/size').split()]
        # The visible face is at positive local X, facing the camera after
        # the world's pi-yaw panel rotation.
        x = local_pose[0] + size[0] / 2
        y, z = local_pose[1:3]
        hy, hz = size[1] / 2, size[2] / 2
        corners = np.array([[x, y - hy, z - hz], [x, y + hy, z - hz],
                            [x, y + hy, z + hz], [x, y - hy, z + hz]])
        faces[name] = (panel_R @ corners.T).T + panel_t
    return faces, pose


def gazebo_service(service, request_type, request):
    command = ['ign', 'service', '-s', service, '--reqtype', request_type,
               '--reptype', 'ignition.msgs.Boolean', '--timeout', '10000',
               '--req', request]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    if result.returncode != 0 or 'data: true' not in result.stdout:
        raise RuntimeError(f'Gazebo service {service} failed: '
                           f'{result.stdout} {result.stderr}')


def replace_panel(variant, pose):
    """Swap the fixed panel for a project variant at a known world pose."""
    for name in ('elevator_button', *(f'elevator_button_v{i}' for i in range(5))):
        request = f'name: "{name}" type: 2'
        # Missing entities are expected; only the create result must succeed.
        subprocess.run([
            'ign', 'service', '-s', '/world/button_press/remove',
            '--reqtype', 'ignition.msgs.Entity',
            '--reptype', 'ignition.msgs.Boolean', '--timeout', '3000',
            '--req', request,
        ], capture_output=True, text=True, timeout=5)
    model_path = (Path('/workspace/ros2_ws/src/piper_elevator_gazebo') /
                  'sim_variants/models' / f'elevator_button_{variant}/model.sdf')
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    x, y, z, roll, pitch, yaw = pose
    # Gazebo EntityFactory accepts SDF with an explicit world pose quaternion.
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    qx = sr*cp*cy-cr*sp*sy
    qy = cr*sp*cy+sr*cp*sy
    qz = cr*cp*sy-sr*sp*cy
    qw = cr*cp*cy+sr*sp*sy
    request = (f'sdf_filename: "{model_path}" name: "elevator_button_{variant}" '
               f'pose {{ position {{ x: {x} y: {y} z: {z} }} '
               f'orientation {{ x: {qx} y: {qy} z: {qz} w: {qw} }} }}')
    gazebo_service('/world/button_press/create',
                   'ignition.msgs.EntityFactory', request)
    return model_path


def jittered_pose(base_pose, rng):
    values = np.asarray(base_pose, dtype=float).copy()
    values[:3] += rng.uniform([-0.012, -0.015, -0.010],
                              [0.012, 0.015, 0.010])
    values[3:] += rng.uniform([-0.02, -0.02, -0.035],
                              [0.02, 0.02, 0.035])
    return values.tolist()


def project_labels(faces, world_to_camera, camera_info, image_shape, depth=None):
    """Return YOLO rows and diagnostics. Reject the whole frame on bad geometry."""
    height, width = image_shape[:2]
    K = np.asarray(camera_info.k).reshape(3, 3)
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError('invalid camera intrinsics')
    rows, diagnostics = [], {}
    for name, world_corners in faces.items():
        camera = (world_to_camera[:3, :3] @ world_corners.T).T + world_to_camera[:3, 3]
        if np.any(camera[:, 2] <= 0.1):
            diagnostics[name] = 'behind_camera'
            continue
        u = K[0, 0] * camera[:, 0] / camera[:, 2] + K[0, 2]
        v = K[1, 1] * camera[:, 1] / camera[:, 2] + K[1, 2]
        left, right = float(u.min()), float(u.max())
        top, bottom = float(v.min()), float(v.max())
        if left < 0 or top < 0 or right >= width or bottom >= height:
            diagnostics[name] = 'outside_image'
            continue
        if right - left < 8 or bottom - top < 8:
            diagnostics[name] = 'too_small'
            continue
        if depth is not None:
            cx, cy = int((left + right) / 2), int((top + bottom) / 2)
            patch = depth[max(0, cy-2):cy+3, max(0, cx-2):cx+3]
            valid = patch[np.isfinite(patch) & (patch > 0.05)]
            if valid.size and float(np.median(valid)) < float(camera[:, 2].mean()) - 0.03:
                diagnostics[name] = 'occluded'
                continue
        rows.append((CLASSES.index(name), (left + right) / (2*width),
                     (top + bottom) / (2*height), (right-left)/width,
                     (bottom-top)/height))
        diagnostics[name] = 'ok'
    return rows, diagnostics


class Collector(Node):
    def __init__(self):
        super().__init__('sim_panel_dataset_collector')
        self.set_parameters([Parameter('use_sim_time', value=True)])
        self.bridge = CvBridge()
        self.joints = None
        self.panel_seen = False
        self.info = None
        self.images = {}
        self.depths = {}
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.arm = ActionClient(self, FollowJointTrajectory,
                                '/arm_controller/follow_joint_trajectory')
        self.create_subscription(JointState, '/piper_pika/joint_states',
                                 self.on_joints, 10)
        self.create_subscription(JointState, '/elevator_button/joint_states',
                                 self.on_panel, 10)
        self.create_subscription(CameraInfo, '/camera/color/camera_info',
                                 self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/color/image_raw',
                                 self.on_image, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw',
                                 self.on_depth, qos_profile_sensor_data)

    def on_joints(self, message):
        values = dict(zip(message.name, message.position))
        if all(j in values for j in JOINTS):
            self.joints = {j: float(values[j]) for j in JOINTS}

    def on_panel(self, message):
        self.panel_seen = bool(message.name)

    def on_info(self, message):
        self.info = message

    @staticmethod
    def stamp(message):
        s = message.header.stamp
        return s.sec * 1_000_000_000 + s.nanosec

    def on_image(self, message):
        self.images[self.stamp(message)] = message
        if len(self.images) > 50:
            self.images.pop(next(iter(self.images)))

    def on_depth(self, message):
        self.depths[self.stamp(message)] = message
        if len(self.depths) > 50:
            self.depths.pop(next(iter(self.depths)))

    def wait(self, predicate, timeout, description):
        end = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if predicate():
                return
        raise TimeoutError(f'timed out waiting for {description}')

    def spin_for(self, seconds):
        end = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=min(0.05, end-time.monotonic()))

    def move(self, target, seconds, timeout):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(JOINTS)
        point = JointTrajectoryPoint()
        point.positions = [target[j] for j in JOINTS]
        point.time_from_start = Duration(seconds=seconds).to_msg()
        goal.trajectory.points = [point]
        sent = self.arm.send_goal_async(goal)
        self.wait(sent.done, timeout, 'trajectory acceptance')
        handle = sent.result()
        if not handle.accepted:
            raise RuntimeError('arm controller rejected trajectory')
        result = handle.get_result_async()
        self.wait(result.done, timeout, 'trajectory completion')
        outcome = result.result().result
        if outcome.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(f'arm trajectory failed: {outcome.error_string}')
        self.wait(lambda: self.joints is not None and all(
            abs(self.joints[j]-target[j]) < 0.02 for j in JOINTS),
            3.0, 'joint position feedback')


def parse_angles(value):
    angles = [float(x) for x in value.split(',')]
    if not angles or any(not math.isfinite(x) or abs(x) > 20 for x in angles):
        raise argparse.ArgumentTypeError('angles must be within ±20 degrees')
    return angles


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='enable Gazebo arm movement')
    parser.add_argument('--joint1-angles', type=parse_angles,
                        default=parse_angles('-15,-10,-5,0,5,10,15'))
    parser.add_argument('--joint5-angles', type=parse_angles,
                        default=parse_angles('-5,0,5'))
    parser.add_argument('--move-seconds', type=float, default=3.0)
    parser.add_argument('--settle-seconds', type=float, default=1.0)
    parser.add_argument('--timeout', type=float, default=40.0)
    parser.add_argument('--scene-id', default='fixed_panel',
                        help='identifier for grouping train/val/test later')
    parser.add_argument('--variants', default='',
                        help='comma-separated panel variants v0,...,v4; empty keeps original')
    parser.add_argument('--jitters', type=int, default=1,
                        help='number of random panel poses per variant')
    parser.add_argument('--seed', type=int, default=20260923)
    parser.add_argument('--no-depth-check', action='store_true')
    parser.add_argument('--output', type=Path,
                        default=Path('/workspace/sim-dataset/Raw'))
    return parser.parse_args()


def main():
    args = arguments()
    if not args.execute:
        raise SystemExit('Pass --execute to move the simulated arm.')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.scene_id):
        raise SystemExit('--scene-id must contain only letters, digits, _ or -')
    if min(args.move_seconds, args.settle_seconds, args.timeout) <= 0:
        raise SystemExit('time values must be positive')
    faces, panel_pose = read_panel_geometry()
    base_pose = list(panel_pose)
    variants = [part.strip() for part in args.variants.split(',') if part.strip()]
    if any(part not in {f'v{i}' for i in range(5)} for part in variants):
        raise SystemExit('--variants accepts only v0,v1,v2,v3,v4')
    if args.jitters < 1:
        raise SystemExit('--jitters must be positive')
    run = args.output / datetime.now().strftime('%Y%m%d_%H%M%S')
    for part in ('images', 'labels', 'review', 'metadata'):
        (run / part).mkdir(parents=True, exist_ok=False)
    rclpy.init()
    node = Collector()
    start = None
    try:
        node.wait(lambda: node.panel_seen and node.joints is not None
                  and node.info is not None and node.images,
                  args.timeout, 'Gazebo panel, arm, camera info, and images')
        if not node.arm.wait_for_server(timeout_sec=args.timeout):
            raise RuntimeError('arm controller action is unavailable')
        start = dict(node.joints)
        manifest = {'scene_id': args.scene_id, 'panel_pose_xyz_rpy': panel_pose,
                    'classes': CLASSES, 'joint1_angles_deg': args.joint1_angles,
                    'joint5_angles_deg': args.joint5_angles,
                    'start_joints_rad': start, 'variants': variants,
                    'jitters': args.jitters, 'seed': args.seed, 'samples': []}
        rng = np.random.default_rng(args.seed)
        scene_specs = ((variant, index) for variant in (variants or [None])
                       for index in range(args.jitters if variant else 1))
        for variant, index in scene_specs:
            if variant is not None:
                panel_pose = jittered_pose(base_pose, rng)
                model_path = replace_panel(variant, panel_pose)
                faces, _ = read_panel_geometry(model_path, panel_pose)
                node.spin_for(1.0)
            scene = (f'{args.scene_id}_{variant}_p{index:02d}' if variant
                     else args.scene_id)
            print(f'Scene {scene}: panel pose {panel_pose}', flush=True)
            for a1 in args.joint1_angles:
              for a5 in args.joint5_angles:
                target = dict(start)
                target['joint1'] += math.radians(a1)
                target['joint5'] += math.radians(a5)
                node.move(target, args.move_seconds, args.timeout)
                node.spin_for(args.settle_seconds)
                previous = max(node.images)
                node.wait(lambda: max(node.images) > previous,
                          args.timeout, 'fresh settled image')
                stamp = max(node.images)
                message = node.images[stamp]
                def matching_depth_stamp():
                    if not node.depths:
                        return None
                    nearest = min(node.depths, key=lambda candidate: abs(candidate-stamp))
                    return nearest if abs(nearest-stamp) <= 20_000_000 else None

                if not args.no_depth_check:
                    try:
                        node.wait(lambda: matching_depth_stamp() is not None,
                                  2.0, 'depth image within 20 ms of color image')
                    except TimeoutError:
                        # The fixed panel is fully visible in this workcell.
                        # Keep the RGB/TF label and record that the optional
                        # depth occlusion check could not be performed.
                        pass
                source_time = Time.from_msg(message.header.stamp)
                frame = message.header.frame_id
                node.wait(lambda: node.tf_buffer.can_transform(
                    frame, 'world', source_time), args.timeout,
                    'world-to-camera TF at image timestamp')
                tf = node.tf_buffer.lookup_transform(frame, 'world', source_time)
                image = node.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
                depth = None
                if not args.no_depth_check and matching_depth_stamp() is not None:
                    depth_msg = node.depths[matching_depth_stamp()]
                    depth = node.bridge.imgmsg_to_cv2(depth_msg).astype(np.float64)
                    if depth_msg.encoding == '16UC1':
                        depth *= 0.001
                rows, diagnostics = project_labels(
                    faces, transform_matrix(tf), node.info, image.shape, depth)
                name = f'{scene}_j1_{a1:+g}_j5_{a5:+g}'
                if len(rows) != len(TARGETS):
                    print(f'Skipping {name}: {diagnostics}', flush=True)
                    manifest['samples'].append({'name': name, 'saved': False,
                                                'diagnostics': diagnostics})
                    continue
                if not cv2.imwrite(str(run / 'images' / f'{name}.png'), image):
                    raise RuntimeError(f'could not save image {name}')
                (run / 'labels' / f'{name}.txt').write_text(''.join(
                    f'{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n'
                    for cls, cx, cy, w, h in rows), encoding='utf-8')
                review = image.copy()
                height, width = image.shape[:2]
                for cls, cx, cy, w, h in rows:
                    x1, y1 = round((cx-w/2)*width), round((cy-h/2)*height)
                    x2, y2 = round((cx+w/2)*width), round((cy+h/2)*height)
                    cv2.rectangle(review, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(review, CLASSES[cls], (x1, max(14, y1-4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imwrite(str(run / 'review' / f'{name}.png'), review)
                metadata = {'name': name, 'scene_id': scene,
                            'variant': variant, 'pose_jitter_index': index,
                            'stamp_ns': stamp, 'image_frame': frame,
                            'joint_offsets_deg': {'joint1': a1, 'joint5': a5},
                            'actual_joints_rad': node.joints,
                            'camera_k': list(node.info.k),
                            'world_to_camera': transform_matrix(tf).tolist(),
                            'panel_pose_xyz_rpy': panel_pose,
                            'depth_check': depth is not None,
                            'diagnostics': diagnostics}
                (run / 'metadata' / f'{name}.json').write_text(
                    json.dumps(metadata, indent=2), encoding='utf-8')
                manifest['samples'].append({'name': name, 'saved': True})
                print(f'Saved {name}: {len(rows)} labels', flush=True)
        (run / 'manifest.json').write_text(json.dumps(manifest, indent=2),
                                            encoding='utf-8')
        print(f'Dataset capture: {run}', flush=True)
    finally:
        if start is not None:
            try:
                node.move(start, args.move_seconds, args.timeout)
            except Exception as exc:
                print(f'WARNING: failed to return to start pose: {exc}', flush=True)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
