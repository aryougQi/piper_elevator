#!/usr/bin/env python3
"""Reproducible simulated elevator-panel dataset collector.

Runs inside the piper_ros2 container with the Gazebo virtual hardware
launched. For every (panel variant, pose jitter) combination the original
panel is removed and the variant spawned at a known jittered pose; the arm
sweeps a joint1 x joint5 grid; at each pose one frame is captured and
YOLO ground truth is projected from the KNOWN button geometry through the
camera intrinsics and TF. No detector output is used for labels.

Usage (inside container):
  python3 collect_panel_data.py --light L0 --variants v0,v1 \
      --jitters 2 --out /workspace/ros2_ws/sim_capture/trial --seed 123
"""

import argparse
import json
import math
import subprocess
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

JOINTS = tuple(f'joint{i}' for i in range(1, 7))

# Class ids follow dataset/data.yaml of the training project. The vertical
# cabin panel has no "4" tile, and its bell / handset tiles have no class.
CLASS_IDS = {'1': 0, '2': 1, '3': 2, 'up': 10, 'down': 11,
             'open': 12, 'close': 13}
ID_NAMES = {v: k for k, v in CLASS_IDS.items()}
# Visible button-face centers relative to wall_panel (x, y, z), meters.
# x = link center 0.010 + half thickness 0.004 (front surface).
FACE_X = 0.014
BUTTON_LOCAL = {
    '3': (FACE_X, 0.0, 0.078), '2': (FACE_X, 0.0, 0.039),
    '1': (FACE_X, 0.0, 0.0),
    'open': (FACE_X, -0.0298, -0.039), 'close': (FACE_X, 0.0298, -0.039),
    'up': (FACE_X, 0.0, -0.078), 'down': (FACE_X, 0.0, -0.117),
}
BUTTON_SIZE = 0.033  # square button face, meters
# Regions that must NOT be labeled; recorded for false-positive analysis.
IGNORED_LOCAL = {
    'alarm_button': {'center': (0.012, -0.0298, 0.117),
                     'size': (0.033, 0.033, 0.033)},
    'intercom_button': {'center': (0.012, 0.0298, 0.117),
                        'size': (0.033, 0.033, 0.033)},
}
PANEL_BASE_POS = np.array([0.55, 0.03, 0.43])
PANEL_BASE_YAW = math.pi

IGN_CREATE = ['ign', 'service', '-s', '/world/button_press/create',
              '--reqtype', 'ignition.msgs.EntityFactory',
              '--reptype', 'ignition.msgs.Boolean', '--timeout', '10000']
IGN_REMOVE = ['ign', 'service', '-s', '/world/button_press/remove',
              '--reqtype', 'ignition.msgs.Entity',
              '--reptype', 'ignition.msgs.Boolean', '--timeout', '10000']

VARIANT_SDF = ('/workspace/ros2_ws/src/piper_elevator_gazebo/sim_variants/'
               'models/elevator_button_{v}/model.sdf')
ALL_PANEL_NAMES = tuple(['elevator_button'] +
                        [f'elevator_button_v{v}' for v in '01234'])


def quat_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rpy_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def tf_to_matrix(t):
    q = t.transform.rotation
    m = quat_to_matrix(q.x, q.y, q.z, q.w)
    p = t.transform.translation
    T = np.eye(4)
    T[:3, :3] = m
    T[:3, 3] = [p.x, p.y, p.z]
    return T


def scene_seed(light, variant, jitter, base_seed):
    tag = f'{light}|{variant}|{jitter}'.encode()
    return int(base_seed + zlib.crc32(tag)) % (2 ** 31)


class Collector(Node):
    def __init__(self):
        super().__init__('panel_data_collector')
        self.set_parameters([Parameter('use_sim_time', value=True)])
        self.bridge = CvBridge()
        self.joints = None
        self.latest_image = None
        self.latest_depth = None
        self.camera_info = None
        self.panel_js = None
        # Image subscriptions are created on demand: the gz->ROS bridge is
        # lazy, so sensor rendering (and its ~25x real-time-factor cost in
        # this container) only runs while a subscriber exists.
        self._image_subs = []
        # TF runs on a helper node with its own spin thread so that lookups
        # with a timeout work while the main node is blocked in actions.
        self.tf_helper = Node('panel_collector_tf')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer, self.tf_helper, spin_thread=True)
        self.create_subscription(JointState, '/piper_pika/joint_states',
                                 self._on_joints, 10)
        self.create_subscription(JointState, '/elevator_button/joint_states',
                                 self._on_panel_js, 10)
        self.arm = ActionClient(self, FollowJointTrajectory,
                                '/arm_controller/follow_joint_trajectory')

    def start_image_stream(self):
        """Subscribe to camera topics; rendering starts on the next frame."""
        if self._image_subs:
            return
        self.latest_image = None
        self.latest_depth = None
        self._image_subs = [
            self.create_subscription(Image, '/camera/color/image_raw',
                                     self._on_image, qos_profile_sensor_data),
            self.create_subscription(
                Image, '/camera/aligned_depth_to_color/image_raw',
                self._on_depth, qos_profile_sensor_data),
            self.create_subscription(CameraInfo, '/camera/color/camera_info',
                                     self._on_info, qos_profile_sensor_data),
        ]

    def stop_image_stream(self):
        """Unsubscribe; the lazy bridge stops relaying and rendering pauses."""
        for sub in self._image_subs:
            self.destroy_subscription(sub)
        self._image_subs = []
        self.latest_image = None
        self.latest_depth = None

    def _on_image(self, msg):
        self.latest_image = msg
        self._image_seq += 1

    _image_seq = 0

    def _on_depth(self, msg):
        self.latest_depth = msg

    def _on_info(self, msg):
        self.camera_info = msg

    def _on_joints(self, msg):
        values = dict(zip(msg.name, msg.position))
        if all(j in values for j in JOINTS):
            self.joints = {j: float(values[j]) for j in JOINTS}

    def _on_panel_js(self, msg):
        self.panel_js = dict(zip(msg.name, msg.position)) if msg.name else {}

    def spin_once(self, timeout):
        if rclpy.ok():
            rclpy.spin_once(self, timeout_sec=timeout)

    def spin_until(self, predicate, timeout, description):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if predicate():
                return
        raise TimeoutError(f'timed out waiting for {description}')

    def spin_for(self, seconds):
        deadline = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.05, max(0.0,
                                          deadline - time.monotonic())))

    def move_arm(self, positions, duration, timeout):
        """Send a trajectory and wait until the joints actually arrive.

        The action result response is unreliable here: with 30 Hz camera and
        depth subscriptions active, the small reliable response is starved by
        the image flood on this DDS domain. Goal acceptance still works and
        the controller always executes, so we poll joint feedback instead.
        """
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(JOINTS)
        point = JointTrajectoryPoint()
        point.positions = [positions[j] for j in JOINTS]
        point.time_from_start = Duration(seconds=duration).to_msg()
        goal.trajectory.points = [point]
        sent = self.arm.send_goal_async(goal)
        self.spin_until(sent.done, min(timeout, 10.0), 'trajectory acceptance')
        handle = sent.result()
        if not handle.accepted:
            raise RuntimeError('arm trajectory was rejected')
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.spin_once(0.05)
            if self.joints is not None and all(
                    abs(self.joints[j] - positions[j]) < 0.02 for j in JOINTS):
                return
        raise TimeoutError(
            'joints did not reach target; last='
            f'{self.joints} target={positions}')


def ign_create_panel(name, sdf_path, pos, quat):
    req = (f'sdf_filename: "{sdf_path}" name: "{name}" '
           f'pose {{ position {{ x: {pos[0]} y: {pos[1]} z: {pos[2]} }} '
           f'orientation {{ x: {quat[0]} y: {quat[1]} z: {quat[2]} w: {quat[3]} }} }}')
    result = subprocess.run(IGN_CREATE + ['--req', req],
                            capture_output=True, text=True, timeout=30)
    if 'data: true' not in result.stdout:
        raise RuntimeError(f'panel create failed: {result.stdout} {result.stderr}')


def ign_try_remove(name):
    req = f'name: "{name}" type: 2'
    result = subprocess.run(IGN_REMOVE + ['--req', req],
                            capture_output=True, text=True, timeout=30)
    return 'data: true' in result.stdout


def swap_panel(node, variant, pose7):
    """Replace the current panel with the requested variant (pose7: xyz+rpy)."""
    for name in ALL_PANEL_NAMES:
        ign_try_remove(name)  # missing entities are expected; ignore
    pos = pose7[:3]
    quat = rpy_to_quat(*pose7[3:])
    sdf = VARIANT_SDF.format(v=variant)
    ign_create_panel(f'elevator_button_v{variant}', sdf, pos, quat)
    node.panel_js = None
    node.spin_until(lambda: node.panel_js is not None, 15.0,
                    f'variant {variant} joint states')


def project_points(points_world, T_world_cam, K):
    """Project world points (N,3) into pixels; returns uv (N,2) and camera coords."""
    pts_cam = (T_world_cam[:3, :3] @ points_world.T).T + T_world_cam[:3, 3]
    uv = np.zeros((len(pts_cam), 2))
    valid = pts_cam[:, 2] > 1e-6
    uv[valid, 0] = K[0, 0] * pts_cam[valid, 0] / pts_cam[valid, 2] + K[0, 2]
    uv[valid, 1] = K[1, 1] * pts_cam[valid, 1] / pts_cam[valid, 2] + K[1, 2]
    return uv, pts_cam


def face_corners(center_local, size, R_panel, t_panel):
    """World-frame corners of a square panel-parallel face."""
    half = size / 2.0
    cx, cy, cz = center_local
    local = np.array([
        [cx, cy - half, cz + half], [cx, cy + half, cz + half],
        [cx, cy + half, cz - half], [cx, cy - half, cz - half],
    ])
    return (R_panel @ local.T).T + t_panel


def depth_at(depth_img, encoding, u, v, win=3):
    """Median depth in a small window around (u, v) in meters."""
    h, w = depth_img.shape[:2]
    x0, x1 = max(0, int(u) - win), min(w, int(u) + win + 1)
    y0, y1 = max(0, int(v) - win), min(h, int(v) + win + 1)
    patch = depth_img[y0:y1, x0:x1].astype(np.float64)
    if encoding == '16UC1':
        patch *= 0.001
    patch = patch[np.isfinite(patch) & (patch > 0.05)]
    return float(np.median(patch)) if patch.size else None


def build_labels(node, K, T_world_cam, R_panel, t_panel, depth_img, depth_enc,
                 image_shape):
    """Project every labeled button; returns YOLO rows, diagnostics, ignored."""
    H, W = image_shape
    rows, diagnostics, ignored = [], [], []
    for name, cls_id in CLASS_IDS.items():
        center = BUTTON_LOCAL[name]
        press = float((node.panel_js or {}).get(f'button_{name}_press_joint', 0.0))
        center = (center[0] + press, center[1], center[2])
        corners = face_corners(center, BUTTON_SIZE, R_panel, t_panel)
        uv, pts_cam = project_points(corners, T_world_cam, K)
        zvals = pts_cam[:, 2]
        diag = {'class': name}
        if np.any(zvals < 0.15):
            diag['status'] = 'behind_or_too_close'
            diagnostics.append(diag)
            continue
        x1, y1 = uv.min(axis=0)
        x2, y2 = uv.max(axis=0)
        cx_px, cy_px = float(uv[:, 0].mean()), float(uv[:, 1].mean())
        diag['raw_box'] = [round(float(x1), 1), round(float(y1), 1),
                           round(float(x2), 1), round(float(y2), 1)]
        if not (0 <= cx_px < W and 0 <= cy_px < H):
            diag['status'] = 'center_outside'
            diagnostics.append(diag)
            continue
        inside_frac = (
            (min(x2, W - 1) - max(x1, 0)) * (min(y2, H - 1) - max(y1, 0))
            / max((x2 - x1) * (y2 - y1), 1e-6)
        )
        if inside_frac < 0.4:
            diag['status'] = f'mostly_outside({inside_frac:.2f})'
            diagnostics.append(diag)
            continue
        if (x2 - x1) < 6 or (y2 - y1) < 6:
            diag['status'] = 'too_small'
            diagnostics.append(diag)
            continue
        expected = float(np.linalg.norm(pts_cam.mean(axis=0)))
        measured = (depth_at(depth_img, depth_enc, cx_px, cy_px)
                    if depth_img is not None else None)
        diag['range_m'] = round(expected, 3)
        if measured is not None:
            diag['measured_depth_m'] = round(measured, 3)
            if measured > expected + 0.05:
                diag['status'] = 'occluded'
                diagnostics.append(diag)
                continue
        bx1 = float(np.clip(x1, 0, W - 1)); by1 = float(np.clip(y1, 0, H - 1))
        bx2 = float(np.clip(x2, 0, W - 1)); by2 = float(np.clip(y2, 0, H - 1))
        diag['status'] = 'ok' if inside_frac > 0.999 else 'partial'
        diagnostics.append(diag)
        rows.append((cls_id,
                     (bx1 + bx2) / 2 / W, (by1 + by2) / 2 / H,
                     (bx2 - bx1) / W, (by2 - by1) / H))
    for iname, spec in IGNORED_LOCAL.items():
        corners = face_corners(spec['center'], spec['size'][1], R_panel, t_panel)
        uv, pts_cam = project_points(corners, T_world_cam, K)
        if np.all(pts_cam[:, 2] > 0.15):
            x1, y1 = uv.min(axis=0); x2, y2 = uv.max(axis=0)
            if x2 > 0 and y2 > 0 and x1 < W and y1 < H:
                ignored.append({'name': iname,
                                'box_px': [round(float(x1), 1), round(float(y1), 1),
                                           round(float(x2), 1), round(float(y2), 1)]})
    return rows, diagnostics, ignored


def draw_review(image, rows, ignored):
    out = image.copy()
    H, W = out.shape[:2]
    for cls_id, cx, cy, w, h in rows:
        x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * H)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, ID_NAMES[cls_id], (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    for region in ignored:
        x1, y1, x2, y2 = (int(round(v)) for v in region['box_px'])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 215, 255), 2)
        cv2.putText(out, 'ign:' + region['name'], (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1)
    return out


def jitter_pose(rng):
    return (
        PANEL_BASE_POS[0] + rng.uniform(-0.012, 0.012),
        PANEL_BASE_POS[1] + rng.uniform(-0.015, 0.015),
        PANEL_BASE_POS[2] + rng.uniform(-0.010, 0.010),
        rng.uniform(-0.03, 0.03),
        rng.uniform(-0.03, 0.03),
        PANEL_BASE_YAW + rng.uniform(-0.035, 0.035),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--light', required=True)
    parser.add_argument('--variants', default='v0')
    parser.add_argument('--jitters', type=int, default=1)
    parser.add_argument('--angles1', default='-15,-10,-5,0,5,10,15')
    parser.add_argument('--angles5', default='-5,0,5')
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--settle', type=float, default=0.8)
    parser.add_argument('--move-seconds', type=float, default=2.5)
    parser.add_argument('--no-swap', action='store_true')
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    angles1 = [float(a) for a in args.angles1.split(',')]
    angles5 = [float(a) for a in args.angles5.split(',')]
    out = Path(args.out)
    for sub in ('images', 'labels', 'meta', 'review'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    log_path = out / 'collection_log.jsonl'

    rclpy.init()
    node = Collector()
    start_pose = None
    try:
        node.start_image_stream()
        node.spin_until(lambda: node.joints is not None and node.latest_image is not None
                        and node.camera_info is not None and node.panel_js is not None,
                        90.0, 'camera, joints, and panel')
        if not node.arm.wait_for_server(timeout_sec=30.0):
            raise RuntimeError('arm trajectory action is unavailable')
        K = np.array(node.camera_info.k, dtype=float).reshape(3, 3)
        start_pose = dict(node.joints)
        node.stop_image_stream()
        print(f'K=fx:{K[0,0]:.1f} fy:{K[1,1]:.1f} cx:{K[0,2]:.1f} cy:{K[1,2]:.1f} '
              f'distortion={list(node.camera_info.d)}', flush=True)
        print(f'start_pose={start_pose}', flush=True)

        seq = 0
        with log_path.open('a', encoding='utf-8') as log:
            for variant in variants:
                for jit in range(args.jitters):
                    seed = scene_seed(args.light, variant, jit, args.seed)
                    rng = np.random.default_rng(seed)
                    pose7 = jitter_pose(rng)
                    if not args.no_swap:
                        swap_panel(node, variant, pose7)
                    node.spin_for(1.5)  # render settle after panel swap
                    R_panel = quat_to_matrix(*rpy_to_quat(*pose7[3:]))
                    t_panel = np.array(pose7[:3])
                    for a1 in angles1:
                        for a5 in angles5:
                            target = dict(start_pose)
                            target['joint1'] += math.radians(a1)
                            target['joint5'] += math.radians(a5)
                            node.move_arm(target, args.move_seconds, 30.0)
                            node.spin_for(args.settle)
                            image_msg = node.latest_image
                            image = node.bridge.imgmsg_to_cv2(
                                image_msg, desired_encoding='bgr8')
                            depth_img, depth_enc = None, None
                            if node.latest_depth is not None:
                                depth_enc = node.latest_depth.encoding
                                depth_img = node.bridge.imgmsg_to_cv2(
                                    node.latest_depth)
                            stamp = image_msg.header.stamp
                            T_world_cam = None
                            tf_mode = 'none'
                            try:
                                tf = node.tf_buffer.lookup_transform(
                                    'camera_color_optical_frame', 'world', stamp,
                                    timeout=Duration(seconds=1.0))
                                T_world_cam = tf_to_matrix(tf)
                                tf_mode = 'stamped'
                            except Exception as exc:
                                tf = node.tf_buffer.lookup_transform(
                                    'camera_color_optical_frame', 'world')
                                T_world_cam = tf_to_matrix(tf)
                                tf_mode = f'latest({type(exc).__name__})'
                            rows, diagnostics, ignored = build_labels(
                                node, K, T_world_cam, R_panel, t_panel,
                                depth_img, depth_enc, image.shape[:2])
                            name = (f'{args.light}_{variant}_j{jit}_'
                                    f'a{a1:+03.0f}_b{a5:+02.0f}_{seq:04d}')
                            cv2.imwrite(str(out / 'images' / f'{name}.png'), image)
                            with (out / 'labels' / f'{name}.txt').open('w') as f:
                                for cls_id, cx, cy, w, h in rows:
                                    f.write(f'{cls_id} {cx:.6f} {cy:.6f} '
                                            f'{w:.6f} {h:.6f}\n')
                            meta = {
                                'name': name,
                                'light': args.light,
                                'variant': variant,
                                'jitter_index': jit,
                                'scene_seed': seed,
                                'panel_pose': {'position': list(pose7[:3]),
                                               'rpy': list(pose7[3:])},
                                'joint_offsets_deg': {'joint1': a1, 'joint5': a5},
                                'joints_rad': {j: round(v, 5)
                                               for j, v in node.joints.items()},
                                'button_press_rad': node.panel_js,
                                'camera_K': [K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
                                'tf_world_to_camera': T_world_cam.tolist(),
                                'tf_mode': tf_mode,
                                'n_boxes': len(rows),
                                'boxes': [{'class': ID_NAMES[r[0]],
                                           'box_xywhn': [round(x, 5) for x in r[1:]]}
                                          for r in rows],
                                'ignored_regions': ignored,
                                'button_diagnostics': diagnostics,
                            }
                            (out / 'meta' / f'{name}.json').write_text(
                                json.dumps(meta, indent=1))
                            cv2.imwrite(str(out / 'review' / f'{name}.png'),
                                        draw_review(image, rows, ignored))
                            log.write(json.dumps({
                                'name': name, 'light': args.light,
                                'variant': variant, 'jitter': jit,
                                'a1': a1, 'a5': a5, 'n_boxes': len(rows)}) + '\n')
                            log.flush()
                            seq += 1
                            print(f'{name}: {len(rows)} boxes', flush=True)
        print(f'COLLECTION_DONE: {seq} frames -> {out}', flush=True)
    finally:
        if start_pose is not None:
            try:
                print('Returning to initial joint pose...', flush=True)
                node.move_arm(start_pose, args.move_seconds, 30.0)
            except Exception as exc:
                print(f'WARNING: could not return to initial pose: {exc}',
                      flush=True)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
