#!/usr/bin/env python3
"""Read-only comparison of a frozen plan point with fresh RGB-D geometry."""

from collections import OrderedDict
import json
from pathlib import Path
import time

from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection2DArray

from piper_elevator_app.detector_core import Detection, project_camera_point, project_pixel, robust_box_depth
from piper_elevator_app.motion_core import quaternion_to_matrix


def stamp(message):
    return Time.from_msg(message.header.stamp).nanoseconds


def main():
    rclpy.init()
    node = Node('coarse_projection_readonly')
    buffer = Buffer(cache_time=Duration(seconds=20))
    listener = TransformListener(buffer, node)
    bridge = CvBridge()
    depths, detections = OrderedDict(), OrderedDict()
    state, camera, base = {}, {}, {}
    qos = QoSProfile(depth=3, reliability=ReliabilityPolicy.BEST_EFFORT)
    subscriptions = []

    def cache_depth(message):
        depths[stamp(message)] = message
        while len(depths) > 120:
            depths.popitem(last=False)

    def cache_detections(message):
        detections[stamp(message)] = message
        while len(detections) > 80:
            detections.popitem(last=False)

    def camera_info(message):
        camera.update(k=list(message.k), d=list(message.d), model=message.distortion_model,
                      frame=message.header.frame_id)

    def base_point(message):
        base.update(point=[message.pose.position.x, message.pose.position.y, message.pose.position.z],
                    stamp_ns=stamp(message), frame=message.header.frame_id)

    subscriptions.append(node.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', cache_depth, qos))
    subscriptions.append(node.create_subscription(Detection2DArray, '/button_detections/raw', cache_detections, qos))
    subscriptions.append(node.create_subscription(CameraInfo, '/camera/color/camera_info', camera_info, qos))
    subscriptions.append(node.create_subscription(String, '/button_approach_planner/observation_status',
                                                  lambda message: state.update(json.loads(message.data)), qos))
    subscriptions.append(node.create_subscription(PoseStamped, '/button_pose_base', base_point,
                         QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)))
    deadline = time.monotonic() + 7
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=.02)
    if not camera or not state:
        raise RuntimeError('CameraInfo or planner status unavailable')
    matrix = np.asarray(camera['k']).reshape(3, 3)
    distortion = np.asarray(camera['d'])
    frozen = np.asarray(state['last_execution']['planned_button'])
    rows, errors = [], []
    for capture, message in detections.items():
        if capture not in depths:
            continue
        try:
            tf = buffer.lookup_transform('base_link', camera['frame'], Time(nanoseconds=capture)).transform
        except TransformException as error:
            errors.append(str(error))
            continue
        rotation = quaternion_to_matrix([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w])
        translation = np.array([tf.translation.x, tf.translation.y, tf.translation.z])
        depth = bridge.imgmsg_to_cv2(depths[capture], 'passthrough')
        for item in message.detections:
            if not item.results or item.results[0].hypothesis.class_id != state['selected_button']:
                continue
            u, v = item.bbox.center.position.x, item.bbox.center.position.y
            w, h = item.bbox.size_x, item.bbox.size_y
            box = Detection(u-w/2, v-h/2, u+w/2, v+h/2, item.results[0].hypothesis.score, 0, state['selected_button'])
            z = robust_box_depth(depth, box, .001, .45, .1, 2., 20, method='weighted')
            if z is None:
                continue
            measured_camera = project_pixel(matrix, u, v, z, distortion, camera['model'])
            measured_base = translation + rotation @ measured_camera
            frozen_camera = rotation.T @ (frozen-translation)
            expected = project_camera_point(matrix, frozen_camera, distortion, camera['model'])
            row = {'stamp_ns': capture, 'pixel': [u, v], 'depth_m': z,
                   'measured_base': measured_base.tolist(), 'planned_pixel': expected.tolist(),
                   'pixel_error': float(np.linalg.norm(np.array([u,v])-expected)),
                   'base_error_mm': ((measured_base-frozen)*1000).tolist(),
                   'base_distance_mm': float(np.linalg.norm(measured_base-frozen)*1000),
                   'camera_translation': translation.tolist(), 'camera_rotation': rotation.tolist()}
            if base:
                row['latched_base_pixel'] = project_camera_point(
                    matrix, rotation.T @ (np.asarray(base['point'])-translation), distortion, camera['model']).tolist()
            rows.append(row)
    result = {'selected': state['selected_button'], 'planned_button': frozen.tolist(),
              'latched_base': base, 'samples': rows, 'tf_errors': errors,
              'last_execution': state.get('last_execution')}
    output = Path(__file__).resolve().parents[1] / 'data/coarse_detection_loss/projection_geometry.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({'selected': result['selected'], 'planned_button': result['planned_button'],
                      'latched_base': base, 'sample_count': len(rows), 'sample': rows[-1] if rows else None,
                      'tf_error_count': len(errors)}, indent=2))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
