"""Read-only, one-shot geometry snapshot for the 2026-09-08 execution failure."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import rclpy
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionFK
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from rosidl_runtime_py.convert import message_to_ordereddict
from sensor_msgs.msg import JointState
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src/piper_elevator_app/scripts'))
from diagnose_rgbd import Diagnostics, quaternion
from piper_elevator_app.coarse_approach_core import CameraModel, check_camera_view
from piper_elevator_app.motion_core import camera_level_roll_error, quaternion_to_matrix


DATA_DIR = Path(__file__).resolve().parents[1] / 'data'


def arrays(transform):
    t = transform.transform
    return np.array([t.translation.x, t.translation.y, t.translation.z]), quaternion(t.rotation)


def main():
    rclpy.init()
    node = Diagnostics()
    messages = {}
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
    for topic, kind in [
        ('/button_approach_pose', PoseStamped),
        ('/button_approach/status', String),
        ('/button_approach_planner/observation_status', String),
    ]:
        node.subscriptions_owned.append(node.create_subscription(
            kind, topic,
            lambda message, topic=topic: messages.update({topic: message_to_ordereddict(message)}),
            qos,
        ))
    for topic, kind in [
        ('/arm_controller/controller_state', JointTrajectoryControllerState),
        ('/feedback/joint_states', JointState),
        ('/piper_pika/joint_states', JointState),
    ]:
        node.subscriptions_owned.append(node.create_subscription(
            kind, topic,
            lambda message, topic=topic: messages.update({topic: message_to_ordereddict(message)}),
            10,
        ))
    try:
        node.collect(3)
        names = [
            'base_frame', 'end_effector_link', 'camera_frame', 'approach_distance_m',
            'observation_stable_samples', 'observation_minimum_samples',
            'observation_window_max_seconds', 'observation_normal_tolerance_rad',
            'post_execution_observation_timeout_seconds', 'max_target_drift_m',
            'maximum_execution_position_error_m', 'maximum_execution_orientation_error_rad',
            'maximum_camera_tilt_rad', 'maximum_camera_roll_rad',
            'visibility_button_radius_m', 'visibility_position_uncertainty_m',
            'visibility_image_margin_ratio', 'visibility_minimum_depth_m',
            'visibility_maximum_depth_m', 'execution_joint_tolerance_rad',
            'execution_stable_velocity_rad_s', 'execution_stop_hold_seconds',
        ]
        params = node.parameters('/button_approach_planner', names)
        base, tip, camera = [params[n] for n in ('base_frame', 'end_effector_link', 'camera_frame')]
        transforms = {
            'base_to_tip': node.tf_buffer.lookup_transform(base, tip, Time()),
            'base_to_camera': node.tf_buffer.lookup_transform(base, camera, Time()),
            'tip_to_camera': node.tf_buffer.lookup_transform(tip, camera, Time()),
        }
        result = {
            'read_only': True,
            'snapshot_stamp_ns': node.get_clock().now().nanoseconds,
            'parameters': params,
            'topics': messages,
            'transforms': {key: message_to_ordereddict(value) for key, value in transforms.items()},
            'limitations': [
                'Snapshot collected after the failure; current TF is not the exact execution-end TF.',
                'The current /button_approach_pose can be replaced by a nominal observation target after execution.',
                'Previous rgbd_diagnostic.json is a separate observation session, not this plan frozen observation.',
            ],
        }
        with open(DATA_DIR / 'post_execution_rgbd_diagnostic.json') as f:
            report = json.load(f)
        with open(DATA_DIR / 'rgbd_diagnostic.json') as f:
            previous = json.load(f)
        button = np.asarray(report['base_position']['median_m'])
        normal = np.asarray(report['base_normal']['mean_unit'])
        old_normal = np.asarray(previous['base_normal']['mean_unit'])
        tool_position, tool_q = arrays(transforms['base_to_tip'])
        camera_position, camera_q = arrays(transforms['base_to_camera'])
        mount_position, mount_q = arrays(transforms['tip_to_camera'])
        camera_rotation = quaternion_to_matrix(camera_q)
        tip_to_button = button - tool_position
        camera_point = camera_rotation.T @ (button - camera_position)
        info = node.camera_info['/camera/color/camera_info']
        model = CameraModel(**{key: info[key] for key in ('width', 'height', 'k', 'd', 'distortion_model')})
        kwargs = {
            'button_radius_m': params['visibility_button_radius_m'],
            'position_uncertainty_m': params['visibility_position_uncertainty_m'],
            'image_margin_ratio': params['visibility_image_margin_ratio'],
            'minimum_depth_m': params['visibility_minimum_depth_m'],
            'maximum_depth_m': params['visibility_maximum_depth_m'],
            'maximum_tilt_rad': params['maximum_camera_tilt_rad'],
            'maximum_roll_rad': params['maximum_camera_roll_rad'],
        }
        view = check_camera_view(button, normal, tool_position, tool_q, mount_position, mount_q, model, **kwargs)
        angle = lambda x, y: math.degrees(math.acos(float(np.clip(np.dot(x, y), -1, 1))))
        normal_distance = float(tip_to_button @ normal)
        result['geometry'] = {
            'base_button_m': button.tolist(), 'base_normal': normal.tolist(),
            'tip_to_button_distance_m': float(np.linalg.norm(tip_to_button)),
            'tip_to_button_normal_distance_m': normal_distance,
            'tip_to_button_tangential_distance_m': float(np.linalg.norm(tip_to_button - normal_distance * normal)),
            'camera_point_m': camera_point.tolist(),
            'camera_tilt_degrees': angle(camera_rotation[:, 2], normal),
            'camera_roll_degrees': math.degrees(camera_level_roll_error(camera_q, normal, [0, 0, 1])),
            'view_safe': bool(view[0]), 'view_message': view[1],
            'previous_session_normal_difference_degrees': angle(normal, old_normal),
            'previous_session_button_difference_mm': 1000 * float(np.linalg.norm(button - np.asarray(previous['base_position']['median_m']))),
        }
        pose_msg = messages.get('/button_approach_pose')
        if pose_msg:
            pose = pose_msg['pose']
            target_position = np.asarray([pose['position'][key] for key in ('x', 'y', 'z')])
            target_q = np.asarray([pose['orientation'][key] for key in ('x', 'y', 'z', 'w')])
            result['geometry']['distance_from_current_published_approach_m'] = float(np.linalg.norm(tool_position - target_position))
            result['geometry']['angle_from_current_published_approach_degrees'] = math.degrees(2 * math.acos(float(np.clip(abs(np.dot(target_q, tool_q)), 0, 1))))
        controller = messages.get('/arm_controller/controller_state')
        if controller:
            # Forward kinematics only calculates a pose; it cannot command motion.
            client = node.create_client(GetPositionFK, '/compute_fk')
            if client.wait_for_service(timeout_sec=2):
                request = GetPositionFK.Request()
                request.header.frame_id = base
                request.fk_link_names = [tip]
                request.robot_state.joint_state.name = controller['joint_names']
                desired = controller.get('reference', controller.get('desired'))
                request.robot_state.joint_state.position = desired['positions']
                request.robot_state.is_diff = True
                future = client.call_async(request)
                rclpy.spin_until_future_complete(node, future, timeout_sec=3)
                if future.done() and future.result() is not None:
                    fk = future.result()
                    result['controller_final_reference_fk'] = message_to_ordereddict(fk)
                    if fk.error_code.val == 1 and fk.pose_stamped:
                        target = fk.pose_stamped[0].pose
                        target_position = np.array([target.position.x, target.position.y, target.position.z])
                        target_q = quaternion(target.orientation)
                        result['geometry']['distance_from_controller_reference_m'] = float(np.linalg.norm(tool_position - target_position))
                        result['geometry']['angle_from_controller_reference_degrees'] = math.degrees(2 * math.acos(float(np.clip(abs(np.dot(target_q, tool_q)), 0, 1))))
            node.destroy_client(client)
        output = DATA_DIR / 'post_execution_geometry_diagnostic.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
