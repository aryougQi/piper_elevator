"""Replay saved geometry without modifying any live parameters or commands."""

import json
import math
from pathlib import Path

import numpy as np
import rclpy

from post_execution_snapshot import Diagnostics
from piper_elevator_app.coarse_approach_core import (
    CameraModel, check_camera_view, joint_configuration_is_safe,
    parse_arm_joint_limits,
)


def main():
    directory = Path(__file__).resolve().parents[1] / 'data'
    with open(directory / 'post_execution_geometry_diagnostic.json') as f:
        snapshot = json.load(f)
    with open(directory / 'post_execution_rgbd_diagnostic.json') as f:
        observations = json.load(f)
    info = observations['camera_info']['/camera/color/camera_info']
    model = CameraModel(**{k: info[k] for k in ('width', 'height', 'k', 'd', 'distortion_model')})
    p = snapshot['parameters']
    g = snapshot['geometry']
    def arrays(key):
        t = snapshot['transforms'][key]['transform']
        return [t['translation'][k] for k in ('x', 'y', 'z')], [t['rotation'][k] for k in ('x', 'y', 'z', 'w')]
    tip, q = arrays('base_to_tip')
    mount, mount_q = arrays('tip_to_camera')
    kwargs = dict(
        button_radius_m=p['visibility_button_radius_m'],
        position_uncertainty_m=p['visibility_position_uncertainty_m'],
        image_margin_ratio=p['visibility_image_margin_ratio'],
        minimum_depth_m=p['visibility_minimum_depth_m'],
        maximum_depth_m=p['visibility_maximum_depth_m'],
        maximum_tilt_rad=math.radians(15),
        maximum_roll_rad=p['maximum_camera_roll_rad'],
    )
    view = check_camera_view(g['base_button_m'], g['base_normal'], tip, q, mount, mount_q, model, **kwargs)
    result = {
        'read_only': True, 'source_snapshot_stamp_ns': snapshot['snapshot_stamp_ns'],
        'only_camera_limit_changed_in_replay': {'maximum_camera_tilt_rad': kwargs['maximum_tilt_rad']},
        'view_safe_at_15_degrees': bool(view[0]), 'view_message': view[1],
        'distance_to_servo_target_with_0_03m_standoff_m': float(np.linalg.norm(
            np.asarray(g['base_button_m']) - 0.03 * np.asarray(g['base_normal']) - np.asarray(tip))),
        'normal_standoff_exceeds_0_08m': g['tip_to_button_normal_distance_m'] >= 0.08,
        'limitation': 'Current capture geometry only; not the plan frozen target or its exact verification time.',
    }
    rclpy.init()
    node = Diagnostics()
    try:
        node.collect(1)
        params = node.parameters('/button_approach_planner', [
            'joint_limit_margin_rad', 'wrist_singularity_joint', 'minimum_abs_wrist_bend_rad',
        ])
        urdf = node.parameters('/move_group', ['robot_description'])['robot_description']
        joints = snapshot['topics']['/feedback/joint_states']
        positions = dict(zip(joints['name'], joints['position']))
        limits = parse_arm_joint_limits(urdf, joints['name'])
        margins = {name: min(positions[name] - bounds[0], bounds[1] - positions[name]) for name, bounds in limits.items()}
        safe, detail = joint_configuration_is_safe(
            positions, limits, params['joint_limit_margin_rad'],
            params['wrist_singularity_joint'], params['minimum_abs_wrist_bend_rad'],
        )
        result['joint_safety'] = {
            'parameters': params, 'safe': bool(safe), 'detail': detail,
            'limits_rad': limits, 'joint_margin_rad': margins,
            'minimum_margin_rad': min(margins.values()),
            'absolute_wrist_bend_rad': abs(positions[params['wrist_singularity_joint']]),
        }
    finally:
        node.destroy_node()
        rclpy.shutdown()
    output = directory / 'post_execution_replay_diagnostic.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
