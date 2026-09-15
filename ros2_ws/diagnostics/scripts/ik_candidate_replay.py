#!/usr/bin/env python3
"""Replay a saved observation through current production IK; never command motion."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import rclpy
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from rosidl_runtime_py.convert import message_to_ordereddict

from ik_candidate_diagnostic import Diagnostics, FrozenPlanner, call_summary, json_value, pose_key
from piper_elevator_app.coarse_approach_core import CameraModel


DATA_DIR = Path(__file__).resolve().parents[1] / 'data'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=str(DATA_DIR / 'ik_candidate_diagnostic.json'))
    parser.add_argument('--output', default=str(DATA_DIR / 'ik_candidate_production_fix.json'))
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error('--output already exists; choose a new path')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(args.input).read_text())
    rclpy.init()
    node = Diagnostics()
    planner = FrozenPlanner(node)
    result = {'read_only': True, 'source': args.input, 'frozen_snapshot_stamp_ns': data['snapshot_stamp_ns'], 'selected_button': data['selected_button']}
    try:
        planner.values.update(data['parameters'])
        planner._base_frame = planner.values['base_frame']
        planner._end_effector_link = planner.values['end_effector_link']
        planner._workspace_min = np.array(planner.values['workspace_min'])
        planner._workspace_max = np.array(planner.values['workspace_max'])
        planner._arm_joint_limits = data['joint_limits']
        planner._latest_joint_positions = data['current']['joints']
        mount = data['transforms']['tip_to_camera']['transform']
        observed = {
            'button': np.array(data['observation']['button']),
            'normal': np.array(data['observation']['normal']),
            'camera_orientation': np.array(data['observation']['camera_orientation']),
            'tip_to_camera_translation': np.array([mount['translation'][key] for key in ('x', 'y', 'z')]),
            'tip_to_camera_quaternion': np.array([mount['rotation'][key] for key in ('x', 'y', 'z', 'w')]),
            'camera_model': CameraModel(**{key: data['camera_info'][key] for key in ('width', 'height', 'k', 'd', 'distortion_model')}),
            'camera_frame': planner.values['camera_frame'],
            'stamp_ns': data['snapshot_stamp_ns'],
            'selected_button': data['selected_button'],
        }
        planner._ik_client = node.create_client(GetPositionIK, planner.values['ik_service'])
        planner._fk_client = node.create_client(GetPositionFK, planner.values['fk_service'])
        if not planner._ik_client.wait_for_service(timeout_sec=8.):
            raise ValueError('IK service unavailable')
        targets = list(planner._candidate_poses(observed))
        planner.targets = {pose_key(pose): index for index, pose in enumerate(targets)}
        planner._planning_deadline = time.monotonic() + planner.values['planning_budget_seconds']
        started = time.monotonic()
        solutions = planner._solve_visible_candidates(observed)
        result.update({'elapsed_seconds': time.monotonic() - started, **call_summary(planner.calls), 'production_diagnostic': planner._last_ik_search_diagnostic, 'calls': planner.calls, 'solutions': []})
        for score, joints, target in solutions:
            actual = planner._fk_pose(joints)
            result['solutions'].append({'score': score, 'joints': joints, 'target': message_to_ordereddict(target), 'fk': message_to_ordereddict(actual), 'endpoint_validation': planner._validate_endpoint(actual, target, observed)})
    except Exception as error:
        result['error'] = repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(result, indent=2, default=json_value) + '\n')
        print(json.dumps({key: value for key, value in result.items() if key not in ('calls', 'solutions')}, indent=2, default=json_value))
        print('endpoint_validation', json.dumps([item['endpoint_validation'] for item in result.get('solutions', [])], default=json_value))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
