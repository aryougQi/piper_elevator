#!/usr/bin/env python3
"""Compare frozen coarse targets through computation-only MoveIt services."""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetMotionPlan, GetPositionFK
from rclpy.node import Node
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.set_message import set_message_fields

from ik_candidate_diagnostic import FrozenPlanner, json_value
from piper_elevator_app.coarse_approach_core import CameraModel


def build_observation(data):
    mount = data['transforms']['tip_to_camera']['transform']
    return {
        'button': np.array(data['observation']['button']),
        'normal': np.array(data['observation']['normal']),
        'camera_orientation': np.array(data['observation']['camera_orientation']),
        'tip_to_camera_translation': np.array([mount['translation'][key] for key in ('x', 'y', 'z')]),
        'tip_to_camera_quaternion': np.array([mount['rotation'][key] for key in ('x', 'y', 'z', 'w')]),
        'camera_model': CameraModel(**{key: data['camera_info'][key] for key in ('width', 'height', 'k', 'd', 'distortion_model')}),
        'camera_frame': data['parameters']['camera_frame'],
        'stamp_ns': data['snapshot_stamp_ns'],
        'selected_button': data['selected_button'],
    }


def candidate_metrics(row, start, names, minimum_bend):
    joints = row['joints']
    delta = {name: float(joints[name] - start[name]) for name in names}
    return {
        'joint_delta_rad': delta,
        'wrist_twist_travel_rad': abs(delta['joint4']) + abs(delta['joint6']),
        'wrist_bend_rad': abs(joints['joint5']),
        'wrist_bend_extra_margin_rad': abs(joints['joint5']) - minimum_bend,
        'maximum_joint_travel_rad': max(abs(value) for value in delta.values()),
        'joint_distance_l1_rad': sum(abs(value) for value in delta.values()),
        'joint_distance_l2_rad': math.sqrt(sum(value * value for value in delta.values())),
    }


def trajectory_metrics(trajectory):
    jt = trajectory.joint_trajectory
    position = np.array([point.positions for point in jt.points])
    delta = np.diff(position, axis=0)
    travel = np.abs(delta).sum(axis=0)
    direct = np.abs(position[-1] - position[0])
    endpoint = jt.points[-1].time_from_start
    reversals = {}
    for i, name in enumerate(jt.joint_names):
        signs = np.sign(delta[np.abs(delta[:, i]) > .002, i])
        reversals[name] = int(np.count_nonzero(np.diff(signs)))
    return {
        'point_count': len(jt.points),
        'duration_seconds': endpoint.sec + endpoint.nanosec * 1.e-9,
        'joint_path_l1_rad': float(travel.sum()),
        'joint_path_l2_rad': float(np.linalg.norm(delta, axis=1).sum()),
        'joint_direct_l1_rad': float(direct.sum()),
        'joint_backtracking_rad': float((travel - direct).sum()),
        'joint_path_detour_ratio': float(np.linalg.norm(delta, axis=1).sum() / max(np.linalg.norm(position[-1] - position[0]), 1.e-9)),
        'wrist_twist_travel_rad': float(sum(travel[jt.joint_names.index(name)] for name in ('joint4', 'joint6'))),
        'travel_by_joint_rad': dict(zip(jt.joint_names, travel.tolist())),
        'direction_reversals_over_2mrad': reversals,
    }


def make_request(planner, joints, pipeline, algorithm):
    request = GetMotionPlan.Request()
    motion = request.motion_plan_request
    motion.group_name = planner.values['planning_group']
    motion.pipeline_id = pipeline
    motion.planner_id = algorithm
    motion.num_planning_attempts = int(planner.values['planning_attempts'])
    motion.allowed_planning_time = float(planner.values['planning_time_seconds'])
    motion.max_velocity_scaling_factor = float(planner.values['velocity_scaling'])
    motion.max_acceleration_scaling_factor = float(planner.values['acceleration_scaling'])
    motion.start_state.is_diff = True
    motion.start_state.joint_state.name = list(planner._latest_joint_positions)
    motion.start_state.joint_state.position = list(planner._latest_joint_positions.values())
    motion.workspace_parameters.header.frame_id = planner._base_frame
    motion.workspace_parameters.min_corner.x, motion.workspace_parameters.min_corner.y, motion.workspace_parameters.min_corner.z = planner._workspace_min.tolist()
    motion.workspace_parameters.max_corner.x, motion.workspace_parameters.max_corner.y, motion.workspace_parameters.max_corner.z = planner._workspace_max.tolist()
    motion.goal_constraints = [planner._joint_goal_constraints(joints)]
    return request


DATA_DIR = Path(__file__).resolve().parents[1] / 'data'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=str(DATA_DIR / 'ik_candidate_diagnostic.json'))
    parser.add_argument('--baseline', default=str(DATA_DIR / 'ik_candidate_production_fix.json'))
    parser.add_argument('--output', default=str(DATA_DIR / 'approach_quality_benchmark.json'))
    parser.add_argument('--repeats', type=int, default=3, choices=(1, 2, 3))
    parser.add_argument('--pilz', action='store_true')
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error('--output already exists; choose a new path')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(args.input).read_text())
    baseline = min(json.loads(Path(args.baseline).read_text())['solutions'], key=lambda row: row['score'])
    rclpy.init()
    node = Node('approach_quality_readonly_benchmark')
    planner = FrozenPlanner(node)
    planner.values.update(data['parameters'])
    planner._base_frame = planner.values['base_frame']
    planner._end_effector_link = planner.values['end_effector_link']
    planner._workspace_min = np.array(planner.values['workspace_min'])
    planner._workspace_max = np.array(planner.values['workspace_max'])
    planner._arm_joint_limits = data['joint_limits']
    planner._latest_joint_positions = data['current']['joints'].copy()
    planner._fk_client = node.create_client(GetPositionFK, planner.values['fk_service'])
    client = node.create_client(GetMotionPlan, '/plan_kinematic_path')
    observed = build_observation(data)
    names = planner.values['home_joint_names']
    reserve = planner.values['joint_goal_tolerance_rad'] + planner.values['execution_joint_tolerance_rad']
    minimum_bend = planner.values['minimum_abs_wrist_bend_rad'] + reserve
    unique = []
    for row in data['expanded_tilt_roll']['calls']:
        if row['code'] != 1 or not row['joint_safe']:
            continue
        if any(max(abs(row['joints'][name] - prior['joints'][name]) for name in names) < .01 for prior in unique):
            continue
        unique.append(row)
    for row in unique:
        row['quality'] = candidate_metrics(row, planner._latest_joint_positions, names, minimum_bend)
    low_twist = min(unique, key=lambda row: row['quality']['wrist_twist_travel_rad'])
    balanced = min((row for row in unique if row['quality']['wrist_bend_extra_margin_rad'] >= .08), key=lambda row: row['quality']['wrist_twist_travel_rad'])
    chosen = [('baseline_first', baseline), ('minimum_twist', low_twist), ('balanced_twist_bend', balanced)]
    result = {
        'read_only': True,
        'source': args.input,
        'baseline_source': args.baseline,
        'frozen_snapshot_stamp_ns': data['snapshot_stamp_ns'],
        'selected_button': data['selected_button'],
        'planning_start': planner._latest_joint_positions,
        'services_called': ['/plan_kinematic_path', planner.values['fk_service']],
        'limitations': [
            'Frozen observed geometry and Home joint start; current MoveIt planning scene.',
            'Planning only; no commands, action clients, publishers, robot execution, or online parameter changes.',
            'Bounded sample does not establish an execution success rate or cover other buttons/views.',
            'Waypoint path metrics do not prove acceleration or jerk continuity on the real controller.',
        ],
        'planning_parameters': {key: planner.values[key] for key in ('planning_group', 'planning_time_seconds', 'planning_attempts', 'velocity_scaling', 'acceleration_scaling', 'action_timeout_seconds')},
        'expanded_unique_candidates': [{'joints': row['joints'], 'target': row['target'], 'quality': row['quality']} for row in unique],
        'selected_candidates': [],
        'runs': [],
    }
    try:
        if not client.wait_for_service(timeout_sec=8.):
            raise ValueError('Computation-only GetMotionPlan service unavailable')
        for label, row in chosen:
            pose = PoseStamped()
            set_message_fields(pose, row['target'])
            planner._planning_deadline = time.monotonic() + 10.
            actual = planner._fk_pose(row['joints'])
            candidate = {
                'label': label,
                'joints': row['joints'],
                'target': row['target'],
                'quality': candidate_metrics(row, planner._latest_joint_positions, names, minimum_bend),
                'joint_validation': planner._joint_configuration_is_safe(row['joints'], reserve_rad=reserve),
                'endpoint_validation': planner._validate_endpoint(actual, pose, observed),
            }
            result['selected_candidates'].append(candidate)
            print(json.dumps(candidate, default=json_value), flush=True)
        runs = [(label, row, 'ompl', 'RRTConnectkConfigDefault', repeat) for repeat in range(args.repeats) for label, row in chosen]
        if args.pilz:
            runs.extend((label, row, 'pilz_industrial_motion_planner', 'PTP', 0) for label, row in chosen if label != 'minimum_twist')
        for label, row, pipeline, algorithm, repeat in runs:
            pose = PoseStamped()
            set_message_fields(pose, row['target'])
            request = make_request(planner, row['joints'], pipeline, algorithm)
            started = time.monotonic()
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=8.)
            run = {'candidate': label, 'pipeline_id': pipeline, 'planner_id': algorithm, 'repeat': repeat, 'elapsed_seconds': time.monotonic() - started}
            if not future.done() or future.result() is None:
                client.remove_pending_request(future)
                run['error'] = 'Computation-only planning service timed out'
            else:
                response = future.result().motion_plan_response
                run.update({'error_code': response.error_code.val, 'planning_time_seconds': response.planning_time})
                trajectory = response.trajectory
                if response.error_code.val == 1 and trajectory.joint_trajectory.points:
                    run['quality'] = trajectory_metrics(trajectory)
                    run['trajectory_validation'] = planner._trajectory_wrist_is_safe(trajectory)
                    run['duration_validation'] = 0. < run['quality']['duration_seconds'] <= planner.values['action_timeout_seconds']
                    endpoint = dict(zip(trajectory.joint_trajectory.joint_names, trajectory.joint_trajectory.points[-1].positions))
                    planner._planning_deadline = time.monotonic() + 10.
                    run['endpoint_validation'] = planner._validate_endpoint(planner._fk_pose(endpoint), pose, observed, position_tolerance=planner.values['position_tolerance_m'])
                    run['start_maximum_error_rad'] = max(abs(value - planner._latest_joint_positions[name]) for name, value in zip(trajectory.joint_trajectory.joint_names, trajectory.joint_trajectory.points[0].positions))
                    run['accepted'] = bool(run['trajectory_validation'][0] and run['duration_validation'] and run['endpoint_validation'][0] and run['start_maximum_error_rad'] <= planner.values['execution_start_tolerance_rad'])
                    run['trajectory'] = message_to_ordereddict(trajectory)
            result['runs'].append(run)
            print(json.dumps({key: value for key, value in run.items() if key != 'trajectory'}, default=json_value), flush=True)
    except Exception as error:
        result['error'] = repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(result, indent=2, default=json_value) + '\n')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
