#!/usr/bin/env python3
"""Read-only, frozen multi-scene regression of coarse-plan quality fallback."""

import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import rclpy
from moveit_msgs.action import MoveGroup
from moveit_msgs.srv import GetMotionPlan, GetPositionFK, GetPositionIK
from rclpy.node import Node
from rosidl_runtime_py.convert import message_to_ordereddict

from approach_quality_benchmark import build_observation, make_request
from ik_candidate_diagnostic import FrozenPlanner, json_value
from piper_elevator_app.approach_quality import configuration_quality, trajectory_quality
from piper_elevator_app.motion_core import matrix_to_quaternion, quaternion_to_matrix


class ServiceOnlyPlanner(FrozenPlanner):
    """Adapt the production baseline to GetMotionPlan; create no action clients."""

    def _plan_constraints(self, constraints, path_constraints=None):
        goal = {joint.joint_name: joint.position for joint in constraints.joint_constraints}
        request = make_request(self, goal, 'ompl', 'RRTConnectkConfigDefault')
        request.motion_plan_request.goal_constraints = [constraints]
        request.motion_plan_request.allowed_planning_time = max(.001, min(
            request.motion_plan_request.allowed_planning_time,
            self._planning_deadline - time.monotonic(),
        ))
        if path_constraints is not None:
            request.motion_plan_request.path_constraints = path_constraints
        response = self._call_moveit_service(
            self._quality_plan_client, request, self._planning_deadline,
        ).motion_plan_response
        if response.error_code.val != 1 or not response.trajectory.joint_trajectory.points:
            return None, f'MoveIt planning error {response.error_code.val}'
        result = MoveGroup.Result()
        result.error_code = response.error_code
        result.trajectory_start = response.trajectory_start
        result.planned_trajectory = response.trajectory
        result.planning_time = response.planning_time
        return result, ''

    def _publish_observation_diagnostics(self):
        pass


def scenes(data, low):
    base = build_observation(data)
    specifications = [
        ('recorded_high', 'recorded high button geometry', None, 0.),
        ('recorded_low', 'recorded historical low button geometry', low['geometry']['base_button_m'], 0.),
        ('synthetic_middle', 'synthetic middle target', [.56, .035, .35], 0.),
        ('synthetic_middle_left', 'synthetic middle target shifted +0.09m in base Y', [.56, .125, .35], 0.),
        ('synthetic_middle_right', 'synthetic middle target shifted -0.09m in base Y', [.56, -.055, .35], 0.),
        ('synthetic_high_normal_left', 'recorded high position with synthetic +8deg normal yaw', None, 8.),
        ('synthetic_high_normal_right', 'recorded high position with synthetic -8deg normal yaw', None, -8.),
    ]
    for name, provenance, button, yaw in specifications:
        observed = copy.deepcopy(base)
        if button is not None:
            observed['button'] = np.array(button, dtype=float)
        if name == 'recorded_low':
            observed['normal'] = np.array(low['geometry']['base_normal'])
        elif yaw:
            angle = math.radians(yaw)
            rotation = np.array([[math.cos(angle), -math.sin(angle), 0.],
                                 [math.sin(angle), math.cos(angle), 0.],
                                 [0., 0., 1.]])
            observed['normal'] = rotation @ observed['normal']
        observed['selected_button'] = name
        yield name, provenance, observed


def quality(planner, result):
    trajectory = result.planned_trajectory
    endpoint = dict(zip(trajectory.joint_trajectory.joint_names, trajectory.joint_trajectory.points[-1].positions))
    configuration = configuration_quality(
        endpoint, planner._latest_joint_positions, planner._arm_joint_limits,
        minimum_abs_wrist_bend=planner.values['minimum_abs_wrist_bend_rad'],
    )
    path = trajectory_quality(trajectory, start_positions=planner._latest_joint_positions)
    end_time = trajectory.joint_trajectory.points[-1].time_from_start
    return {
        'configuration': configuration,
        'path': path,
        'combined_cost': configuration['cost'] + path['cost'],
        'duration_seconds': end_time.sec + end_time.nanosec * 1.e-9,
        'endpoint': endpoint,
    }


DATA_DIR = Path(__file__).resolve().parents[1] / 'data'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=str(DATA_DIR / 'ik_candidate_diagnostic.json'))
    parser.add_argument('--low-input', default=str(DATA_DIR / 'post_execution_geometry_diagnostic.json'))
    parser.add_argument('--previous-benchmark', default=str(DATA_DIR / 'approach_quality_benchmark.json'))
    parser.add_argument('--output', default=str(DATA_DIR / 'approach_quality_matrix.json'))
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error('--output already exists; choose a new path')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(args.input).read_text())
    low = json.loads(Path(args.low_input).read_text())
    prior = json.loads(Path(args.previous_benchmark).read_text())
    bent = next(row['joints'] for row in prior['selected_candidates'] if row['label'] == 'balanced_twist_bend')
    starts = [('home', data['current']['joints']), ('verified_bent_approach', bent)]
    rclpy.init()
    node = Node('approach_quality_readonly_matrix')
    planner = ServiceOnlyPlanner(node)
    planner.values.update(data['parameters'])
    planner._base_frame = planner.values['base_frame']
    planner._end_effector_link = planner.values['end_effector_link']
    planner._workspace_min = np.array(planner.values['workspace_min'])
    planner._workspace_max = np.array(planner.values['workspace_max'])
    planner._arm_joint_limits = data['joint_limits']
    planner._ik_client = node.create_client(GetPositionIK, planner.values['ik_service'])
    planner._fk_client = node.create_client(GetPositionFK, planner.values['fk_service'])
    planner._quality_plan_client = node.create_client(GetMotionPlan, '/plan_kinematic_path')
    result = {
        'read_only': True,
        'source_files': [args.input, args.low_input, args.previous_benchmark],
        'services_called': ['/compute_ik', '/compute_fk', '/plan_kinematic_path'],
        'limitations': [
            'Fourteen synthetic/frozen planning cases are not a statistical guarantee for arbitrary target positions or executed motion.',
            'Two historical real target geometries and five explicitly synthetic geometries; all use current calibration and current MoveIt collision scene.',
            'Home and one prior FK/collision-validated non-Home configuration; robot does not physically move between start states.',
            'Frozen geometry bypasses live acquisition and target freshness; all geometric endpoint, joint, collision planner and trajectory checks remain active.',
            'The baseline is adapted from the production plan-only MoveGroup action to equivalent computation-only GetMotionPlan request.',
            'Improvement invokes the actual production _improve_coarse_plan and retains its validated baseline on optional failure.',
        ],
        'parameters': planner.values.copy(),
        'starts': dict(starts),
        'cases': [],
    }
    try:
        if not hasattr(planner, '_improve_coarse_plan'):
            raise ValueError('Production quality integration not yet available')
        for start_name, start in starts:
            planner._latest_joint_positions = start.copy()
            planner._planning_deadline = time.monotonic() + 10.
            start_pose = planner._fk_pose(start)
            q = start_pose.pose.orientation
            start_rotation = quaternion_to_matrix([q.x, q.y, q.z, q.w])
            for scene_name, provenance, observed in scenes(data, low):
                observed['camera_orientation'] = matrix_to_quaternion(
                    start_rotation @ quaternion_to_matrix(observed['tip_to_camera_quaternion'])
                )
                planner._last_plan_quality_diagnostic = {}
                planner._planning_deadline = time.monotonic() + planner.values['planning_budget_seconds']
                started = time.monotonic()
                row = {
                    'scene': scene_name,
                    'provenance': provenance,
                    'start': start_name,
                    'button': observed['button'].tolist(),
                    'normal': observed['normal'].tolist(),
                    'baseline_success': False,
                    'final_success': False,
                    'baseline_attempts': [],
                }
                try:
                    candidates = planner._solve_visible_candidates(observed)
                    row['ik_diagnostic'] = copy.deepcopy(planner._last_ik_search_diagnostic)
                    baseline = None
                    for index, (_, joints, target) in enumerate(candidates[:planner.values['maximum_candidate_plans']]):
                        planned, message = planner._plan_constraints(planner._joint_goal_constraints(joints))
                        accepted = False
                        if planned is not None:
                            accepted, message = planner._validate_planned_candidate(planned, target, observed)
                        row['baseline_attempts'].append({'candidate_index': index, 'accepted': accepted, 'message': message})
                        if accepted:
                            baseline = planned
                            row['baseline_success'] = True
                            break
                    if baseline is not None:
                        row['baseline_quality'] = quality(planner, baseline)
                        improved, selected_target, detail = planner._improve_coarse_plan(
                            baseline, target, observed, candidates, message,
                        )
                        row['quality_diagnostic'] = copy.deepcopy(planner._last_plan_quality_diagnostic)
                        row['retained_same_baseline_object'] = improved is baseline
                        row['final_success'], row['final_validation'] = planner._validate_planned_candidate(improved, selected_target, observed)
                        row['final_quality'] = quality(planner, improved)
                        row['final_message'] = detail
                        row['baseline_trajectory'] = message_to_ordereddict(baseline.planned_trajectory)
                        row['chosen_trajectory'] = message_to_ordereddict(improved.planned_trajectory)
                except Exception as error:
                    row['error'] = repr(error)
                row['elapsed_seconds'] = time.monotonic() - started
                result['cases'].append(row)
                print(json.dumps({
                    'scene': scene_name, 'start': start_name,
                    'baseline_success': row['baseline_success'],
                    'final_success': row['final_success'],
                    'baseline_retained': row.get('retained_same_baseline_object'),
                    'baseline_attempts': row['baseline_attempts'],
                    'quality_reason': row.get('quality_diagnostic', {}).get('reason'),
                    'error': row.get('error'),
                }, default=json_value), flush=True)
        rows = result['cases']
        result['summary'] = {
            'cases': len(rows),
            'baseline_successes': sum(row['baseline_success'] for row in rows),
            'baseline_plus_improvement_successes': sum(row['final_success'] for row in rows),
            'baseline_successes_lost': sum(row['baseline_success'] and not row['final_success'] for row in rows),
            'baseline_retained': sum(row.get('retained_same_baseline_object', False) for row in rows),
            'quality_replacements': sum(row['final_success'] and not row.get('retained_same_baseline_object', True) for row in rows),
            'unexpected_errors': sum('error' in row for row in rows),
        }
        print(json.dumps(result['summary']), flush=True)
    finally:
        Path(args.output).write_text(json.dumps(result, indent=2, default=json_value) + '\n')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
