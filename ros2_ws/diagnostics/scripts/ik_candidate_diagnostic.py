#!/usr/bin/env python3
"""Read-only candidate reproduction: subscriptions and IK/FK/validity only."""

import argparse
from array import array
from collections import Counter
import json
import math
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import rclpy
from moveit_msgs.srv import GetPositionFK, GetPositionIK, GetStateValidity
from moveit_msgs.msg import Constraints, JointConstraint
from rcl_interfaces.srv import GetParameters
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from rosidl_runtime_py.convert import message_to_ordereddict
from sensor_msgs.msg import JointState
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src/piper_elevator_app/scripts'))
from diagnose_rgbd import Diagnostics, quaternion
from piper_elevator_app.button_approach_planner import ButtonApproachPlanner
from piper_elevator_app.coarse_approach_core import CameraModel, stable_observation_window
from piper_elevator_app.motion_core import camera_level_roll_error, level_limited_camera_orientation, quaternion_to_matrix


def arrays(transform):
    tf = transform.transform
    return np.array([tf.translation.x, tf.translation.y, tf.translation.z]), np.array(quaternion(tf.rotation))


def pose_key(pose):
    p, q = pose.pose.position, pose.pose.orientation
    return tuple(round(value, 9) for value in (p.x, p.y, p.z, q.x, q.y, q.z, q.w))


def json_value(value):
    if isinstance(value, (np.ndarray, array)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def call_summary(rows):
    safe = [row for row in rows if row['code'] == 1 and row['joint_safe']]
    unique = []
    for row in safe:
        joints = row['joints']
        if not any(max(abs(joints[f'joint{i}'] - old[f'joint{i}']) for i in range(1, 7)) < .01 for old in unique):
            unique.append(joints)
    return {
        'attempts': len(rows), 'codes': dict(Counter(row['code'] for row in rows)),
        'safe_results': len(safe),
        'unique_safe_solutions': len(unique),
        'safe_wrist_bend_range_rad': ([min(abs(row['joints']['joint5']) for row in safe), max(abs(row['joints']['joint5']) for row in safe)] if safe else None),
        'rejection_groups': dict(Counter(row['joint_reason'].split(':')[0] for row in rows if row['code'] == 1 and not row['joint_safe'])),
    }


def sample_targets(planner, observed, expanded=False):
    normal = observed['normal']
    level = quaternion_to_matrix(level_limited_camera_orientation(normal.copy(), observed['camera_orientation'].copy(), np.array([0., 0., 1.]), 0.))
    roll = planner.values['candidate_roll_rad']
    orientations = [(normal, None), (normal, 0.), (normal, -roll), (normal, roll)]
    tilts = [planner.values['candidate_tilt_rad']]
    if expanded:
        tilts.append(planner.values['maximum_camera_tilt_rad'] - 1.e-9)
    for tilt in tilts:
        for tangent in (level[:, 0], level[:, 1]):
            for sign in (-1., 1.):
                direction = math.cos(tilt) * normal + sign * math.sin(tilt) * tangent
                for value in ([None, -roll, roll] if expanded else [None]):
                    orientations.append((direction, value))
    targets = []
    for direction, value in orientations:
        for offset in planner.values['approach_distance_offsets_m']:
            target = planner._candidate_pose(observed, planner.values['approach_distance_m'] + float(offset), direction, value)
            if planner._view_is_safe(target, observed)[0]:
                targets.append(target)
    return targets


def diagnostic_sweep(planner, targets, constrained):
    current = planner._latest_joint_positions.copy()
    seeds = [current.copy()]
    for bend, roll in ((-.60, 0.), (.60, 0.), (-.60, .5), (-.60, -.5), (.60, .5), (.60, -.5)):
        seed = dict(current, joint5=bend, joint4=current['joint4'] + roll, joint6=current['joint6'] - roll)
        for name, (lower, upper) in planner._arm_joint_limits.items():
            seed[name] = float(np.clip(seed[name], lower + .01, upper - .01))
        seeds.append(seed)
    started = time.monotonic()
    planner.calls = []
    for target in targets:
        for seed in (seeds[1:3] if constrained else seeds):
            request = GetPositionIK.Request()
            ik = request.ik_request
            ik.group_name = planner.values['planning_group']
            ik.ik_link_name = planner._end_effector_link
            ik.pose_stamped = target
            ik.avoid_collisions = True
            ik.robot_state.is_diff = True
            ik.robot_state.joint_state.name = list(seed)
            ik.robot_state.joint_state.position = list(seed.values())
            ik.timeout = Duration(seconds=planner.values['ik_timeout_seconds']).to_msg()
            if constrained:
                reserve = planner.values['joint_goal_tolerance_rad'] + planner.values['execution_joint_tolerance_rad']
                margin = planner.values['joint_limit_margin_rad'] + reserve
                bend = planner.values['minimum_abs_wrist_bend_rad'] + reserve
                ik.constraints = Constraints(name='existing_safe_joint_bounds')
                for name, (lo, hi) in planner._arm_joint_limits.items():
                    lo, hi = lo + margin, hi - margin
                    if name == planner.values['wrist_singularity_joint']:
                        if seed[name] < 0:
                            hi = min(hi, -bend)
                        else:
                            lo = max(lo, bend)
                    ik.constraints.joint_constraints.append(JointConstraint(joint_name=name, position=(lo + hi) / 2., tolerance_above=(hi - lo) / 2., tolerance_below=(hi - lo) / 2., weight=1.))
            planner._call_moveit_service(planner._ik_client, request)
    return {'elapsed_seconds': time.monotonic() - started, **call_summary(planner.calls), 'calls': planner.calls}


class FrozenPlanner(ButtonApproachPlanner):
    """Reuse production geometry/search without initializing command endpoints."""

    def __init__(self, node):
        self.node = node
        self.values = {}
        self._declare_parameters()
        self._lock = threading.Lock()
        self._arm_joint_limits = None
        self.calls = []
        self.targets = {}

    def declare_parameter(self, name, value):
        self.values[name] = value

    def get_parameter(self, name):
        return SimpleNamespace(value=self.values[name])

    def get_clock(self):
        return self.node.get_clock()

    def get_logger(self):
        return self.node.get_logger()

    def _fresh_joint_positions(self):
        return self._latest_joint_positions.copy()

    def _call_moveit_service(self, client, request, deadline=None):
        started = time.monotonic()
        timeout = min(5., max(.001, deadline - started)) if deadline else 5.
        if not client.wait_for_service(timeout_sec=min(5., timeout)):
            raise ValueError('Service unavailable: ' + client.srv_name)
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)
        if not future.done() or future.result() is None:
            client.remove_pending_request(future)
            raise ValueError('Service timeout: ' + client.srv_name)
        result = future.result()
        if isinstance(request, GetPositionIK.Request):
            joints = dict(zip(result.solution.joint_state.name, result.solution.joint_state.position))
            reserve = self.values['joint_goal_tolerance_rad'] + self.values['execution_joint_tolerance_rad']
            safe, reason = self._joint_configuration_is_safe(joints, reserve_rad=reserve)
            self.calls.append({
                'target_index': self.targets.get(pose_key(request.ik_request.pose_stamped), -1),
                'target': message_to_ordereddict(request.ik_request.pose_stamped),
                'seed': dict(zip(request.ik_request.robot_state.joint_state.name, request.ik_request.robot_state.joint_state.position)),
                'elapsed_seconds': time.monotonic() - started,
                'code': result.error_code.val, 'joints': joints,
                'joint_safe': bool(safe), 'joint_reason': reason,
                'margin_to_bounds_rad': {name: min(value - self._arm_joint_limits[name][0], self._arm_joint_limits[name][1] - value) for name, value in joints.items() if name in self._arm_joint_limits},
            })
        return result


DATA_DIR = Path(__file__).resolve().parents[1] / 'data'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(DATA_DIR / 'ik_candidate_diagnostic.json'))
    parser.add_argument('--seconds', type=float, default=6.)
    args = parser.parse_args()
    if not 1. <= args.seconds <= 30.:
        parser.error('--seconds must be between 1 and 30')
    if Path(args.output).exists():
        parser.error('--output already exists; choose a new path to preserve prior evidence')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Diagnostics()
    planner = FrozenPlanner(node)
    messages = {}
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    for topic, kind, qos in [
        ('/button_approach_planner/observation_status', String, latched),
        ('/button_approach/status', String, latched),
        ('/piper_pika/joint_states', JointState, 10),
        ('/feedback/joint_states', JointState, 10),
    ]:
        node.subscriptions_owned.append(node.create_subscription(kind, topic, lambda message, topic=topic: messages.update({topic: message}), qos))
    result = {'read_only': True, 'limitations': ['This is a current stationary snapshot, not the failed request frozen observation. No motion endpoint is created or called.']}
    try:
        params = node.parameters('/button_approach_planner', list(planner.values))
        planner.values.update(params)
        planner._base_frame = params['base_frame']
        planner._end_effector_link = params['end_effector_link']
        planner._workspace_min = np.array(params['workspace_min'])
        planner._workspace_max = np.array(params['workspace_max'])
        planner._planning_deadline = time.monotonic() + 90.
        node.collect(1.)
        planner._description_client = node.create_client(GetParameters, params['robot_description_service'])
        planner._ik_client = node.create_client(GetPositionIK, params['ik_service'])
        planner._fk_client = node.create_client(GetPositionFK, params['fk_service'])
        validity = node.create_client(GetStateValidity, '/check_state_validity')
        planner._load_arm_joint_limits()
        elapsed = node.collect(args.seconds)
        joint_message = messages[params['joint_state_topic']]
        planner._latest_joint_positions = dict(zip(joint_message.name, joint_message.position))
        result.update({
            'parameters': params, 'joint_limits': planner._arm_joint_limits,
            'snapshot_stamp_ns': node.get_clock().now().nanoseconds,
            'selected_button': node.selected,
            'topics': {key: message_to_ordereddict(value) for key, value in messages.items()},
            'sample_seconds': elapsed,
        })
        transforms = {key: node.tf_buffer.lookup_transform(params['base_frame'] if key != 'tip_to_camera' else params['end_effector_link'], frame, Time()) for key, frame in [('base_to_tip', params['end_effector_link']), ('base_to_camera', params['camera_frame']), ('tip_to_camera', params['camera_frame'])]}
        result['transforms'] = {key: message_to_ordereddict(value) for key, value in transforms.items()}
        model = CameraModel(**{key: node.camera_info[params['camera_info_topic']][key] for key in ('width', 'height', 'k', 'd', 'distortion_model')})
        result['camera_info'] = node.camera_info[params['camera_info_topic']]
        mount_p, mount_q = arrays(transforms['tip_to_camera'])
        observations = []
        errors = Counter()
        for message, received, age, selected in node.surface:
            if selected != node.selected or message.header.frame_id != params['camera_frame']:
                continue
            try:
                cp, cq = arrays(node.tf_buffer.lookup_transform(params['base_frame'], message.header.frame_id, Time.from_msg(message.header.stamp)))
                rotation = quaternion_to_matrix(cq)
                point = np.array([message.pose.position.x, message.pose.position.y, message.pose.position.z])
                normal = rotation @ quaternion_to_matrix(quaternion(message.pose.orientation))[:, 2]
                if normal @ (rotation @ point) < 0:
                    normal = -normal
                observations.append({'button': cp + rotation @ point, 'normal': normal, 'camera_orientation': cq, 'received_at': received, 'stamp_ns': Time.from_msg(message.header.stamp).nanoseconds, 'selected_button': selected, 'tip_to_camera_translation': mount_p, 'tip_to_camera_quaternion': mount_q, 'camera_model': model, 'camera_frame': message.header.frame_id})
            except Exception as error:
                errors[str(error)] += 1
        window = observations[-params['observation_stable_samples']:]
        button, normal, detail = stable_observation_window(window, params['observation_minimum_samples'], params['observation_position_tolerance_m'], params['observation_normal_tolerance_rad'])
        result['observation'] = {'samples': len(window), 'detail': detail, 'errors': dict(errors), 'raw_button': [item['button'].tolist() for item in window], 'raw_normal': [item['normal'].tolist() for item in window]}
        if normal is None:
            raise ValueError('Snapshot not stable: ' + detail)
        observed = dict(window[-1], button=button, normal=normal)
        result['observation'].update(button=button.tolist(), normal=normal.tolist(), camera_orientation=observed['camera_orientation'].tolist())
        tp, tq = arrays(transforms['base_to_tip'])
        current_pose = planner._make_pose(tp, tq, node.get_clock().now().to_msg())
        cam_p, cam_q = arrays(transforms['base_to_camera'])
        angle = math.degrees(math.acos(float(np.clip(quaternion_to_matrix(cam_q)[:, 2] @ normal, -1., 1.))))
        result['current'] = {
            'joints': planner._latest_joint_positions,
            'view': planner._view_is_safe(current_pose, observed),
            'handover_view': planner._view_is_safe(current_pose, observed, handover=True),
            'camera_tilt_deg': angle,
            'camera_roll_deg': math.degrees(camera_level_roll_error(cam_q, normal, np.array([0., 0., 1.]))),
            'normal_distance_m': float((button - tp) @ normal),
            'joint_safe': planner._joint_configuration_is_safe(planner._latest_joint_positions),
        }
        check = GetStateValidity.Request()
        check.group_name = params['planning_group']
        check.robot_state.is_diff = True
        check.robot_state.joint_state.name = list(planner._latest_joint_positions)
        check.robot_state.joint_state.position = list(planner._latest_joint_positions.values())
        result['current']['state_validity'] = message_to_ordereddict(planner._call_moveit_service(validity, check))
        targets = list(planner._candidate_poses(observed))
        planner.targets = {pose_key(pose): index for index, pose in enumerate(targets)}
        result['candidate_targets'] = [{'index': index, 'pose': message_to_ordereddict(pose), 'view': planner._view_is_safe(pose, observed)} for index, pose in enumerate(targets)]
        result['candidate_count'] = len(targets)
        for label, budget, maximum in [('production_budget', params['ik_search_budget_seconds'], params['maximum_ik_solutions']), ('full_coverage', 30., 10000)]:
            planner.calls = []
            planner.values['ik_search_budget_seconds'] = budget
            planner.values['maximum_ik_solutions'] = maximum
            planner._planning_deadline = time.monotonic() + budget + 2.
            started = time.monotonic()
            solutions = planner._solve_visible_candidates(observed)
            rows = planner.calls
            result[label] = {'elapsed_seconds': time.monotonic() - started, 'attempts': len(rows), 'codes': dict(Counter(row['code'] for row in rows)), 'rejections': dict(Counter(row['joint_reason'] for row in rows if row['code'] == 1 and not row['joint_safe'])), 'solutions': [{'cost': score, 'joints': joints, 'target': message_to_ordereddict(target)} for score, joints, target in solutions], 'calls': rows}
            print(label, json.dumps(call_summary(rows)), 'solutions', len(solutions), flush=True)
        result['legacy_target_outer'] = diagnostic_sweep(planner, targets, False)
        print('legacy_target_outer', json.dumps(call_summary(result['legacy_target_outer']['calls'])), flush=True)
        result['constrained_ik'] = diagnostic_sweep(planner, targets, True)
        print('constrained_ik', json.dumps(call_summary(result['constrained_ik']['calls'])), flush=True)
        planner.values['candidate_tilt_rad'] = params['maximum_camera_tilt_rad'] - 1.e-9
        extended_targets = list(planner._candidate_poses(observed))
        result['configured_maximum_tilt_sweep'] = diagnostic_sweep(planner, extended_targets, True)
        print('configured_maximum_tilt_sweep', json.dumps(call_summary(result['configured_maximum_tilt_sweep']['calls'])), flush=True)
        planner.values['candidate_tilt_rad'] = params['candidate_tilt_rad']
        expanded = sample_targets(planner, observed, expanded=True)
        result['expanded_tilt_roll'] = diagnostic_sweep(planner, expanded, False)
        print('expanded_tilt_roll', json.dumps(call_summary(result['expanded_tilt_roll']['calls'])), flush=True)
        level = quaternion_to_matrix(level_limited_camera_orientation(normal.copy(), observed['camera_orientation'].copy(), np.array([0., 0., 1.]), 0.))
        result['normal_perturbations'] = []
        for axis in (0, 1):
            for sign in (-1., 1.):
                delta = math.radians(2.)
                shifted = dict(observed, normal=math.cos(delta) * normal + sign * math.sin(delta) * level[:, axis])
                comparison = {'axis': axis, 'degrees': 2. * sign, 'normal': shifted['normal'].tolist()}
                for label, use_expanded in [('legacy', False), ('expanded', True)]:
                    sweep = diagnostic_sweep(planner, sample_targets(planner, shifted, expanded=use_expanded), False)
                    comparison[label] = {key: value for key, value in sweep.items() if key != 'calls'}
                result['normal_perturbations'].append(comparison)
                print('normal_perturbation', json.dumps(comparison), flush=True)
        result['moveit_kinematics'] = node.parameters('/move_group', ['robot_description_kinematics.arm.kinematics_solver', 'robot_description_kinematics.arm.kinematics_solver_timeout', 'robot_description_kinematics.arm.kinematics_solver_search_resolution'])
    except Exception as error:
        result['error'] = repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(result, indent=2, default=json_value) + '\n')
        print(json.dumps({'output': args.output, 'selected_button': result.get('selected_button'), 'current': result.get('current'), 'candidate_count': result.get('candidate_count'), 'error': result.get('error')}, indent=2, default=json_value), flush=True)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
