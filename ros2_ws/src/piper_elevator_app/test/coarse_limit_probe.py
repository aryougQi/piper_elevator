"""Record coarse planning limits in the dedicated ROS 42 simulation only."""

import argparse
import json
import math
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import rclpy
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from moveit_msgs.msg import DisplayTrajectory
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger


JOINT_NAMES = [f'joint{index}' for index in range(1, 7)]
BUTTON_NAMES = {'1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm'}


def await_result(node, future, timeout):
    deadline = time.monotonic() + timeout
    while not future.done() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    if not future.done():
        raise TimeoutError('Service result did not arrive before probe deadline')
    return future.result()


def spin_for(node, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)


def joint_ranges(samples, limits):
    ranges = {}
    first_violation = None
    maximum_violation = None
    for point_index, positions in enumerate(samples):
        for name in JOINT_NAMES:
            if name not in positions:
                continue
            value = float(positions[name])
            if not math.isfinite(value):
                raise ValueError(f'Non-finite {name} at point {point_index}')
            bounds = ranges.setdefault(name, {'minimum': value, 'maximum': value})
            bounds['minimum'] = min(bounds['minimum'], value)
            bounds['maximum'] = max(bounds['maximum'], value)
            lower, upper = limits[name]
            bound = lower if value < lower else upper if value > upper else None
            if bound is None:
                continue
            violation = {
                'point_index': point_index,
                'joint': name,
                'value_rad': value,
                'bound_rad': bound,
                'delta_rad': abs(value - bound),
            }
            if first_violation is None:
                first_violation = violation
            if maximum_violation is None or violation['delta_rad'] > (
                maximum_violation['delta_rad']
            ):
                maximum_violation = violation
    return {
        'samples': len(samples),
        'ranges': ranges,
        'first_violation': first_violation,
        'maximum_violation': maximum_violation,
    }


def endpoint_margins(positions, limits):
    if positions is None:
        return None
    return {
        name: {
            'position_rad': positions.get(name),
            'lower_margin_rad': positions[name] - limits[name][0],
            'upper_margin_rad': limits[name][1] - positions[name],
        }
        for name in JOINT_NAMES if name in positions
    }


def write_report(output, report):
    records = report['cycles'] if report['operation'] == 'execute_cycles' else report['attempts']
    report['successes'] = sum(bool(item.get('success')) for item in records)
    report['failures'] = sum(item.get('success') is False for item in records)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempts', type=int, default=3, choices=range(1, 6))
    parser.add_argument('--execute-cycles', action='store_true')
    parser.add_argument('--cycles', type=int, default=5, choices=range(1, 6))
    parser.add_argument('--buttons', default='')
    parser.add_argument(
        '--output', type=Path,
        default=Path(__file__).resolve().parents[3] / 'diagnostics' / 'data' / 'coarse_limits_probe.json',
    )
    arguments = parser.parse_args()
    if os.environ.get('ROS_DOMAIN_ID') != '42':
        raise RuntimeError('Probe refuses to run outside ROS_DOMAIN_ID=42')
    buttons = [button.strip() for button in arguments.buttons.split(',') if button.strip()]
    if buttons and not arguments.execute_cycles:
        raise RuntimeError('--buttons requires the explicit --execute-cycles mode')
    if any(button not in BUTTON_NAMES for button in buttons):
        raise RuntimeError('Probe received an unknown button label')

    rclpy.init()
    node = Node('coarse_limit_probe', parameter_overrides=[
        Parameter('use_sim_time', Parameter.Type.BOOL, True),
    ])
    displays = []
    feedback = []
    selected = {'button': None, 'acknowledged_at': 0.0}
    observation = {'surface_stamps': [], 'surface_received_at': 0.0}
    live = {'feedback_received_at': 0.0, 'status': None, 'status_received_at': 0.0}
    statuses = []
    try:
        parameters = node.create_client(
            GetParameters, '/button_approach_planner/get_parameters',
        )
        if not parameters.wait_for_service(timeout_sec=10.0):
            raise RuntimeError('Simulation planner parameter service unavailable')
        def confirm_simulation():
            result = await_result(node, parameters.call_async(
                GetParameters.Request(names=['simulation_mode', 'use_sim_time']),
            ), 10.0)
            if len(result.values) != 2 or any(
                value.type != 1 or not value.bool_value for value in result.values
            ):
                raise RuntimeError(
                    'Probe requires planner simulation_mode=true and use_sim_time=true'
                )

        confirm_simulation()

        description = node.create_client(GetParameters, '/move_group/get_parameters')
        if not description.wait_for_service(timeout_sec=10.0):
            raise RuntimeError('MoveIt robot description service unavailable')
        result = await_result(node, description.call_async(
            GetParameters.Request(names=['robot_description']),
        ), 10.0)
        robot = ET.fromstring(result.values[0].string_value)
        limits = {}
        for joint in robot.findall('joint'):
            if joint.get('name') in JOINT_NAMES:
                limit = joint.find('limit')
                limits[joint.get('name')] = [
                    float(limit.get('lower')), float(limit.get('upper')),
                ]
        if set(limits) != set(JOINT_NAMES):
            raise RuntimeError('MoveIt description does not bound all six arm joints')

        def on_feedback(message):
            feedback.append(dict(zip(message.name, message.position)))
            live['feedback_received_at'] = time.monotonic()

        def on_display(message):
            for trajectory_index, trajectory in enumerate(message.trajectory):
                joint_trajectory = trajectory.joint_trajectory
                samples = [
                    dict(zip(joint_trajectory.joint_names, point.positions))
                    for point in joint_trajectory.points
                ]
                displays.append({
                    'received_monotonic': time.monotonic(),
                    'trajectory_index': trajectory_index,
                    'joint_names': list(joint_trajectory.joint_names),
                    'points': samples,
                    'raw_points': [{
                        'positions': list(point.positions),
                        'velocities': list(point.velocities),
                        'accelerations': list(point.accelerations),
                        'time_from_start': {
                            'sec': point.time_from_start.sec,
                            'nanosec': point.time_from_start.nanosec,
                        },
                    } for point in joint_trajectory.points],
                    'summary': joint_ranges(samples, limits),
                })

        def on_selected(message):
            if selected['button'] != message.data:
                selected['acknowledged_at'] = time.monotonic()
            selected['button'] = message.data

        def on_surface(message):
            stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            observation['surface_stamps'].append((time.monotonic(), stamp))
            observation['surface_stamps'] = observation['surface_stamps'][-100:]
            observation['surface_received_at'] = time.monotonic()

        def on_status(message):
            received_at = time.monotonic()
            live['status'] = message.data
            live['status_received_at'] = received_at
            statuses.append({'received_monotonic': received_at, 'status': message.data})

        node.create_subscription(JointState, '/piper_pika/joint_states', on_feedback, 20)
        node.create_subscription(DisplayTrajectory, '/display_planned_path', on_display, 20)
        node.create_subscription(String, '/button_selected', on_selected, QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        ))
        node.create_subscription(PoseStamped, '/button_surface_pose', on_surface, 20)
        node.create_subscription(String, '/button_approach/status', on_status, QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        ))
        planner = node.create_client(Trigger, '/button_approach_planner/plan')
        if not planner.wait_for_service(timeout_sec=10.0):
            raise RuntimeError('Coarse plan service unavailable')
        spin_for(node, 1.0)
        report = {
            'ros_domain_id': 42,
            'simulation_mode': True,
            'operation': 'execute_cycles' if arguments.execute_cycles else 'plan_only',
            'selected_button': selected['button'],
            'urdf_limits': limits,
            'initial_feedback': feedback[-1] if feedback else None,
            'attempts': [],
            'cycles': [],
        }
        write_report(arguments.output, report)

        def service_stage(label, client, timeout):
            confirm_simulation()
            display_start = len(displays)
            feedback_start = len(feedback)
            status_start = len(statuses)
            started = time.monotonic()
            try:
                response = await_result(node, client.call_async(Trigger.Request()), timeout)
                success, message = response.success, response.message
            except TimeoutError as error:
                success, message = False, str(error)
            spin_for(node, 0.15)
            return {
                'stage': label,
                'success': success,
                'message': message,
                'elapsed_seconds': time.monotonic() - started,
                'selected_button': selected['button'],
                'feedback': joint_ranges(feedback[feedback_start:], limits),
                'endpoint_margins': endpoint_margins(feedback[-1] if feedback else None, limits),
                'endpoint_abs_wrist_bend_rad': (
                    abs(feedback[-1]['joint5'])
                    if feedback and 'joint5' in feedback[-1] else None
                ),
                'display_trajectories': displays[display_start:],
                'statuses': statuses[status_start:],
            }

        def wait_for_observation(after, button, timeout=15.0):
            started = time.monotonic()
            last_age = None
            stable_count = 0
            while time.monotonic() - started < timeout:
                rclpy.spin_once(node, timeout_sec=0.05)
                baseline = max(after, selected['acknowledged_at'])
                frames = {}
                if selected['button'] == button:
                    for received, stamp in observation['surface_stamps']:
                        if received > baseline:
                            frames.setdefault(stamp, received)
                stamps = set(frames)
                stable_count = len(stamps)
                third_frame_received_at = (
                    sorted(frames.values())[2] if stable_count >= 3 else math.inf
                )
                now = node.get_clock().now().nanoseconds * 1e-9
                last_age = now - max(stamps) if stamps else None
                if (
                    selected['button'] == button
                    and stable_count >= 3
                    and live['status'] == 'TARGET_READY'
                    and live['status_received_at'] > third_frame_received_at
                    and last_age is not None and 0.0 <= last_age <= 0.5
                    and time.monotonic() - live['feedback_received_at'] <= 0.5
                ):
                    return {
                        'stage': 'fresh_observation', 'success': True,
                        'message': 'Fresh TARGET_READY observation confirmed',
                        'elapsed_seconds': time.monotonic() - started,
                        'distinct_surface_frames': stable_count,
                        'surface_age_ros_seconds': last_age,
                        'selected_button': selected['button'],
                        'observation_baseline_monotonic': baseline,
                        'third_frame_received_monotonic': third_frame_received_at,
                        'target_ready_received_monotonic': live['status_received_at'],
                    }
            return {
                'stage': 'fresh_observation', 'success': False,
                'message': 'Fresh TARGET_READY observation did not arrive',
                'elapsed_seconds': time.monotonic() - started,
                'distinct_surface_frames': stable_count,
                'surface_age_ros_seconds': last_age,
                'selected_button': selected['button'],
                'planner_status': live['status'],
            }

        if arguments.execute_cycles:
            home = node.create_client(Trigger, '/button_approach_planner/return_home')
            execute = node.create_client(Trigger, '/button_approach_planner/execute')
            for client in (home, execute):
                if not client.wait_for_service(timeout_sec=10.0):
                    raise RuntimeError('Simulation Home/Execute service unavailable')
            selection = node.create_publisher(String, '/button_selection', QoSProfile(
                depth=1, reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ))
            targets = buttons or [selected['button']]
            if any(button not in BUTTON_NAMES for button in targets):
                raise RuntimeError('No selected button; specify --buttons explicitly')
            halted = False
            for cycle_index in range(arguments.cycles):
                for button in targets:
                    cycle = {'cycle': cycle_index + 1, 'button': button, 'stages': []}
                    report['cycles'].append(cycle)
                    result = service_stage('home', home, 65.0)
                    cycle['stages'].append(result)
                    write_report(arguments.output, report)
                    if result['success']:
                        ready_after = time.monotonic()
                        if buttons:
                            selection.publish(String(data=button))
                        result = wait_for_observation(ready_after, button)
                        cycle['stages'].append(result)
                        write_report(arguments.output, report)
                    if result['success']:
                        result = service_stage('plan', planner, 50.0)
                        cycle['stages'].append(result)
                        write_report(arguments.output, report)
                    if result['success']:
                        result = service_stage('execute', execute, 65.0)
                        cycle['stages'].append(result)
                    cycle['success'] = all(stage['success'] for stage in cycle['stages'])
                    if not cycle['success']:
                        cycle['failure_stage'] = result['stage']
                        report['halted_after_failure'] = True
                        halted = True
                    write_report(arguments.output, report)
                    print(json.dumps({
                        'cycle': cycle['cycle'], 'button': button,
                        'success': cycle['success'],
                        'stages': [{key: stage.get(key) for key in (
                            'stage', 'success', 'message', 'elapsed_seconds',
                        )} for stage in cycle['stages']],
                    }), flush=True)
                    if halted:
                        break
                if halted:
                    break
            print(json.dumps({
                'output': str(arguments.output),
                'successes': report['successes'], 'failures': report['failures'],
            }), flush=True)
            return

        for index in range(arguments.attempts):
            display_start = len(displays)
            feedback_start = len(feedback)
            started = time.monotonic()
            response = await_result(node, planner.call_async(Trigger.Request()), 60.0)
            spin_for(node, 0.5)
            trajectory_records = displays[display_start:]
            attempt = {
                'attempt': index + 1,
                'success': response.success,
                'message': response.message,
                'elapsed_seconds': time.monotonic() - started,
                'selected_button': selected['button'],
                'feedback': joint_ranges(feedback[feedback_start:], limits),
                'display_trajectories': trajectory_records,
            }
            report['attempts'].append(attempt)
            write_report(arguments.output, report)
            print(json.dumps({
                'attempt': attempt['attempt'],
                'stage': 'plan',
                **{key: attempt[key] for key in (
                    'success', 'message', 'elapsed_seconds',
                )},
            }), flush=True)
        print(json.dumps({
            'output': str(arguments.output),
            'successes': report['successes'], 'failures': report['failures'],
        }), flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
