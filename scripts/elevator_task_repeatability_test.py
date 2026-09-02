#!/usr/bin/env python3
"""Run repeatable end-to-end elevator-button trials and write CSV metrics."""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer
from tf2_ros import TransformException
from tf2_ros import TransformListener


DEFAULT_BUTTONS = ['1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm']
VISUAL_TERMINAL_PREFIXES = ('COMPLETE:', 'FAILED:', 'STOPPED')


def metric(status, name):
    match = re.search(rf'{re.escape(name)}=([-+0-9.]+)(\w+)', status)
    return match.group(1) + match.group(2) if match else ''


class TaskRepeatabilityTest(Node):
    """Track sequence-separated task results and press timing messages."""

    def __init__(self):
        super().__init__(f'elevator_task_repeatability_test_{os.getpid()}')
        latched_qos = QoSProfile(
            depth=50,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.result_sequence = 0
        self.latest_result = ''
        self.completion_sequence = 0
        self.latest_completion = False
        self.active_button = ''
        self.visual_statuses = []
        self.timing_sequence = 0
        self.latest_timing = None
        self.latest_phase = 'IDLE'
        self.latest_joints = {}
        self.trace_button = ''
        self.trace_started = 0.0
        self.trace_samples = []
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
            spin_thread=False,
        )

        self.command_publisher = self.create_publisher(
            String,
            '/elevator_task/command',
            10,
        )
        self.create_subscription(
            String,
            '/elevator_task/result',
            self._result_callback,
            latched_qos,
        )
        self.create_subscription(
            Bool,
            '/elevator_task/completed',
            self._completion_callback,
            latched_qos,
        )
        self.create_subscription(
            String,
            '/elevator_task/active_button',
            self._active_button_callback,
            latched_qos,
        )
        self.create_subscription(
            String,
            '/button_visual_servo/status',
            self._visual_status_callback,
            latched_qos,
        )
        self.create_subscription(
            String,
            '/elevator_task/status',
            self._status_callback,
            latched_qos,
        )
        self.create_subscription(
            JointState,
            '/piper_pika/joint_states',
            self._joint_state_callback,
            10,
        )
        self.create_timer(0.02, self._sample_endpoint)
        self.create_subscription(
            String,
            '/button_press/timing',
            self._timing_callback,
            latched_qos,
        )
        self.stop_client = self.create_client(
            Trigger,
            '/elevator_task_manager/stop',
        )

    def _result_callback(self, message):
        self.result_sequence += 1
        self.latest_result = str(message.data)

    def _completion_callback(self, message):
        self.completion_sequence += 1
        self.latest_completion = bool(message.data)

    def _active_button_callback(self, message):
        self.active_button = str(message.data)

    def _visual_status_callback(self, message):
        self.visual_statuses.append(str(message.data))

    def _timing_callback(self, message):
        try:
            payload = json.loads(str(message.data))
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warning('Ignoring malformed press timing JSON')
            return
        if not isinstance(payload, dict):
            return
        self.timing_sequence += 1
        self.latest_timing = payload

    def _status_callback(self, message):
        self.latest_phase = str(message.data).partition(':')[0]

    def _joint_state_callback(self, message):
        if len(message.name) == len(message.position):
            self.latest_joints = dict(zip(message.name, message.position))

    def _sample_endpoint(self):
        if not self.trace_button:
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                'pika_fingertip_center_link',
                Time(),
            )
        except TransformException:
            return
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        sample = {
            'button': self.trace_button,
            'elapsed_seconds': time.monotonic() - self.trace_started,
            'phase': self.latest_phase,
            'x': translation.x,
            'y': translation.y,
            'z': translation.z,
            'qx': rotation.x,
            'qy': rotation.y,
            'qz': rotation.z,
            'qw': rotation.w,
        }
        for name in (f'joint{index}' for index in range(1, 7)):
            sample[name] = self.latest_joints.get(name, '')
        self.trace_samples.append(sample)

    def spin_for(self, seconds):
        deadline = time.monotonic() + max(0.0, float(seconds))
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(
                self,
                timeout_sec=min(0.10, deadline - time.monotonic()),
            )

    def wait_for_dependencies(self, timeout):
        deadline = time.monotonic() + float(timeout)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.10)
            if (
                self.command_publisher.get_subscription_count() > 0
                and self.stop_client.service_is_ready()
            ):
                return
        raise RuntimeError('task command subscriber or stop service unavailable')

    def run_trial(self, button, timeout):
        self.spin_for(0.20)
        result_baseline = self.result_sequence
        completion_baseline = self.completion_sequence
        timing_baseline = self.timing_sequence
        visual_baseline = len(self.visual_statuses)
        started = time.monotonic()
        self.trace_button = str(button)
        self.trace_started = started
        self.trace_samples = []

        self.command_publisher.publish(String(data=f'press {button}'))
        print(f'[command] press {button}', flush=True)
        deadline = started + float(timeout)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.10)
            if (
                self.result_sequence > result_baseline
                and self.latest_result.startswith(('COMPLETE:', 'FAILED:'))
            ):
                break
        else:
            self.request_stop()
            raise TimeoutError(
                f'button `{button}` timed out after {timeout:.1f}s'
            )

        self.spin_for(0.20)
        trace = self.trace_samples
        self.trace_button = ''
        self.trace_samples = []
        fresh_visual = self.visual_statuses[visual_baseline:]
        visual_terminal = next(
            (
                status for status in reversed(fresh_visual)
                if status.startswith(VISUAL_TERMINAL_PREFIXES)
            ),
            '',
        )
        timing = (
            self.latest_timing
            if self.timing_sequence > timing_baseline
            else None
        )
        completed = (
            self.completion_sequence > completion_baseline
            and self.latest_completion
        )
        result = self.latest_result
        passed = result.startswith('COMPLETE:') and completed
        if passed and (
            timing is None
            or not bool(timing.get('completed'))
            or str(timing.get('button')) != str(button)
        ):
            passed = False
            result += '; missing or mismatched successful press timing'

        return {
            'passed': passed,
            'total_seconds': time.monotonic() - started,
            'task_result': result,
            'visual_status': visual_terminal,
            'timing': timing or {},
            'trace': trace,
        }

    def request_stop(self):
        if not self.stop_client.service_is_ready():
            return
        future = self.stop_client.call_async(Trigger.Request())
        deadline = time.monotonic() + 5.0
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.10)
            if future.done():
                break
        # The stop service only acknowledges the request.  The manager still
        # retracts and returns home asynchronously; do not let that terminal
        # result leak into the next button trial.
        quiescence_deadline = time.monotonic() + 20.0
        while rclpy.ok() and time.monotonic() < quiescence_deadline:
            rclpy.spin_once(self, timeout_sec=0.10)
            if not self.active_button:
                return
        raise TimeoutError('task manager did not become idle after stop')


def arguments():
    parser = argparse.ArgumentParser(
        description='Run full elevator tasks for every configured button.',
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='required acknowledgement that this script moves the arm',
    )
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('--buttons', nargs='+', default=DEFAULT_BUTTONS)
    parser.add_argument('--task-timeout', type=float, default=180.0)
    parser.add_argument('--dependency-timeout', type=float, default=20.0)
    parser.add_argument('--settle-seconds', type=float, default=0.5)
    parser.add_argument('--continue-on-failure', action='store_true')
    parser.add_argument('--log-dir', default='/workspace/test_logs')
    return parser.parse_args()


def main():
    args = arguments()
    if not args.execute:
        print(
            'Refusing to move without --execute. Run:\n'
            '  ./scripts/test_elevator_task.sh --execute',
            file=sys.stderr,
        )
        return 2
    if args.cycles < 1:
        print('--cycles must be at least 1', file=sys.stderr)
        return 2
    unknown = [button for button in args.buttons if button not in DEFAULT_BUTTONS]
    if unknown:
        print(f'unsupported buttons: {unknown}', file=sys.stderr)
        return 2

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = log_dir / f'elevator_task_{timestamp}.csv'
    trace_path = log_dir / f'endpoint_trace_{timestamp}.csv'
    phase_names = [
        'setup', 'baseline', 'approach', 'press', 'hold',
        'retract', 'release_wait',
    ]
    fieldnames = [
        'cycle', 'button', 'started_at', 'result', 'total_seconds',
        'standoff', 'lateral', 'angle', 'press_total_seconds',
        *[f'press_{name}_seconds' for name in phase_names],
        'task_result', 'visual_status',
    ]

    failures = 0
    trace_fieldnames = [
        'cycle', 'button', 'elapsed_seconds', 'phase',
        'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw',
        'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6',
    ]
    rclpy.init()
    node = TaskRepeatabilityTest()
    try:
        node.wait_for_dependencies(args.dependency_timeout)
        with (
            csv_path.open('w', newline='', encoding='utf-8') as csv_file,
            trace_path.open('w', newline='', encoding='utf-8') as trace_file,
        ):
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            trace_writer = csv.DictWriter(
                trace_file,
                fieldnames=trace_fieldnames,
            )
            trace_writer.writeheader()
            stop_after_failure = False
            for cycle in range(1, args.cycles + 1):
                for button in args.buttons:
                    started_at = datetime.now().isoformat(timespec='seconds')
                    try:
                        trial = node.run_trial(button, args.task_timeout)
                    except Exception as error:
                        trial = {
                            'passed': False,
                            'total_seconds': 0.0,
                            'task_result': f'FAILED: {error}',
                            'visual_status': '',
                            'timing': {},
                            'trace': list(node.trace_samples),
                        }
                        node.trace_button = ''
                        node.trace_samples = []
                    result = 'PASS' if trial['passed'] else 'FAIL'
                    if not trial['passed']:
                        failures += 1
                    timing = trial['timing']
                    phases = timing.get('phases', {})
                    row = {
                        'cycle': cycle,
                        'button': button,
                        'started_at': started_at,
                        'result': result,
                        'total_seconds': f"{trial['total_seconds']:.3f}",
                        'standoff': metric(trial['visual_status'], 'standoff'),
                        'lateral': metric(trial['visual_status'], 'lateral'),
                        'angle': metric(trial['visual_status'], 'angle'),
                        'press_total_seconds': timing.get('total_seconds', ''),
                        'task_result': trial['task_result'],
                        'visual_status': trial['visual_status'],
                    }
                    for phase in phase_names:
                        row[f'press_{phase}_seconds'] = phases.get(phase, '')
                    writer.writerow(row)
                    csv_file.flush()
                    for sample in trial['trace']:
                        trace_writer.writerow({'cycle': cycle, **sample})
                    trace_file.flush()
                    print(
                        f'[{result}] cycle={cycle} button={button} '
                        f"total={trial['total_seconds']:.2f}s "
                        f"press={timing.get('total_seconds', 'n/a')}s "
                        f"result={trial['task_result']}",
                        flush=True,
                    )
                    if not trial['passed'] and not args.continue_on_failure:
                        stop_after_failure = True
                        break
                    node.spin_for(args.settle_seconds)
                if stop_after_failure:
                    break
    except KeyboardInterrupt:
        node.request_stop()
        failures += 1
    except Exception as error:
        print(f'[FAIL] {error}', file=sys.stderr)
        failures += 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print(
        f'Summary: failures={failures}, log={csv_path}, '
        f'trace={trace_path}',
        flush=True,
    )
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
