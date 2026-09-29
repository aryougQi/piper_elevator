#!/usr/bin/env python3
"""Run repeated end-to-end elevator-task trials and summarize failures."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import re
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from rcl_interfaces.srv import GetParameters
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger


TERMINAL = ('COMPLETE:', 'FAILED:', 'STOPPED:')


class StabilityProbe(Node):
    def __init__(self):
        super().__init__('elevator_stability_probe')
        qos = QoSProfile(
            depth=50,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.statuses = []
        self.results = []
        self.completions = []
        self.latest_joint_at = 0.0
        self.latest_joint_names = set()
        self.create_subscription(String, '/elevator_task/status', self._status, qos)
        self.create_subscription(String, '/elevator_task/result', self._result, qos)
        self.create_subscription(Bool, '/elevator_task/completed', self._completed, qos)
        self.create_subscription(JointState, '/piper_pika/joint_states', self._joint_state, 10)
        self.command_pub = self.create_publisher(String, '/elevator_task/command', qos)
        self.reset_client = self.create_client(Trigger, '/elevator_task_manager/reset')
        self.move_group_params = self.create_client(
            GetParameters, '/move_group/get_parameters'
        )

    def _status(self, message):
        text = str(message.data)
        self.statuses.append((time.monotonic(), text))
        print(f'[status] {text}', flush=True)

    def _result(self, message):
        self.results.append((time.monotonic(), str(message.data)))

    def _completed(self, message):
        self.completions.append((time.monotonic(), bool(message.data)))

    def _joint_state(self, message):
        self.latest_joint_at = time.monotonic()
        self.latest_joint_names = set(message.name)

    def spin(self, seconds):
        deadline = time.monotonic() + max(0.0, seconds)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.1, deadline - time.monotonic()))

    def wait_for_stack(self, timeout):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if self.reset_client.wait_for_service(timeout_sec=0.2):
                if (
                    self.statuses
                    and self.statuses[-1][1].startswith('IDLE')
                    and self.move_group_params.service_is_ready()
                    and time.monotonic() - self.latest_joint_at < 1.0
                    and {'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'}
                    <= self.latest_joint_names
                ):
                    # The parameter service appears before MoveIt's planning
                    # context has finished loading. Let the action servers
                    # and controllers settle before issuing the first home.
                    self.spin(8.0)
                    return
            rclpy.spin_once(self, timeout_sec=0.1)
        raise TimeoutError('elevator_task_manager did not become IDLE')

    def publish_command(self, button):
        deadline = time.monotonic() + 5.0
        while (
            self.command_pub.get_subscription_count() < 1
            and time.monotonic() < deadline
        ):
            self.spin(0.1)
        self.command_pub.publish(String(data=f'press {button}'))
        self.spin(0.5)

    def wait_terminal(self, start_index, timeout):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            for _, text in self.statuses[start_index:]:
                if text.startswith(TERMINAL):
                    return text
            rclpy.spin_once(self, timeout_sec=0.1)
        return f'FAILED: test timeout after {timeout:.1f}s'

    def reset(self, timeout):
        if not self.reset_client.wait_for_service(timeout_sec=min(timeout, 2.0)):
            return False, 'reset service unavailable'
        future = self.reset_client.call_async(Trigger.Request())
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if future.done():
                response = future.result()
                return bool(response and response.success), (
                    response.message if response else 'no reset response'
                )
        return False, 'reset timed out'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--button', default='3')
    parser.add_argument('--timeout', type=float, default=240.0)
    parser.add_argument('--reset-timeout', type=float, default=20.0)
    parser.add_argument('--mode', choices=('stable', 'yolo'), required=True)
    parser.add_argument(
        '--log-dir',
        default='/workspace/ros2_ws/diagnostics/data/stability',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.runs < 1 or args.timeout <= 0:
        raise SystemExit('--runs and --timeout must be positive')
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = log_dir / f'elevator_stability_{args.mode}_{timestamp}.csv'
    fields = ['mode', 'run', 'button', 'started_at', 'duration_s', 'passed',
              'terminal_status', 'replans', 'failure_reason']

    rclpy.init()
    node = StabilityProbe()
    rows = []
    try:
        print('Waiting for the complete task stack...', flush=True)
        node.wait_for_stack(args.timeout)
        for run in range(1, args.runs + 1):
            print(f'\n=== {args.mode} trial {run}/{args.runs} ===', flush=True)
            start = time.monotonic()
            status_index = len(node.statuses)
            started_at = datetime.now().isoformat(timespec='seconds')
            node.publish_command(args.button)
            terminal = node.wait_terminal(status_index, args.timeout)
            duration = time.monotonic() - start
            trial_statuses = [text for _, text in node.statuses[status_index:]]
            replans = sum('COARSE_REACQUIRING' in text for text in trial_statuses)
            passed = terminal.startswith('COMPLETE:')
            failure_reason = '' if passed else terminal
            rows.append({
                'mode': args.mode,
                'run': run,
                'button': args.button,
                'started_at': started_at,
                'duration_s': f'{duration:.2f}',
                'passed': str(passed).lower(),
                'terminal_status': terminal,
                'replans': replans,
                'failure_reason': failure_reason,
            })
            print(f'[trial] passed={passed} duration={duration:.1f}s '
                  f'replans={replans} result={terminal}', flush=True)
            if run < args.runs:
                ok, message = node.reset(args.reset_timeout)
                print(f'[reset] success={ok} message={message}', flush=True)
                if not ok:
                    raise RuntimeError(message)
                node.wait_for_stack(args.reset_timeout)
    finally:
        with csv_path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        rclpy.shutdown()

    passed = sum(row['passed'] == 'true' for row in rows)
    failures = {}
    for row in rows:
        if row['passed'] != 'true':
            reason = re.sub(r'^FAILED:\s*', '', row['failure_reason'])
            failures[reason] = failures.get(reason, 0) + 1
    print(f'\nSUMMARY mode={args.mode} passed={passed}/{len(rows)} '
          f'success_rate={passed / len(rows):.1%}')
    if failures:
        print('FAILURES:')
        for reason, count in sorted(failures.items(), key=lambda item: -item[1]):
            print(f'  {count}x {reason}')
    print(f'CSV: {csv_path}')
    return 0 if passed == len(rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
