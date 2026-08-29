#!/usr/bin/env python3
"""Run guarded end-to-end button visual-servo repeatability trials."""

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import re
import sys
import time

from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger


ACTIVE_STATUS_PREFIXES = (
    'STARTING_MOVEIT_SERVO',
    'VISUAL_TRACKING',
    'CARTESIAN_HANDOFF',
    'CARTESIAN_EXECUTING',
    'STOPPING',
)
TERMINAL_STATUS_PREFIXES = ('COMPLETE:', 'FAILED:', 'STOPPED')


class RepeatabilityTest(Node):
    def __init__(self):
        super().__init__(f'visual_servo_repeatability_test_{os.getpid()}')
        qos = QoSProfile(
            depth=50,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.status_messages = []
        self.latest_surface_at = 0.0
        self.create_subscription(
            String,
            '/button_visual_servo/status',
            self._status_callback,
            qos,
        )
        self.create_subscription(
            PoseStamped,
            '/button_surface_pose',
            self._surface_callback,
            10,
        )
        self._service_clients = {
            'plan': self.create_client(
                Trigger,
                '/button_approach_planner/plan',
            ),
            'execute': self.create_client(
                Trigger,
                '/button_approach_planner/execute',
            ),
            'return_home': self.create_client(
                Trigger,
                '/button_approach_planner/return_home',
            ),
            'servo_start': self.create_client(
                Trigger,
                '/button_visual_servo/start',
            ),
            'servo_stop': self.create_client(
                Trigger,
                '/button_visual_servo/stop',
            ),
        }

    def _status_callback(self, message):
        text = str(message.data)
        self.status_messages.append((time.monotonic(), text))
        print(f'[status] {text}', flush=True)

    def _surface_callback(self, message):
        del message
        self.latest_surface_at = time.monotonic()

    def spin_for(self, seconds):
        deadline = time.monotonic() + max(0.0, float(seconds))
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(
                self,
                timeout_sec=min(0.10, deadline - time.monotonic()),
            )

    def wait_for_dependencies(self, timeout):
        for name, client in self._service_clients.items():
            if not client.wait_for_service(timeout_sec=float(timeout)):
                raise RuntimeError(f'required service is unavailable: {name}')
        self.wait_for_fresh_surface(timeout)

    def wait_for_fresh_surface(self, timeout):
        deadline = time.monotonic() + float(timeout)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.10)
            if time.monotonic() - self.latest_surface_at <= 1.0:
                return
        raise RuntimeError('no fresh /button_surface_pose within timeout')

    def latest_status(self):
        self.spin_for(0.25)
        return self.status_messages[-1][1] if self.status_messages else ''

    def call(self, name, timeout):
        client = self._service_clients[name]
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + float(timeout)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.10)
            if future.done():
                result = future.result()
                if result is None:
                    raise RuntimeError(f'{name} service returned no response')
                print(
                    f'[{name}] success={result.success} '
                    f'message={result.message}',
                    flush=True,
                )
                return bool(result.success), str(result.message)
        raise TimeoutError(f'{name} service timed out after {timeout:.1f}s')

    def wait_for_terminal_status(self, after_index, timeout):
        deadline = time.monotonic() + float(timeout)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.10)
            for _, status in self.status_messages[after_index:]:
                if status.startswith(TERMINAL_STATUS_PREFIXES):
                    return status
        return f'FAILED: test timeout after {timeout:.1f}s'


def arguments():
    parser = argparse.ArgumentParser(
        description=(
            'Run coarse approach and visual servo repeatedly, recording '
            'timing and terminal status.'
        ),
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='required acknowledgement that this script may move the arm',
    )
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--servo-timeout', type=float, default=120.0)
    parser.add_argument('--service-timeout', type=float, default=60.0)
    parser.add_argument('--dependency-timeout', type=float, default=10.0)
    parser.add_argument('--settle-seconds', type=float, default=1.0)
    parser.add_argument('--return-timeout', type=float, default=60.0)
    parser.add_argument(
        '--skip-coarse',
        action='store_true',
        help='start visual servo from the current pose',
    )
    parser.add_argument(
        '--continue-on-failure',
        action='store_true',
        help='continue remaining trials after a failed trial',
    )
    parser.add_argument(
        '--log-dir',
        default='/workspace/test_logs',
    )
    return parser.parse_args()


def metric(status, name):
    match = re.search(rf'{re.escape(name)}=([-+0-9.]+)(\w+)', status)
    return match.group(1) + match.group(2) if match else ''


def main():
    args = arguments()
    if not args.execute:
        print(
            'Refusing to move the arm without --execute.\n'
            'Review the workspace, keep the emergency stop ready, then run:\n'
            '  ./scripts/test_visual_servo.sh --execute',
            file=sys.stderr,
        )
        return 2
    if args.runs < 1:
        print('--runs must be at least 1', file=sys.stderr)
        return 2

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = log_dir / f'visual_servo_{timestamp}.csv'
    fieldnames = [
        'run',
        'started_at',
        'plan_seconds',
        'execute_seconds',
        'servo_seconds',
        'return_seconds',
        'total_seconds',
        'result',
        'standoff',
        'lateral',
        'angle',
        'status',
    ]

    rclpy.init()
    node = RepeatabilityTest()
    active_trial = False
    failures = 0
    try:
        print('Checking services and fresh RGB-D surface pose...', flush=True)
        node.wait_for_dependencies(args.dependency_timeout)
        status = node.latest_status()
        if status.startswith(ACTIVE_STATUS_PREFIXES):
            raise RuntimeError(
                'visual servo is already active; call '
                '/button_visual_servo/stop before testing'
            )

        with csv_path.open('w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for run_number in range(1, args.runs + 1):
                print(
                    f'\n=== Trial {run_number}/{args.runs} ===',
                    flush=True,
                )
                trial_started = time.monotonic()
                wall_started = datetime.now().isoformat(timespec='seconds')
                plan_seconds = 0.0
                execute_seconds = 0.0
                servo_seconds = 0.0
                return_seconds = 0.0
                terminal = 'FAILED: trial did not start'
                active_trial = True
                try:
                    if not args.skip_coarse:
                        step_started = time.monotonic()
                        success, message = node.call(
                            'plan',
                            args.service_timeout,
                        )
                        plan_seconds = time.monotonic() - step_started
                        if not success:
                            raise RuntimeError(
                                f'coarse plan failed: {message}'
                            )

                        step_started = time.monotonic()
                        success, message = node.call(
                            'execute',
                            args.service_timeout,
                        )
                        execute_seconds = time.monotonic() - step_started
                        if not success:
                            raise RuntimeError(
                                f'coarse execution failed: {message}'
                            )
                        node.spin_for(args.settle_seconds)

                    status_index = len(node.status_messages)
                    step_started = time.monotonic()
                    success, message = node.call(
                        'servo_start',
                        args.service_timeout,
                    )
                    if not success:
                        raise RuntimeError(
                            f'visual servo start failed: {message}'
                        )
                    terminal = node.wait_for_terminal_status(
                        status_index,
                        args.servo_timeout,
                    )
                    servo_seconds = time.monotonic() - step_started
                except Exception as error:
                    terminal = f'FAILED: {error}'

                visual_passed = terminal.startswith('COMPLETE:')
                if not visual_passed:
                    try:
                        node.call('servo_stop', 5.0)
                    except Exception as stop_error:
                        print(f'[warning] stop failed: {stop_error}')

                return_error = None
                step_started = time.monotonic()
                try:
                    node.spin_for(args.settle_seconds)
                    print('[return] MoveIt planning to home...', flush=True)
                    success, message = node.call(
                        'return_home',
                        args.return_timeout,
                    )
                    if not success:
                        raise RuntimeError(message)
                    node.spin_for(args.settle_seconds)
                    node.wait_for_fresh_surface(args.dependency_timeout)
                    print(
                        '[return] home reached and fresh button surface '
                        'restored',
                        flush=True,
                    )
                except Exception as error:
                    return_error = str(error)
                    terminal += f'; RESET_FAILED: {return_error}'
                return_seconds = time.monotonic() - step_started
                active_trial = False

                passed = visual_passed and return_error is None
                if not passed:
                    failures += 1
                total_seconds = time.monotonic() - trial_started
                result = 'PASS' if passed else 'FAIL'
                print(
                    f'[{result}] trial={run_number} '
                    f'total={total_seconds:.2f}s status={terminal}',
                    flush=True,
                )
                writer.writerow({
                    'run': run_number,
                    'started_at': wall_started,
                    'plan_seconds': f'{plan_seconds:.3f}',
                    'execute_seconds': f'{execute_seconds:.3f}',
                    'servo_seconds': f'{servo_seconds:.3f}',
                    'return_seconds': f'{return_seconds:.3f}',
                    'total_seconds': f'{total_seconds:.3f}',
                    'result': result,
                    'standoff': metric(terminal, 'standoff'),
                    'lateral': metric(terminal, 'lateral'),
                    'angle': metric(terminal, 'angle'),
                    'status': terminal,
                })
                csv_file.flush()
                if return_error is not None:
                    print(
                        '[FAIL] MoveIt home was not restored; '
                        'remaining trials are blocked for safety',
                        flush=True,
                    )
                    break
                if not passed and not args.continue_on_failure:
                    break
                node.spin_for(args.settle_seconds)
    except KeyboardInterrupt:
        print('\nInterrupted by operator.', file=sys.stderr)
        failures += 1
    except Exception as error:
        print(f'Preflight failed: {error}', file=sys.stderr)
        failures += 1
    finally:
        if active_trial:
            try:
                node.call('servo_stop', 5.0)
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()

    log_result = str(csv_path) if csv_path.exists() else 'not created'
    print(
        f'\nSummary: failures={failures}, log={log_result}',
        flush=True,
    )
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
