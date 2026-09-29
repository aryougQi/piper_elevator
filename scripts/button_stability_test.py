#!/usr/bin/env python3
"""Drive every panel button several times and report stability.

One trial is a complete end-to-end task: the elevator task manager homes the
arm, selects the button, plans, executes the coarse approach, runs visual
servo, presses, and returns home.  The script drives that state machine over
`/elevator_task/command` instead of talking to the individual nodes, so a
passing trial exercises exactly the production path.

Two independent checks decide PASS/FAIL:

1. the manager reports ``COMPLETE:`` on ``/elevator_task/result``;
2. the Gazebo joint for that button really travelled at least
   ``--pressed-depth`` metres, read from ``/elevator_button/joint_states``.

Check 2 is what makes this a stability test rather than a status-log test: a
task can only report COMPLETE after the press executor itself observed either
a contact or joint depression, and the joint trace is the raw ground truth.

Run through ``./scripts/test_button_stability.sh``.
"""

import argparse
import csv
from datetime import datetime
import math
import os
from pathlib import Path
import statistics
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_msgs.msg import Int8
from std_msgs.msg import String
from std_srvs.srv import Trigger


# Every control on the vertical cabin panel whose label exists in the
# 14-class detector.  The bell and handset tiles have no class, so they can
# never be selected and are not testable.
BUTTONS = ('3', '2', '1', 'open', 'close', 'up', 'down')

# The approach planner refuses to home without a complete, fresh sample of
# these joints (`joint_state_max_age_seconds` is 0.5 s), so the preflight has
# to satisfy exactly the same condition or the very first trial fails on a
# startup race rather than on panel stability.
ARM_JOINTS = ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')
ARM_FEEDBACK_TOPIC = '/piper_pika/joint_states'
ARM_FEEDBACK_MAX_AGE_S = 0.5
# `/button_surface_pose` runs at roughly 10-13 Hz in simulation.  Requiring a
# burst of them proves the detector, TF chain and depth pipeline all converged.
SURFACE_MIN_SAMPLES = 40
# Highest-confidence control, used only to probe the perception chain.
WARMUP_BUTTON = '3'

LATCHED = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

# `/elevator_task/result` carries non-terminal chatter as well: it is
# republished as 'RUNNING: button=<b>' as soon as a command is accepted, and
# as 'No task has run' after every reset.  Only these prefixes end a trial.
TERMINAL_RESULT_PREFIXES = ('COMPLETE:', 'FAILED:', 'REJECTED:')

# The manager reaches home on its own before every task, so a trial that ends
# without home means the next trial would start from an unknown pose.
HOME_LOST_MARKERS = ('home_reached=false',)


class StabilityTrial(Node):
    """Command the task manager and observe the physical button travel."""

    def __init__(self):
        super().__init__(f'button_stability_test_{os.getpid()}')
        self.results = []
        self.statuses = []
        self.completions = []
        self.deepest_depression_m = {}
        self.joint_samples = 0
        self.arm_feedback_at = 0.0
        self.arm_joint_positions = {}
        self.halt_joints = None
        self.visual_status = ''
        self.last_visual_tracking_status = ''
        self.surface_times = []

        self.create_subscription(
            PoseStamped,
            '/button_surface_pose',
            self._surface_callback,
            10,
        )
        self.create_subscription(
            String, '/elevator_task/result', self._result_callback, LATCHED
        )
        self.create_subscription(
            String, '/elevator_task/status', self._status_callback, LATCHED
        )
        self.create_subscription(
            Bool, '/elevator_task/completed', self._completion_callback,
            LATCHED,
        )
        self.create_subscription(
            JointState,
            '/elevator_button/joint_states',
            self._joint_callback,
            10,
        )
        self.create_subscription(
            JointState,
            ARM_FEEDBACK_TOPIC,
            self._arm_joint_callback,
            10,
        )
        self.create_subscription(
            Int8, '/servo_node/status', self._servo_status_callback, 10,
        )
        self.create_subscription(
            String, '/button_visual_servo/status',
            self._visual_status_callback, 10,
        )
        self._command = self.create_publisher(
            String, '/elevator_task/command', 10
        )
        self._selection = self.create_publisher(
            String, '/button_selection', LATCHED
        )
        self._reset = self.create_client(Trigger, '/elevator_task_manager/reset')
        self._stop = self.create_client(Trigger, '/elevator_task_manager/stop')

    def _result_callback(self, message):
        self.results.append(str(message.data))

    def _status_callback(self, message):
        self.statuses.append(str(message.data))

    def _completion_callback(self, message):
        self.completions.append(bool(message.data))

    def _joint_callback(self, message):
        self.joint_samples += 1
        positions = dict(zip(message.name, message.position))
        for button in BUTTONS:
            position = positions.get(button_joint(button))
            if position is None:
                continue
            # The press joint is built with travel [-4 mm, 0].
            depth = max(0.0, -float(position))
            if depth > self.deepest_depression_m.get(button, 0.0):
                self.deepest_depression_m[button] = depth

    def _arm_joint_callback(self, message):
        positions = dict(zip(message.name, message.position))
        if all(name in positions for name in ARM_JOINTS) and all(
            math.isfinite(float(positions[name])) for name in ARM_JOINTS
        ):
            self.arm_feedback_at = time.monotonic()
            self.arm_joint_positions = {
                name: float(positions[name]) for name in ARM_JOINTS
            }

    def _servo_status_callback(self, message):
        if int(message.data) == 2 and self.halt_joints is None:
            self.halt_joints = dict(self.arm_joint_positions)

    def _visual_status_callback(self, message):
        self.visual_status = str(message.data)
        if self.visual_status.startswith((
            'VISUAL_', 'VISION_LOSS_', 'ORIENTING_',
            'LOCKED_APPROACH_', 'REACQUIRING_',
        )):
            self.last_visual_tracking_status = self.visual_status

    def _surface_callback(self, message):
        del message
        self.surface_times.append(time.monotonic())

    def arm_feedback_fresh(self):
        return (
            time.monotonic() - self.arm_feedback_at
            <= ARM_FEEDBACK_MAX_AGE_S
        )

    def observation_pipeline_live(self):
        """The RGB-D target stream must be flowing, not merely subscribed.

        A cold stack publishes a first valid detection long before the
        planner's frozen observation window is usable.  Starting trials then
        would record a start-up artefact as a button failure, so require a
        burst of surface poses with the newest one still fresh.
        """
        if len(self.surface_times) < SURFACE_MIN_SAMPLES:
            return False
        return (
            time.monotonic() - self.surface_times[-1]
            <= ARM_FEEDBACK_MAX_AGE_S
        )

    # -- primitives ------------------------------------------------------

    def spin_for(self, seconds):
        deadline = time.monotonic() + max(0.0, float(seconds))
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.10, deadline
                                                  - time.monotonic()))

    def wait_for_stack(self, timeout):
        """Block until a task can actually be commanded.

        Everything the state machine's first phase needs must already be
        satisfiable: the manager itself, a subscriber for commands, the reset
        service, the panel joint feedback, and a fresh complete sample of the
        arm feedback the approach planner validates before homing.
        """
        deadline = time.monotonic() + float(timeout)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.20)
            ready = (
                self.results
                and self._command.get_subscription_count() > 0
                and self._reset.service_is_ready()
                and self.joint_samples > 0
                and self.arm_feedback_fresh()
            )
            if ready:
                # Let controllers and the camera chain settle before the first
                # MoveIt plan, so trial 1 is not penalised by start-up jitter.
                self.spin_for(2.0)
                return
        raise RuntimeError(
            'task stack not ready: need /elevator_task/result, a subscriber '
            'on /elevator_task/command, /elevator_task_manager/reset, '
            '/elevator_button/joint_states and fresh complete '
            f'{ARM_FEEDBACK_TOPIC}'
        )

    def warm_up_perception(self, timeout):
        """Prove the detector, TF chain and depth pipeline all work.

        `/button_surface_pose` is only published for a *selected* and stable
        button, so the perception chain cannot be checked passively.  Select a
        high-confidence class, require a continuous burst of surface poses,
        then clear the selection so trial 1 measures the button rather than
        the start-up of the stack.
        """
        self.surface_times.clear()
        deadline = time.monotonic() + float(timeout)
        while rclpy.ok() and time.monotonic() < deadline:
            self._selection.publish(String(data=WARMUP_BUTTON))
            self.spin_for(0.5)
            if self.observation_pipeline_live():
                self._selection.publish(String(data=''))
                self.spin_for(1.0)
                self.surface_times.clear()
                return True
        self._selection.publish(String(data=''))
        return False

    def reset(self, timeout):
        """Clear the manager latch so the next trial starts from IDLE."""
        deadline = time.monotonic() + float(timeout)
        while rclpy.ok() and time.monotonic() < deadline:
            if not self._reset.service_is_ready():
                self.spin_for(0.20)
                continue
            future = self._reset.call_async(Trigger.Request())
            while rclpy.ok() and not future.done():
                if time.monotonic() > deadline:
                    return False
                rclpy.spin_once(self, timeout_sec=0.10)
            response = future.result()
            if response is not None and response.success:
                return True
            # 'Cannot reset while a task is running' - let it settle.
            self.spin_for(0.5)
        return False

    def stop(self):
        if not self._stop.service_is_ready():
            self._stop.wait_for_service(timeout_sec=2.0)
        future = self._stop.call_async(Trigger.Request())
        deadline = time.monotonic() + 10.0
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.10)
        return future.done()

    def run_trial(self, button, timeout):
        """Publish one press command and wait for the manager's verdict.

        The manager's first result is 'RUNNING: button=<b>'; only COMPLETE,
        FAILED or REJECTED end the trial.  A busy manager rejects a command
        on the status topic without publishing a result, so that is watched
        too and must not be mistaken for a long-running task.
        """
        result_baseline = len(self.results)
        status_baseline = len(self.statuses)
        self.deepest_depression_m[button] = 0.0
        self.halt_joints = None
        self._command.publish(String(data=f'press {button}'))

        deadline = time.monotonic() + float(timeout)
        next_progress_at = time.monotonic() + 10.0
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.20)
            if time.monotonic() >= next_progress_at:
                print(
                    f'[PROGRESS] {button} {self.last_visual_tracking_status}',
                    flush=True,
                )
                next_progress_at = time.monotonic() + 10.0
            for text in self.results[result_baseline:]:
                if text.startswith(TERMINAL_RESULT_PREFIXES):
                    return text
            for text in self.statuses[status_baseline:]:
                if text.startswith('REJECTED:'):
                    return f'FAILED: {text}'
        return (
            f'FAILED: no terminal result within {timeout:.0f}s; '
            f'last status={self.statuses[-1] if self.statuses else "<none>"}'
        )


def button_joint(button):
    """Gazebo press joint name for a panel control."""
    return f'button_{button}_press_joint'


def parse_buttons(value):
    requested = [item.strip() for item in str(value).split(',') if item.strip()]
    if not requested:
        raise argparse.ArgumentTypeError('at least one button is required')
    unknown = [item for item in requested if item not in BUTTONS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f'unknown button(s) {unknown}; choose from {list(BUTTONS)}'
        )
    return requested


def arguments():
    parser = argparse.ArgumentParser(
        description=(
            'Press every panel button several times through the elevator '
            'task state machine and report per-button stability.'
        ),
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='required acknowledgement that this script moves the arm',
    )
    parser.add_argument(
        '--buttons',
        type=parse_buttons,
        default=list(BUTTONS),
        help=f'comma separated subset of {list(BUTTONS)}',
    )
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument(
        '--trial-timeout',
        type=float,
        default=240.0,
        help='wall-clock budget for one complete press task',
    )
    parser.add_argument(
        '--dependency-timeout',
        type=float,
        default=300.0,
        help='how long to wait for a cold stack to become commandable',
    )
    parser.add_argument('--settle-seconds', type=float, default=1.0)
    parser.add_argument(
        '--pressed-depth',
        type=float,
        default=0.0020,
        help='joint travel that proves the button was physically depressed',
    )
    parser.add_argument(
        '--continue-on-home-failure',
        action='store_true',
        help=(
            'keep testing after a trial that did not return home. Unsafe in '
            'general: the next trial would start from an unknown pose.'
        ),
    )
    parser.add_argument(
        '--log-dir',
        default='/workspace/test_logs',
    )
    return parser.parse_args()


def write_report(path, rows, buttons, rounds, passed, failures):
    """Write the operator-facing markdown report."""
    lines = [
        '# Button stability report',
        '',
        f'- generated: {datetime.now().isoformat(timespec="seconds")}',
        f'- buttons: {", ".join(buttons)}',
        f'- rounds per button: {rounds}',
        f'- trials: {len(rows)}, passed: {passed}, failed: {failures}',
        '',
        '## Per button',
        '',
        '| button | trials | pass | pass rate | median s | max s | '
        'min press depth (mm) |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ]
    for button in buttons:
        subset = [row for row in rows if row['button'] == button]
        if not subset:
            continue
        good = [row for row in subset if row['result'] == 'PASS']
        durations = [float(row['seconds']) for row in subset]
        depths = [float(row['press_depth_mm']) for row in good]
        lines.append(
            f'| {button} | {len(subset)} | {len(good)} | '
            f'{100.0 * len(good) / len(subset):.0f}% | '
            f'{statistics.median(durations):.1f} | '
            f'{max(durations):.1f} | '
            f'{min(depths):.2f} |' if good else
            f'| {button} | {len(subset)} | 0 | 0% | '
            f'{statistics.median(durations):.1f} | {max(durations):.1f} | - |'
        )
    if fail_rows := [row for row in rows if row['result'] != 'PASS']:
        lines += ['', '## Failures', '']
        for row in fail_rows:
            lines.append(
                f'- round {row["round"]} `{row["button"]}`: {row["detail"]}'
            )
    lines += ['', '## All trials', '',
              '| # | button | round | result | seconds | press depth (mm) |',
              '| --- | --- | --- | --- | --- | --- |']
    for index, row in enumerate(rows, start=1):
        lines.append(
            f'| {index} | {row["button"]} | {row["round"]} | '
            f'{row["result"]} | {row["seconds"]} | {row["press_depth_mm"]} |'
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = arguments()
    if not args.execute:
        print(
            'Refusing to move the arm without --execute.\n'
            'Review the workcell, keep the emergency stop ready, then run:\n'
            '  ./scripts/test_button_stability.sh --execute '
            '--rounds 3',
            file=sys.stderr,
        )
        return 2
    if args.rounds < 1:
        print('--rounds must be at least 1', file=sys.stderr)
        return 2

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = log_dir / f'button_stability_{stamp}.csv'
    report_path = log_dir / f'button_stability_{stamp}.md'

    rows = []
    failures = 0
    aborted = ''
    rclpy.init()
    node = StabilityTrial()
    try:
        print('Waiting for the task stack...', flush=True)
        node.wait_for_stack(args.dependency_timeout)
        print('Task stack ready. Warming up perception...', flush=True)
        if not node.warm_up_perception(args.dependency_timeout):
            raise RuntimeError(
                'perception never produced a continuous /button_surface_pose '
                'stream; check the detector, camera TF and depth topics'
            )
        print('Perception warmed up. Starting trials.\n', flush=True)

        with csv_path.open('w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    'button', 'round', 'result', 'seconds',
                    'press_depth_mm', 'detail',
                ],
            )
            writer.writeheader()

            for button in args.buttons:
                for round_number in range(1, args.rounds + 1):
                    print(
                        f'=== {button} round {round_number}/{args.rounds} ===',
                        flush=True,
                    )
                    if not node.reset(20.0):
                        aborted = (
                            'task manager did not accept a reset between '
                            'trials'
                        )
                        print(f'[ABORT] {aborted}', flush=True)
                        break

                    started = time.monotonic()
                    detail = node.run_trial(button, args.trial_timeout)
                    seconds = time.monotonic() - started
                    depth = node.deepest_depression_m.get(button, 0.0)

                    manager_ok = detail.startswith('COMPLETE:')
                    physical_ok = depth >= args.pressed_depth
                    passed = manager_ok and physical_ok
                    if manager_ok and not physical_ok:
                        detail += (
                            f'; joint travel {depth * 1000.0:.2f} mm < '
                            f'{args.pressed_depth * 1000.0:.2f} mm'
                        )
                    if not manager_ok and node.halt_joints:
                        detail += f'; servo_halt_joints={node.halt_joints}'
                    if not manager_ok:
                        detail += (
                            f'; visual_status={node.visual_status}'
                            f'; last_tracking={node.last_visual_tracking_status}'
                        )

                    if not passed:
                        failures += 1
                    print(
                        f'[{"PASS" if passed else "FAIL"}] {button} '
                        f'round={round_number} seconds={seconds:.1f} '
                        f'depth={depth * 1000.0:.2f}mm detail={detail}',
                        flush=True,
                    )
                    row = {
                        'button': button,
                        'round': round_number,
                        'result': 'PASS' if passed else 'FAIL',
                        'seconds': f'{seconds:.1f}',
                        'press_depth_mm': f'{depth * 1000.0:.3f}',
                        'detail': detail,
                    }
                    rows.append(row)
                    writer.writerow(row)
                    csv_file.flush()

                    home_lost = any(
                        marker in detail for marker in HOME_LOST_MARKERS
                    )
                    if home_lost and not args.continue_on_home_failure:
                        aborted = (
                            'the arm did not return home; remaining trials '
                            'are blocked for safety (pass '
                            '--continue-on-home-failure to override)'
                        )
                        print(f'[ABORT] {aborted}', flush=True)
                        break
                    if home_lost:
                        print(
                            '[WARN] home not confirmed; continuing because '
                            '--continue-on-home-failure was given',
                            flush=True,
                        )
                    node.spin_for(args.settle_seconds)
                if aborted:
                    break
    except KeyboardInterrupt:
        print('\nInterrupted by operator; stopping the task.', file=sys.stderr)
        try:
            node.stop()
        except Exception:
            pass
        failures += 1
        aborted = 'interrupted by operator'
    except Exception as error:
        print(f'Preflight failed: {error}', file=sys.stderr)
        failures += 1
        aborted = str(error)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    passed = sum(1 for row in rows if row['result'] == 'PASS')
    if rows:
        write_report(
            report_path, rows, args.buttons, args.rounds, passed, failures
        )

    print('\n================ STABILITY SUMMARY ================', flush=True)
    for button in args.buttons:
        subset = [row for row in rows if row['button'] == button]
        if not subset:
            print(f'{button:>6}: not run', flush=True)
            continue
        good = sum(1 for row in subset if row['result'] == 'PASS')
        durations = [float(row['seconds']) for row in subset]
        depths = [
            float(row['press_depth_mm'])
            for row in subset if row['result'] == 'PASS'
        ]
        depth_text = f'{min(depths):.2f}mm' if depths else 'n/a'
        print(
            f'{button:>6}: {good}/{len(subset)} pass, '
            f'median {statistics.median(durations):.1f}s, '
            f'min press depth {depth_text}',
            flush=True,
        )
    print(
        f'TOTAL : {passed}/{len(rows)} pass, {failures} failure(s)',
        flush=True,
    )
    if aborted:
        print(f'ABORTED: {aborted}', flush=True)
    print(f'CSV   : {csv_path}', flush=True)
    if rows:
        print(f'REPORT: {report_path}', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
