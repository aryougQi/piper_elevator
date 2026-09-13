#!/usr/bin/env python3
"""Gate real-arm commands around trajectories and live Servo control."""

import threading
import time

from action_msgs.msg import GoalStatus
from action_msgs.msg import GoalStatusArray
from sensor_msgs.msg import JointState

from piper_elevator_app.control_gate_core import ControlGatePolicy
from piper_elevator_app.control_gate_core import desired_control_mode
from piper_elevator_app.control_gate_core import (
    joint_positions_within_tolerance,
)

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from std_srvs.srv import SetBool


class TrajectoryControlGate(Node):
    """Forward CAN commands for an action or a live Servo heartbeat."""

    def __init__(self):
        """Configure action-state gating and crash-safe time limits."""
        super().__init__('piper_pika_control_gate')
        self.declare_parameter(
            'status_topic',
            '/arm_controller/follow_joint_trajectory/_action/status',
        )
        self.declare_parameter('gate_service_name', '/control_enable')
        self.declare_parameter(
            'servo_gate_service_name',
            '/servo_control_enable',
        )
        self.declare_parameter('maximum_trajectory_gate_seconds', 45.0)
        self.declare_parameter(
            'trajectory_command_topic',
            '/control/joint_states',
        )
        self.declare_parameter(
            'trajectory_feedback_topic',
            '/feedback/joint_states',
        )
        self.declare_parameter(
            'trajectory_joint_names',
            ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
        )
        self.declare_parameter('trajectory_settle_tolerance_rad', 0.025)
        self.declare_parameter('trajectory_settle_minimum_seconds', 0.25)
        self.declare_parameter('trajectory_settle_timeout_seconds', 25.0)
        self.declare_parameter('servo_authorization_service', '~/servo_enable')
        self.declare_parameter('servo_heartbeat_timeout_seconds', 0.75)
        self.declare_parameter('hardware_gate_ack_timeout_seconds', 1.5)
        self._condition = threading.Condition()
        self._authorization_callback_group = ReentrantCallbackGroup()

        self._active_states = {
            GoalStatus.STATUS_ACCEPTED,
            GoalStatus.STATUS_EXECUTING,
            GoalStatus.STATUS_CANCELING,
        }
        self._gate_mode = None
        self._desired_gate_mode = None
        self._pending_gate_future = None
        self._pending_gate_request = None
        self._pending_gate_started_at = None
        self._uncertain_gate_mode = None
        self._trajectory_target = {}
        self._trajectory_feedback = {}
        self._trajectory_terminal_at = None
        self._trajectory_joint_names = list(
            self.get_parameter('trajectory_joint_names').value
        )
        self._policy = ControlGatePolicy(
            maximum_trajectory_seconds=float(
                self.get_parameter(
                    'maximum_trajectory_gate_seconds'
                ).value
            ),
            servo_heartbeat_seconds=float(
                self.get_parameter(
                    'servo_heartbeat_timeout_seconds'
                ).value
            ),
            trajectory_settle_seconds=float(
                self.get_parameter(
                    'trajectory_settle_timeout_seconds'
                ).value
            ),
        )
        self._gate_clients = {
            'trajectory': self.create_client(
                SetBool,
                str(self.get_parameter('gate_service_name').value),
            ),
            'servo': self.create_client(
                SetBool,
                str(
                    self.get_parameter('servo_gate_service_name').value
                ),
            ),
        }
        self.create_subscription(
            GoalStatusArray,
            str(self.get_parameter('status_topic').value),
            self._status_callback,
            10,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter('trajectory_command_topic').value),
            self._trajectory_command_callback,
            10,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter('trajectory_feedback_topic').value),
            self._trajectory_feedback_callback,
            10,
        )
        self.create_service(
            SetBool,
            str(
                self.get_parameter('servo_authorization_service').value
            ),
            self._servo_authorization_callback,
            callback_group=self._authorization_callback_group,
        )
        self.create_timer(0.2, self._watchdog_callback)

    def _status_callback(self, message):
        with self._condition:
            self._update_status_locked(message)

    def _update_status_locked(self, message):
        active = any(
            status.status in self._active_states
            for status in message.status_list
        )
        if active and self._policy.servo_authorized:
            self.get_logger().error(
                'Trajectory became active during Servo authorization; '
                'revoking Servo and switching command modes fail-closed'
            )
            self._policy.update_servo(False, time.monotonic())
        was_active = self._policy.trajectory_active
        if active and not was_active:
            self._trajectory_target = {}
            self._trajectory_terminal_at = None
        needs_settling = (
            was_active
            and not active
            and bool(self._trajectory_target)
        )
        self._policy.update_trajectory(
            active,
            time.monotonic(),
            needs_settling=needs_settling,
        )
        if needs_settling:
            self._trajectory_terminal_at = time.monotonic()
            self.get_logger().info(
                'Trajectory action finished; holding the final command until '
                'real joint feedback settles'
            )
        self._update_gate()

    @staticmethod
    def _joint_map(message):
        return {
            str(name): float(message.position[index])
            for index, name in enumerate(message.name)
            if index < len(message.position)
        }

    def _trajectory_command_callback(self, message):
        with self._condition:
            if self._policy.trajectory_active or self._policy.trajectory_settling:
                self._trajectory_target = self._joint_map(message)

    def _trajectory_feedback_callback(self, message):
        with self._condition:
            self._update_feedback_locked(message)

    def _update_feedback_locked(self, message):
        self._trajectory_feedback = self._joint_map(message)
        minimum_settle = float(
            self.get_parameter('trajectory_settle_minimum_seconds').value
        )
        if (
            self._policy.trajectory_settling
            and self._trajectory_terminal_at is not None
            and time.monotonic() - self._trajectory_terminal_at
            >= minimum_settle
            and self._trajectory_tracking_complete()
        ):
            self._policy.complete_trajectory_settling()
            self._trajectory_terminal_at = None
            self.get_logger().info(
                'Real joint feedback reached the final trajectory command'
            )
            self._update_gate()

    def _trajectory_tracking_complete(self):
        return joint_positions_within_tolerance(
            self._trajectory_target,
            self._trajectory_feedback,
            self._trajectory_joint_names,
            float(
                self.get_parameter(
                    'trajectory_settle_tolerance_rad'
                ).value
            ),
        )

    def _servo_authorization_callback(self, request, response):
        enabled = bool(request.data)
        timeout = float(
            self.get_parameter('hardware_gate_ack_timeout_seconds').value
        )
        deadline = time.monotonic() + timeout
        with self._condition:
            if enabled and (
                self._policy.trajectory_active
                or self._policy.trajectory_settling
                or self._uncertain_gate_mode is not None
            ):
                response.success = False
                response.message = (
                    'Servo authorization rejected while trajectory control '
                    'is active or settling, or hardware gate state is unknown'
                )
                return response
            self._policy.update_servo(enabled, time.monotonic())
            self._update_gate()
            while True:
                pending_mode = (
                    self._pending_gate_request[0]
                    if self._pending_gate_request is not None else None
                )
                confirmed = (
                    self._policy.servo_authorized
                    and self._gate_mode == 'servo'
                    and self._pending_gate_future is None
                    and self._uncertain_gate_mode is None
                ) if enabled else (
                    self._gate_mode != 'servo'
                    and pending_mode != 'servo'
                    and self._uncertain_gate_mode != 'servo'
                )
                if confirmed:
                    response.success = True
                    response.message = (
                        'Hardware Servo control confirmed; heartbeat accepted'
                        if enabled else 'Hardware Servo control closure confirmed'
                    )
                    return response
                if enabled and not self._policy.servo_authorized:
                    response.success = False
                    response.message = 'Servo authorization was revoked'
                    return response
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    self._policy.update_servo(False, time.monotonic())
                    if pending_mode is not None:
                        self._uncertain_gate_mode = pending_mode
                    elif self._gate_mode == 'servo':
                        self._uncertain_gate_mode = 'servo'
                    self._update_gate()
                    response.success = False
                    response.message = (
                        'Hardware Servo gate confirmation timed out; '
                        'authorization revoked, closure not yet confirmed'
                    )
                    return response
                # A delayed initial acknowledgment must not expire the
                # heartbeat before its bounded authorization wait finishes.
                if enabled:
                    self._policy.update_servo(True, time.monotonic())
                self._condition.wait(min(remaining, 0.05))

    def _watchdog_callback(self):
        self._update_gate()

    def _update_gate(self):
        with self._condition:
            self._update_gate_locked()

    def _update_gate_locked(self):
        if (
            self._pending_gate_started_at is not None
            and self._uncertain_gate_mode is None
            and time.monotonic() - self._pending_gate_started_at > float(
                self.get_parameter('hardware_gate_ack_timeout_seconds').value
            )
        ):
            self._uncertain_gate_mode = self._pending_gate_request[0]
            self._policy.update_servo(False, time.monotonic())
            if self._uncertain_gate_mode == 'trajectory':
                self._policy.trajectory_timed_out = True
            self.get_logger().error(
                'Hardware gate acknowledgment timed out; state is unknown. '
                'New enables are blocked until the pending request returns '
                'and hardware closure is confirmed'
            )
        desired_mode, timed_out_now = desired_control_mode(
            self._policy,
            time.monotonic()
        )
        if timed_out_now:
            self.get_logger().error(
                'Trajectory command gate exceeded its active/settling hard '
                'time limit; closing the real-arm command path'
            )
        self._set_gate_mode(desired_mode)

    def _set_gate_mode(self, desired_mode):
        with self._condition:
            self._set_gate_mode_locked(desired_mode)

    def _set_gate_mode_locked(self, desired_mode):
        if desired_mode not in {None, 'trajectory', 'servo'}:
            raise ValueError(f'Unsupported control gate mode: {desired_mode}')
        self._desired_gate_mode = desired_mode
        if self._pending_gate_future is not None:
            return
        if (
            self._gate_mode == self._desired_gate_mode
            and self._uncertain_gate_mode is None
        ):
            return

        if self._uncertain_gate_mode is not None:
            request_mode = self._uncertain_gate_mode
            enable = False
        elif self._gate_mode is not None:
            request_mode = self._gate_mode
            enable = False
        else:
            request_mode = self._desired_gate_mode
            enable = True
        if request_mode is None:
            return
        client = self._gate_clients[request_mode]
        if not client.wait_for_service(timeout_sec=0.0):
            return
        request = SetBool.Request()
        request.data = enable
        self._pending_gate_request = (request_mode, enable)
        self._pending_gate_started_at = time.monotonic()
        self._pending_gate_future = client.call_async(request)
        self._pending_gate_future.add_done_callback(
            self._gate_response_callback
        )

    def _gate_response_callback(self, future):
        with self._condition:
            self._gate_response_locked(future)
            self._condition.notify_all()

    def _gate_response_locked(self, future):
        requested_mode, requested_enable = self._pending_gate_request
        self._pending_gate_future = None
        self._pending_gate_request = None
        self._pending_gate_started_at = None
        request_succeeded = False
        try:
            result = future.result()
        except Exception as error:
            self._uncertain_gate_mode = requested_mode
            self._policy.update_servo(False, time.monotonic())
            if requested_mode == 'trajectory':
                self._policy.trajectory_timed_out = True
            self.get_logger().error(
                f'Hardware command gate service failed: {error}'
            )
        else:
            if result is not None and result.success:
                self._gate_mode = (
                    requested_mode if requested_enable else None
                )
                if (
                    not requested_enable
                    and self._uncertain_gate_mode == requested_mode
                ):
                    self._uncertain_gate_mode = None
                request_succeeded = True
                state = 'opened' if requested_enable else 'closed'
                self.get_logger().info(
                    f'Hardware {requested_mode} command gate confirmed '
                    f'{state}'
                )
            else:
                if result is None:
                    self._uncertain_gate_mode = requested_mode
                    self._policy.update_servo(False, time.monotonic())
                    if requested_mode == 'trajectory':
                        self._policy.trajectory_timed_out = True
                message = '' if result is None else result.message
                self.get_logger().error(
                    'Hardware command gate request was rejected'
                    + (f': {message}' if message else '')
                )
        # Reconcile a desired-state change that arrived while the successful
        # service request was pending. Failures retry on the 200 ms watchdog
        # instead of creating a tight asynchronous retry loop.
        if request_succeeded:
            self._set_gate_mode(self._desired_gate_mode)


def main(args=None):
    """Run the real-arm trajectory control gate."""
    rclpy.init(args=args)
    node = TrajectoryControlGate()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
