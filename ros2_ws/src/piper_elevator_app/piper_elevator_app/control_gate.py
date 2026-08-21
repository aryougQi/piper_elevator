#!/usr/bin/env python3

import time

from action_msgs.msg import GoalStatus
from action_msgs.msg import GoalStatusArray
from std_srvs.srv import SetBool

import rclpy
from rclpy.node import Node


class TrajectoryControlGate(Node):
    """Forward CAN commands only while a trajectory action is active."""

    def __init__(self):
        super().__init__('piper_pika_control_gate')
        self.declare_parameter(
            'status_topic',
            '/arm_controller/follow_joint_trajectory/_action/status',
        )
        self.declare_parameter('gate_service_name', '/control_enable')
        self.declare_parameter('status_timeout_seconds', 1.0)

        self._active_states = {
            GoalStatus.STATUS_ACCEPTED,
            GoalStatus.STATUS_EXECUTING,
            GoalStatus.STATUS_CANCELING,
        }
        self._gate_open = None
        self._last_status_time = None
        self._client = self.create_client(
            SetBool,
            str(self.get_parameter('gate_service_name').value),
        )
        self.create_subscription(
            GoalStatusArray,
            str(self.get_parameter('status_topic').value),
            self._status_callback,
            10,
        )
        self.create_timer(0.2, self._watchdog_callback)

    def _status_callback(self, message):
        self._last_status_time = time.monotonic()
        active = any(
            status.status in self._active_states
            for status in message.status_list
        )
        self._set_gate(active)

    def _watchdog_callback(self):
        if self._last_status_time is None:
            self._set_gate(False)
            return
        timeout = float(
            self.get_parameter('status_timeout_seconds').value
        )
        if time.monotonic() - self._last_status_time > timeout:
            self._set_gate(False)

    def _set_gate(self, open_gate):
        if self._gate_open is open_gate:
            return
        if not self._client.wait_for_service(timeout_sec=0.0):
            return
        request = SetBool.Request()
        request.data = bool(open_gate)
        self._client.call_async(request)
        self._gate_open = open_gate
        state = 'opened' if open_gate else 'closed'
        self.get_logger().info(f'Hardware command gate {state}')


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryControlGate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
