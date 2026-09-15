#!/usr/bin/env python3
"""Plan to a captured pose, preview it, and execute only after typed confirmation."""
import argparse
import json
import math
import time
from pathlib import Path

import rclpy
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import Constraints, JointConstraint, DisplayTrajectory
from moveit_msgs.srv import GetMotionPlan
from sensor_msgs.msg import JointState


def wait(node, future, seconds):
    rclpy.spin_until_future_complete(node, future, timeout_sec=seconds)
    if not future.done():
        raise RuntimeError('ROS request timed out')
    return future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sample', type=Path)
    args = parser.parse_args()
    names = ['joint' + str(i) for i in range(1, 7)]
    sample = json.loads(args.sample.read_text())
    target = sample['joints']
    if sample.get('joint_names') != names or len(target) != 6 or not all(math.isfinite(x) for x in target):
        raise ValueError('Invalid sample joint names or positions')
    rclpy.init()
    node = rclpy.create_node('return_to_handeye_sample')
    latest = {}
    minimum_stamp_ns = 0
    def feedback(msg):
        values = dict(zip(msg.name, msg.position))
        age = (node.get_clock().now().nanoseconds - msg.header.stamp.sec * 10**9 - msg.header.stamp.nanosec) / 1e9
        stamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        if stamp_ns >= minimum_stamp_ns and all(n in values and math.isfinite(values[n]) for n in names) and 0 <= age < 0.5:
            latest.update(values=[values[n] for n in names], received=time.monotonic())
    sub = node.create_subscription(JointState, '/feedback/joint_states', feedback, 10)
    def fresh():
        nonlocal minimum_stamp_ns
        minimum_stamp_ns = node.get_clock().now().nanoseconds
        latest.clear()
        deadline = time.monotonic() + 5
        while not latest and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        if not latest:
            raise RuntimeError('No fresh actual joint feedback')
        return latest['values']
    handle = None
    finished = False
    try:
        start = fresh()
        client = node.create_client(GetMotionPlan, '/plan_kinematic_path')
        if not client.wait_for_service(timeout_sec=5):
            raise RuntimeError('MoveIt planning service unavailable')
        request = GetMotionPlan.Request()
        plan = request.motion_plan_request
        plan.group_name = 'arm'
        plan.pipeline_id = 'ompl'
        plan.num_planning_attempts = 5
        plan.allowed_planning_time = 10.0
        plan.max_velocity_scaling_factor = 0.05
        plan.max_acceleration_scaling_factor = 0.05
        plan.start_state.is_diff = True
        plan.start_state.joint_state.name = names
        plan.start_state.joint_state.position = start
        goal = Constraints()
        for name, value in zip(names, target):
            goal.joint_constraints.append(JointConstraint(joint_name=name, position=value,
                tolerance_above=0.001, tolerance_below=0.001, weight=1.0))
        plan.goal_constraints = [goal]
        response = wait(node, client.call_async(request), 20).motion_plan_response
        if response.error_code.val != 1 or not response.trajectory.joint_trajectory.points:
            raise RuntimeError(f'Planning failed: {response.error_code.val}; no motion sent')
        trajectory = response.trajectory
        display = DisplayTrajectory(trajectory_start=response.trajectory_start, trajectory=[trajectory])
        pub = node.create_publisher(DisplayTrajectory, '/display_planned_path',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        for _ in range(20):
            pub.publish(display)
            rclpy.spin_once(node, timeout_sec=0.1)
        print('Plan ready. Inspect the trajectory in RViz (MotionPlanning / Planned Path).')
        print('The physical board and surroundings may not be in the collision scene.')
        print('Target joints (rad):', target)
        if input('If the full path is clear, type EXECUTE then Enter; anything else cancels: ').strip() != 'EXECUTE':
            print('Cancelled; no motion sent.')
            return
        current = fresh()
        planned_start = dict(zip(trajectory.joint_trajectory.joint_names,
                                 trajectory.joint_trajectory.points[0].positions))
        if any(n not in planned_start or abs(v - planned_start[n]) > 0.006 for n, v in zip(names, current)):
            raise RuntimeError('Arm moved since planning; rerun to replan')
        action = ActionClient(node, ExecuteTrajectory, '/execute_trajectory')
        if not action.wait_for_server(timeout_sec=5):
            raise RuntimeError('ExecuteTrajectory unavailable')
        handle = wait(node, action.send_goal_async(ExecuteTrajectory.Goal(trajectory=trajectory)), 15)
        if not handle.accepted:
            raise RuntimeError('Execution rejected')
        result = wait(node, handle.get_result_async(), 180)
        finished = True
        if result.status != 4 or result.result.error_code.val != 1:
            raise RuntimeError(f'Execution failed: status={result.status}, code={result.result.error_code.val}')
        deadline = time.monotonic() + 30.0
        stable_since = None
        error = float('inf')
        while time.monotonic() < deadline:
            actual = fresh()
            error = max(abs(a-b) for a,b in zip(actual, target))
            if error <= 0.006:
                if stable_since is None:
                    stable_since = time.monotonic()
                if time.monotonic() - stable_since >= 1.0:
                    break
            else:
                stable_since = None
        else:
            raise RuntimeError(f'Actual feedback did not remain at target for 1 second; last error {error:.6f} rad')
        print(f'Execution finished; maximum actual joint error: {error:.6f} rad')
        if error > 0.006:
            raise RuntimeError('Actual pose differs from target; do not count as a repeat yet')
        print('Now run move_and_check.py --capture-only to save the repeat sample.')
    finally:
        if handle is not None and handle.accepted and not finished:
            print('Requesting trajectory cancellation; verify the arm has stopped.')
            try:
                wait(node, handle.cancel_goal_async(), 5)
            except Exception as exc:
                print('Cancellation could not be confirmed:', exc)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
