"""Coarse action lifecycle checks without ROS initialization or hardware."""

from concurrent.futures import Future
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import Constraints, RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from piper_elevator_app import button_approach_planner as module
from piper_elevator_app.button_approach_planner import ButtonApproachPlanner


def completed(value):
    future = Future()
    future.set_result(value)
    return future


def terminal(status=5, error=1):
    return SimpleNamespace(
        status=status,
        result=SimpleNamespace(error_code=SimpleNamespace(val=error)),
    )


class GoalHandle:
    def __init__(self, result=None, accepted=True, terminate_on_cancel=False):
        self.accepted = accepted
        self.result_future = Future() if result is None else completed(result)
        self.terminate_on_cancel = terminate_on_cancel
        self.cancel_count = 0

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        self.cancel_count += 1
        if self.terminate_on_cancel and not self.result_future.done():
            self.result_future.set_result(terminal())
        return completed(SimpleNamespace(goals_canceling=[]))


class ActionClient:
    def __init__(self, handle=None, send_future=None):
        self.send_future = send_future if send_future is not None else completed(handle)
        self.goals = []
        self.on_server_wait = None

    def wait_for_server(self, timeout_sec):
        if self.on_server_wait is not None:
            self.on_server_wait()
        return True

    def send_goal_async(self, goal):
        self.goals.append(goal)
        return self.send_future


def trajectory(duration=2, start=0.0, end=0.1):
    result = RobotTrajectory()
    result.joint_trajectory.joint_names = ['joint1']
    first = JointTrajectoryPoint()
    first.positions = [start]
    last = JointTrajectoryPoint()
    last.positions = [end]
    last.time_from_start.sec = duration
    result.joint_trajectory.points = [first, last]
    return result


class Planner(ButtonApproachPlanner):
    def __init__(self):
        self.values = {
            'allow_execution': True,
            'simulation_mode': True,
            'camera_calibration_valid': True,
            'plan_max_age_seconds': 30.0,
            'target_max_age_seconds': 1.0,
            'surface_normal_max_age_seconds': 0.5,
            'max_target_drift_m': 0.015,
            'observation_normal_tolerance_rad': 0.08,
            'joint_state_max_age_seconds': 0.5,
            'simulation_future_stamp_tolerance_seconds': 0.02,
            'joint_state_boundary_tolerance_rad': 1e-4,
            'execution_start_tolerance_rad': 0.02,
            'action_timeout_seconds': 30.0,
            'cancellation_timeout_seconds': 3.0,
            'execution_timeout_margin_seconds': 5.0,
            'execution_joint_tolerance_rad': 0.01,
            'execution_stable_samples': 3,
            'execution_stable_velocity_rad_s': 0.03,
            'execution_stop_hold_seconds': 0.3,
            'execution_settle_timeout_seconds': 0.5,
            'home_joint_names': ['joint1'],
            'home_joint_positions_rad': [0.0],
            'home_joint_tolerance_rad': 0.015,
            'joint_goal_tolerance_rad': 0.005,
            'home_execution_retries': 2,
            'home_retry_delay_seconds': 0.1,
            'planning_group': 'arm',
            'planning_attempts': 5,
            'planning_time_seconds': 3.0,
            'velocity_scaling': 0.15,
            'acceleration_scaling': 0.15,
            'gripper_motion_seconds': 1.0,
            'closed_gripper_position_m': 0.0,
        }
        self._lock = threading.Lock()
        self._busy = False
        self._motion_stop_unconfirmed = False
        self._base_frame = 'base_link'
        self._workspace_min = np.array([-1.0, -1.0, 0.0])
        self._workspace_max = np.array([1.0, 1.0, 1.5])
        self._latest_joint_positions = {'joint1': 0.0, 'center_joint': 0.012}
        self._arm_joint_limits = {'joint1': (-2.0, 2.0)}
        self._latest_joint_received_at = time.monotonic()
        self._latest_received_at = time.monotonic()
        self._latest_button = PoseStamped()
        self._planned_trajectory = trajectory()
        self._planned_target = PoseStamped()
        self._planned_button = np.zeros(3)
        self._latest_observation = {
            'button': np.zeros(3),
            'normal': np.array([1.0, 0.0, 0.0]),
            'received_at': self._latest_received_at,
            'selected_button': '3',
            'stamp_ns': 10_000_000_000,
        }
        self._planned_observation = dict(self._latest_observation)
        self._plan_created_at = time.monotonic()
        self._execute_client = ActionClient(GoalHandle(result=terminal(status=4)))
        self._move_group_client = ActionClient(GoalHandle())
        self._gripper_client = ActionClient(GoalHandle())
        self.statuses = []
        self.waits = []

    def get_parameter(self, name):
        return SimpleNamespace(value=self.values[name])

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=10_000_000_000))

    def _string_parameter(self, name):
        return str(self.values[name])

    def _publish_status(self, message):
        self.statuses.append(message)

    def _wait_for_future(self, future, timeout):
        self.waits.append(timeout)
        return future.result() if future.done() else None

    def _wait_for_real_joint_endpoint(self, trajectory):
        return True, 'settled'

    def _verify_approach_reached(self, target):
        return True, 'verified'

    def _load_arm_joint_limits(self):
        return None

    def _normalized_joint_positions(self, context):
        del context
        positions = self._latest_joint_positions.copy()
        if (
            time.monotonic() - self._latest_joint_received_at
            > self.values['joint_state_max_age_seconds']
            or any(name not in positions for name in self.values['home_joint_names'])
            or not all(np.isfinite(value) for value in positions.values())
        ):
            raise ValueError('Operation requires fresh complete real joint feedback')
        return positions

    def _motion_trajectory_is_safe(self, trajectory):
        return True, 'motion limits verified'

    def _trajectory_wrist_is_safe(self, trajectory):
        return True, 'handover limits verified'


def test_execution_rejects_recently_received_but_expired_capture():
    planner = Planner()
    planner._execution_button = planner._planned_button.copy()
    planner._execution_observation = dict(planner._planned_observation)
    planner._latest_observation['stamp_ns'] = 9_400_000_000
    valid, reason = planner._execution_target_is_current()
    assert not valid
    assert 'Paired observation is stale' in reason


class ArmLimitPlanner(Planner):
    _normalized_joint_positions = ButtonApproachPlanner._normalized_joint_positions
    _motion_trajectory_is_safe = ButtonApproachPlanner._motion_trajectory_is_safe
    _trajectory_wrist_is_safe = ButtonApproachPlanner._trajectory_wrist_is_safe

    def __init__(self):
        super().__init__()
        self._arm_joint_limits = dict(zip(
            [f'joint{index}' for index in range(1, 7)],
            [(-2.6179938, 2.6179938), (0.0, 3.1415926),
             (-2.9670597, 0.0), (-1.7453292, 1.7453292),
             (-1.2217304, 1.2217304), (-2.0943951, 2.0943951)],
        ))
        self.values.update({
            'home_joint_names': list(self._arm_joint_limits),
            'home_joint_positions_rad': [0.0] * 6,
            'joint_state_boundary_tolerance_rad': 1e-4,
            'trajectory_boundary_tolerance_rad': 1e-9,
            'joint_limit_margin_rad': 0.15,
            'wrist_singularity_joint': 'joint5',
            'minimum_abs_wrist_bend_rad': 0.4,
        })
        self._latest_joint_positions = dict.fromkeys(self._arm_joint_limits, 0.0)
        self._latest_joint_positions['center_joint'] = 0.012

    def get_logger(self):
        return SimpleNamespace(info=self.statuses.append)


def arm_trajectory(start=None, end=None):
    result = RobotTrajectory()
    result.joint_trajectory.joint_names = [f'joint{index}' for index in range(1, 7)]
    first = JointTrajectoryPoint()
    first.positions = [0.0] * 6 if start is None else list(start)
    last = JointTrajectoryPoint()
    last.positions = [0.1, 0.4, -0.7, 0.1, -0.6, 0.1] if end is None else list(end)
    last.time_from_start.sec = 2
    result.joint_trajectory.points = [first, last]
    return result


def test_live_joint3_roundoff_is_normalized_in_moveit_start_then_executes():
    planner = ArmLimitPlanner()
    live_value = 1.3014661678278e-08
    planner._latest_joint_positions['joint3'] = live_value
    wrapped = terminal(status=4)
    wrapped.result.planned_trajectory = arm_trajectory()
    planner._move_group_client = ActionClient(GoalHandle(result=wrapped))
    result, message = planner._plan_constraints(Constraints())
    assert result is not None, message
    state = planner._move_group_client.goals[0].request.start_state.joint_state
    sent = dict(zip(state.name, state.position))
    assert sent['joint3'] == 0.0
    assert sent['center_joint'] == 0.012
    assert planner._latest_joint_positions['joint3'] == live_value
    success, message = planner._execute_trajectory(result.planned_trajectory)
    assert success, message


@pytest.mark.parametrize('joint_index', range(6))
@pytest.mark.parametrize('side', ['lower', 'upper'])
def test_each_measured_joint_rejects_real_bound_violation_before_execution(joint_index, side):
    planner = ArmLimitPlanner()
    name = f'joint{joint_index + 1}'
    lower, upper = planner._arm_joint_limits[name]
    value = lower - 1.01e-4 if side == 'lower' else upper + 1.01e-4
    planner._latest_joint_positions[name] = value
    start = [0.0] * 6
    start[joint_index] = lower if side == 'lower' else upper
    success, message = planner._execute_trajectory(arm_trajectory(start=start))
    assert not success and name in message
    assert not planner._execute_client.goals


def test_home_uses_actual_asymmetric_arm_bounds():
    planner = ArmLimitPlanner()
    by_name = {joint.joint_name: joint for joint in planner._home_constraints().joint_constraints}
    assert by_name['joint2'].tolerance_below == 0.0
    assert by_name['joint2'].tolerance_above == pytest.approx(0.005)
    assert by_name['joint3'].tolerance_above == 0.0
    assert by_name['joint3'].tolerance_below == pytest.approx(0.005)
    for name, joint in by_name.items():
        lower, upper = planner._arm_joint_limits[name]
        assert lower <= joint.position - joint.tolerance_below
        assert joint.position + joint.tolerance_above <= upper


def test_home_goal_band_reserves_tracking_error_inside_acceptance_band():
    planner = ArmLimitPlanner()
    goal = {joint.joint_name: joint for joint in planner._home_constraints().joint_constraints}
    acceptance = {joint.joint_name: joint for joint in planner._home_constraints(acceptance=True).joint_constraints}
    tracking = planner.values['execution_joint_tolerance_rad']
    for name in planner.values['home_joint_names']:
        target, actual = goal[name], acceptance[name]
        if actual.tolerance_above > 0.0:
            assert target.tolerance_above + tracking <= actual.tolerance_above
        if actual.tolerance_below > 0.0:
            assert target.tolerance_below + tracking <= actual.tolerance_below
    assert acceptance['joint1'].tolerance_above == 0.015
    assert acceptance['joint2'].tolerance_below == 0.0
    assert acceptance['joint3'].tolerance_above == 0.0


@pytest.mark.parametrize('home_tolerance', [0.009, 0.010])
def test_home_configuration_requires_room_for_execution_error(home_tolerance):
    planner = ArmLimitPlanner()
    planner.values['home_joint_tolerance_rad'] = home_tolerance
    with pytest.raises(ValueError, match='must exceed execution tracking'):
        planner._home_constraints()


def home_feedback_clock(planner, monkeypatch, update=None):
    clock = SimpleNamespace(now=time.monotonic(), samples=0)
    planner._latest_joint_received_at = clock.now

    def sleep(seconds):
        clock.now += seconds
        clock.samples += 1
        planner._latest_joint_received_at = clock.now
        if update is not None:
            update(clock.samples)

    monkeypatch.setattr(module.time, 'monotonic', lambda: clock.now)
    monkeypatch.setattr(module.time, 'sleep', sleep)
    return clock


def test_already_home_verifies_fresh_stationary_feedback_without_motion(monkeypatch):
    planner = ArmLimitPlanner()
    planner._latest_joint_positions.update({
        'joint1': -0.0011591, 'joint2': 0.0116417, 'joint3': 9.24e-8,
        'joint4': 0.0133130, 'joint5': -0.0022470, 'joint6': 0.0123175,
    })
    clock = home_feedback_clock(planner, monkeypatch)
    response = planner._return_home_callback(None, SimpleNamespace())
    assert response.success, response.message
    assert 'already reached and stable' in response.message
    assert clock.samples >= 3
    assert not planner._move_group_client.goals
    assert not planner._execute_client.goals
    assert not planner._busy
    assert planner._planned_trajectory is None


def test_home_noop_rejects_drift_outside_actual_home_band(monkeypatch):
    planner = ArmLimitPlanner()
    home_feedback_clock(planner, monkeypatch, lambda _: planner._latest_joint_positions.update(joint1=0.016))
    response = planner._return_home_callback(None, SimpleNamespace())
    assert not response.success and 'outside the configured home region' in response.message
    assert not planner._move_group_client.goals
    assert not planner._execute_client.goals


def test_home_noop_requires_stationary_feedback_inside_region(monkeypatch):
    planner = ArmLimitPlanner()
    home_feedback_clock(planner, monkeypatch, lambda index: planner._latest_joint_positions.update(joint1=0.01 if index % 2 else -0.01))
    response = planner._return_home_callback(None, SimpleNamespace())
    assert not response.success and 'did not remain stationary' in response.message
    assert not planner._move_group_client.goals
    assert not planner._execute_client.goals


def test_home_noop_cannot_reuse_one_cached_feedback_sample(monkeypatch):
    planner = ArmLimitPlanner()
    clock = SimpleNamespace(now=time.monotonic())
    planner._latest_joint_received_at = clock.now
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock.now)
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: setattr(clock, 'now', clock.now + seconds))
    response = planner._return_home_callback(None, SimpleNamespace())
    assert not response.success
    assert not planner._move_group_client.goals
    assert not planner._execute_client.goals


@pytest.mark.parametrize('actual_joint1, expected_success', [(0.004, True), (0.016, False)])
def test_regular_home_verifies_actual_home_band_after_execution(monkeypatch, actual_joint1, expected_success):
    planner = ArmLimitPlanner()
    planner._latest_joint_positions['joint1'] = 0.2
    home_feedback_clock(planner, monkeypatch)
    calls = []
    planner._plan_constraints = lambda constraints: (
        SimpleNamespace(planned_trajectory=arm_trajectory()), ''
    )

    def execute(_):
        calls.append('execute')
        planner._latest_joint_positions['joint1'] = actual_joint1
        return True, 'trajectory endpoint verified'

    planner._execute_trajectory = execute
    response = planner._return_home_callback(None, SimpleNamespace())
    assert response.success is expected_success, response.message
    assert calls == ['execute']
    if not expected_success:
        assert 'outside the configured home region' in response.message


def test_planned_handover_reserves_tracking_error_inside_all_joint_limits():
    planner = ArmLimitPlanner()
    endpoint = [0.1, 0.4, -0.159, 0.1, -0.6, 0.1]
    trajectory = arm_trajectory(end=endpoint)
    assert planner._motion_trajectory_is_safe(trajectory)[0]
    safe, message = planner._trajectory_wrist_is_safe(trajectory)
    assert not safe and 'joint3' in message
    endpoint[2] = -0.161
    assert planner._trajectory_wrist_is_safe(arm_trajectory(end=endpoint))[0]
    endpoint[2] = -0.151
    assert planner._joint_configuration_is_safe(dict(zip(
        planner.values['home_joint_names'], endpoint,
    )))[0]


def active_arm(planner, handle=None):
    handle = GoalHandle() if handle is None else handle
    planner._active_arm_goal_handle = handle
    planner._active_arm_result_future = handle.result_future
    planner._execution_limit_violation = None
    return handle


@pytest.mark.parametrize('joint_index', range(6))
@pytest.mark.parametrize('side', ['lower', 'upper'])
def test_active_feedback_cancels_once_for_each_true_joint_bound_violation(joint_index, side):
    planner = ArmLimitPlanner()
    handle = active_arm(planner)
    name = f'joint{joint_index + 1}'
    lower, upper = planner._arm_joint_limits[name]
    value = lower - 0.001 if side == 'lower' else upper + 0.001
    planner._latest_joint_positions[name] = value
    planner._monitor_active_joint_limits()
    planner._monitor_active_joint_limits()
    assert handle.cancel_count == 1
    assert name in planner._execution_limit_violation
    assert 'outside' in planner._execution_limit_violation
    assert planner._latest_joint_positions[name] == value


@pytest.mark.parametrize('joint_index', range(6))
@pytest.mark.parametrize('side', ['lower', 'upper'])
def test_active_feedback_allows_tiny_boundary_roundoff_without_cancel(joint_index, side):
    planner = ArmLimitPlanner()
    handle = active_arm(planner)
    name = f'joint{joint_index + 1}'
    lower, upper = planner._arm_joint_limits[name]
    planner._latest_joint_positions[name] = lower - 1.3e-8 if side == 'lower' else upper + 1.3e-8
    planner._monitor_active_joint_limits()
    assert handle.cancel_count == 0
    assert planner._execution_limit_violation is None


def test_limit_monitor_does_not_cancel_completed_arm_action():
    planner = ArmLimitPlanner()
    handle = active_arm(planner, GoalHandle(result=terminal(status=4)))
    planner._latest_joint_positions['joint3'] = 0.001
    planner._monitor_active_joint_limits()
    assert handle.cancel_count == 0


def test_limit_monitor_does_not_apply_old_snapshot_to_new_handle(monkeypatch):
    planner = ArmLimitPlanner()
    old_handle = active_arm(planner)
    new_handle = GoalHandle()
    original = module.normalize_joint_positions

    def swap_handle(*args, **kwargs):
        planner._active_arm_goal_handle = new_handle
        planner._active_arm_result_future = new_handle.result_future
        raise ValueError('joint3 old action feedback outside bounds')

    monkeypatch.setattr(module, 'normalize_joint_positions', swap_handle)
    planner._monitor_active_joint_limits()
    assert old_handle.cancel_count == 0
    assert new_handle.cancel_count == 0
    assert planner._execution_limit_violation is None
    monkeypatch.setattr(module, 'normalize_joint_positions', original)
    planner._latest_joint_positions['joint3'] = 0.001
    planner._monitor_active_joint_limits()
    assert new_handle.cancel_count == 1


def test_limit_monitor_rechecks_terminal_race_before_cancellation(monkeypatch):
    planner = ArmLimitPlanner()
    handle = active_arm(planner)

    def complete_during_validation(*args, **kwargs):
        handle.result_future.set_result(terminal(status=4))
        raise ValueError('joint3 feedback outside bounds at terminal transition')

    monkeypatch.setattr(module, 'normalize_joint_positions', complete_during_validation)
    planner._monitor_active_joint_limits()
    assert handle.cancel_count == 0
    assert planner._execution_limit_violation is None


def test_planning_only_goal_has_no_active_arm_limit_monitor():
    planner = ArmLimitPlanner()
    handle = GoalHandle()
    planner._move_group_client = ActionClient(handle)
    planner._latest_joint_positions['joint3'] = 0.001
    planner._monitor_active_joint_limits()
    assert handle.cancel_count == 0


def test_execution_prioritizes_limit_error_and_clears_active_handle():
    planner = ArmLimitPlanner()
    handle = GoalHandle(terminate_on_cancel=True)

    def accept_then_update_feedback(goal):
        planner._execute_client.goals.append(goal)
        planner._latest_joint_positions['joint3'] = 0.001
        return completed(handle)

    planner._execute_client.send_goal_async = accept_then_update_feedback
    success, message = planner._execute_trajectory(arm_trajectory())
    assert not success
    assert 'joint3 active arm feedback' in message
    assert 'outside' in message
    assert handle.cancel_count == 1
    assert planner._active_arm_goal_handle is None
    assert planner._motion_stop_unconfirmed


def test_limit_cancellation_is_not_resent_when_result_times_out():
    planner = ArmLimitPlanner()
    handle = GoalHandle()

    def accept_then_update_feedback(goal):
        planner._execute_client.goals.append(goal)
        planner._latest_joint_positions['joint3'] = 0.001
        return completed(handle)

    planner._execute_client.send_goal_async = accept_then_update_feedback
    success, message = planner._execute_trajectory(arm_trajectory())
    assert not success and 'joint3 active arm feedback' in message
    assert handle.cancel_count == 1
    assert planner._motion_stop_unconfirmed
    assert planner._active_arm_goal_handle is None


def test_endpoint_wait_rejects_true_measured_violation_without_settle_delay(monkeypatch):
    planner = ArmLimitPlanner()
    planner._latest_joint_positions['joint3'] = 0.001

    def unexpected_sleep(_):
        raise AssertionError('True bound violation must fail without waiting')

    monkeypatch.setattr(module.time, 'sleep', unexpected_sleep)
    success, message = ButtonApproachPlanner._wait_for_real_joint_endpoint(planner, arm_trajectory())
    assert not success and 'joint3 executed joint feedback' in message


def test_stationary_grossly_out_of_bounds_feedback_cannot_clear_stop_guard(monkeypatch):
    planner = Planner()
    planner._execute_client = ActionClient(GoalHandle(terminate_on_cancel=True))
    planner._execute_trajectory(trajectory())
    clock = stationary_feedback(planner, monkeypatch, [2.001] * 5)
    assert planner._motion_stop_unconfirmed
    for _ in range(5):
        clock.now += 0.1
        planner._latest_joint_received_at = clock.now
        planner._latest_joint_positions['joint1'] = 2.0 + 1.3e-8
        planner._update_physical_stop_guards()
    assert not planner._motion_stop_unconfirmed


def test_execution_consumes_plan_on_controller_error():
    planner = Planner()
    planner._execute_client = ActionClient(GoalHandle(result=terminal(status=6, error=-4)))
    response = planner._execute_callback(None, SimpleNamespace())
    assert not response.success
    assert planner._planned_trajectory is None
    assert planner._execution_observation is not None
    assert not planner._busy
    second = planner._execute_callback(None, SimpleNamespace())
    assert not second.success
    assert len(planner._execute_client.goals) == 1


def test_stale_target_consumes_plan_without_sending():
    planner = Planner()
    planner._latest_received_at -= 2.0
    response = planner._execute_callback(None, SimpleNamespace())
    assert not response.success
    assert 'stale' in response.message
    assert planner._planned_trajectory is None
    assert not planner._execute_client.goals


def test_moved_target_consumes_plan_without_sending():
    planner = Planner()
    planner._latest_button.pose.position.x = 0.1
    response = planner._execute_callback(None, SimpleNamespace())
    assert not response.success
    assert 'moved' in response.message
    assert planner._planned_trajectory is None
    assert not planner._execute_client.goals


def test_target_is_rechecked_after_waiting_for_action_server():
    planner = Planner()
    planner._execute_client.on_server_wait = lambda: setattr(
        planner._latest_button.pose.position, 'x', 0.1
    )
    response = planner._execute_callback(None, SimpleNamespace())
    assert not response.success
    assert 'moved before execution' in response.message
    assert not planner._execute_client.goals


def test_changed_normal_at_same_position_is_rejected_before_motion():
    planner = Planner()
    planner._execute_client.on_server_wait = lambda: planner._latest_observation.update(
        normal=np.array([np.cos(0.2), np.sin(0.2), 0.0])
    )
    response = planner._execute_callback(None, SimpleNamespace())
    assert not response.success
    assert 'Panel normal changed before execution' in response.message
    assert not planner._execute_client.goals


def test_changed_selection_at_same_position_is_rejected_before_motion():
    planner = Planner()
    planner._execute_client.on_server_wait = lambda: planner._latest_observation.update(
        selected_button='4'
    )
    response = planner._execute_callback(None, SimpleNamespace())
    assert not response.success
    assert 'Selected button changed before execution' in response.message
    assert not planner._execute_client.goals


def test_lost_stable_observation_is_rejected_before_motion():
    planner = Planner()
    planner._execute_client.on_server_wait = lambda: setattr(
        planner, '_latest_observation', None
    )
    response = planner._execute_callback(None, SimpleNamespace())
    assert not response.success
    assert 'Stable paired observation unavailable' in response.message
    assert not planner._execute_client.goals


def test_execution_rejects_changed_or_stale_joint_start():
    planner = Planner()
    planner._latest_joint_positions['joint1'] = 0.1
    success, message = planner._execute_trajectory(trajectory())
    assert not success and 'planned start' in message
    planner._latest_joint_positions['joint1'] = 0.0
    planner._latest_joint_received_at -= 1.0
    success, message = planner._execute_trajectory(trajectory())
    assert not success and 'fresh complete' in message
    assert not planner._execute_client.goals


def test_execution_rejects_duration_exceeding_budget_before_motion():
    planner = Planner()
    success, message = planner._execute_trajectory(trajectory(duration=31))
    assert not success and 'budget' in message
    assert not planner._execute_client.goals


def test_execution_wait_uses_trajectory_duration_and_margin():
    planner = Planner()
    success, _ = planner._execute_trajectory(trajectory(duration=2))
    assert success
    assert planner.waits == [3.0, 7.0]


def test_execution_timeout_cancels_then_requires_measured_physical_stop(monkeypatch):
    planner = Planner()
    handle = GoalHandle(terminate_on_cancel=True)
    planner._execute_client = ActionClient(handle)
    success, message = planner._execute_trajectory(trajectory())
    assert not success and 'termination confirmed' in message
    assert handle.cancel_count == 1
    assert planner._motion_stop_unconfirmed
    stationary_feedback(planner, monkeypatch)
    assert not planner._motion_stop_unconfirmed


def test_cancel_acknowledgement_does_not_unblock_motion_without_result(monkeypatch):
    planner = Planner()
    handle = GoalHandle()
    planner._execute_client = ActionClient(handle)
    success, message = planner._execute_trajectory(trajectory())
    assert not success and 'stop unconfirmed' in message
    assert planner._motion_stop_unconfirmed
    planner._execute_trajectory(trajectory())
    assert len(planner._execute_client.goals) == 1
    handle.result_future.set_result(terminal())
    assert planner._motion_stop_unconfirmed
    stationary_feedback(planner, monkeypatch)
    assert not planner._motion_stop_unconfirmed


def test_late_goal_acceptance_requires_terminal_result_and_physical_stop(monkeypatch):
    planner = Planner()
    send_future = Future()
    planner._execute_client = ActionClient(send_future=send_future)
    success, message = planner._execute_trajectory(trajectory())
    assert not success and 'goal response timed out' in message
    assert planner._motion_stop_unconfirmed
    handle = GoalHandle()
    send_future.set_result(handle)
    assert handle.cancel_count == 1
    assert planner._motion_stop_unconfirmed
    handle.result_future.set_result(terminal())
    assert planner._motion_stop_unconfirmed
    stationary_feedback(planner, monkeypatch)
    assert not planner._motion_stop_unconfirmed


def test_late_goal_rejection_unblocks_motion():
    planner = Planner()
    send_future = Future()
    planner._execute_client = ActionClient(send_future=send_future)
    planner._execute_trajectory(trajectory())
    send_future.set_result(GoalHandle(accepted=False))
    assert not planner._motion_stop_unconfirmed


def test_result_transport_error_still_cancels_and_blocks_motion():
    planner = Planner()
    handle = GoalHandle()

    def fail_result_request():
        raise RuntimeError('result transport unavailable')

    handle.get_result_async = fail_result_request
    planner._execute_client = ActionClient(handle)
    success, message = planner._execute_trajectory(trajectory())
    assert not success and 'result unavailable' in message
    assert handle.cancel_count == 1
    assert planner._motion_stop_unconfirmed


def test_late_acceptance_is_canceled_even_when_result_request_fails():
    planner = Planner()
    send_future = Future()
    planner._execute_client = ActionClient(send_future=send_future)
    planner._execute_trajectory(trajectory())
    handle = GoalHandle()

    def fail_result_request():
        raise RuntimeError('result transport unavailable')

    handle.get_result_async = fail_result_request
    send_future.set_result(handle)
    assert handle.cancel_count == 1
    assert planner._motion_stop_unconfirmed


def test_canceled_action_does_not_report_success_from_payload_alone():
    planner = Planner()
    planner._execute_client = ActionClient(GoalHandle(result=terminal(status=5)))
    success, message = planner._execute_trajectory(trajectory())
    assert not success and 'did not finish successfully' in message


def stationary_feedback(planner, monkeypatch, values=None):
    clock = SimpleNamespace(now=time.monotonic() + 0.05)
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock.now)
    for value in ([0.0] * 5 if values is None else values):
        clock.now += 0.1
        planner._latest_joint_received_at = clock.now
        planner._latest_joint_positions = {'joint1': value}
        planner._update_physical_stop_guards()
    return clock


def test_controller_error_blocks_until_fresh_physical_stop(monkeypatch):
    planner = Planner()
    planner._execute_client = ActionClient(GoalHandle(result=terminal(status=6, error=-4)))
    success, message = planner._execute_trajectory(trajectory())
    assert not success and 'stationary real joint feedback' in message
    assert planner._motion_stop_unconfirmed
    stationary_feedback(planner, monkeypatch)
    assert not planner._motion_stop_unconfirmed


def test_endpoint_timeout_blocks_until_physical_stop(monkeypatch):
    planner = Planner()
    planner._wait_for_real_joint_endpoint = lambda _: (False, 'endpoint timeout')
    success, message = planner._execute_trajectory(trajectory())
    assert not success and 'endpoint timeout' in message
    assert planner._motion_stop_unconfirmed
    stationary_feedback(planner, monkeypatch)
    assert not planner._motion_stop_unconfirmed


def test_unresolved_action_does_not_clear_from_stationary_feedback(monkeypatch):
    planner = Planner()
    planner._execute_client = ActionClient(GoalHandle())
    planner._execute_trajectory(trajectory())
    stationary_feedback(planner, monkeypatch)
    assert planner._motion_stop_unconfirmed


def test_physical_stop_rejects_moving_feedback(monkeypatch):
    planner = Planner()
    planner._execute_client = ActionClient(GoalHandle(terminate_on_cancel=True))
    planner._execute_trajectory(trajectory())
    stationary_feedback(planner, monkeypatch, [0.0, 0.01, 0.02, 0.03, 0.04])
    assert planner._motion_stop_unconfirmed


def test_physical_stop_requires_hold_duration_and_distinct_samples(monkeypatch):
    planner = Planner()
    planner._execute_client = ActionClient(GoalHandle(terminate_on_cancel=True))
    planner._execute_trajectory(trajectory())
    clock = stationary_feedback(planner, monkeypatch, [0.0, 0.0, 0.0])
    assert planner._motion_stop_unconfirmed
    for _ in range(5):
        planner._update_physical_stop_guards()
    assert planner._motion_stop_unconfirmed
    clock.now += 0.2
    planner._latest_joint_received_at = clock.now
    planner._update_physical_stop_guards()
    assert not planner._motion_stop_unconfirmed


def test_physical_stop_missing_or_stale_feedback_resets_hold(monkeypatch):
    planner = Planner()
    planner._execute_client = ActionClient(GoalHandle(terminate_on_cancel=True))
    planner._execute_trajectory(trajectory())
    clock = stationary_feedback(planner, monkeypatch, [0.0, 0.0, 0.0])
    planner._latest_joint_positions = {}
    planner._update_physical_stop_guards()
    planner._latest_joint_positions = {'joint1': 0.0}
    for _ in range(3):
        clock.now += 0.1
        planner._latest_joint_received_at = clock.now
        planner._update_physical_stop_guards()
    assert planner._motion_stop_unconfirmed
    clock.now += 0.6
    planner._update_physical_stop_guards()
    for _ in range(3):
        clock.now += 0.1
        planner._latest_joint_received_at = clock.now
        planner._update_physical_stop_guards()
    assert planner._motion_stop_unconfirmed
    clock.now += 0.2
    planner._latest_joint_received_at = clock.now
    planner._update_physical_stop_guards()
    assert not planner._motion_stop_unconfirmed


def test_physical_stop_requires_every_configured_arm_joint(monkeypatch):
    planner = Planner()
    planner.values['home_joint_names'] = [f'joint{index}' for index in range(1, 7)]
    planner._latest_joint_positions = dict.fromkeys(planner.values['home_joint_names'], 0.0)
    planner._execute_client = ActionClient(GoalHandle(terminate_on_cancel=True))
    planner._execute_trajectory(trajectory())
    clock = stationary_feedback(planner, monkeypatch)
    assert planner._motion_stop_unconfirmed
    for _ in range(5):
        clock.now += 0.1
        planner._latest_joint_received_at = clock.now
        planner._latest_joint_positions = dict.fromkeys(planner.values['home_joint_names'], 0.0)
        planner._update_physical_stop_guards()
    assert not planner._motion_stop_unconfirmed


def test_planning_sends_explicit_real_start_and_preserves_gripper_state():
    planner = Planner()
    wrapped = terminal(status=4)
    wrapped.result.planned_trajectory = trajectory()
    planner._move_group_client = ActionClient(GoalHandle(result=wrapped))
    result, _ = planner._plan_constraints(Constraints())
    assert result is not None
    request = planner._move_group_client.goals[0].request
    state = request.start_state
    assert state.is_diff
    assert dict(zip(state.joint_state.name, state.joint_state.position)) == {
        'joint1': 0.0, 'center_joint': 0.012,
    }
    assert request.pipeline_id == 'ompl'
    assert request.planner_id == 'RRTConnectkConfigDefault'
    assert planner.waits == [3.0, 5.0]


def test_planning_sends_normalized_snapshot_without_modifying_raw_feedback():
    planner = Planner()
    planner._latest_joint_positions = {'joint1': 1.3014661678278e-08, 'center_joint': 0.012}
    planner._normalized_joint_positions = lambda context: {'joint1': 0.0, 'center_joint': 0.012}
    wrapped = terminal(status=4)
    wrapped.result.planned_trajectory = trajectory()
    planner._move_group_client = ActionClient(GoalHandle(result=wrapped))
    result, _ = planner._plan_constraints(Constraints())
    assert result is not None
    state = planner._move_group_client.goals[0].request.start_state.joint_state
    assert dict(zip(state.name, state.position)) == {'joint1': 0.0, 'center_joint': 0.012}
    assert planner._latest_joint_positions['joint1'] == 1.3014661678278e-08


def test_invalid_normalized_start_stops_planning_without_retryable_error():
    planner = Planner()

    def reject_start(context):
        raise ValueError(f'{context}: joint3 exceeds effective bounds by 0.001rad')

    planner._normalized_joint_positions = reject_start
    result, message = planner._plan_constraints(Constraints())
    assert result is None and 'joint3 exceeds effective bounds' in message
    assert not planner._retryable_planning_failure(message)
    assert not planner._move_group_client.goals


@pytest.mark.parametrize('coarse', [False, True])
def test_execute_checks_stage_trajectory_bounds_before_sending(coarse):
    planner = Planner()
    planner._executing_coarse_target = coarse
    reason = 'coarse joint6 margin violated' if coarse else 'home joint3 spline exceeds upper bound'
    if coarse:
        planner._trajectory_wrist_is_safe = lambda _: (False, reason)
    else:
        planner._motion_trajectory_is_safe = lambda _: (False, reason)
    success, message = planner._execute_trajectory(trajectory())
    assert not success and reason in message
    assert not planner._execute_client.goals


def test_execute_start_rejects_out_of_bounds_feedback_before_action():
    planner = Planner()

    def reject_start(context):
        raise ValueError(f'{context}: joint2 exceeds effective bounds')

    planner._normalized_joint_positions = reject_start
    success, message = planner._execute_trajectory(trajectory())
    assert not success and 'joint2 exceeds effective bounds' in message
    assert not planner._execute_client.goals


def test_start_comparison_uses_raw_feedback_after_normalized_bound_check():
    planner = Planner()
    planner._latest_joint_positions['joint1'] = 0.020001
    planner._normalized_joint_positions = lambda context: {'joint1': 0.02}
    success, message = planner._trajectory_start_is_current(trajectory(start=0.0))
    assert not success and 'planned start' in message


@pytest.mark.parametrize('boundary', ['lower', 'upper'])
def test_home_goal_tolerances_stay_inside_each_joint_bound(boundary):
    planner = Planner()
    names = [f'joint{number}' for number in range(1, 7)]
    planner.values['home_joint_names'] = names
    planner.values['home_joint_positions_rad'] = [0.0] * 6
    bounds = (0.0, 2.0) if boundary == 'lower' else (-2.0, 0.0)
    planner._arm_joint_limits = dict.fromkeys(names, bounds)
    constraints = planner._home_constraints()
    assert len(constraints.joint_constraints) == 6
    for goal in constraints.joint_constraints:
        lower, upper = planner._arm_joint_limits[goal.joint_name]
        assert goal.position - goal.tolerance_below >= lower
        assert goal.position + goal.tolerance_above <= upper
        if boundary == 'lower':
            assert goal.tolerance_below == 0.0
            assert goal.tolerance_above == pytest.approx(0.005)
        else:
            assert goal.tolerance_above == 0.0
            assert goal.tolerance_below == pytest.approx(0.005)


@pytest.mark.parametrize('joint_index', range(6))
def test_home_rejects_every_joint_outside_effective_bounds(joint_index):
    planner = Planner()
    names = [f'joint{number}' for number in range(1, 7)]
    planner.values['home_joint_names'] = names
    planner.values['home_joint_positions_rad'] = [0.0] * 6
    planner._arm_joint_limits = dict.fromkeys(names, (-2.0, 2.0))
    planner.values['home_joint_positions_rad'][joint_index] = 2.001
    with pytest.raises(ValueError, match=names[joint_index]):
        planner._home_constraints()


def test_planning_rejects_incomplete_start_without_sending():
    planner = Planner()
    planner._latest_joint_positions = {'center_joint': 0.012}
    result, message = planner._plan_constraints(Constraints())
    assert result is None and 'fresh complete' in message
    assert not planner._move_group_client.goals


def test_planning_timeout_cancels_and_remains_blocked_until_terminal_result():
    planner = Planner()
    handle = GoalHandle()
    planner._move_group_client = ActionClient(handle)
    result, message = planner._plan_constraints(Constraints())
    assert result is None and 'stop unconfirmed' in message
    assert handle.cancel_count == 1
    assert planner._motion_stop_unconfirmed
    handle.result_future.set_result(terminal())
    assert not planner._motion_stop_unconfirmed


def test_planning_obeys_overall_deadline():
    planner = Planner()
    planner._planning_deadline = time.monotonic() - 0.1
    result, message = planner._plan_constraints(Constraints())
    assert result is None and 'budget exhausted' in message
    assert not planner._move_group_client.goals


def test_gripper_timeout_cancels_and_blocks_when_stop_is_unconfirmed():
    planner = Planner()
    handle = GoalHandle()
    planner._gripper_client = ActionClient(handle)
    success, message = planner._close_gripper()
    assert not success and 'stop unconfirmed' in message
    assert handle.cancel_count == 1
    assert planner._motion_stop_unconfirmed


def endpoint_check(monkeypatch, samples):
    planner = Planner()
    clock = SimpleNamespace(now=100.0, index=0)
    planner._latest_joint_received_at = clock.now
    planner._latest_joint_positions = {'joint1': 0.1}

    def sleep(seconds):
        clock.now += seconds
        if clock.index < len(samples):
            value = samples[clock.index]
            clock.index += 1
            if value is not None:
                planner._latest_joint_received_at = clock.now
                planner._latest_joint_positions = {'joint1': value}

    monkeypatch.setattr(module.time, 'monotonic', lambda: clock.now)
    monkeypatch.setattr(module.time, 'sleep', sleep)
    return ButtonApproachPlanner._wait_for_real_joint_endpoint(planner, trajectory())


def test_endpoint_requires_distinct_feedback_receipts(monkeypatch):
    success, message = endpoint_check(monkeypatch, [0.1])
    assert not success and 'stable_samples=1/3' in message


def test_endpoint_accepts_three_fresh_stationary_samples(monkeypatch):
    success, message = endpoint_check(monkeypatch, [0.1, 0.1, 0.1])
    assert success and 'stable_samples=3' in message


def test_endpoint_rejects_motion_even_inside_endpoint_tolerance(monkeypatch):
    success, _ = endpoint_check(monkeypatch, [0.092, 0.096, 0.100, 0.104, 0.108])
    assert not success
