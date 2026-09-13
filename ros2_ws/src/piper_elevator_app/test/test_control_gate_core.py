"""Regression tests for the real-arm command gate policy."""

from piper_elevator_app.control_gate_core import ControlGatePolicy
from piper_elevator_app.control_gate_core import desired_control_mode
from piper_elevator_app.control_gate_core import (
    joint_positions_within_tolerance,
)


def make_policy():
    """Build the production policy limits used by the real launch."""
    return ControlGatePolicy(
        maximum_trajectory_seconds=45.0,
        servo_heartbeat_seconds=0.75,
        trajectory_settle_seconds=25.0,
    )


def test_silent_active_action_does_not_close_after_one_second():
    """An action status is durable state and need not be a 1 Hz heartbeat."""
    policy = make_policy()
    policy.update_trajectory(True, now=10.0)

    assert policy.desired_open(now=10.0) == (True, False)
    assert policy.desired_open(now=11.1) == (True, False)
    assert policy.desired_open(now=40.0) == (True, False)


def test_explicit_terminal_action_status_closes_gate():
    """A terminal action status closes the trajectory gate immediately."""
    policy = make_policy()
    policy.update_trajectory(True, now=10.0)
    policy.update_trajectory(False, now=14.0)

    assert policy.desired_open(now=14.0) == (False, False)


def test_hard_timeout_closes_and_latches_until_terminal_status():
    """A crashed action cannot leave or repeatedly reopen the gate."""
    policy = make_policy()
    policy.update_trajectory(True, now=10.0)

    assert policy.desired_open(now=55.1) == (False, True)
    assert policy.desired_open(now=55.2) == (False, False)
    policy.update_trajectory(True, now=56.0)
    assert policy.desired_open(now=56.0) == (False, False)

    policy.update_trajectory(False, now=57.0)
    assert policy.desired_open(now=57.0) == (False, False)
    policy.update_trajectory(True, now=58.0)
    assert policy.desired_open(now=58.0) == (True, False)


def test_servo_authorization_requires_fresh_heartbeat():
    """Servo retains its intentionally short crash-detection heartbeat."""
    policy = make_policy()
    policy.update_servo(True, now=10.0)

    assert policy.desired_open(now=10.70) == (True, False)
    assert policy.desired_open(now=10.76) == (False, False)


def test_control_modes_are_explicit_and_mutually_exclusive():
    policy = make_policy()
    assert desired_control_mode(policy, now=1.0) == (None, False)

    policy.update_trajectory(True, now=2.0)
    assert desired_control_mode(policy, now=2.1) == ('trajectory', False)

    policy.update_trajectory(False, now=3.0)
    policy.update_servo(True, now=3.1)
    assert desired_control_mode(policy, now=3.2) == ('servo', False)

    policy.update_servo(False, now=3.3)
    assert desired_control_mode(policy, now=3.4) == (None, False)


def test_trajectory_timeout_cannot_be_bypassed_by_servo_heartbeat():
    """A timed-out trajectory latches the whole command gate closed."""
    policy = make_policy()
    policy.update_trajectory(True, now=10.0)
    policy.update_servo(True, now=55.05)

    assert policy.desired_open(now=55.1) == (False, True)


def test_terminal_action_stays_open_until_real_feedback_settles():
    policy = make_policy()
    policy.update_trajectory(True, now=10.0)
    policy.update_trajectory(False, now=14.0, needs_settling=True)

    assert policy.desired_open(now=20.0) == (True, False)
    policy.complete_trajectory_settling()
    assert policy.desired_open(now=20.0) == (False, False)


def test_post_action_settling_has_a_hard_timeout():
    policy = make_policy()
    policy.update_trajectory(True, now=10.0)
    policy.update_trajectory(False, now=14.0, needs_settling=True)

    assert policy.desired_open(now=39.1) == (False, True)


def test_joint_tracking_requires_every_joint_inside_tolerance():
    target = {'joint1': 0.5, 'joint5': -0.6}
    assert joint_positions_within_tolerance(
        target,
        {'joint1': 0.51, 'joint5': -0.59},
        ['joint1', 'joint5'],
        0.025,
    )
    assert not joint_positions_within_tolerance(
        target,
        {'joint1': 0.51, 'joint5': -0.9},
        ['joint1', 'joint5'],
        0.025,
    )
