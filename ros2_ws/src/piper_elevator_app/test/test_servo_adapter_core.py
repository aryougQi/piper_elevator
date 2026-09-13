"""Unit tests for bounded simulated Servo tracking compensation."""

import math

import pytest

from piper_elevator_app.servo_adapter_core import compensated_positions


def compensate(desired, current=0.0, **overrides):
    arguments = {
        'joint_names': ['joint1'],
        'desired_positions': [desired],
        'current_by_name': {'joint1': current},
        'position_gain': 2.0,
        'maximum_lead_rad': 0.08,
        'lower_by_name': {'joint1': -1.0},
        'upper_by_name': {'joint1': 1.0},
    }
    arguments.update(overrides)
    return compensated_positions(**arguments)[0]


def test_small_position_error_is_amplified_relative_to_feedback():
    assert compensate(0.12, current=0.10) == pytest.approx(0.14)
    assert compensate(0.08, current=0.10) == pytest.approx(0.06)


def test_position_lead_and_joint_limits_are_bounded():
    assert compensate(0.8, current=0.1) == pytest.approx(0.18)
    assert compensate(
        1.0,
        current=0.98,
        upper_by_name={'joint1': 1.0},
        position_gain=4.0,
    ) == pytest.approx(1.0)


@pytest.mark.parametrize(
    'overrides',
    [
        {'position_gain': 0.9},
        {'maximum_lead_rad': 0.0},
        {'desired_positions': []},
        {'desired_positions': [math.nan]},
        {'current_by_name': {}},
        {'lower_by_name': {'joint1': 1.0}},
    ],
)
def test_invalid_or_unsafe_inputs_are_rejected(overrides):
    with pytest.raises(ValueError):
        compensate(0.1, **overrides)
