"""Check actual published Servo commands at phase and safety boundaries."""

import math

import numpy as np
import pytest

from piper_elevator_app.button_visual_servo import ButtonVisualServo
from piper_elevator_app.motion_core import matrix_to_quaternion
from piper_elevator_app.motion_core import quaternion_to_matrix
from test_servo_normal_consistency import CAMERA_ORIENTATION
from test_servo_normal_consistency import TrackingHarness
from test_servo_normal_consistency import assert_held
from test_servo_normal_consistency import frame


class OrientationDriftHarness(TrackingHarness):
    def __init__(self, *args, drift_axis='tilt', **kwargs):
        self.drift_axis = drift_axis
        super().__init__(*args, **kwargs)

    def _current_servo_pose(self):
        angle = math.radians(10.0 if self.index >= 9 else 0.0)
        cosine, sine = math.cos(angle), math.sin(angle)
        if self.drift_axis == 'tilt':
            rotation = np.array([
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ])
        else:
            rotation = np.array([
                [1.0, 0.0, 0.0],
                [0.0, cosine, -sine],
                [0.0, sine, cosine],
            ])
        orientation = matrix_to_quaternion(
            rotation @ quaternion_to_matrix(CAMERA_ORIENTATION),
        )
        return self.position.copy(), orientation.copy(), orientation.copy()


@pytest.mark.parametrize('drift_axis', ['tilt', 'roll'])
def test_orientation_guard_stops_smoothed_inward_speed_immediately(
    monkeypatch, drift_axis,
):
    servo = OrientationDriftHarness(
        [frame(sequence) for sequence in range(1, 13)], monkeypatch,
        at_target=False, drift_axis=drift_axis,
    )
    servo.run()
    assert any(index == 8 and linear[0] > 0.02
               for index, linear, _ in servo.commands)
    for index, linear, _ in servo.commands:
        if index >= 9:
            assert abs(linear[0]) < 1e-12
    assert abs(servo._last_linear_command[0]) < 1e-12


def test_orientation_guard_preserves_retreat(monkeypatch):
    servo = OrientationDriftHarness(
        [frame(sequence) for sequence in range(1, 13)], monkeypatch,
        at_target=False,
    )
    servo.position[0] = 0.43
    servo.run()
    assert all(linear[0] < 0.0 for index, linear, _ in servo.commands
               if index >= 9)


def test_reorientation_drops_previous_reacquisition_translation(monkeypatch):
    servo = TrackingHarness(
        [frame(1), frame(2)] + [frame(2, visible=False)] * 40
        + [frame(3, 10.0), frame(4, 10.0)],
        monkeypatch, at_target=False,
    )
    servo.run()
    assert any(index == 41 and np.linalg.norm(linear) > 0.001
               for index, linear, _ in servo.commands)
    for index, linear, _ in servo.commands:
        if index >= 42:
            np.testing.assert_array_equal(linear, np.zeros(3))
    np.testing.assert_array_equal(servo._last_linear_command, np.zeros(3))


def test_zero_blind_speed_clears_smoothed_commands(monkeypatch):
    servo = TrackingHarness(
        [frame(sequence) for sequence in range(1, 10)]
        + [frame(9, visible=False)] * 3,
        monkeypatch, at_target=False, vision_loss_speed_scale=0.0,
    )
    servo.run()
    assert any(index == 8 and linear[0] > 0.02
               for index, linear, _ in servo.commands)
    for index, linear, angular in servo.commands:
        if index >= 9:
            np.testing.assert_array_equal(linear, np.zeros(3))
            np.testing.assert_array_equal(angular, np.zeros(3))


class CreepingHarness(TrackingHarness):
    """Blind frames while the tip creeps toward the locked standoff."""

    def __init__(self, *args, step=0.0002, **kwargs):
        self.step = step
        super().__init__(*args, **kwargs)

    def advance(self, period):
        previous = self.index
        result = super().advance(period)
        if self.index > previous and previous >= 8:
            self.position = self.position + np.array([self.step, 0.0, 0.0])
        return result


def test_blind_approach_keeps_constant_speed_until_the_standoff(monkeypatch):
    """A proportional blind command decays to nothing at the last millimetres."""
    servo = CreepingHarness(
        [frame(sequence) for sequence in range(1, 9)]
        + [frame(sequence, visible=False) for sequence in range(9, 90)],
        monkeypatch, at_target=False,
    )
    servo.position[0] = 0.362
    servo.run()
    blind = [
        linear[0] for index, linear, _ in servo.commands if index >= 17
    ]
    assert blind
    # 0.020 m/s * vision_loss_speed_scale 0.5, held until the ramp takes over.
    assert min(blind) >= 0.008
    assert max(blind) <= 0.012


def test_blind_continuation_holds_once_it_stops_progressing(monkeypatch):
    servo = TrackingHarness(
        [frame(sequence) for sequence in range(1, 9)]
        + [frame(sequence, visible=False) for sequence in range(9, 500)],
        monkeypatch, at_target=False,
    )
    locked, source, message = servo.run(timeout=60.0)
    assert locked is None
    assert 'RGB-D loss exceeded bounded Servo continuation' in message
    moving = [
        index for index, linear, _ in servo.commands
        if index >= 9 and abs(linear[0]) > 0.001
    ]
    assert moving
    # vision_loss_stall_seconds = 3.0 s at 50 Hz, then hold instead of pushing.
    assert max(moving) <= 9 + 165
    assert_held(servo, [9 + 200])


def test_zero_vector_limit_is_a_stop():
    command = np.array([0.03, -0.02, 0.01])
    np.testing.assert_array_equal(
        ButtonVisualServo._limit_vector(command, 0.0), np.zeros(3),
    )


@pytest.mark.parametrize('maximum', [-0.1, math.nan, math.inf, -math.inf])
def test_invalid_vector_limits_reject_commands(maximum):
    with pytest.raises(ValueError, match='norm limit'):
        ButtonVisualServo._limit_vector(np.array([0.03, 0.0, 0.0]), maximum)


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
def test_nonfinite_commands_are_rejected(value):
    with pytest.raises(ValueError, match='Command vector'):
        ButtonVisualServo._limit_vector(np.array([value, 0.0, 0.0]), 0.08)
