"""Exercise final-approach normal monitoring without DDS or robot motion."""

import math
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from piper_elevator_app import button_visual_servo
from piper_elevator_app.button_visual_servo import ButtonVisualServo
from piper_elevator_app.motion_core import matrix_to_quaternion


CONFIG = yaml.safe_load(
    (Path(__file__).parents[1] / 'config' / 'button_visual_servo.yaml').read_text()
)['button_visual_servo']['ros__parameters']
BUTTON = np.array([0.4, 0.0, 0.3])
CAMERA_ORIENTATION = matrix_to_quaternion(np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
]))


def frame(sequence, angle=0.0, *, visible=True):
    return SimpleNamespace(sequence=sequence, angle=angle, visible=visible)


def aligned_frames():
    return [frame(sequence) for sequence in range(1, 6)]


class TrackingHarness:
    _track_visually = ButtonVisualServo._track_visually
    _servo_target = ButtonVisualServo._servo_target
    _controlled_angular_error = ButtonVisualServo._controlled_angular_error
    _level_roll_error = ButtonVisualServo._level_roll_error
    _roll_within_tolerance = ButtonVisualServo._roll_within_tolerance
    _limit_vector = staticmethod(ButtonVisualServo._limit_vector)
    _limit_level_roll_speed = ButtonVisualServo._limit_level_roll_speed
    _roll_status = staticmethod(ButtonVisualServo._roll_status)
    _linear_speed_multiplier = ButtonVisualServo._linear_speed_multiplier
    _blind_approach_speed = ButtonVisualServo._blind_approach_speed
    _vision_loss_continuation_seconds = (
        ButtonVisualServo._vision_loss_continuation_seconds
    )
    _level_roll_tolerance = ButtonVisualServo._level_roll_tolerance
    _wrist_limit_guard_margin = ButtonVisualServo._wrist_limit_guard_margin
    _wrist_limit_guard_freezes_roll = (
        ButtonVisualServo._wrist_limit_guard_freezes_roll
    )
    _wrist_limit_guard_status_text = (
        ButtonVisualServo._wrist_limit_guard_status_text
    )
    _smooth_servo_command = ButtonVisualServo._smooth_servo_command
    _publish_zero_twist = ButtonVisualServo._publish_zero_twist

    def __init__(self, frames, monkeypatch, *, at_target=True, **parameters):
        self.frames = frames
        self.parameters = dict(CONFIG, **parameters)
        self._condition = threading.Condition()
        self._level_reference_axis = np.array([0.0, 0.0, 1.0])
        self._wrist_guard_position = None
        self._wrist_guard_state_received_at = 0.0
        self._wrist_guard_reported = False
        self.position = np.array([0.37 if at_target else 0.34, 0.0, 0.3])
        self.now = 100.0
        self.index = 0
        self.stopped = False
        self.source_times = {}
        self.commands = []
        self.statuses = []
        self.targets = []
        self._stop_event = SimpleNamespace(
            is_set=lambda: self.stopped,
            wait=self.advance,
        )
        self._target_publisher = SimpleNamespace(publish=self.targets.append)
        monkeypatch.setattr(button_visual_servo.time, 'monotonic', lambda: self.now)
        self.install_frame()

    def get_parameter(self, name):
        return SimpleNamespace(value=self.parameters[name])

    def install_frame(self):
        current = self.frames[self.index]
        if not current.visible:
            self._observation = None
            return
        angle = math.radians(current.angle)
        normal = np.array([math.cos(angle), math.sin(angle), 0.0])
        self._observation = (
            BUTTON.copy(), normal,
            self.source_times.setdefault(current.sequence, self.now),
            current.sequence,
        )

    def advance(self, period):
        self.now += period
        self.index += 1
        if self.index >= len(self.frames):
            self.stopped = True
            return True
        self.install_frame()
        return False

    def _standoff_distance(self):
        return self.parameters['standoff_distance_m']

    def _servo_safety_failure(self):
        return None

    def _set_hardware_servo_gate(self, enabled):
        return True, ''

    def _current_servo_pose(self):
        return (
            self.position.copy(), CAMERA_ORIENTATION.copy(),
            CAMERA_ORIENTATION.copy(),
        )

    def _make_pose(self, position, orientation):
        return position.copy(), orientation.copy()

    def _publish_status(self, message):
        self.statuses.append((self.index, message))

    def _publish_twist(self, linear, angular):
        self.commands.append((self.index, linear.copy(), angular.copy()))

    def run(self, timeout=5.0):
        return self._track_visually(self.now + timeout)


def assert_held(servo, indices):
    for index in indices:
        commands = [item for item in servo.commands if item[0] == index]
        assert commands, f'No hold command on frame {index}'
        for _, linear, angular in commands:
            np.testing.assert_array_equal(linear, np.zeros(3))
            np.testing.assert_array_equal(angular, np.zeros(3))


@pytest.mark.parametrize('angle', [0.0, 9.0])
def test_consistent_fresh_normals_finish_with_locked_orientation(monkeypatch, angle):
    servo = TrackingHarness(aligned_frames() + [frame(6, angle)], monkeypatch)
    locked, source, message = servo.run()
    assert source == 'visual_target', message
    np.testing.assert_allclose(locked[1], [1.0, 0.0, 0.0])
    for _, orientation in servo.targets:
        np.testing.assert_allclose(orientation, CAMERA_ORIENTATION, atol=1e-12)


def test_three_new_conflicting_normals_fail_instead_of_completing(monkeypatch):
    servo = TrackingHarness(
        aligned_frames() + [frame(sequence, 30.0) for sequence in range(6, 9)],
        monkeypatch,
    )
    locked, source, message = servo.run()
    assert locked is None
    assert source == ''
    assert 'fresh surface normal disagrees' in message
    assert 'observations=3/3' in message
    assert_held(servo, [5, 6, 7])


def test_single_conflict_resets_completion_until_new_normal_frames(monkeypatch):
    servo = TrackingHarness(
        aligned_frames() + [frame(6, 30.0), frame(7), frame(8)], monkeypatch,
    )
    locked, source, message = servo.run()
    assert source == 'visual_target', message
    assert locked is not None
    assert servo.index == 7
    assert_held(servo, [5])


def test_conflict_blocks_axial_motion_then_allows_consistent_recovery(monkeypatch):
    servo = TrackingHarness(
        aligned_frames() + [frame(6, 30.0), frame(7)],
        monkeypatch, at_target=False,
    )
    locked, _, _ = servo.run()
    assert locked is None
    assert_held(servo, [5])
    assert any(index == 6 and linear[0] > 0.0
               for index, linear, _ in servo.commands)


def test_wrist_limit_guard_relaxes_roll_only_near_the_joint_bound(monkeypatch):
    servo = TrackingHarness(aligned_frames(), monkeypatch)

    def set_wrist(position):
        servo._wrist_guard_position = position
        # The harness patches the module clock, so use its monotonic value.
        servo._wrist_guard_state_received_at = servo.now

    # Far from the limit: the configured 3 deg acceptance is unchanged.
    set_wrist(-0.60)
    assert servo._wrist_limit_guard_margin() == pytest.approx(0.6217, abs=1e-3)
    assert servo._level_roll_tolerance() == pytest.approx(0.05235)
    assert servo._roll_within_tolerance(math.radians(5.0)) is False
    assert not servo._wrist_limit_guard_freezes_roll()

    # Inside the relax margin: 6 deg is accepted so leveling stops early.
    set_wrist(-1.05)
    assert servo._level_roll_tolerance() == pytest.approx(0.10472)
    assert servo._roll_within_tolerance(math.radians(5.0)) is True
    assert not servo._wrist_limit_guard_freezes_roll()

    # Inside the hold margin: accept the current roll and stop pushing.
    set_wrist(-1.18)
    assert servo._wrist_limit_guard_freezes_roll() is True
    assert servo._roll_within_tolerance(math.radians(30.0)) is True
    command = servo._limit_level_roll_speed(
        np.array([0.10, 0.20, 0.30]),
        np.array([0.0, 0.0, 1.0]),
    )
    np.testing.assert_allclose(command, [0.10, 0.20, 0.0])

    # Stale joint feedback must fall back to the strict 3 deg acceptance.
    set_wrist(-1.05)
    servo._wrist_guard_state_received_at = servo.now - 5.0
    assert servo._wrist_limit_guard_margin() is None
    assert servo._level_roll_tolerance() == pytest.approx(0.05235)
    assert servo._roll_within_tolerance(math.radians(5.0)) is False


def test_repeated_conflicting_sequence_does_not_count_as_new_evidence(monkeypatch):
    servo = TrackingHarness(
        aligned_frames() + [frame(6, 30.0)] * 5 + [frame(7), frame(8)],
        monkeypatch,
    )
    locked, source, message = servo.run()
    assert source == 'visual_target', message
    assert locked is not None
    assert servo.index == 11
    assert_held(servo, range(5, 10))
    conflicts = [message for _, message in servo.statuses
                 if 'FINAL_APPROACH_NORMAL_CONFLICT' in message]
    assert len(conflicts) == 5
    assert all('observations=1/3' in message for message in conflicts)


def test_vision_loss_cannot_clear_conflict_or_complete_from_locked_target(monkeypatch):
    servo = TrackingHarness(
        aligned_frames() + [frame(6, 30.0)]
        + [frame(6, visible=False)] * 6 + [frame(7), frame(8)],
        monkeypatch,
    )
    locked, source, message = servo.run()
    assert source == 'visual_target', message
    assert locked is not None
    assert servo.index == 13
    assert_held(servo, range(5, 12))


def test_conflict_hold_remains_bounded_by_tracking_deadline(monkeypatch):
    servo = TrackingHarness(
        aligned_frames() + [frame(6, 30.0)]
        + [frame(6, visible=False)] * 20,
        monkeypatch,
    )
    locked, source, message = servo.run(timeout=0.25)
    assert locked is None
    assert source == ''
    assert message == 'visual tracking timeout'
    assert_held(servo, range(5, servo.index))


@pytest.mark.parametrize('parameters', [
    {'maximum_normal_change_rad': 0.0},
    {'maximum_normal_change_rad': math.nan},
    {'maximum_normal_change_rad': math.pi},
    {'required_conflicting_normal_observations': 0},
])
def test_invalid_normal_monitor_limits_reject_tracking(monkeypatch, parameters):
    servo = TrackingHarness(aligned_frames(), monkeypatch, **parameters)
    locked, source, message = servo.run()
    assert locked is None
    assert source == ''
    assert message == 'invalid final-approach normal consistency limits'
    assert not servo.commands
