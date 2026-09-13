"""Shared coarse-to-Servo capture geometry and pre-motion rejection."""

import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from piper_elevator_app.motion_core import check_servo_capture
from piper_elevator_app.motion_core import matrix_to_quaternion
from piper_elevator_app.button_visual_servo import ButtonVisualServo


def camera_orientation(tilt=0.0, roll=0.0):
    theta, phi = np.radians([tilt, roll])
    level = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    tilt_rotation = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(theta), -math.sin(theta)],
        [0.0, math.sin(theta), math.cos(theta)],
    ])
    roll_rotation = np.array([
        [math.cos(phi), -math.sin(phi), 0.0],
        [math.sin(phi), math.cos(phi), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return matrix_to_quaternion(level @ tilt_rotation @ roll_rotation)


def capture(**overrides):
    parameters = {
        'button_position': [0.4, 0.0, 0.3],
        'surface_normal': [1.0, 0.0, 0.0],
        'tool_position': [0.26, 0.0, 0.3],
        'camera_orientation': camera_orientation(),
        'maximum_tilt_rad': math.radians(15.0),
        'maximum_roll_rad': math.radians(15.0),
        'minimum_standoff_m': 0.08,
        'target_standoff_m': 0.03,
        'maximum_start_error_m': 0.20,
        **overrides,
    }
    return check_servo_capture(**parameters)


@pytest.mark.parametrize('tilt', [0.0, 7.0, 11.5, 15.0])
def test_captures_orientation_that_servo_can_correct(tilt):
    safe, detail = capture(camera_orientation=camera_orientation(tilt))
    assert safe, detail
    assert f'tilt={tilt:.2f}deg' in detail
    assert 'standoff=140.0mm (minimum=80.0mm)' in detail


@pytest.mark.parametrize('tilt', [15.01, 20.0, 180.0])
def test_rejects_camera_outside_capture_tilt(tilt):
    safe, detail = capture(camera_orientation=camera_orientation(tilt))
    assert not safe
    assert 'Camera tilt outside Servo capture range' in detail
    assert f'tilt={tilt:.2f}deg (maximum=15.00deg)' in detail


@pytest.mark.parametrize('roll', [-15.0, 15.0])
def test_roll_boundary_is_inclusive(roll):
    safe, detail = capture(camera_orientation=camera_orientation(roll=roll))
    assert safe, detail


@pytest.mark.parametrize('roll', [-15.01, 20.0])
def test_rejects_roll_outside_capture_range(roll):
    safe, detail = capture(camera_orientation=camera_orientation(roll=roll))
    assert not safe
    assert 'Camera roll outside Servo capture range' in detail


@pytest.mark.parametrize('standoff', [0.03, 0.07999, -0.03])
def test_rejects_insufficient_space_to_correct_orientation(standoff):
    safe, detail = capture(tool_position=[0.4 - standoff, 0.0, 0.3])
    assert not safe
    assert 'TCP standoff below capture minimum' in detail
    assert 'minimum=80.0mm' in detail


def test_minimum_capture_standoff_is_inclusive():
    safe, detail = capture(tool_position=[0.32, 0.0, 0.3])
    assert safe, detail


@pytest.mark.parametrize('target_error,safe_expected', [(0.20, True), (0.20001, False)])
def test_target_error_limit_is_inclusive(target_error, safe_expected):
    safe, detail = capture(tool_position=[0.37 - target_error, 0.0, 0.3])
    assert safe is safe_expected, detail
    assert 'maximum=200.0mm' in detail


def test_capture_distance_includes_lateral_error():
    safe, detail = capture(tool_position=[0.26, 0.20, 0.3])
    assert not safe
    assert 'Target outside Servo capture distance' in detail


@pytest.mark.parametrize('overrides', [
    {'button_position': [math.nan, 0.0, 0.3]},
    {'button_position': [0.4, 0.0]},
    {'tool_position': [0.26, math.inf, 0.3]},
    {'surface_normal': [0.0, 0.0, 0.0]},
    {'surface_normal': [1.0, math.nan, 0.0]},
    {'camera_orientation': [0.0, 0.0, 0.0, 0.0]},
    {'camera_orientation': [0.0, 0.0, 1.0]},
    {'camera_orientation': [math.nan, 0.0, 0.0, 1.0]},
    {'level_reference_axis': [0.0, 0.0, 0.0]},
    {'level_reference_axis': [1.0, 0.0, 0.0]},
    {'maximum_tilt_rad': math.nan},
    {'maximum_tilt_rad': -0.01},
    {'maximum_roll_rad': math.inf},
    {'minimum_standoff_m': 0.03},
    {'target_standoff_m': 0.0},
    {'maximum_start_error_m': 0.0},
])
def test_invalid_geometry_or_limits_fail_closed(overrides):
    safe, detail = capture(**overrides)
    assert not safe
    assert detail.startswith('Invalid Servo capture geometry:')


def test_capture_does_not_mutate_input_arrays():
    normal = np.array([2.0, 0.0, 0.0])
    vertical = np.array([0.0, 0.0, 2.0])
    safe, detail = capture(surface_normal=normal, level_reference_axis=vertical)
    assert safe, detail
    assert normal.tolist() == [2.0, 0.0, 0.0]
    assert vertical.tolist() == [0.0, 0.0, 2.0]


class ServoStartHarness:
    _cleanup_servo_session = ButtonVisualServo._cleanup_servo_session
    _signal_alignment_finished = (
        ButtonVisualServo._signal_alignment_finished
    )

    def __init__(self, tilt=0.0, standoff=0.14):
        self.tilt = tilt
        self.standoff = standoff
        self.statuses = []
        self.gate_calls = []
        self.resume_calls = 0
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._running = True
        self._alignment_finished = threading.Event()
        self._alignment_result = (False, 'visual servo has not run')
        self._owns_servo = False
        self._workspace_min = [-0.65, -0.65, 0.02]
        self._workspace_max = [0.65, 0.65, 0.75]

    def get_parameter(self, name):
        return SimpleNamespace(value={
            'observation_timeout_seconds': 8.25,
            'handover_maximum_camera_tilt_rad': math.radians(15.0),
            'handover_maximum_camera_roll_rad': math.radians(15.0),
            'handover_minimum_standoff_m': 0.08,
            'maximum_start_error_m': 0.20,
            'level_reference_axis': [0.0, 0.0, 1.0],
        }[name])

    def _wait_for_observation(self, *args):
        return np.array([0.4, 0.0, 0.3]), np.array([1.0, 0.0, 0.0]), 1.0, 1

    def _current_servo_pose(self):
        return (
            np.array([0.4 - self.standoff, 0.0, 0.3]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            camera_orientation(self.tilt),
        )

    def _standoff_distance(self):
        return 0.03

    def _servo_target(self, button, normal, tool, camera, distance):
        return button - distance * normal, tool

    def _resume_moveit_servo(self):
        self.resume_calls += 1
        return False, 'test ends before starting motion'

    def _set_hardware_servo_gate(self, enabled, **kwargs):
        self.gate_calls.append(enabled)
        return True, ''

    def _publish_status(self, status):
        self.statuses.append(status)

    def _publish_zero_twist(self):
        pass

    def _pause_moveit_servo(self, **kwargs):
        return True, ''

    def _publish_completion(self, completed):
        self.completed = completed

    def get_logger(self):
        return SimpleNamespace(error=lambda text: None)


@pytest.mark.parametrize('tilt,standoff', [(20.0, 0.14), (0.0, 0.03)])
def test_unsafe_handover_cannot_resume_servo_or_enable_hardware(tilt, standoff):
    from piper_elevator_app.button_visual_servo import ButtonVisualServo

    servo = ServoStartHarness(tilt, standoff)
    ButtonVisualServo._run_servo(servo)
    assert servo.resume_calls == 0
    assert True not in servo.gate_calls
    assert not servo.completed
    assert servo.statuses[-1].startswith('FAILED: unsafe Servo handover;')


def test_observed_eleven_point_five_degree_handover_can_reach_servo_start():
    from piper_elevator_app.button_visual_servo import ButtonVisualServo

    servo = ServoStartHarness(tilt=11.5)
    ButtonVisualServo._run_servo(servo)
    assert servo.resume_calls == 1
    assert True not in servo.gate_calls
    assert servo.statuses[-1] == 'FAILED: test ends before starting motion'
