"""Distinguish current RGB-D blockers from historical observation counters."""

from collections import deque
import math
from types import SimpleNamespace

from geometry_msgs.msg import PoseStamped
import pytest
from rclpy.time import Time

from piper_elevator_app import button_approach_planner as module
from piper_elevator_app.motion_core import matrix_to_quaternion
from test_approach_planning import LEVEL_CAMERA, PlannerHarness, make_transform


class DiagnosticsHarness(PlannerHarness):
    def __init__(self, monkeypatch):
        self.clock = 100.0
        monkeypatch.setattr(module.time, 'monotonic', lambda: self.clock)
        super().__init__()
        self.values.update(
            observation_stable_samples=40,
            observation_minimum_samples=8,
            observation_window_max_seconds=6.0,
            planning_observation_wait_seconds=0.06,
        )
        self._observations = deque(maxlen=40)
        self._lookup_message_transform = lambda *args: make_transform(
            [0.3, 0.0, 0.3], matrix_to_quaternion(LEVEL_CAMERA),
        )
        self._tf_buffer = SimpleNamespace(
            lookup_transform=lambda *args, **kwargs: make_transform(
                [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
            ),
        )

    def advance(self, seconds):
        self.clock += seconds
        self.now_ns += int(round(seconds * 1e9))

    def publish(self, angle=0.0, *, camera_frame=None):
        message = PoseStamped()
        message.header.frame_id = camera_frame or self.values['camera_frame']
        message.header.stamp = Time(nanoseconds=self.now_ns).to_msg()
        message.pose.position.z = 0.2
        message.pose.orientation.y = math.sin(math.radians(angle) / 2.0)
        message.pose.orientation.w = math.cos(math.radians(angle) / 2.0)
        self._surface_pose_callback(message)


def test_valid_frames_resolve_current_rejection_but_keep_aged_history(monkeypatch):
    planner = DiagnosticsHarness(monkeypatch)
    planner.publish(camera_frame='wrong_camera')
    rejected = planner._observation_diagnostics()
    assert rejected['blocking_reason'] == 'surface_input_rejected'
    assert rejected['current_rejection'] == 'frame_mismatch'
    assert rejected['last_rejection_age_seconds'] == 0.0

    planner.advance(0.1)
    planner.publish()
    recovered = planner._observation_diagnostics()
    assert recovered['blocking_reason'] == 'acquiring_window'
    assert recovered['current_rejection'] == ''
    assert recovered['last_rejection'] == 'frame_mismatch'
    assert recovered['last_rejection_age_seconds'] == pytest.approx(0.1)


def test_timeout_reports_current_normal_window_and_only_wait_counts(monkeypatch):
    planner = DiagnosticsHarness(monkeypatch)
    planner._reject_surface_observation(
        'geometry_or_tf', 'age=0.124s limit=0.030s',
    )
    planner._observation_counts['stable_windows'] = 6544
    for index in range(40):
        planner.advance(0.1)
        planner.publish(25.0 if index % 2 else -25.0)
    assert planner._latest_observation is None
    assert 'normal uncertainty=' in planner._observation_detail
    assert 'limit=5.0deg' in planner._observation_detail
    next_index = 40

    def receive(seconds):
        nonlocal next_index
        planner.advance(seconds)
        if round(seconds * 1e9) == 0:
            return
        planner.publish(25.0 if next_index % 2 else -25.0)
        next_index += 1

    monkeypatch.setattr(module.time, 'sleep', receive)
    with pytest.raises(ValueError) as raised:
        planner._wait_for_planning_observation()
    message = str(raised.value)
    assert 'reason=unstable_window' in message
    assert 'window=40/40' in message
    assert 'current_rejection=none' in message
    assert 'wait_counts=' in message
    assert '6544' not in message
    assert 'age=0.124s' not in message

    diagnostic = planner._observation_diagnostics()
    assert diagnostic['counts_scope'] == 'node_lifetime'
    assert diagnostic['counts']['stable_windows'] == 6544
    assert diagnostic['last_rejection_age_seconds'] > 4.0
    waited = diagnostic['last_observation_wait']
    assert waited['outcome'] == 'timed_out'
    assert waited['elapsed_seconds'] == pytest.approx(0.06)
    assert waited['counts']['received'] > 0
    assert waited['counts']['accepted_frames'] == waited['counts']['received']
    assert waited['counts']['unstable_windows'] == waited['counts']['received']
    assert waited['counts'].get('stable_windows', 0) == 0
    assert waited['counts'].get('rejected_geometry_or_tf', 0) == 0


def test_old_stable_window_with_no_new_images_reports_input_staleness(monkeypatch):
    planner = DiagnosticsHarness(monkeypatch)
    for _ in range(8):
        planner.advance(0.1)
        planner.publish()
    assert planner._observation_diagnostics()['ready']
    planner.advance(0.6)
    monkeypatch.setattr(module.time, 'sleep', planner.advance)
    with pytest.raises(ValueError, match='reason=surface_input_stale'):
        planner._wait_for_planning_observation()
    waited = planner._observation_diagnostics()['last_observation_wait']
    assert waited['counts'] == {}
    assert waited['outcome'] == 'timed_out'


def test_success_records_this_wait_separately_from_old_windows(monkeypatch):
    planner = DiagnosticsHarness(monkeypatch)
    planner._observation_counts['stable_windows'] = 6544
    planner.values['planning_observation_wait_seconds'] = 0.5

    def receive(seconds):
        planner.advance(seconds)
        planner.publish()

    monkeypatch.setattr(module.time, 'sleep', receive)
    accepted = planner._wait_for_planning_observation()
    assert accepted is not planner._latest_observation
    diagnostic = planner._observation_diagnostics()
    assert diagnostic['ready']
    assert diagnostic['blocking_reason'] == ''
    waited = diagnostic['last_observation_wait']
    assert waited['outcome'] == 'ready'
    assert waited['counts'] == {
        'received': 8,
        'accepted_frames': 8,
        'unstable_windows': 7,
        'stable_windows': 1,
    }


def test_no_input_reports_missing_stream_instead_of_window_noise(monkeypatch):
    planner = DiagnosticsHarness(monkeypatch)
    monkeypatch.setattr(module.time, 'sleep', planner.advance)
    with pytest.raises(ValueError, match='reason=no_surface_input'):
        planner._wait_for_planning_observation()
