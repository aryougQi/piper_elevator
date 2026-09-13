"""Exercise image, TF, and press authorization freshness without DDS."""

import threading
import time
from types import SimpleNamespace

from geometry_msgs.msg import PoseStamped, TransformStamped
import numpy as np
import pytest
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from piper_elevator_app.button_press_executor import ButtonPressExecutor
from piper_elevator_app.button_visual_servo import ButtonVisualServo


def transform(stamp_ns=9_950_000_000, child='camera'):
    result = TransformStamped()
    result.header.stamp = Time(nanoseconds=stamp_ns).to_msg()
    result.child_frame_id = child
    result.transform.rotation.w = 1.0
    return result


def surface(stamp_ns=9_950_000_000, x=0.0):
    result = PoseStamped()
    result.header.frame_id = 'camera'
    result.header.stamp = Time(nanoseconds=stamp_ns).to_msg()
    result.pose.position.x = x
    result.pose.position.z = 0.4
    result.pose.orientation.w = 1.0
    return result


class Parameters:
    values = {
        'target_max_age_seconds': 0.75,
        'maximum_tf_fallback_age_seconds': 0.02,
        'maximum_target_jump_m': 0.015,
        'world_position_smoothing_alpha': 0.25,
        'world_normal_smoothing_alpha': 0.20,
        'tf_timeout_seconds': 0.25,
        'feedback_timeout_seconds': 0.25,
        'simulation_mode': False,
        'simulation_future_stamp_tolerance_seconds': 0.02,
        'contact_detection_mode': 'torque',
        'geometry_press_enabled': False,
        'geometry_press_surface_travel_m': 0.030,
        'maximum_approach_travel_m': 0.038,
        'allow_execution': True,
        'torque_thresholds_calibrated': True,
    }

    def get_parameter(self, name):
        return SimpleNamespace(value=self.values[name])

    def get_clock(self):
        return SimpleNamespace(now=lambda: Time(nanoseconds=self.now_ns))

    def get_logger(self):
        return SimpleNamespace(
            warning=lambda text, **kwargs: self.logs.append(text)
        )


class VisualHarness(Parameters):
    _surface_pose_callback = ButtonVisualServo._surface_pose_callback
    _surface_stamp_is_fresh = ButtonVisualServo._surface_stamp_is_fresh
    _button_selection_callback = ButtonVisualServo._button_selection_callback
    _observation_is_fresh_locked = (
        ButtonVisualServo._observation_is_fresh_locked
    )
    _current_servo_pose = ButtonVisualServo._current_servo_pose
    _future_stamp_tolerance = ButtonVisualServo._future_stamp_tolerance

    def __init__(self):
        self.now_ns = 10_000_000_000
        self.logs = []
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._running = False
        self._selected_button = '2'
        self._selection_changed_stamp_ns = 0
        self._selection_generation = 0
        self._observation_stamp_ns = 0
        self._observation_sequence = 0
        self._observation = None
        self._observation_anchor = None
        self._filtered_world_position = None
        self._filtered_world_normal = None
        self._base_frame = 'base'
        self._camera_frame = 'camera'
        self._end_effector_link = 'tip'

    def _lookup_surface_transform(self, frame, stamp):
        return transform()

    def _publish_status(self, text):
        self.logs.append(text)

    def _publish_completion(self, value):
        self.completion = value


@pytest.mark.parametrize('stamp_ns', [0, 9_000_000_000, 10_050_000_000])
def test_surface_rejects_missing_old_and_future_capture_time(stamp_ns):
    servo = VisualHarness()
    servo._surface_pose_callback(surface(stamp_ns))
    assert servo._observation is None
    assert servo._observation_sequence == 0


def test_surface_duplicate_and_out_of_order_frames_do_not_refresh_tracking():
    servo = VisualHarness()
    servo._surface_pose_callback(surface())
    accepted = servo._observation
    servo._surface_pose_callback(surface())
    servo._surface_pose_callback(surface(9_940_000_000))
    assert servo._observation is accepted
    assert servo._observation_sequence == 1
    servo.now_ns = 10_800_000_000
    assert not servo._observation_is_fresh_locked()


def test_vision_loss_budget_includes_age_before_callback(monkeypatch):
    servo = VisualHarness()
    monkeypatch.setattr(time, 'monotonic', lambda: 100.0)
    servo._surface_pose_callback(surface(9_600_000_000))
    assert servo._observation[2] == pytest.approx(99.6)


def test_selection_change_during_tf_lookup_cannot_install_old_target():
    servo = VisualHarness()

    def switch_selection(frame, stamp):
        servo._button_selection_callback(String(data='3'))
        return transform()

    servo._lookup_surface_transform = switch_selection
    servo._surface_pose_callback(surface())
    assert servo._selected_button == '3'
    assert servo._observation is None
    assert servo._observation_anchor is None


def test_surface_expiring_during_tf_lookup_is_rejected():
    servo = VisualHarness()

    def delayed_lookup(frame, stamp):
        servo.now_ns += 1_000_000_000
        return transform()

    servo._lookup_surface_transform = delayed_lookup
    servo._surface_pose_callback(surface())
    assert servo._observation is None


def test_out_of_order_callback_completion_keeps_newer_frame():
    servo = VisualHarness()
    original_lookup = servo._lookup_surface_transform

    def complete_newer_callback(frame, stamp):
        servo._lookup_surface_transform = original_lookup
        servo._surface_pose_callback(surface(9_960_000_000, x=0.001))
        return transform()

    servo._lookup_surface_transform = complete_newer_callback
    servo._surface_pose_callback(surface())
    assert servo._observation_stamp_ns == 9_960_000_000
    assert servo._observation_sequence == 1


def test_changing_selection_stops_active_servo():
    servo = VisualHarness()
    servo._running = True
    servo._button_selection_callback(String(data='3'))
    assert servo._stop_event.is_set()


class PressHarness(Parameters):
    _current_motion_state = ButtonPressExecutor._current_motion_state
    _visual_completion_callback = (
        ButtonPressExecutor._visual_completion_callback
    )
    _button_selection_callback = ButtonPressExecutor._button_selection_callback
    _effort_callback = ButtonPressExecutor._effort_callback
    _fresh_effort_sample = ButtonPressExecutor._fresh_effort_sample
    _effort_feedback_stale = ButtonPressExecutor._effort_feedback_stale
    _start_callback = ButtonPressExecutor._start_callback
    _contact_mode = ButtonPressExecutor._contact_mode
    _contact_mode_needs_torque = (
        ButtonPressExecutor._contact_mode_needs_torque
    )
    _geometry_press_enabled = ButtonPressExecutor._geometry_press_enabled
    _geometry_press_travel = ButtonPressExecutor._geometry_press_travel
    _future_stamp_tolerance = ButtonPressExecutor._future_stamp_tolerance

    def __init__(self):
        self.now_ns = 10_000_000_000
        self.logs = []
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._running = False
        self._stop_in_progress = False
        self._stop_generation = 0
        self._owns_servo = False
        self._cleanup_confirmed = True
        self._cleanup_message = ''
        self._visual_completed = False
        self._visual_completion_level = False
        self._visual_completion_button = ''
        self._selected_button = '2'
        self._latest_effort = None
        self._effort_sequence = 0
        self._effort_stamp_ns = 0
        self._base_frame = 'base'
        self._camera_frame = 'camera'
        self._end_effector_link = 'tip'
        self._motion_state_stamp_ns = 123
        self._tf_buffer = SimpleNamespace(
            lookup_transform=lambda *a, **k: transform()
        )

    def _publish_status(self, text):
        self.logs.append(text)

    def _make_contact_detector(self):
        return object()

    def _publish_servo_claim(self, value):
        pass

    def _publish_completion(self, value):
        pass

    def _run_press(self):
        raise AssertionError('This test must not run a press worker')


@pytest.mark.parametrize('harness,method', [
    (VisualHarness, '_current_servo_pose'),
    (PressHarness, '_current_motion_state'),
])
@pytest.mark.parametrize('bad_frame', ['tip', 'camera'])
@pytest.mark.parametrize('bad_stamp', [0, 9_700_000_000, 10_010_000_000])
def test_live_feedback_rejects_stale_or_future_tf(
    harness, method, bad_frame, bad_stamp,
):
    node = harness()

    def lookup(base, child, *args, **kwargs):
        stamp = bad_stamp if child == bad_frame else 9_950_000_000
        return transform(stamp, child)

    node._tf_buffer = SimpleNamespace(lookup_transform=lookup)
    assert getattr(node, method)() is None
    assert node.logs
    if isinstance(node, PressHarness):
        assert node._motion_state_stamp_ns == 0


@pytest.mark.parametrize('harness,method', [
    (VisualHarness, '_current_servo_pose'),
    (PressHarness, '_current_motion_state'),
])
def test_live_feedback_accepts_fresh_tf(harness, method):
    node = harness()
    node._tf_buffer = SimpleNamespace(
        lookup_transform=lambda *a, **k: transform()
    )
    state = getattr(node, method)()
    assert state is not None
    np.testing.assert_array_equal(state[0], [0.0, 0.0, 0.0])


def _one_step_future_lookup(base, child, *args, **kwargs):
    # Gazebo advances sim time in 1 ms steps, so the newest TF can carry a
    # stamp one step ahead of the /clock sample the node is holding.
    stamp = 10_001_000_000 if child == 'tip' else 9_950_000_000
    return transform(stamp, child)


@pytest.mark.parametrize('harness,method', [
    (VisualHarness, '_current_servo_pose'),
    (PressHarness, '_current_motion_state'),
])
def test_simulation_accepts_one_step_future_tf(harness, method, monkeypatch):
    monkeypatch.setattr(
        Parameters,
        'values',
        dict(Parameters.values, simulation_mode=True),
    )
    node = harness()
    node._tf_buffer = SimpleNamespace(
        lookup_transform=_one_step_future_lookup
    )
    assert getattr(node, method)() is not None
    assert not node.logs


@pytest.mark.parametrize('harness,method', [
    (VisualHarness, '_current_servo_pose'),
    (PressHarness, '_current_motion_state'),
])
def test_real_hardware_still_rejects_future_tf(harness, method):
    node = harness()
    node._tf_buffer = SimpleNamespace(
        lookup_transform=_one_step_future_lookup
    )
    assert getattr(node, method)() is None
    assert node.logs


def test_planner_future_stamp_tolerance_is_simulation_only():
    from test_approach_planning import PlannerHarness

    planner = PlannerHarness()
    assert planner.values['simulation_mode'] is False
    assert planner._future_stamp_tolerance() == 0.0
    planner.values['simulation_mode'] = True
    assert planner._future_stamp_tolerance() == pytest.approx(0.02)
    planner.values['simulation_future_stamp_tolerance_seconds'] = -0.5
    assert planner._future_stamp_tolerance() == 0.0


def test_press_handoff_is_consumed_and_requires_new_alignment(monkeypatch):
    press = PressHarness()
    workers = []
    monkeypatch.setattr(
        threading, 'Thread',
        lambda **kwargs: SimpleNamespace(start=lambda: workers.append(kwargs)),
    )
    press._visual_completion_callback(Bool(data=True))
    assert press._start_callback(None, Trigger.Response()).success
    assert len(workers) == 1
    assert not press._visual_completed
    press._running = False
    press._visual_completion_callback(Bool(data=True))
    assert not press._start_callback(None, Trigger.Response()).success
    press._visual_completion_callback(Bool(data=False))
    press._visual_completion_callback(Bool(data=True))
    assert press._start_callback(None, Trigger.Response()).success
    assert len(workers) == 2


def test_press_rechecks_handoff_after_preflight():
    press = PressHarness()
    press._visual_completion_callback(Bool(data=True))

    def revoked_during_validation():
        press._visual_completion_callback(Bool(data=False))
        return object()

    press._make_contact_detector = revoked_during_validation
    response = press._start_callback(None, Trigger.Response())
    assert not response.success
    assert not press._running


def test_press_selection_change_revokes_old_alignment():
    press = PressHarness()
    press._visual_completion_callback(Bool(data=True))
    press._button_selection_callback(String(data='2'))
    assert press._visual_completed
    press._button_selection_callback(String(data='3'))
    press._visual_completion_callback(Bool(data=True))
    assert not press._start_callback(None, Trigger.Response()).success
    assert not press._visual_completed


def test_press_selection_change_during_preflight_revokes_authorization():
    press = PressHarness()
    press._visual_completion_callback(Bool(data=True))

    def changed_during_validation():
        press._button_selection_callback(String(data='3'))
        return object()

    press._make_contact_detector = changed_during_validation
    assert not press._start_callback(None, Trigger.Response()).success
    assert not press._running


def test_press_selection_change_stops_active_press():
    press = PressHarness()
    press._running = True
    press._button_selection_callback(String(data='3'))
    assert press._stop_event.is_set()


def effort(stamp_ns):
    message = JointState()
    message.header.stamp = Time(nanoseconds=stamp_ns).to_msg()
    message.name = ['joint1']
    message.effort = [0.5]
    return message


@pytest.mark.parametrize('stamp_ns', [0, 9_700_000_000, 10_010_000_000])
def test_press_rejects_stale_missing_future_effort(stamp_ns):
    press = PressHarness()
    press._effort_callback(effort(stamp_ns))
    assert press._fresh_effort_sample() is None
    assert press._effort_feedback_stale()


def test_simulation_accepts_one_step_future_effort(monkeypatch):
    monkeypatch.setattr(
        Parameters,
        'values',
        dict(Parameters.values, simulation_mode=True),
    )
    press = PressHarness()
    press._effort_callback(effort(10_001_000_000))
    assert press._fresh_effort_sample() is not None
    assert not press._effort_feedback_stale()


def test_real_hardware_still_rejects_future_effort():
    press = PressHarness()
    press._effort_callback(effort(10_001_000_000))
    assert press._fresh_effort_sample() is None
    assert press._effort_feedback_stale()


def test_press_repeated_or_out_of_order_effort_cannot_count_as_new_contact():
    press = PressHarness()
    press._effort_callback(effort(9_950_000_000))
    first = press._fresh_effort_sample()
    assert first is not None
    press._effort_callback(effort(9_950_000_000))
    press._effort_callback(effort(9_940_000_000))
    assert press._fresh_effort_sample(after_sequence=first[3]) is None
    press._effort_callback(effort(9_960_000_000))
    assert press._fresh_effort_sample(after_sequence=first[3]) is not None
    press.now_ns = 10_300_000_000
    assert press._effort_feedback_stale()
    assert press._fresh_effort_sample() is None


def test_effort_freshness_budget_includes_delivery_delay(monkeypatch):
    press = PressHarness()
    monkeypatch.setattr(time, 'monotonic', lambda: 100.0)
    press._effort_callback(effort(9_800_000_000))
    assert press._fresh_effort_sample()[2] == pytest.approx(99.8)
    monkeypatch.setattr(time, 'monotonic', lambda: 100.1)
    assert press._effort_feedback_stale()
