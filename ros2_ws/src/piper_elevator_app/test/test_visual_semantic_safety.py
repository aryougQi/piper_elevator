"""Semantic conflict gates exercised without ROS nodes or robot commands."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from rclpy.time import Time
from std_msgs.msg import String
from std_srvs.srv import Trigger

from piper_elevator_app.button_visual_servo import ButtonVisualServo
from test_approach_servo_handoff import ServoStartHarness
from test_servo_normal_consistency import (
    TrackingHarness, aligned_frames, assert_held, frame,
)
from test_servo_observation_safety import VisualHarness, surface


def tracking_payload(stamp_ns=9_950_000_000, reason='direction_conflict'):
    positive = reason == ''
    return {
        'stamp': {'sec': stamp_ns // 1_000_000_000,
                  'nanosec': stamp_ns % 1_000_000_000},
        'frame_id': 'camera',
        'reason': reason,
        'selected': {
            'class_name': '2',
            'stable_detection': positive,
            'depth_valid': positive,
            'measured': {'class_name': '2'} if positive else None,
        },
    }


def status_message(stamp_ns=9_950_000_000, reason='direction_conflict'):
    return String(data=json.dumps(tracking_payload(stamp_ns, reason)))


class SemanticHarness(VisualHarness):
    _tracking_state_callback = ButtonVisualServo._tracking_state_callback
    _semantic_conflict_locked = ButtonVisualServo._semantic_conflict_locked
    _publish_twist = ButtonVisualServo._publish_twist
    _publish_zero_twist = ButtonVisualServo._publish_zero_twist

    def __init__(self):
        super().__init__()
        self.commands = []
        self._twist_publisher = SimpleNamespace(publish=self.commands.append)
        self._servo_command_started_at = 0.0
        self._running = True

    @property
    def held(self):
        with self._condition:
            return bool(self._semantic_conflict_locked())


def assert_zero_command(command):
    assert [command.twist.linear.x, command.twist.linear.y, command.twist.linear.z] == [0.0] * 3
    assert [command.twist.angular.x, command.twist.angular.y, command.twist.angular.z] == [0.0] * 3


@pytest.mark.parametrize('reason', ['direction_conflict', 'projection_conflict'])
def test_trusted_conflict_immediately_holds_and_blocks_later_nonzero_publish(reason):
    servo = SemanticHarness()
    servo._surface_pose_callback(surface())
    servo._tracking_state_callback(status_message(reason=reason))
    assert servo.held
    assert not servo._stop_event.is_set()
    assert_zero_command(servo.commands[-1])
    servo._publish_twist(np.array([0.1, 0.2, 0.3]), np.array([0.3, 0.2, 0.1]))
    assert_zero_command(servo.commands[-1])
    assert servo._servo_command_started_at == 0.0


@pytest.mark.parametrize('invalid_field', [
    'wrong_selection', 'wrong_frame', 'zero_stamp', 'old_stamp', 'future_stamp',
    'fractional_seconds', 'boolean_seconds', 'nanoseconds_range', 'unknown_reason',
    'missing_selected', 'malformed_json',
])
def test_invalid_or_unrelated_status_does_not_block_current_task(invalid_field):
    servo = SemanticHarness()
    payload = tracking_payload()
    if invalid_field == 'wrong_selection':
        payload['selected']['class_name'] = '3'
    elif invalid_field == 'wrong_frame':
        payload['frame_id'] = 'another_camera'
    elif invalid_field == 'zero_stamp':
        payload['stamp'] = {'sec': 0, 'nanosec': 0}
    elif invalid_field == 'old_stamp':
        payload['stamp'] = {'sec': 9, 'nanosec': 0}
    elif invalid_field == 'future_stamp':
        payload['stamp'] = {'sec': 10, 'nanosec': 50_000_000}
    elif invalid_field == 'fractional_seconds':
        payload['stamp']['sec'] = 9.0
    elif invalid_field == 'boolean_seconds':
        payload['stamp']['sec'] = True
    elif invalid_field == 'nanoseconds_range':
        payload['stamp']['nanosec'] = 1_000_000_000
    elif invalid_field == 'unknown_reason':
        payload['reason'] = 'ordinary_missing_frame'
    elif invalid_field == 'missing_selected':
        del payload['selected']
    message = String(data='{' if invalid_field == 'malformed_json' else json.dumps(payload))
    servo._tracking_state_callback(message)
    assert not servo.held
    assert not servo.commands


@pytest.mark.parametrize('surface_first', [False, True])
def test_recovery_requires_new_positive_and_corresponding_fresh_surface(surface_first):
    servo = SemanticHarness()
    servo._surface_pose_callback(surface(9_940_000_000))
    servo._tracking_state_callback(status_message(9_950_000_000))
    if surface_first:
        servo._surface_pose_callback(surface(9_970_000_000))
    else:
        servo._tracking_state_callback(status_message(9_970_000_000, ''))
    assert servo.held
    if surface_first:
        servo._tracking_state_callback(status_message(9_970_000_000, ''))
    else:
        servo._surface_pose_callback(surface(9_970_000_000))
    assert not servo.held
    servo._publish_twist(np.array([0.1, 0.0, 0.0]), np.zeros(3))
    assert servo.commands[-1].twist.linear.x == 0.1


def test_old_and_duplicate_positive_messages_cannot_clear_newer_negative_evidence():
    servo = SemanticHarness()
    servo._surface_pose_callback(surface(9_970_000_000))
    servo._tracking_state_callback(status_message(9_950_000_000, ''))
    servo._tracking_state_callback(status_message(9_960_000_000))
    for stamp_ns in (9_950_000_000, 9_960_000_000, 9_950_000_000):
        servo._tracking_state_callback(status_message(stamp_ns, ''))
        assert servo.held
    servo._tracking_state_callback(status_message(9_970_000_000, ''))
    assert not servo.held
    servo._tracking_state_callback(status_message(9_960_000_000))
    assert not servo.held


def test_negative_wins_same_capture_disagreement_even_after_positive_arrives_first():
    servo = SemanticHarness()
    servo._surface_pose_callback(surface())
    servo._tracking_state_callback(status_message(reason=''))
    servo._tracking_state_callback(status_message())
    assert servo.held
    servo._tracking_state_callback(status_message(reason=''))
    assert servo.held


@pytest.mark.parametrize('field,value', [
    ('stable_detection', False), ('depth_valid', False),
    ('stable_detection', 1), ('depth_valid', 'true'),
    ('measured', None), ('measured', {'class_name': '3'}),
])
def test_incomplete_positive_quality_does_not_release_conflict(field, value):
    servo = SemanticHarness()
    servo._tracking_state_callback(status_message())
    servo._surface_pose_callback(surface(9_970_000_000))
    payload = tracking_payload(9_970_000_000, '')
    payload['selected'][field] = value
    servo._tracking_state_callback(String(data=json.dumps(payload)))
    assert servo.held


def test_previous_selection_negative_is_rejected_after_selecting_same_label_again():
    servo = SemanticHarness()
    servo._tracking_state_callback(status_message())
    servo._running = False
    servo._button_selection_callback(String(data='3'))
    servo.now_ns += 10_000_000
    servo._button_selection_callback(String(data='2'))
    servo._tracking_state_callback(status_message())
    assert not servo.held


def start_harness(monkeypatch):
    payload = {
        'schema_version': 1, 'handover_id': 'semantic-test',
        'selected_button': '2', 'frame_id': 'base_link',
        'observation_stamp_ns': 9_950_000_000, 'verified_at_ns': 10_000_000_000,
        'button': [0.4, 0.0, 0.3], 'normal': [1.0, 0.0, 0.0],
        'tcp_position': [0.26, 0.0, 0.3], 'tcp_orientation': [0.0, 0.0, 0.0, 1.0],
        'joint_positions': {'joint1': 0.0},
    }
    servo = ServoStartHarness(payload, monkeypatch)
    servo._camera_frame = 'camera'
    return servo


def test_start_cannot_clear_existing_same_selection_conflict(monkeypatch):
    servo = start_harness(monkeypatch)
    servo._tracking_state_callback(status_message())
    for _ in range(2):
        response = servo._start_callback(None, Trigger.Response())
        assert not response.success
        assert 'direction_conflict' in response.message
        assert servo.claim_calls == 0
        assert not servo.started_threads
        servo.now_ns += 1_000_000_000


def test_negative_arriving_during_handover_claim_prevents_worker_start(monkeypatch):
    servo = start_harness(monkeypatch)
    servo.during_claim = lambda: servo._tracking_state_callback(status_message())
    response = servo._start_callback(None, Trigger.Response())
    assert not response.success
    assert 'direction_conflict' in response.message
    assert servo.claim_calls == 1
    assert not servo.started_threads


def test_conflict_after_alignment_prevents_press_claim(monkeypatch):
    servo = start_harness(monkeypatch)
    servo._running = True
    servo._handoff_ready = True
    servo._tracking_state_callback(status_message())
    response = servo._claim_for_press_callback(None, Trigger.Response())
    assert not response.success
    assert 'direction_conflict' in response.message
    assert not servo._press_release_requested.is_set()


def test_conflict_during_press_wait_revokes_completion(monkeypatch):
    servo = start_harness(monkeypatch)
    servo._running = True
    servo._handoff_ready = True
    servo._tracking_state_callback(status_message())
    assert not servo._hold_for_press_claim()
    assert servo.completions[-1] is False


def test_conflict_arriving_during_press_claim_wait_prevents_success(monkeypatch):
    servo = start_harness(monkeypatch)
    servo._running = True
    servo._handoff_ready = True
    servo._handoff_pending = True

    def released_after_conflict():
        servo._tracking_state_callback(status_message())
        return True

    servo._handoff_released = SimpleNamespace(is_set=released_after_conflict)
    response = servo._claim_for_press_callback(None, Trigger.Response())
    assert not response.success
    assert 'direction_conflict' in response.message


def test_mock_without_tracking_state_keeps_existing_command_path():
    servo = SemanticHarness()
    servo._publish_twist(np.array([0.1, 0.0, 0.0]), np.zeros(3))
    assert servo.commands[-1].twist.linear.x == 0.1


class SemanticTrackingHarness(TrackingHarness):
    _tracking_state_callback = ButtonVisualServo._tracking_state_callback
    _surface_stamp_is_fresh = ButtonVisualServo._surface_stamp_is_fresh
    _observation_is_fresh_locked = ButtonVisualServo._observation_is_fresh_locked

    def __init__(self, frames, monkeypatch, *, conflict_index=5, recovery_index=None,
                 reason='direction_conflict', **parameters):
        self.conflict_index = conflict_index
        self.recovery_index = recovery_index
        self.reason = reason
        self._selected_button = '2'
        self._camera_frame = 'camera'
        self._selection_changed_stamp_ns = 0
        self._observation_stamp_ns = 0
        self._running = True
        super().__init__(frames, monkeypatch, **parameters)

    def get_clock(self):
        return SimpleNamespace(now=lambda: Time(nanoseconds=int(self.now * 1e9)))

    def install_frame(self):
        super().install_frame()
        stamp = int(self.now * 1e9)
        if self._observation is not None:
            self._observation_stamp_ns = stamp
        if self.index == self.conflict_index:
            self._tracking_state_callback(status_message(stamp, self.reason))
        if self.index == self.recovery_index:
            self._tracking_state_callback(status_message(stamp, ''))


@pytest.mark.parametrize('reason', ['direction_conflict', 'projection_conflict'])
@pytest.mark.parametrize('at_target', [False, True])
def test_final_approach_cannot_blindly_continue_after_semantic_conflict(
    monkeypatch, reason, at_target,
):
    servo = SemanticTrackingHarness(
        aligned_frames() + [frame(6, visible=False)] * 4,
        monkeypatch, reason=reason, at_target=at_target,
    )
    locked, _, _ = servo.run()
    assert locked is None
    assert_held(servo, [5, 6, 7, 8])
    assert any('SEMANTIC_CONFLICT_HOLD' in message for _, message in servo.statuses)
    assert not any('VISION_LOSS_CONTINUING' in message for _, message in servo.statuses)


def test_final_approach_resumes_only_after_positive_surface_pair(monkeypatch):
    servo = SemanticTrackingHarness(
        aligned_frames() + [frame(6, visible=False), frame(6, visible=False), frame(7), frame(8)],
        monkeypatch, recovery_index=7, at_target=False,
    )
    servo.run()
    assert_held(servo, [5, 6])
    assert any(index >= 7 and linear[0] > 0.0 for index, linear, _ in servo.commands)


def test_semantic_hold_expires_without_using_blind_motion_budget(monkeypatch):
    servo = SemanticTrackingHarness(
        aligned_frames() + [frame(6, visible=False)] * 8,
        monkeypatch, at_target=False, observation_timeout_seconds=0.05,
    )
    locked, _, message = servo.run()
    assert locked is None
    assert message == 'unresolved semantic conflict: direction_conflict'
    assert_held(servo, range(5, servo.index + 1))
