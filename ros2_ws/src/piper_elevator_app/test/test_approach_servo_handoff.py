"""Exercise coarse-to-Servo authorization without robot motion."""

import copy
import json
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.time import Time
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from piper_elevator_app import button_approach_planner as planner_module
from piper_elevator_app import button_visual_servo as servo_module
from piper_elevator_app.button_visual_servo import ButtonVisualServo
from piper_elevator_app.handover_core import decode_coarse_handover
from test_approach_execution import ArmLimitPlanner, arm_trajectory
from test_approach_verification import VerificationHarness


@pytest.fixture
def verified_planner(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(planner_module.time, 'monotonic', lambda: clock[0])
    planner = VerificationHarness(clock)
    monkeypatch.setattr(planner_module.time, 'sleep', planner.sleep)
    success, message = planner._verify_approach_reached(planner.target)
    assert success, message
    return planner


def claim(planner):
    return planner._claim_servo_handover_callback(None, Trigger.Response())


def test_verified_coarse_pose_issues_one_consumable_handover(verified_planner):
    planner = verified_planner
    result = claim(planner)
    assert result.success, result.message
    payload = json.loads(result.message)
    assert payload['schema_version'] == 1
    assert payload['handover_id']
    assert payload['selected_button'] == '2'
    assert payload['frame_id'] == planner._base_frame
    assert 0 < payload['observation_stamp_ns'] <= payload['verified_at_ns']
    np.testing.assert_allclose(payload['button'], planner.button)
    np.testing.assert_allclose(payload['normal'], [1.0, 0.0, 0.0])
    assert payload['joint_positions'] == planner._latest_joint_positions
    assert len(payload['tcp_position']) == 3
    assert len(payload['tcp_orientation']) == 4
    assert planner._verified_handover is None
    assert not claim(planner).success


def test_claim_returns_current_verified_near_view(verified_planner):
    planner = verified_planner
    latest = copy.deepcopy(planner._latest_observation)
    latest['button'][1] += 0.001
    angle = math.radians(1.0)
    latest['normal'] = np.array([math.cos(angle), math.sin(angle), 0.0])
    planner._latest_observation = latest
    result = claim(planner)
    assert result.success, result.message
    payload = json.loads(result.message)
    np.testing.assert_allclose(payload['button'], latest['button'])
    np.testing.assert_allclose(payload['normal'], latest['normal'])
    assert payload['observation_stamp_ns'] == latest['stamp_ns']


@pytest.mark.parametrize('state', [
    'busy', 'unconfirmed_stop', 'missing_token',
])
def test_unavailable_coarse_handover_rejects_claim(verified_planner, state):
    planner = verified_planner
    if state == 'busy':
        planner._busy = True
    elif state == 'unconfirmed_stop':
        planner._motion_stop_unconfirmed = True
    else:
        planner._verified_handover = None
    assert not claim(planner).success


def test_expired_coarse_handover_rejects_claim(verified_planner):
    planner = verified_planner
    planner._verified_handover['verified_at_ns'] = (
        planner.now_ns - 121_000_000_000
    )
    assert not claim(planner).success


@pytest.mark.parametrize('clear', ['plan_reset', 'new_selection'])
def test_new_coarse_task_invalidates_previous_handover(
    verified_planner, clear,
):
    planner = verified_planner
    if clear == 'plan_reset':
        with planner._lock:
            planner._clear_stored_plan_locked()
    else:
        planner._selection_callback(String(data='3'))
    assert planner._verified_handover is None
    assert not claim(planner).success


@pytest.mark.parametrize('failure', [
    'different_button', 'target_jump', 'joint_motion', 'stale_observation',
    'stale_joint_state', 'stale_tcp', 'moved_tcp', 'camera_out_of_view',
])
def test_claim_rechecks_current_scene_and_robot(verified_planner, failure):
    planner = verified_planner
    if failure == 'different_button':
        planner._selected_button = '3'
    elif failure == 'target_jump':
        planner._latest_observation['button'][1] += 0.016
    elif failure == 'joint_motion':
        planner._latest_joint_positions['joint1'] += 0.021
    elif failure == 'stale_observation':
        planner._latest_observation['stamp_ns'] -= 2_000_000_000
    elif failure == 'stale_joint_state':
        planner._latest_joint_stamp_ns -= 2_000_000_000
        planner._latest_joint_received_at -= 2.0
    elif failure == 'stale_tcp':
        planner.stale_tf = True
    elif failure == 'moved_tcp':
        planner.actual.pose.position.y += 0.10
    else:
        planner._latest_observation['tip_to_camera_translation'][0] += 0.2
    result = claim(planner)
    assert not result.success, result.message


def test_selection_during_tcp_lookup_invalidates_claim(verified_planner):
    planner = verified_planner
    planner.on_actual_transform = lambda: planner._selection_callback(
        String(data='3'),
    )
    result = claim(planner)
    assert not result.success
    assert planner._verified_handover is None


def test_concurrent_claim_cannot_consume_handover_twice(verified_planner):
    planner = verified_planner
    competing = []
    planner.on_actual_transform = lambda: competing.append(claim(planner))
    result = claim(planner)
    assert result.success, result.message
    assert len(competing) == 1
    assert not competing[0].success
    assert planner._verified_handover is None


def test_robot_motion_during_tcp_lookup_rejects_claim(verified_planner):
    planner = verified_planner

    def move_joint():
        planner._latest_joint_positions['joint1'] += 0.021

    planner.on_actual_transform = move_joint
    result = claim(planner)
    assert not result.success


class ServoStartHarness(ButtonVisualServo):
    """Retain start checks and stub transport and thread launch."""

    _level_roll_tolerance = ButtonVisualServo._level_roll_tolerance
    _wrist_limit_guard_margin = (
        ButtonVisualServo._wrist_limit_guard_margin
    )
    _wrist_limit_guard_freezes_roll = (
        ButtonVisualServo._wrist_limit_guard_freezes_roll
    )
    _wrist_limit_guard_status_text = (
        ButtonVisualServo._wrist_limit_guard_status_text
    )

    def __init__(self, payload, monkeypatch):
        self.values = {}
        self._declare_parameters()
        self.values.update({
            'allow_execution': True,
            'camera_calibration_valid': True,
        })
        self.now_ns = payload['verified_at_ns']
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._press_claim_event = threading.Event()
        self._running = False
        self._starting = False
        self._stop_pending = False
        self._owns_servo = False
        self._cleanup_confirmed = True
        self._handoff_ready = False
        self._handoff_pending = False
        self._handoff_abort = False
        self._handoff_released = threading.Event()
        self._press_release_requested = threading.Event()
        self._alignment_finished = threading.Event()
        self._alignment_result = (False, 'visual servo has not run')
        self._alignment_thread = None
        self._wrist_guard_position = None
        self._wrist_guard_state_received_at = 0.0
        self._wrist_guard_reported = False
        self._selected_button = '2'
        self._selection_generation = 1
        self._selection_changed_stamp_ns = 0
        self._observation_stamp_ns = 0
        self._observation_sequence = 1
        self._observation = None
        self._base_frame = 'base_link'
        self._camera_frame = self.values['camera_frame']
        self._observation_anchor = np.array([0.9, 0.0, 0.3])
        self._filtered_world_position = self._observation_anchor.copy()
        self._filtered_world_normal = np.array([0.0, 1.0, 0.0])
        self.payload = copy.deepcopy(payload)
        self.claim_calls = 0
        self.during_claim = None
        self.started_threads = []
        self.statuses = []
        self.completions = []
        self.pause_calls = 0
        self.gate_calls = []
        self.real_thread = threading.Thread

        def thread(**kwargs):
            def start():
                # Stub workers never execute the real Servo loop; they only
                # report the alignment result the start service waits for.
                self.started_threads.append(kwargs)
                self._signal_alignment_finished(
                    True,
                    'COMPLETE: harness alignment',
                )

            return SimpleNamespace(
                start=start,
                is_alive=lambda: False,
            )

        monkeypatch.setattr(servo_module.threading, 'Thread', thread)

    def declare_parameter(self, name, value):
        self.values[name] = value

    def get_parameter(self, name):
        return SimpleNamespace(value=self.values[name])

    def get_clock(self):
        return SimpleNamespace(now=lambda: Time(nanoseconds=self.now_ns))

    def get_logger(self):
        return SimpleNamespace(error=lambda *args: None)

    def _claim_coarse_handover(self):
        self.claim_calls += 1
        if self.during_claim is not None:
            self.during_claim()
        if self.payload is None:
            raise ValueError('No verified coarse handover')
        return decode_coarse_handover(
            json.dumps(self.payload), selected_button=self._selected_button,
            frame_id=self._base_frame, now_ns=self.now_ns,
        )

    def _run_servo(self):
        raise AssertionError('Tests must not execute a Servo worker')

    def _publish_completion(self, completed):
        self.completions.append(completed)

    def _publish_status(self, status):
        self.statuses.append(status)

    def _publish_zero_twist(self):
        pass

    def _pause_moveit_servo(self, **kwargs):
        self.pause_calls += 1
        return True, ''

    def _set_hardware_servo_gate(self, enabled, **kwargs):
        self.gate_calls.append(enabled)
        return True, ''


@pytest.fixture
def servo_start(verified_planner, monkeypatch):
    result = claim(verified_planner)
    assert result.success, result.message
    return ServoStartHarness(json.loads(result.message), monkeypatch)


def start(servo):
    return servo._start_callback(None, Trigger.Response())


def test_start_reports_success_only_after_the_alignment_finishes(
    servo_start, monkeypatch,
):
    """A caller must not see success while the Servo is still aligning."""
    servo = servo_start
    release = threading.Event()

    def delayed_worker():
        release.wait(5.0)
        servo._signal_alignment_finished(True, 'COMPLETE: delayed alignment')

    servo._run_servo = delayed_worker
    monkeypatch.setattr(servo_module.threading, 'Thread', servo.real_thread)
    responses = []
    caller = servo.real_thread(
        target=lambda: responses.append(
            servo._start_callback(None, Trigger.Response())
        ),
    )
    caller.start()
    caller.join(0.2)
    assert responses == []
    release.set()
    caller.join(5.0)
    assert responses
    assert responses[0].success
    assert responses[0].message == 'COMPLETE: delayed alignment'


def test_start_reports_the_alignment_failure(servo_start, monkeypatch):
    servo = servo_start

    def failing_worker():
        servo._signal_alignment_finished(
            False,
            'FAILED: no initial RGB-D surface pose',
        )

    servo._run_servo = failing_worker
    monkeypatch.setattr(servo_module.threading, 'Thread', servo.real_thread)
    result = start(servo)
    assert not result.success
    assert 'no initial RGB-D surface pose' in result.message


def test_visual_start_anchors_to_consumed_near_view(servo_start):
    servo = servo_start
    result = start(servo)
    assert result.success, result.message
    assert servo.claim_calls == 1
    assert len(servo.started_threads) == 1
    assert servo._running
    assert not servo._starting
    np.testing.assert_allclose(
        servo._observation_anchor, servo.payload['button'],
    )
    np.testing.assert_allclose(
        servo._filtered_world_position, servo.payload['button'],
    )
    np.testing.assert_allclose(
        servo._filtered_world_normal, servo.payload['normal'],
    )
    assert servo.values['maximum_target_jump_m'] == 0.015


def test_visual_start_requires_handover_even_with_an_existing_target(
    servo_start,
):
    servo = servo_start
    old_anchor = servo._observation_anchor.copy()
    servo.payload = None
    result = start(servo)
    assert not result.success
    assert not servo.started_threads
    assert not servo._running
    assert not servo._starting
    np.testing.assert_array_equal(servo._observation_anchor, old_anchor)


@pytest.mark.parametrize('interrupt', ['stop', 'selection', 'press_busy'])
def test_cancel_or_selection_during_claim_cannot_launch_worker(
    servo_start, interrupt,
):
    servo = servo_start
    stop_threads = []

    def during_claim():
        if interrupt == 'stop':
            thread = servo.real_thread(target=lambda: servo._stop_callback(
                None, Trigger.Response(),
            ))
            stop_threads.append(thread)
            thread.start()
            assert servo._stop_event.wait(1.0)
        elif interrupt == 'selection':
            servo._button_selection_callback(String(data='3'))
        else:
            servo._press_servo_claim_callback(Bool(data=True))

    servo.during_claim = during_claim
    result = start(servo)
    for thread in stop_threads:
        thread.join(1.0)
        assert not thread.is_alive()
    assert not result.success
    assert not servo.started_threads
    assert not servo._running
    assert not servo._starting


def test_second_start_cannot_race_a_pending_claim(servo_start):
    servo = servo_start
    nested = []
    servo.during_claim = lambda: nested.append(start(servo))
    result = start(servo)
    assert result.success, result.message
    assert len(nested) == 1
    assert not nested[0].success
    assert servo.claim_calls == 1
    assert len(servo.started_threads) == 1


def test_idle_visual_stop_does_not_pause_press_session(servo_start):
    servo = servo_start
    servo._press_claim_event.set()
    response = servo._stop_callback(None, Trigger.Response())
    assert response.success, response.message
    assert servo.pause_calls == 0
    assert not servo.gate_calls
    assert servo._press_claim_event.is_set()


def test_old_press_boolean_cannot_authorize_visual_release(servo_start):
    servo = servo_start
    servo._running = True
    servo._owns_servo = True
    servo._handoff_ready = True
    servo._press_servo_claim_callback(Bool(data=True))
    assert not servo._press_release_requested.is_set()
    assert not servo._handoff_released.is_set()
    assert servo._owns_servo


def test_near_view_handover_rebases_old_visual_filter_across_nodes(
    verified_planner, monkeypatch,
):
    planner = verified_planner
    far_button = planner._execution_observation['button'].copy()
    planner.button[0] += 0.022
    success, detail = planner._verify_approach_reached(planner.target)
    assert success, detail
    servo = ServoStartHarness(planner._verified_handover, monkeypatch)
    servo._observation_anchor = far_button.copy()
    servo._filtered_world_position = far_button.copy()
    servo._filtered_world_normal = np.array([1.0, 0.0, 0.0])
    servo._coarse_handover_client = SimpleNamespace(
        wait_for_service=lambda **kwargs: True,
        call_async=lambda request: claim(planner),
    )
    servo._claim_coarse_handover = (
        ButtonVisualServo._claim_coarse_handover.__get__(servo)
    )
    servo._wait_for_future = lambda future, *args: future
    transform = TransformStamped()
    transform.transform.rotation.w = 1.0
    servo._lookup_surface_transform = lambda *args: transform
    message = PoseStamped()
    message.header.frame_id = 'camera_color_optical_frame'
    message.header.stamp = Time(nanoseconds=servo.now_ns).to_msg()
    message.pose.position.x = float(planner.button[0])
    message.pose.position.z = float(planner.button[2])
    orientation = planner._verified_handover['tcp_orientation']
    (message.pose.orientation.x, message.pose.orientation.y,
     message.pose.orientation.z, message.pose.orientation.w) = orientation

    servo._surface_pose_callback(message)
    assert servo._observation is None
    assert any('IGNORED_TARGET_JUMP' in status for status in servo.statuses)
    response = start(servo)
    assert response.success, response.message
    assert planner._verified_handover is None
    np.testing.assert_allclose(servo._observation_anchor, planner.button)
    np.testing.assert_allclose(servo._filtered_world_position, planner.button)
    sequence = servo._observation_sequence
    servo.now_ns += 50_000_000
    message.header.stamp = Time(nanoseconds=servo.now_ns).to_msg()
    servo._surface_pose_callback(message)
    assert servo._observation_sequence == sequence + 1
    np.testing.assert_allclose(servo._observation[0], planner.button)
    assert servo.values['maximum_target_jump_m'] == 0.015


def test_real_joint_limit_failure_never_issues_visual_handover():
    planner = ArmLimitPlanner()
    planner._planned_trajectory = arm_trajectory()
    planner._latest_joint_positions['joint3'] = 0.000768
    planner._verified_handover = {'handover_id': 'obsolete'}
    response = planner._execute_callback(None, Trigger.Response())
    assert not response.success
    assert 'joint3' in response.message
    assert planner._last_execution_diagnostic['phase'] == 'motion_failed'
    assert planner._verified_handover is None
    assert not claim(planner).success
    assert not planner._execute_client.goals
    assert planner.values['joint_state_boundary_tolerance_rad'] == 0.0001
