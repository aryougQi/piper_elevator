"""Exercise exclusive Servo ownership and acknowledged session release."""

import time
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from piper_elevator_app.button_visual_servo import ButtonVisualServo
from test_approach_servo_handoff import ServoStartHarness
from test_servo_normal_consistency import CAMERA_ORIENTATION


PAYLOAD = {
    'schema_version': 1,
    'handover_id': 'verified-coarse-session',
    'selected_button': '2',
    'frame_id': 'base_link',
    'verified_at_ns': 10_000_000_000,
    'observation_stamp_ns': 10_000_000_000,
    'button': [0.4, 0.0, 0.3],
    'normal': [1.0, 0.0, 0.0],
    'tcp_position': [0.26, 0.0, 0.3],
    'tcp_orientation': CAMERA_ORIENTATION.tolist(),
    'joint_positions': {'joint5': 0.6},
}


class LifecycleHarness(ServoStartHarness):
    _run_servo = ButtonVisualServo._run_servo
    _resume_moveit_servo = ButtonVisualServo._resume_moveit_servo
    _pause_moveit_servo = ButtonVisualServo._pause_moveit_servo

    def __init__(self, monkeypatch):
        super().__init__(PAYLOAD, monkeypatch)
        self._running = True
        self._servo_started = False
        self._servo_start_client = 'start'
        self._servo_unpause_client = 'unpause'
        self._servo_pause_client = 'pause'
        self._level_reference_axis = np.array([0.0, 0.0, 1.0])
        self._workspace_min = np.array([-0.65, -0.65, 0.02])
        self._workspace_max = np.array([0.65, 0.65, 0.75])
        self.events = []
        self.service_results = {}
        self.on_service = None
        self.on_initial_pose = None
        self.tracked = False
        self.track_failure = None
        self.hold_result = True
        self.gate_result = True

    def _wait_for_observation(self, *args):
        return (np.array(PAYLOAD['button']), np.array(PAYLOAD['normal']),
                time.monotonic(), 1)

    def _current_servo_pose(self):
        if not self.tracked and self.on_initial_pose is not None:
            self.on_initial_pose()
        return (
            np.array([0.37 if self.tracked else 0.26, 0.0, 0.3]),
            CAMERA_ORIENTATION.copy(), CAMERA_ORIENTATION.copy(),
        )

    def _call_servo_service(self, client, label):
        self.events.append(('service', client))
        if self.on_service is not None:
            self.on_service(client)
        return self.service_results.get(client, (True, ''))

    def _set_hardware_servo_gate(self, enabled, **kwargs):
        self.events.append(('gate', enabled))
        self.gate_calls.append(enabled)
        return self.gate_result, 'gate failure' if not self.gate_result else ''

    def _track_visually(self, deadline, initial):
        self.events.append(('track',))
        if self.track_failure is not None:
            return None, '', self.track_failure
        self.tracked = True
        return (initial[0].copy(), initial[1].copy()), 'visual_target', ''

    def _decelerate_servo_to_hold(self):
        self._publish_zero_twist()
        return True, ''

    def _hold_for_press_claim(self):
        self._publish_zero_twist()
        self.events.append(('release',))
        return self.hold_result

    def _publish_zero_twist(self):
        self.events.append(('zero',))

    def get_logger(self):
        return SimpleNamespace(
            error=lambda *args: None, info=lambda *args: None,
        )


@pytest.fixture
def servo(monkeypatch):
    return LifecycleHarness(monkeypatch)


def test_worker_releases_publisher_before_press_can_own_session(servo):
    servo._run_servo()
    assert not servo._running
    assert servo._handoff_pending
    assert servo._owns_servo
    assert servo._handoff_released.is_set()
    release_index = servo.events.index(('release',))
    assert servo.events[release_index + 1:] == []
    assert ('service', 'pause') not in servo.events
    assert ('gate', False) not in servo.events
    assert servo.completions[-1] is True


def test_press_acknowledgement_ends_visual_ownership(servo):
    servo._run_servo()
    servo._press_servo_claim_callback(Bool(data=True))
    assert not servo._handoff_pending
    assert not servo._owns_servo
    assert servo._cleanup_confirmed
    events = servo.events.copy()
    servo._handoff_deadline = time.monotonic() - 1.0
    servo._check_pending_handoff()
    assert servo.events == events


@pytest.mark.parametrize('abort', ['timeout', 'negative_ack'])
def test_unacknowledged_press_handover_closes_owned_session(servo, abort):
    servo._run_servo()
    if abort == 'timeout':
        servo._handoff_deadline = time.monotonic() - 1.0
    else:
        servo._press_servo_claim_callback(Bool(data=False))
    servo._check_pending_handoff()
    assert not servo._handoff_pending
    assert not servo._owns_servo
    assert servo._cleanup_confirmed
    assert servo.events[-3:] == [
        ('zero',), ('service', 'pause'), ('gate', False),
    ]
    assert servo.completions[-1] is False


def test_failed_handover_cleanup_retains_ownership_for_stop_retry(servo):
    servo._run_servo()
    servo.service_results['pause'] = (False, 'pause rejected')
    servo._handoff_deadline = time.monotonic() - 1.0
    servo._check_pending_handoff()
    assert servo._owns_servo
    assert not servo._cleanup_confirmed
    assert not servo._start_callback(None, Trigger.Response()).success
    servo.service_results['pause'] = (True, '')
    result = servo._stop_callback(None, Trigger.Response())
    assert result.success, result.message
    assert not servo._owns_servo
    assert servo._cleanup_confirmed


def test_stop_does_not_repeat_failed_worker_cleanup_before_explicit_retry(
    servo,
):
    tracking_started = Event()
    servo.service_results['pause'] = (False, 'pause rejected')

    def wait_for_stop(deadline, initial):
        tracking_started.set()
        assert servo._stop_event.wait(2.0)
        return None, '', 'visual servo stopped'

    servo._track_visually = wait_for_stop
    worker = servo.real_thread(target=servo._run_servo)
    worker.start()
    try:
        assert tracking_started.wait(2.0)
        response = servo._stop_callback(None, Trigger.Response())
        worker.join(2.0)
        assert not worker.is_alive()
        assert not response.success
        assert 'cleanup is unconfirmed' in response.message
        assert servo._owns_servo
        assert not servo._cleanup_confirmed
        assert servo.events.count(('service', 'pause')) == 1
        assert servo.events.count(('gate', False)) == 1

        servo.service_results['pause'] = (True, '')
        response = servo._stop_callback(None, Trigger.Response())
        assert response.success, response.message
        assert not servo._owns_servo
        assert servo._cleanup_confirmed
        assert servo.events.count(('service', 'pause')) == 2
        assert servo.events.count(('gate', False)) == 2
    finally:
        servo._stop_event.set()
        worker.join(2.0)


def test_start_is_blocked_until_pending_handover_resolves(servo):
    servo._run_servo()
    result = servo._start_callback(None, Trigger.Response())
    assert not result.success
    assert servo.claim_calls == 0


@pytest.mark.parametrize('service', ['start', 'unpause'])
def test_rejected_servo_start_or_unpause_cannot_complete(servo, service):
    servo.service_results[service] = (False, f'{service} rejected')
    servo._run_servo()
    assert ('gate', True) not in servo.events
    assert ('track',) not in servo.events
    assert True not in servo.completions
    assert not servo._running
    assert not servo._handoff_pending
    assert not servo._owns_servo
    assert any(f'{service} rejected' in status for status in servo.statuses)


def test_stop_during_start_acknowledgement_cannot_unpause_or_enable(servo):
    def stop_after_start(client):
        if client == 'start':
            servo._stop_event.set()

    servo.on_service = stop_after_start
    servo._run_servo()
    assert ('service', 'start') in servo.events
    assert ('service', 'unpause') not in servo.events
    assert ('gate', True) not in servo.events
    assert ('track',) not in servo.events
    assert not servo._owns_servo
    assert servo.completions[-1] is False


def test_stop_before_motion_does_not_open_any_shared_controls(servo):
    servo.on_initial_pose = servo._stop_event.set
    servo._run_servo()
    assert not servo.events
    assert not servo._running
    assert servo.completions[-1] is False


def test_standalone_alignment_timeout_closes_session(servo):
    servo.hold_result = False
    servo._run_servo()
    assert not servo._owns_servo
    assert not servo._handoff_pending
    assert not servo._handoff_released.is_set()
    assert servo.completions == [True, False]
    assert servo.events[-3:] == [
        ('zero',), ('service', 'pause'), ('gate', False),
    ]


@pytest.mark.parametrize('failure', [
    'inactive', 'not_ready', 'stopped', 'already_claimed',
])
def test_press_claim_requires_live_completed_visual_session(servo, failure):
    servo._handoff_ready = True
    if failure == 'inactive':
        servo._running = False
    elif failure == 'not_ready':
        servo._handoff_ready = False
    elif failure == 'stopped':
        servo._stop_event.set()
    else:
        servo._press_release_requested.set()
    response = servo._claim_for_press_callback(None, Trigger.Response())
    assert not response.success
    assert not servo.events


@pytest.mark.parametrize('release', [True, False])
def test_press_claim_waits_for_worker_release_or_stop(servo, release):
    servo._handoff_ready = True
    replies = []
    thread = servo.real_thread(target=lambda: replies.append(
        servo._claim_for_press_callback(None, Trigger.Response()),
    ))
    thread.start()
    try:
        assert servo._press_release_requested.wait(1.0)
        assert not replies
        with servo._condition:
            if release:
                servo._running = False
                servo._handoff_pending = True
                servo._handoff_released.set()
            else:
                servo._stop_event.set()
            servo._condition.notify_all()
        thread.join(1.0)
        assert not thread.is_alive()
        assert len(replies) == 1
        assert replies[0].success is release
    finally:
        servo._stop_event.set()
        with servo._condition:
            servo._condition.notify_all()
        thread.join(1.0)


def test_gate_cannot_reopen_when_stop_is_already_pending(servo):
    servo._stop_event.set()
    result, _ = ButtonVisualServo._set_hardware_servo_gate(servo, True)
    assert not result


def test_gate_availability_wait_does_not_override_new_stop(servo):
    servo.values['hardware_gate_required'] = True
    servo._last_gate_heartbeat = 0.0
    requests = []

    def stop_while_waiting(**kwargs):
        servo._stop_event.set()
        return True

    servo._hardware_gate_client = SimpleNamespace(
        wait_for_service=stop_while_waiting,
        call_async=lambda request: requests.append(request) or object(),
    )
    servo._wait_for_future = lambda *args, **kwargs: SimpleNamespace(
        success=True, message='',
    )
    result, _ = ButtonVisualServo._set_hardware_servo_gate(servo, True)
    assert not result
    assert not requests
