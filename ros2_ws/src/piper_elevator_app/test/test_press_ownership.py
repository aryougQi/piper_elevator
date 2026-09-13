"""Exercise acknowledged handoff and stop ownership without ROS motion."""

import threading
from types import SimpleNamespace

import pytest
import numpy as np
from std_srvs.srv import Trigger

from piper_elevator_app.button_press_executor import ButtonPressExecutor


class PressHarness:
    _run_press = ButtonPressExecutor._run_press
    _stop_callback = ButtonPressExecutor._stop_callback
    _cleanup_servo_control = ButtonPressExecutor._cleanup_servo_control
    _resume_moveit_servo = ButtonPressExecutor._resume_moveit_servo

    def __init__(self):
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._running = False
        self._stop_in_progress = False
        self._stop_generation = 0
        self._owns_servo = False
        self._cleanup_confirmed = True
        self._cleanup_message = ''
        self._servo_started = False
        self._visual_completed = False
        self._visual_completion_button = ''
        self._visual_handoff_client = 'handoff'
        self._servo_start_client = 'start'
        self._servo_unpause_client = 'unpause'
        self.calls = []
        self.statuses = []
        self.completions = []
        self.claims = []
        self.parameters = {
            'continuous_servo_handoff': True,
            'handoff_timeout_seconds': 2.0,
            'stop_timeout_seconds': 0.10,
        }
        self.responses = {
            'handoff': (False, 'handoff rejected'),
            'start': (True, 'started'),
            'unpause': (False, 'unpause rejected'),
        }
        self.pause_result = (True, '')
        self.gate_result = (True, '')

    def get_parameter(self, name):
        return SimpleNamespace(value=self.parameters[name])

    def get_logger(self):
        return SimpleNamespace(info=self.statuses.append, error=self.statuses.append)

    def _call_trigger(self, client, label, **kwargs):
        self.calls.append(client)
        return self.responses[client]

    def _publish_zero_twist(self):
        self.calls.append('zero')

    def _pause_moveit_servo(self, wait):
        assert wait
        self.calls.append('pause')
        return self.pause_result

    def _set_hardware_servo_gate(self, enabled, wait=True):
        assert wait
        self.calls.append('enable' if enabled else 'disable')
        return self.gate_result

    def _publish_servo_claim(self, claimed):
        self.claims.append(claimed)

    def _publish_completion(self, completed):
        self.completions.append(completed)

    def _publish_status(self, text):
        self.statuses.append(text)

    def _start_timed_phase(self, phase, status=None):
        self._phase_timer.start(phase)

    def _publish_timing(self, *args):
        pass


@pytest.mark.parametrize('continuous', [False, True])
def test_rejected_handoff_never_writes_or_stops_visual_control(continuous):
    press = PressHarness()
    press.parameters['continuous_servo_handoff'] = continuous
    press._running = True
    press._run_press()

    assert press.calls == ['handoff']
    assert press.claims == [False]
    assert press.completions == [False]
    assert not press._owns_servo
    assert not press._running


def test_stop_during_handoff_cleans_claim_without_opening_gate():
    press = PressHarness()
    press._running = True

    def claim_then_stop(*args, **kwargs):
        press.calls.append('handoff')
        press._stop_event.set()
        return True, 'claimed'

    press._call_trigger = claim_then_stop
    press._run_press()

    assert press.calls == ['handoff', 'zero', 'pause', 'disable']
    assert press.claims == [True, False]
    assert press.completions == [False]
    assert press._cleanup_confirmed
    assert not press._running


def test_idle_stop_is_idempotent_and_does_not_touch_other_owner():
    press = PressHarness()
    for _ in range(2):
        response = press._stop_callback(None, Trigger.Response())
        assert response.success
    assert press.calls == []


def test_active_stop_does_not_claim_success_before_worker_exit():
    press = PressHarness()
    press._running = True
    response = press._stop_callback(None, Trigger.Response())

    assert not response.success
    assert press._running
    assert press._stop_event.is_set()
    assert not press._stop_in_progress


@pytest.mark.parametrize('failed_operation', ['pause', 'gate'])
def test_cleanup_failure_retains_ownership_for_explicit_retry(failed_operation):
    press = PressHarness()
    press._owns_servo = True
    if failed_operation == 'pause':
        press.pause_result = (False, 'pause unconfirmed')
    else:
        press.gate_result = (False, 'gate unconfirmed')

    response = press._stop_callback(None, Trigger.Response())
    assert not response.success
    assert press._owns_servo
    assert press.claims == [True]

    press.pause_result = press.gate_result = (True, '')
    assert press._stop_callback(None, Trigger.Response()).success
    assert not press._owns_servo
    assert press.claims == [True, False]


def test_unpause_rejection_cannot_be_reported_as_resumed():
    press = PressHarness()
    success, message = press._resume_moveit_servo()

    assert not success
    assert message == 'unpause rejected'
    assert press.calls == ['start', 'unpause']


def test_stop_prevents_new_servo_resume():
    press = PressHarness()
    press._stop_event.set()
    assert not press._resume_moveit_servo()[0]
    assert press.calls == []


@pytest.mark.parametrize('previously_started', [False, True])
def test_stop_during_unpause_ack_is_not_reported_as_resumed(previously_started):
    press = PressHarness()
    press._servo_started = previously_started

    def acknowledge(client, label, **kwargs):
        press.calls.append(client)
        if client == 'unpause':
            press._stop_event.set()
        return True, 'acknowledged'

    press._call_trigger = acknowledge
    resumed, message = press._resume_moveit_servo()
    assert not resumed
    assert 'stopped' in message
    assert press.calls == (['unpause'] if previously_started else ['start', 'unpause'])


class HardwareGateHarness(PressHarness):
    _set_hardware_servo_gate = ButtonPressExecutor._set_hardware_servo_gate

    def __init__(self):
        super().__init__()
        self._owns_servo = True
        self._last_gate_heartbeat = 0.0
        self.parameters.update({
            'hardware_gate_required': True,
            'hardware_gate_heartbeat_seconds': 0.20,
        })
        self._hardware_gate_client = SimpleNamespace(
            wait_for_service=lambda **kwargs: True,
            call_async=lambda request: self.calls.append(request.data),
        )

    def _wait_for_future(self, future, timeout):
        return SimpleNamespace(success=True, message='hardware enabled')


@pytest.mark.parametrize('change', ['stop', 'ownership_lost'])
def test_hardware_enable_rechecks_control_after_service_availability_wait(change):
    press = HardwareGateHarness()

    def ready_after_revocation(**kwargs):
        with press._condition:
            if change == 'stop':
                press._stop_event.set()
            else:
                press._owns_servo = False
        return True

    press._hardware_gate_client.wait_for_service = ready_after_revocation
    success, _ = press._set_hardware_servo_gate(True)
    assert not success
    assert press.calls == []


def test_stop_during_hardware_enable_ack_leaves_cleanup_to_owner():
    press = HardwareGateHarness()

    def stopped_before_ack(future, timeout):
        with press._condition:
            press._stop_event.set()
        return SimpleNamespace(success=True, message='hardware enabled')

    press._wait_for_future = stopped_before_ack
    success, _ = press._set_hardware_servo_gate(True)
    assert not success
    assert press.calls == [True]
    assert press._owns_servo
    assert press._last_gate_heartbeat == 0.0


class HoldHarness(PressHarness):
    _collect_torque_baseline = ButtonPressExecutor._collect_torque_baseline
    _settle_servo_origin = ButtonPressExecutor._settle_servo_origin

    def __init__(self):
        super().__init__()
        self.parameters.update({
            'baseline_timeout_seconds': 0.10,
            'servo_settle_timeout_seconds': 0.10,
            'servo_settle_required_samples': 3,
            'servo_settle_position_tolerance_m': 0.0002,
            'servo_settle_direction_tolerance_rad': 0.005,
        })
        self._motion_state_stamp_ns = 0

    def _start_timed_phase(self, *args):
        pass

    def _refresh_gate_or_raise(self):
        self.calls.append('heartbeat')

    def _wait_period(self):
        pass

    def _current_motion_state(self):
        self._motion_state_stamp_ns += 1
        return np.zeros(3), np.array([0.0, 0.0, 1.0])

    def _fresh_effort_sample(self, after_sequence):
        return ['joint1'], [0.1], 0.0, after_sequence + 1


def test_torque_baseline_keeps_inherited_zero_hold_and_gate_heartbeat():
    press = HoldHarness()
    detector = SimpleNamespace(baseline_ready=False, samples=0)

    def add_sample(*args):
        detector.samples += 1
        detector.baseline_ready = detector.samples == 3

    detector.add_baseline_sample = add_sample
    press._collect_torque_baseline(detector)
    assert press.calls == ['zero', 'heartbeat'] * 3


@pytest.mark.parametrize('maintain_control', [False, True])
def test_settling_only_maintains_control_before_release(maintain_control):
    press = HoldHarness()
    state = press._settle_servo_origin(maintain_control=maintain_control)

    np.testing.assert_array_equal(state[0], np.zeros(3))
    assert press.calls == (['zero', 'heartbeat'] * 3 if maintain_control else [])
