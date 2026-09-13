"""Confirm hardware acknowledgments before reporting Servo readiness."""

from concurrent.futures import Future
import threading
from types import SimpleNamespace

from std_srvs.srv import SetBool

from piper_elevator_app.control_gate import TrajectoryControlGate
from piper_elevator_app.control_gate_core import ControlGatePolicy


class Client:
    def __init__(self, ready=True):
        self.ready = ready
        self.requests = []
        self.called = threading.Event()

    def wait_for_service(self, timeout_sec):
        return self.ready

    def call_async(self, request):
        future = Future()
        self.requests.append((request.data, future))
        self.called.set()
        return future

    def reply(self, index, success):
        self.requests[index][1].set_result(
            SetBool.Response(success=success, message='hardware response')
        )


class GateHarness:
    _servo_authorization_callback = (
        TrajectoryControlGate._servo_authorization_callback
    )
    _update_gate = TrajectoryControlGate._update_gate
    _update_gate_locked = TrajectoryControlGate._update_gate_locked
    _set_gate_mode = TrajectoryControlGate._set_gate_mode
    _set_gate_mode_locked = TrajectoryControlGate._set_gate_mode_locked
    _gate_response_callback = TrajectoryControlGate._gate_response_callback
    _gate_response_locked = TrajectoryControlGate._gate_response_locked

    def __init__(self):
        self._condition = threading.Condition()
        self._policy = ControlGatePolicy()
        self._gate_mode = None
        self._desired_gate_mode = None
        self._pending_gate_future = None
        self._pending_gate_request = None
        self._pending_gate_started_at = None
        self._uncertain_gate_mode = None
        self._gate_clients = {'servo': Client(), 'trajectory': Client()}
        self.messages = []

    def get_parameter(self, name):
        assert name == 'hardware_gate_ack_timeout_seconds'
        return SimpleNamespace(value=0.10)

    def get_logger(self):
        return SimpleNamespace(info=self.messages.append, error=self.messages.append)

    def call(self, enabled):
        return self._servo_authorization_callback(
            SetBool.Request(data=enabled), SetBool.Response(),
        )

    def call_in_thread(self, enabled):
        result = []
        worker = threading.Thread(target=lambda: result.append(self.call(enabled)))
        worker.start()
        return worker, result


def test_enable_waits_until_hardware_acknowledges_open_mode():
    gate = GateHarness()
    worker, result = gate.call_in_thread(True)
    client = gate._gate_clients['servo']
    assert client.called.wait(1.0)
    assert result == []
    assert gate._gate_mode is None

    client.reply(0, True)
    worker.join(1.0)
    assert not worker.is_alive()
    assert result[0].success
    assert gate._gate_mode == 'servo'


def test_enable_timeout_keeps_pending_unknown_then_closes_late_enable():
    gate = GateHarness()
    worker, result = gate.call_in_thread(True)
    client = gate._gate_clients['servo']
    assert client.called.wait(1.0)
    worker.join(1.0)

    assert not worker.is_alive()
    assert not result[0].success
    assert not gate._policy.servo_authorized
    assert gate._uncertain_gate_mode == 'servo'
    assert gate._pending_gate_future is client.requests[0][1]
    assert not gate.call(True).success
    assert len(client.requests) == 1

    client.reply(0, True)
    assert [enabled for enabled, _ in client.requests] == [True, False]
    assert gate._gate_mode == 'servo'
    assert gate._uncertain_gate_mode == 'servo'
    assert not gate.call(True).success

    client.reply(1, True)
    assert gate._gate_mode is None
    assert gate._uncertain_gate_mode is None
    assert gate._pending_gate_future is None
    assert gate.call(False).success


def test_rejected_hardware_disable_cannot_report_control_released():
    gate = GateHarness()
    gate._gate_mode = 'servo'
    worker, result = gate.call_in_thread(False)
    client = gate._gate_clients['servo']
    assert client.called.wait(1.0)
    client.reply(0, False)
    worker.join(1.0)

    assert not worker.is_alive()
    assert not result[0].success
    assert gate._gate_mode == 'servo'
    assert gate._uncertain_gate_mode == 'servo'
    assert not gate.call(True).success


def test_unavailable_hardware_service_is_not_servo_ready():
    gate = GateHarness()
    gate._gate_clients['servo'].ready = False

    response = gate.call(True)
    assert not response.success
    assert not gate._policy.servo_authorized
    assert gate._gate_mode is None


def test_idle_disable_does_not_touch_trajectory_gate():
    gate = GateHarness()
    assert gate.call(False).success
    assert gate._gate_clients['servo'].requests == []
    assert gate._gate_clients['trajectory'].requests == []


def test_lost_enable_response_is_unknown_until_explicit_close_confirmation():
    gate = GateHarness()
    worker, result = gate.call_in_thread(True)
    client = gate._gate_clients['servo']
    assert client.called.wait(1.0)
    client.requests[0][1].set_exception(RuntimeError('response lost'))
    worker.join(1.0)

    assert not worker.is_alive()
    assert not result[0].success
    assert gate._uncertain_gate_mode == 'servo'
    assert not gate.call(True).success
    gate._update_gate()
    assert [enabled for enabled, _ in client.requests] == [True, False]
    client.reply(1, True)
    assert gate._uncertain_gate_mode is None
    assert gate._gate_mode is None
