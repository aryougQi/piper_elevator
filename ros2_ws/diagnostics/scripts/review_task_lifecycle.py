"""Offline reproductions for the 2026-09-13 review; no ROS requests sent."""

from pathlib import Path
import runpy
import threading
import time
from types import SimpleNamespace

from piper_elevator_app.elevator_task_manager import ElevatorTaskManager, TaskFailure


def reproduce_stop():
    sent = []
    started = time.monotonic()
    future = SimpleNamespace(
        done=lambda: time.monotonic() - started >= 0.15,
        cancelled=lambda: False,
        exception=lambda: None,
        result=lambda: SimpleNamespace(success=True, message='motion finished'),
    )
    stop = threading.Event()
    stop.set()
    harness = SimpleNamespace(
        _condition=threading.Condition(),
        _stop_event=stop,
        _trigger_clients={'execute': SimpleNamespace(
            wait_for_service=lambda **kwargs: True,
            call_async=lambda request: sent.append('execute') or future,
        )},
    )
    harness._check_stopped = lambda: ElevatorTaskManager._check_stopped(harness)
    try:
        ElevatorTaskManager._call_trigger(harness, 'execute', 1.0)
    except TaskFailure as error:
        elapsed = time.monotonic() - started
        assert sent == ['execute'] and elapsed >= 0.15
        print(f'Stop already set, but execute sent and awaited {elapsed:.3f}s: {error}')


def reproduce_terminal_race():
    workspace = Path(__file__).resolve().parents[2]
    test = workspace / 'src/piper_elevator_app/test/test_task_execution.py'
    base = runpy.run_path(str(test))['TaskHarness']

    class Harness(base):
        def _publish_active_button(self, button):
            # Schedule a new command at the first publication after the old
            # task releases its lock. This is an allowed callback interleaving.
            if button == '' and not self._busy:
                self._busy = True
                self._task_sequence += 1
                self._active_button = '3'
                self._publish_completion(False)
                self._publish_result('RUNNING: button=3')
            super()._publish_active_button(button)

    harness = Harness()
    harness._run_task(1, '2')
    assert harness._busy and harness._active_button == '3'
    assert harness.completed and harness.result.startswith('COMPLETE: button=2')
    print(f'New task active={harness._active_button}; completed={harness.completed}; '
          f'published active={harness.active_button!r}; result={harness.result}')


def reproduce_repeated_tf_contact():
    from piper_elevator_app.button_press_executor import ButtonPressExecutor
    from piper_elevator_app.press_core import StallContactDetector

    statuses = []
    reads = []
    parameters = {'simulation_mode': False, 'maximum_approach_travel_m': 0.038}
    harness = SimpleNamespace(
        _stop_event=threading.Event(),
        _motion_state_stamp_ns=1_000_000_000,
        get_parameter=lambda key: SimpleNamespace(value=parameters[key]),
        _start_timed_phase=lambda *args: None,
        _motion_timeout_seconds=lambda: 15.0,
        _geometry_press_enabled=lambda: False,
        _stall_detection_enabled=lambda: True,
        _make_stall_detector=StallContactDetector,
        _motion_speed=lambda key: 0.010,
        # Reuse one still-valid TF frame at 50 Hz. Six reads span 100 ms,
        # below the configured 250 ms TF freshness threshold.
        _guard_motion=lambda *args: reads.append(1) or (None, 0.006),
        _publish_zero_twist=lambda: None,
        _publish_status=statuses.append,
        _guard_deadline=lambda *args: None,
        _refresh_gate_or_raise=lambda: None,
        _publish_smoothed_linear=lambda *args: None,
        _line_tracking_command=lambda *args: None,
        _wait_period=lambda: None,
    )
    travel = ButtonPressExecutor._approach_until_contact(harness, None, None, None)
    assert len(reads) == 6 and travel == 0.006
    assert statuses[0].startswith('CONTACT_DETECTED_BY_STALL')
    print(f'One TF frame reused {len(reads)} times: {statuses[0]}')


if __name__ == '__main__':
    reproduce_stop()
    reproduce_terminal_race()
    reproduce_repeated_tf_contact()
