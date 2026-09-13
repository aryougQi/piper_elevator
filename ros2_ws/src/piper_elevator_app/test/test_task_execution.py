"""Exercise task motion and recovery state without sending ROS requests."""

import threading
from types import SimpleNamespace

import pytest

from piper_elevator_app import elevator_task_manager as manager_module
from piper_elevator_app.elevator_task_manager import ElevatorTaskManager
from piper_elevator_app.elevator_task_manager import TaskFailure


class TaskHarness:
    _run_task = ElevatorTaskManager._run_task
    _recover = ElevatorTaskManager._recover
    _check_stopped = ElevatorTaskManager._check_stopped
    _seconds = ElevatorTaskManager._seconds
    _phase = ElevatorTaskManager._phase

    def __init__(self, failures=None, **parameters):
        self.parameters = {
            'return_home_before_task': True,
            'return_home_after_failure': True,
            'clear_selection_after_task': True,
            'home_timeout_seconds': 60.0,
            'planning_timeout_seconds': 30.0,
            'execution_timeout_seconds': 60.0,
            'visual_timeout_seconds': 120.0,
            'press_timeout_seconds': 30.0,
            'stop_timeout_seconds': 9.0,
            'recovery_retry_seconds': 5.0,
            **parameters,
        }
        self.failures = failures or {}
        self.calls = []
        self.statuses = []
        self.errors = []
        self.selections = []
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._busy = True
        self._task_sequence = 1
        self._active_button = '2'
        self._selection_publisher = SimpleNamespace(
            publish=lambda message: self.selections.append(message.data)
        )

    def get_parameter(self, name):
        return SimpleNamespace(value=self.parameters[name])

    def get_logger(self):
        return SimpleNamespace(error=self.errors.append)

    def _call_trigger(self, key, timeout, ignore_stop=False):
        self.calls.append(key)
        failure = self.failures.get((key, self.calls.count(key)))
        if failure is not None:
            raise TaskFailure(failure)
        return 'confirmed'

    def _wait_for_required_services(self):
        pass

    def _ensure_unique_nodes(self):
        pass

    def _select_and_wait_for_target(self, button):
        assert button == '2'

    def _wait_for_post_motion_target(self, button):
        self.calls.append('post_motion_target')

    def _start_and_wait_for_completion(self, prefix, timeout):
        self.calls.append(prefix)

    def _publish_status(self, status):
        self.statuses.append(status)

    def _publish_active_button(self, button):
        self.active_button = button

    def _publish_completion(self, completed):
        self.completed = completed

    def _publish_result(self, result):
        self.result = result


@pytest.mark.parametrize('failure', [
    'execute rejected: Motion completed; post-motion verification failed',
    'service timed out: execute',
])
def test_failed_execute_requires_confirmed_recovery_home(failure):
    task = TaskHarness({('execute', 1): failure})
    task._run_task(1, '2')

    assert task.calls == [
        'home', 'plan', 'execute', 'press_stop', 'visual_stop',
        'clear_plan', 'home',
    ]
    assert not task.completed
    assert failure in task.result
    assert 'home_reached=true' in task.result
    assert 'RECOVERING_HOME' in task.statuses
    assert not task._busy
    assert task.selections == ['']


def test_failed_recovery_cannot_report_previous_home_as_current():
    task = TaskHarness({
        ('execute', 1): 'execute post-motion verification failed',
        ('home', 2): 'home feedback did not settle',
    })
    task._run_task(1, '2')

    assert task.calls.count('home') == 2
    assert not task.completed
    assert 'home_reached=false' in task.result
    assert task.errors == [
        'Failed to recover home: home feedback did not settle'
    ]


def test_recovery_retries_while_the_planner_is_still_busy(monkeypatch):
    """An aborted execution releases the planner a moment later."""
    task = TaskHarness(
        {
            ('execute', 1): 'execute rejected: MoveIt execution error -4',
            ('home', 2): 'home rejected: Planner is busy',
        },
        recovery_retry_seconds=1.0,
    )
    monkeypatch.setattr(manager_module.time, 'sleep', lambda _: None)
    task._run_task(1, '2')

    assert task.calls.count('home') == 3
    assert 'home_reached=true' in task.result
    assert not task.errors


def test_disabled_recovery_does_not_claim_home_after_execute_failure():
    task = TaskHarness(
        {('execute', 1): 'execute post-motion verification failed'},
        return_home_after_failure=False,
    )
    task._run_task(1, '2')

    assert task.calls.count('home') == 1
    assert not task.completed
    assert 'home_reached=false' in task.result


def test_normal_task_completes_only_after_final_home():
    task = TaskHarness()
    task._run_task(1, '2')

    assert task.calls == [
        'home', 'plan', 'execute', 'post_motion_target', 'visual',
        'press', 'home',
    ]
    assert task.completed
    assert task.result == 'COMPLETE: button=2 pressed; home reached'
    assert not task._busy


def test_initial_home_failure_requires_another_confirmed_home():
    task = TaskHarness({('home', 1): 'initial home timed out'})
    task._run_task(1, '2')

    assert task.calls == [
        'home', 'press_stop', 'visual_stop', 'clear_plan', 'home',
    ]
    assert not task.completed
    assert 'home_reached=true' in task.result
    assert 'initial home timed out' in task.result


def test_planning_failure_preserves_confirmed_home_without_motion():
    task = TaskHarness({('plan', 1): 'no stable observation'})
    task._run_task(1, '2')

    assert task.calls == [
        'home', 'plan', 'press_stop', 'visual_stop', 'clear_plan',
    ]
    assert not task.completed
    assert 'home_reached=true' in task.result


@pytest.mark.parametrize('stop_service', ['press_stop', 'visual_stop'])
@pytest.mark.parametrize('at_home', [False, True])
def test_unconfirmed_stop_blocks_recovery_and_home_claim(stop_service, at_home):
    task = TaskHarness({(stop_service, 1): 'stop cleanup timed out'})

    assert not task._recover(at_home)
    assert task.calls == ['press_stop', 'visual_stop', 'clear_plan']
    assert any('home blocked' in message for message in task.errors)


class PostMotionHarness(TaskHarness):
    _selected_callback = ElevatorTaskManager._selected_callback
    _detection_callback = ElevatorTaskManager._detection_callback
    _surface_callback = ElevatorTaskManager._surface_callback

    def __init__(
        self, failures=None, *, selected='2', detected=True,
        fresh_detection=True, fresh_surfaces=3,
    ):
        super().__init__(
            failures,
            post_motion_target_wait_timeout_seconds=0.08,
            required_post_motion_surface_observations=3,
            selection_publish_period_seconds=1.0,
        )
        self._selected_button = '2'
        self._selected_sequence = 5
        self._detection_valid = True
        self._detection_sequence = 5
        self._surface_sequence = 5
        self._visual_status = 'IGNORED_TARGET_JUMP'
        self._visual_status_sequence = 5
        self.selected_response = selected
        self.detected = detected
        self.fresh_detection = fresh_detection
        self.fresh_surfaces = fresh_surfaces
        self._selection_publisher = SimpleNamespace(publish=self._receive_selection)

    def _receive_selection(self, message):
        self.selections.append(message.data)
        if not message.data:
            return
        self._selected_callback(SimpleNamespace(data=self.selected_response))
        if self.fresh_detection:
            self._detection_callback(SimpleNamespace(data=self.detected))
        for _ in range(self.fresh_surfaces):
            self._surface_callback(None)

    def _wait_for_post_motion_target(self, button):
        self.calls.append('post_motion_target')
        ElevatorTaskManager._wait_for_post_motion_target(self, button)


def test_old_visual_anchor_rejection_does_not_block_atomic_start_handoff():
    task = PostMotionHarness()
    task._run_task(1, '2')

    assert task._visual_status == 'IGNORED_TARGET_JUMP'
    assert task.calls == [
        'home', 'plan', 'execute', 'post_motion_target', 'visual',
        'press', 'home',
    ]
    assert task.completed


@pytest.mark.parametrize('failure', [
    'post-motion geometry verification failed', 'execute timed out',
])
def test_failed_execute_cannot_enter_visual_start_despite_available_new_frames(
    failure,
):
    task = PostMotionHarness({('execute', 1): failure})
    task._run_task(1, '2')

    assert 'post_motion_target' not in task.calls
    assert 'visual' not in task.calls
    assert not task.completed


@pytest.mark.parametrize('observation', [
    {'selected': '3'},
    {'detected': False},
    {'fresh_detection': False},
    {'fresh_surfaces': 0},
    {'fresh_surfaces': 2},
])
def test_post_motion_wait_still_requires_selected_valid_and_new_rgbd(observation):
    task = PostMotionHarness(**observation)

    with pytest.raises(TaskFailure, match='no fresh RGB-D target'):
        task._wait_for_post_motion_target('2')
