"""Static integration contracts for the elevator task state machine."""

from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[1]


def test_task_manager_wires_every_motion_stage_and_safe_home():
    config = yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'elevator_task.yaml').read_text()
    )['elevator_task_manager']['ros__parameters']
    source = (
        PACKAGE_ROOT
        / 'piper_elevator_app'
        / 'elevator_task_manager.py'
    ).read_text()

    assert config['button_selection_topic'] == '/button_selection'
    assert config['plan_service'].endswith('/plan')
    assert config['execute_service'].endswith('/execute')
    assert config['visual_start_service'].endswith('/start')
    assert config['press_start_service'].endswith('/start')
    assert config['home_service'].endswith('/return_home')
    assert config['return_home_before_task'] is True
    assert config['return_home_after_failure'] is True
    assert config['clear_selection_after_task'] is True
    assert config['post_motion_target_wait_timeout_seconds'] > 0.0
    assert config['required_unique_nodes'] == [
        '/button_detector',
        '/button_approach_planner',
        '/button_visual_servo',
        '/button_press_executor',
    ]
    assert "self._phase('COARSE_PLANNING'" in source
    assert "self._phase('WAITING_FOR_VISUAL_TARGET'" in source
    assert "self._phase('VISUAL_SERVO'" in source
    assert "self._phase('PRESSING'" in source
    assert "self._phase('HOMING_FINAL'" in source
    assert '_recover(at_home)' in source
    assert '_ensure_unique_nodes()' in source
    assert '_wait_for_post_motion_target(button)' in source
    assert 'self._surface_sequence > surface_baseline' in source


def test_task_outputs_are_latched_and_completion_is_sequence_guarded():
    source = (
        PACKAGE_ROOT
        / 'piper_elevator_app'
        / 'elevator_task_manager.py'
    ).read_text()

    assert 'DurabilityPolicy.TRANSIENT_LOCAL' in source
    assert '_visual_completion_sequence' in source
    assert '_press_completion_sequence' in source
    assert '> completion_baseline' in source
    assert 'self._publish_completion(success and at_home)' in source
    assert 'self._clients =' not in source
    assert 'self._trigger_clients =' in source


def test_top_level_launch_contains_complete_simulation_stack():
    launch_source = (
        PACKAGE_ROOT / 'launch' / 'elevator_task.launch.py'
    ).read_text()

    for component in (
        'gazebo_hardware.launch.py',
        'button_detector.launch.py',
        'piper_pika_moveit.launch.py',
        'button_approach_planner.launch.py',
        'button_visual_servo.launch.py',
        'button_press.launch.py',
        "executable='elevator_task_manager'",
    ):
        assert component in launch_source
