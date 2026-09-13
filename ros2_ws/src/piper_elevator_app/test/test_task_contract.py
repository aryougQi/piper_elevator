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
    assert config['approach_status_topic'] == '/button_approach/status'
    assert config['plan_service'].endswith('/plan')
    assert config['execute_service'].endswith('/execute')
    assert config['visual_start_service'].endswith('/start')
    assert config['press_start_service'].endswith('/start')
    assert config['home_service'].endswith('/return_home')
    assert config['return_home_before_task'] is True
    assert config['return_home_after_failure'] is True
    assert config['clear_selection_after_task'] is True
    assert config['post_motion_target_wait_timeout_seconds'] > 0.0
    assert config['post_motion_target_wait_timeout_seconds'] <= 5.0
    assert config['required_post_motion_surface_observations'] >= 3
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
    assert 'surface_baseline + required_surfaces' in source
    assert "self._approach_status == 'TARGET_READY'" in source
    post_motion_source = source.split(
        'def _wait_for_post_motion_target', 1
    )[1].split('def _start_and_wait_for_completion', 1)[0]
    assert '_visual_status' not in post_motion_source
    assert 'Visual start atomically claims that planner-approved target' in (
        post_motion_source
    )


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
    simulation_planner_block = launch_source.split(
        "simulation_planner = include(",
        1,
    )[1].split(
        "simulation_visual = include(",
        1,
    )[0]
    assert "'publish_camera_tf': 'false'" in simulation_planner_block


def test_real_launch_uses_feedback_and_trajectory_proxy():
    moveit_source = (
        PACKAGE_ROOT / 'launch' / 'piper_pika_real.launch.py'
    ).read_text()
    real_source = (
        PACKAGE_ROOT / 'launch' / 'button_approach_real.launch.py'
    ).read_text()

    assert "'external_hardware': 'false'" in moveit_source
    assert "'joint_states_topic': '/piper_pika/joint_states'" in moveit_source
    assert "'enable_timeout': LaunchConfiguration('enable_timeout')" in (
        moveit_source
    )
    assert "DeclareLaunchArgument('enable_timeout', default_value='15.0')" in (
        moveit_source
    )
    visual_block = real_source.split(
        'visual_servo = IncludeLaunchDescription',
        1,
    )[1].split(
        'button_press = IncludeLaunchDescription',
        1,
    )[0]
    assert "'hardware_gate_required': 'true'" in visual_block
    assert "'servo_gate_service_name': '/servo_control_enable'" in (
        moveit_source
    )


def test_real_driver_separates_planned_and_streaming_control_modes():
    driver = (
        PACKAGE_ROOT.parent
        / 'agx_arm_ros'
        / 'src'
        / 'agx_arm_ctrl'
        / 'agx_arm_ctrl'
        / 'agx_arm_ctrl_single_node.py'
    ).read_text()
    gate = (
        PACKAGE_ROOT
        / 'piper_elevator_app'
        / 'control_gate.py'
    ).read_text()

    assert '"servo_control_enable"' in driver
    assert 'motion_mode="j"' in driver
    assert 'motion_mode="js"' in driver
    assert 'self._external_control_mode == "servo"' in driver
    assert "'trajectory': self.create_client(" in gate
    assert "'servo': self.create_client(" in gate
    assert 'Servo authorization rejected while trajectory control' in gate


def test_real_camera_is_pinned_to_validated_device():
    camera_launch = (
        PACKAGE_ROOT / 'launch' / 'realsense_button_detector.launch.py'
    ).read_text()
    real_launch = (
        PACKAGE_ROOT / 'launch' / 'button_approach_real.launch.py'
    ).read_text()

    assert "default_value='_315122272433'" in camera_launch
    assert "'serial_no': LaunchConfiguration('camera_serial_no')" in (
        camera_launch
    )
    assert "'enable_sync': 'true'" in camera_launch
    assert "'depth_module.color_profile': '848x480x30'" in camera_launch
    assert "'depth_module.depth_profile': '848x480x30'" in camera_launch
    assert "default_value='_315122272433'" in real_launch
    calibrated_defaults = (
        ('camera_x', '-0.0525784297'),
        ('camera_y', '0.0004861476'),
        ('camera_z', '-0.1399236010'),
        ('camera_roll', '-1.2709909926'),
        ('camera_pitch', '-1.5255574198'),
        ('camera_yaw', '1.2243363470'),
    )
    top_level = (
        PACKAGE_ROOT / 'launch' / 'elevator_task.launch.py'
    ).read_text()
    for name, value in calibrated_defaults:
        declaration = (
            f"DeclareLaunchArgument('{name}', default_value='{value}')"
        )
        assert declaration in real_launch
        assert declaration in top_level
