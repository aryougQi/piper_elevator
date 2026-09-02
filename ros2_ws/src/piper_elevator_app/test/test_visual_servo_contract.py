"""Configuration and integration contracts for fine visual servoing."""

import math
from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[1]


def test_visual_servo_has_safe_three_cm_contract():
    config = yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'button_visual_servo.yaml').read_text()
    )['button_visual_servo']['ros__parameters']

    assert config['surface_pose_topic'] == '/button_surface_pose'
    assert config['camera_frame'] == 'camera_color_optical_frame'
    assert config['end_effector_link'] == 'pika_fingertip_center_link'
    assert config['standoff_distance_m'] == 0.030
    assert config['required_alignment_observations'] >= 2
    assert config['required_locked_alignment_cycles'] >= 5
    assert config['required_post_orientation_observations'] >= 3
    assert config['reacquisition_search_enabled'] is True
    assert config['reacquisition_initial_hold_seconds'] >= 0.5
    assert config['reacquisition_search_radius_m'] <= 0.012
    assert config['reacquisition_search_speed_mps'] <= 0.012
    assert config['reacquisition_maximum_axial_drift_m'] <= 0.002
    assert config['minimum_standoff_m'] <= config['standoff_distance_m']
    assert (
        config['standoff_distance_m'] - config['distance_tolerance_m']
        >= config['minimum_standoff_m']
    )
    assert config['press_servo_claim_topic'] == (
        '/button_press/servo_claimed'
    )
    assert config['press_claim_timeout_seconds'] <= 3.0
    assert config['maximum_linear_speed_mps'] <= 0.08
    assert config['maximum_linear_acceleration_mps2'] <= 0.30
    assert (
        0.0 < config['axial_approach_full_speed_angle_rad']
        < config['axial_approach_stop_angle_rad']
        <= config['perpendicular_tolerance_rad']
    )
    assert 1.0 <= config['simulation_linear_speed_multiplier'] <= 10.0
    assert config['level_roll_enabled'] is True
    assert config['orientation_control_enabled'] is True
    assert config['maximum_target_jump_m'] <= 0.015
    assert config['button_selection_topic'] == '/button_selection'

    source = (
        PACKAGE_ROOT
        / 'piper_elevator_app'
        / 'button_visual_servo.py'
    ).read_text()
    assert 'message_stamp_ns < self._selection_changed_stamp_ns' in source
    assert 'button_base - self._observation_anchor' in source
    assert 'target_roll >= roll_tolerance' in source
    assert config['level_reference_axis'] == [0.0, 0.0, 1.0]
    assert config['target_level_roll_rad'] == 0.0
    assert (
        config['target_level_roll_rad']
        < config['level_roll_tolerance_rad']
        <= math.radians(3.0)
    )
    assert config['maximum_level_roll_speed_radps'] <= 0.30
    assert config['maximum_angular_speed_radps'] <= 0.35
    assert config['maximum_angular_acceleration_radps2'] <= 1.20
    assert 'final_planning_pipeline' not in config
    assert 'visual_handoff_distance_m' not in config
    assert config['allow_execution'] is False
    assert config['camera_calibration_valid'] is False


def test_detector_publishes_only_fitted_surface_pose_for_servo():
    config = yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'button_visual_servo.yaml').read_text()
    )['button_visual_servo']['ros__parameters']
    detector = (
        PACKAGE_ROOT
        / 'piper_elevator_app'
        / 'yolo_button_detector.py'
    ).read_text()
    servo = (
        PACKAGE_ROOT
        / 'piper_elevator_app'
        / 'button_visual_servo.py'
    ).read_text()

    assert 'estimate_surface_normal(' in detector
    assert '_publish_surface_pose(' in detector
    assert 'suppress_incomplete_layout=(' in detector
    assert "'~/start'" in servo
    assert "'~/stop'" in servo
    assert '_publish_twist(' in servo
    assert '_linear_speed_multiplier' in servo
    assert "'vision_limit'" not in servo
    assert "'final_planning_pipeline'" not in servo
    assert 'required_stable_observations' in servo
    assert 'tool_orientation_for_camera_direction' in servo
    assert 'camera_level_roll_error' in servo
    assert 'roll={self._roll_status(level_roll)}' in servo
    assert 'always request an explicit unpause' in servo
    assert "'servo_status_topic'" in servo
    assert '_servo_safety_failure' in servo
    assert 'halted at a singularity (status=2)' in servo
    assert 'IGNORED_TARGET_JUMP:' in servo
    assert 'VISION_LOSS_CONTINUING' in servo
    assert 'VISION_LOSS_HOLDING' in servo
    assert 'orientation_prioritized_linear_command(' in servo
    assert "servo_phase = 'ORIENTING'" in servo
    assert "servo_phase = 'REACQUIRING'" in servo
    assert "servo_phase = 'FINAL_APPROACH'" in servo
    assert "desired_linear = np.zeros(3)" in servo
    assert '_roll_within_tolerance(level_roll)' in servo
    assert 'FINAL_APPROACH_ORIENTATION_GUARD' in servo
    assert config['expected_observation_gap_seconds'] <= 0.20
    assert config['vision_loss_continuation_seconds'] <= 8.0
    assert config['vision_loss_speed_scale'] <= 0.50
    assert config['vision_loss_continuation_max_distance_m'] <= 0.10
    assert config['vision_loss_max_travel_m'] <= 0.080
    assert config['required_locked_target_stable_cycles'] >= 5
    assert "if using_locked_observation:" in servo
    assert 'locked_target_stable_cycles' in servo
    assert '_track_visually(\n                deadline,\n                initial,' in servo
    assert '_plan_final_pose_with_retries' not in servo
    tracking_source = servo[
        servo.index('    def _track_visually('):
        servo.index('    @staticmethod\n    def _limit_vector')
    ]
    assert '_decelerate_servo_to_hold' not in tracking_source
    assert 'holding level pose for fresh RGB-D reacquisition' in (
        tracking_source
    )
    assert 'PHASE_COMPLETE: REACQUIRING' in tracking_source
    assert 'tangential_spiral_offset(' in tracking_source
    assert 'reacquisition search exceeded axial drift limit' in (
        tracking_source
    )
    assert 'desired_linear -= np.dot(' in tracking_source
    assert "elif servo_phase == 'REACQUIRING':" in tracking_source
    assert '_decelerate_servo_to_hold' in servo
    assert '_hold_for_press_claim' in servo
    assert 'COMPLETE: continuous Servo alignment' in servo
    assert 'CARTESIAN_HANDOFF' not in servo


def test_moveit_servo_is_smoothed_and_starts_with_moveit():
    servo_config = yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'piper_pika_servo.yaml').read_text()
    )
    moveit_launch = (
        PACKAGE_ROOT
        / 'launch'
        / 'piper_pika_moveit.launch.py'
    ).read_text()

    assert servo_config['command_in_type'] == 'speed_units'
    assert servo_config['publish_period'] <= 0.02
    assert 'ButterworthFilterPlugin' in servo_config[
        'smoothing_filter_plugin_name'
    ]
    assert servo_config['command_out_topic'] == (
        '/arm_controller/joint_trajectory'
    )
    assert servo_config['lower_singularity_threshold'] == 55.0
    assert servo_config['hard_stop_singularity_threshold'] == 100.0
    assert servo_config['leaving_singularity_threshold_multiplier'] == 2.0
    assert 'servo_node_main' in moveit_launch
    assert "'start_moveit_servo'" in moveit_launch
    assert "'/servo_node/raw_joint_trajectory'" in moveit_launch
    assert "executable='simulation_servo_adapter'" in moveit_launch


def test_simulation_servo_adapter_is_bounded_and_simulation_only():
    adapter_config = yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'simulation_servo_adapter.yaml').read_text()
    )['simulation_servo_adapter']['ros__parameters']
    launch_source = (
        PACKAGE_ROOT / 'launch' / 'piper_pika_moveit.launch.py'
    ).read_text()

    assert 1.0 < adapter_config['position_gain'] <= 3.0
    assert 0.0 < adapter_config['maximum_lead_rad'] <= 0.10
    assert adapter_config['joint_state_timeout_seconds'] <= 0.20
    assert len(adapter_config['joint_names']) == 6
    assert len(adapter_config['lower_limits_rad']) == 6
    assert len(adapter_config['upper_limits_rad']) == 6
    assert 'if simulation:' in launch_source
    assert 'simulation_servo_adapter.yaml' in launch_source
