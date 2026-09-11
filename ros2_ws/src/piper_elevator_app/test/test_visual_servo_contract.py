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
    assert config['standoff_distance_m'] == 0.03
    assert config['minimum_standoff_m'] <= config['standoff_distance_m']
    assert config['visual_handoff_distance_m'] > config[
        'standoff_distance_m'
    ]
    assert config['vision_loss_handoff_max_distance_m'] >= config[
        'visual_handoff_distance_m'
    ]
    assert config['maximum_final_cartesian_distance_m'] <= 0.065
    assert config['maximum_linear_speed_mps'] <= 0.08
    assert config['maximum_linear_acceleration_mps2'] <= 0.30
    assert config['level_roll_enabled'] is True
    assert config['level_reference_axis'] == [0.0, 0.0, 1.0]
    assert math.radians(5.0) <= config['maximum_level_roll_rad'] <= (
        math.radians(15.0)
    )
    assert config['maximum_level_roll_speed_radps'] <= 0.20
    assert config['final_planning_pipeline'] == (
        'pilz_industrial_motion_planner'
    )
    assert config['final_planner_id'] == 'LIN'
    assert config['allow_execution'] is False
    assert config['camera_calibration_valid'] is False


def test_detector_publishes_only_fitted_surface_pose_for_servo():
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
    assert "'~/start'" in servo
    assert "'~/stop'" in servo
    assert '_publish_twist(' in servo
    assert "'vision_limit'" in servo
    assert "'final_planning_pipeline'" in servo
    assert 'required_stable_observations' in servo
    assert 'tool_orientation_for_camera_direction' in servo
    assert 'camera_level_roll_error' in servo
    assert 'roll={self._roll_status(level_roll)}' in servo
    assert 'always request an explicit unpause' in servo
    assert "'servo_status_topic'" in servo
    assert '_servo_safety_failure' in servo
    assert 'halted at a singularity (status=2)' in servo


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
    assert 'servo_node_main' in moveit_launch
    assert "'start_moveit_servo'" in moveit_launch
