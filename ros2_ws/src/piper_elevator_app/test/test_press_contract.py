"""Configuration and integration contracts for guarded button pressing."""

from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[1]


def test_real_press_is_disabled_until_six_joint_calibration():
    config = yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'button_press.yaml').read_text()
    )['button_press_executor']['ros__parameters']

    assert config['visual_completion_topic'] == (
        '/button_visual_servo/completed'
    )
    assert config['servo_claim_topic'] == '/button_press/servo_claimed'
    assert config['continuous_servo_handoff'] is True
    assert config['timing_topic'] == '/button_press/timing'
    assert config['effort_topic'] == '/feedback/joint_states'
    assert config['button_selected_topic'] == '/button_selected'
    assert config['simulation_button_joint_topic'] == (
        '/elevator_button/joint_states'
    )
    buttons = ['1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm']
    assert config['simulation_button_names'] == buttons
    assert config['simulation_button_joint_names'] == [
        f'button_{button}_press_joint' for button in buttons
    ]
    assert config['simulation_contacts_topics'] == [
        f'/elevator_button/button_{button}/contacts' for button in buttons
    ]
    assert config['arm_joint_names'] == [
        'joint1',
        'joint2',
        'joint3',
        'joint4',
        'joint5',
        'joint6',
    ]
    assert config['torque_thresholds_calibrated'] is False
    assert config['allow_execution'] is False
    assert len(config['joint_torque_delta_thresholds_nm']) == 6
    assert len(config['joint_torque_absolute_limits_nm']) == 6


def test_press_motion_has_bounded_contact_and_retract_contract():
    config = yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'button_press.yaml').read_text()
    )['button_press_executor']['ros__parameters']
    source = (
        PACKAGE_ROOT
        / 'piper_elevator_app'
        / 'button_press_executor.py'
    ).read_text()

    assert config['maximum_approach_travel_m'] <= 0.04
    assert config['motion_timeout_seconds'] <= 15.0
    assert config['approach_speed_mps'] <= 0.01
    assert config['press_speed_mps'] < config['approach_speed_mps']
    assert config['retract_speed_mps'] <= 0.03
    assert config['retract_slow_speed_mps'] < config['retract_speed_mps']
    assert config['retract_slowdown_distance_m'] <= 0.003
    assert config['maximum_acceleration_mps2'] <= 0.12
    assert config['press_extension_m'] <= 0.003
    assert config['hold_seconds'] >= 0.25
    assert config['maximum_lateral_drift_m'] <= 0.003
    assert config['lateral_correction_gain'] > 0.0
    assert config['maximum_lateral_correction_speed_mps'] <= 0.006
    assert config['retract_tolerance_m'] <= 0.001
    assert config['simulation_retract_tolerance_m'] <= 0.0008
    assert config['contact_consecutive_samples'] >= 3
    assert config['emergency_threshold_multiplier'] > 1.0
    assert '_collect_torque_baseline' in source
    assert '_approach_until_contact' in source
    assert '_line_tracking_command' in source
    assert '_retract_to_start' in source
    assert 'slow_speed if distance <= slowdown_distance else fast_speed' in (
        source
    )
    retract_gain_source = (
        'error\n'
        '                * 2.0\n'
        '                * self._simulation_motion_multiplier()'
    )
    assert retract_gain_source in source
    assert 'retract timed out:' in source
    assert 'remaining={distance * 1000.0:.1f}mm' in source
    assert '_settle_servo_origin' in source
    assert config['servo_settle_required_samples'] >= 5
    assert config['servo_settle_position_tolerance_m'] <= 0.0002
    assert 'position_delta <= position_tolerance' in source
    assert 'direction_delta <= direction_tolerance' in source
    assert '_pause_moveit_servo(wait=True)' in source
    assert '_publish_servo_claim(True)' in source
    assert 'do not restart/unpause it' in source
    assert '_servo_safety_failure' in source
    assert config['servo_status_topic'] == '/servo_node/status'
    assert '_set_hardware_servo_gate' in source
    assert '_press_simulated_button' in source
    assert '_publish_timing' in source
    assert 'simulation_pressed_depth_m' in source


def test_simulation_requires_real_gazebo_contact_and_button_travel():
    config = yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'button_press.yaml').read_text()
    )['button_press_executor']['ros__parameters']
    source = (
        PACKAGE_ROOT
        / 'piper_elevator_app'
        / 'button_press_executor.py'
    ).read_text()

    assert 0.001 <= config['simulation_pressed_depth_m'] <= 0.004
    assert config['simulation_release_tolerance_m'] < (
        config['simulation_pressed_depth_m']
    )
    assert 1.0 <= config['simulation_speed_multiplier'] <= 10.0
    assert 'simulation_motion_timeout_seconds' not in config
    assert 'simulation_retract_timeout_seconds' not in config
    assert 'simulated_contact_travel_m' not in config
    assert '_button_selection_callback' in source
    assert '_simulation_contacts_callback' in source
    assert '_simulation_true_contacts' in source
    assert '_simulation_button_joint_callback' in source
    assert '_active_simulation_button' in source
    assert 'BUTTON_DEPRESSED' in source
    assert 'BUTTON_RELEASED' in source
    assert 'RETRACTING remaining=' in source


def test_visual_servo_exposes_latched_completion_handshake():
    visual_config = yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'button_visual_servo.yaml').read_text()
    )['button_visual_servo']['ros__parameters']
    visual_source = (
        PACKAGE_ROOT
        / 'piper_elevator_app'
        / 'button_visual_servo.py'
    ).read_text()

    assert visual_config['completion_topic'] == (
        '/button_visual_servo/completed'
    )
    assert '_publish_completion(True)' in visual_source
    assert '_hold_for_press_claim' in visual_source


def test_control_gate_has_crash_safe_servo_heartbeat():
    gate = (
        PACKAGE_ROOT / 'piper_elevator_app' / 'control_gate.py'
    ).read_text()

    assert 'servo_authorization_service' in gate
    assert 'servo_heartbeat_timeout_seconds' in gate
    assert 'self._trajectory_active or self._servo_authorized' in gate
