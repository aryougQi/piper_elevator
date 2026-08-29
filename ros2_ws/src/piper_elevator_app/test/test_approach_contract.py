"""Contract tests for the first coarse button approach."""

from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_coarse_approach_constrains_position_but_not_orientation():
    config = yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'button_approach.yaml').read_text()
    )['button_approach_planner']['ros__parameters']

    assert config['end_effector_link'] == 'pika_fingertip_center_link'
    assert config['button_surface_pose_topic'] == '/button_surface_pose'
    assert config['close_gripper_before_plan'] is True
    assert config['closed_gripper_position_m'] == 0.0
    assert 'orientation_tolerance_rad' not in config
    assert config['pointing_tolerance_rad'] <= 0.14
    assert config['roll_tolerance_rad'] <= 0.26
    assert config['wrist_safe_joints'] == ['joint4', 'joint5', 'joint6']
    assert config['wrist_safe_centers_rad'][1] == -0.60
    assert config['wrist_safe_tolerances_rad'] == [1.20, 0.15, 1.50]
    joint5_minimum = abs(config['wrist_safe_centers_rad'][1]) - config[
        'wrist_safe_tolerances_rad'
    ][1]
    assert joint5_minimum >= config['minimum_abs_wrist_bend_rad']
    assert config['home_joint_names'] == [
        'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'
    ]
    assert config['home_joint_positions_rad'] == [0.0] * 6

    source = (
        PACKAGE_ROOT
        / 'piper_elevator_app'
        / 'button_approach_planner.py'
    ).read_text()
    assert "self._publish_status('CLOSING_GRIPPER')" in source
    assert "'pika_left_finger_joint'" in source
    assert "'pika_right_finger_joint'" in source
    assert 'OrientationConstraint' in source
    assert 'constraints.orientation_constraints' in source
    assert 'absolute_z_axis_tolerance' in source
    assert 'orientation_from_approach_direction' in source
    assert 'surface_normal_max_age_seconds' in source
    assert 'constraints.position_constraints = [position]' in source
    assert 'constraints.joint_constraints = ' in source
    assert "'~/return_home'" in source
    assert "constraints.name = 'home'" in source
    assert "self._publish_status('HOME_EXECUTING')" in source
    assert '_trajectory_wrist_is_safe' in source
    assert 'wrist singularity guard' in source
    assert 'DisplayTrajectory' in source
    assert '_publish_display_trajectory(result)' in source
    assert '_verify_approach_reached(target)' in source
    assert 'controller reported success but actual TCP missed the plan' in source
