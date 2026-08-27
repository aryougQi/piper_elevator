"""Contract tests for the first coarse button approach."""

from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_coarse_approach_uses_closed_fingertip_center_and_locked_roll():
    config = yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'button_approach.yaml').read_text()
    )['button_approach_planner']['ros__parameters']

    assert config['end_effector_link'] == 'pika_fingertip_center_link'
    assert config['close_gripper_before_plan'] is True
    assert config['closed_gripper_position_m'] == 0.0
    assert config['orientation_tolerance_rad'] <= 0.02

    source = (
        PACKAGE_ROOT
        / 'piper_elevator_app'
        / 'button_approach_planner.py'
    ).read_text()
    assert "self._publish_status('CLOSING_GRIPPER')" in source
    assert "'pika_left_finger_joint'" in source
    assert "'pika_right_finger_joint'" in source
    assert 'tool_rotation = tool_transform.transform.rotation' in source
    assert 'orientation = self._orientation_constraint(target)' in source
