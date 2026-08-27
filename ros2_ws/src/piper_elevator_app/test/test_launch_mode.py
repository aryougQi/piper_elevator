"""Unit tests for legacy and external-hardware MoveIt policy."""

from piper_elevator_app.launch_mode import select_moveit_launch_mode


def test_external_hardware_uses_description_only():
    mode = select_moveit_launch_mode(True)

    assert mode.description_file == 'piper_pika_description.urdf.xacro'
    assert mode.start_robot_state_publisher is False
    assert mode.start_ros2_control is False
    assert mode.start_controller_spawners is False
    assert mode.default_joint_states_topic == '/piper_pika/joint_states'


def test_legacy_mode_preserves_existing_proxy():
    mode = select_moveit_launch_mode(False)

    assert mode.description_file == 'piper_pika.urdf.xacro'
    assert mode.start_robot_state_publisher is True
    assert mode.start_ros2_control is True
    assert mode.start_controller_spawners is True
    assert mode.default_joint_states_topic == 'control/joint_states'
