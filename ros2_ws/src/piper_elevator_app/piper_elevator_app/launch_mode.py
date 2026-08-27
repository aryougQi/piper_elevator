"""Pure launch policy for legacy and external ROS 2 control hardware."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MoveItLaunchMode:
    """Processes and description selected for one MoveIt launch mode."""

    description_file: str
    start_robot_state_publisher: bool
    start_ros2_control: bool
    start_controller_spawners: bool
    default_joint_states_topic: str


def select_moveit_launch_mode(external_hardware: bool) -> MoveItLaunchMode:
    """Return launch behavior without depending on the launch framework."""
    if external_hardware:
        return MoveItLaunchMode(
            description_file='piper_pika_description.urdf.xacro',
            start_robot_state_publisher=False,
            start_ros2_control=False,
            start_controller_spawners=False,
            default_joint_states_topic='/piper_pika/joint_states',
        )
    return MoveItLaunchMode(
        description_file='piper_pika.urdf.xacro',
        start_robot_state_publisher=True,
        start_ros2_control=True,
        start_controller_spawners=True,
        default_joint_states_topic='control/joint_states',
    )
