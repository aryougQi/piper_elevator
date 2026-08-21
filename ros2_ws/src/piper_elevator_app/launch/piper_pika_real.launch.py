import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils.launch_utils import DeclareBooleanLaunchArg


def generate_launch_description():
    app_share = get_package_share_directory('piper_elevator_app')
    arm_ctrl_share = get_package_share_directory('agx_arm_ctrl')

    arm_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                arm_ctrl_share,
                'launch',
                'start_single_agx_arm.launch.py',
            )
        ),
        launch_arguments={
            'can_port': LaunchConfiguration('can_port'),
            'arm_type': 'piper',
            'effector_type': 'none',
            'auto_enable': LaunchConfiguration('auto_enable'),
            'speed_percent': LaunchConfiguration('speed_percent'),
            'tcp_offset': LaunchConfiguration('pika_tcp_offset'),
            # Commands remain blocked unless the action-aware gate below runs.
            'control_enabled': 'false',
        }.items(),
    )

    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                app_share,
                'launch',
                'piper_pika_moveit.launch.py',
            )
        ),
        launch_arguments={
            'use_rviz': LaunchConfiguration('use_rviz'),
            'pika_tcp_offset': LaunchConfiguration('pika_tcp_offset'),
            'joint_states_topic': '/piper_pika/joint_states',
            'controller_output_topic': '/control/joint_states',
            # The current task does not actuate the Pika jaws on real hardware.
            'start_pika_controller': 'false',
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('can_port', default_value='can0'),
        DeclareLaunchArgument(
            'pika_serial_port', default_value='/dev/ttyUSB60'
        ),
        DeclareLaunchArgument('speed_percent', default_value='10'),
        DeclareLaunchArgument(
            'pika_tcp_offset',
            default_value='[0.006, 0.0, 0.189, 0.0, 0.0, 0.0]',
        ),
        DeclareBooleanLaunchArg('use_rviz', default_value=True),
        DeclareBooleanLaunchArg('auto_enable', default_value=False),
        DeclareBooleanLaunchArg(
            'hardware_commands_enabled',
            default_value=False,
            description=(
                'Allow the action-aware control gate to forward trajectories.'
            ),
        ),
        DeclareBooleanLaunchArg(
            'start_pika_driver',
            default_value=False,
            description=(
                'Read the real Pika center_joint from its serial port.'
            ),
        ),
        arm_driver,
        Node(
            package='piper_elevator_app',
            executable='piper_pika_joint_state_mux',
            output='screen',
            parameters=[{
                'arm_topic': '/feedback/joint_states',
                'gripper_topic': '/gripper/joint_state',
                'output_topic': '/piper_pika/joint_states',
                'gripper_joint_name': 'center_joint',
                'default_gripper_position': 0.0,
            }],
        ),
        Node(
            package='sensor_tools',
            executable='serial_gripper_imu',
            name='pika_serial_gripper',
            output='screen',
            parameters=[{
                'serial_port': LaunchConfiguration('pika_serial_port'),
                'joint_name': 'center_joint',
                'motor_current_limit': 1000.0,
                'motor_current_redundancy': 500.0,
                'mit_mode': True,
                'ctrl_rate': 50.0,
            }],
            remappings=[
                ('/gripper/joint_state', '/gripper/joint_state'),
                (
                    '/gripper/joint_state_ctrl',
                    '/pika_gripper/control_disabled',
                ),
                ('/joint_state_info', '/pika_gripper/state_info_unused'),
                ('/joint_state_gripper', '/pika_gripper/state_unused'),
            ],
            condition=IfCondition(LaunchConfiguration('start_pika_driver')),
        ),
        moveit,
        Node(
            package='piper_elevator_app',
            executable='piper_pika_control_gate',
            name='piper_pika_control_gate',
            output='screen',
            parameters=[{
                'status_topic': (
                    '/arm_controller/'
                    'follow_joint_trajectory/_action/status'
                ),
                'gate_service_name': '/control_enable',
                'status_timeout_seconds': 1.0,
            }],
            condition=IfCondition(
                LaunchConfiguration('hardware_commands_enabled')
            ),
        ),
    ])
