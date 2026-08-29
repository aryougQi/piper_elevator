"""Bring up the complete elevator task stack and its state machine."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.conditions import UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils.launch_utils import DeclareBooleanLaunchArg


def include(package, filename, arguments, condition=None):
    """Create a compact include action for one package launch file."""
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory(package),
                'launch',
                filename,
            )
        ),
        launch_arguments=arguments.items(),
        condition=condition,
    )


def generate_launch_description():
    app_share = get_package_share_directory('piper_elevator_app')
    simulation = LaunchConfiguration('simulation_mode')
    simulation_condition = IfCondition(simulation)
    real_condition = UnlessCondition(simulation)

    gazebo = include(
        'piper_elevator_gazebo',
        'gazebo_hardware.launch.py',
        {
            'gui': LaunchConfiguration('gazebo_gui'),
        },
        condition=simulation_condition,
    )
    simulation_detector = include(
        'piper_elevator_app',
        'button_detector.launch.py',
        {
            'use_sim_time': 'true',
            'confidence_threshold': LaunchConfiguration(
                'simulation_confidence_threshold'
            ),
            'simulation_layout_relabel': 'true',
        },
        condition=simulation_condition,
    )
    simulation_moveit = include(
        'piper_elevator_app',
        'piper_pika_moveit.launch.py',
        {
            'external_hardware': 'true',
            'use_sim_time': 'true',
            'use_rviz': LaunchConfiguration('use_rviz'),
            'start_pika_controller': 'false',
            'pika_tcp_offset': LaunchConfiguration('pika_tcp_offset'),
        },
        condition=simulation_condition,
    )
    simulation_planner = include(
        'piper_elevator_app',
        'button_approach_planner.launch.py',
        {
            'use_sim_time': 'true',
            'simulation_mode': 'true',
            'camera_calibration_valid': 'true',
            'allow_execution': 'true',
        },
        condition=simulation_condition,
    )
    simulation_visual = include(
        'piper_elevator_app',
        'button_visual_servo.launch.py',
        {
            'use_sim_time': 'true',
            'simulation_mode': 'true',
            'camera_calibration_valid': 'true',
            'allow_execution': 'true',
        },
        condition=simulation_condition,
    )
    simulation_press = include(
        'piper_elevator_app',
        'button_press.launch.py',
        {
            'use_sim_time': 'true',
            'simulation_mode': 'true',
            'allow_execution': 'true',
        },
        condition=simulation_condition,
    )

    real_stack = include(
        'piper_elevator_app',
        'button_approach_real.launch.py',
        {
            'can_port': LaunchConfiguration('can_port'),
            'pika_serial_port': LaunchConfiguration('pika_serial_port'),
            'speed_percent': LaunchConfiguration('speed_percent'),
            'pika_tcp_offset': LaunchConfiguration('pika_tcp_offset'),
            'use_rviz': LaunchConfiguration('use_rviz'),
            'start_camera': LaunchConfiguration('start_camera'),
            'start_pika_driver': LaunchConfiguration('start_pika_driver'),
            'auto_enable': LaunchConfiguration('auto_enable'),
            'hardware_commands_enabled': LaunchConfiguration(
                'hardware_commands_enabled'
            ),
            'publish_camera_tf': LaunchConfiguration('publish_camera_tf'),
            'camera_calibration_valid': LaunchConfiguration(
                'camera_calibration_valid'
            ),
            'allow_execution': LaunchConfiguration('allow_execution'),
            'camera_x': LaunchConfiguration('camera_x'),
            'camera_y': LaunchConfiguration('camera_y'),
            'camera_z': LaunchConfiguration('camera_z'),
            'camera_roll': LaunchConfiguration('camera_roll'),
            'camera_pitch': LaunchConfiguration('camera_pitch'),
            'camera_yaw': LaunchConfiguration('camera_yaw'),
        },
        condition=real_condition,
    )

    manager = Node(
        package='piper_elevator_app',
        executable='elevator_task_manager',
        name='elevator_task_manager',
        output='screen',
        parameters=[
            os.path.join(app_share, 'config', 'elevator_task.yaml'),
            {
                'use_sim_time': ParameterValue(
                    simulation,
                    value_type=bool,
                ),
            },
        ],
    )

    return LaunchDescription([
        DeclareBooleanLaunchArg('simulation_mode', default_value=True),
        DeclareBooleanLaunchArg('gazebo_gui', default_value=True),
        DeclareBooleanLaunchArg('use_rviz', default_value=True),
        DeclareLaunchArgument(
            'simulation_confidence_threshold',
            default_value='0.05',
        ),
        DeclareLaunchArgument('can_port', default_value='can0'),
        DeclareLaunchArgument(
            'pika_serial_port', default_value='/dev/ttyUSB60'
        ),
        DeclareLaunchArgument('speed_percent', default_value='10'),
        DeclareLaunchArgument(
            'pika_tcp_offset',
            default_value='[0.006, 0.0, 0.189, 0.0, 0.0, 0.0]',
        ),
        DeclareBooleanLaunchArg('start_camera', default_value=True),
        DeclareBooleanLaunchArg('start_pika_driver', default_value=False),
        DeclareBooleanLaunchArg('auto_enable', default_value=False),
        DeclareBooleanLaunchArg(
            'hardware_commands_enabled', default_value=False
        ),
        DeclareBooleanLaunchArg('publish_camera_tf', default_value=False),
        DeclareBooleanLaunchArg(
            'camera_calibration_valid', default_value=False
        ),
        DeclareBooleanLaunchArg('allow_execution', default_value=False),
        DeclareLaunchArgument('camera_x', default_value='0.0'),
        DeclareLaunchArgument('camera_y', default_value='0.0'),
        DeclareLaunchArgument('camera_z', default_value='0.0'),
        DeclareLaunchArgument('camera_roll', default_value='0.0'),
        DeclareLaunchArgument('camera_pitch', default_value='0.0'),
        DeclareLaunchArgument('camera_yaw', default_value='0.0'),
        gazebo,
        simulation_detector,
        simulation_moveit,
        simulation_planner,
        simulation_visual,
        simulation_press,
        real_stack,
        manager,
    ])
