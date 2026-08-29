import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    app_share = get_package_share_directory('piper_elevator_app')
    parameters = os.path.join(app_share, 'config', 'button_press.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('simulation_mode', default_value='false'),
        DeclareLaunchArgument('allow_execution', default_value='false'),
        DeclareLaunchArgument(
            'hardware_gate_required',
            default_value='false',
        ),
        Node(
            package='piper_elevator_app',
            executable='button_press_executor',
            name='button_press_executor',
            output='screen',
            parameters=[
                parameters,
                {
                    'simulation_mode': ParameterValue(
                        LaunchConfiguration('simulation_mode'),
                        value_type=bool,
                    ),
                    'allow_execution': ParameterValue(
                        LaunchConfiguration('allow_execution'),
                        value_type=bool,
                    ),
                    'hardware_gate_required': ParameterValue(
                        LaunchConfiguration('hardware_gate_required'),
                        value_type=bool,
                    ),
                    'use_sim_time': ParameterValue(
                        LaunchConfiguration('use_sim_time'),
                        value_type=bool,
                    ),
                },
            ],
        ),
    ])
