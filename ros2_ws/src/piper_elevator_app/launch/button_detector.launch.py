from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import os


def generate_launch_description():
    package_share = get_package_share_directory('piper_elevator_app')
    parameters = os.path.join(package_share, 'config', 'button_detector.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        # This launch is used by Gazebo. RealSense launch loads the 0.40
        # production threshold directly from button_detector.yaml.
        DeclareLaunchArgument('confidence_threshold', default_value='0.05'),
        DeclareLaunchArgument('simulation_layout_relabel', default_value='false'),
        Node(
            package='piper_elevator_app',
            executable='button_detector',
            name='button_detector',
            output='screen',
            parameters=[
                parameters,
                {
                    'use_sim_time': ParameterValue(
                        LaunchConfiguration('use_sim_time'),
                        value_type=bool,
                    ),
                    'confidence_threshold': ParameterValue(
                        LaunchConfiguration('confidence_threshold'),
                        value_type=float,
                    ),
                    'simulation_layout_relabel': ParameterValue(
                        LaunchConfiguration('simulation_layout_relabel'),
                        value_type=bool,
                    ),
                },
            ],
        ),
    ])
