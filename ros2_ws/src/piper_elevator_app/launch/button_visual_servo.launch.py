import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    app_share = get_package_share_directory('piper_elevator_app')
    parameters = os.path.join(
        app_share,
        'config',
        'button_visual_servo.yaml',
    )
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('simulation_mode', default_value='false'),
        DeclareLaunchArgument(
            'camera_calibration_valid',
            default_value='false',
        ),
        DeclareLaunchArgument('allow_execution', default_value='false'),
        DeclareLaunchArgument(
            'hardware_gate_required',
            default_value='false',
        ),
        DeclareLaunchArgument('level_roll_enabled', default_value='true'),
        DeclareLaunchArgument(
            'perpendicular_tolerance_rad',
            default_value='0.05236',
        ),
        DeclareLaunchArgument(
            'axial_approach_full_speed_angle_rad',
            default_value='0.03491',
        ),
        DeclareLaunchArgument(
            'axial_approach_stop_angle_rad',
            default_value='0.05236',
        ),
        Node(
            package='piper_elevator_app',
            executable='button_visual_servo',
            name='button_visual_servo',
            output='screen',
            parameters=[
                parameters,
                {
                    'simulation_mode': ParameterValue(
                        LaunchConfiguration('simulation_mode'),
                        value_type=bool,
                    ),
                    'camera_calibration_valid': ParameterValue(
                        LaunchConfiguration('camera_calibration_valid'),
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
                    'level_roll_enabled': ParameterValue(
                        LaunchConfiguration('level_roll_enabled'),
                        value_type=bool,
                    ),
                    'perpendicular_tolerance_rad': ParameterValue(
                        LaunchConfiguration('perpendicular_tolerance_rad'),
                        value_type=float,
                    ),
                    'axial_approach_full_speed_angle_rad': ParameterValue(
                        LaunchConfiguration(
                            'axial_approach_full_speed_angle_rad'
                        ),
                        value_type=float,
                    ),
                    'axial_approach_stop_angle_rad': ParameterValue(
                        LaunchConfiguration('axial_approach_stop_angle_rad'),
                        value_type=float,
                    ),
                    'use_sim_time': ParameterValue(
                        LaunchConfiguration('use_sim_time'),
                        value_type=bool,
                    ),
                },
            ],
        ),
    ])
