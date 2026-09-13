import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    app_share = get_package_share_directory('piper_elevator_app')
    parameters = os.path.join(
        app_share,
        'config',
        'button_approach.yaml',
    )
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'publish_camera_tf',
            default_value='false',
            description=(
                'Publish a camera TF only after supplying calibrated values.'
            ),
        ),
        DeclareLaunchArgument('camera_parent_frame', default_value='tcp_link'),
        DeclareLaunchArgument(
            'camera_child_frame', default_value='camera_link'
        ),
        DeclareLaunchArgument('camera_x', default_value='0.0'),
        DeclareLaunchArgument('camera_y', default_value='0.0'),
        DeclareLaunchArgument('camera_z', default_value='0.0'),
        DeclareLaunchArgument('camera_roll', default_value='0.0'),
        DeclareLaunchArgument('camera_pitch', default_value='0.0'),
        DeclareLaunchArgument('camera_yaw', default_value='0.0'),
        DeclareLaunchArgument('simulation_mode', default_value='false'),
        DeclareLaunchArgument(
            'constrain_coarse_orientation',
            default_value='true',
        ),
        DeclareLaunchArgument(
            'preserve_wrist_roll_from_current',
            default_value='false',
        ),
        DeclareLaunchArgument(
            'preserve_coarse_camera_orientation',
            default_value='false',
        ),
        DeclareLaunchArgument(
            'coarse_vertical_offset_m',
            default_value='0.0',
        ),
        DeclareLaunchArgument(
            'camera_calibration_valid',
            default_value='false',
        ),
        DeclareLaunchArgument('allow_execution', default_value='false'),
        DeclareLaunchArgument(
            'observation_stable_samples', default_value='20'
        ),
        DeclareLaunchArgument(
            'observation_window_max_seconds', default_value='3.0'
        ),
        DeclareLaunchArgument(
            'planning_observation_wait_seconds', default_value='6.0'
        ),
        DeclareLaunchArgument(
            'post_execution_observation_timeout_seconds', default_value='6.0'
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_extrinsic_tf',
            output='screen',
            arguments=[
                '--x', LaunchConfiguration('camera_x'),
                '--y', LaunchConfiguration('camera_y'),
                '--z', LaunchConfiguration('camera_z'),
                '--roll', LaunchConfiguration('camera_roll'),
                '--pitch', LaunchConfiguration('camera_pitch'),
                '--yaw', LaunchConfiguration('camera_yaw'),
                '--frame-id', LaunchConfiguration('camera_parent_frame'),
                '--child-frame-id', LaunchConfiguration('camera_child_frame'),
            ],
            condition=IfCondition(LaunchConfiguration('publish_camera_tf')),
        ),
        Node(
            package='piper_elevator_app',
            executable='button_approach_planner',
            name='button_approach_planner',
            output='screen',
            parameters=[
                parameters,
                {
                    'simulation_mode': ParameterValue(
                        LaunchConfiguration('simulation_mode'),
                        value_type=bool,
                    ),
                    'constrain_coarse_orientation': ParameterValue(
                        LaunchConfiguration('constrain_coarse_orientation'),
                        value_type=bool,
                    ),
                    'preserve_wrist_roll_from_current': ParameterValue(
                        LaunchConfiguration(
                            'preserve_wrist_roll_from_current'
                        ),
                        value_type=bool,
                    ),
                    'preserve_coarse_camera_orientation': ParameterValue(
                        LaunchConfiguration(
                            'preserve_coarse_camera_orientation'
                        ),
                        value_type=bool,
                    ),
                    'coarse_vertical_offset_m': ParameterValue(
                        LaunchConfiguration('coarse_vertical_offset_m'),
                        value_type=float,
                    ),
                    'camera_calibration_valid': ParameterValue(
                        LaunchConfiguration('camera_calibration_valid'),
                        value_type=bool,
                    ),
                    'allow_execution': ParameterValue(
                        LaunchConfiguration('allow_execution'),
                        value_type=bool,
                    ),
                    'observation_stable_samples': ParameterValue(
                        LaunchConfiguration('observation_stable_samples'),
                        value_type=int,
                    ),
                    'observation_window_max_seconds': ParameterValue(
                        LaunchConfiguration('observation_window_max_seconds'),
                        value_type=float,
                    ),
                    'planning_observation_wait_seconds': ParameterValue(
                        LaunchConfiguration('planning_observation_wait_seconds'),
                        value_type=float,
                    ),
                    'post_execution_observation_timeout_seconds': ParameterValue(
                        LaunchConfiguration(
                            'post_execution_observation_timeout_seconds'
                        ),
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
