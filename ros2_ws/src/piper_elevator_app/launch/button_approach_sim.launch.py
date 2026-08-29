import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
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
    servo_parameters = os.path.join(
        app_share,
        'config',
        'button_visual_servo.yaml',
    )
    press_parameters = os.path.join(
        app_share,
        'config',
        'button_press.yaml',
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument(
            'pika_tcp_offset',
            default_value='[0.006, 0.0, 0.189, 0.0, 0.0, 0.0]',
        ),
        DeclareLaunchArgument('button_base_x', default_value='0.55'),
        DeclareLaunchArgument('button_base_y', default_value='0.0'),
        DeclareLaunchArgument('button_base_z', default_value='0.30'),
        DeclareLaunchArgument('auto_execute', default_value='true'),
        IncludeLaunchDescription(
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
            }.items(),
        ),
        # Pika eye-in-hand 暂定外参：D405 位于 TCP 后方 10 cm，光轴与
        # TCP +Z 同向。真机必须用手眼标定结果替换。
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='simulated_camera_tf',
            output='screen',
            arguments=[
                '--x', '0.0',
                '--y', '0.0',
                '--z', '-0.10',
                '--qx', '0.0',
                '--qy', '0.0',
                '--qz', '0.0',
                '--qw', '1.0',
                '--frame-id', 'tcp_link',
                '--child-frame-id', 'camera_color_optical_frame',
            ],
        ),
        Node(
            package='piper_elevator_app',
            executable='mock_button_pose',
            name='mock_button_pose',
            output='screen',
            parameters=[{
                'frame_id': 'camera_color_optical_frame',
                # 按钮固定在更远的 base_link 位置；发布器实时转换到相机系。
                'fixed_frame_id': 'base_link',
                'x': ParameterValue(
                    LaunchConfiguration('button_base_x'),
                    value_type=float,
                ),
                'y': ParameterValue(
                    LaunchConfiguration('button_base_y'),
                    value_type=float,
                ),
                'z': ParameterValue(
                    LaunchConfiguration('button_base_z'),
                    value_type=float,
                ),
                # 仿真按钮面板法向沿 base_link +X。
                'normal_x': 1.0,
                'normal_y': 0.0,
                'normal_z': 0.0,
                'publish_rate_hz': 5.0,
            }],
        ),
        Node(
            package='piper_elevator_app',
            executable='button_approach_planner',
            name='button_approach_planner',
            output='screen',
            parameters=[
                parameters,
                {
                    'simulation_mode': True,
                    'allow_execution': True,
                    # The fake Pika controller exposes only center_joint; its
                    # mimic finger joints are not command interfaces.
                    'close_gripper_before_plan': False,
                    'auto_plan_execute': ParameterValue(
                        LaunchConfiguration('auto_execute'),
                        value_type=bool,
                    ),
                },
            ],
        ),
        Node(
            package='piper_elevator_app',
            executable='button_visual_servo',
            name='button_visual_servo',
            output='screen',
            parameters=[
                servo_parameters,
                {
                    'simulation_mode': True,
                    'allow_execution': True,
                },
            ],
        ),
        Node(
            package='piper_elevator_app',
            executable='button_press_executor',
            name='button_press_executor',
            output='screen',
            parameters=[
                press_parameters,
                {
                    'simulation_mode': True,
                    'allow_execution': True,
                },
            ],
        ),
    ])
