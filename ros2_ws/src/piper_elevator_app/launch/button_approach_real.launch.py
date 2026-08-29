import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from moveit_configs_utils.launch_utils import DeclareBooleanLaunchArg


def generate_launch_description():
    app_share = get_package_share_directory('piper_elevator_app')

    real_arm = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(app_share, 'launch', 'piper_pika_real.launch.py')
        ),
        launch_arguments={
            'can_port': LaunchConfiguration('can_port'),
            'pika_serial_port': LaunchConfiguration('pika_serial_port'),
            'speed_percent': LaunchConfiguration('speed_percent'),
            'pika_tcp_offset': LaunchConfiguration('pika_tcp_offset'),
            'use_rviz': LaunchConfiguration('use_rviz'),
            'auto_enable': LaunchConfiguration('auto_enable'),
            'hardware_commands_enabled': LaunchConfiguration(
                'hardware_commands_enabled'
            ),
            'start_pika_driver': LaunchConfiguration('start_pika_driver'),
        }.items(),
    )

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                app_share,
                'launch',
                'realsense_button_detector.launch.py',
            )
        ),
        condition=IfCondition(LaunchConfiguration('start_camera')),
    )

    planner = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                app_share,
                'launch',
                'button_approach_planner.launch.py',
            )
        ),
        launch_arguments={
            'publish_camera_tf': LaunchConfiguration('publish_camera_tf'),
            'camera_parent_frame': 'tcp_link',
            'camera_child_frame': 'camera_link',
            'camera_x': LaunchConfiguration('camera_x'),
            'camera_y': LaunchConfiguration('camera_y'),
            'camera_z': LaunchConfiguration('camera_z'),
            'camera_roll': LaunchConfiguration('camera_roll'),
            'camera_pitch': LaunchConfiguration('camera_pitch'),
            'camera_yaw': LaunchConfiguration('camera_yaw'),
            'simulation_mode': 'false',
            'camera_calibration_valid': LaunchConfiguration(
                'camera_calibration_valid'
            ),
            'allow_execution': LaunchConfiguration('allow_execution'),
            'hardware_gate_required': 'true',
        }.items(),
    )

    visual_servo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                app_share,
                'launch',
                'button_visual_servo.launch.py',
            )
        ),
        launch_arguments={
            'simulation_mode': 'false',
            'camera_calibration_valid': LaunchConfiguration(
                'camera_calibration_valid'
            ),
            'allow_execution': LaunchConfiguration('allow_execution'),
        }.items(),
    )

    button_press = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                app_share,
                'launch',
                'button_press.launch.py',
            )
        ),
        launch_arguments={
            'simulation_mode': 'false',
            'allow_execution': LaunchConfiguration('allow_execution'),
            'hardware_gate_required': 'true',
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
        real_arm,
        camera,
        planner,
        visual_servo,
        button_press,
    ])
