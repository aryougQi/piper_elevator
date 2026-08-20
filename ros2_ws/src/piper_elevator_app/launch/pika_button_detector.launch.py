import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    app_share = get_package_share_directory('piper_elevator_app')
    sensor_tools_share = get_package_share_directory('sensor_tools')
    detector_parameters = os.path.join(
        app_share,
        'config',
        'button_detector.yaml',
    )

    fisheye_port = LaunchConfiguration('fisheye_port')

    return LaunchDescription([
        DeclareLaunchArgument(
            'fisheye_port',
            default_value='6',
            description='Number in /dev/videoN for the Pika fisheye camera.',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    sensor_tools_share,
                    'launch',
                    'open_fisheye.launch.py',
                )
            ),
            launch_arguments={'fisheye_port': fisheye_port}.items(),
        ),
        Node(
            package='piper_elevator_app',
            executable='button_detector',
            name='button_detector',
            output='screen',
            parameters=[detector_parameters],
        ),
    ])
