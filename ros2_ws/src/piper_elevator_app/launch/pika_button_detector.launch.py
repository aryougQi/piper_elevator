import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    app_share = get_package_share_directory('piper_elevator_app')
    detector_parameters = os.path.join(
        app_share,
        'config',
        'button_detector.yaml',
    )

    fisheye_port = LaunchConfiguration('fisheye_port')
    camera_fps = LaunchConfiguration('camera_fps')
    camera_height = LaunchConfiguration('camera_height')
    camera_width = LaunchConfiguration('camera_width')

    return LaunchDescription([
        DeclareLaunchArgument(
            'fisheye_port',
            default_value='6',
            description='Number in /dev/videoN for the Pika fisheye camera.',
        ),
        DeclareLaunchArgument(
            'camera_fps',
            default_value='30',
            description=(
                'Camera FPS. The detector keeps only the newest frame when '
                'inference runs slower than the camera.'
            ),
        ),
        DeclareLaunchArgument(
            'camera_height',
            default_value='480',
            description='Pika fisheye image height.',
        ),
        DeclareLaunchArgument(
            'camera_width',
            default_value='640',
            description='Pika fisheye image width.',
        ),
        Node(
            package='piper_elevator_app',
            executable='pika_fisheye_camera',
            name='camera_fisheye',
            output='screen',
            parameters=[{
                'camera_port': ParameterValue(
                    fisheye_port,
                    value_type=int,
                ),
                'camera_fps': ParameterValue(
                    camera_fps,
                    value_type=int,
                ),
                'camera_height': ParameterValue(
                    camera_height,
                    value_type=int,
                ),
                'camera_width': ParameterValue(
                    camera_width,
                    value_type=int,
                ),
            }],
        ),
        Node(
            package='piper_elevator_app',
            executable='button_detector',
            name='button_detector',
            output='screen',
            # Keep this legacy launcher self-contained even though the main
            # application now uses RealSense RGB-D by default.
            parameters=[
                detector_parameters,
                {
                    'use_depth': False,
                    'color_topic': '/camera_fisheye/color/image_raw',
                    'camera_info_topic': (
                        '/camera_fisheye/color/camera_info'
                    ),
                    'reliable_input': True,
                },
            ],
        ),
    ])
