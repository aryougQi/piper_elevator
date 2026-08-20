from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os


def generate_launch_description():
    package_share = get_package_share_directory('piper_elevator_app')
    parameters = os.path.join(package_share, 'config', 'button_detector.yaml')

    return LaunchDescription([
        Node(
            package='piper_elevator_app',
            executable='button_detector',
            name='button_detector',
            output='screen',
            parameters=[parameters],
        ),
    ])
