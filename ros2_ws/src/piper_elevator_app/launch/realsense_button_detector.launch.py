import fcntl
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.actions import OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


_CAMERA_LOCK_HANDLE = None
_CAMERA_LOCK_PATH = '/dev/shm/piper_elevator_realsense.lock'


def _acquire_camera_lock(context):
    """Prevent two host-IPC containers from opening the D405 together."""
    del context
    global _CAMERA_LOCK_HANDLE
    handle = open(_CAMERA_LOCK_PATH, 'a+', encoding='utf-8')
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(
            'RealSense button detector is already running. Stop the '
            'existing launch before starting another one.'
        ) from error
    handle.seek(0)
    handle.truncate()
    handle.write(f'pid={os.getpid()}\n')
    handle.flush()
    _CAMERA_LOCK_HANDLE = handle
    return []


def generate_launch_description():
    app_share = get_package_share_directory('piper_elevator_app')
    realsense_share = get_package_share_directory('realsense2_camera')
    detector_parameters = os.path.join(
        app_share,
        'config',
        'button_detector.yaml',
    )

    return LaunchDescription([
        OpaqueFunction(function=_acquire_camera_lock),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(realsense_share, 'launch', 'rs_launch.py')
            ),
            launch_arguments={
                'camera_name': 'camera',
                'camera_namespace': '',
                'enable_color': 'true',
                'enable_depth': 'true',
                'align_depth.enable': 'true',
                'pointcloud.enable': 'false',
            }.items(),
        ),
        Node(
            package='piper_elevator_app',
            executable='button_detector',
            name='button_detector',
            output='screen',
            parameters=[
                detector_parameters,
                {
                    'use_depth': True,
                    'reliable_input': False,
                    'color_topic': '/camera/color/image_raw',
                    'depth_topic': (
                        '/camera/aligned_depth_to_color/image_raw'
                    ),
                    'camera_info_topic': '/camera/color/camera_info',
                },
            ],
        ),
    ])
