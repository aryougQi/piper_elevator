import os
import sys
from glob import glob

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('piper_elevator_app')
    config = os.path.join(share, 'config', 'sam2_button_tracker.yaml')
    # Torch wheels ship a newer cuDNN than the CUDA 12 ONNX runtime. Keep
    # this library precedence local to SAM2 so YOLO retains its CUDA 12 stack.
    torch_libraries = [lib for root in sys.path for lib in glob(os.path.join(root, 'nvidia', '*', 'lib'))]
    library_path = os.pathsep.join(torch_libraries + [os.environ.get('LD_LIBRARY_PATH', '')])
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('device', default_value='cuda'),
        DeclareLaunchArgument('enabled', default_value='true'),
        DeclareLaunchArgument('debug_image', default_value='true'),
        DeclareLaunchArgument('compile_model', default_value='true'),
        Node(
            package='piper_elevator_app',
            executable='sam2_button_tracker',
            name='sam2_button_tracker',
            output='screen',
            additional_env={
                'LD_LIBRARY_PATH': library_path,
                # Persistent cache across compose run containers. The compose
                # service mounts this directory; no model inputs are cached.
                'TORCHINDUCTOR_CACHE_DIR': os.environ.get('TORCHINDUCTOR_CACHE_DIR', '/tmp/torchinductor_root'),
                'TQDM_DISABLE': '1',
            },
            parameters=[config, {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'device': LaunchConfiguration('device'),
                'enabled': LaunchConfiguration('enabled'),
                'debug_image': LaunchConfiguration('debug_image'),
                'compile_model': ParameterValue(LaunchConfiguration('compile_model'), value_type=bool),
            }],
        ),
    ])
