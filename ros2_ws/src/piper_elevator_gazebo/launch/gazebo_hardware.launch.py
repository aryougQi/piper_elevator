"""Launch only Gazebo virtual hardware and hardware-facing ROS interfaces."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import OpaqueFunction
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro


def _register_description_resource_paths():
    """Expose dependency share roots for package:// meshes in Fortress."""
    description_packages = (
        'agx_arm_description',
        'pika_gripper_description',
        'realsense2_description',
    )
    resource_roots = [
        os.path.dirname(get_package_share_directory(package_name))
        for package_name in description_packages
    ]
    for variable in ('GZ_SIM_RESOURCE_PATH', 'IGN_GAZEBO_RESOURCE_PATH'):
        existing = os.environ.get(variable, '')
        paths = resource_roots + ([existing] if existing else [])
        os.environ[variable] = os.pathsep.join(paths)


def _launch_setup(context):
    _register_description_resource_paths()
    gazebo_share = get_package_share_directory('piper_elevator_gazebo')
    ros_gz_share = get_package_share_directory('ros_gz_sim')
    robot_file = os.path.join(
        gazebo_share,
        'urdf',
        'piper_pika_gazebo.urdf.xacro',
    )
    mappings = {
        'camera_xyz': LaunchConfiguration('camera_xyz').perform(context),
        'camera_rpy': LaunchConfiguration('camera_rpy').perform(context),
        'controllers_file': os.path.join(
            gazebo_share,
            'config',
            'gazebo_controllers.yaml',
        ),
    }
    robot_description = xacro.process_file(
        robot_file,
        mappings=mappings,
    ).toxml()

    world = LaunchConfiguration('world').perform(context)
    verbose = LaunchConfiguration('verbose').perform(context)
    gui = LaunchConfiguration('gui').perform(context).casefold() == 'true'
    gz_arguments = f'-r -v {verbose} {world}'
    if not gui:
        gz_arguments += ' -s'

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_share, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': gz_arguments,
            'gz_version': '6',
            'on_exit_shutdown': 'true',
        }.items(),
    )
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[
            {'robot_description': robot_description},
            {'use_sim_time': True},
        ],
        remappings=[('joint_states', '/piper_pika/joint_states')],
    )
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-world',
            'button_press',
            '-topic',
            '/robot_description',
            '-name',
            'piper_pika',
            '-allow_renaming',
            'false',
        ],
    )
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        parameters=[{
            'config_file': os.path.join(
                gazebo_share,
                'config',
                'hardware_bridge.yaml',
            ),
            'override_frame_id': 'camera_color_optical_frame',
            'use_sim_time': True,
        }],
    )
    joint_state_relay = Node(
        package='topic_tools',
        executable='relay',
        name='piper_pika_joint_state_relay',
        output='screen',
        arguments=['/joint_states', '/piper_pika/joint_states'],
        parameters=[{'use_sim_time': True}],
    )
    controller_spawners = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=[
            'joint_state_broadcaster',
            'arm_controller',
            'pika_gripper_controller',
            '--controller-manager',
            '/controller_manager',
            '--controller-manager-timeout',
            '60',
        ],
    )

    return [
        gazebo,
        robot_state_publisher,
        bridge,
        joint_state_relay,
        spawn_robot,
        RegisterEventHandler(
            OnProcessExit(
                target_action=spawn_robot,
                on_exit=[controller_spawners],
            )
        ),
    ]


def generate_launch_description():
    gazebo_share = get_package_share_directory('piper_elevator_gazebo')
    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument(
            'world',
            default_value=os.path.join(
                gazebo_share,
                'worlds',
                'button_press.sdf',
            ),
        ),
        DeclareLaunchArgument('camera_xyz', default_value='0 0 -0.10'),
        DeclareLaunchArgument('camera_rpy', default_value='0 0 0'),
        DeclareLaunchArgument('verbose', default_value='2'),
        OpaqueFunction(function=_launch_setup),
    ])
