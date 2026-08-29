import ast
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launch_utils import DeclareBooleanLaunchArg

from piper_elevator_app.launch_mode import select_moveit_launch_mode
import yaml


def _make_moveit_config(context):
    app_share = get_package_share_directory('piper_elevator_app')
    external_hardware = LaunchConfiguration('external_hardware').perform(
        context
    ).casefold() == 'true'
    mode = select_moveit_launch_mode(external_hardware)
    tcp_offset = ast.literal_eval(
        LaunchConfiguration('pika_tcp_offset').perform(context)
    )
    if len(tcp_offset) != 6:
        raise ValueError('pika_tcp_offset must contain six numbers')

    mappings = {
        'initial_positions_file': os.path.join(
            app_share,
            'config',
            'piper_pika_initial_positions.yaml',
        ),
        'tcp_offset_xyz': ' '.join(str(value) for value in tcp_offset[:3]),
        'tcp_offset_rpy': ' '.join(str(value) for value in tcp_offset[3:]),
    }
    return (
        MoveItConfigsBuilder(
            'piper_pika',
            package_name='piper_elevator_app',
        )
        .robot_description(
            file_path=f'config/{mode.description_file}',
            mappings=mappings,
        )
        .robot_description_semantic(
            file_path='config/piper_pika.srdf',
        )
        .robot_description_kinematics(
            file_path='config/piper_pika_kinematics.yaml',
        )
        .joint_limits(
            file_path='config/piper_pika_joint_limits.yaml',
        )
        .trajectory_execution(
            file_path='config/piper_pika_moveit_controllers.yaml',
        )
        .to_moveit_configs()
    )


def _launch_setup(context):
    app_share = get_package_share_directory('piper_elevator_app')
    agx_moveit_share = get_package_share_directory('agx_arm_moveit')
    moveit_config = _make_moveit_config(context)
    external_hardware = LaunchConfiguration('external_hardware').perform(
        context
    ).casefold() == 'true'
    mode = select_moveit_launch_mode(external_hardware)
    joint_states_topic = LaunchConfiguration('joint_states_topic')
    controller_output_topic = LaunchConfiguration(
        'controller_output_topic'
    )
    controllers = ['joint_state_broadcaster', 'arm_controller']
    if LaunchConfiguration('start_pika_controller').perform(
        context
    ).casefold() == 'true':
        controllers.append('pika_gripper_controller')

    move_group_parameters = [
        moveit_config.to_dict(),
        {
            'publish_robot_description_semantic': True,
            'allow_trajectory_execution': True,
            'publish_planning_scene': True,
            'publish_geometry_updates': True,
            'publish_state_updates': True,
            'publish_transforms_updates': True,
            'monitor_dynamics': False,
            'use_sim_time': ParameterValue(
                LaunchConfiguration('use_sim_time'),
                value_type=bool,
            ),
        },
    ]

    nodes = []
    if mode.start_robot_state_publisher:
        nodes.append(
            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                output='screen',
                remappings=[('joint_states', joint_states_topic)],
                parameters=[
                    moveit_config.robot_description,
                    {
                        'use_sim_time': ParameterValue(
                            LaunchConfiguration('use_sim_time'),
                            value_type=bool,
                        ),
                    },
                ],
            )
        )
    nodes.append(
        Node(
            package='moveit_ros_move_group',
            executable='move_group',
            output='screen',
            parameters=move_group_parameters,
            remappings=[('joint_states', joint_states_topic)],
            additional_env={'DISPLAY': os.environ.get('DISPLAY', '')},
        )
    )
    if LaunchConfiguration('start_moveit_servo').perform(
        context
    ).casefold() == 'true':
        with open(
            os.path.join(
                app_share,
                'config',
                'piper_pika_servo.yaml',
            ),
            encoding='utf-8',
        ) as servo_file:
            servo_config = yaml.safe_load(servo_file)
        servo_config['use_gazebo'] = (
            LaunchConfiguration('use_sim_time').perform(context).casefold()
            == 'true'
        )
        servo_config['joint_topic'] = joint_states_topic.perform(context)
        nodes.append(
            Node(
                package='moveit_servo',
                executable='servo_node_main',
                name='servo_node',
                output='screen',
                parameters=[
                    {'moveit_servo': servo_config},
                    moveit_config.robot_description,
                    moveit_config.robot_description_semantic,
                    moveit_config.robot_description_kinematics,
                    moveit_config.joint_limits,
                    {
                        'use_sim_time': ParameterValue(
                            LaunchConfiguration('use_sim_time'),
                            value_type=bool,
                        ),
                    },
                ],
            )
        )
    if mode.start_ros2_control:
        nodes.append(
            Node(
                package='controller_manager',
                executable='ros2_control_node',
                output='screen',
                parameters=[
                    moveit_config.robot_description,
                    os.path.join(
                        app_share,
                        'config',
                        'piper_pika_ros2_controllers.yaml',
                    ),
                ],
                remappings=[('joint_states', controller_output_topic)],
            )
        )
    if mode.start_controller_spawners:
        nodes.append(
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=controllers + [
                    '--controller-manager',
                    '/controller_manager',
                ],
                output='screen',
            )
        )
    nodes.append(
        Node(
            package='rviz2',
            executable='rviz2',
            output='log',
            arguments=[
                '-d',
                os.path.join(agx_moveit_share, 'config', 'moveit.rviz'),
            ],
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.robot_description_kinematics,
                moveit_config.planning_pipelines,
                moveit_config.joint_limits,
                {
                    'use_sim_time': ParameterValue(
                        LaunchConfiguration('use_sim_time'),
                        value_type=bool,
                    ),
                },
            ],
            remappings=[('joint_states', joint_states_topic)],
            condition=IfCondition(LaunchConfiguration('use_rviz')),
        )
    )
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareBooleanLaunchArg('use_rviz', default_value=True),
        DeclareBooleanLaunchArg(
            'external_hardware',
            default_value=False,
            description=(
                'Connect MoveIt to an already running controller manager.'
            ),
        ),
        DeclareBooleanLaunchArg('use_sim_time', default_value=False),
        DeclareBooleanLaunchArg(
            'start_moveit_servo',
            default_value=True,
            description='Start the smooth Cartesian velocity controller.',
        ),
        DeclareBooleanLaunchArg(
            'start_pika_controller',
            default_value=True,
            description='Start the simulated Pika trajectory controller.',
        ),
        DeclareLaunchArgument(
            'joint_states_topic',
            default_value=PythonExpression([
                "'/piper_pika/joint_states' if '",
                LaunchConfiguration('external_hardware'),
                "'.lower() == 'true' else 'control/joint_states'",
            ]),
            description='Joint feedback consumed by MoveIt and TF.',
        ),
        DeclareLaunchArgument(
            'controller_output_topic',
            default_value='control/joint_states',
            description='Joint commands emitted by the ros2_control proxy.',
        ),
        DeclareLaunchArgument(
            'pika_tcp_offset',
            default_value='[0.006, 0.0, 0.189, 0.0, 0.0, 0.0]',
            description='Pika TCP [x, y, z, rx, ry, rz] from its base.',
        ),
        OpaqueFunction(function=_launch_setup),
    ])
