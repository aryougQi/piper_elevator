"""Bring up the complete elevator task stack and its state machine."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.conditions import UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils.launch_utils import DeclareBooleanLaunchArg


def include(package, filename, arguments, condition=None):
    """Create a compact include action for one package launch file."""
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory(package),
                'launch',
                filename,
            )
        ),
        launch_arguments=arguments.items(),
        condition=condition,
    )


def generate_launch_description():
    app_share = get_package_share_directory('piper_elevator_app')
    simulation = LaunchConfiguration('simulation_mode')
    simulation_condition = IfCondition(simulation)
    real_condition = UnlessCondition(simulation)

    gazebo = include(
        'piper_elevator_gazebo',
        'gazebo_hardware.launch.py',
        {
            'gui': LaunchConfiguration('gazebo_gui'),
        },
        condition=simulation_condition,
    )
    simulation_detector = include(
        'piper_elevator_app',
        'button_detector.launch.py',
        {
            'use_sim_time': 'true',
            'confidence_threshold': LaunchConfiguration(
                'simulation_confidence_threshold'
            ),
            # Gazebo's floor indicator can look like an arrow to the real
            # model. Recover semantics from the known simulated 3x3 layout;
            # close-range tracking continues to use visual identity.
            'simulation_layout_relabel': 'true',
            # CI and headless development hosts may not expose CUDA.  The
            # simulation remains deterministic on CPU; real hardware keeps
            # the production CUDA defaults in the standalone launches.
            'inference_device': 'auto',
            # The shipped ONNX model has a fixed 1280x1280 input. Keep that
            # contract; the latest-frame queue prevents stale-frame buildup.
            'model_input_size': '1280',
        },
        condition=simulation_condition,
    )
    simulation_moveit = include(
        'piper_elevator_app',
        'piper_pika_moveit.launch.py',
        {
            'external_hardware': 'true',
            # Gazebo owns the model with its control plugin and camera chain.
            'publish_robot_state': 'false',
            'use_sim_time': 'true',
            'use_rviz': LaunchConfiguration('use_rviz'),
            'start_pika_controller': 'false',
            'pika_tcp_offset': LaunchConfiguration('pika_tcp_offset'),
        },
        condition=simulation_condition,
    )
    simulation_planner = include(
        'piper_elevator_app',
        'button_approach_planner.launch.py',
        {
            'use_sim_time': 'true',
            'simulation_mode': 'true',
            # Gazebo's robot_state_publisher already owns the complete camera
            # chain. Explicitly block parent real-camera launch arguments from
            # leaking into this included launch and creating duplicate TF.
            'publish_camera_tf': 'false',
            'camera_calibration_valid': 'true',
            'allow_execution': 'true',
        },
        condition=simulation_condition,
    )
    simulation_visual = include(
        'piper_elevator_app',
        'button_visual_servo.launch.py',
        {
            'use_sim_time': 'true',
            'simulation_mode': 'true',
            # SAM2 is optional on CPU-only simulation hosts; enable it with
            # enable_sam2:=true when a CUDA runtime is available.
            'require_sam2_tracking': LaunchConfiguration('enable_sam2'),
            'camera_calibration_valid': 'true',
            'allow_execution': 'true',
            # SAM2 surface poses arrive at about 4-5 Hz in this simulation;
            # accept one normal inference interval without declaring stale.
            'expected_observation_gap_seconds': '0.40',
        },
        condition=simulation_condition,
    )
    simulation_tracker = include(
        'piper_elevator_app',
        'sam2_button_tracker.launch.py',
        {
            'use_sim_time': 'true',
            'device': 'cuda',
            'enabled': LaunchConfiguration('enable_sam2'),
            'debug_image': LaunchConfiguration('sam2_debug_image'),
        },
        condition=simulation_condition,
    )
    simulation_press = include(
        'piper_elevator_app',
        'button_press.launch.py',
        {
            'use_sim_time': 'true',
            'simulation_mode': 'true',
            'allow_execution': 'true',
        },
        condition=simulation_condition,
    )

    real_stack = include(
        'piper_elevator_app',
        'button_approach_real.launch.py',
        {
            'can_port': LaunchConfiguration('can_port'),
            'pika_serial_port': LaunchConfiguration('pika_serial_port'),
            'camera_serial_no': LaunchConfiguration('camera_serial_no'),
            'speed_percent': LaunchConfiguration('speed_percent'),
            'pika_tcp_offset': LaunchConfiguration('pika_tcp_offset'),
            'use_rviz': LaunchConfiguration('use_rviz'),
            'start_camera': LaunchConfiguration('start_camera'),
            'start_pika_driver': LaunchConfiguration('start_pika_driver'),
            'auto_enable': LaunchConfiguration('auto_enable'),
            'enable_timeout': LaunchConfiguration('enable_timeout'),
            'hardware_commands_enabled': LaunchConfiguration(
                'hardware_commands_enabled'
            ),
            'publish_camera_tf': LaunchConfiguration('publish_camera_tf'),
            'camera_calibration_valid': LaunchConfiguration(
                'camera_calibration_valid'
            ),
            'allow_execution': LaunchConfiguration('allow_execution'),
            'coarse_vertical_offset_m': LaunchConfiguration(
                'coarse_vertical_offset_m'
            ),
            'camera_x': LaunchConfiguration('camera_x'),
            'camera_y': LaunchConfiguration('camera_y'),
            'camera_z': LaunchConfiguration('camera_z'),
            'camera_roll': LaunchConfiguration('camera_roll'),
            'camera_pitch': LaunchConfiguration('camera_pitch'),
            'camera_yaw': LaunchConfiguration('camera_yaw'),
        },
        condition=real_condition,
    )

    manager = Node(
        package='piper_elevator_app',
        executable='elevator_task_manager',
        name='elevator_task_manager',
        output='screen',
        parameters=[
            os.path.join(app_share, 'config', 'elevator_task.yaml'),
            {
                'require_sam2_tracking': ParameterValue(simulation, value_type=bool),
                'use_sim_time': ParameterValue(
                    simulation,
                    value_type=bool,
                ),
            },
        ],
    )

    return LaunchDescription([
        DeclareBooleanLaunchArg('simulation_mode', default_value=True),
        DeclareBooleanLaunchArg(
            'enable_sam2',
            default_value=True,
            description='Run SAM2 in simulation; requires a working CUDA runtime',
        ),
        DeclareBooleanLaunchArg(
            'sam2_debug_image',
            default_value=True,
            description='Publish the annotated SAM2 debug image',
        ),
        # Headless/container runs have no X server; starting the GUI aborts
        # Gazebo before sensors and controllers come up. Users can opt in with
        # gazebo_gui:=true on a desktop.
        DeclareBooleanLaunchArg('gazebo_gui', default_value=False),
        DeclareBooleanLaunchArg('use_rviz', default_value=True),
        DeclareLaunchArgument(
            'simulation_confidence_threshold',
            default_value='0.05',
        ),
        DeclareLaunchArgument('can_port', default_value='can0'),
        DeclareLaunchArgument(
            'pika_serial_port', default_value='/dev/ttyUSB60'
        ),
        DeclareLaunchArgument(
            'camera_serial_no', default_value='_315122272433'
        ),
        DeclareLaunchArgument('speed_percent', default_value='10'),
        DeclareLaunchArgument(
            'pika_tcp_offset',
            default_value='[0.006, 0.0, 0.189, 0.0, 0.0, 0.0]',
        ),
        DeclareBooleanLaunchArg('start_camera', default_value=True),
        DeclareBooleanLaunchArg('start_pika_driver', default_value=False),
        DeclareBooleanLaunchArg('auto_enable', default_value=False),
        DeclareLaunchArgument('enable_timeout', default_value='15.0'),
        DeclareBooleanLaunchArg(
            'hardware_commands_enabled', default_value=False
        ),
        # Match the real approach entry point: retained hand-eye parameters
        # require explicit validation before enabling real-camera motion.
        DeclareBooleanLaunchArg('publish_camera_tf', default_value=False),
        DeclareBooleanLaunchArg(
            'camera_calibration_valid', default_value=False
        ),
        DeclareBooleanLaunchArg('allow_execution', default_value=False),
        DeclareLaunchArgument(
            'coarse_vertical_offset_m',
            default_value='0.0',
        ),
        DeclareLaunchArgument('camera_x', default_value='-0.0525784297'),
        DeclareLaunchArgument('camera_y', default_value='0.0004861476'),
        DeclareLaunchArgument('camera_z', default_value='-0.1399236010'),
        DeclareLaunchArgument('camera_roll', default_value='-1.2709909926'),
        DeclareLaunchArgument('camera_pitch', default_value='-1.5255574198'),
        DeclareLaunchArgument('camera_yaw', default_value='1.2243363470'),
        gazebo,
        simulation_detector,
        simulation_moveit,
        simulation_planner,
        simulation_tracker,
        simulation_visual,
        simulation_press,
        real_stack,
        manager,
    ])
