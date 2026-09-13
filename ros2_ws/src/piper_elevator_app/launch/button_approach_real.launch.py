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
            'enable_timeout': LaunchConfiguration('enable_timeout'),
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
        launch_arguments={
            'camera_serial_no': LaunchConfiguration('camera_serial_no'),
        }.items(),
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
            'constrain_coarse_orientation': 'true',
            'preserve_coarse_camera_orientation': 'false',
            'coarse_vertical_offset_m': LaunchConfiguration(
                'coarse_vertical_offset_m'
            ),
            'preserve_wrist_roll_from_current': 'false',
            'camera_calibration_valid': LaunchConfiguration(
                'camera_calibration_valid'
            ),
            'allow_execution': LaunchConfiguration('allow_execution'),
            'observation_stable_samples': LaunchConfiguration(
                'observation_stable_samples'
            ),
            'observation_window_max_seconds': LaunchConfiguration(
                'observation_window_max_seconds'
            ),
            'planning_observation_wait_seconds': LaunchConfiguration(
                'planning_observation_wait_seconds'
            ),
            'post_execution_observation_timeout_seconds': LaunchConfiguration(
                'post_execution_observation_timeout_seconds'
            ),
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
            'hardware_gate_required': 'true',
            # Keep real and simulation phase transitions identical.  Hardware
            # safety comes from lower-level speed, acceleration and command
            # gates rather than accepting a looser visual pose.
            'level_roll_enabled': 'true',
            'perpendicular_tolerance_rad': '0.05236',
            'axial_approach_full_speed_angle_rad': '0.03491',
            'axial_approach_stop_angle_rad': '0.05236',
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
        DeclareLaunchArgument(
            'camera_serial_no', default_value='_315122272433'
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
        DeclareLaunchArgument('enable_timeout', default_value='15.0'),
        DeclareBooleanLaunchArg(
            'hardware_commands_enabled', default_value=False
        ),
        # The currently checked-in hand-eye result has not passed the latest
        # independent validation.  Keep the real stack inert until a new
        # result is explicitly approved at launch time.
        DeclareBooleanLaunchArg('publish_camera_tf', default_value=False),
        DeclareBooleanLaunchArg(
            'camera_calibration_valid', default_value=False
        ),
        DeclareBooleanLaunchArg('allow_execution', default_value=False),
        # D405 replay needs a longer window to reduce normal uncertainty while
        # retaining the same angular and position acceptance thresholds.
        DeclareLaunchArgument(
            'observation_stable_samples', default_value='40'
        ),
        DeclareLaunchArgument(
            'observation_window_max_seconds', default_value='6.0'
        ),
        DeclareLaunchArgument(
            'planning_observation_wait_seconds', default_value='8.0'
        ),
        DeclareLaunchArgument(
            'post_execution_observation_timeout_seconds', default_value='8.0'
        ),
        DeclareLaunchArgument(
            'coarse_vertical_offset_m',
            default_value='0.0',
            description=(
                'Base-frame vertical offset applied to the real coarse pose.'
            ),
        ),
        # D405 315122272433, result_20260902_linear/handeye_result.json.
        # This historical result is retained for explicit diagnostic runs;
        # it is not the default production calibration.
        DeclareLaunchArgument('camera_x', default_value='-0.0525784297'),
        DeclareLaunchArgument('camera_y', default_value='0.0004861476'),
        DeclareLaunchArgument('camera_z', default_value='-0.1399236010'),
        DeclareLaunchArgument('camera_roll', default_value='-1.2709909926'),
        DeclareLaunchArgument('camera_pitch', default_value='-1.5255574198'),
        DeclareLaunchArgument('camera_yaw', default_value='1.2243363470'),
        real_arm,
        camera,
        planner,
        visual_servo,
        button_press,
    ])
