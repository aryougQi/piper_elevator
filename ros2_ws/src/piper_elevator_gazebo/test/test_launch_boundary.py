"""Keep the Gazebo launch limited to virtual hardware processes."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]


def test_hardware_launch_uses_only_infrastructure_nodes():
    launch_source = (
        PACKAGE_ROOT / 'launch' / 'gazebo_hardware.launch.py'
    ).read_text(encoding='utf-8')
    script_file = REPOSITORY_ROOT / 'scripts' / 'gazebo_hardware.sh'
    script_source = (
        script_file.read_text(encoding='utf-8')
        if script_file.is_file()
        else ''
    )
    combined = launch_source + script_source

    for required in (
        'ros_gz_sim',
        'ros_gz_bridge',
        'robot_state_publisher',
        'relay',
    ):
        assert required in combined

    # Gazebo converts package:// mesh URIs to model:// URIs.  The parent
    # share directories must therefore be present in both Fortress resource
    # path variables so the official package meshes remain reusable.
    for required_resource in (
        'agx_arm_description',
        'pika_gripper_description',
        'realsense2_description',
        'GZ_SIM_RESOURCE_PATH',
        'IGN_GAZEBO_RESOURCE_PATH',
    ):
        assert required_resource in launch_source

    for forbidden in (
        'depth_adapter',
        'move_group',
        'rviz2',
        'button_detector',
        'button_approach_planner',
        'mock_button_pose',
    ):
        assert forbidden not in combined
