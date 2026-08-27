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


def test_compose_does_not_disable_gazebo_opengl():
    compose_file = REPOSITORY_ROOT / 'docker-compose.yml'
    if not compose_file.is_file():
        # Package-only installs and the runtime container do not include the
        # repository-level Compose file.
        return
    compose_source = compose_file.read_text(encoding='utf-8')

    # Gazebo Fortress renders its QtQuick scene through GLX / EGL.  These
    # overrides prevent Qt from creating that context and crash MinimalScene
    # as soon as the GUI window is exposed.
    for forbidden_setting in (
        'QT_OPENGL: software',
        'QT_XCB_GL_INTEGRATION: none',
        'LIBGL_ALWAYS_SOFTWARE: "1"',
    ):
        assert forbidden_setting not in compose_source


def test_compose_passes_active_xauthority_to_gui_containers():
    compose_file = REPOSITORY_ROOT / 'docker-compose.yml'
    launcher_file = REPOSITORY_ROOT / 'scripts' / 'gazebo_hardware.sh'
    if not compose_file.is_file() or not launcher_file.is_file():
        return

    compose_source = compose_file.read_text(encoding='utf-8')
    launcher_source = launcher_file.read_text(encoding='utf-8')
    assert 'XAUTHORITY: /tmp/.docker.xauth' in compose_source
    assert (
        '${XAUTHORITY:-/dev/null}:/tmp/.docker.xauth:ro'
        in compose_source
    )
    assert (
        'Gazebo GUI cannot read the current Xauthority file.'
        in launcher_source
    )
