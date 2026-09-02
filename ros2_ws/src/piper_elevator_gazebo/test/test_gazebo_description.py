"""Contract tests for the Gazebo robot and button descriptions."""

from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def expand_robot():
    """Expand and parse the Gazebo robot xacro."""
    xacro_file = PACKAGE_ROOT / 'urdf' / 'piper_pika_gazebo.urdf.xacro'
    completed = subprocess.run(
        ['xacro', str(xacro_file)],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout, ET.fromstring(completed.stdout)


def test_gazebo_robot_uses_physical_control_and_pika_camera():
    expanded_xml, robot = expand_robot()

    assert robot.findtext('ros2_control/hardware/plugin') == (
        'gz_ros2_control/GazeboSimSystem'
    )
    assert 'mock_components/GenericSystem' not in expanded_xml
    assert expanded_xml.count(
        'gz_ros2_control::GazeboSimROS2ControlPlugin'
    ) == 1
    assert robot.find("link[@name='camera_link']") is not None
    assert robot.find(
        "link[@name='camera_color_optical_frame']"
    ) is not None
    assert robot.find(
        "gazebo[@reference='camera_link']/sensor[@type='rgbd_camera']"
    ) is not None
    assert 'sensor_d405' not in expanded_xml
    assert 'realsense2_description' not in expanded_xml

    camera_joint = robot.find("joint[@name='pika_camera_joint']")
    assert camera_joint.find('parent').get('link') == (
        'pika_gripper_base_link'
    )
    assert camera_joint.find('child').get('link') == 'camera_link'
    camera_xyz = [
        float(value)
        for value in camera_joint.find('origin').get('xyz').split()
    ]
    assert camera_xyz == [-0.074017, 0.0, 0.069]
    camera_rpy = [
        float(value)
        for value in camera_joint.find('origin').get('rpy').split()
    ]
    assert camera_rpy == [0.0, -1.57079632679, 0.0]

    camera_sensor = robot.find(
        "gazebo[@reference='camera_link']/sensor[@name='pika_fisheye']"
    )
    assert camera_sensor.findtext('topic') == '/pika_fisheye'

    bridge = (
        PACKAGE_ROOT / 'config' / 'hardware_bridge.yaml'
    ).read_text(encoding='utf-8')
    assert '/pika_fisheye/image' in bridge
    assert '/piper_d405' not in bridge
    for button in (
        '1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm'
    ):
        assert f'/elevator_button/button_{button}/contacts' in bridge
    assert 'ros_gz_interfaces/msg/Contacts' in bridge


def test_gazebo_control_joint_contract():
    _, robot = expand_robot()
    control = robot.find("ros2_control[@name='PiperPikaGazeboSystem']")

    assert [joint.get('name') for joint in control.findall('joint')] == [
        'joint1',
        'joint2',
        'joint3',
        'joint4',
        'joint5',
        'joint6',
        'center_joint',
        'pika_left_finger_joint',
        'pika_right_finger_joint',
    ]

    # urdf2sdf drops movable child links without inertia, which would also
    # remove center_joint and make the gripper controller fail to activate.
    actuator = robot.find("link[@name='pika_gripper_actuator_link']")
    assert actuator.find('inertial') is not None

    initial_positions = {
        joint.get('name'): float(
            joint.find("state_interface[@name='position']/param").text
        )
        for joint in control.findall('joint')
    }
    assert 0.0 < initial_positions['joint2'] < 3.1415926
    assert -2.9670597 < initial_positions['joint3'] < 0.0

    left = control.find("joint[@name='pika_left_finger_joint']")
    right = control.find("joint[@name='pika_right_finger_joint']")
    assert left.find('command_interface').get('name') == 'position'
    assert right.find('command_interface').get('name') == 'position'
    assert robot.find(
        "joint[@name='pika_left_finger_joint']/mimic"
    ) is None
    assert robot.find(
        "joint[@name='pika_right_finger_joint']/mimic"
    ) is None

    controllers = yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'gazebo_controllers.yaml').read_text(
            encoding='utf-8'
        )
    )
    assert controllers['gz_ros2_control']['ros__parameters'][
        'position_proportional_gain'
    ] == 0.1
    assert controllers['pika_gripper_controller']['ros__parameters'][
        'joints'
    ] == [
        'center_joint',
        'pika_left_finger_joint',
        'pika_right_finger_joint',
    ]
    arm = controllers['arm_controller']['ros__parameters']
    assert arm['open_loop_control'] is False
    assert arm['allow_partial_joints_goal'] is False
    assert arm['constraints']['goal_time'] <= 1.0
    for joint_name in [
        'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'
    ]:
        assert arm['constraints'][joint_name]['trajectory'] <= 0.12
        assert arm['constraints'][joint_name]['goal'] <= 0.025


def test_every_button_has_physical_travel_and_own_contact_sensor():
    model_file = (
        PACKAGE_ROOT / 'models' / 'elevator_button' / 'model.sdf'
    )
    model_xml = model_file.read_text(encoding='utf-8')
    model = ET.fromstring(model_xml).find('model')
    buttons = ('1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm')
    for button in buttons:
        joint = model.find(f"joint[@name='button_{button}_press_joint']")
        assert joint is not None
        assert joint.get('type') == 'prismatic'
        assert float(joint.findtext('axis/limit/lower')) == -0.004
        assert float(joint.findtext('axis/limit/upper')) == 0.0
        assert float(joint.findtext('axis/dynamics/spring_reference')) == 0.0
        assert float(joint.findtext('axis/dynamics/spring_stiffness')) > 0.0
        link = model.find(f"link[@name='button_{button}_face']")
        assert link is not None
        assert link.find("sensor[@type='contact']") is not None
        texture = link.findtext('visual/material/pbr/metal/albedo_map')
        assert texture == (
            f'model://elevator_button/textures/button_{button}.png'
        )
    assert model.find(
        "plugin[@name='gz::sim::systems::TouchPlugin']"
    ) is not None
    assert '/button_pose' not in model_xml
    assert 'Elevator' not in model_xml


def test_button_fixture_looks_like_a_complete_cabin_panel():
    model_file = (
        PACKAGE_ROOT / 'models' / 'elevator_button' / 'model.sdf'
    )
    model = ET.parse(model_file).getroot().find('model')
    panel = model.find(
        "link[@name='wall_panel']/visual[@name='panel_visual']"
    )
    panel_size = [
        float(value)
        for value in panel.findtext('geometry/box/size').split()
    ]
    assert panel_size == [0.012, 0.27, 0.32]
    assert model.find(
        "link[@name='wall_panel']/visual[@name='floor_display']"
    ) is not None
    for control in (
        '1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm'
    ):
        assert model.find(f"link[@name='button_{control}_face']") is not None


def test_world_contains_only_the_button_fixture_boundary():
    world_file = PACKAGE_ROOT / 'worlds' / 'button_press.sdf'
    world_xml = world_file.read_text(encoding='utf-8')
    world = ET.fromstring(world_xml).find('world')

    fixture = world.find("include[uri='model://elevator_button']")
    assert fixture is not None
    fixture_pose = [float(value) for value in fixture.findtext('pose').split()]
    assert fixture_pose[:3] == [0.55, 0.03, 0.30]
    assert 'door_' not in world_xml
    assert 'Elevator' not in world_xml


def test_package_assets_never_install_python_cache_files():
    setup_source = (PACKAGE_ROOT / 'setup.py').read_text(
        encoding='utf-8'
    )

    assert "'__pycache__' not in path.parts" in setup_source
    assert "path.suffix not in {'.pyc', '.pyo'}" in setup_source
