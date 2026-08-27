"""Regression tests for shared and legacy Piper/Pika descriptions."""

from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE_ROOT / 'config'


def expand(filename):
    """Expand one project xacro and parse its robot element."""
    command = ['xacro', str(CONFIG / filename)]
    if filename == 'piper_pika.urdf.xacro':
        command.append(
            'initial_positions_file:='
            f'{CONFIG / "piper_pika_initial_positions.yaml"}'
        )
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return ET.fromstring(completed.stdout)


def test_description_only_robot_has_no_ros2_control():
    robot = expand('piper_pika_description.urdf.xacro')

    assert robot.find('ros2_control') is None
    assert robot.find(
        "link[@name='pika_fingertip_center_link']"
    ) is not None
    assert robot.find("link[@name='tcp_link']") is not None

    fingertip_joint = robot.find(
        "joint[@name='pika_fingertip_center_joint']"
    )
    assert fingertip_joint.find('parent').get('link') == (
        'pika_gripper_base_link'
    )
    assert fingertip_joint.find('child').get('link') == (
        'pika_fingertip_center_link'
    )
    assert fingertip_joint.find('origin').get('xyz') == '0.006 0 0.189'

    tcp_joint = robot.find("joint[@name='tcp_joint']")
    assert tcp_joint.find('parent').get('link') == (
        'pika_fingertip_center_link'
    )
    assert tcp_joint.find('origin').get('xyz') == '0 0 0'


def test_legacy_wrapper_keeps_fake_system_contract():
    robot = expand('piper_pika.urdf.xacro')
    control = robot.find("ros2_control[@name='PiperPikaFakeSystem']")

    assert control.findtext('hardware/plugin') == (
        'mock_components/GenericSystem'
    )
    assert [joint.get('name') for joint in control.findall('joint')] == [
        'joint1',
        'joint2',
        'joint3',
        'joint4',
        'joint5',
        'joint6',
        'center_joint',
    ]


def test_pika_finger_mounts_preserve_official_geometry_and_mirroring():
    robot = expand('piper_pika_description.urdf.xacro')
    left = robot.find("joint[@name='pika_left_finger_joint']")
    right = robot.find("joint[@name='pika_right_finger_joint']")

    assert left.find('origin').get('xyz') == '0.0 0.041 0.08'
    assert right.find('origin').get('xyz') == '0.0 -0.041 0.08'
    assert left.find('mimic').get('joint') == 'center_joint'
    assert left.find('mimic').get('multiplier') == '0.5'
    assert right.find('mimic').get('joint') == 'center_joint'
    assert right.find('mimic').get('multiplier') == '-0.5'
    assert right.find('axis').get('xyz') == '0 0 1'
    assert right.find('limit').get('lower') == '-0.049'
    assert right.find('limit').get('upper') == '0.0'
