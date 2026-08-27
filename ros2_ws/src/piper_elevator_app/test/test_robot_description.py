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
    assert robot.find("link[@name='tcp_link']") is not None


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
