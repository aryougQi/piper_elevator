"""Launch-wrapper checks with fake Docker/CAN tools; never touch hardware."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture
def launcher(tmp_path):
    root = tmp_path/'project'
    (root/'scripts').mkdir(parents=True)
    (root/'ros2_ws/install').mkdir(parents=True)
    (root/'ros2_ws/install/setup.bash').touch()
    source = Path(__file__).resolve().parents[3]/'scripts/start_real.sh'
    shutil.copy(source, root/'scripts/start_real.sh')
    (root/'config').mkdir()
    shutil.copy(source.parents[1]/'config/real_handeye.json', root/'config/real_handeye.json')
    binaries = tmp_path/'bin'
    binaries.mkdir()
    mock = '''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = [Path(sys.argv[0]).name, *sys.argv[1:]]
if args[0] == 'sudo': args = args[1:]
with open(os.environ['MOCK_LOG'], 'a') as stream:
    stream.write(json.dumps(args)+'\\n')
if args[:3] == ['ip', '-json', '-details']:
    state = os.environ.get('MOCK_STATE', 'down')
    print(json.dumps([{'flags': [] if state == 'down' else ['UP'], 'linkinfo': {
        'info_kind': 'can', 'info_data': {'state': 'ERROR-ACTIVE',
        'bittiming': {'bitrate': 500000 if state == 'wrong_rate' else 1000000}}}}]))
if args[0] == 'ip' and 'set' in args and os.environ.get('MOCK_FAIL_SET'):
    sys.exit(1)
'''
    for name in ['ip', 'sudo', 'docker']:
        path = binaries/name
        path.write_text(mock)
        path.chmod(0o755)
    log = tmp_path/'commands.jsonl'

    def run(*args, state='down', fail_set=False):
        log.write_text('')
        env = dict(os.environ, PATH=str(binaries)+':'+os.environ['PATH'],
                   MOCK_LOG=str(log), MOCK_STATE=state, ROS_DOMAIN_ID='0')
        if fail_set:
            env['MOCK_FAIL_SET'] = '1'
        result = subprocess.run(['bash', str(root/'scripts/start_real.sh'),
                                 '--no-rviz', *args], env=env,
                                capture_output=True, text=True, timeout=10,
                                cwd=tmp_path)
        return result, [json.loads(line) for line in log.read_text().splitlines()]
    return run


def test_dry_run_has_no_external_actions(launcher):
    result, commands = launcher('--dry-run')
    assert result.returncode == 0
    assert commands == []
    assert 'auto_enable:=false' in result.stdout


def test_down_can_is_configured_before_launch(launcher):
    result, commands = launcher()
    assert result.returncode == 0, result.stderr
    config = ['ip', 'link', 'set', 'dev', 'can0', 'type', 'can', 'bitrate', '1000000']
    up = ['ip', 'link', 'set', 'dev', 'can0', 'up']
    launch = next(c for c in commands if c[:3] == ['docker', 'compose', 'run'])
    assert commands.index(config) < commands.index(up) < commands.index(launch)
    assert 'simulation_mode:=false' in launch
    assert 'allow_execution:=false' in launch
    assert 'camera_serial_no:=_315122272433' in launch


def test_ready_can_is_never_reset(launcher):
    result, commands = launcher(state='ready')
    assert result.returncode == 0
    assert not any(c[0] == 'ip' and 'set' in c for c in commands)


@pytest.mark.parametrize('options', [{'state': 'wrong_rate'}, {'fail_set': True}])
def test_can_failure_blocks_launch(launcher, options):
    result, commands = launcher(**options)
    assert result.returncode != 0
    assert not any(c[:3] == ['docker', 'compose', 'run'] for c in commands)


@pytest.mark.parametrize('args', [
    ['--execute', 'camera_calibration_valid:=false'], ['allow_execution:=true'], ['simulation_mode:=true'],
    ['--execute', '--camera-serial', '315122272440'],
])
def test_invalid_execution_arguments_fail_before_hardware(launcher, args):
    result, commands = launcher(*args)
    assert result.returncode != 0
    assert commands == []


def test_explicit_calibration_and_execute_are_forwarded(launcher):
    args = ['--execute', 'camera_calibration_valid:=true', 'publish_camera_tf:=true']
    args += [name+':=0.01' for name in ('camera_x', 'camera_y', 'camera_z',
                                      'camera_roll', 'camera_pitch', 'camera_yaw')]
    result, commands = launcher(*args, state='ready')
    assert result.returncode == 0, result.stderr
    launch = next(c for c in commands if c[:3] == ['docker', 'compose', 'run'])
    for argument in ['auto_enable:=true', 'hardware_commands_enabled:=true',
                     'allow_execution:=true', 'camera_yaw:=0.01']:
        assert argument in launch


def test_execute_loads_user_selected_historical_profile(launcher):
    result, commands = launcher('--execute', state='ready')
    assert result.returncode == 0, result.stderr
    launch = next(c for c in commands if c[:3] == ['docker', 'compose', 'run'])
    assert 'publish_camera_tf:=true' in launch
    assert 'camera_calibration_valid:=true' in launch
    assert 'camera_x:=-0.05257842972237375' in launch
    assert 'camera_yaw:=1.224336346953496' in launch
