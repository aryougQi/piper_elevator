"""Configuration invariants shared by coarse planning launch entry points."""

import ast
import math
from pathlib import Path

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
POLICY_PARAMETERS = (
    'constrain_coarse_orientation',
    'preserve_coarse_camera_orientation',
    'preserve_wrist_roll_from_current',
    'coarse_vertical_offset_m',
)
OBSERVATION_PARAMETER_TYPES = {
    'observation_stable_samples': 'int',
    'observation_window_max_seconds': 'float',
    'planning_observation_wait_seconds': 'float',
    'post_execution_observation_timeout_seconds': 'float',
}


@pytest.fixture
def config():
    return yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'button_approach.yaml').read_text()
    )['button_approach_planner']['ros__parameters']


def launch_tree(filename):
    return ast.parse((PACKAGE_ROOT / 'launch' / filename).read_text())


def launch_defaults(tree):
    defaults = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == 'DeclareLaunchArgument'
            and node.args
        ):
            continue
        for keyword in node.keywords:
            if keyword.arg == 'default_value':
                defaults[ast.literal_eval(node.args[0])] = ast.literal_eval(
                    keyword.value
                )
    return defaults


def coarse_overrides(tree):
    """Find the coarse policy map without importing or starting ROS launch."""
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        values = {
            key.value: value
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant)
        }
        if all(name in values for name in POLICY_PARAMETERS):
            matches.append(values)
    assert len(matches) == 1
    return matches[0]


def launch_value(node, defaults):
    if isinstance(node, ast.Call):
        assert isinstance(node.func, ast.Name)
        if node.func.id == 'ParameterValue':
            return launch_value(node.args[0], defaults)
        assert node.func.id == 'LaunchConfiguration'
        return defaults[ast.literal_eval(node.args[0])]
    return ast.literal_eval(node)


def normalized(value):
    if isinstance(value, str):
        return yaml.safe_load(value)
    return value


def test_coarse_policy_allows_small_tilt_without_pinning_current_pose(config):
    assert config['constrain_coarse_orientation'] is True
    assert config['preserve_coarse_camera_orientation'] is False
    assert config['preserve_wrist_roll_from_current'] is False
    assert config['coarse_vertical_offset_m'] == 0.0
    assert 0.0 < config['candidate_tilt_rad'] <= (
        config['maximum_camera_tilt_rad']
    ) <= math.radians(10.0) + 1e-6
    assert 0.0 < config['candidate_roll_rad'] <= (
        config['maximum_camera_roll_rad']
    ) <= 0.60 + 1e-6


def test_visibility_budget_covers_execution_position_error(config):
    assert config['camera_frame']
    assert config['camera_info_topic']
    assert config['visibility_button_radius_m'] > 0.0
    assert config['visibility_position_uncertainty_m'] >= (
        config['maximum_execution_position_error_m']
    ) >= config['position_tolerance_m'] > 0.0
    assert 0.0 < config['visibility_image_margin_ratio'] < 0.5
    assert 0.0 < config['visibility_minimum_depth_m'] < (
        config['visibility_maximum_depth_m']
    )
    offsets = config['approach_distance_offsets_m']
    assert offsets and offsets[0] == 0.0
    assert all(math.isfinite(value) and value >= 0.0 for value in offsets)
    assert offsets == sorted(set(offsets))
    assert config['approach_distance_m'] > (
        config['visibility_button_radius_m']
        + config['visibility_position_uncertainty_m']
    )


def test_controller_success_and_home_goals_fit_actual_tracking_budget(config):
    path = (
        PACKAGE_ROOT.parent / 'piper_elevator_gazebo'
        / 'config' / 'gazebo_controllers.yaml'
    )
    arm = yaml.safe_load(path.read_text())['arm_controller']['ros__parameters']
    for name in config['home_joint_names']:
        assert 0.0 < arm['constraints'][name]['goal'] <= (
            config['execution_joint_tolerance_rad']
        )
    assert config['home_joint_tolerance_rad'] >= (
        config['joint_goal_tolerance_rad']
        + config['execution_joint_tolerance_rad']
    )


def test_observation_and_execution_require_stable_fresh_feedback(config):
    assert config['observation_stable_samples'] >= 3
    assert config['execution_stable_samples'] >= 3
    assert 0.0 < config['maximum_tf_fallback_age_seconds'] <= 0.03
    assert 0.0 < config['observation_position_tolerance_m'] <= (
        config['visibility_position_uncertainty_m']
    )
    assert 0.0 < config['observation_normal_tolerance_rad'] <= (
        config['maximum_camera_tilt_rad']
    )
    assert config['post_execution_observation_timeout_seconds'] > (
        config['surface_normal_max_age_seconds']
    ) > config['maximum_tf_fallback_age_seconds']
    assert 0.0 < config['joint_goal_tolerance_rad'] <= (
        config['execution_joint_tolerance_rad']
    ) <= config['execution_start_tolerance_rad']
    assert config['joint_limit_margin_rad'] > (
        config['servo_joint_limit_margin_rad']
        + config['execution_joint_tolerance_rad']
    )
    assert config['minimum_abs_wrist_bend_rad'] > (
        config['execution_joint_tolerance_rad']
    )
    assert 0.0 < config['execution_stable_velocity_rad_s'] <= 0.03


def test_joint_boundary_tolerances_preserve_servo_handover_margin(config):
    servo = yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'piper_pika_servo.yaml').read_text()
    )
    assert 0.0 < config['trajectory_boundary_tolerance_rad'] < (
        config['joint_state_boundary_tolerance_rad']
    ) < min(
        config['joint_goal_tolerance_rad'],
        config['execution_joint_tolerance_rad'],
    )
    assert config['joint_limit_margin_rad'] >= servo['joint_limit_margin']
    assert config['joint_state_boundary_tolerance_rad'] < (
        config['joint_limit_margin_rad'] - servo['joint_limit_margin']
    )


def test_post_motion_capture_limits_match_servo_start_without_relaxing_plans(config):
    servo = yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'button_visual_servo.yaml').read_text()
    )['button_visual_servo']['ros__parameters']
    for name in ('handover_maximum_camera_tilt_rad',
                 'handover_maximum_camera_roll_rad', 'handover_minimum_standoff_m'):
        assert config[name] == pytest.approx(servo[name], abs=1e-7, rel=0.0)
    assert config['maximum_camera_tilt_rad'] < (
        config['handover_maximum_camera_tilt_rad']
    ) <= 0.60 + 1e-6
    assert config['servo_standoff_distance_m'] == servo['standoff_distance_m']
    assert config['servo_maximum_start_error_m'] == servo['maximum_start_error_m']
    assert config['handover_minimum_standoff_m'] > (
        servo['standoff_distance_m'] + servo['distance_tolerance_m']
    )
    assert servo['axial_approach_stop_angle_rad'] < config['maximum_camera_tilt_rad']


def test_candidate_search_and_planning_have_finite_budgets(config):
    assert 0 < config['maximum_candidate_plans'] <= (
        config['maximum_ik_solutions']
    )
    assert 0.0 < config['ik_timeout_seconds'] < (
        config['ik_search_budget_seconds']
    ) < config['planning_budget_seconds']
    assert 0.0 < config['planning_time_seconds'] < (
        config['planning_budget_seconds']
    )
    assert config['planning_attempts'] > 0
    assert config['planning_request_attempts'] > 0
    for name in (
        'moveit_service_timeout_seconds',
        'execution_timeout_margin_seconds',
        'cancellation_timeout_seconds',
    ):
        assert math.isfinite(config[name]) and config[name] > 0.0
    for name in ('ik_service', 'fk_service', 'robot_description_service'):
        assert config[name].startswith('/')
    assert config['allow_execution'] is False
    assert config['camera_calibration_valid'] is False
    assert config['auto_plan_execute'] is False
    assert 0.0 < config['velocity_scaling'] <= 0.10
    assert 0.0 < config['acceleration_scaling'] <= 0.10


@pytest.mark.parametrize('filename', [
    'button_approach_planner.launch.py',
    'button_approach_real.launch.py',
    'button_approach_sim.launch.py',
])
def test_launch_entry_points_share_coarse_policy(config, filename):
    tree = launch_tree(filename)
    defaults = launch_defaults(tree)
    overrides = coarse_overrides(tree)
    for name in POLICY_PARAMETERS:
        assert normalized(launch_value(overrides[name], defaults)) == (
            config[name]
        ), f'{filename}: {name}'


def test_task_entry_point_does_not_force_a_vertical_shift(config):
    defaults = launch_defaults(launch_tree('elevator_task.launch.py'))
    assert normalized(defaults['coarse_vertical_offset_m']) == (
        config['coarse_vertical_offset_m']
    )


def test_generic_observation_launch_defaults_preserve_yaml_and_types(config):
    tree = launch_tree('button_approach_planner.launch.py')
    defaults = launch_defaults(tree)
    overrides = coarse_overrides(tree)
    for name, expected_type in OBSERVATION_PARAMETER_TYPES.items():
        assert normalized(launch_value(overrides[name], defaults)) == config[name]
        value = overrides[name]
        assert isinstance(value, ast.Call)
        assert value.func.id == 'ParameterValue'
        assert any(
            keyword.arg == 'value_type'
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == expected_type
            for keyword in value.keywords
        )


def test_real_observation_window_is_longer_without_relaxing_gates(config):
    tree = launch_tree('button_approach_real.launch.py')
    defaults = launch_defaults(tree)
    overrides = coarse_overrides(tree)
    expected = {
        'observation_stable_samples': 40,
        'observation_window_max_seconds': 6.0,
        'planning_observation_wait_seconds': 8.0,
        'post_execution_observation_timeout_seconds': 8.0,
    }
    for name, value in expected.items():
        assert normalized(launch_value(overrides[name], defaults)) == value
        assert isinstance(overrides[name], ast.Call)
        assert overrides[name].func.id == 'LaunchConfiguration'
        assert ast.literal_eval(overrides[name].args[0]) == name
    for name in (
        'observation_minimum_samples', 'observation_normal_tolerance_rad',
        'observation_position_tolerance_m', 'planning_budget_seconds',
    ):
        assert name not in overrides
    assert config['observation_minimum_samples'] == 8
    assert config['observation_normal_tolerance_rad'] == pytest.approx(
        math.radians(5.0), abs=1e-6,
    )
    assert config['planning_budget_seconds'] == 30.0


def test_simulation_does_not_inherit_real_observation_window():
    tree = launch_tree('button_approach_sim.launch.py')
    defaults = launch_defaults(tree)
    overrides = coarse_overrides(tree)
    for name in OBSERVATION_PARAMETER_TYPES:
        assert name not in defaults
        assert name not in overrides


def test_surface_observation_uses_one_timestamp_for_both_tf_lookups():
    source = (
        PACKAGE_ROOT / 'piper_elevator_app' / 'button_approach_planner.py'
    ).read_text()
    tree = ast.parse(source)
    callback = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == '_surface_pose_callback'
    )
    lookups = [
        node for node in ast.walk(callback)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == '_lookup_message_transform'
    ]
    assert len(lookups) == 2
    assert all(
        len(call.args) == 3
        and isinstance(call.args[2], ast.Name)
        and call.args[2].id == 'stamp'
        for call in lookups
    )
