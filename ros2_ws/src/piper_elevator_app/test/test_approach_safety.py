"""Exercise the coarse boundary pipeline against the real Piper URDF."""

import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from piper_elevator_app.button_approach_planner import ButtonApproachPlanner
from piper_elevator_app.joint_limits_core import (
    trajectory_position_limit_violation,
)


NAMES = [f'joint{number}' for number in range(1, 7)]
URDF_PATH = (
    Path(__file__).resolve().parents[2]
    / 'agx_arm_ros/src/agx_arm_description/agx_arm_urdf/piper/urdf'
    / 'piper_description.urdf'
)
URDF = ET.parse(URDF_PATH).getroot()
LIMITS = {
    joint.get('name'): (
        float(joint.find('limit').get('lower')),
        float(joint.find('limit').get('upper')),
    )
    for joint in URDF.findall('joint')
    if joint.get('name') in NAMES
}
SAFE = dict(zip(NAMES, [0.0, 1.0, -1.0, 0.0, -0.6, 0.0]))


class PlannerParameters:
    values = {
        'home_joint_names': NAMES,
        'wrist_singularity_joint': 'joint5',
        'minimum_abs_wrist_bend_rad': 0.4,
        'joint_limit_margin_rad': 0.15,
        'joint_goal_tolerance_rad': 0.005,
        'execution_joint_tolerance_rad': 0.010,
        'joint_state_boundary_tolerance_rad': 0.0001,
        'trajectory_boundary_tolerance_rad': 1.0e-9,
        'joint_state_max_age_seconds': 0.5,
    }
    _load_arm_joint_limits = ButtonApproachPlanner._load_arm_joint_limits
    _fresh_joint_positions = ButtonApproachPlanner._fresh_joint_positions
    _normalized_joint_positions = (
        ButtonApproachPlanner._normalized_joint_positions
    )
    _joint_configuration_is_safe = (
        ButtonApproachPlanner._joint_configuration_is_safe
    )
    _joint_goal_constraints = ButtonApproachPlanner._joint_goal_constraints
    _motion_trajectory_is_safe = (
        ButtonApproachPlanner._motion_trajectory_is_safe
    )
    _trajectory_wrist_is_safe = ButtonApproachPlanner._trajectory_wrist_is_safe

    def __init__(self, positions=None):
        self._arm_joint_limits = dict(LIMITS)
        self._lock = threading.Lock()
        self._latest_joint_positions = dict(
            SAFE if positions is None else positions
        )
        self._latest_joint_received_at = time.monotonic()
        self.logs = []

    def get_parameter(self, name):
        return SimpleNamespace(value=self.values[name])

    def _string_parameter(self, name):
        return str(self.values[name])

    def get_logger(self):
        return SimpleNamespace(info=self.logs.append)


def trajectory(positions, names=None, velocities=None, accelerations=None):
    names = NAMES if names is None else names
    points = []
    for index, position in enumerate(positions):
        points.append(SimpleNamespace(
            positions=[position[name] for name in names]
            if isinstance(position, dict) else list(position),
            velocities=[] if velocities is None else list(velocities[index]),
            accelerations=(
                [] if accelerations is None else list(accelerations[index])
            ),
            time_from_start=SimpleNamespace(sec=index, nanosec=0),
        ))
    return SimpleNamespace(joint_trajectory=SimpleNamespace(
        joint_names=names, points=points,
    ))


def test_real_robot_has_asymmetric_home_boundaries():
    assert set(LIMITS) == set(NAMES)
    assert LIMITS['joint2'][0] == 0.0 < LIMITS['joint2'][1]
    assert LIMITS['joint3'][0] < LIMITS['joint3'][1] == 0.0


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('side', [0, 1])
def test_roundoff_at_each_urdf_boundary_is_normalized_in_a_separate_copy(
    name, side,
):
    boundary = LIMITS[name][side]
    raw = dict(SAFE)
    raw[name] = boundary + (-1.0 if side == 0 else 1.0) * 5.0e-5
    raw['center_joint'] = 0.02
    planner = PlannerParameters(raw)
    result = planner._normalized_joint_positions('test boundary')
    assert result[name] == boundary
    assert result['center_joint'] == 0.02
    assert planner._latest_joint_positions == raw
    assert result is not planner._latest_joint_positions
    assert len(planner.logs) == 1


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('side', [0, 1])
def test_nontrivial_feedback_violation_at_each_boundary_is_rejected(
    name, side,
):
    raw = dict(SAFE)
    raw[name] = LIMITS[name][side] + (-1.0 if side == 0 else 1.0) * 0.001
    planner = PlannerParameters(raw)
    with pytest.raises(ValueError, match=name):
        planner._normalized_joint_positions('unsafe feedback')
    assert planner._latest_joint_positions == raw


@pytest.mark.parametrize('invalid', [math.nan, math.inf, -math.inf])
def test_normalization_does_not_accept_nonfinite_feedback(invalid):
    raw = dict(SAFE)
    raw['joint3'] = invalid
    with pytest.raises(ValueError, match='Fresh, complete'):
        PlannerParameters(raw)._normalized_joint_positions('invalid feedback')


def test_stale_feedback_is_rejected_before_normalization():
    planner = PlannerParameters()
    planner._latest_joint_received_at = time.monotonic() - 1.0
    with pytest.raises(ValueError, match='Fresh, complete'):
        planner._normalized_joint_positions('stale feedback')


def test_observed_home_roundoff_is_removed_without_weakening_trajectory_limits(
):
    raw = dict.fromkeys(NAMES, 0.0)
    raw['joint3'] = 2.4585283000380797e-8
    planner = PlannerParameters(raw)
    valid, reason = planner._motion_trajectory_is_safe(trajectory([raw, SAFE]))
    assert not valid
    assert 'joint3' in reason and 'point 0' in reason
    normalized = planner._normalized_joint_positions('observed Home')
    valid, reason = planner._trajectory_wrist_is_safe(
        trajectory([normalized, SAFE])
    )
    assert valid, reason
    assert planner._latest_joint_positions['joint3'] == raw['joint3']


def test_transit_from_home_can_cross_straight_wrist_before_safe_handover():
    home = dict.fromkeys(NAMES, 0.0)
    intermediate = {**SAFE, 'joint5': -0.2}
    valid, reason = PlannerParameters()._trajectory_wrist_is_safe(
        trajectory([home, intermediate, SAFE]),
    )
    assert valid, reason


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('side', [0, 1])
def test_candidate_plan_and_actual_endpoint_have_nested_limit_reserves(
    name, side,
):
    planner = PlannerParameters()
    boundary = LIMITS[name][side]
    inward = 1.0 if side == 0 else -1.0
    candidate = {**SAFE, name: boundary + inward * 0.165001}
    assert planner._joint_configuration_is_safe(
        candidate, reserve_rad=0.015
    )[0]
    planned = {**SAFE, name: boundary + inward * 0.160001}
    assert not planner._joint_configuration_is_safe(
        planned, reserve_rad=0.015
    )[0]
    valid, reason = planner._trajectory_wrist_is_safe(
        trajectory([SAFE, planned])
    )
    assert valid, reason
    actual = {**SAFE, name: boundary + inward * 0.150001}
    assert planner._joint_configuration_is_safe(actual)[0]
    assert not planner._trajectory_wrist_is_safe(trajectory([SAFE, actual]))[0]
    outside = {**SAFE, name: boundary + inward * 0.149999}
    assert not planner._joint_configuration_is_safe(outside)[0]


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('side', [0, 1])
def test_goal_region_and_tracking_error_stay_inside_actual_joint_margin(
    name, side,
):
    planner = PlannerParameters()
    inward = 1.0 if side == 0 else -1.0
    center = {**SAFE, name: LIMITS[name][side] + inward * 0.162}
    constraints = planner._joint_goal_constraints(center)
    goal = next(
        item for item in constraints.joint_constraints
        if item.joint_name == name
    )
    lower_goal = goal.position - goal.tolerance_below
    upper_goal = goal.position + goal.tolerance_above
    lower, upper = LIMITS[name]
    assert lower_goal >= lower + 0.16 - 1e-12
    assert upper_goal <= upper - 0.16 + 1e-12
    assert lower_goal - 0.01 >= lower + 0.15 - 1e-12
    assert upper_goal + 0.01 <= upper - 0.15 + 1e-12
    assert min(goal.tolerance_below, goal.tolerance_above) == pytest.approx(
        0.002
    )


@pytest.mark.parametrize('sign', [-1.0, 1.0])
def test_wrist_bend_reserves_cover_goal_width_and_execution_error(sign):
    planner = PlannerParameters()
    candidate = {**SAFE, 'joint5': sign * 0.415001}
    assert planner._joint_configuration_is_safe(
        candidate, reserve_rad=0.015
    )[0]
    planned = {**SAFE, 'joint5': sign * 0.410001}
    assert planner._trajectory_wrist_is_safe(trajectory([SAFE, planned]))[0]
    actual = {**SAFE, 'joint5': sign * 0.400001}
    assert planner._joint_configuration_is_safe(actual)[0]
    assert not planner._trajectory_wrist_is_safe(trajectory([SAFE, actual]))[0]
    unsafe = {**SAFE, 'joint5': sign * 0.399999}
    assert not planner._joint_configuration_is_safe(unsafe)[0]


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('side', [0, 1])
def test_intermediate_joint_bounds_apply_even_when_endpoint_is_safe(
    name, side,
):
    start = {
        **SAFE, name: LIMITS[name][side] + (-1.0 if side == 0 else 1.0) * 1e-5
    }
    valid, reason = PlannerParameters()._trajectory_wrist_is_safe(
        trajectory([start, SAFE]),
    )
    assert not valid
    assert name in reason


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('quintic', [False, True])
def test_spline_extrema_cannot_escape_real_limits_between_valid_points(
    name, side, quintic,
):
    inward = 1.0 if side == 0 else -1.0
    point = {**SAFE, name: LIMITS[name][side] + inward * 0.01}
    first_velocity = [0.0] * 6
    last_velocity = [0.0] * 6
    first_velocity[NAMES.index(name)] = -inward
    last_velocity[NAMES.index(name)] = inward
    motion = trajectory(
        [point, point], velocities=[first_velocity, last_velocity],
        accelerations=[[0.0] * 6, [0.0] * 6] if quintic else None,
    )
    valid, reason = PlannerParameters()._motion_trajectory_is_safe(motion)
    assert not valid
    assert name in reason and 'segment' in reason


@pytest.mark.parametrize('sec,nanosec', [
    (-1, 0), (math.nan, 0), (math.inf, 0), (1, math.nan), (1, 1e9), (0, 0),
])
def test_invalid_or_nonincreasing_trajectory_time_is_rejected(sec, nanosec):
    motion = trajectory([SAFE, SAFE])
    motion.joint_trajectory.points[-1].time_from_start = SimpleNamespace(
        sec=sec, nanosec=nanosec,
    )
    assert not PlannerParameters()._motion_trajectory_is_safe(motion)[0]


@pytest.mark.parametrize('sec,nanosec', [(1, 0), (0, 1)])
def test_delayed_first_point_cannot_add_unchecked_controller_interpolation(
    sec, nanosec,
):
    motion = trajectory([SAFE, SAFE])
    motion.joint_trajectory.points[0].time_from_start = SimpleNamespace(
        sec=sec, nanosec=nanosec,
    )
    motion.joint_trajectory.points[1].time_from_start.sec = 2
    valid, reason = PlannerParameters()._motion_trajectory_is_safe(motion)
    assert not valid
    assert 'start at time zero' in reason


def test_delayed_joint3_goal_rejected_despite_safe_transmitted_segment():
    near_limit = dict(SAFE, joint3=-0.01)
    later = dict(SAFE, joint3=-0.11)
    inward = [0.0, 0.0, -0.1, 0.0, 0.0, 0.0]
    moving_outward = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]
    motion = trajectory([near_limit, later], velocities=[inward, inward])
    for index, sample in enumerate(motion.joint_trajectory.points):
        sample.time_from_start.sec = index + 1
    assert trajectory_position_limit_violation(
        NAMES, motion.joint_trajectory.points, LIMITS, 1e-9,
    ) is None

    # JTC would prepend the current state, with the same position but an
    # outward velocity. Its unseen initial cubic reaches +0.015 rad at 0.5 s.
    controller_path = trajectory(
        [near_limit, near_limit, later],
        velocities=[moving_outward, inward, inward],
    )
    message = trajectory_position_limit_violation(
        NAMES, controller_path.joint_trajectory.points, LIMITS, 1e-9,
    )
    assert 'joint3 segment 0->1 time=0.5s' in message
    assert 'value=0.015' in message
    valid, reason = PlannerParameters()._motion_trajectory_is_safe(motion)
    assert not valid
    assert 'start at time zero' in reason


@pytest.mark.parametrize('points,names', [
    ([], NAMES),
    ([[0.6], [0.6]], ['joint5']),
    ([[0.0] * 6, [0.0] * 6], NAMES[:5] + ['joint5']),
    ([[0.0] * 5, [0.0] * 5], NAMES),
])
def test_incomplete_or_ambiguous_trajectories_are_rejected(points, names):
    result = PlannerParameters()._trajectory_wrist_is_safe(
        trajectory(points, names),
    )
    assert not result[0]
