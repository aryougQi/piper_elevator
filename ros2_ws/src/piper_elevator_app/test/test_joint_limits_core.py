"""Regress physical Piper limits, encoder roundoff and spline overshoot."""

import copy
import math
import re
from types import SimpleNamespace

import numpy as np
import pytest

from piper_elevator_app.joint_limits_core import bounded_goal_tolerances
from piper_elevator_app.joint_limits_core import merge_joint_limits
from piper_elevator_app.joint_limits_core import normalize_joint_positions
from piper_elevator_app.joint_limits_core import (
    trajectory_position_limit_violation,
)


# Piper URDF bounds, including the asymmetric joint2 and joint3 home limits.
LIMITS = {
    'joint1': (-2.6179938, 2.6179938),
    'joint2': (0.0, 3.1415926),
    'joint3': (-2.9670597, 0.0),
    'joint4': (-1.7453292, 1.7453292),
    'joint5': (-1.2217304, 1.2217304),
    'joint6': (-2.0943951, 2.0943951),
}
NAMES = list(LIMITS)
MIDDLE = {name: (low + high) / 2.0 for name, (low, high) in LIMITS.items()}


def point(positions=None, at=0.0, velocities=None, accelerations=None):
    nanos = round(at * 1e9)
    values = dict(MIDDLE, **(positions or {}))
    return SimpleNamespace(
        positions=[values[name] for name in NAMES],
        velocities=[] if velocities is None else list(velocities),
        accelerations=[] if accelerations is None else list(accelerations),
        time_from_start=SimpleNamespace(
            sec=nanos // 10**9, nanosec=nanos % 10**9,
        ),
    )


def derivatives(name, value):
    return [value if joint == name else 0.0 for joint in NAMES]


def violation(points, names=NAMES, limits=LIMITS, tolerance=1e-9):
    return trajectory_position_limit_violation(
        names, points, limits, tolerance,
    )


def test_moveit_overrides_never_expand_or_disable_physical_bounds():
    overrides = {
        'joint1': {
            'has_position_limits': True,
            'min_position': -3., 'max_position': 1.,
        },
        'joint2': {
            'has_position_limits': True,
            'min_position': 0.1, 'max_position': 4.,
        },
        'joint3': {'has_position_limits': False},
        'joint4': {'has_velocity_limits': True, 'max_velocity': 5.0},
        'center_joint': {'has_position_limits': True},
    }
    result = merge_joint_limits(LIMITS, overrides)
    assert result['joint1'] == (LIMITS['joint1'][0], 1.0)
    assert result['joint2'] == (0.1, LIMITS['joint2'][1])
    assert result['joint3'] == LIMITS['joint3']
    assert result['joint4'] == LIMITS['joint4']
    assert LIMITS['joint1'][1] == 2.6179938


@pytest.mark.parametrize('override', [
    None,
    {'has_position_limits': 'true'},
    {'has_position_limits': True, 'min_position': 0.0},
    {'has_position_limits': True,
     'min_position': math.nan, 'max_position': 1.},
    {'has_position_limits': True,
     'min_position': 0., 'max_position': math.inf},
    {'has_position_limits': True, 'min_position': 1., 'max_position': -1.},
    {'has_position_limits': True, 'min_position': 3., 'max_position': 4.},
])
def test_invalid_or_disjoint_moveit_position_overrides_are_rejected(override):
    with pytest.raises(ValueError, match='joint1'):
        merge_joint_limits(LIMITS, {'joint1': override})


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('side', ['lower', 'upper'])
def test_every_boundary_only_clamps_small_deviations(name, side):
    low, high = LIMITS[name]
    boundary = low if side == 'lower' else high
    actual = boundary + (-1 if side == 'lower' else 1) * 5e-5
    positions = dict(MIDDLE, **{name: actual, 'center_joint': 0.04})
    before = positions.copy()
    normalized, corrections = normalize_joint_positions(
        positions, LIMITS, 1e-4, context='live state',
    )
    assert normalized[name] == boundary
    assert normalized['center_joint'] == 0.04
    assert corrections == {name: (actual, boundary)}
    assert positions == before


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('side', ['lower', 'upper'])
def test_every_boundary_rejects_real_limit_violations(name, side):
    low, high = LIMITS[name]
    actual = low - 2e-4 if side == 'lower' else high + 2e-4
    with pytest.raises(ValueError) as failure:
        normalize_joint_positions(dict(MIDDLE, **{name: actual}), LIMITS, 1e-4)
    text = str(failure.value)
    assert name in text
    assert 'value=' in text and 'outside [' in text and 'excess=' in text


def test_live_joint3_positive_roundoff_at_home_is_normalized():
    positions = dict.fromkeys(NAMES, 0.0)
    positions.update(joint2=-2e-8, joint3=1.3e-8)
    normalized, corrections = normalize_joint_positions(
        positions, LIMITS, 1e-4,
    )
    assert normalized == dict.fromkeys(NAMES, 0.0)
    assert set(corrections) == {'joint2', 'joint3'}


@pytest.mark.parametrize('positions, tolerance', [
    ({}, 1e-4),
    (dict(MIDDLE, joint1=math.nan), 1e-4),
    (dict(MIDDLE, center_joint=math.inf), 1e-4),
    (dict(MIDDLE, joint4='invalid'), 1e-4),
    (MIDDLE, -1.0),
    (MIDDLE, math.nan),
    (MIDDLE, None),
])
def test_normalization_rejects_invalid_data(positions, tolerance):
    with pytest.raises(ValueError):
        normalize_joint_positions(positions, LIMITS, tolerance)


def test_home_goal_tolerances_are_one_sided_at_asymmetric_joint_limits():
    assert bounded_goal_tolerances(0., *LIMITS['joint2'], 0.015) == (0., 0.015)
    assert bounded_goal_tolerances(0., *LIMITS['joint3'], 0.015) == (0.015, 0.)
    assert bounded_goal_tolerances(-0.6, *LIMITS['joint5'], 0.005, 0.15) == \
        (0.005, 0.005)


def test_goal_tolerances_respect_reserved_margin_on_both_sides():
    below, above = bounded_goal_tolerances(0.155, 0., 1., 0.01, 0.15)
    assert below == pytest.approx(0.005)
    assert above == 0.01
    below, above = bounded_goal_tolerances(0.845, 0., 1., 0.01, 0.15)
    assert below == 0.01
    assert above == pytest.approx(0.005)


@pytest.mark.parametrize('args', [
    (-1e-8, 0., 1., 0.01),
    (1.01, 0., 1., 0.01),
    (0.1, 0., 1., 0.01, 0.15),
    (0.5, 0., 1., 0.01, 0.6),
    (math.nan, 0., 1., 0.01),
    (0.5, 0., 1., -0.01),
    (None, 0., 1., 0.01),
])
def test_goal_center_outside_the_effective_envelope_is_rejected(args):
    with pytest.raises(ValueError):
        bounded_goal_tolerances(*args)


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('side', ['lower', 'upper'])
def test_linear_waypoints_cover_every_arm_bound(name, side):
    low, high = LIMITS[name]
    boundary = low if side == 'lower' else high
    path = [point(), point({name: boundary}, at=1.0)]
    assert violation(path) is None
    path[1].positions[NAMES.index(name)] += (
        (-1 if side == 'lower' else 1) * 1e-5
    )
    message = violation(path)
    assert name in message
    assert 'point 1 time=1s' in message
    assert 'excess=' in message


@pytest.mark.parametrize('name, near, velocity', [
    ('joint2', 0.01, -0.2), ('joint3', -0.01, 0.2),
])
def test_cubic_extrema_detect_hidden_overshoot(name, near, velocity):
    path = [
        point({name: near}, velocities=derivatives(name, velocity)),
        point({name: near}, at=1., velocities=derivatives(name, -velocity)),
    ]
    message = violation(path)
    assert name in message
    assert 'segment 0->1 time=0.5s' in message
    assert 'excess=0.04' in message


def test_cubic_short_overshoot_is_not_missed_by_fixed_time_sampling():
    stationary, other = 0.003, 2.0
    start = -1e-6
    end = start + 1. / 3. - (stationary + other) / 2. + stationary * other
    path = [
        point({'joint3': start}, velocities=derivatives(
            'joint3', stationary * other,
        )),
        point({'joint3': end}, at=1., velocities=derivatives(
            'joint3', (1. - stationary) * (1. - other),
        )),
    ]
    message = violation(path)
    assert 'joint3 segment 0->1 time=0.003s' in message
    # The first regular 10 ms sample has already returned inside the bound.
    at_10ms = start + 0.01**3 / 3. - (stationary + other) * 0.01**2 / 2. \
        + stationary * other * 0.01
    assert at_10ms < 0.0


def test_quintic_checks_exact_interior_extremum_with_duration_scaling():
    start_time, duration = 0.37, 0.41
    # p(u)=-0.01+u^2*(1-u)^2*(1+u), maximum at (sqrt(41)-1)/10.
    path = [
        point({'joint3': -0.01}, at=start_time, velocities=[0.] * 6,
              accelerations=derivatives('joint3', 2. / duration**2)),
        point({'joint3': -0.01}, at=start_time + duration, velocities=[0.] * 6,
              accelerations=derivatives('joint3', 4. / duration**2)),
    ]
    message = violation(path)
    assert 'joint3 segment 0->1' in message
    time_value = float(re.search(r'time=([^s]+)s', message).group(1))
    expected = start_time + duration * (math.sqrt(41.) - 1.) / 10.
    assert time_value == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize('mode', ['linear', 'cubic', 'quintic'])
def test_legitimate_path_from_home_to_interior_has_no_hidden_overshoot(mode):
    velocity = None if mode == 'linear' else [0.] * 6
    acceleration = [0.] * 6 if mode == 'quintic' else None
    path = [
        point(dict.fromkeys(NAMES, 0.), velocities=velocity,
              accelerations=acceleration),
        point(MIDDLE, at=2.0, velocities=velocity, accelerations=acceleration),
    ]
    assert violation(path) is None


@pytest.mark.parametrize('defect', [
    'missing_joint', 'duplicate_joint', 'empty', 'positions', 'nonfinite',
    'velocity_length', 'velocity_nan', 'mixed_velocity', 'mixed_acceleration',
    'acceleration_without_velocity', 'duplicate_time', 'backward_time',
    'negative_time', 'invalid_nanoseconds', 'fractional_seconds',
])
def test_trajectory_rejects_malformed_states_derivatives_and_times(defect):
    path = [point(), point(at=1.)]
    names = NAMES.copy()
    if defect == 'missing_joint':
        names.pop()
    elif defect == 'duplicate_joint':
        names[-1] = names[0]
    elif defect == 'empty':
        path = []
    elif defect == 'positions':
        path[0].positions.pop()
    elif defect == 'nonfinite':
        path[0].positions[0] = math.inf
    elif defect == 'velocity_length':
        path[0].velocities = [0.]
    elif defect == 'velocity_nan':
        path[0].velocities = [math.nan] * 6
    elif defect == 'mixed_velocity':
        path[0].velocities = [0.] * 6
    elif defect == 'mixed_acceleration':
        path[0].velocities = path[1].velocities = [0.] * 6
        path[0].accelerations = [0.] * 6
    elif defect == 'acceleration_without_velocity':
        path[0].accelerations = [0.] * 6
    elif defect == 'duplicate_time':
        path[1].time_from_start = path[0].time_from_start
    elif defect == 'backward_time':
        path.reverse()
    elif defect == 'negative_time':
        path[0].time_from_start.sec = -1
    elif defect == 'invalid_nanoseconds':
        path[0].time_from_start.nanosec = 10**9
    elif defect == 'fractional_seconds':
        path[0].time_from_start.sec = 0.5
    message = violation(path, names=names)
    assert message is not None
    assert message.startswith('Invalid trajectory:')


def test_trajectory_epsilon_does_not_modify_input_or_hide_larger_excess():
    path = [point({'joint3': 5e-10}), point(at=1.)]
    before = copy.deepcopy(path)
    assert violation(path) is None
    assert path == before
    path[0].positions[2] = 5e-8
    assert 'joint3 point 0' in violation(path)


def test_degenerate_stationary_quintic_has_no_spurious_root_failure():
    path = [
        point(at=at, velocities=np.zeros(6), accelerations=np.zeros(6))
        for at in (0., 1., 2.)
    ]
    assert violation(path) is None
