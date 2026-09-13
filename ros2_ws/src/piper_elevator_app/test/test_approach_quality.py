"""Soft preferences preserve bounded joint motion and expose tradeoffs."""

import copy
import math
from types import SimpleNamespace

from piper_elevator_app.approach_quality import configuration_quality
from piper_elevator_app.approach_quality import trajectory_quality
from piper_elevator_app.coarse_approach_core import joint_configuration_cost

import pytest


LIMITS = {f'joint{index}': (-math.pi, math.pi) for index in range(1, 7)}
HOME = {name: 0.0 for name in LIMITS}


def configuration(**updates):
    return dict(HOME, joint5=0.6, **updates)


def trajectory(rows, names=('joint4', 'joint5', 'joint6')):
    return SimpleNamespace(
        joint_names=list(names),
        points=[SimpleNamespace(positions=list(row)) for row in rows],
    )


def test_smaller_wrist_rotations_score_better_and_keep_baseline():
    small = configuration(joint4=0.2, joint6=-0.2)
    large = configuration(joint4=1.2, joint6=-1.2)
    low = configuration_quality(small, HOME, LIMITS)
    high = configuration_quality(large, HOME, LIMITS)
    assert low['cost'] < high['cost']
    assert low['preference_cost'] < high['preference_cost']
    assert low['wrist_roll_motion_rad'] == pytest.approx(0.4)
    assert high['wrist_roll_motion_rad'] == pytest.approx(2.4)
    assert low['baseline_cost'] == joint_configuration_cost(
        small, HOME, LIMITS)
    assert low['cost'] == pytest.approx(
        low['baseline_cost'] + sum(low['preference_terms'].values()))


def test_same_travel_concentrated_on_one_joint_has_extra_soft_cost():
    balanced = configuration_quality(
        configuration(joint1=0.5, joint2=0.5), HOME, LIMITS)
    concentrated = configuration_quality(
        configuration(joint1=1.0), HOME, LIMITS)
    assert concentrated['preference_terms']['largest_joint_motion'] > (
        balanced['preference_terms']['largest_joint_motion'])


def test_home_bend_does_not_prefer_a_wrist_branch():
    positive = configuration_quality(dict(HOME, joint5=0.6), HOME, LIMITS)
    negative = configuration_quality(dict(HOME, joint5=-0.6), HOME, LIMITS)
    assert positive['cost'] == negative['cost']
    assert positive['wrist_branch_change_penalty'] == 0.0
    assert negative['wrist_branch_change_penalty'] == 0.0


def test_existing_wrist_branch_has_a_finite_fading_preference():
    target = dict(HOME, joint5=-0.6)
    current = dict(HOME, joint5=0.6)
    opposite = configuration_quality(target, current, LIMITS)
    near_home = configuration_quality(target, dict(HOME, joint5=1e-7), LIMITS)
    matching = configuration_quality(target, target, LIMITS)
    assert opposite['wrist_branch_change_penalty'] == 1.0
    assert 0.0 < near_home['wrist_branch_change_penalty'] < 1e-5
    assert matching['wrist_branch_change_penalty'] == 0.0
    assert math.isfinite(opposite['cost'])


def test_wrist_clearance_preference_saturates_and_never_rejects():
    near = configuration_quality(dict(HOME, joint5=0.401), HOME, LIMITS)
    comfortable = configuration_quality(dict(HOME, joint5=0.6), HOME, LIMITS)
    bent = configuration_quality(dict(HOME, joint5=1.1), HOME, LIMITS)
    below = configuration_quality(dict(HOME, joint5=0.1), HOME, LIMITS)
    assert near['preference_terms']['wrist_clearance'] > 0.0
    assert comfortable['preference_terms']['wrist_clearance'] == 0.0
    assert bent['preference_terms']['wrist_clearance'] == 0.0
    assert below['wrist_clearance_rad'] < 0.0
    assert math.isfinite(below['cost'])


def test_bounded_positions_crossing_pi_are_not_wrapped():
    current = dict(HOME, joint6=-3.0)
    target = configuration(joint6=3.0)
    result = configuration_quality(target, current, LIMITS)
    assert result['joint_motion_rad']['joint6'] == pytest.approx(6.0)
    assert result['max_joint_motion_rad'] == pytest.approx(6.0)
    path = trajectory([[3.0]], names=('joint6',))
    metric = trajectory_quality(path, start_positions={'joint6': -3.0})
    assert metric['path_length_rad'] == pytest.approx(6.0)
    assert metric['wrist_total_variation_rad'] == pytest.approx(6.0)


def test_path_excursion_and_wrist_backtracking_are_penalized():
    direct = trajectory_quality(trajectory([[0, 0, 0], [1, 0, 0]]))
    detour = trajectory_quality(trajectory([[0, 0, 0], [2, 0, 0], [1, 0, 0]]))
    assert direct['path_length_rad'] == pytest.approx(1.0)
    assert direct['backtracking_rad'] == 0.0
    assert detour['path_length_rad'] == pytest.approx(3.0)
    assert detour['direct_distance_rad'] == pytest.approx(1.0)
    assert detour['excess_path_rad'] == pytest.approx(2.0)
    assert detour['wrist_backtracking_rad'] == pytest.approx(2.0)
    assert detour['cost'] > direct['cost']


def test_geometric_detour_without_axis_reversal_is_still_visible():
    direct = trajectory_quality(trajectory([[0, 0, 0], [1, 1, 0]]))
    detour = trajectory_quality(trajectory([[0, 0, 0], [1, 0, 0], [1, 1, 0]]))
    assert detour['backtracking_rad'] == 0.0
    assert detour['excess_path_rad'] == pytest.approx(2.0 - math.sqrt(2.0))
    assert detour['cost'] > direct['cost']


def test_metrics_ignore_retiming_and_collinear_waypoint_density():
    sparse = trajectory([[0, 0, 0], [1, 1, 1]])
    dense = trajectory([[0, 0, 0], [0.25, 0.25, 0.25], [1, 1, 1]])
    dense.points[1].time_from_start = SimpleNamespace(sec=500, nanosec=0)
    first = trajectory_quality(sparse)
    second = trajectory_quality(dense)
    assert first['cost'] == pytest.approx(second['cost'])
    assert first['path_length_rad'] == pytest.approx(second['path_length_rad'])


def test_start_state_is_accounted_for_without_modifying_message():
    message = trajectory([[1, 0.6, -1]])
    before = copy.deepcopy(message)
    result = trajectory_quality(
        SimpleNamespace(joint_trajectory=message), start_positions=HOME)
    assert result['path_length_rad'] == pytest.approx(math.sqrt(2.36))
    assert result['wrist_total_variation_rad'] == pytest.approx(2.6)
    assert message == before
    no_start = trajectory_quality(message)
    assert no_start['cost'] == 0.0


@pytest.mark.parametrize('positions,current,limits', [
    ({}, HOME, LIMITS),
    (configuration(), {}, LIMITS),
    (configuration(), HOME, {}),
    (dict(HOME, joint5=math.nan), HOME, LIMITS),
    (dict(HOME, joint5=math.inf), HOME, LIMITS),
    (configuration(), HOME, dict(LIMITS, joint1=(1.0, -1.0))),
    (configuration(), HOME, dict(LIMITS, joint1=('invalid', 1.0))),
])
def test_invalid_configuration_can_be_handled_by_baseline_fallback(
    positions, current, limits,
):
    with pytest.raises(ValueError):
        configuration_quality(positions, current, limits)


@pytest.mark.parametrize('path', [
    None,
    trajectory([]),
    trajectory([[0.0]], names=('joint4', 'joint4')),
    trajectory([[0.0]], names=('',)),
    trajectory([[0.0, 1.0]]),
    trajectory([[0.0, math.nan, 0.0]]),
])
def test_invalid_trajectory_metrics_raise_for_caller_fallback(path):
    with pytest.raises(ValueError):
        trajectory_quality(path)
