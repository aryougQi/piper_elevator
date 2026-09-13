"""Soft motion preferences for already validated coarse-approach candidates.

These costs never establish reachability, collision safety, joint clearance,
or controller smoothness. Bounded joint differences are deliberately not
wrapped. Lower costs are preferred; no metric defines a rejection threshold.
"""

import math
from collections.abc import Mapping

import numpy as np

from .coarse_approach_core import joint_configuration_cost


def _finite(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{label} must be finite') from error
    if not math.isfinite(result):
        raise ValueError(f'{label} must be finite')
    return result


def configuration_quality(
    positions, current, limits, *, wrist_joint='joint5',
    minimum_abs_wrist_bend=0.4,
    wrist_joints=('joint4', 'joint5', 'joint6'),
):
    """Return baseline cost, soft preference cost and physical motion metrics.

    The finite wrist-clearance penalty falls to zero at 0.15 rad beyond the
    existing protection line. It does not keep rewarding greater wrist bend.
    A branch-change preference fades to zero as either bend approaches Home.
    Invalid inputs raise ValueError so callers can retain their baseline.
    """
    if (
        not all(isinstance(value, Mapping)
                for value in (positions, current, limits))
        or not limits
    ):
        raise ValueError('Positions, current and limits need joint mappings')
    minimum = _finite(minimum_abs_wrist_bend, 'Minimum wrist bend')
    if minimum < 0.0:
        raise ValueError('Minimum wrist bend must be non-negative')
    if wrist_joint not in limits:
        raise ValueError(f'Missing wrist joint: {wrist_joint}')
    try:
        target = {name: _finite(positions[name], name) for name in limits}
        start = {name: _finite(current[name], name) for name in limits}
        bounds = {
            name: tuple(_finite(value, name) for value in values)
            for name, values in limits.items()
        }
    except (KeyError, TypeError) as error:
        raise ValueError('Incomplete joint configuration') from error
    baseline = joint_configuration_cost(target, start, bounds)
    if not math.isfinite(baseline):
        raise ValueError('Configuration does not have a finite baseline cost')

    delta = {name: target[name] - start[name] for name in limits}
    wrist_names = set(wrist_joints).intersection(limits)
    roll_names = wrist_names - {wrist_joint}
    wrist_motion = sum(abs(delta[name]) for name in wrist_names)
    roll_motion = sum(abs(delta[name]) for name in roll_names)
    max_motion = max(abs(value) for value in delta.values())
    clearance = abs(target[wrist_joint]) - minimum
    bend_penalty = min(1.0, max(0.0, 1.0 - clearance / 0.15)) ** 2
    branch_penalty = 0.0
    if target[wrist_joint] * start[wrist_joint] < 0.0:
        branch_penalty = (
            min(1.0, abs(start[wrist_joint]) / 0.15)
            * min(1.0, abs(target[wrist_joint]) / 0.15)
        )
    terms = {
        'wrist_motion': 0.08 * sum(
            delta[name] ** 2 for name in wrist_names),
        'wrist_roll_motion': 0.04 * sum(
            delta[name] ** 2 for name in roll_names),
        'largest_joint_motion': 0.04 * max_motion ** 2,
        'wrist_clearance': 0.10 * bend_penalty,
        'wrist_branch_change': 0.08 * branch_penalty,
    }
    preference = sum(terms.values())
    return {
        'cost': baseline + preference,
        'baseline_cost': baseline,
        'preference_cost': preference,
        'preference_terms': terms,
        'total_motion_rad': sum(abs(value) for value in delta.values()),
        'max_joint_motion_rad': max_motion,
        'wrist_motion_rad': wrist_motion,
        'wrist_roll_motion_rad': roll_motion,
        'wrist_clearance_rad': clearance,
        'wrist_branch_change_penalty': branch_penalty,
        'joint_motion_rad': {
            name: abs(value) for name, value in delta.items()
        },
    }


def trajectory_quality(
    trajectory, *, start_positions=None,
    wrist_joints=('joint4', 'joint5', 'joint6'),
):
    """Measure joint-space travel in returned waypoints without changing them.

    Accepts JointTrajectory or RobotTrajectory. Optional start positions add
    the initial displacement to the metrics, without inserting a ROS point.
    L2 waypoint path length and per-joint total variation describe geometry,
    not the controller's interpolated curve or velocity/acceleration/jerk.
    Retiming or adding collinear points therefore does not change the cost.
    """
    path = getattr(trajectory, 'joint_trajectory', trajectory)
    try:
        names = list(path.joint_names)
        points = list(path.points)
    except (AttributeError, TypeError) as error:
        raise ValueError('Expected a joint trajectory') from error
    if (
        not names or not all(isinstance(name, str) and name for name in names)
        or len(names) != len(set(names)) or not points
    ):
        raise ValueError('Trajectory needs unique joints and non-empty points')
    rows = []
    if start_positions is not None:
        if not isinstance(start_positions, Mapping):
            raise ValueError('Start positions must be a joint mapping')
        try:
            rows.append([
                _finite(start_positions[name], name) for name in names
            ])
        except KeyError as error:
            raise ValueError(
                f'Start position missing {error.args[0]}'
            ) from error
    for index, point in enumerate(points):
        try:
            values = np.asarray(point.positions, dtype=np.float64)
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f'Invalid positions at point {index}') from error
        if values.shape != (len(names),) or not np.all(np.isfinite(values)):
            raise ValueError(f'Invalid positions at point {index}')
        rows.append(values)
    values = np.asarray(rows, dtype=np.float64)
    steps = np.diff(values, axis=0)
    endpoint_delta = values[-1] - values[0]
    variations = np.sum(np.abs(steps), axis=0)
    direct_variations = np.abs(endpoint_delta)
    excess_variations = np.maximum(0.0, variations - direct_variations)
    length = float(np.sum(np.linalg.norm(steps, axis=1)))
    direct_distance = float(np.linalg.norm(endpoint_delta))
    excess = max(0.0, length - direct_distance)
    wrist_indices = [
        index for index, name in enumerate(names) if name in wrist_joints
    ]
    wrist_variation = float(np.sum(variations[wrist_indices]))
    wrist_backtracking = float(np.sum(excess_variations[wrist_indices]))
    backtracking = float(np.sum(excess_variations))
    terms = {
        'path_length': 0.05 * length,
        'excess_path': 0.15 * excess,
        'wrist_travel': 0.04 * wrist_variation,
        'backtracking': 0.06 * backtracking,
        'wrist_backtracking': 0.06 * wrist_backtracking,
    }
    result = {
        'cost': sum(terms.values()),
        'preference_terms': terms,
        'path_length_rad': length,
        'waypoint_path_length_rad': length,
        'direct_distance_rad': direct_distance,
        'excess_path_rad': excess,
        'waypoint_excess_path_rad': excess,
        'total_variation_rad': float(np.sum(variations)),
        'max_joint_variation_rad': float(np.max(variations)),
        'wrist_total_variation_rad': wrist_variation,
        'backtracking_rad': backtracking,
        'wrist_backtracking_rad': wrist_backtracking,
        'joint_total_variation_rad': {
            name: float(variations[index]) for index, name in enumerate(names)
        },
    }
    if not all(math.isfinite(value) for value in (
        result['cost'], length, direct_distance, result['total_variation_rad'],
    )):
        raise ValueError('Trajectory metrics must be finite')
    return result
