"""Bounded arm states, goal tolerances and controller spline validation."""

from collections.abc import Mapping
import math

import numpy as np
from numpy.polynomial import Polynomial


def _checked_limits(limits):
    if not isinstance(limits, Mapping) or not limits:
        raise ValueError('Arm position limits must be a non-empty mapping')
    result = {}
    for name, bounds in limits.items():
        if not isinstance(name, str) or not name:
            raise ValueError('Joint names must be non-empty strings')
        try:
            lower, upper = map(float, bounds)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f'{name}: expected lower and upper limits'
            ) from error
        if (
            not math.isfinite(lower) or not math.isfinite(upper)
            or lower >= upper
        ):
            raise ValueError(f'{name}: invalid limits [{lower}, {upper}] rad')
        result[name] = (lower, upper)
    return result


def _nonnegative(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} must be numeric') from error
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f'{name} must be finite and non-negative')
    return value


def _violation(name, value, lower, upper, tolerance, context):
    excess = max(lower - value, value - upper, 0.0)
    if excess <= tolerance:
        return None
    return (
        f'{name} {context}: value={value:.12g} rad outside '
        f'[{lower:.12g}, {upper:.12g}] rad; excess={excess:.12g} rad, '
        f'tolerance={tolerance:.12g} rad'
    )


def merge_joint_limits(urdf_limits, overrides):
    """Intersect enabled MoveIt position overrides with immutable URDF bounds.

    Unrelated joints and velocity/acceleration override fields are ignored.
    Disabling a position override never disables the physical URDF limits.
    """
    merged = _checked_limits(urdf_limits)
    if not isinstance(overrides, Mapping):
        raise ValueError('MoveIt joint limit overrides must be a mapping')
    for name, (lower, upper) in merged.items():
        override = overrides.get(name, {})
        if not isinstance(override, Mapping):
            raise ValueError(
                f'{name}: joint limit override must be a mapping'
            )
        enabled = override.get('has_position_limits', False)
        if not isinstance(enabled, bool):
            raise ValueError(f'{name}: has_position_limits must be boolean')
        if not enabled:
            continue
        try:
            requested = _checked_limits({name: (
                override['min_position'], override['max_position'],
            )})[name]
        except KeyError as error:
            raise ValueError(
                f'{name}: enabled position override needs both bounds'
            ) from error
        effective = (max(lower, requested[0]), min(upper, requested[1]))
        if effective[0] >= effective[1]:
            raise ValueError(
                f'{name}: MoveIt bounds {requested} and URDF bounds '
                f'{(lower, upper)} have no usable intersection'
            )
        merged[name] = effective
    return merged


def normalize_joint_positions(
    positions, limits, tolerance_rad, context='joint state',
):
    """Clamp only boundary roundoff, retaining every extra joint entry."""
    bounds = _checked_limits(limits)
    tolerance = _nonnegative(tolerance_rad, 'State bound tolerance')
    if not isinstance(positions, Mapping):
        raise ValueError(f'{context}: joint positions must be a mapping')
    normalized = {}
    for name, value in positions.items():
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f'{name} {context}: position is not numeric'
            ) from error
        if not math.isfinite(value):
            raise ValueError(
                f'{name} {context}: non-finite value={value} rad; '
                f'bounds={bounds.get(name, "uncontrolled")} rad'
            )
        normalized[name] = value
    corrections = {}
    for name, (lower, upper) in bounds.items():
        if name not in normalized:
            raise ValueError(
                f'{name} {context}: required arm joint is missing; '
                f'bounds=[{lower:.12g}, {upper:.12g}] rad'
            )
        value = normalized[name]
        message = _violation(name, value, lower, upper, tolerance, context)
        if message:
            raise ValueError(message)
        bounded = min(upper, max(lower, value))
        if bounded != value:
            normalized[name] = bounded
            corrections[name] = (value, bounded)
    return normalized, corrections


def bounded_goal_tolerances(position, lower, upper, tolerance, margin=0.0):
    """Return (below, above) without extending a goal outside its envelope."""
    lower, upper = _checked_limits({'goal': (lower, upper)})['goal']
    try:
        position = float(position)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('Goal position must be numeric') from error
    tolerance = _nonnegative(tolerance, 'Goal tolerance')
    margin = _nonnegative(margin, 'Goal margin')
    lower += margin
    upper -= margin
    if lower > upper or not math.isfinite(position):
        raise ValueError('Goal margin or position creates an invalid envelope')
    message = _violation('goal', position, lower, upper, 0.0, 'center')
    if message:
        raise ValueError(message)
    return min(tolerance, position - lower), min(tolerance, upper - position)


def _point_vector(point, field, count, index, optional=False):
    values = np.asarray(getattr(point, field, []), dtype=np.float64)
    if optional and values.shape == (0,):
        return None
    if values.shape != (count,) or not np.all(np.isfinite(values)):
        raise ValueError(
            f'point {index}: {field} must contain {count} finite values'
        )
    return values


def _segment_polynomial(first, second, duration, joint_index):
    p0, v0, a0, _ = first
    p1, v1, a1, _ = second
    start, delta = p0[joint_index], p1[joint_index] - p0[joint_index]
    if v0 is None:
        return Polynomial([start, delta])
    initial = v0[joint_index] * duration
    final = v1[joint_index] * duration
    if a0 is None:
        return Polynomial([
            start, initial,
            3.0 * delta - 2.0 * initial - final,
            -2.0 * delta + initial + final,
        ])
    initial_acc = a0[joint_index] * duration * duration
    final_acc = a1[joint_index] * duration * duration
    return Polynomial([
        start, initial, initial_acc / 2.0,
        10.0 * delta - 6.0 * initial - 4.0 * final
        - 1.5 * initial_acc + 0.5 * final_acc,
        -15.0 * delta + 8.0 * initial + 7.0 * final
        + 1.5 * initial_acc - final_acc,
        6.0 * delta - 3.0 * initial - 3.0 * final
        - 0.5 * initial_acc + 0.5 * final_acc,
    ])


def trajectory_position_limit_violation(
    joint_names, points, limits, tolerance_rad,
):
    """Check waypoints and all extrema of JTC linear/cubic/quintic segments.

    Derivative fields must be uniformly present or absent across all points.
    Accelerations require velocities. Times must be non-negative and strictly
    increasing. Every controlled arm joint must occur exactly once. Extrema
    use polynomial derivative roots in normalized segment time, not sampling.
    The caller must include the actual start state as the first waypoint.
    """
    try:
        bounds = _checked_limits(limits)
        tolerance = _nonnegative(tolerance_rad, 'Trajectory bound tolerance')
        names = list(joint_names)
        if (
            not names
            or not all(isinstance(name, str) and name for name in names)
            or len(names) != len(set(names))
        ):
            raise ValueError(
                'Trajectory joint names must be non-empty and unique'
            )
        missing = set(bounds) - set(names)
        if missing:
            raise ValueError(
                f'Trajectory missing arm joints: {sorted(missing)}'
            )
        points = list(points)
        if not points:
            raise ValueError('Trajectory contains no points')
        data = []
        for index, point in enumerate(points):
            position = _point_vector(point, 'positions', len(names), index)
            velocity = _point_vector(
                point, 'velocities', len(names), index, optional=True,
            )
            acceleration = _point_vector(
                point, 'accelerations', len(names), index, optional=True,
            )
            if acceleration is not None and velocity is None:
                raise ValueError(
                    f'point {index}: accelerations need velocities'
                )
            if data and (
                (velocity is None) != (data[0][1] is None)
                or (acceleration is None) != (data[0][2] is None)
            ):
                raise ValueError(
                    f'point {index}: mixed derivative availability'
                )
            stamp = point.time_from_start
            sec, nanosec = float(stamp.sec), float(stamp.nanosec)
            if (
                not math.isfinite(sec) or not math.isfinite(nanosec)
                or sec < 0.0 or not 0.0 <= nanosec < 1e9
                or not sec.is_integer() or not nanosec.is_integer()
            ):
                raise ValueError(f'point {index}: invalid time_from_start')
            elapsed = sec + nanosec * 1e-9
            if (
                not math.isfinite(elapsed)
                or (data and elapsed <= data[-1][3])
            ):
                raise ValueError(
                    f'point {index}: times must strictly increase'
                )
            for name, (lower, upper) in bounds.items():
                message = _violation(
                    name, position[names.index(name)], lower, upper, tolerance,
                    f'point {index} time={elapsed:.12g}s',
                )
                if message:
                    return message
            data.append((position, velocity, acceleration, elapsed))

        for index, (first, second) in enumerate(zip(data, data[1:])):
            duration = second[3] - first[3]
            for name, (lower, upper) in bounds.items():
                with np.errstate(
                    over='raise', invalid='raise', divide='raise'
                ):
                    polynomial = _segment_polynomial(
                        first, second, duration, names.index(name),
                    )
                    if not np.all(np.isfinite(polynomial.coef)):
                        raise ValueError(
                            f'{name} segment {index}: invalid spline'
                        )
                    roots = polynomial.deriv().roots()
                    if not np.all(np.isfinite(roots)):
                        raise ValueError(
                            f'{name} segment {index}: invalid spline extrema'
                        )
                    extrema = [
                        float(root.real) for root in roots
                        if abs(root.imag) <= 1e-9 and 0.0 < root.real < 1.0
                    ]
                    for fraction in sorted(extrema):
                        value = float(polynomial(fraction))
                        if not math.isfinite(value):
                            raise ValueError(
                                f'{name} segment {index}: invalid value'
                            )
                        message = _violation(
                            name, value, lower, upper, tolerance,
                            f'segment {index}->{index + 1} '
                            f'time={first[3] + fraction * duration:.12g}s',
                        )
                        if message:
                            return message
        return None
    except (AttributeError, TypeError, ValueError, OverflowError,
            FloatingPointError, np.linalg.LinAlgError) as error:
        return f'Invalid trajectory: {error}'
