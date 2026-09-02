"""Pure helpers for compensating simulated Servo position tracking lag."""

import math


def compensated_positions(
    joint_names,
    desired_positions,
    current_by_name,
    position_gain,
    maximum_lead_rad,
    lower_by_name,
    upper_by_name,
):
    """Return bounded position leads relative to measured joint positions."""
    names = list(joint_names)
    desired = list(desired_positions)
    gain = float(position_gain)
    maximum_lead = float(maximum_lead_rad)

    if not names or len(names) != len(desired):
        raise ValueError('joint names and desired positions must match')
    if not math.isfinite(gain) or gain < 1.0:
        raise ValueError('position_gain must be finite and at least 1.0')
    if not math.isfinite(maximum_lead) or maximum_lead <= 0.0:
        raise ValueError('maximum_lead_rad must be finite and positive')

    compensated = []
    for name, target in zip(names, desired):
        if (
            name not in current_by_name
            or name not in lower_by_name
            or name not in upper_by_name
        ):
            raise ValueError(f'missing state or limits for joint {name!r}')
        current = float(current_by_name[name])
        target = float(target)
        lower = float(lower_by_name[name])
        upper = float(upper_by_name[name])
        if not all(math.isfinite(value) for value in (
            current, target, lower, upper,
        )):
            raise ValueError(f'non-finite data for joint {name!r}')
        if lower >= upper:
            raise ValueError(f'invalid limits for joint {name!r}')

        lead = gain * (target - current)
        lead = max(-maximum_lead, min(maximum_lead, lead))
        compensated.append(max(lower, min(upper, current + lead)))

    return compensated
