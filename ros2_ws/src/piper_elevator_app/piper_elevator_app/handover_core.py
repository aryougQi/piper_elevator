"""Validate the coarse approach evidence consumed by visual Servo."""

import json
import math

import numpy as np


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite number')
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f'{name} must be a finite number') from error
    if not math.isfinite(number):
        raise ValueError(f'{name} must be a finite number')
    return number


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value


def _vector(value, size, name, *, normalize=False):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f'{name} must contain {size} finite numbers')
    vector = np.array([
        _finite_number(component, name) for component in value
    ], dtype=np.float64)
    if normalize:
        scale = float(np.max(np.abs(vector)))
        if scale == 0.0:
            raise ValueError(f'{name} must be nonzero')
        # Scale first so finite very large/small inputs cannot overflow or
        # underflow their norm into a false valid direction.
        vector /= scale
        vector /= np.linalg.norm(vector)
    return vector


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate handover field: {key}')
        result[key] = value
    return result


def _json_float(value):
    return _finite_number(float(value), 'JSON number')


def _reject_json_constant(value):
    raise ValueError(f'Nonfinite JSON number: {value}')


def decode_coarse_handover(
    message: str,
    *,
    selected_button: str,
    frame_id: str,
    now_ns: int,
    maximum_age_seconds: float = 0.75,
    future_tolerance_seconds: float = 0.02,
) -> dict:
    """Decode schema 1 evidence; the planner owns one-use token consumption.

    Observation source age must be in (-future_tolerance, maximum_age]. A
    claim may refresh the observation after verification, so verification
    need not be newer than the observation or share its freshness deadline.
    """
    if not isinstance(message, str):
        raise ValueError('Coarse handover message must be a JSON string')
    selected_button = _text(selected_button, 'Expected selected_button')
    frame_id = _text(frame_id, 'Expected frame_id')
    now_ns = _positive_integer(now_ns, 'now_ns')
    maximum_age_seconds = _finite_number(
        maximum_age_seconds, 'maximum_age_seconds')
    future_tolerance_seconds = _finite_number(
        future_tolerance_seconds, 'future_tolerance_seconds')
    if maximum_age_seconds <= 0.0 or future_tolerance_seconds < 0.0:
        raise ValueError('Maximum age must be positive and future tolerance nonnegative')
    try:
        payload = json.loads(
            message, object_pairs_hook=_json_object,
            parse_float=_json_float, parse_constant=_reject_json_constant)
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError(f'Invalid coarse handover JSON: {error}') from error
    if not isinstance(payload, dict):
        raise ValueError('Coarse handover JSON must be an object')
    if type(payload.get('schema_version')) is not int or payload['schema_version'] != 1:
        raise ValueError('Unsupported coarse handover schema_version; expected 1')
    _text(payload.get('handover_id'), 'handover_id')
    button_name = _text(payload.get('selected_button'), 'selected_button')
    if button_name.casefold() != selected_button.casefold():
        raise ValueError('Coarse handover selected_button does not match the current selection')
    if _text(payload.get('frame_id'), 'frame_id') != frame_id:
        raise ValueError('Coarse handover frame_id does not match the expected frame')
    observation_stamp = _positive_integer(
        payload.get('observation_stamp_ns'), 'observation_stamp_ns')
    verified_stamp = _positive_integer(payload.get('verified_at_ns'), 'verified_at_ns')
    age_ns = now_ns - observation_stamp
    if age_ns <= -future_tolerance_seconds * 1e9:
        raise ValueError('Coarse handover observation_stamp_ns is in the future')
    if age_ns > maximum_age_seconds * 1e9:
        raise ValueError('Coarse handover observation_stamp_ns is stale')
    if verified_stamp - now_ns > future_tolerance_seconds * 1e9:
        raise ValueError('Coarse handover verified_at_ns is in the future')
    for name, size, normalize in (
        ('button', 3, False), ('normal', 3, True),
        ('tcp_position', 3, False), ('tcp_orientation', 4, True),
    ):
        payload[name] = _vector(payload.get(name), size, name, normalize=normalize)
    joints = payload.get('joint_positions')
    if not isinstance(joints, dict) or not joints:
        raise ValueError('joint_positions must be a nonempty object')
    payload['joint_positions'] = {
        _text(name, 'Joint name'): _finite_number(value, f'Joint {name}')
        for name, value in joints.items()
    }
    return payload
