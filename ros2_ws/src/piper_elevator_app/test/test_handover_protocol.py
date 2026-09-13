"""Validate coarse/Servo evidence without DDS or robot motion."""

import json
import math

import numpy as np
import pytest

from piper_elevator_app.handover_core import decode_coarse_handover


NOW_NS = 1788866811120085000


def evidence(**changes):
    payload = {
        'schema_version': 1,
        'handover_id': 'approach-down-1',
        'selected_button': 'down',
        'frame_id': 'base_link',
        'observation_stamp_ns': NOW_NS - 100_000_000,
        'verified_at_ns': NOW_NS - 5_000_000_000,
        'button': [0.55, 0.03, 0.45],
        'normal': [2.0, 0.0, 0.0],
        'tcp_position': [0.35, 0.02, 0.45],
        'tcp_orientation': [0.0, 0.0, 0.0, 2.0],
        'joint_positions': {'joint1': 0, 'joint5': -0.6},
    }
    payload.update(changes)
    return payload


def decode(payload=None, **arguments):
    values = dict(selected_button='down', frame_id='base_link', now_ns=NOW_NS)
    values.update(arguments)
    return decode_coarse_handover(json.dumps(evidence() if payload is None else payload), **values)


def test_verified_handover_normalizes_geometry_and_preserves_extension_fields():
    original = evidence(diagnostics={'path': 'pilz_ptp'})
    result = decode(original, selected_button='DOWN')
    for name in ('button', 'normal', 'tcp_position', 'tcp_orientation'):
        assert isinstance(result[name], np.ndarray)
        assert result[name].dtype == np.float64
    np.testing.assert_allclose(result['button'], original['button'])
    np.testing.assert_allclose(result['tcp_position'], original['tcp_position'])
    np.testing.assert_allclose(result['normal'], [1., 0., 0.])
    np.testing.assert_allclose(result['tcp_orientation'], [0., 0., 0., 1.])
    assert all(type(value) is float for value in result['joint_positions'].values())
    assert result['diagnostics'] == original['diagnostics']
    assert result['handover_id'] == original['handover_id']
    assert result['verified_at_ns'] < result['observation_stamp_ns']
    assert original['normal'] == [2., 0., 0.]


@pytest.mark.parametrize('missing', list(evidence()))
def test_missing_required_field_is_rejected(missing):
    payload = evidence()
    del payload[missing]
    with pytest.raises(ValueError, match=missing):
        decode(payload)


@pytest.mark.parametrize('message', [
    '', '{', 'null', '[]', 'true', '1', '"payload"',
    b'{}', None, {},
])
def test_malformed_or_nonobject_message_is_rejected(message):
    with pytest.raises(ValueError):
        decode_coarse_handover(
            message, selected_button='down', frame_id='base_link', now_ns=NOW_NS)


@pytest.mark.parametrize(('field', 'invalid'), [
    ('schema_version', True), ('schema_version', 1.0), ('schema_version', '1'),
    ('schema_version', 2), ('handover_id', ''), ('handover_id', '  '),
    ('handover_id', 123), ('selected_button', ''), ('selected_button', 'up'),
    ('selected_button', True), ('selected_button', ' down'),
    ('frame_id', ''), ('frame_id', 'BASE_LINK'), ('frame_id', 1),
])
def test_schema_and_identity_are_strict(field, invalid):
    with pytest.raises(ValueError, match=field):
        decode(evidence(**{field: invalid}))


@pytest.mark.parametrize('field', ['observation_stamp_ns', 'verified_at_ns'])
@pytest.mark.parametrize('invalid', [True, 0, -1, 1.0, '1788866811120085000', None])
def test_timestamps_must_be_positive_integer_nanoseconds(field, invalid):
    with pytest.raises(ValueError, match=field):
        decode(evidence(**{field: invalid}))


@pytest.mark.parametrize(('age_ns', 'accepted'), [
    (750_000_000, True), (750_000_001, False),
    (0, True), (-19_999_999, True), (-20_000_000, False),
    (-20_000_001, False),
])
def test_observation_freshness_boundaries_keep_nanosecond_precision(age_ns, accepted):
    payload = evidence(observation_stamp_ns=NOW_NS - age_ns)
    if accepted:
        decode(payload)
    else:
        with pytest.raises(ValueError, match='observation_stamp_ns'):
            decode(payload)


def test_claim_can_refresh_an_observation_long_after_initial_verification():
    decode(evidence(verified_at_ns=1, observation_stamp_ns=NOW_NS))
    decode(evidence(verified_at_ns=NOW_NS + 20_000_000))
    with pytest.raises(ValueError, match='verified_at_ns.*future'):
        decode(evidence(verified_at_ns=NOW_NS + 20_000_001))


@pytest.mark.parametrize(('field', 'size'), [
    ('button', 3), ('normal', 3), ('tcp_position', 3), ('tcp_orientation', 4),
])
@pytest.mark.parametrize('invalid', [None, 'vector', {}, [], [1.0, 2.0]])
def test_geometry_requires_exact_array_dimensions(field, size, invalid):
    with pytest.raises(ValueError, match=field):
        decode(evidence(**{field: invalid}))


@pytest.mark.parametrize(('field', 'size'), [
    ('button', 3), ('normal', 3), ('tcp_position', 3), ('tcp_orientation', 4),
])
@pytest.mark.parametrize('invalid', [True, None, '1.0', [1.0], math.nan, math.inf, -math.inf])
def test_geometry_components_must_be_finite_numbers(field, size, invalid):
    with pytest.raises(ValueError):
        decode(evidence(**{field: [invalid] + [1.] * (size - 1)}))


@pytest.mark.parametrize(('field', 'size'), [('normal', 3), ('tcp_orientation', 4)])
def test_normalization_rejects_zero_and_handles_finite_extreme_magnitudes(field, size):
    with pytest.raises(ValueError, match=field + '.*nonzero'):
        decode(evidence(**{field: [0.] * size}))
    for magnitude in (1e308, 1e-320):
        vector = decode(evidence(**{field: [magnitude] * size}))[field]
        assert np.isfinite(vector).all()
        assert np.linalg.norm(vector) == pytest.approx(1.)


@pytest.mark.parametrize('invalid', [
    {}, [], None, {'': 0.1}, {' ': 0.1}, {'joint1': True},
    {'joint1': '0.1'}, {'joint1': None}, {'joint1': math.nan},
    {'joint1': math.inf}, {'joint1': -math.inf}, {'joint1': 10 ** 400},
])
def test_joint_map_must_have_names_and_finite_numeric_positions(invalid):
    with pytest.raises(ValueError):
        decode(evidence(joint_positions=invalid))


def test_duplicate_json_fields_and_nonfinite_extensions_are_rejected():
    message = json.dumps(evidence())[:-1] + ', "selected_button": "up"}'
    with pytest.raises(ValueError, match='Duplicate'):
        decode_coarse_handover(
            message, selected_button='down', frame_id='base_link', now_ns=NOW_NS)
    with pytest.raises(ValueError, match='finite'):
        decode(evidence(extension=math.inf))
    message = json.dumps(evidence())[:-1] + ', "extension": 1e999}'
    with pytest.raises(ValueError, match='finite'):
        decode_coarse_handover(
            message, selected_button='down', frame_id='base_link', now_ns=NOW_NS)


@pytest.mark.parametrize('arguments', [
    {'selected_button': ''}, {'frame_id': ''}, {'now_ns': 0}, {'now_ns': True},
    {'maximum_age_seconds': 0.}, {'maximum_age_seconds': -1.},
    {'maximum_age_seconds': True}, {'maximum_age_seconds': math.inf},
    {'future_tolerance_seconds': -1.}, {'future_tolerance_seconds': math.nan},
])
def test_invalid_consumer_context_is_rejected(arguments):
    with pytest.raises(ValueError):
        decode(**arguments)


def test_decoding_does_not_consume_or_share_token_state():
    first = decode()
    first['button'][0] = 42.
    second = decode()
    assert first['handover_id'] == second['handover_id']
    np.testing.assert_allclose(second['button'], [.55, .03, .45])
