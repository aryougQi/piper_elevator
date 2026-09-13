"""Recover semantic holds without bypassing verified coarse handovers."""

import time

import numpy as np
import pytest
from std_srvs.srv import Trigger

from test_servo_observation_safety import surface, transform
from test_visual_semantic_safety import (
    SemanticHarness, start_harness, status_message,
)


@pytest.mark.parametrize('lag_frames', [1, 2, 4])
def test_positive_stream_does_not_starve_delayed_surface_recovery(lag_frames):
    servo = SemanticHarness()
    servo._running = False
    servo._tracking_state_callback(status_message(
        9_950_000_000, 'projection_conflict',
    ))
    for index in range(1, lag_frames + 2):
        capture = 10_000_000_000 + index * 100_000_000
        servo.now_ns = capture + 50_000_000
        servo._tracking_state_callback(status_message(capture, ''))
        if index > lag_frames:
            servo._surface_pose_callback(surface(capture - lag_frames * 100_000_000))
        else:
            assert servo.held
    assert not servo.held
    assert servo._observation_is_fresh_locked()
    assert not servo.commands


def test_expired_recovery_candidate_is_replaced_by_new_positive_capture():
    servo = SemanticHarness()
    servo._running = False
    servo._tracking_state_callback(status_message())
    servo._tracking_state_callback(status_message(9_970_000_000, ''))
    assert servo.held
    servo.now_ns = 10_800_000_000
    servo._tracking_state_callback(status_message(10_770_000_000, ''))
    assert servo._semantic_positive_stamp_ns == 10_770_000_000
    assert servo.held
    servo._surface_pose_callback(surface(10_770_000_000))
    assert not servo.held


def recovering_start(monkeypatch, *, near_x=0.02, deliver_surface=True):
    servo = start_harness(monkeypatch)
    servo._observation_anchor = np.array([0.0, 0.0, 0.4])
    servo._filtered_world_position = servo._observation_anchor.copy()
    servo._filtered_world_normal = np.array([0.0, 0.0, 1.0])
    servo._observation_stamp_ns = 9_940_000_000
    servo._observation = (
        servo._observation_anchor.copy(), servo._filtered_world_normal.copy(),
        time.monotonic(), 1,
    )
    servo._lookup_surface_transform = lambda frame, stamp: transform()
    servo._tracking_state_callback(status_message(
        9_950_000_000, 'projection_conflict',
    ))
    servo._tracking_state_callback(status_message(9_970_000_000, ''))
    if deliver_surface:
        servo._surface_pose_callback(surface(9_970_000_000, x=near_x))
    assert servo._observation_stamp_ns == 9_940_000_000
    if deliver_surface:
        assert any('IGNORED_TARGET_JUMP' in message for message in servo.statuses)
    assert servo._semantic_conflict_locked() == 'projection_conflict'
    servo.payload.update(
        observation_stamp_ns=9_970_000_000,
        button=[near_x, 0.0, 0.4], normal=[0.0, 0.0, 1.0],
    )
    return servo


def test_positive_but_rejected_near_view_requires_a_verified_coarse_token(monkeypatch):
    servo = recovering_start(monkeypatch)
    old_anchor = servo._observation_anchor.copy()
    servo.payload = None
    for expected_claims in range(1, 4):
        response = servo._start_callback(None, Trigger.Response())
        assert not response.success
        assert 'no verified coarse handover' in response.message
        assert 're-run coarse approach' in response.message
        assert servo.claim_calls == expected_claims
        np.testing.assert_array_equal(servo._observation_anchor, old_anchor)
        assert servo._semantic_conflict_locked() == 'projection_conflict'
    assert not servo.started_threads
    assert not servo._running
    assert not servo._starting
    assert not servo.gate_calls


def test_verified_near_handover_replaces_old_anchor_before_start(monkeypatch):
    servo = recovering_start(monkeypatch)
    # More positive statuses arrive while claiming; they must not move the
    # recovery requirement beyond an already valid token's capture.
    servo.during_claim = lambda: servo._tracking_state_callback(
        status_message(9_980_000_000, ''),
    )
    response = servo._start_callback(None, Trigger.Response())
    assert response.success, response.message
    assert servo.claim_calls == 1
    assert len(servo.started_threads) == 1
    np.testing.assert_array_equal(servo._observation_anchor, [0.02, 0.0, 0.4])
    assert servo._observation_stamp_ns == 9_970_000_000
    assert servo._semantic_conflict_locked() == ''
    assert not servo.gate_calls


def test_hold_clearing_during_claim_keeps_original_fresh_evidence_snapshot(monkeypatch):
    servo = recovering_start(monkeypatch, near_x=0.0, deliver_surface=False)

    def receive_delayed_surface():
        servo._surface_pose_callback(surface(9_970_000_000))
        assert servo._semantic_conflict_locked() == ''
        servo._tracking_state_callback(status_message(9_990_000_000, ''))

    servo.during_claim = receive_delayed_surface
    response = servo._start_callback(None, Trigger.Response())
    assert response.success, response.message
    assert servo.claim_calls == 1
    assert len(servo.started_threads) == 1
    assert servo._observation_stamp_ns == 9_970_000_000


@pytest.mark.parametrize('token_stamp', [9_000_000_000, 9_960_000_000])
def test_stale_or_pre_recovery_token_cannot_reset_anchor(monkeypatch, token_stamp):
    servo = recovering_start(monkeypatch)
    old_anchor = servo._observation_anchor.copy()
    servo.payload['observation_stamp_ns'] = token_stamp
    response = servo._start_callback(None, Trigger.Response())
    assert not response.success
    assert 're-run coarse approach' in response.message
    assert servo.claim_calls == 1
    assert not servo.started_threads
    np.testing.assert_array_equal(servo._observation_anchor, old_anchor)
    assert servo._semantic_conflict_locked() == 'projection_conflict'


def test_positive_evidence_expiring_during_claim_prevents_start(monkeypatch):
    servo = recovering_start(monkeypatch)
    old_anchor = servo._observation_anchor.copy()

    def delayed_claim():
        servo.now_ns += 1_000_000_000
        servo.payload['observation_stamp_ns'] = servo.now_ns - 20_000_000

    servo.during_claim = delayed_claim
    response = servo._start_callback(None, Trigger.Response())
    assert not response.success
    assert 'positive RGB-D evidence expired' in response.message
    assert not servo.started_threads
    np.testing.assert_array_equal(servo._observation_anchor, old_anchor)


@pytest.mark.parametrize('recovers_again', [False, True])
def test_new_negative_during_claim_aborts_even_if_positive_returns(
    monkeypatch, recovers_again,
):
    servo = recovering_start(monkeypatch)
    old_anchor = servo._observation_anchor.copy()

    def conflict_during_claim():
        servo._tracking_state_callback(status_message(
            9_980_000_000, 'direction_conflict',
        ))
        if recovers_again:
            servo._tracking_state_callback(status_message(9_990_000_000, ''))
            servo._surface_pose_callback(surface(9_990_000_000))
        servo.payload['observation_stamp_ns'] = 9_990_000_000

    servo.during_claim = conflict_during_claim
    response = servo._start_callback(None, Trigger.Response())
    assert not response.success
    assert 'New semantic conflict during coarse handover' in response.message
    assert servo.claim_calls == 1
    assert not servo.started_threads
    np.testing.assert_array_equal(servo._observation_anchor, old_anchor)
    assert not servo.gate_calls
