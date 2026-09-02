"""Tests for joint-effort contact detection."""

import numpy as np
import pytest

from piper_elevator_app.press_core import JointEffortContactDetector
from piper_elevator_app.press_core import PhaseTimer
from piper_elevator_app.press_core import simulated_button_depression


def test_phase_timer_accumulates_repeated_named_phases():
    samples = iter([10.0, 10.5, 11.5, 12.0, 13.0, 13.5])
    timer = PhaseTimer(clock=lambda: next(samples))

    timer.start('approach')
    timer.start('press')
    timer.start('approach')
    timer.stop()
    snapshot = timer.snapshot()

    assert snapshot['phases'] == {
        'approach': 2.0,
        'press': 0.5,
    }
    assert snapshot['total_seconds'] == 3.5


JOINTS = [f'joint{index}' for index in range(1, 7)]


def make_detector(**overrides):
    arguments = {
        'joint_names': JOINTS,
        'delta_thresholds': [1.0] * 6,
        'absolute_limits': [20.0] * 6,
        'baseline_sample_count': 3,
        'consecutive_samples': 2,
        'smoothing_alpha': 1.0,
    }
    arguments.update(overrides)
    return JointEffortContactDetector(**arguments)


def establish_baseline(detector, sample=None):
    values = [0.0] * 6 if sample is None else sample
    for _ in range(3):
        detector.add_baseline_sample(JOINTS, values)


def test_uses_median_baseline_and_joint_name_order():
    detector = make_detector()
    detector.add_baseline_sample(JOINTS, [1.0] * 6)
    detector.add_baseline_sample(JOINTS, [9.0] * 6)
    reverse_names = list(reversed(JOINTS))
    detector.add_baseline_sample(reverse_names, [1.0] * 6)

    assert detector.baseline_ready
    assert detector.baseline == pytest.approx([1.0] * 6)


def test_contact_requires_consecutive_filtered_increments():
    detector = make_detector()
    establish_baseline(detector)

    first = detector.update(JOINTS, [0.0, 1.2, 0.0, 0.0, 0.0, 0.0])
    second = detector.update(JOINTS, [0.0, 1.3, 0.0, 0.0, 0.0, 0.0])

    assert not first.contact
    assert second.contact
    assert not second.emergency


def test_contact_run_resets_after_a_quiet_sample():
    detector = make_detector()
    establish_baseline(detector)
    detector.update(JOINTS, [1.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    detector.update(JOINTS, [0.0] * 6)

    result = detector.update(JOINTS, [1.2, 0.0, 0.0, 0.0, 0.0, 0.0])

    assert not result.contact


def test_large_increment_or_absolute_torque_is_emergency():
    detector = make_detector(emergency_multiplier=2.0)
    establish_baseline(detector)

    residual_trip = detector.update(
        JOINTS,
        [0.0, 0.0, 2.1, 0.0, 0.0, 0.0],
    )
    absolute_trip = detector.update(
        JOINTS,
        [0.0, 0.0, 0.0, 21.0, 0.0, 0.0],
    )

    assert residual_trip.emergency
    assert absolute_trip.emergency


def test_rejects_nan_feedback_and_unconfigured_thresholds():
    detector = make_detector()
    assert not detector.add_baseline_sample(
        JOINTS,
        [np.nan] * 6,
    )
    with pytest.raises(ValueError):
        make_detector(delta_thresholds=[0.0] * 6)


def test_simulated_button_depression_uses_negative_prismatic_travel():
    assert simulated_button_depression(0.0, -0.0025) == pytest.approx(
        0.0025
    )
    assert simulated_button_depression(0.0, 0.001) == 0.0
    with pytest.raises(ValueError):
        simulated_button_depression(0.0, np.nan)
