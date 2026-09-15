"""Geometric failure cases for RGB-D surface estimation."""

import numpy as np
import pytest

from piper_elevator_app.plane_core import fit_plane_consensus


def plane(seed=7, count=300):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-0.04, 0.04, (count, 2))
    z = 0.5 - 0.3 * xy[:, 0] + 0.15 * xy[:, 1]
    return np.column_stack((xy, z)), rng


def test_tilted_plane_with_outliers_recovers_normal():
    points, rng = plane()
    points[:, 2] += rng.normal(0, 0.0002, len(points))
    points[:60, 2] += rng.uniform(0.015, 0.06, 60)
    result = fit_plane_consensus(points, threshold_m=0.001)
    expected = np.array([0.3, -0.15, 1.0])
    expected /= np.linalg.norm(expected)
    assert result is not None
    assert np.degrees(np.arccos(np.clip(result.normal @ expected, -1, 1))) < 0.3
    assert result.inlier_ratio == pytest.approx(0.8)
    assert result.rmse_m < 0.0003
    assert result.offset < 0


def test_two_surfaces_without_dominant_support_are_rejected():
    points, _ = plane()
    points[::2, 2] += 0.03
    assert fit_plane_consensus(points, threshold_m=0.001) is None


def test_holes_leaving_only_a_depth_scanline_are_rejected():
    x = np.linspace(-0.04, 0.04, 100)
    points = np.column_stack((x, np.zeros_like(x), np.full_like(x, 0.5)))
    assert fit_plane_consensus(points) is None


def test_nonfinite_points_do_not_become_plane_evidence():
    points, _ = plane(count=40)
    points[:20] = np.nan
    assert fit_plane_consensus(points, minimum_samples=30) is None


def test_replay_is_deterministic():
    points, rng = plane()
    points[:40, 2] += rng.uniform(0.01, 0.1, 40)
    first = fit_plane_consensus(points)
    second = fit_plane_consensus(points)
    np.testing.assert_array_equal(first.normal, second.normal)
    assert first.inlier_ratio == second.inlier_ratio


@pytest.mark.parametrize('kwargs', [
    {'threshold_m': float('nan')}, {'threshold_m': 0},
    {'minimum_inlier_ratio': 0.5}, {'minimum_inlier_ratio': 1.1},
    {'iterations': 0}, {'minimum_samples': 2},
])
def test_invalid_quality_parameters_fail_explicitly(kwargs):
    points, _ = plane()
    with pytest.raises(ValueError):
        fit_plane_consensus(points, **kwargs)
