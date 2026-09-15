"""Bounded, deterministic plane consensus for small RGB-D regions.

Algorithm reference: Open3D PointCloud.segment_plane (MIT), see
docs/vision_geometry.md. This NumPy implementation is independently written.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlaneEstimate:
    normal: np.ndarray
    offset: float
    inlier_ratio: float
    rmse_m: float


def fit_plane_consensus(points, *, threshold_m=0.004, minimum_samples=30,
                        minimum_inlier_ratio=0.70, iterations=96):
    """Return a supported plane, or None for ambiguous/degenerate geometry.

    Distances are perpendicular to the plane, not differences in depth.
    No previous-frame points or hole filling are used as new evidence.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must have shape (N, 3)')
    if not np.isfinite(threshold_m) or threshold_m <= 0:
        raise ValueError('threshold_m must be finite and positive')
    if not 0.5 < minimum_inlier_ratio <= 1.0:
        raise ValueError('minimum_inlier_ratio must be in (0.5, 1]')
    if iterations < 1 or minimum_samples < 3:
        raise ValueError('iterations must be positive and minimum_samples >= 3')
    points = points[np.all(np.isfinite(points), axis=1)]
    required = max(minimum_samples, int(np.ceil(len(points) * minimum_inlier_ratio)))
    if len(points) < required:
        return None
    rng = np.random.default_rng(0)
    best_mask = None
    best_score = (0, -np.inf)
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-12:
            continue
        normal /= length
        distances = np.abs((points - a) @ normal)
        mask = distances <= threshold_m
        count = int(mask.sum())
        score = (count, -float(np.mean(distances[mask] ** 2)))
        if count >= required and score > best_score:
            best_mask, best_score = mask, score
    if best_mask is None:
        return None
    # Refine and re-evaluate consensus. A failed quality gate never falls back
    # to an unconditional SVD plane, which would defeat rejection.
    for _ in range(3):
        support = points[best_mask]
        center = support.mean(axis=0)
        _, singular, right = np.linalg.svd(support - center, full_matrices=False)
        spread = singular / np.sqrt(len(support))
        if spread[1] < 1e-4 or spread[1] < 0.05 * spread[0]:
            return None
        normal = right[-1]
        distances = np.abs((points - center) @ normal)
        mask = distances <= threshold_m
        if int(mask.sum()) < required:
            return None
        if np.array_equal(mask, best_mask):
            break
        best_mask = mask
    if np.dot(normal, center) < 0:
        normal = -normal
    return PlaneEstimate(
        normal=normal,
        offset=-float(normal @ center),
        inlier_ratio=float(mask.mean()),
        rmse_m=float(np.sqrt(np.mean(distances[mask] ** 2))),
    )
