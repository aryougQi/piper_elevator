"""Offline-only button geometry experiments; not imported by production nodes.

Open-source algorithm references and limitations: docs/vision_geometry.md.
Boxes use source-image (x1, y1, x2, y2), with exclusive upper pixel bounds.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from piper_elevator_app.plane_core import fit_plane_consensus


@dataclass(frozen=True)
class CenterEstimate:
    center: object = None
    corners: object = None
    reason: str = 'no_button_boundary'
    candidates: int = 0
    edge_refined: bool = False
    edge_residual_px: object = None


@dataclass(frozen=True)
class SurfaceEstimate:
    plane: object = None
    panel_point: object = None
    reason: str = 'insufficient_panel_support'
    samples: int = 0
    quadrants: int = 0


def _box(box):
    box = np.asarray(box, dtype=float)
    if box.shape != (4,) or not np.all(np.isfinite(box)):
        raise ValueError('box must contain four finite xyxy coordinates')
    if np.any(box[2:] <= box[:2]):
        raise ValueError('box must have positive width and height')
    return box


def _expanded(box, scale):
    center = (box[:2] + box[2:]) / 2
    half = (box[2:] - box[:2]) * scale / 2
    return np.r_[center - half, center + half]


def _rect_mask(shape, box):
    x1, y1, x2, y2 = box
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)


def _refine_quad_edges(contour, corners):
    """Fit the middle of each observed side, excluding rounded corner pixels."""
    points = contour.reshape(-1, 2).astype(float)
    lines, residuals = [], []
    for a, b in zip(corners, np.roll(corners, -1, axis=0)):
        delta = b-a
        length = np.linalg.norm(delta)
        if length < 8:
            return None
        tangent = delta/length
        relative = points-a
        along = relative @ tangent
        distances = np.abs(relative @ np.array([-tangent[1], tangent[0]]))
        support = points[(along > .15*length) & (along < .85*length) & (distances <= 2.5)]
        if len(support) < 5:
            return None
        vx, vy, x, y = cv2.fitLine(support.astype(np.float32), cv2.DIST_L1, 0, .01, .01).reshape(4)
        normal = np.array([-vy, vx], dtype=float)
        residual = float(np.percentile(np.abs((support-[x, y]) @ normal), 90))
        if residual > 1.2:
            return None
        lines.append(np.r_[normal, -normal @ [x, y]])
        residuals.append(residual)
    refined = []
    for previous, current in zip(np.roll(lines, 1, axis=0), lines):
        point = np.cross(previous, current)
        if abs(point[2]) < .1:
            return None
        refined.append(point[:2]/point[2])
    refined = np.array(refined)
    if (not cv2.isContourConvex(refined.astype(np.float32).reshape(-1, 1, 2))
            or np.max(np.linalg.norm(refined-corners, axis=1)) > 3):
        return None
    return refined, max(residuals)


def refine_rectangular_center(image, box, *, refine_edges=True):
    """Locate a complete quadrilateral boundary; reject icons and ambiguity.

    The diagonal intersection is the projected center of a planar rectangle.
    No box-center fallback is reported as a successful refinement. Circular,
    occluded and heavily distorted buttons are outside this first prototype.
    """
    box = _box(box)
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError('image must be uint8 grayscale or BGR')
    height, width = image.shape[:2]
    if box[0] <= 1 or box[1] <= 1 or box[2] >= width-1 or box[3] >= height-1:
        return CenterEstimate(reason='clipped_detection')
    size = box[2:] - box[:2]
    if np.min(size) < 12:
        return CenterEstimate(reason='button_too_small')
    roi = _expanded(box, 1.3)
    x1, y1 = np.maximum(np.floor(roi[:2]), 0).astype(int)
    x2, y2 = np.minimum(np.ceil(roi[2:]), [width, height]).astype(int)
    gray = image[y1:y2, x1:x2]
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks = [cv2.Canny(gray, 40, 120), threshold, cv2.bitwise_not(threshold)]
    candidates = []
    box_center = (box[:2] + box[2:]) / 2
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = abs(cv2.contourArea(contour))
            if not 0.45 <= area / np.prod(size) <= 1.35:
                continue
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            local = polygon.reshape(4, 2)
            if (np.any(local <= 1) or np.any(local[:, 0] >= x2-x1-2)
                    or np.any(local[:, 1] >= y2-y1-2)):
                continue
            corners = local.astype(float) + [x1, y1]
            extent = np.ptp(corners, axis=0)
            if np.any(extent / size < 0.65) or np.any(extent / size > 1.25):
                continue
            # Reject jagged icon contours approximated by an enclosing quad.
            if abs(cv2.contourArea(polygon) - area) / area > 0.08:
                continue
            edge_result = (_refine_quad_edges(contour + np.array([x1, y1]), corners)
                           if refine_edges else None)
            if edge_result is not None:
                corners = edge_result[0]
            homogeneous = np.column_stack((corners, np.ones(4)))
            p = np.cross(np.cross(homogeneous[0], homogeneous[2]),
                         np.cross(homogeneous[1], homogeneous[3]))
            if abs(p[2]) < 1e-9:
                continue
            center = p[:2] / p[2]
            if np.linalg.norm((center - box_center) / size) > 0.25:
                continue
            score = area / np.prod(size)
            candidates.append((score, center, corners, edge_result))
    if not candidates:
        return CenterEstimate()
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    best = candidates[0]
    for candidate in candidates[1:]:
        if (candidate[0] >= 0.8 * best[0]
                and np.linalg.norm(candidate[1]-best[1]) > 0.08*np.min(size)):
            return CenterEstimate(reason='ambiguous_boundaries', candidates=len(candidates))
    return CenterEstimate(best[1], best[2], 'accepted', len(candidates),
                          best[3] is not None, None if best[3] is None else best[3][1])


def panel_sampling_mask(shape, target_box, other_boxes=(), *, outer_scale=2.5,
                        exclusion_scale=1.25, panel_roi=None):
    """Sample around the button, excluding expanded boxes of every detection.

    panel_roi, if supplied, is an independently known panel xyxy boundary;
    without it, surrounding background is not guaranteed to be excluded.
    """
    target_box = _box(target_box)
    if not np.isfinite(outer_scale) or not np.isfinite(exclusion_scale):
        raise ValueError('sampling scales must be finite')
    if not outer_scale > exclusion_scale >= 1:
        raise ValueError('require outer_scale > exclusion_scale >= 1')
    mask = _rect_mask(shape, _expanded(target_box, outer_scale))
    for box in [target_box, *other_boxes]:
        mask &= ~_rect_mask(shape, _expanded(_box(box), exclusion_scale))
    if panel_roi is not None:
        mask &= _rect_mask(shape, _box(panel_roi))
    return mask


def camera_rays(pixels, camera_matrix, distortion=(), distortion_model=''):
    """Unproject color pixels using the matching color camera calibration."""
    matrix = np.asarray(camera_matrix, dtype=float)
    if (matrix.shape != (3, 3) or not np.all(np.isfinite(matrix))
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0):
        raise ValueError('camera matrix must have finite positive focal lengths')
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 1, 2)
    coefficients = np.asarray(distortion, dtype=float).reshape(-1)
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(pixels)):
        raise ValueError('pixels and distortion must be finite')
    if coefficients.size and np.any(coefficients):
        if distortion_model in ('plumb_bob', 'rational_polynomial'):
            normalized = cv2.undistortPoints(pixels, matrix, coefficients)
        elif distortion_model in ('equidistant', 'fisheye') and coefficients.size == 4:
            normalized = cv2.fisheye.undistortPoints(pixels, matrix, coefficients)
        else:
            raise ValueError('unsupported distortion model or coefficient count')
        normalized = normalized.reshape(-1, 2)
    else:
        normalized = (pixels.reshape(-1, 2) - matrix[:2, 2]) / [matrix[0, 0], matrix[1, 1]]
    return np.column_stack((normalized, np.ones(len(pixels))))


def estimate_sampled_surface(depth, mask, center, camera_matrix, *, distortion=(),
                             distortion_model='', unit_scale=0.001,
                             min_depth_m=0.1, max_depth_m=2.0,
                             threshold_m=0.0015, minimum_samples=60,
                             max_samples=800, minimum_inlier_ratio=0.70,
                             require_surrounding=True, max_tilt_degrees=60):
    """Estimate panel orientation, never substitute panel depth for button depth."""
    if depth.ndim != 2 or mask.shape != depth.shape or mask.dtype != bool:
        raise ValueError('depth and boolean mask must have matching 2D shapes')
    center = np.asarray(center, dtype=float)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError('center must contain two finite pixel coordinates')
    if (not np.isfinite(unit_scale) or unit_scale <= 0
            or not 0 < min_depth_m < max_depth_m
            or minimum_samples < 3 or max_samples < minimum_samples
            or not 0 < max_tilt_degrees < 90):
        raise ValueError('invalid depth, sample, or tilt bounds')
    yy, xx = np.nonzero(mask)
    depths = depth[yy, xx].astype(float)
    if np.issubdtype(depth.dtype, np.integer):
        depths *= unit_scale
    valid = np.isfinite(depths) & (depths >= min_depth_m) & (depths <= max_depth_m)
    yy, xx, depths = yy[valid], xx[valid], depths[valid]
    if len(depths) < minimum_samples:
        return SurfaceEstimate(samples=len(depths))
    if len(depths) > max_samples:
        indices = np.random.default_rng(0).choice(len(depths), max_samples, replace=False)
        yy, xx, depths = yy[indices], xx[indices], depths[indices]
    points = camera_rays(np.column_stack((xx, yy)), camera_matrix,
                         distortion, distortion_model) * depths[:, None]
    fit = fit_plane_consensus(points, threshold_m=threshold_m,
                              minimum_samples=minimum_samples,
                              minimum_inlier_ratio=minimum_inlier_ratio)
    if fit is None:
        return SurfaceEstimate(reason='no_plane_consensus', samples=len(points))
    inliers = np.abs(points @ fit.normal + fit.offset) <= threshold_m
    sectors = (xx >= center[0]).astype(int) + 2*(yy >= center[1]).astype(int)
    quadrants = int(np.count_nonzero(np.bincount(sectors[inliers], minlength=4) >= 8))
    if require_surrounding and quadrants < 3:
        return SurfaceEstimate(reason='one_sided_support', samples=len(points), quadrants=quadrants)
    if fit.normal[2] < np.cos(np.deg2rad(max_tilt_degrees)):
        return SurfaceEstimate(reason='excessive_tilt', samples=len(points), quadrants=quadrants)
    ray = camera_rays([center], camera_matrix, distortion, distortion_model)[0]
    denominator = float(ray @ fit.normal)
    z = -fit.offset / denominator if abs(denominator) > 1e-9 else float('nan')
    if not min_depth_m <= z <= max_depth_m:
        return SurfaceEstimate(reason='invalid_panel_intersection', samples=len(points), quadrants=quadrants)
    return SurfaceEstimate(fit, ray*z, 'accepted', len(points), quadrants)


def measure_button_point(depth, center_estimate, camera_matrix, *, radius_px=3,
                         unit_scale=0.001, distortion=(), distortion_model='',
                         max_depth_spread_m=0.006):
    """Independent center depth; return None for missing/ambiguous button depth."""
    if center_estimate.center is None:
        return None
    if (depth.ndim != 2 or radius_px < 1 or not np.isfinite(unit_scale)
            or unit_scale <= 0 or max_depth_spread_m <= 0):
        raise ValueError('invalid button depth parameters')
    center = center_estimate.center
    yy, xx = np.ogrid[:depth.shape[0], :depth.shape[1]]
    mask = (xx-center[0])**2 + (yy-center[1])**2 <= radius_px**2
    values = depth[mask].astype(float)
    if np.issubdtype(depth.dtype, np.integer):
        values *= unit_scale
    valid = values[np.isfinite(values) & (values >= 0.1) & (values <= 2.0)]
    if len(valid) < max(5, int(np.ceil(0.7*len(values)))):
        return None
    if np.percentile(valid, 90)-np.percentile(valid, 10) > max_depth_spread_m:
        return None
    return camera_rays([center], camera_matrix, distortion, distortion_model)[0] * np.median(valid)
