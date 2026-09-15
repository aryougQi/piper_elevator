"""Offline geometry behavior, including cases that must reject a measurement."""

import cv2
import numpy as np
import pytest

from piper_elevator_app.experimental_button_geometry import (
    CenterEstimate, camera_rays, estimate_sampled_surface, measure_button_point,
    panel_sampling_mask, refine_rectangular_center,
)


MATRIX = np.array([[200., 0, 100], [0, 200., 100], [0, 0, 1]])
BOX = [70, 70, 130, 130]


def test_rectangle_center_ignores_off_center_digit_and_box_bias():
    image = np.full((200, 200, 3), 160, np.uint8)
    cv2.rectangle(image, (68, 72), (128, 132), (20, 20, 20), -1)
    cv2.putText(image, '1', (105, 113), cv2.FONT_HERSHEY_SIMPLEX, .8, (250, 250, 250), 2)
    result = refine_rectangular_center(image, [72, 68, 134, 136])
    assert result.reason == 'accepted'
    np.testing.assert_allclose(result.center, [98, 102], atol=1)


def test_perspective_rectangle_uses_diagonal_intersection():
    image = np.full((200, 200), 180, np.uint8)
    corners = np.array([[60, 65], [145, 75], [128, 140], [73, 137]], np.int32)
    cv2.fillConvexPoly(image, corners, 20)
    result = refine_rectangular_center(image, [59, 64, 146, 141])
    h = np.column_stack((corners, np.ones(4)))
    center = np.cross(np.cross(h[0], h[2]), np.cross(h[1], h[3]))
    np.testing.assert_allclose(result.center, center[:2]/center[2], atol=1.5)


def test_small_icon_without_button_outline_is_not_refined():
    image = np.full((200, 200), 170, np.uint8)
    cv2.rectangle(image, (91, 89), (108, 110), 20, -1)
    assert refine_rectangular_center(image, BOX).center is None


def test_clipped_button_is_rejected():
    image = np.zeros((200, 200), np.uint8)
    assert refine_rectangular_center(image, [-5, 40, 60, 100]).reason == 'clipped_detection'


def test_occluded_outline_is_rejected():
    image = np.full((200, 200), 170, np.uint8)
    cv2.rectangle(image, (70, 70), (130, 130), 20, -1)
    cv2.rectangle(image, (70, 70), (110, 130), 170, -1)
    assert refine_rectangular_center(image, BOX).center is None


def test_sampling_excludes_selected_neighbors_and_outside_panel():
    mask = panel_sampling_mask((200, 200), BOX, [[135, 70, 165, 130]],
                               panel_roi=[40, 40, 180, 180])
    assert not mask[100, 100]
    assert not mask[100, 145]
    assert not mask[30, 100]
    assert mask[50, 100]


def tilted_depth():
    yy, xx = np.indices((200, 200))
    return (0.5 / (1 + .2*(xx-100)/200 - .1*(yy-100)/200)).astype(np.float32)


def test_annulus_plane_ignores_protruding_button_and_neighbor():
    depth = tilted_depth()
    depth[70:131, 70:131] -= .015
    depth[70:131, 140:165] -= .02
    mask = panel_sampling_mask(depth.shape, BOX, [[140, 70, 165, 131]])
    result = estimate_sampled_surface(depth, mask, [100, 100], MATRIX)
    normal = np.array([.2, -.1, 1.]); normal /= np.linalg.norm(normal)
    assert result.reason == 'accepted'
    np.testing.assert_allclose(result.plane.normal, normal, atol=1e-5)
    assert result.panel_point[2] == pytest.approx(.5, abs=1e-6)
    button = measure_button_point(depth, CenterEstimate(center=np.array([100., 100.])), MATRIX)
    assert button[2] == pytest.approx(.485, abs=1e-5)


def test_one_sided_depth_does_not_authorize_panel_orientation():
    depth = tilted_depth()
    mask = panel_sampling_mask(depth.shape, BOX)
    mask[:, 100:] = False
    result = estimate_sampled_surface(depth, mask, [100, 100], MATRIX)
    assert result.reason == 'one_sided_support'
    assert result.plane is None


def test_no_button_depth_never_falls_back_to_panel_intersection():
    depth = tilted_depth()
    depth[90:111, 90:111] = 0
    center = CenterEstimate(center=np.array([100., 100.]))
    assert measure_button_point(depth, center, MATRIX) is None
    mask = panel_sampling_mask(depth.shape, BOX)
    assert estimate_sampled_surface(depth, mask, center.center, MATRIX).plane is not None


def test_uint16_and_float_depth_agree():
    mask = panel_sampling_mask((200, 200), BOX)
    a = estimate_sampled_surface(np.full((200, 200), 500, np.uint16), mask, [100, 100], MATRIX)
    b = estimate_sampled_surface(np.full((200, 200), .5, np.float32), mask, [100, 100], MATRIX)
    np.testing.assert_allclose(a.panel_point, b.panel_point)


def test_projection_uses_color_distortion():
    points = np.array([[.1, -.04, .5], [-.1, .03, .6]])
    distortion = np.array([.1, -.05, .001, .002, 0])
    pixels, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), MATRIX, distortion)
    rays = camera_rays(pixels, MATRIX, distortion, 'plumb_bob')
    np.testing.assert_allclose(rays, points/points[:, 2, None], atol=1e-7)


def test_production_sources_have_no_experiment_imports():
    from pathlib import Path
    root = Path(__file__).parents[1]/'piper_elevator_app'
    for name in ['detector_core.py', 'yolo_button_detector.py']:
        source = (root/name).read_text()
        assert 'experimental_button_geometry' not in source
        assert 'plane_core' not in source
