"""Unit tests for model decoding, tracking, and RGB-D geometry."""

import cv2
import numpy as np
import pytest

from piper_elevator_app.detector_core import Detection
from piper_elevator_app.detector_core import estimate_surface_normal
from piper_elevator_app.detector_core import filter_detections_by_class
from piper_elevator_app.detector_core import project_pixel
from piper_elevator_app.detector_core import robust_box_depth
from piper_elevator_app.detector_core import relabel_three_by_three_panel
from piper_elevator_app.detector_core import TemporalButtonTracker
from piper_elevator_app.detector_core import YoloOnnxDetector


CLASS_NAMES = [
    'button_up',
    'button_3',
    'button_2',
    'button_1',
    'button_open',
    'button_close',
    'button_down',
    'display_2',
    'display_1',
    'display_3',
]


class FakeInput:
    """Minimal ONNX input descriptor."""

    name = 'images'
    shape = [1, 3, 640, 640]


class FakeModelMetadata:
    """Minimal ONNX metadata descriptor."""

    custom_metadata_map = {
        'names': "{0: 'UP', 1: 'down', 2: 'open'}",
    }


class FakeSession:
    """Minimal ONNX Runtime test double."""

    def __init__(self, output):
        self.output = output
        self.input_blob = None
        self.run_count = 0

    def get_inputs(self):
        return [FakeInput()]

    def run(self, output_names, inputs):
        del output_names
        self.run_count += 1
        self.input_blob = inputs['images']
        return [self.output]

    def get_modelmeta(self):
        return FakeModelMetadata()


def make_detection(
    center_x=320.0,
    center_y=240.0,
    confidence=0.90,
    class_id=3,
):
    """Create one square detection for tracking tests."""
    return Detection(
        center_x - 40.0,
        center_y - 40.0,
        center_x + 40.0,
        center_y + 40.0,
        confidence,
        class_id,
        CLASS_NAMES[class_id],
    )


def test_decodes_yolov8_output_and_filters_display():
    output = np.zeros((1, 14, 3), dtype=np.float32)
    output[0, 0:4, 0] = [320.0, 320.0, 100.0, 100.0]
    output[0, 4 + 3, 0] = 0.90
    output[0, 0:4, 1] = [120.0, 120.0, 80.0, 80.0]
    output[0, 4 + 7, 1] = 0.99
    output[0, 0:4, 2] = [500.0, 400.0, 60.0, 60.0]
    output[0, 4 + 1, 2] = 0.20
    fake_session = FakeSession(output)
    detector = YoloOnnxDetector(
        model_path=None,
        class_names=CLASS_NAMES,
        target_classes=[name for name in CLASS_NAMES if 'button_' in name],
        confidence_threshold=0.60,
        session=fake_session,
    )

    detections = detector.infer(
        np.full((480, 640, 3), 50, dtype=np.uint8)
    )

    assert fake_session.input_blob.shape == (1, 3, 640, 640)
    assert len(detections) == 1
    detection = detections[0]
    assert detection.class_name == 'button_1'
    assert detection.confidence == pytest.approx(0.90)
    assert detection.center == pytest.approx((320.0, 240.0))
    assert detection.width == pytest.approx(100.0)
    assert detection.height == pytest.approx(100.0)


def test_decodes_yolov10_end_to_end_output_and_model_metadata():
    output = np.zeros((1, 300, 6), dtype=np.float32)
    output[0, 0] = [270.0, 270.0, 370.0, 370.0, 0.90, 1.0]
    output[0, 1] = [100.0, 100.0, 180.0, 180.0, 0.95, 2.0]
    detector = YoloOnnxDetector(
        model_path=None,
        class_names=['__model_metadata__'],
        target_classes=['UP', 'down'],
        confidence_threshold=0.60,
        session=FakeSession(output),
    )

    detections = detector.infer(
        np.full((480, 640, 3), 50, dtype=np.uint8)
    )

    assert len(detections) == 1
    detection = detections[0]
    assert detection.class_name == 'down'
    assert detection.confidence == pytest.approx(0.90)
    assert detection.center == pytest.approx((320.0, 240.0))
    assert detection.width == pytest.approx(100.0)
    assert detection.height == pytest.approx(100.0)


def test_warmup_runs_requested_number_of_dummy_frames():
    output = np.zeros((1, 300, 6), dtype=np.float32)
    session = FakeSession(output)
    detector = YoloOnnxDetector(
        model_path=None,
        class_names=['__model_metadata__'],
        target_classes=['*'],
        session=session,
    )

    detector.warmup(2)

    assert session.run_count == 2
    assert session.input_blob.shape == (1, 3, 640, 640)


def test_tracker_requires_stability_and_tolerates_one_miss():
    tracker = TemporalButtonTracker(
        required_stable_frames=3,
        max_missed_frames=2,
        smoothing_alpha=0.5,
    )

    first, first_valid = tracker.update(
        [make_detection()],
        640,
        480,
    )
    second, second_valid = tracker.update(
        [make_detection(322.0, 239.0)],
        640,
        480,
    )
    third, third_valid = tracker.update(
        [make_detection(321.0, 241.0)],
        640,
        480,
    )
    missed, missed_valid = tracker.update([], 640, 480)
    recovered, recovered_valid = tracker.update(
        [make_detection(320.0, 240.0)],
        640,
        480,
    )

    assert first is not None and not first_valid
    assert second is not None and not second_valid
    assert third is not None and third_valid
    assert missed is None and not missed_valid
    assert recovered is not None and recovered_valid


def test_tracker_does_not_switch_silently_to_another_class():
    tracker = TemporalButtonTracker(required_stable_frames=2)
    tracker.update([make_detection(class_id=3)], 640, 480)
    selected, valid = tracker.update(
        [make_detection(center_x=500.0, class_id=4)],
        640,
        480,
    )

    assert selected is not None
    assert selected.class_name == 'button_open'
    assert not valid
    assert tracker.stable_frames == 1


def test_filters_coordinates_to_operator_selected_button_class():
    detections = [
        Detection(10, 10, 30, 30, 0.9, 53, '3'),
        Detection(40, 10, 60, 30, 0.8, 256, 'UP'),
    ]

    assert filter_detections_by_class(detections, '') == []
    assert filter_detections_by_class(detections, '  up  ') == [
        detections[1]
    ]


def test_simulation_panel_layout_recovers_all_nine_button_labels():
    labels = ['1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm']
    class_names = labels + ['wrong']
    detections = []
    for column_x in (100.0, 250.0, 400.0):
        for row_y in (80.0, 220.0, 350.0):
            detections.append(
                Detection(
                    column_x - 20.0,
                    row_y - 20.0,
                    column_x + 20.0,
                    row_y + 20.0,
                    0.5,
                    9,
                    'wrong',
                )
            )

    relabeled = relabel_three_by_three_panel(
        list(reversed(detections)),
        labels,
        class_names,
    )

    by_name = {detection.class_name: detection for detection in relabeled}
    assert set(by_name) == set(labels)
    assert by_name['alarm'].center == pytest.approx((400.0, 350.0))
    assert by_name['1'].center == pytest.approx((100.0, 80.0))


def test_simulation_layout_does_not_relabel_an_incomplete_close_view():
    detections = [make_detection(class_id=index) for index in range(8)]

    assert relabel_three_by_three_panel(
        detections,
        ['1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm'],
        ['1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm'],
    ) == detections


def test_robust_depth_rejects_holes_and_outliers():
    depth_image = np.full((300, 300), 800, dtype=np.uint16)
    depth_image[130:135, 130:135] = 0
    depth_image[160:165, 160:165] = 5000
    detection = Detection(100, 100, 200, 200, 0.9, 3, 'button_1')

    depth_m = robust_box_depth(
        depth_image,
        detection,
        unit_scale=0.001,
        inner_ratio=0.5,
        min_depth_m=0.1,
        max_depth_m=2.0,
        min_samples=20,
    )

    assert depth_m == pytest.approx(0.8)


def test_robust_depth_keeps_gazebo_float_metres_unscaled():
    depth_image = np.full((300, 300), 0.8, dtype=np.float32)
    depth_image[130:135, 130:135] = np.nan
    depth_image[160:165, 160:165] = 5.0
    detection = Detection(100, 100, 200, 200, 0.9, 3, 'button_1')

    depth_m = robust_box_depth(
        depth_image,
        detection,
        unit_scale=0.001,
        inner_ratio=0.5,
        min_depth_m=0.1,
        max_depth_m=2.0,
        min_samples=20,
    )

    assert depth_m == pytest.approx(0.8)


def test_projects_pixel_to_camera_coordinates():
    camera_matrix = np.asarray(
        [
            [600.0, 0.0, 320.0],
            [0.0, 600.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )

    position = project_pixel(camera_matrix, 380.0, 210.0, 0.8)

    assert position == pytest.approx([0.08, -0.04, 0.8])


def test_projects_distorted_realsense_pixel_to_camera_coordinates():
    camera_matrix = np.asarray(
        [
            [432.35, 0.0, 430.15],
            [0.0, 431.36, 245.92],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.asarray(
        [-0.0541, 0.0622, -0.00029, 0.00033, -0.0206],
        dtype=np.float64,
    )
    expected = np.asarray([0.18, -0.09, 0.8], dtype=np.float64)
    projected, _ = cv2.projectPoints(
        expected.reshape(1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        camera_matrix,
        distortion,
    )
    u, v = projected.reshape(2)

    position = project_pixel(
        camera_matrix,
        u,
        v,
        expected[2],
        distortion,
        'plumb_bob',
    )

    assert position == pytest.approx(expected, abs=1e-6)


def test_estimates_button_surface_normal_from_depth_plane():
    camera_matrix = np.asarray([
        [400.0, 0.0, 160.0],
        [0.0, 400.0, 120.0],
        [0.0, 0.0, 1.0],
    ])
    desired_normal = np.asarray([0.12, -0.05, 1.0])
    desired_normal /= np.linalg.norm(desired_normal)
    rows, columns = np.indices((240, 320))
    rays_x = (columns - 160.0) / 400.0
    rays_y = (rows - 120.0) / 400.0
    depth = 0.8 / (
        desired_normal[0] * rays_x
        + desired_normal[1] * rays_y
        + desired_normal[2]
    )
    depth[100:103, 150:153] = np.nan
    depth[130:133, 170:173] = 1.6
    detection = Detection(100, 60, 220, 180, 0.9, 3, 'button_1')

    normal = estimate_surface_normal(
        depth.astype(np.float32),
        detection,
        camera_matrix,
        unit_scale=0.001,
        inner_ratio=0.8,
        min_depth_m=0.1,
        max_depth_m=2.0,
        min_samples=30,
        max_residual_m=0.003,
    )

    assert normal == pytest.approx(desired_normal, abs=2.0e-3)
