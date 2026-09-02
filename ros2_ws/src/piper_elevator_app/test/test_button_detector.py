"""Unit tests for model decoding, tracking, and RGB-D geometry."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from piper_elevator_app.detector_core import Detection
from piper_elevator_app.detector_core import estimate_surface_normal
from piper_elevator_app.detector_core import extract_padded_square_crop
from piper_elevator_app.detector_core import filter_detections_by_class
from piper_elevator_app.detector_core import project_camera_point
from piper_elevator_app.detector_core import project_pixel
from piper_elevator_app.detector_core import preserve_projected_target_label
from piper_elevator_app.detector_core import preserve_tracked_label
from piper_elevator_app.detector_core import robust_box_depth
from piper_elevator_app.detector_core import remap_crop_detections
from piper_elevator_app.detector_core import relabel_three_by_three_panel
from piper_elevator_app.detector_core import (
    relabel_three_by_three_panel_with_status,
)
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

    assert selected is None
    assert not valid
    assert tracker.current is not None
    assert tracker.current.class_name == 'button_1'
    assert tracker.missed_frames == 1


def test_tracker_rejects_a_distant_detection_with_the_same_class():
    tracker = TemporalButtonTracker(
        required_stable_frames=2,
        max_missed_frames=5,
    )
    tracker.update([make_detection(center_x=120.0)], 640, 480)

    selected, valid = tracker.update(
        [make_detection(center_x=520.0, confidence=0.99)],
        640,
        480,
    )

    assert selected is None
    assert not valid
    assert tracker.current is not None
    assert tracker.current.center == pytest.approx((120.0, 240.0))
    assert tracker.missed_frames == 1


def test_tracker_reacquires_distant_layout_confirmed_identity():
    tracker = TemporalButtonTracker(
        required_stable_frames=3,
        max_missed_frames=80,
        smoothing_alpha=0.35,
    )
    tracker.update([make_detection(center_x=120.0)], 848, 480)
    tracker.update([], 848, 480)

    moved = make_detection(center_x=620.0, confidence=0.8)
    selected, valid = tracker.update(
        [moved],
        848,
        480,
        allow_global_reacquisition=True,
    )

    assert selected == moved
    assert not valid
    assert tracker.current == moved
    assert tracker.stable_frames == 1
    assert tracker.missed_frames == 0

    tracker.update(
        [make_detection(center_x=621.0, confidence=0.8)],
        848,
        480,
        allow_global_reacquisition=True,
    )
    _, valid = tracker.update(
        [make_detection(center_x=622.0, confidence=0.8)],
        848,
        480,
        allow_global_reacquisition=True,
    )
    assert valid


def test_tracker_does_not_globally_reacquire_ambiguous_identity():
    tracker = TemporalButtonTracker(required_stable_frames=2)
    tracker.update([make_detection(center_x=120.0)], 848, 480)

    selected, valid = tracker.update(
        [
            make_detection(center_x=500.0, confidence=0.9),
            make_detection(center_x=700.0, confidence=0.8),
        ],
        848,
        480,
        allow_global_reacquisition=True,
    )

    assert selected is None
    assert not valid
    assert tracker.current.center == pytest.approx((120.0, 240.0))


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


def test_simulation_layout_reports_authoritative_identity():
    detections = []
    for row in range(3):
        for column in range(3):
            center_x = 100.0 + column * 80.0
            center_y = 100.0 + row * 80.0
            detections.append(
                Detection(
                    center_x - 20.0,
                    center_y - 20.0,
                    center_x + 20.0,
                    center_y + 20.0,
                    0.8,
                    0,
                    'raw',
                )
            )

    relabeled, confirmed = relabel_three_by_three_panel_with_status(
        detections,
        ['1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm'],
        ['1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm', 'raw'],
        maximum_panel_box_size=80.0,
        drop_auxiliary_detections=True,
    )

    assert confirmed
    assert [item.class_name for item in relabeled] == [
        '1', '4', 'open', '2', 'up', 'close', '3', 'down', 'alarm'
    ]


def test_incomplete_layout_is_not_authoritative():
    detections = [make_detection(center_x=100.0 + index * 20.0) for index in range(8)]

    relabeled, confirmed = relabel_three_by_three_panel_with_status(
        detections,
        ['1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm'],
        ['1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm'],
    )

    assert relabeled == detections
    assert not confirmed


def test_simulation_layout_regularizes_individual_box_center_bias():
    labels = ['1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm']
    detections = []
    for row, y in enumerate((100.0, 180.0, 260.0)):
        for column, x in enumerate((200.0, 280.0, 360.0)):
            biased_y = y + (12.0 if (row, column) == (1, 0) else 0.0)
            detections.append(
                Detection(
                    x - 20.0,
                    biased_y - 20.0,
                    x + 20.0,
                    biased_y + 20.0,
                    0.8,
                    4,
                    'up',
                )
            )

    relabeled = relabel_three_by_three_panel(
        detections,
        labels,
        labels,
    )
    by_name = {item.class_name: item for item in relabeled}

    assert by_name['4'].center == pytest.approx((200.0, 180.0))
    assert by_name['up'].center == pytest.approx((280.0, 180.0))


def test_simulation_layout_does_not_relabel_an_incomplete_close_view():
    detections = [make_detection(class_id=index) for index in range(8)]

    assert relabel_three_by_three_panel(
        detections,
        ['1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm'],
        ['1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm'],
    ) == detections


def test_simulation_layout_can_suppress_incomplete_initial_acquisition():
    detections = [make_detection(class_id=index) for index in range(8)]

    assert relabel_three_by_three_panel(
        detections,
        ['1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm'],
        ['1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm'],
        suppress_incomplete_layout=True,
    ) == []


def test_simulation_layout_rejects_eight_buttons_plus_floor_indicator():
    labels = ['1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm']
    detections = []
    for y in (100.0, 180.0, 260.0):
        for x in (200.0, 280.0, 360.0):
            detections.append(
                Detection(x, y, x + 50, y + 50, 0.8, 4, 'up')
            )
    detections.pop()
    floor_indicator = Detection(265, 20, 315, 55, 0.99, 4, 'up')

    assert relabel_three_by_three_panel(
        detections + [floor_indicator],
        labels,
        labels,
        suppress_incomplete_layout=True,
    ) == []


def test_simulation_layout_ignores_smaller_floor_indicator():
    labels = ['1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm']
    class_names = labels + ['empty']
    detections = []
    for y in (100.0, 180.0, 260.0):
        for x in (200.0, 280.0, 360.0):
            detections.append(
                Detection(x, y, x + 50, y + 50, 0.8, 9, 'wrong')
            )
    indicator = Detection(270, 30, 300, 65, 0.9, 9, 'empty')

    relabeled = relabel_three_by_three_panel(
        detections + [indicator], labels, class_names
    )

    assert indicator in relabeled
    by_name = {item.class_name: item for item in relabeled}
    assert set(labels).issubset(by_name)
    assert by_name['down'].center == pytest.approx((385.0, 205.0))


def test_simulation_layout_can_drop_false_positive_auxiliary_icon():
    labels = ['1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm']
    detections = []
    for y in (100.0, 180.0, 260.0):
        for x in (200.0, 280.0, 360.0):
            detections.append(
                Detection(x, y, x + 50, y + 50, 0.8, 4, 'up')
            )
    false_up = Detection(270, 30, 300, 65, 0.99, 4, 'up')

    relabeled = relabel_three_by_three_panel(
        detections + [false_up],
        labels,
        labels,
        drop_auxiliary_detections=True,
    )

    assert len(relabeled) == 9
    assert false_up not in relabeled
    assert [item.class_name for item in relabeled].count('up') == 1


def test_simulation_layout_does_not_relabel_large_close_view_boxes():
    detections = [
        Detection(
            10.0 * index,
            10.0 * index,
            10.0 * index + 120.0,
            10.0 * index + 120.0,
            0.8,
            index,
            f'raw_{index}',
        )
        for index in range(9)
    ]

    assert relabel_three_by_three_panel(
        detections,
        ['1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm'],
        ['1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm'],
        maximum_panel_box_size=80.0,
    ) == detections

    assert relabel_three_by_three_panel(
        detections,
        ['1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm'],
        ['1', '2', '3', '4', 'up', 'down', 'open', 'close', 'alarm'],
        maximum_panel_box_size=80.0,
        suppress_incomplete_layout=True,
    ) == []


def test_tracking_preserves_selected_label_at_close_range():
    tracked = Detection(100, 100, 150, 150, 0.8, 8, 'alarm')
    same_button_wrong_label = Detection(
        103, 102, 153, 152, 0.9, 4, 'up'
    )
    neighbor = Detection(180, 100, 230, 150, 0.9, 5, 'close')

    corrected = preserve_tracked_label(
        [same_button_wrong_label, neighbor],
        'alarm',
        tracked,
        minimum_iou=0.15,
    )

    assert corrected[0].class_name == 'alarm'
    assert corrected[0].class_id == tracked.class_id
    assert corrected[1] == neighbor


def test_tracking_does_not_jump_to_a_distant_detection():
    tracked = Detection(100, 100, 150, 150, 0.8, 8, 'alarm')
    distant = Detection(300, 300, 350, 350, 0.9, 4, 'up')

    assert preserve_tracked_label(
        [distant], 'alarm', tracked, minimum_iou=0.15
    ) == [distant]


def test_tracking_survives_local_motion_without_overlap():
    tracked = Detection(100, 100, 140, 140, 0.8, 8, 'alarm')
    moved_wrong_label = Detection(142, 100, 182, 140, 0.9, 4, 'up')

    corrected = preserve_tracked_label(
        [moved_wrong_label],
        'alarm',
        tracked,
        minimum_iou=0.15,
        maximum_center_distance=45.0,
    )

    assert corrected[0].class_name == 'alarm'
    assert corrected[0].class_id == tracked.class_id


def test_tracking_ignores_distant_same_name_detection():
    tracked = Detection(100, 100, 140, 140, 0.8, 8, 'alarm')
    same_button_wrong_label = Detection(108, 102, 148, 142, 0.7, 4, 'up')
    distant_same_name = Detection(300, 300, 340, 340, 0.99, 8, 'alarm')

    corrected = preserve_tracked_label(
        [distant_same_name, same_button_wrong_label],
        'alarm',
        tracked,
        minimum_iou=0.15,
        maximum_center_distance=30.0,
    )

    assert corrected[0] == distant_same_name
    assert corrected[1].class_name == 'alarm'
    assert corrected[1].class_id == tracked.class_id


def test_tracking_never_relabels_before_initial_semantic_acquisition():
    wrong_label = Detection(100, 100, 140, 140, 0.9, 4, 'up')

    assert preserve_tracked_label(
        [wrong_label],
        'alarm',
        tracked=None,
        minimum_iou=0.15,
        maximum_center_distance=50.0,
    ) == [wrong_label]


def test_projected_target_reacquires_unique_nearby_detection():
    target = Detection(390, 210, 430, 250, 0.7, 4, 'up')
    neighbor = Detection(520, 210, 560, 250, 0.9, 8, 'alarm')

    corrected, authoritative = preserve_projected_target_label(
        [neighbor, target],
        'alarm',
        [410.0, 230.0],
        maximum_distance=60.0,
        ambiguity_margin=20.0,
    )

    assert authoritative
    assert corrected[1].class_name == 'alarm'
    assert corrected[0] == neighbor


def test_projected_target_rejects_ambiguous_neighbors():
    left = Detection(350, 210, 390, 250, 0.8, 4, 'up')
    right = Detection(430, 210, 470, 250, 0.8, 6, 'down')

    corrected, authoritative = preserve_projected_target_label(
        [left, right],
        'alarm',
        [410.0, 230.0],
        maximum_distance=60.0,
        ambiguity_margin=20.0,
    )

    assert not authoritative
    assert corrected == [left, right]


def test_projected_crop_pads_an_edge_without_moving_source_pixels():
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[0, 0] = [10, 20, 30]

    crop, origin_x, origin_y = extract_padded_square_crop(
        image,
        [10.0, 15.0],
        80,
    )

    assert crop.shape == (80, 80, 3)
    assert (origin_x, origin_y) == (-30, -25)
    assert crop[25, 30].tolist() == [10, 20, 30]
    assert crop[0, 0].tolist() == [114, 114, 114]


def test_projected_crop_detection_maps_back_and_clips_padding():
    local = Detection(20, 15, 70, 65, 0.8, 4, 'up')

    mapped = remap_crop_detections(
        [local],
        origin_x=-30,
        origin_y=-25,
        image_width=100,
        image_height=80,
    )

    assert len(mapped) == 1
    assert mapped[0].x1 == pytest.approx(0.0)
    assert mapped[0].y1 == pytest.approx(0.0)
    assert mapped[0].x2 == pytest.approx(40.0)
    assert mapped[0].y2 == pytest.approx(40.0)
    assert mapped[0].class_name == 'up'


def test_real_detector_configuration_holds_identity_through_short_dropout():
    config = yaml.safe_load(
        (Path(__file__).parents[1] / 'config' / 'button_detector.yaml')
        .read_text()
    )['button_detector']['ros__parameters']

    assert config['simulation_layout_relabel'] is False
    assert config['preserve_selected_track_identity'] is True
    assert config['max_missed_frames'] >= 50
    assert config['projected_reacquisition_enabled'] is True
    assert config['projected_local_detection_enabled'] is True
    assert config['projected_local_crop_size_px'] >= (
        2 * config['projected_reacquisition_radius_px']
    )
    assert config['button_base_topic'] == '/button_pose_base'
    assert config['projected_reacquisition_radius_px'] <= 60.0
    assert (
        config['projected_reacquisition_ambiguity_margin_px'] >= 20.0
    )

    source = (
        Path(__file__).parents[1]
        / 'piper_elevator_app'
        / 'yolo_button_detector.py'
    ).read_text()
    assert '_project_locked_button' in source
    assert 'preserve_projected_target_label(' in source
    assert '_infer_projected_local_target' in source
    assert 'self._locked_button_base is not None' in source


def test_complete_simulation_launch_enables_layout_semantics():
    launch_source = (
        Path(__file__).parents[1] / 'launch' / 'elevator_task.launch.py'
    ).read_text()

    assert "'simulation_layout_relabel': 'true'" in launch_source

    detector_source = (
        Path(__file__).parents[1]
        / 'piper_elevator_app'
        / 'yolo_button_detector.py'
    ).read_text()
    assert 'maximum_panel_box_size=0.25 * min(width, height)' in detector_source


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

    pixel = project_camera_point(camera_matrix, position)
    assert pixel == pytest.approx([380.0, 210.0])


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
    assert project_camera_point(
        camera_matrix,
        expected,
        distortion,
        'plumb_bob',
    ) == pytest.approx([u, v], abs=1e-6)


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
