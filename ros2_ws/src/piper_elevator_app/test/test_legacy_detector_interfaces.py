"""Exercise old recognition with the restored coarse/Servo observation contract."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from sensor_msgs.msg import Image

from piper_elevator_app.detector_core import Detection, TemporalButtonTracker
from piper_elevator_app.yolo_button_detector import ButtonDetector
from test_servo_observation_safety import surface
from test_visual_semantic_safety import SemanticHarness, status_message


class Detector(ButtonDetector):
    def __init__(self):
        self.values = {}
        self._declare_parameters()
        self._selected_button_class = '2'
        self._camera_frame = 'camera'
        self._use_depth = True
        self._camera_matrix = np.array([[200., 0, 80.], [0, 200., 60.], [0, 0, 1.]])
        self._distortion_coefficients = np.array([])
        self._distortion_model = ''
        self._filtered_position = None
        self._filtered_surface_normal = None
        self._tracker = TemporalButtonTracker(required_stable_frames=2)
        self.detections = [Detection(50, 30, 110, 90, .9, 0, '2')]
        self.outputs = {name: [] for name in ('pose', 'surface_pose', 'pixel', 'valid', 'confidence', 'tracking_state')}
        for name, messages in self.outputs.items():
            setattr(self, '_' + name + '_publisher', SimpleNamespace(publish=messages.append))

    def declare_parameter(self, name, value):
        self.values[name] = value

    def get_parameter(self, name):
        return SimpleNamespace(value=self.values[name])

    def _detect(self, image):
        return self.detections

    def _publish_detections(self, *args):
        pass

    def _publish_debug(self, *args):
        pass

    def frame(self, stamp=9_970_000_000, depth=.4):
        image = Image()
        image.header.frame_id = 'camera'
        image.header.stamp.sec, image.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
        self._process_frame(image, np.zeros((120, 160, 3), np.uint8),
                            np.full((120, 160), depth, np.float32))
        return json.loads(self.outputs['tracking_state'][-1].data)


def test_old_detector_publishes_atomic_pose_normal_and_tracking_stamp():
    detector = Detector()
    assert not detector.frame()['selected']['stable_detection']
    assert not detector.outputs['pose']
    state = detector.frame()
    assert state['selected']['depth_valid']
    assert state['selected']['surface_valid']
    assert state['selected']['measured']['class_name'] == '2'
    pose, normal = detector.outputs['pose'][-1], detector.outputs['surface_pose'][-1]
    assert pose.header == normal.header
    assert normal.header.stamp.nanosec == state['stamp']['nanosec']
    assert pose.pose.position == normal.pose.position
    assert normal.pose.orientation.w != 0.0
    detector.frame(stamp=9_980_000_000, depth=.5)
    assert detector.outputs['pose'][-1].pose.position.z == pytest.approx(.5)


@pytest.mark.parametrize('failure', ['missing', 'wrong_class', 'depth_hole'])
def test_missing_or_invalid_detection_cannot_publish_a_stale_surface(failure):
    detector = Detector()
    detector.frame()
    detector.frame()
    before = len(detector.outputs['surface_pose'])
    if failure == 'missing':
        detector.detections = []
    if failure == 'wrong_class':
        detector.detections = [Detection(50, 30, 110, 90, .9, 1, '3')]
    state = detector.frame(depth=0.0 if failure == 'depth_hole' else .4)
    assert not state['selected']['depth_valid']
    assert len(detector.outputs['surface_pose']) == before


def test_old_detector_status_is_understood_by_new_servo_semantic_gate():
    detector = Detector()
    detector.frame(stamp=9_960_000_000)
    detector.frame()
    servo = SemanticHarness()
    servo._tracking_state_callback(status_message(9_950_000_000))
    assert servo.held
    servo._tracking_state_callback(detector.outputs['tracking_state'][-1])
    assert servo.held  # matching fresh surface still required
    servo._surface_pose_callback(surface(9_970_000_000))
    assert not servo.held
