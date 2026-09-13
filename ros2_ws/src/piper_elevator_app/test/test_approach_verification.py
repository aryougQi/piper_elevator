"""Verify post-motion acceptance with production geometry and observation filtering."""

import copy
from collections import deque
import math
from types import SimpleNamespace

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped
from rclpy.time import Time
from std_srvs.srv import Trigger

from piper_elevator_app import button_approach_planner as module
from piper_elevator_app.motion_core import matrix_to_quaternion
from piper_elevator_app.motion_core import quaternion_to_matrix
from test_approach_planning import LEVEL_CAMERA, PlannerHarness, make_transform
from test_approach_planning import observation, trajectory


def rotation_z(degrees):
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class VerificationHarness(PlannerHarness):
    """Feed real surface callbacks while advancing a deterministic clock."""

    def __init__(self, clock):
        super().__init__()
        self.clock = clock
        self.values.update({
            'observation_minimum_samples': 8,
            'observation_stable_samples': 40,
            'observation_window_max_seconds': 6.0,
            'post_execution_observation_timeout_seconds': 8.0,
            'allow_execution': True,
            'simulation_mode': True,
        })
        self._selected_button = '2'
        self._observations = deque(maxlen=40)
        self._observation_not_before_stamp_ns = 0
        self._last_execution_diagnostic = {}
        self._latest_joint_stamp_ns = self.now_ns
        self._latest_received_at = 0.0
        self._latest_button = None
        self._execution_observation = observation()
        self._execution_observation['selected_button'] = '2'
        self._execution_button = self._execution_observation['button'].copy()
        self.target = self._make_pose(
            [0.3, 0.0, 0.3], matrix_to_quaternion(LEVEL_CAMERA),
            Time(nanoseconds=self.now_ns).to_msg(),
        )
        self.actual = copy.deepcopy(self.target)
        self.camera_mount_translation = np.zeros(3)
        self.button = self._execution_button.copy()
        self.normal_degrees = 0.0
        self.frame_limit = None
        self.frames = 0
        self.sleeps = 0
        self.refresh_joints = True
        self.stale_tf = False
        self.on_sleep = None
        self.on_actual_transform = None
        self.actual_transform_calls = 0
        self._lookup_message_transform = lambda target, source, stamp: (
            self._camera_transform() if target == self._base_frame
            else self.lookup_transform(target, source, stamp)
        )
        self._tf_buffer = SimpleNamespace(lookup_transform=self.lookup_transform)

    def advance(self, seconds):
        self.clock[0] += seconds
        self.now_ns += int(round(seconds * 1e9))
        if self.refresh_joints:
            self._latest_joint_received_at = self.clock[0]
            self._latest_joint_stamp_ns = self.now_ns

    def sleep(self, seconds):
        self.sleeps += 1
        self.advance(seconds)
        if self.on_sleep is not None:
            self.on_sleep()
        if self.frame_limit is None or self.frames < self.frame_limit:
            self.publish_frame()

    def _tip_transform(self):
        position = self.actual.pose.position
        orientation = self.actual.pose.orientation
        return make_transform(
            [position.x, position.y, position.z],
            [orientation.x, orientation.y, orientation.z, orientation.w],
        )

    def _camera_transform(self):
        position, orientation = self._transform_arrays(self._tip_transform())
        return make_transform(
            position + quaternion_to_matrix(orientation) @ self.camera_mount_translation,
            orientation,
        )

    def lookup_transform(self, target_frame, source_frame, *args, **kwargs):
        if target_frame == self._end_effector_link:
            return make_transform(
                self.camera_mount_translation, [0.0, 0.0, 0.0, 1.0],
            )
        assert (target_frame, source_frame) == (
            self._base_frame, self._end_effector_link,
        )
        self.actual_transform_calls += 1
        if self.on_actual_transform is not None:
            self.on_actual_transform()
        transform = self._tip_transform()
        stamp_ns = self.now_ns - (1_000_000_000 if self.stale_tf else 0)
        transform.header.stamp = Time(nanoseconds=stamp_ns).to_msg()
        return transform

    def set_actual_rotation(self, rotation):
        q = matrix_to_quaternion(rotation)
        orientation = self.actual.pose.orientation
        orientation.x, orientation.y, orientation.z, orientation.w = map(float, q)

    def publish_frame(self, stamp_ns=None):
        self.frames += 1
        message = PoseStamped()
        message.header.frame_id = self.values['camera_frame']
        message.header.stamp = Time(
            nanoseconds=self.now_ns if stamp_ns is None else stamp_ns,
        ).to_msg()
        position, orientation = self._transform_arrays(self._camera_transform())
        camera_rotation = quaternion_to_matrix(orientation)
        local_point = camera_rotation.T @ (self.button - position)
        message.pose.position.x, message.pose.position.y, message.pose.position.z = \
            map(float, local_point)
        surface_rotation = rotation_z(self.normal_degrees) @ LEVEL_CAMERA
        q = matrix_to_quaternion(camera_rotation.T @ surface_rotation)
        message.pose.orientation.x, message.pose.orientation.y, \
            message.pose.orientation.z, message.pose.orientation.w = map(float, q)
        self._surface_pose_callback(message)


@pytest.fixture
def planner(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    harness = VerificationHarness(clock)
    monkeypatch.setattr(module.time, 'sleep', harness.sleep)
    return harness


def test_changed_normal_is_accepted_when_current_camera_geometry_is_safe(planner):
    planner.normal_degrees = 7.0

    success, message = planner._verify_approach_reached(planner.target)

    assert success, message
    assert planner.frames == 8
    assert planner.clock[0] == pytest.approx(100.4)
    assert planner._last_execution_diagnostic


def test_handover_accepts_servo_capture_angle_without_relaxing_planning(planner):
    planner.set_actual_rotation(rotation_z(11.5) @ LEVEL_CAMERA)
    planning_safe, planning_message = planner._validate_endpoint(
        planner.actual, planner.target, planner._execution_observation,
    )

    success, message = planner._verify_approach_reached(planner.target)

    assert not planning_safe
    assert 'tilt' in planning_message.lower()
    assert success, message


@pytest.mark.parametrize('distance', [0.075, 0.25])
def test_handover_requires_clearance_and_servo_capture_distance(planner, distance):
    planner.actual.pose.position.x = planner.button[0] - distance
    planner.target.pose.position.x = planner.actual.pose.position.x
    if distance < 0.08:
        planner.camera_mount_translation[2] = -0.1

    success, message = planner._verify_approach_reached(planner.target)

    assert not success, message
    assert any(word in message.lower() for word in ('standoff', 'capture', 'start'))
    assert planner.actual_transform_calls > 0


def test_old_forty_frame_window_is_replaced_by_eight_post_motion_frames(planner):
    old = copy.deepcopy(planner._execution_observation)
    planner._observations.extend(copy.deepcopy(old) for _ in range(40))
    planner._latest_observation = copy.deepcopy(old)
    planner.frame_limit = 8
    baseline = planner.now_ns

    success, message = planner._verify_approach_reached(planner.target)

    assert success, message
    assert len(planner._observations) == 8
    assert all(item['stamp_ns'] > baseline for item in planner._observations)
    assert planner.frames == 8
    assert planner.clock[0] == pytest.approx(100.4)


@pytest.mark.parametrize('offset_ns', [-1, 0])
def test_queued_capture_at_or_before_post_motion_baseline_is_rejected(
    planner, offset_ns,
):
    planner._observation_not_before_stamp_ns = planner.now_ns

    planner.publish_frame(stamp_ns=planner.now_ns + offset_ns)

    assert not planner._observations
    assert planner._latest_observation is None


def test_in_flight_surface_callback_cannot_refill_new_post_motion_window(planner):
    def lookup_after_window_reset(*args):
        planner._observation_not_before_stamp_ns = planner.now_ns
        return planner._camera_transform()

    planner._lookup_message_transform = lookup_after_window_reset
    planner.publish_frame()

    assert not planner._observations
    assert planner._latest_observation is None


@pytest.mark.parametrize('failure, expected', [
    ('tilt', 'tilt'),
    ('roll', 'roll'),
    ('field_of_view', 'image safety margin'),
    ('tcp', 'TCP endpoint error'),
    ('orientation', 'orientation'),
    ('joint_margin', 'limit margin'),
    ('wrist', 'wrist'),
    ('joint_age', 'joint feedback'),
    ('tf_age', 'transform is stale'),
])
def test_current_endpoint_safety_remains_required(planner, failure, expected):
    if failure == 'tilt':
        planner.set_actual_rotation(rotation_z(20.0) @ LEVEL_CAMERA)
    elif failure == 'roll':
        angle = math.radians(25.0)
        c, s = math.cos(angle), math.sin(angle)
        planner.set_actual_rotation(np.array([
            [1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c],
        ]) @ LEVEL_CAMERA)
    elif failure == 'field_of_view':
        planner.actual.pose.position.y = 0.09
        planner.target.pose.position.y = 0.09
    elif failure == 'tcp':
        planner.actual.pose.position.x += 0.02
    elif failure == 'orientation':
        planner.set_actual_rotation(rotation_z(12.0) @ LEVEL_CAMERA)
        q = matrix_to_quaternion(rotation_z(-10.0) @ LEVEL_CAMERA)
        orientation = planner.target.pose.orientation
        orientation.x, orientation.y, orientation.z, orientation.w = map(float, q)
    elif failure == 'joint_margin':
        planner._latest_joint_positions['joint1'] = 1.99
    elif failure == 'wrist':
        planner._latest_joint_positions['joint5'] = 0.0
    elif failure == 'joint_age':
        planner.refresh_joints = False
        planner._latest_joint_received_at = 99.0
    elif failure == 'tf_age':
        planner.stale_tf = True

    success, message = planner._verify_approach_reached(planner.target)

    assert not success
    assert expected.lower() in message.lower()
    assert planner.actual_transform_calls > 0
    assert planner.clock[0] <= 108.1


def test_temporary_geometry_failure_can_recover_within_verification_budget(planner):
    planner.set_actual_rotation(rotation_z(20.0) @ LEVEL_CAMERA)

    def recover():
        if planner.sleeps == 12:
            planner.set_actual_rotation(LEVEL_CAMERA)

    planner.on_sleep = recover
    success, message = planner._verify_approach_reached(planner.target)

    assert success, message
    assert planner.sleeps >= 12
    assert planner.actual_transform_calls >= 2
    assert planner.clock[0] < 108.0


def test_seven_fresh_frames_cannot_complete_verification(planner):
    planner.frame_limit = 7

    success, _ = planner._verify_approach_reached(planner.target)

    assert not success
    assert planner.actual_transform_calls == 0
    assert 108.0 <= planner.clock[0] <= 108.1


def test_observation_expiring_during_tcp_lookup_cannot_be_accepted(planner):
    planner.on_actual_transform = lambda: planner.advance(0.6)

    success, message = planner._verify_approach_reached(planner.target)

    assert not success, message
    assert planner.actual_transform_calls > 0
    assert planner.clock[0] >= 108.0


def test_selection_change_during_tcp_lookup_cannot_be_accepted(planner):
    planner.on_actual_transform = lambda: planner._selection_callback(
        SimpleNamespace(data='3'),
    )

    success, message = planner._verify_approach_reached(planner.target)

    assert not success
    assert 'Selected button changed' in message
    assert planner.actual_transform_calls == 1
    assert planner.clock[0] == pytest.approx(100.4)


def test_button_displacement_is_rejected_after_first_stable_new_window(planner):
    planner.button[1] += 0.04

    success, message = planner._verify_approach_reached(planner.target)

    assert not success
    assert 'button moved' in message.lower()
    assert planner.actual_transform_calls == 0
    assert planner.frames == 8


def test_new_stable_displaced_target_during_tcp_lookup_cannot_be_accepted(planner):
    def publish_displaced_window():
        planner.on_actual_transform = None
        planner.button[1] += 0.04
        for _ in range(8):
            planner.advance(0.001)
            planner.publish_frame()

    planner.on_actual_transform = publish_displaced_window

    success, message = planner._verify_approach_reached(planner.target)

    assert not success
    assert 'button moved' in message.lower()
    assert planner.actual_transform_calls == 1
    assert planner.sleeps <= 9


@pytest.mark.parametrize('trajectory_success', [False, True])
def test_execution_reports_motion_and_post_motion_failures_separately(
    planner, trajectory_success,
):
    observed = copy.deepcopy(planner._execution_observation)
    planner._planned_trajectory = trajectory(planner._latest_joint_positions)
    planner._planned_target = copy.deepcopy(planner.target)
    planner._planned_button = observed['button'].copy()
    planner._planned_observation = observed
    planner._plan_created_at = planner.clock[0]
    planner._latest_received_at = planner.clock[0]
    planner._latest_button = planner._make_pose(
        observed['button'], [0.0, 0.0, 0.0, 1.0],
        Time(nanoseconds=planner.now_ns).to_msg(),
    )
    planner._execute_trajectory = lambda _: (trajectory_success, 'controller failed')
    planner.set_actual_rotation(rotation_z(20.0) @ LEVEL_CAMERA)

    response = planner._execute_callback(None, Trigger.Response())

    assert not response.success
    assert not planner._busy
    assert not planner._executing_coarse_target
    assert planner._planned_trajectory is None
    if trajectory_success:
        assert response.message.startswith(
            'Trajectory reached; post-motion verification failed: ',
        )
        assert planner.statuses[-1].startswith('APPROACH_VERIFICATION_FAILED')
        assert 'tilt' in response.message.lower()
        assert planner.actual_transform_calls > 0
    else:
        assert response.message == 'controller failed'
        assert planner.statuses[-1].startswith('EXECUTION_FAILED')
        assert planner.actual_transform_calls == 0
