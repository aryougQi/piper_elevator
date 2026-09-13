"""Exercise coarse planning with real geometry and in-memory MoveIt replies."""

import copy
from collections import Counter, deque
import math
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK
from rclpy.time import Time
from std_srvs.srv import Trigger

from piper_elevator_app.button_approach_planner import ButtonApproachPlanner
from piper_elevator_app.coarse_approach_core import CameraModel
from piper_elevator_app.motion_core import matrix_to_quaternion
from piper_elevator_app.motion_core import quaternion_to_matrix


LEVEL_CAMERA = np.array([
    [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0],
])
JOINT_NAMES = [f'joint{index}' for index in range(1, 7)]


class PlannerHarness(ButtonApproachPlanner):
    """Run production methods without initializing a node or ROS transport."""

    def __init__(self):
        self.values = {}
        self._declare_parameters()
        self.values['observation_minimum_samples'] = 3
        self._lock = threading.Lock()
        self._base_frame = 'base_link'
        self._end_effector_link = 'tip'
        self._workspace_min = np.array([-0.65, -0.65, 0.02])
        self._workspace_max = np.array([0.65, 0.65, 0.75])
        self._arm_joint_limits = {name: (-2.0, 2.0) for name in JOINT_NAMES}
        self._latest_joint_positions = dict.fromkeys(JOINT_NAMES, 0.0)
        self._latest_joint_positions['joint5'] = 0.6
        self._latest_joint_received_at = time.monotonic()
        self._planning_deadline = time.monotonic() + 30.0
        self._ik_client = object()
        self._busy = False
        self._motion_stop_unconfirmed = False
        self._latest_observation = None
        self._latest_approach = None
        self._selected_button = ''
        self._selection_changed_stamp_ns = 0
        self._camera_model = camera_model()
        self._observations = deque(maxlen=3)
        self._observation_counts = Counter()
        self._observation_detail = 'waiting for surface poses'
        self._observation_last_rejection = ''
        self._surface_input_received_at = None
        self._surface_input_stamp_ns = 0
        self._clear_stored_plan_locked()
        self.statuses = []
        self.published = []
        publisher = SimpleNamespace(publish=self.published.append)
        self._button_base_publisher = publisher
        self._approach_publisher = publisher
        self.now_ns = 10_000_000_000

    def declare_parameter(self, name, value):
        self.values[name] = value

    def get_parameter(self, name):
        return SimpleNamespace(value=self.values[name])

    def get_clock(self):
        return SimpleNamespace(now=lambda: Time(nanoseconds=self.now_ns))

    def get_logger(self):
        return SimpleNamespace(
            warning=lambda *args, **kwargs: None,
            info=lambda *args, **kwargs: None,
        )

    def _publish_status(self, status):
        self.statuses.append(status)

    def _publish_display_trajectory(self, result):
        self.published.append(result)


def camera_model():
    return CameraModel(
        640, 480, [600., 0., 320., 0., 600., 240., 0., 0., 1.], [],
    )


def observation():
    return {
        'button': np.array([0.5, 0.0, 0.3]),
        'normal': np.array([1.0, 0.0, 0.0]),
        'tip_to_camera_translation': np.zeros(3),
        'tip_to_camera_quaternion': np.array([0.0, 0.0, 0.0, 1.0]),
        'camera_orientation': matrix_to_quaternion(LEVEL_CAMERA),
        'camera_model': camera_model(),
        'camera_frame': 'camera_color_optical_frame',
        'stamp_ns': 9_950_000_000,
        'received_at': time.monotonic(),
    }


def trajectory(joints):
    return SimpleNamespace(joint_trajectory=SimpleNamespace(
        joint_names=list(joints),
        points=[SimpleNamespace(
            positions=list(joints.values()),
            time_from_start=SimpleNamespace(sec=second, nanosec=0),
        ) for second in (0, 2)],
    ))


def ik_result(joints=None):
    result = GetPositionIK.Response()
    result.error_code.val = (
        MoveItErrorCodes.SUCCESS if joints else MoveItErrorCodes.NO_IK_SOLUTION
    )
    if joints:
        result.solution.joint_state.name = list(joints)
        result.solution.joint_state.position = list(joints.values())
    return result


def test_candidates_compensate_nonidentity_mount_and_limit_camera_angles():
    planner = PlannerHarness()
    observed = observation()
    mount = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    observed['tip_to_camera_quaternion'] = matrix_to_quaternion(mount)
    observed['tip_to_camera_translation'] = np.array([0.03, 0.02, 0.0])
    poses = list(planner._candidate_poses(observed))
    assert poses
    assert len(poses) <= 99
    for pose in poses:
        assert planner._view_is_safe(pose, observed)[0]
        tool = pose.pose.orientation
        rotation = quaternion_to_matrix([tool.x, tool.y, tool.z, tool.w])
        camera_z = (rotation @ mount)[:, 2]
        tilt = math.acos(float(np.clip(
            camera_z @ observed['normal'], -1., 1.,
        )))
        assert tilt <= planner.values['maximum_camera_tilt_rad'] + 1e-12
    nominal = planner._candidate_pose(
        observed, planner.values['approach_distance_m'],
        observed['normal'], None,
    )
    assert not planner._view_is_safe(nominal, observed)[0]
    assert poses[0].pose.position.x == pytest.approx(0.33)


def test_candidates_stay_inside_the_tilt_limit_with_sampling_margin():
    """Sampling exactly at the limit made the planner reject its own pose."""
    planner = PlannerHarness()
    observed = observation()
    margin = planner.values['candidate_tilt_margin_rad']
    maximum = planner.values['maximum_camera_tilt_rad']
    poses = list(planner._candidate_poses(observed))
    assert poses
    mount = quaternion_to_matrix(observed['tip_to_camera_quaternion'])
    for pose in poses:
        tool = pose.pose.orientation
        rotation = quaternion_to_matrix([tool.x, tool.y, tool.z, tool.w])
        camera_z = (rotation @ mount)[:, 2]
        tilt = math.acos(float(np.clip(
            camera_z @ observed['normal'], -1., 1.,
        )))
        assert tilt <= maximum - margin + 1e-9


def test_candidates_reject_uncompensated_mount_that_pushes_button_offscreen():
    planner = PlannerHarness()
    observed = observation()
    observed['tip_to_camera_translation'] = np.array([0.4, 0.0, 0.0])
    assert list(planner._candidate_poses(observed)) == []


def test_joint_goal_uses_all_arm_joints_without_base_axis_pose_constraints():
    planner = PlannerHarness()
    goal = planner._joint_goal_constraints(planner._latest_joint_positions)
    assert [joint.joint_name for joint in goal.joint_constraints] == \
        JOINT_NAMES
    assert goal.position_constraints == []
    assert goal.orientation_constraints == []
    for joint in goal.joint_constraints:
        expected = planner._latest_joint_positions[joint.joint_name]
        assert joint.position == expected
        tolerance = planner.values['joint_goal_tolerance_rad']
        assert joint.tolerance_above == tolerance
        assert joint.tolerance_below == tolerance


def test_ik_checks_collisions_both_wrist_branches_and_ranks_safe_solutions():
    planner = PlannerHarness()
    observed = observation()
    target = next(planner._candidate_poses(observed))
    planner._candidate_poses = lambda _: iter([target])
    current = planner._latest_joint_positions
    replies = [
        dict(current, joint1=1.99),
        dict(current, joint1=0.4, joint5=-0.6),
        dict(current, joint1=0.1, joint5=0.6),
        dict(current, joint1=0.4, joint5=-0.6),
        None,
        dict(current, joint1=0.2, joint5=-0.6),
        None,
    ]
    requests = []

    def solve(client, request, deadline):
        assert client is planner._ik_client
        assert deadline <= planner._planning_deadline
        requests.append(request)
        return ik_result(replies[len(requests) - 1])

    planner._call_moveit_service = solve
    candidates = planner._solve_visible_candidates(observed)
    assert len(requests) == 7
    assert len(candidates) == 3
    scores = [item[0] for item in candidates]
    assert scores == sorted(scores)
    assert candidates[0][1]['joint1'] == 0.1
    wrist_seeds = set()
    for request in requests:
        ik = request.ik_request
        assert ik.avoid_collisions
        assert ik.group_name == planner.values['planning_group']
        assert ik.ik_link_name == 'tip'
        assert 0 < ik.timeout.nanosec <= 50_000_000
        wrist_seeds.add(dict(zip(
            ik.robot_state.joint_state.name,
            ik.robot_state.joint_state.position,
        ))['joint5'])
    assert {-0.6, 0.6} <= wrist_seeds
    requests.clear()
    repeated = planner._solve_visible_candidates(observed)
    assert [(cost, joints) for cost, joints, _ in repeated] == [
        (cost, joints) for cost, joints, _ in candidates
    ]


def test_ik_stops_at_solution_count_and_expired_budget():
    planner = PlannerHarness()
    observed = observation()
    target = next(planner._candidate_poses(observed))
    planner._candidate_poses = lambda _: iter([target] * 100)
    planner.values['maximum_ik_solutions'] = 2
    requests = []

    def solve(client, request, deadline):
        requests.append(request)
        return ik_result(dict(
            planner._latest_joint_positions, joint1=0.05 * len(requests),
        ))

    planner._call_moveit_service = solve
    assert len(planner._solve_visible_candidates(observed)) == 2
    assert len(requests) == 2
    planner._planning_deadline = time.monotonic() - 1.0
    assert planner._solve_visible_candidates(observed) == []
    assert len(requests) == 2


@pytest.mark.parametrize('failure', ['tilt', 'roll', 'field_of_view'])
def test_endpoint_checks_actual_camera_with_legacy_flags_false(failure):
    planner = PlannerHarness()
    planner.values['constrain_coarse_orientation'] = False
    planner.values['preserve_coarse_camera_orientation'] = False
    observed = observation()
    target = next(planner._candidate_poses(observed))
    actual = copy.deepcopy(target)
    if failure == 'field_of_view':
        observed['tip_to_camera_translation'] = np.array([0.2, 0.0, 0.0])
    else:
        angle = math.radians(25.0)
        c, s = math.cos(angle), math.sin(angle)
        rotation = (
            np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
            if failure == 'tilt'
            else np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
        ) @ LEVEL_CAMERA
        q = matrix_to_quaternion(rotation)
        orientation = actual.pose.orientation
        orientation.x, orientation.y = map(float, q[:2])
        orientation.z, orientation.w = map(float, q[2:])
    safe, reason = planner._validate_endpoint(actual, target, observed)
    assert not safe
    expected = 'image safety margin' if failure == 'field_of_view' else failure
    assert expected in reason


def configure_planning(planner, observed, count=3):
    targets = list(planner._candidate_poses(observed))[:count]
    joints = [dict(planner._latest_joint_positions, joint1=0.1 * (i + 1))
              for i in range(count)]
    planner._latest_observation = observed
    planner._solve_visible_candidates = lambda _: list(zip(
        range(count), joints, targets,
    ))
    planner._fk_pose = lambda actual: targets[min(
        range(count), key=lambda i: abs(joints[i]['joint1'] - actual['joint1'])
    )]
    return joints, targets


def test_plan_retries_after_moveit_failure_and_unsafe_trajectory():
    planner = PlannerHarness()
    joints, targets = configure_planning(planner, observation())
    attempted = []

    def plan(constraints):
        attempted.append(constraints)
        if len(attempted) == 1:
            return None, 'MoveIt planning error -1'
        selected = dict(joints[len(attempted) - 1])
        if len(attempted) == 2:
            selected['joint1'] = 3.0
        return SimpleNamespace(planned_trajectory=trajectory(selected)), ''

    planner._plan_constraints = plan
    result = planner._plan_callback(None, Trigger.Response())
    assert result.success, result.message
    assert len(attempted) == 3
    assert planner._planned_target == targets[2]
    assert not planner._busy
    assert planner._planning_deadline is None


@pytest.mark.parametrize('drift, succeeds', [(0.004, True), (0.05, False)])
def test_plan_freezes_observation_and_rejects_excessive_drift(drift, succeeds):
    planner = PlannerHarness()
    observed = observation()
    original_button = observed['button'].copy()
    joints, _ = configure_planning(planner, observed, count=1)

    def plan(constraints):
        planner._latest_observation['button'][1] += drift
        planner._latest_observation['received_at'] = time.monotonic()
        return SimpleNamespace(planned_trajectory=trajectory(joints[0])), ''

    planner._plan_constraints = plan
    result = planner._plan_callback(None, Trigger.Response())
    assert result.success is succeeds, result.message
    if succeeds:
        assert planner._planned_button == pytest.approx(original_button)
        frozen = planner._planned_observation['button']
        assert frozen == pytest.approx(original_button)
        assert planner._planned_observation is not planner._latest_observation
    else:
        assert 'Target changed' in result.message
        assert planner._planned_trajectory is None
    assert not planner._busy


def make_transform(position, orientation):
    transform = TransformStamped()
    translation = transform.transform.translation
    translation.x, translation.y, translation.z = map(float, position)
    rotation = transform.transform.rotation
    rotation.x, rotation.y, rotation.z, rotation.w = map(float, orientation)
    return transform


def test_new_selection_clears_plan_and_rejects_queued_old_camera_frame():
    planner = PlannerHarness()
    planner._latest_observation = observation()
    planner._planned_trajectory = object()
    planner._observations.append(observation())
    planner._selection_callback(SimpleNamespace(data='button_2'))
    assert planner._selected_button == 'button_2'
    assert planner._planned_trajectory is None
    assert planner._latest_observation is None
    assert len(planner._observations) == 0
    queued = PoseStamped()
    queued.header.frame_id = planner.values['camera_frame']
    queued.header.stamp = Time(nanoseconds=9_900_000_000).to_msg()
    queued.pose.position.z = 0.2
    queued.pose.orientation.w = 1.0
    planner._surface_pose_callback(queued)
    assert planner._latest_observation is None
    assert len(planner._observations) == 0


def test_surface_pose_requires_distinct_stable_frames_and_resets_on_jump():
    planner = PlannerHarness()
    base_tf = make_transform(
        [0.3, 0.0, 0.3], matrix_to_quaternion(LEVEL_CAMERA),
    )
    mount_tf = make_transform([0., 0., 0.], [0., 0., 0., 1.])
    planner._lookup_message_transform = lambda *args: base_tf
    planner._tf_buffer = SimpleNamespace(
        lookup_transform=lambda *a, **k: mount_tf,
    )

    def publish(stamp_ns, x=0.0):
        message = PoseStamped()
        message.header.frame_id = planner.values['camera_frame']
        message.header.stamp = Time(nanoseconds=stamp_ns).to_msg()
        message.pose.position.x = x
        message.pose.position.z = 0.2
        message.pose.orientation.w = 1.0
        planner._surface_pose_callback(message)

    publish(9_800_000_000)
    publish(9_800_000_000)
    assert len(planner._observations) == 1
    assert planner._latest_observation is None
    publish(9_850_000_000, 0.001)
    assert planner._latest_observation is None
    publish(9_900_000_000, 0.002)
    accepted = planner._latest_observation
    assert accepted['button'] == pytest.approx([0.5, -0.001, 0.3])
    assert accepted['normal'] == pytest.approx([1.0, 0.0, 0.0])
    assert accepted['stamp_ns'] == 9_900_000_000
    publish(9_950_000_000, 0.05)
    assert planner._latest_observation is None
    assert planner._latest_approach is None
    assert len(planner._observations) == 1


def test_planning_waits_for_new_stable_observation_without_using_stale_cache(
    monkeypatch,
):
    planner = PlannerHarness()
    planner._latest_observation = observation()
    planner._latest_observation['received_at'] = time.monotonic() - 2.0
    refreshed = observation()
    pauses = []

    def receive_after_pause(duration):
        pauses.append(duration)
        planner._latest_observation = refreshed

    monkeypatch.setattr(
        'piper_elevator_app.button_approach_planner.time.sleep',
        receive_after_pause,
    )
    accepted = planner._wait_for_planning_observation()
    assert pauses
    assert accepted['received_at'] == refreshed['received_at']
    assert accepted is not refreshed


def test_observation_wait_consumes_old_plan_and_honors_bounded_timeout(
    monkeypatch,
):
    planner = PlannerHarness()
    planner._planned_trajectory = object()
    planner.values['planning_observation_wait_seconds'] = 0.04
    clock = [100.0]
    monkeypatch.setattr(
        'piper_elevator_app.button_approach_planner.time.monotonic',
        lambda: clock[0],
    )
    monkeypatch.setattr(
        'piper_elevator_app.button_approach_planner.time.sleep',
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    result = planner._plan_callback(None, Trigger.Response())
    assert not result.success
    assert 'Timed out waiting for stable' in result.message
    assert clock[0] <= 100.06
    assert planner._planned_trajectory is None
    assert not planner._busy


def test_observation_wait_is_within_overall_planning_budget(monkeypatch):
    planner = PlannerHarness()
    clock = [100.0]
    planner._planning_deadline = 100.02
    monkeypatch.setattr(
        'piper_elevator_app.button_approach_planner.time.monotonic',
        lambda: clock[0],
    )
    monkeypatch.setattr(
        'piper_elevator_app.button_approach_planner.time.sleep',
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    with pytest.raises(ValueError, match='Timed out waiting for stable'):
        planner._wait_for_planning_observation()
    assert clock[0] == pytest.approx(100.02)


def test_normal_window_handles_depth_noise_and_invalidates_real_rotation():
    planner = PlannerHarness()
    planner._observations = deque(maxlen=20)
    planner._lookup_message_transform = lambda *args: make_transform(
        [0.3, 0, 0.3], matrix_to_quaternion(LEVEL_CAMERA))
    planner._tf_buffer = SimpleNamespace(lookup_transform=lambda *a, **k:
                                        make_transform([0, 0, 0], [0, 0, 0, 1]))

    def publish(index, angle):
        planner.now_ns = 10_000_000_000 + index * 100_000_000
        message = PoseStamped()
        message.header.frame_id = planner.values['camera_frame']
        message.header.stamp = Time(nanoseconds=planner.now_ns).to_msg()
        message.pose.position.z = 0.2
        message.pose.orientation.y = math.sin(math.radians(angle) / 2)
        message.pose.orientation.w = math.cos(math.radians(angle) / 2)
        planner._surface_pose_callback(message)

    for index in range(20):
        publish(index, -8 if index % 2 else 8)
    assert planner._latest_observation is not None
    assert planner._latest_observation['normal'] == pytest.approx([1, 0, 0])
    for index in range(20, 30):
        publish(index, 20)
    assert planner._latest_observation is None
    for index in range(30, 40):
        publish(index, 20)
    assert planner._latest_observation is not None
    # A source timestamp gap must flush old samples even if callbacks arrive together.
    publish(80, 20)
    assert planner._latest_observation is None
    assert len(planner._observations) == 1


@pytest.mark.parametrize('drift, succeeds', [(0.0, True), (0.05, False)])
def test_endpoint_waits_for_recovery_and_rechecks_original_target(monkeypatch, drift, succeeds):
    planner = PlannerHarness()
    observed = observation()
    joints, _ = configure_planning(planner, observed, count=1)

    def plan(constraints):
        planner._latest_observation = None
        return SimpleNamespace(planned_trajectory=trajectory(joints[0])), ''

    def recover(seconds):
        refreshed = observation()
        refreshed['button'][1] += drift
        planner._latest_observation = refreshed

    planner._plan_constraints = plan
    monkeypatch.setattr('piper_elevator_app.button_approach_planner.time.sleep', recover)
    result = planner._plan_callback(None, Trigger.Response())
    assert result.success is succeeds, result.message
    if not succeeds:
        assert 'Target changed' in result.message
        assert planner._planned_trajectory is None


def test_normal_filter_is_in_base_frame_when_camera_rotates():
    planner = PlannerHarness()
    planner._observations = deque(maxlen=20)
    planner._tf_buffer = SimpleNamespace(lookup_transform=lambda *a, **k:
                                        make_transform([0, 0, 0], [0, 0, 0, 1]))
    for index in range(20):
        angle = math.radians(-10 if index % 2 else 10)
        local_rotation = quaternion_to_matrix([0, math.sin(angle / 2), 0,
                                               math.cos(angle / 2)])
        camera_rotation = LEVEL_CAMERA @ local_rotation
        camera_tf = make_transform([0.3, 0, 0.3], matrix_to_quaternion(camera_rotation))
        planner._lookup_message_transform = lambda *args: camera_tf
        planner.now_ns = 10_000_000_000 + index * 100_000_000
        message = PoseStamped()
        message.header.frame_id = planner.values['camera_frame']
        message.header.stamp = Time(nanoseconds=planner.now_ns).to_msg()
        point = camera_rotation.T @ np.array([0.2, 0, 0])
        message.pose.position.x, message.pose.position.y, message.pose.position.z = map(float, point)
        message.pose.orientation.y = -math.sin(angle / 2)
        message.pose.orientation.w = math.cos(angle / 2)
        planner._surface_pose_callback(message)
    assert planner._latest_observation['button'] == pytest.approx([0.5, 0, 0.3])
    assert planner._latest_observation['normal'] == pytest.approx([1, 0, 0])


def test_position_outlier_does_not_restart_twenty_frame_acquisition():
    planner = PlannerHarness()
    planner.values['observation_minimum_samples'] = 8
    planner._observations = deque(maxlen=20)
    planner._lookup_message_transform = lambda *args: make_transform(
        [0.3, 0, 0.3], matrix_to_quaternion(LEVEL_CAMERA))
    planner._tf_buffer = SimpleNamespace(lookup_transform=lambda *a, **k:
                                        make_transform([0, 0, 0], [0, 0, 0, 1]))

    def publish(index, depth):
        planner.now_ns = 10_000_000_000 + index * 100_000_000
        message = PoseStamped()
        message.header.frame_id = planner.values['camera_frame']
        message.header.stamp = Time(nanoseconds=planner.now_ns).to_msg()
        message.pose.position.z = depth
        message.pose.orientation.w = 1.0
        planner._surface_pose_callback(message)

    for i in range(8): publish(i, .2)
    assert planner._latest_observation is not None
    publish(8, .214)
    assert planner._latest_observation is None
    assert len(planner._observations) == 9
    for i in range(9, 12): publish(i, .2)
    assert planner._latest_observation is not None
    assert planner._latest_observation['button'] == pytest.approx([.5, 0, .3])
    # A real 2 cm displacement must stop use of the old target immediately.
    publish(12, .22)
    assert planner._latest_observation is None
    # A larger identity jump resets the window outright.
    publish(13, .27)
    assert planner._latest_observation is None
    assert len(planner._observations) == 1


def test_recent_receipt_does_not_refresh_expired_capture_before_planning(monkeypatch):
    planner = PlannerHarness()
    old = observation()
    old['stamp_ns'] = planner.now_ns - 600_000_000
    planner._latest_observation = old
    refreshed = observation()
    pauses = []

    def receive(duration):
        pauses.append(duration)
        planner._latest_observation = refreshed

    monkeypatch.setattr('piper_elevator_app.button_approach_planner.time.sleep', receive)
    assert planner._wait_for_planning_observation()['stamp_ns'] == refreshed['stamp_ns']
    assert pauses


@pytest.mark.parametrize('stamp_ns', [0, 10_100_000_000, 9_400_000_000])
def test_diagnostics_distinguish_bad_capture_time_from_unstable_geometry(stamp_ns):
    planner = PlannerHarness()
    message = PoseStamped()
    message.header.frame_id = planner.values['camera_frame']
    message.header.stamp = Time(nanoseconds=stamp_ns).to_msg()
    planner._surface_pose_callback(message)
    diagnostic = planner._observation_diagnostics()
    assert diagnostic['counts'] == {'received': 1, 'rejected_invalid_stamp_or_age': 1}
    assert diagnostic['last_rejection'] == 'invalid_stamp_or_age'
    assert not diagnostic['ready']


def test_capture_expires_while_waiting_for_tf():
    planner = PlannerHarness()
    message = PoseStamped()
    message.header.frame_id = planner.values['camera_frame']
    message.header.stamp = Time(nanoseconds=planner.now_ns - 400_000_000).to_msg()
    message.pose.position.z = 0.2
    message.pose.orientation.w = 1.0

    def slow_transform(*args):
        planner.now_ns += 200_000_000
        return make_transform([0.3, 0, 0.3], matrix_to_quaternion(LEVEL_CAMERA))

    planner._lookup_message_transform = slow_transform
    planner._tf_buffer = SimpleNamespace(lookup_transform=lambda *a, **k:
                                        make_transform([0, 0, 0], [0, 0, 0, 1]))
    planner._surface_pose_callback(message)
    assert len(planner._observations) == 0
    assert planner._observation_last_rejection == 'expired_during_tf'


def test_capture_gap_flushes_window_even_when_callbacks_arrive_together():
    planner = PlannerHarness()
    planner._observations.append(dict(observation(), stamp_ns=9_000_000_000))
    planner._lookup_message_transform = lambda *args: make_transform(
        [0.3, 0, 0.3], matrix_to_quaternion(LEVEL_CAMERA))
    planner._tf_buffer = SimpleNamespace(lookup_transform=lambda *a, **k:
                                        make_transform([0, 0, 0], [0, 0, 0, 1]))
    message = PoseStamped()
    message.header.frame_id = planner.values['camera_frame']
    message.header.stamp = Time(nanoseconds=planner.now_ns).to_msg()
    message.pose.position.z = 0.2
    message.pose.orientation.w = 1.0
    planner._surface_pose_callback(message)
    assert len(planner._observations) == 1
    assert planner._observation_counts['window_resets'] == 1
    assert 'source_gap=1.000s' in planner._observation_detail


def test_real_noise_window_converges_without_relaxing_normal_drift_check():
    planner = PlannerHarness()
    planner.values.update(observation_stable_samples=40,
                          observation_minimum_samples=8,
                          observation_window_max_seconds=6.0)
    planner._observations = deque(maxlen=40)
    planner._lookup_message_transform = lambda *args: make_transform(
        [0.3, 0, 0.3], matrix_to_quaternion(LEVEL_CAMERA))
    planner._tf_buffer = SimpleNamespace(lookup_transform=lambda *a, **k:
                                        make_transform([0, 0, 0], [0, 0, 0, 1]))

    def publish(index, degrees):
        planner.now_ns = 10_000_000_000 + index * 100_000_000
        message = PoseStamped()
        message.header.frame_id = planner.values['camera_frame']
        message.header.stamp = Time(nanoseconds=planner.now_ns).to_msg()
        message.pose.position.z = 0.2
        angle = math.radians(degrees) / 2
        message.pose.orientation.y = math.sin(angle)
        message.pose.orientation.w = math.cos(angle)
        planner._surface_pose_callback(message)

    for index in range(20):
        publish(index, 15 if index % 2 else -15)
    assert planner._latest_observation is None
    for index in range(20, 40):
        publish(index, 15 if index % 2 else -15)
    assert planner._latest_observation is not None
    assert planner._latest_observation['normal'] == pytest.approx([1, 0, 0])
    for index in range(40, 60):
        publish(index, 30)
    assert planner._latest_observation is None
    for index in range(60, 80):
        publish(index, 30)
    assert planner._latest_observation is not None
