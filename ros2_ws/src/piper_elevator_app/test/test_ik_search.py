"""Bounded IK coverage and failure diagnostics using production safety checks."""

import copy
import math

import numpy as np
import pytest
from moveit_msgs.msg import MoveItErrorCodes

from piper_elevator_app.motion_core import (
    camera_level_roll_error, level_limited_camera_orientation,
    quaternion_to_matrix,
)
from test_approach_planning import PlannerHarness, ik_result, observation


def pose_key(target):
    position = target.pose.position
    orientation = target.pose.orientation
    return tuple(round(value, 8) for value in (
        position.x, position.y, position.z,
        orientation.x, orientation.y, orientation.z, orientation.w,
    ))


def search_setup(monkeypatch, count=1, budget=8.0):
    planner = PlannerHarness()
    observed = observation()
    observed['selected_button'] = '2'
    base = next(planner._candidate_poses(observed))
    targets = [copy.deepcopy(base) for _ in range(count)]
    for index, target in enumerate(targets):
        target.pose.position.x += index * 0.0001
    planner._candidate_poses = lambda _: iter(targets)
    clock = [100.0]
    planner._planning_deadline = clock[0] + 30.0
    planner._latest_joint_received_at = clock[0]
    planner.values['ik_search_budget_seconds'] = budget
    monkeypatch.setattr(
        'piper_elevator_app.button_approach_planner.time.monotonic',
        lambda: clock[0],
    )
    return planner, observed, targets, clock


def test_last_pose_gets_current_seed_before_alternate_seed_budget(monkeypatch):
    planner, observed, targets, clock = search_setup(monkeypatch, count=24)
    planner.values['maximum_ik_solutions'] = 1
    requests = []
    desired = pose_key(targets[-1])
    solution = dict(planner._latest_joint_positions, joint1=0.1)

    def solve(client, request, deadline):
        assert client is planner._ik_client
        assert deadline == pytest.approx(108.0)
        requests.append(request)
        clock[0] += 0.06
        if pose_key(request.ik_request.pose_stamped) == desired:
            return ik_result(solution)
        return ik_result()

    planner._call_moveit_service = solve
    solutions = planner._solve_visible_candidates(observed)

    assert len(solutions) == 1
    assert solutions[0][1] == solution
    assert pose_key(solutions[0][2]) == desired
    assert [pose_key(request.ik_request.pose_stamped) for request in requests] == [
        pose_key(target) for target in targets
    ]
    for request in requests:
        ik = request.ik_request
        assert ik.avoid_collisions
        assert ik.timeout.sec == 0
        assert 0 < ik.timeout.nanosec <= 50_000_000
        assert dict(zip(
            ik.robot_state.joint_state.name,
            ik.robot_state.joint_state.position,
        )) == planner._latest_joint_positions
    diagnostic = planner._last_ik_search_diagnostic
    assert diagnostic['candidates_attempted'] == len(targets)
    assert diagnostic['accepted_solutions'] == 1
    assert diagnostic['stop_reason'] == 'solution_limit'
    assert diagnostic['elapsed_seconds'] == pytest.approx(1.44)


def test_geometrically_identical_candidates_are_not_repeated():
    planner = PlannerHarness()
    observed = observation()
    targets = list(planner._candidate_poses(observed))
    assert targets
    assert len(set(map(pose_key, targets))) == len(targets)
    assert all(planner._view_is_safe(target, observed)[0] for target in targets)
    diagnostic = planner._candidate_view_diagnostic
    assert diagnostic['duplicates'] >= 3
    assert diagnostic['visible'] == len(targets)
    assert diagnostic['generated'] == (
        diagnostic['visible'] + diagnostic['duplicates']
        + sum(diagnostic['view_rejections'].values())
    )


def camera_geometry(target, observed):
    q = target.pose.orientation
    tool_rotation = quaternion_to_matrix([q.x, q.y, q.z, q.w])
    camera_rotation = tool_rotation @ quaternion_to_matrix(
        observed['tip_to_camera_quaternion'],
    )
    tilt = math.acos(float(np.clip(
        camera_rotation[:, 2] @ observed['normal'], -1.0, 1.0,
    )))
    # The test fixture has an identity camera mount.
    roll = camera_level_roll_error(
        [q.x, q.y, q.z, q.w], observed['normal'].copy(), [0.0, 0.0, 1.0],
    )
    return camera_rotation[:, 2], tilt, roll


def test_candidate_search_covers_tilt_roll_combinations_with_original_limits():
    planner = PlannerHarness()
    observed = observation()
    before = copy.deepcopy(planner.values)
    targets = list(planner._candidate_poses(observed))
    angles = [camera_geometry(target, observed)[1:] for target in targets]

    assert targets
    assert any(tilt > math.radians(5.1) for tilt, _ in angles)
    assert any(
        tilt > math.radians(5.1) and abs(roll) > math.radians(9.0)
        for tilt, roll in angles
    )
    assert all(
        tilt <= math.radians(10.0) + 1e-9
        and abs(roll) <= math.radians(15.0) + 1e-9
        for tilt, roll in angles
    )
    assert all(planner._view_is_safe(target, observed)[0] for target in targets)
    assert len(set(map(pose_key, targets))) == len(targets)
    assert planner.values == before


@pytest.mark.parametrize('current_tilt_deg', [7.0, 17.0])
def test_current_non_axis_direction_is_preserved_or_clamped_to_planning_limit(
    current_tilt_deg,
):
    planner = PlannerHarness()
    observed = observation()
    normal = observed['normal']
    level = quaternion_to_matrix(observed['camera_orientation'])
    tangent = (level[:, 0] + 0.3 * level[:, 1]) / math.sqrt(1.09)
    angle = math.radians(current_tilt_deg)
    current_axis = math.cos(angle) * normal + math.sin(angle) * tangent
    observed['camera_orientation'] = level_limited_camera_orientation(
        current_axis, observed['camera_orientation'],
        np.array([0.0, 0.0, 1.0]), 0.0,
    )
    # Candidates are sampled inside the planning limit so the realised tilt
    # after IK cannot exceed it.
    permitted_angle = min(
        angle,
        planner.values['maximum_camera_tilt_rad']
        - planner.values['candidate_tilt_margin_rad'],
    )
    expected_axis = math.cos(permitted_angle) * normal + math.sin(permitted_angle) * tangent

    targets = list(planner._candidate_poses(observed))
    axes = [camera_geometry(target, observed)[0] for target in targets]
    assert any(np.linalg.norm(axis - expected_axis) < 1e-8 for axis in axes)
    assert all(planner._view_is_safe(target, observed)[0] for target in targets)
    assert len(set(map(pose_key, targets))) == len(targets)


def test_clamped_current_direction_still_requires_complete_button_visibility():
    planner = PlannerHarness()
    observed = observation()
    normal = observed['normal']
    level = quaternion_to_matrix(observed['camera_orientation'])
    tangent = (level[:, 0] + level[:, 1]) / math.sqrt(2.0)
    angle = math.radians(17.0)
    observed['camera_orientation'] = level_limited_camera_orientation(
        math.cos(angle) * normal + math.sin(angle) * tangent,
        observed['camera_orientation'], np.array([0.0, 0.0, 1.0]), 0.0,
    )
    limit = planner.values['maximum_camera_tilt_rad']
    clamped_axis = math.cos(limit) * normal + math.sin(limit) * tangent
    for offset in planner.values['approach_distance_offsets_m']:
        target = planner._candidate_pose(
            observed, planner.values['approach_distance_m'] + offset,
            clamped_axis, None,
        )
        safe, reason = planner._view_is_safe(target, observed)
        assert not safe
        assert 'image safety margin' in reason
    targets = list(planner._candidate_poses(observed))
    assert all(
        np.linalg.norm(camera_geometry(target, observed)[0] - clamped_axis) > 1e-8
        for target in targets
    )
    assert all(planner._view_is_safe(target, observed)[0] for target in targets)


def test_expanded_real_candidates_fit_first_seed_pass_with_original_budget(monkeypatch):
    planner = PlannerHarness()
    observed = observation()
    targets = list(planner._candidate_poses(observed))
    clock = [100.0]
    planner._planning_deadline = 130.0
    planner._latest_joint_received_at = clock[0]
    original_budget = planner.values['ik_search_budget_seconds']
    assert original_budget == 8.0
    monkeypatch.setattr(
        'piper_elevator_app.button_approach_planner.time.monotonic',
        lambda: clock[0],
    )
    calls = []

    def solve(client, request, deadline):
        assert deadline == pytest.approx(100.0 + original_budget)
        assert request.ik_request.avoid_collisions
        assert 0 < request.ik_request.timeout.nanosec <= 50_000_000
        calls.append(request)
        clock[0] += 0.06
        return ik_result()

    planner._call_moveit_service = solve
    assert planner._solve_visible_candidates(observed) == []
    assert len(calls) >= len(targets)
    assert [pose_key(request.ik_request.pose_stamped) for request in calls[:len(targets)]] == [
        pose_key(target) for target in targets
    ]
    diagnostic = planner._last_ik_search_diagnostic
    assert diagnostic['candidates_attempted'] == len(targets)
    assert diagnostic['stop_reason'] == 'time_budget'
    assert original_budget <= diagnostic['elapsed_seconds'] <= original_budget + 0.06
    assert planner.values['ik_search_budget_seconds'] == original_budget


@pytest.mark.parametrize('joint,value,reason', [
    ('joint1', 1.84, 'outside its limit margin'),
    ('joint5', 0.405, 'too close to a straight wrist'),
])
def test_ik_success_retains_goal_and_tracking_reserves_in_rejection_report(
    monkeypatch, joint, value, reason,
):
    planner, observed, _, clock = search_setup(monkeypatch)
    unsafe = dict(planner._latest_joint_positions, **{joint: value})

    def solve(client, request, deadline):
        assert request.ik_request.avoid_collisions
        clock[0] += 0.01
        return ik_result(unsafe)

    planner._call_moveit_service = solve
    assert planner._solve_visible_candidates(observed) == []
    diagnostic = planner._last_ik_search_diagnostic
    assert diagnostic['ik_successes'] == diagnostic['ik_calls'] > 0
    assert diagnostic['joint_rejections'] == diagnostic['ik_successes']
    assert diagnostic['accepted_solutions'] == 0
    assert diagnostic['stop_reason'] == 'complete'
    assert diagnostic['joint_rejection_examples']
    for example in diagnostic['joint_rejection_examples']:
        assert joint in example['reason']
        assert reason in example['reason']
        assert example['joints'][joint] == value
    detail = planner._ik_search_failure_detail()
    assert 'selected=2' in detail
    assert reason in detail
    assert 'stop=complete' in detail


def test_unsuccessful_moveit_codes_are_preserved_without_joint_rejections(monkeypatch):
    planner, observed, _, clock = search_setup(monkeypatch)
    codes = [MoveItErrorCodes.NO_IK_SOLUTION, MoveItErrorCodes.FRAME_TRANSFORM_FAILURE]
    expected = {}

    def solve(client, request, deadline):
        assert request.ik_request.avoid_collisions
        code = codes[sum(expected.values()) % len(codes)]
        expected[str(code)] = expected.get(str(code), 0) + 1
        clock[0] += 0.01
        response = ik_result()
        response.error_code.val = code
        return response

    planner._call_moveit_service = solve
    assert planner._solve_visible_candidates(observed) == []
    diagnostic = planner._last_ik_search_diagnostic
    assert diagnostic['ik_return_codes'] == expected
    assert diagnostic['ik_calls'] == sum(expected.values())
    assert diagnostic['ik_successes'] == 0
    assert diagnostic['joint_rejections'] == 0
    assert diagnostic['stop_reason'] == 'complete'
    assert str(MoveItErrorCodes.NO_IK_SOLUTION) in planner._ik_search_failure_detail()


@pytest.mark.parametrize('budget,stop_reason', [(0.13, 'time_budget'), (8.0, 'complete')])
def test_budget_exhaustion_is_distinct_from_completed_search(monkeypatch, budget, stop_reason):
    planner, observed, _, clock = search_setup(monkeypatch, budget=budget)

    def solve(client, request, deadline):
        clock[0] += 0.06
        return ik_result()

    planner._call_moveit_service = solve
    assert planner._solve_visible_candidates(observed) == []
    diagnostic = planner._last_ik_search_diagnostic
    assert diagnostic['stop_reason'] == stop_reason
    assert diagnostic['candidates_attempted'] == 1
    assert diagnostic['ik_calls'] > 0
    if stop_reason == 'time_budget':
        assert diagnostic['elapsed_seconds'] >= budget
    else:
        assert diagnostic['elapsed_seconds'] < budget
    assert f'stop={stop_reason}' in planner._ik_search_failure_detail()


def test_service_failure_is_saved_before_exception_propagates(monkeypatch):
    planner, observed, _, _ = search_setup(monkeypatch)
    message = 'MoveIt service unavailable: /compute_ik'

    def unavailable(client, request, deadline):
        raise ValueError(message)

    planner._call_moveit_service = unavailable
    with pytest.raises(ValueError, match=message):
        planner._solve_visible_candidates(observed)
    diagnostic = planner._last_ik_search_diagnostic
    assert diagnostic['stop_reason'] == 'service_error: ' + message
    assert diagnostic['ik_calls'] == 1
    assert diagnostic['ik_return_codes'] == {}
    assert diagnostic['accepted_solutions'] == 0
    assert message in planner._ik_search_failure_detail()
