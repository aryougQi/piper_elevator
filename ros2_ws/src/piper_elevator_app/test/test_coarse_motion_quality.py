"""Optional posture improvements must retain an already valid coarse plan."""

import copy
from types import SimpleNamespace

import pytest
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetMotionPlan
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint

from test_approach_planning import JOINT_NAMES, PlannerHarness, ik_result, observation


def plan_result(start, endpoint, intermediate=()):
    result = MoveGroup.Result()
    result.error_code.val = MoveItErrorCodes.SUCCESS
    result.trajectory_start.joint_state.name = list(JOINT_NAMES)
    result.trajectory_start.joint_state.position = [
        float(start[name]) for name in JOINT_NAMES
    ]
    trajectory = result.planned_trajectory.joint_trajectory
    trajectory.joint_names = list(JOINT_NAMES)
    for second, positions in enumerate((start, *intermediate, endpoint)):
        point = JointTrajectoryPoint()
        point.positions = [float(positions[name]) for name in JOINT_NAMES]
        point.time_from_start.sec = second
        trajectory.points.append(point)
    return result


def quality_setup(monkeypatch):
    planner = PlannerHarness()
    clock = [100.0]
    monkeypatch.setattr(
        'piper_elevator_app.button_approach_planner.time.monotonic',
        lambda: clock[0],
    )
    planner._planning_deadline = 130.0
    planner._latest_joint_received_at = clock[0]
    planner._load_arm_joint_limits = lambda: None
    planner._quality_plan_client = SimpleNamespace(service_is_ready=lambda: True)
    planner._execute_trajectory = lambda _: pytest.fail(
        'Quality evaluation must not execute a trajectory'
    )
    observed = observation()
    observed['selected_button'] = '2'
    planner._latest_observation = observed
    target = next(planner._candidate_poses(observed))
    start = dict(planner._latest_joint_positions)
    baseline_endpoint = dict(start, joint4=1.2, joint6=-1.2)
    baseline = plan_result(start, baseline_endpoint)
    natural_endpoint = dict(start, joint4=0.1, joint6=-0.1)
    natural = plan_result(start, natural_endpoint)
    planner._fk_pose = lambda _joints, **_kwargs: copy.deepcopy(target)
    candidates = [(0.0, baseline_endpoint, target)]
    return (
        planner, observed, target, start, baseline, natural,
        natural_endpoint, candidates, clock,
    )


def improve(planner, baseline, target, observed, candidates):
    return planner._improve_coarse_plan(
        baseline, target, observed, candidates, 'validated baseline',
    )


def test_improvement_requires_better_real_metrics_and_valid_endpoint(monkeypatch):
    (planner, observed, target, start, baseline, natural,
     endpoint, candidates, _) = quality_setup(monkeypatch)
    before = copy.deepcopy(baseline)
    assert planner._quality_metrics(natural, start)['score'] < (
        planner._quality_metrics(baseline, start)['score'])
    planner._quality_candidate_pool = lambda *_args: [(endpoint, target)]
    planner._plan_quality_candidate = lambda *_args: natural

    selected, selected_target, _message = improve(
        planner, baseline, target, observed, candidates,
    )

    assert selected is natural
    assert selected_target == target
    assert baseline == before
    assert planner._validate_planned_candidate(natural, target, observed)[0]


def test_a_valid_but_worse_plan_keeps_the_baseline(monkeypatch):
    (planner, observed, target, start, baseline, _natural,
     _endpoint, candidates, _) = quality_setup(monkeypatch)
    endpoint = dict(start, joint4=1.5, joint6=-1.5)
    worse = plan_result(start, endpoint)
    assert planner._quality_metrics(worse, start)['score'] > (
        planner._quality_metrics(baseline, start)['score'])
    planner._quality_candidate_pool = lambda *_args: [(endpoint, target)]
    planner._plan_quality_candidate = lambda *_args: worse

    selected, selected_target, _message = improve(
        planner, baseline, target, observed, candidates,
    )

    assert selected is baseline
    assert selected_target == target


@pytest.mark.parametrize('failure', ['joint_limit', 'wrist_bend', 'field_of_view'])
def test_better_score_cannot_override_production_safety(monkeypatch, failure):
    (planner, observed, target, _start, baseline, natural,
     endpoint, candidates, _) = quality_setup(monkeypatch)
    planner._quality_candidate_pool = lambda *_args: [(endpoint, target)]
    planner._plan_quality_candidate = lambda *_args: natural
    if failure in ('joint_limit', 'wrist_bend'):
        joint = 'joint1' if failure == 'joint_limit' else 'joint5'
        value = 2.1 if failure == 'joint_limit' else 0.2
        natural.planned_trajectory.joint_trajectory.points[-1].positions[
            JOINT_NAMES.index(joint)
        ] = value
    else:
        actual = copy.deepcopy(target)
        actual.pose.position.y += 0.5
        planner._fk_pose = lambda _joints, **_kwargs: actual

    selected, _selected_target, _message = improve(
        planner, baseline, target, observed, candidates,
    )

    assert selected is baseline


@pytest.mark.parametrize('stage', ['ik', 'planning', 'fk'])
@pytest.mark.parametrize('error', [ValueError('service unavailable'),
                                   RuntimeError('service transport failed')])
def test_optional_service_errors_keep_the_baseline(monkeypatch, stage, error):
    (planner, observed, target, _start, baseline, natural,
     endpoint, candidates, _) = quality_setup(monkeypatch)
    planner._quality_candidate_pool = lambda *_args: [(endpoint, target)]
    planner._plan_quality_candidate = lambda *_args: natural

    def fail(*_args, **_kwargs):
        raise error

    if stage == 'ik':
        planner._quality_candidate_pool = fail
    elif stage == 'planning':
        planner._plan_quality_candidate = fail
    else:
        planner._fk_pose = fail

    selected, selected_target, _message = improve(
        planner, baseline, target, observed, candidates,
    )

    assert selected is baseline
    assert selected_target == target
    assert not planner._motion_stop_unconfirmed


def test_optional_budget_exhaustion_keeps_the_baseline(monkeypatch):
    (planner, observed, target, _start, baseline, _natural,
     endpoint, candidates, clock) = quality_setup(monkeypatch)
    requests = []
    planner._quality_candidate_pool = lambda *_args: [(endpoint, target)]

    def timeout(_joints, _start, deadline):
        requests.append(deadline)
        clock[0] = deadline
        raise ValueError('MoveIt service budget exhausted')

    planner._plan_quality_candidate = timeout
    selected, _selected_target, _message = improve(
        planner, baseline, target, observed, candidates,
    )

    assert selected is baseline
    assert len(requests) == 1
    assert requests[0] <= 103.0
    assert planner._planning_deadline == 130.0


@pytest.mark.parametrize('skip', ['disabled', 'no_total_budget'])
def test_optional_work_is_skipped_when_disabled_or_budget_is_short(monkeypatch, skip):
    (planner, observed, target, _start, baseline, _natural,
     _endpoint, candidates, _) = quality_setup(monkeypatch)
    if skip == 'disabled':
        planner.values['optimize_coarse_motion'] = False
    else:
        planner._planning_deadline = 100.01
    planner._quality_candidate_pool = lambda *_args: pytest.fail(
        'No optional IK request should be made'
    )
    planner._plan_quality_candidate = lambda *_args: pytest.fail(
        'No optional planning request should be made'
    )

    selected, selected_target, _message = improve(
        planner, baseline, target, observed, candidates,
    )

    assert selected is baseline
    assert selected_target == target


def test_soft_preference_does_not_reject_the_only_valid_awkward_pose(monkeypatch):
    (planner, observed, target, _start, baseline, _natural,
     _endpoint, candidates, _) = quality_setup(monkeypatch)
    point = baseline.planned_trajectory.joint_trajectory.points[-1]
    point.positions[JOINT_NAMES.index('joint5')] = 0.4173
    assert planner._validate_planned_candidate(baseline, target, observed)[0]
    planner._quality_candidate_pool = lambda *_args: []

    selected, _selected_target, _message = improve(
        planner, baseline, target, observed, candidates,
    )

    assert selected is baseline


def test_unavailable_optional_service_returns_baseline_without_waiting(monkeypatch):
    (planner, observed, target, _start, baseline, _natural,
     _endpoint, candidates, _) = quality_setup(monkeypatch)
    planner._quality_plan_client.service_is_ready = lambda: False
    planner._quality_candidate_pool = lambda *_args: pytest.fail(
        'An unavailable optional service must not start IK exploration'
    )

    selected, _selected_target, _message = improve(
        planner, baseline, target, observed, candidates,
    )

    assert selected is baseline


def test_later_planning_failure_retains_the_already_valid_improvement(monkeypatch):
    (planner, observed, target, start, baseline, natural,
     endpoint, candidates, _) = quality_setup(monkeypatch)
    second = dict(start, joint4=0.2, joint6=-0.2)
    planner._quality_candidate_pool = lambda *_args: [
        (endpoint, target), (second, target),
    ]
    calls = []

    def propose(joints, _start, _deadline):
        calls.append(joints)
        if joints == second:
            raise ValueError('Preferred PTP planning error -12')
        return natural

    planner._plan_quality_candidate = propose
    selected, _selected_target, _message = improve(
        planner, baseline, target, observed, candidates,
    )

    assert calls == [endpoint, second]
    assert selected is natural


def test_quality_pool_preserves_original_candidates_and_bounds_collision_checked_ik(monkeypatch):
    (planner, observed, target, start, _baseline, _natural,
     endpoint, candidates, _) = quality_setup(monkeypatch)
    planner.values['motion_quality_maximum_ik_calls'] = 4
    candidate_before = copy.deepcopy(candidates)
    requests = []

    def solve(client, request, deadline):
        assert client is planner._ik_client
        assert deadline == 101.0
        requests.append(request)
        if len(requests) == 1:
            response = ik_result()
            response.error_code.val = MoveItErrorCodes.GOAL_IN_COLLISION
            return response
        if len(requests) == 2:
            return ik_result(dict(endpoint, joint5=0.2))
        return ik_result(endpoint)

    planner._call_moveit_service = solve
    pool = planner._quality_candidate_pool(observed, candidates, start, 101.0)

    assert len(requests) == 4
    assert candidates == candidate_before
    assert any(joints == candidates[0][1] and pose == target for joints, pose in pool)
    assert sum(joints == endpoint for joints, _pose in pool) == 1
    assert not any(joints['joint5'] == 0.2 for joints, _pose in pool)
    for request in requests:
        ik = request.ik_request
        assert ik.avoid_collisions
        assert dict(zip(ik.robot_state.joint_state.name,
                        ik.robot_state.joint_state.position)) == candidates[0][1]
        assert 0.0 < ik.timeout.sec + ik.timeout.nanosec * 1e-9 <= 0.05


@pytest.mark.parametrize('failure', ['error', 'expired'])
def test_optional_ik_failure_preserves_all_original_candidates(monkeypatch, failure):
    (planner, observed, target, start, _baseline, _natural,
     endpoint, candidates, clock) = quality_setup(monkeypatch)
    candidates.append((1.0, endpoint, target))
    calls = []

    def unavailable(*_args):
        calls.append(True)
        raise ValueError('MoveIt service unavailable')

    planner._call_moveit_service = unavailable
    if failure == 'expired':
        clock[0] = 102.0
    pool = planner._quality_candidate_pool(observed, candidates, start, 101.0)

    assert len(calls) == (1 if failure == 'error' else 0)
    assert len(pool) == len(candidates)
    for _score, joints, original_target in candidates:
        assert any(joints == item and original_target == pose for item, pose in pool)


def test_legacy_attempt_order_gets_a_valid_baseline_before_quality_search(monkeypatch):
    (planner, observed, target, start, _baseline, _natural,
     _endpoint, _candidates, _) = quality_setup(monkeypatch)
    endpoints = [dict(start, joint1=value) for value in (0.1, 0.2, 0.3, 0.4)]
    candidates = [(float(index), joints, target)
                  for index, joints in enumerate(endpoints)]
    planner._solve_visible_candidates = lambda _: candidates
    events = []
    baseline = plan_result(start, endpoints[2])

    def legacy_plan(constraints):
        joint1 = next(item.position for item in constraints.joint_constraints
                      if item.joint_name == 'joint1')
        events.append(joint1)
        if len(events) == 1:
            return None, 'MoveIt planning error -1'
        if len(events) == 2:
            return plan_result(start, dict(endpoints[1], joint5=0.1)), ''
        return baseline, ''

    def quality(result, selected_target, frozen, original, message):
        assert events == [0.1, 0.2, 0.3]
        assert result is baseline
        assert original == candidates
        assert planner._validate_planned_candidate(
            result, selected_target, frozen)[0]
        events.append('quality')
        return result, selected_target, message

    planner._plan_constraints = legacy_plan
    planner._improve_coarse_plan = quality
    response = planner._plan_callback(None, Trigger.Response())

    assert response.success, response.message
    assert events == [0.1, 0.2, 0.3, 'quality']
    assert planner._planned_trajectory == baseline.planned_trajectory


def test_quality_transport_is_plan_only_ptp_with_existing_limits(monkeypatch):
    (planner, _observed, _target, start, _baseline, natural,
     endpoint, _candidates, _) = quality_setup(monkeypatch)
    requests = []
    original_parameters = dict(planner.values)

    def service(client, request, deadline):
        assert client is planner._quality_plan_client
        assert isinstance(request, GetMotionPlan.Request)
        assert deadline == 102.0
        requests.append(request)
        response = GetMotionPlan.Response()
        response.motion_plan_response.error_code.val = MoveItErrorCodes.SUCCESS
        response.motion_plan_response.trajectory_start = natural.trajectory_start
        response.motion_plan_response.trajectory = natural.planned_trajectory
        return response

    planner._call_moveit_service = service
    planner._send_action_goal = lambda *_args: pytest.fail(
        'Quality planning must not send an action goal'
    )
    result = planner._plan_quality_candidate(endpoint, start, 102.0)

    assert result.planned_trajectory == natural.planned_trajectory
    assert len(requests) == 1
    request = requests[0].motion_plan_request
    assert request.pipeline_id == 'pilz_industrial_motion_planner'
    assert request.planner_id == 'PTP'
    assert request.max_velocity_scaling_factor == planner.values['velocity_scaling']
    assert request.max_acceleration_scaling_factor == planner.values['acceleration_scaling']
    assert 0.0 < request.allowed_planning_time <= 2.0
    assert dict(zip(request.start_state.joint_state.name,
                    request.start_state.joint_state.position)) == start
    goal = request.goal_constraints[0]
    assert {joint.joint_name: joint.position for joint in goal.joint_constraints} == endpoint
    assert planner.values == original_parameters


@pytest.mark.parametrize('error_code', [MoveItErrorCodes.PLANNING_FAILED,
                                       MoveItErrorCodes.GOAL_IN_COLLISION,
                                       MoveItErrorCodes.NO_IK_SOLUTION])
def test_quality_transport_never_accepts_failed_moveit_response(monkeypatch, error_code):
    (planner, _observed, _target, start, _baseline, natural,
     endpoint, _candidates, _) = quality_setup(monkeypatch)

    def service(*_args):
        response = GetMotionPlan.Response()
        response.motion_plan_response.error_code.val = error_code
        response.motion_plan_response.trajectory = natural.planned_trajectory
        return response

    planner._call_moveit_service = service
    with pytest.raises(ValueError):
        planner._plan_quality_candidate(endpoint, start, 102.0)
