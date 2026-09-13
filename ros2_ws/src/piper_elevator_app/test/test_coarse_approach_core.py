import math

import numpy as np
import pytest

from piper_elevator_app.coarse_approach_core import CameraModel
from piper_elevator_app.coarse_approach_core import check_camera_view
from piper_elevator_app.coarse_approach_core import joint_configuration_cost
from piper_elevator_app.coarse_approach_core import joint_configuration_is_safe
from piper_elevator_app.coarse_approach_core import parse_arm_joint_limits
from piper_elevator_app.motion_core import matrix_to_quaternion
from piper_elevator_app.motion_core import quaternion_to_matrix


LEVEL_CAMERA = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])
CAMERA_K = [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]


def camera_model(distortion=None, model='plumb_bob'):
    return CameraModel(640, 480, CAMERA_K, distortion or [], model)


def check_view(**changes):
    arguments = dict(
        button_position=[0.5, 0.0, 0.3],
        surface_normal=[1.0, 0.0, 0.0],
        tool_position=[0.3, 0.0, 0.3],
        tool_orientation=matrix_to_quaternion(LEVEL_CAMERA),
        tool_to_camera_translation=[0.0, 0.0, 0.0],
        tool_to_camera_quaternion=[0.0, 0.0, 0.0, 1.0],
        camera_model=camera_model(),
        button_radius_m=0.015,
        position_uncertainty_m=0.005,
        image_margin_ratio=0.10,
        minimum_depth_m=0.08,
        maximum_depth_m=1.0,
        maximum_tilt_rad=math.radians(10.0),
        maximum_roll_rad=math.radians(15.0),
    )
    arguments.update(changes)
    return check_camera_view(**arguments)


def axis_rotation(axis, degrees):
    angle = math.radians(degrees) / 2.0
    return quaternion_to_matrix([
        *(np.asarray(axis) * math.sin(angle)), math.cos(angle),
    ])


def test_level_camera_with_panel_normal_along_base_x_is_visible():
    accepted, reason = check_view()
    assert accepted, reason
    assert 'tilt=0.0' in reason
    assert 'roll=0.0' in reason


def test_nonidentity_camera_mount_uses_both_rotation_and_translation():
    mount_rotation = axis_rotation([0.0, 1.0, 0.0], 90.0)
    tool_rotation = LEVEL_CAMERA @ mount_rotation.T
    mount_translation = np.array([0.04, 0.02, -0.01])
    tool_position = (
        np.array([0.3, 0.0, 0.3]) - tool_rotation @ mount_translation
    )

    accepted, reason = check_view(
        tool_position=tool_position,
        tool_orientation=matrix_to_quaternion(tool_rotation),
        tool_to_camera_translation=mount_translation,
        tool_to_camera_quaternion=matrix_to_quaternion(mount_rotation),
    )
    assert accepted, reason


@pytest.mark.parametrize('angle, accepted', [(5.0, True), (16.0, False)])
def test_base_z_rotation_is_camera_tilt_for_a_base_x_panel(angle, accepted):
    rotated = axis_rotation([0.0, 0.0, 1.0], angle) @ LEVEL_CAMERA
    result, reason = check_view(
        tool_position=np.array([0.5, 0.0, 0.3]) - 0.2 * rotated[:, 2],
        tool_orientation=matrix_to_quaternion(rotated),
    )
    assert result is accepted, reason
    if not accepted:
        assert 'tilt' in reason


def test_rotation_around_optical_axis_checks_image_roll_separately():
    rotated = axis_rotation([1.0, 0.0, 0.0], 18.0) @ LEVEL_CAMERA
    accepted, reason = check_view(
        tool_orientation=matrix_to_quaternion(rotated),
    )
    assert not accepted
    assert 'roll' in reason


def test_correct_tcp_position_does_not_imply_button_visible():
    accepted, reason = check_view(
        tool_to_camera_translation=[0.14, 0.0, 0.0],
    )
    assert not accepted
    assert 'image safety margin' in reason


def test_button_behind_camera_rejected_even_with_matching_orientation():
    accepted, reason = check_view(tool_position=[0.6, 0.0, 0.3])
    assert not accepted
    assert 'depth' in reason


def test_complete_button_footprint_must_fit_even_when_center_fits():
    accepted, reason = check_view(button_position=[0.5, -0.075, 0.3])
    assert not accepted
    assert 'image safety margin' in reason
    accepted, reason = check_view(
        button_position=[0.5, -0.075, 0.3],
        button_radius_m=0.005,
        position_uncertainty_m=0.0,
    )
    assert accepted, reason


@pytest.mark.parametrize('model, distortion', [
    ('plumb_bob', [1.0, 0.0, 0.0, 0.0, 0.0]),
    ('rational_polynomial', [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ('equidistant', [3.0, 0.0, 0.0, 0.0]),
])
def test_distortion_changes_border_visibility(model, distortion):
    pose = dict(
        button_position=[0.5, -0.074, 0.3],
        button_radius_m=0.008,
        position_uncertainty_m=0.002,
    )
    accepted, reason = check_view(**pose)
    assert accepted, reason
    accepted, reason = check_view(
        **pose, camera_model=camera_model(distortion, model),
    )
    assert not accepted
    assert 'image safety margin' in reason


def test_calibration_accepts_numpy_arrays_and_does_not_alias_inputs():
    k = np.asarray(CAMERA_K).reshape(3, 3)
    d = np.zeros(5)
    calibrated = CameraModel(np.int64(640), 480, k, d, 'plumb_bob')
    k[0, 0] = 0.0
    d[0] = 2.0
    assert calibrated.k[0, 0] == 600.0
    assert calibrated.d[0] == 0.0
    assert not calibrated.k.flags.writeable


@pytest.mark.parametrize('changes', [
    {'width': 0},
    {'height': 3.5},
    {'k': [0.0] * 9},
    {'k': [math.nan] * 9},
    {'k': [1.0, 2.0]},
    {'d': [math.inf] * 5},
    {'d': [0.0] * 3},
    {'distortion_model': 'unknown'},
    {'distortion_model': np.array(['plumb_bob', 'equidistant'])},
    {'distortion_model': 'equidistant', 'd': []},
])
def test_invalid_camera_calibration_is_rejected(changes):
    arguments = dict(width=640, height=480, k=CAMERA_K, d=[])
    arguments.update(changes)
    with pytest.raises(ValueError):
        CameraModel(**arguments)


@pytest.mark.parametrize('changes', [
    {'button_position': [math.nan, 0.0, 0.3]},
    {'surface_normal': [0.0, 0.0, 0.0]},
    {'tool_orientation': [0.0, 0.0, 0.0, 0.0]},
    {'tool_orientation': [0.0, 1.0]},
    {'tool_to_camera_translation': [math.inf, 0.0, 0.0]},
    {'tool_to_camera_quaternion': [0.0, 0.0, math.nan, 1.0]},
    {'image_margin_ratio': 0.5},
    {'position_uncertainty_m': -0.1},
    {'minimum_depth_m': 2.0},
    {'maximum_tilt_rad': math.nan},
    {'maximum_roll_rad': -0.1},
    {'camera_model': None},
])
def test_invalid_view_geometry_fails_closed(changes):
    accepted, reason = check_view(**changes)
    assert not accepted
    assert 'Invalid' in reason


def test_view_check_does_not_modify_normal_or_pose_arrays():
    normal = np.array([2.0, 0.0, 0.0])
    orientation = matrix_to_quaternion(LEVEL_CAMERA)
    initial_orientation = orientation.copy()
    accepted, reason = check_view(
        surface_normal=normal, tool_orientation=orientation,
    )
    assert accepted, reason
    assert normal == pytest.approx([2.0, 0.0, 0.0])
    assert orientation == pytest.approx(initial_orientation)


ROBOT_XML = '''<robot name="test">
  <joint name="joint1" type="revolute"><limit lower="-2" upper="2"/></joint>
  <joint name="joint5" type="revolute">
    <limit lower="-1.5" upper="1.5"/>
  </joint>
  <joint name="camera" type="fixed"/>
</robot>'''


def test_parse_arm_limits_uses_requested_joint_order_and_ignores_fixed_mount():
    limits = parse_arm_joint_limits(ROBOT_XML, ['joint5', 'joint1'])
    assert list(limits) == ['joint5', 'joint1']
    assert limits == {'joint1': (-2.0, 2.0), 'joint5': (-1.5, 1.5)}


@pytest.mark.parametrize('xml, names', [
    ('<robot>', ['joint1']),
    ('<node/>', ['joint1']),
    (ROBOT_XML, []),
    (ROBOT_XML, None),
    (ROBOT_XML, 'joint1'),
    (ROBOT_XML, [['joint1']]),
    (ROBOT_XML, ['joint1', 'joint1']),
    (ROBOT_XML, ['missing']),
    (ROBOT_XML, ['camera']),
    (ROBOT_XML.replace('lower="-2"', 'lower="nan"'), ['joint1']),
    (ROBOT_XML.replace('upper="2"', 'upper="-3"'), ['joint1']),
    (ROBOT_XML.replace('lower="-2"', ''), ['joint1']),
    (ROBOT_XML.replace('type="revolute"', 'type="continuous"'), ['joint1']),
])
def test_malformed_or_unbounded_arm_limits_are_rejected(xml, names):
    with pytest.raises(ValueError):
        parse_arm_joint_limits(xml, names)


def safe_joints(positions, **changes):
    arguments = dict(
        positions=positions,
        limits=parse_arm_joint_limits(ROBOT_XML, ['joint1', 'joint5']),
        margin=0.1,
        wrist_joint='joint5',
        minimum_abs_wrist_bend=0.2,
    )
    arguments.update(changes)
    return joint_configuration_is_safe(**arguments)


def test_all_joint_limit_margins_apply_even_with_healthy_wrist():
    assert safe_joints({'joint1': 0.0, 'joint5': 0.6})[0]
    accepted, reason = safe_joints({'joint1': 1.95, 'joint5': 0.6})
    assert not accepted
    assert 'joint1' in reason


@pytest.mark.parametrize('bend', [-0.1, 0.0, 0.1])
def test_straight_wrist_is_rejected_in_both_directions(bend):
    accepted, reason = safe_joints({'joint1': 0.0, 'joint5': bend})
    assert not accepted
    assert 'straight wrist' in reason


@pytest.mark.parametrize('positions, changes', [
    ({'joint1': 0.0}, {}),
    ({'joint1': math.nan, 'joint5': 0.6}, {}),
    ({'joint1': 0.0, 'joint5': 0.6}, {'limits': {}}),
    ({'joint1': 0.0, 'joint5': 0.6}, {'limits': ['joint1']}),
    ({'joint1': 0.0, 'joint5': 0.6}, {'margin': math.inf}),
    ({'joint1': 0.0, 'joint5': 0.6}, {'margin': 2.0}),
    ({'joint1': 0.0, 'joint5': 0.6}, {'wrist_joint': 'missing'}),
])
def test_malformed_joint_configurations_fail_closed(positions, changes):
    accepted, reason = safe_joints(positions, **changes)
    assert not accepted
    assert 'Invalid' in reason


def test_candidate_cost_prefers_short_motion_and_clearance():
    limits = parse_arm_joint_limits(ROBOT_XML, ['joint1', 'joint5'])
    current = {'joint1': 0.0, 'joint5': 0.6}
    nearby = {'joint1': 0.2, 'joint5': 0.65}
    distant = {'joint1': 1.5, 'joint5': -0.65}
    assert joint_configuration_cost(nearby, current, limits) < \
        joint_configuration_cost(distant, current, limits)
    near_limit = {'joint1': 1.98, 'joint5': 0.6}
    better_margin = {'joint1': 1.5, 'joint5': 0.6}
    assert joint_configuration_cost(better_margin, near_limit, limits) < \
        joint_configuration_cost(near_limit, near_limit, limits)


@pytest.mark.parametrize('positions', [
    {}, {'joint1': math.nan, 'joint5': 0.5}, {'joint1': 3.0, 'joint5': 0.5},
])
def test_invalid_candidates_cannot_rank_ahead_of_valid_candidates(positions):
    limits = parse_arm_joint_limits(ROBOT_XML, ['joint1', 'joint5'])
    assert math.isinf(joint_configuration_cost(
        positions, {'joint1': 0.0, 'joint5': 0.6}, limits,
    ))


def test_cost_rejects_non_mapping_limits():
    assert math.isinf(joint_configuration_cost({}, {}, np.array([1.0, 2.0])))


def test_normal_window_filters_noise_but_rejects_rotation_and_scatter():
    from piper_elevator_app.coarse_approach_core import stable_surface_normal
    tolerance = math.radians(5)

    def normals(degrees):
        angles = np.radians(degrees)
        return np.column_stack([np.sin(angles), np.zeros(len(angles)), np.cos(angles)])

    # Adjacent raw measurements differ by 16 degrees; the panel is stationary.
    result, _ = stable_surface_normal(normals([-8, 8] * 10), tolerance)
    assert result == pytest.approx([0, 0, 1])
    # A genuine 12-degree step must invalidate a mixed old/new window.
    assert stable_surface_normal(normals([0] * 10 + [12] * 10), tolerance)[0] is None
    assert stable_surface_normal(normals(np.linspace(0, 20, 20)), tolerance)[0] is None
    assert stable_surface_normal(normals([-25, 25] * 10), tolerance)[0] is None
    # Once the new orientation is stable it can be reacquired, not frozen out.
    result, _ = stable_surface_normal(normals([12] * 20), tolerance)
    assert result == pytest.approx(normals([12])[0])


def test_observation_window_tolerates_old_outlier_but_not_current_displacement():
    from piper_elevator_app.coarse_approach_core import stable_observation_window
    samples = [dict(button=np.array([.5, 0, .3]), normal=np.array([1., 0, 0])) for _ in range(12)]
    samples[2]['button'][0] += .014
    point, normal, _ = stable_observation_window(samples, 8, .008, math.radians(5))
    assert point == pytest.approx([.5, 0, .3])
    assert normal == pytest.approx([1, 0, 0])
    samples[-1]['button'][0] += .014
    assert stable_observation_window(samples, 8, .008, math.radians(5))[0] is None
    # A sustained change cannot be hidden by averaging it into the old target.
    for sample in samples[-6:]: sample['button'] = np.array([.52, 0, .3])
    assert stable_observation_window(samples, 8, .008, math.radians(5))[0] is None
