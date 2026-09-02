import math

import numpy as np
import pytest

from piper_elevator_app.motion_core import camera_level_roll_error
from piper_elevator_app.motion_core import (
    camera_centered_tool_approach_position,
)
from piper_elevator_app.motion_core import level_limited_camera_orientation
from piper_elevator_app.motion_core import matrix_to_quaternion
from piper_elevator_app.motion_core import (
    orientation_prioritized_linear_command,
)
from piper_elevator_app.motion_core import position_in_workspace
from piper_elevator_app.motion_core import quaternion_error_rotation_vector
from piper_elevator_app.motion_core import quaternion_to_matrix
from piper_elevator_app.motion_core import tangential_spiral_offset
from piper_elevator_app.motion_core import limit_pose_step
from piper_elevator_app.motion_core import transform_button_to_approach
from piper_elevator_app.motion_core import (
    tool_orientation_for_camera_direction,
)
from piper_elevator_app.motion_core import visual_servo_errors
from piper_elevator_app.motion_core import visual_servo_target


def test_tangential_spiral_waits_then_stays_in_safe_plane_and_radius():
    assert tangential_spiral_offset(
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        0.4,
        0.5,
        0.003,
        1.5,
        0.012,
    ) == pytest.approx([0.0, 0.0, 0.0])

    offset = tangential_spiral_offset(
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        20.0,
        0.5,
        0.003,
        1.5,
        0.012,
    )

    assert np.dot(offset, [1.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert np.linalg.norm(offset) == pytest.approx(0.012)


def test_orientation_priority_stops_only_inward_motion_when_misaligned():
    command, scale = orientation_prioritized_linear_command(
        [0.08, 0.01, -0.02],
        [1.0, 0.0, 0.0],
        math.radians(5.0),
        math.radians(2.0),
        math.radians(3.0),
    )

    assert scale == pytest.approx(0.0)
    assert command == pytest.approx([0.0, 0.01, -0.02])


def test_orientation_priority_blends_inward_motion_after_alignment():
    command, scale = orientation_prioritized_linear_command(
        [0.08, 0.01, 0.0],
        [2.0, 0.0, 0.0],
        math.radians(2.5),
        math.radians(2.0),
        math.radians(3.0),
    )

    assert scale == pytest.approx(0.5)
    assert command == pytest.approx([0.04, 0.01, 0.0])


def test_orientation_priority_never_limits_safe_retreat():
    command, scale = orientation_prioritized_linear_command(
        [-0.03, 0.01, 0.0],
        [1.0, 0.0, 0.0],
        math.radians(10.0),
        math.radians(2.0),
        math.radians(3.0),
    )

    assert scale == pytest.approx(1.0)
    assert command == pytest.approx([-0.03, 0.01, 0.0])


def test_identity_transform_creates_standoff_along_optical_z():
    button, approach, orientation, direction = transform_button_to_approach(
        [0.1, -0.2, 0.5],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        0.08,
    )

    assert button == pytest.approx([0.1, -0.2, 0.5])
    assert approach == pytest.approx([0.1, -0.2, 0.42])
    assert direction == pytest.approx([0.0, 0.0, 1.0])
    assert quaternion_to_matrix(orientation) == pytest.approx(np.eye(3))


def test_optical_z_can_point_along_base_x():
    half = math.sqrt(0.5)
    button, approach, _, direction = transform_button_to_approach(
        [0.0, 0.0, 0.45],
        [0.0, 0.0, 0.25],
        [0.0, half, 0.0, half],
        0.08,
    )

    assert button == pytest.approx([0.45, 0.0, 0.25], abs=1.0e-8)
    assert approach == pytest.approx([0.37, 0.0, 0.25], abs=1.0e-8)
    assert direction == pytest.approx([1.0, 0.0, 0.0], abs=1.0e-8)


def test_camera_centering_compensates_only_surface_tangent_offset():
    approach = camera_centered_tool_approach_position(
        [0.50, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.10, 0.02, 0.08],
        0.15,
    )

    assert approach == pytest.approx([0.35, -0.02, -0.08])
    camera_origin = approach + np.array([0.10, 0.02, 0.08])
    assert camera_origin[1:] == pytest.approx([0.0, 0.0])
    assert np.dot(np.array([0.50, 0.0, 0.0]) - approach, [1, 0, 0]) \
        == pytest.approx(0.15)


def test_camera_centering_without_mount_offset_matches_nominal_approach():
    approach = camera_centered_tool_approach_position(
        [0.1, -0.2, 0.5],
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        0.08,
    )

    assert approach == pytest.approx([0.1, -0.2, 0.42])


def test_camera_centering_shift_can_be_limited_for_reachability():
    approach = camera_centered_tool_approach_position(
        [0.50, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.10, 0.0, 0.08],
        0.15,
        0.045,
    )

    assert approach == pytest.approx([0.35, 0.0, -0.045])


def test_camera_visibility_compensates_only_offset_outside_safe_radius():
    approach = camera_centered_tool_approach_position(
        [0.50, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.10, 0.0, 0.08],
        0.15,
        0.045,
        0.055,
    )

    # The 80 mm tangent camera offset is allowed to remain 55 mm off-axis,
    # so the TCP moves only the minimum 25 mm needed for visibility.
    assert approach == pytest.approx([0.35, 0.0, -0.025])


def test_camera_visibility_needs_no_shift_inside_safe_radius():
    approach = camera_centered_tool_approach_position(
        [0.50, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.10, 0.0, 0.08],
        0.15,
        0.045,
        0.090,
    )

    assert approach == pytest.approx([0.35, 0.0, 0.0])


def test_tool_target_accounts_for_camera_mount_rotation():
    half = math.sqrt(0.5)
    current_camera = [0.0, half, 0.0, half]
    desired_tool = tool_orientation_for_camera_direction(
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        current_camera,
    )
    resulting_camera = (
        quaternion_to_matrix(desired_tool)
        @ quaternion_to_matrix(current_camera)
    )

    assert resulting_camera[:, 2] == pytest.approx(
        [0.0, 0.0, 1.0],
        abs=1.0e-8,
    )


def test_camera_roll_is_softly_clamped_to_level_reference():
    camera_z = np.array([1.0, 0.0, 0.0])
    level_x = np.array([0.0, -1.0, 0.0])
    level_y = np.array([0.0, 0.0, -1.0])
    roll = math.radians(35.0)
    current_x = math.cos(roll) * level_x + math.sin(roll) * level_y
    current_y = -math.sin(roll) * level_x + math.cos(roll) * level_y
    current = matrix_to_quaternion(
        np.column_stack((current_x, current_y, camera_z))
    )

    assert math.degrees(
        camera_level_roll_error(
            current,
            camera_z,
            [0.0, 0.0, 1.0],
        )
    ) == pytest.approx(35.0)

    limited = level_limited_camera_orientation(
        camera_z,
        current,
        [0.0, 0.0, 1.0],
        math.radians(10.0),
    )
    assert quaternion_to_matrix(limited)[:, 2] == pytest.approx(camera_z)
    assert math.degrees(
        camera_level_roll_error(
            limited,
            camera_z,
            [0.0, 0.0, 1.0],
        )
    ) == pytest.approx(10.0)


def test_level_constraint_preserves_roll_when_vertical_is_ambiguous():
    current = [0.0, 0.0, 0.0, 1.0]
    result = level_limited_camera_orientation(
        [0.0, 0.0, 1.0],
        current,
        [0.0, 0.0, 1.0],
        math.radians(10.0),
    )

    assert result == pytest.approx(current)
    assert math.isnan(
        camera_level_roll_error(
            current,
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        )
    )


def test_workspace_bounds_are_inclusive():
    assert position_in_workspace(
        [0.40, 0.0, 0.20],
        [-0.65, -0.65, 0.02],
        [0.65, 0.65, 0.75],
    )
    assert not position_in_workspace(
        [0.70, 0.0, 0.20],
        [-0.65, -0.65, 0.02],
        [0.65, 0.65, 0.75],
    )


def test_rejects_invalid_quaternion():
    with pytest.raises(ValueError):
        transform_button_to_approach(
            [0.0, 0.0, 0.4],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            0.08,
        )


def test_visual_servo_target_is_three_cm_before_surface():
    target_position, target_orientation = visual_servo_target(
        [0.50, 0.10, 0.30],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        0.03,
    )

    assert target_position == pytest.approx([0.47, 0.10, 0.30])
    assert quaternion_to_matrix(target_orientation)[:, 2] == pytest.approx(
        [1.0, 0.0, 0.0],
        abs=1.0e-8,
    )
    errors = visual_servo_errors(
        target_position,
        target_orientation,
        [0.50, 0.10, 0.30],
        [1.0, 0.0, 0.0],
        0.03,
    )
    assert errors == pytest.approx([0.0, 0.0, 0.0], abs=1.0e-8)


def test_visual_servo_step_is_bounded():
    half = math.sqrt(0.5)
    position, orientation = limit_pose_step(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.10, 0.0, 0.0],
        [0.0, half, 0.0, half],
        0.02,
        math.radians(5.0),
    )

    assert position == pytest.approx([0.02, 0.0, 0.0])
    rotated_z = quaternion_to_matrix(orientation)[:, 2]
    assert math.degrees(math.acos(rotated_z[2])) == pytest.approx(5.0)


def test_quaternion_error_is_expressed_as_rotation_vector():
    half = math.sqrt(0.5)

    rotation = quaternion_error_rotation_vector(
        [0.0, 0.0, 0.0, 1.0],
        [0.0, half, 0.0, half],
    )

    assert rotation == pytest.approx([0.0, math.pi / 2.0, 0.0])
