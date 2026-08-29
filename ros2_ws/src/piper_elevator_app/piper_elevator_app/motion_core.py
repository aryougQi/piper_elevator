import math

import numpy as np


def quaternion_to_matrix(quaternion):
    """Return a 3x3 rotation matrix for an xyzw quaternion."""
    values = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm < 1.0e-9:
        raise ValueError('Quaternion must be finite and non-zero')
    x, y, z, w = values / norm
    return np.array([
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ])


def matrix_to_quaternion(matrix):
    """Return a normalized xyzw quaternion for a rotation matrix."""
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError('Rotation matrix must be finite and 3x3')

    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array([
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
            0.25 * scale,
        ])
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(
                1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
            ) * 2.0
            quaternion = np.array([
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
            ])
        elif index == 1:
            scale = math.sqrt(
                1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
            ) * 2.0
            quaternion = np.array([
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
            ])
        else:
            scale = math.sqrt(
                1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
            ) * 2.0
            quaternion = np.array([
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ])
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def orientation_from_approach_direction(direction, reference_quaternion):
    """Align tool +Z with ``direction`` while preserving tool roll.

    The current tool X axis is projected onto the requested surface plane, so
    visual servo corrections do not introduce an unnecessary wrist spin.
    """
    tool_z = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(tool_z))
    if tool_z.shape != (3,) or not math.isfinite(norm) or norm < 1.0e-9:
        raise ValueError('Approach direction must be finite and non-zero')
    tool_z /= norm

    reference = quaternion_to_matrix(reference_quaternion)
    tool_x = reference[:, 0] - np.dot(reference[:, 0], tool_z) * tool_z
    if np.linalg.norm(tool_x) < 1.0e-6:
        tool_x = reference[:, 1] - np.dot(reference[:, 1], tool_z) * tool_z
    if np.linalg.norm(tool_x) < 1.0e-6:
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(fallback, tool_z))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        tool_x = fallback - np.dot(fallback, tool_z) * tool_z
    tool_x /= np.linalg.norm(tool_x)
    tool_y = np.cross(tool_z, tool_x)
    tool_y /= np.linalg.norm(tool_y)
    return matrix_to_quaternion(np.column_stack((tool_x, tool_y, tool_z)))


def camera_level_roll_error(
    camera_orientation,
    approach_direction,
    vertical_direction,
):
    """Return signed optical-axis roll relative to a level camera image.

    Optical-frame ``-Y`` is treated as image-up.  The base vertical vector is
    projected onto the button plane, so the result remains independent of
    camera pitch and yaw.  ``nan`` means the surface normal is parallel to the
    vertical reference and therefore has no well-defined image-up direction.
    """
    camera_z = np.asarray(approach_direction, dtype=np.float64)
    vertical = np.asarray(vertical_direction, dtype=np.float64)
    if camera_z.shape != (3,) or vertical.shape != (3,):
        raise ValueError('Direction vectors must contain exactly three values')
    camera_z_norm = float(np.linalg.norm(camera_z))
    vertical_norm = float(np.linalg.norm(vertical))
    if (
        not math.isfinite(camera_z_norm)
        or not math.isfinite(vertical_norm)
        or camera_z_norm < 1.0e-9
        or vertical_norm < 1.0e-9
    ):
        raise ValueError('Direction vectors must be finite and non-zero')
    camera_z /= camera_z_norm
    vertical /= vertical_norm

    image_up = vertical - np.dot(vertical, camera_z) * camera_z
    image_up_norm = float(np.linalg.norm(image_up))
    if image_up_norm < 1.0e-6:
        return math.nan
    image_up /= image_up_norm
    level_y = -image_up
    level_x = np.cross(level_y, camera_z)
    level_x /= np.linalg.norm(level_x)

    aligned = quaternion_to_matrix(
        orientation_from_approach_direction(
            camera_z,
            camera_orientation,
        )
    )
    current_x = aligned[:, 0]
    return math.atan2(
        float(np.dot(camera_z, np.cross(level_x, current_x))),
        float(np.clip(np.dot(level_x, current_x), -1.0, 1.0)),
    )


def level_limited_camera_orientation(
    approach_direction,
    current_camera_orientation,
    vertical_direction,
    maximum_roll,
):
    """Align camera +Z and softly clamp image roll around that axis."""
    if not math.isfinite(maximum_roll) or maximum_roll < 0.0:
        raise ValueError('Maximum camera roll must be finite and non-negative')
    camera_z = np.asarray(approach_direction, dtype=np.float64)
    camera_z_norm = float(np.linalg.norm(camera_z))
    if (
        camera_z.shape != (3,)
        or not math.isfinite(camera_z_norm)
        or camera_z_norm < 1.0e-9
    ):
        raise ValueError('Approach direction must be finite and non-zero')
    camera_z /= camera_z_norm
    roll = camera_level_roll_error(
        current_camera_orientation,
        camera_z,
        vertical_direction,
    )
    if not math.isfinite(roll):
        return orientation_from_approach_direction(
            camera_z,
            current_camera_orientation,
        )

    vertical = np.asarray(vertical_direction, dtype=np.float64)
    vertical /= np.linalg.norm(vertical)
    image_up = vertical - np.dot(vertical, camera_z) * camera_z
    image_up /= np.linalg.norm(image_up)
    level_y = -image_up
    level_x = np.cross(level_y, camera_z)
    level_x /= np.linalg.norm(level_x)
    limited_roll = float(np.clip(roll, -maximum_roll, maximum_roll))
    cosine = math.cos(limited_roll)
    sine = math.sin(limited_roll)
    camera_x = cosine * level_x + sine * level_y
    camera_y = -sine * level_x + cosine * level_y
    return matrix_to_quaternion(
        np.column_stack((camera_x, camera_y, camera_z))
    )


def tool_orientation_for_camera_direction(
    direction,
    current_tool_orientation,
    current_camera_orientation,
    vertical_direction=None,
    maximum_camera_roll=None,
):
    """Orient the tool so the mounted camera +Z follows ``direction``.

    The fixed tool-to-camera rotation is recovered from the two current TF
    orientations.  This prevents a non-identity camera mount from being
    mistaken for the tool frame.  Roll is preserved unless an optional
    vertical reference and maximum camera roll are supplied.
    """
    current_tool = quaternion_to_matrix(current_tool_orientation)
    current_camera = quaternion_to_matrix(current_camera_orientation)
    if vertical_direction is None or maximum_camera_roll is None:
        desired_camera_quaternion = orientation_from_approach_direction(
            direction,
            current_camera_orientation,
        )
    else:
        desired_camera_quaternion = level_limited_camera_orientation(
            direction,
            current_camera_orientation,
            vertical_direction,
            maximum_camera_roll,
        )
    desired_camera = quaternion_to_matrix(desired_camera_quaternion)
    tool_to_camera = current_tool.T @ current_camera
    desired_tool = desired_camera @ tool_to_camera.T
    return matrix_to_quaternion(desired_tool)


def quaternion_angular_distance(first, second):
    """Return the shortest angular distance between xyzw quaternions."""
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    dot = float(np.clip(abs(np.dot(left, right)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


def quaternion_error_rotation_vector(current, target):
    """Return the base-frame rotation vector from current to target."""
    current_q = np.asarray(current, dtype=np.float64)
    target_q = np.asarray(target, dtype=np.float64)
    current_norm = float(np.linalg.norm(current_q))
    target_norm = float(np.linalg.norm(target_q))
    if (
        current_q.shape != (4,)
        or target_q.shape != (4,)
        or not math.isfinite(current_norm)
        or not math.isfinite(target_norm)
        or current_norm < 1.0e-9
        or target_norm < 1.0e-9
    ):
        raise ValueError('Quaternions must be finite and non-zero')
    x1, y1, z1, w1 = target_q / target_norm
    x2, y2, z2, w2 = current_q / current_norm
    error = np.asarray([
        -w1 * x2 + x1 * w2 - y1 * z2 + z1 * y2,
        -w1 * y2 + x1 * z2 + y1 * w2 - z1 * x2,
        -w1 * z2 - x1 * y2 + y1 * x2 + z1 * w2,
        w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2,
    ])
    if error[3] < 0.0:
        error = -error
    vector_norm = float(np.linalg.norm(error[:3]))
    if vector_norm < 1.0e-9:
        return np.zeros(3)
    angle = 2.0 * math.atan2(vector_norm, float(error[3]))
    return error[:3] * (angle / vector_norm)


def interpolate_quaternion(first, second, fraction):
    """Spherically interpolate between xyzw quaternions."""
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    amount = float(np.clip(fraction, 0.0, 1.0))
    if dot > 0.9995:
        result = left + amount * (right - left)
        return result / np.linalg.norm(result)
    angle = math.acos(float(np.clip(dot, -1.0, 1.0)))
    scale = math.sin(angle)
    result = (
        math.sin((1.0 - amount) * angle) / scale * left
        + math.sin(amount * angle) / scale * right
    )
    return result / np.linalg.norm(result)


def visual_servo_target(
    button_position,
    surface_direction,
    current_orientation,
    standoff_distance,
):
    """Return the perpendicular tool pose at the requested standoff."""
    button = np.asarray(button_position, dtype=np.float64)
    normal = np.asarray(surface_direction, dtype=np.float64)
    if button.shape != (3,) or not np.all(np.isfinite(button)):
        raise ValueError('Button position must be finite and three-dimensional')
    normal_norm = float(np.linalg.norm(normal))
    if normal.shape != (3,) or not math.isfinite(normal_norm):
        raise ValueError('Surface direction must contain three finite values')
    if normal_norm < 1.0e-9:
        raise ValueError('Surface direction must be non-zero')
    if not math.isfinite(standoff_distance) or standoff_distance <= 0.0:
        raise ValueError('Standoff distance must be positive')
    normal /= normal_norm
    target_position = button - float(standoff_distance) * normal
    target_orientation = orientation_from_approach_direction(
        normal,
        current_orientation,
    )
    return target_position, target_orientation


def visual_servo_errors(
    current_position,
    current_orientation,
    button_position,
    surface_direction,
    standoff_distance,
):
    """Return axial, lateral, and perpendicular visual-servo errors."""
    current = np.asarray(current_position, dtype=np.float64)
    button = np.asarray(button_position, dtype=np.float64)
    normal = np.asarray(surface_direction, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    button_delta = button - current
    measured_standoff = float(np.dot(button_delta, normal))
    lateral = button_delta - measured_standoff * normal
    current_z = quaternion_to_matrix(current_orientation)[:, 2]
    perpendicular_error = math.acos(
        float(np.clip(np.dot(current_z, normal), -1.0, 1.0))
    )
    return (
        measured_standoff - float(standoff_distance),
        float(np.linalg.norm(lateral)),
        perpendicular_error,
    )


def limit_pose_step(
    current_position,
    current_orientation,
    target_position,
    target_orientation,
    maximum_translation,
    maximum_rotation,
):
    """Clamp one PBVS correction in translation and rotation."""
    current = np.asarray(current_position, dtype=np.float64)
    target = np.asarray(target_position, dtype=np.float64)
    delta = target - current
    distance = float(np.linalg.norm(delta))
    if distance > maximum_translation:
        delta *= float(maximum_translation) / distance

    angular_distance = quaternion_angular_distance(
        current_orientation,
        target_orientation,
    )
    rotation_fraction = 1.0
    if angular_distance > maximum_rotation:
        rotation_fraction = float(maximum_rotation) / angular_distance
    orientation = interpolate_quaternion(
        current_orientation,
        target_orientation,
        rotation_fraction,
    )
    return current + delta, orientation


def transform_button_to_approach(
    button_in_camera,
    base_from_camera_translation,
    base_from_camera_quaternion,
    approach_distance,
):
    """Transform a button and create a TCP pose before it.

    The camera optical +Z axis is treated as the panel approach direction.
    The TCP +Z axis is aligned with that direction, and the approach position
    stays ``approach_distance`` metres in front of the button.
    """
    button_camera = np.asarray(button_in_camera, dtype=np.float64)
    translation = np.asarray(
        base_from_camera_translation,
        dtype=np.float64,
    )
    if button_camera.shape != (3,) or translation.shape != (3,):
        raise ValueError('Positions must contain exactly three values')
    if not np.all(np.isfinite(button_camera)):
        raise ValueError('Button position must be finite')
    if not math.isfinite(approach_distance) or approach_distance <= 0.0:
        raise ValueError('Approach distance must be positive')

    rotation = quaternion_to_matrix(base_from_camera_quaternion)
    button_base = translation + rotation @ button_camera
    approach_direction = rotation[:, 2]
    approach_position = button_base - approach_distance * approach_direction

    tool_z = approach_direction / np.linalg.norm(approach_direction)
    tool_x = rotation[:, 0] - np.dot(rotation[:, 0], tool_z) * tool_z
    if np.linalg.norm(tool_x) < 1.0e-9:
        fallback = np.array([0.0, 0.0, 1.0])
        tool_x = fallback - np.dot(fallback, tool_z) * tool_z
    tool_x /= np.linalg.norm(tool_x)
    tool_y = np.cross(tool_z, tool_x)
    tool_y /= np.linalg.norm(tool_y)
    orientation = matrix_to_quaternion(
        np.column_stack((tool_x, tool_y, tool_z))
    )
    return button_base, approach_position, orientation, approach_direction


def camera_centered_tool_approach_position(
    button_position,
    surface_normal,
    desired_tool_orientation,
    tool_to_camera_translation,
    approach_distance,
    maximum_tangent_shift=None,
):
    """Place the camera over a button without changing TCP standoff.

    An eye-in-hand camera is normally offset from the fingertip.  Driving the
    fingertip directly onto the button-normal ray can therefore move a button
    close to (or outside) the image boundary, especially for panel-edge
    buttons.  This function compensates only the component of that fixed
    camera offset tangent to the button surface.  The fingertip remains
    ``approach_distance`` in front of the surface while the camera optical
    origin lies on the same normal ray as the selected button.
    """
    button = np.asarray(button_position, dtype=np.float64)
    normal = np.asarray(surface_normal, dtype=np.float64)
    tool_to_camera = np.asarray(
        tool_to_camera_translation,
        dtype=np.float64,
    )
    if button.shape != (3,) or normal.shape != (3,):
        raise ValueError('Button position and surface normal must be 3D')
    if tool_to_camera.shape != (3,):
        raise ValueError('Tool-to-camera translation must be 3D')
    if not (
        np.all(np.isfinite(button))
        and np.all(np.isfinite(normal))
        and np.all(np.isfinite(tool_to_camera))
    ):
        raise ValueError('Camera-centering inputs must be finite')
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1.0e-9:
        raise ValueError('Surface normal must be non-zero')
    if not math.isfinite(approach_distance) or approach_distance <= 0.0:
        raise ValueError('Approach distance must be positive')

    normal /= normal_norm
    desired_tool_matrix = quaternion_to_matrix(
        desired_tool_orientation,
    )
    camera_offset = desired_tool_matrix @ tool_to_camera
    tangent_camera_offset = (
        camera_offset - np.dot(camera_offset, normal) * normal
    )
    if maximum_tangent_shift is not None:
        if (
            not math.isfinite(maximum_tangent_shift)
            or maximum_tangent_shift < 0.0
        ):
            raise ValueError(
                'Maximum camera-centering shift must be non-negative'
            )
        tangent_norm = float(np.linalg.norm(tangent_camera_offset))
        if tangent_norm > maximum_tangent_shift and tangent_norm > 1.0e-9:
            tangent_camera_offset *= maximum_tangent_shift / tangent_norm
    nominal_tool_position = button - approach_distance * normal
    return nominal_tool_position - tangent_camera_offset


def position_in_workspace(position, minimum, maximum):
    point = np.asarray(position, dtype=np.float64)
    lower = np.asarray(minimum, dtype=np.float64)
    upper = np.asarray(maximum, dtype=np.float64)
    if point.shape != (3,) or lower.shape != (3,) or upper.shape != (3,):
        raise ValueError('Workspace vectors must contain three values')
    return bool(
        np.all(np.isfinite(point))
        and np.all(point >= lower)
        and np.all(point <= upper)
    )
