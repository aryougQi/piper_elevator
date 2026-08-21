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
