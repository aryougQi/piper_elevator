"""Geometry and joint checks for coarse approach candidate poses."""

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass

import cv2
import numpy as np

from piper_elevator_app.motion_core import camera_level_roll_error
from piper_elevator_app.motion_core import matrix_to_quaternion
from piper_elevator_app.motion_core import quaternion_to_matrix


def stable_observation_window(samples, minimum_samples, position_tolerance,
                              normal_tolerance):
    """Reject isolated depth errors without mixing a sustained target change.

    Require at least 80% positional inliers and three consecutive recent
    inliers. A newly displaced target invalidates the output immediately;
    only a new coherent window can reacquire it.
    """
    if len(samples) < minimum_samples:
        return None, None, f'acquiring samples {len(samples)}/{minimum_samples}'
    points = np.asarray([sample['button'] for sample in samples])
    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    inliers = np.isfinite(distances) & (distances <= position_tolerance)
    count = int(np.count_nonzero(inliers))
    detail = f'position inliers={count}/{len(samples)}'
    if count < minimum_samples or count < math.ceil(0.8 * len(samples)):
        return None, None, detail
    if not np.all(inliers[-3:]):
        return None, None, detail + '; recent position changed'
    normal, normal_detail = stable_surface_normal(
        [sample['normal'] for sample, keep in zip(samples, inliers) if keep],
        normal_tolerance,
    )
    if normal is None:
        return None, None, detail + '; ' + normal_detail
    return np.mean(points[inliers], axis=0), normal, detail + '; ' + normal_detail


def stable_surface_normal(normals, tolerance):
    """Average base-frame normals only when uncertainty and trend are small.

    Independent depth fits can be noisy even for a stationary panel. Check
    the uncertainty of the mean and the change between window halves instead
    of requiring every pair of raw fits to agree. This is a noise estimate,
    not a guarantee against systematic depth/calibration errors.
    """
    values = np.asarray(normals, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 3 or len(values) < 3
            or not np.all(np.isfinite(values))):
        return None, 'invalid normal samples'
    lengths = np.linalg.norm(values, axis=1)
    if np.any(lengths < 1e-9):
        return None, 'zero normal sample'
    values = values / lengths[:, None]
    mean = np.mean(values, axis=0)
    if np.linalg.norm(mean) < 0.5:
        return None, 'inconsistent normal directions'
    mean /= np.linalg.norm(mean)
    angles = np.arccos(np.clip(values @ mean, -1.0, 1.0))
    uncertainty = 2.0 * float(np.sqrt(np.sum(angles ** 2))) / len(values)
    halves = [np.mean(part, axis=0) for part in np.array_split(values, 2)]
    if any(np.linalg.norm(part) < 0.5 for part in halves):
        return None, 'inconsistent normal directions'
    halves = [part / np.linalg.norm(part) for part in halves]
    trend = math.acos(float(np.clip(halves[0] @ halves[1], -1.0, 1.0)))
    detail = (f'normal uncertainty={math.degrees(uncertainty):.1f}deg, '
              f'window change={math.degrees(trend):.1f}deg, '
              f'limit={math.degrees(tolerance):.1f}deg')
    if uncertainty > tolerance or trend > tolerance:
        return None, detail
    return mean, detail


@dataclass(frozen=True)
class CameraModel:
    """Validated ROS CameraInfo intrinsics for the image being checked."""

    width: int
    height: int
    k: object
    d: object
    distortion_model: str = 'plumb_bob'

    def __post_init__(self):
        """Validate and copy the calibration supplied by CameraInfo."""
        for name in ('width', 'height'):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError(
                    'Camera image dimensions must be positive integers'
                )
        try:
            k = np.asarray(self.k, dtype=np.float64).reshape(3, 3).copy()
            d = np.asarray(self.d, dtype=np.float64).reshape(-1).copy()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                'Camera intrinsics must contain a 3x3 K and numeric D'
            ) from exc
        if not np.all(np.isfinite(k)) or not np.all(np.isfinite(d)):
            raise ValueError('Camera intrinsics must be finite')
        if (
            k[0, 0] <= 0.0 or k[1, 1] <= 0.0
            or not np.allclose(k[2], [0.0, 0.0, 1.0], atol=1.0e-9, rtol=0.0)
            or abs(k[0, 1]) > 1.0e-9 or abs(k[1, 0]) > 1.0e-9
        ):
            raise ValueError(
                'Camera K must have positive focal lengths and zero skew'
            )
        model = self.distortion_model
        if not isinstance(model, str):
            raise ValueError('Camera distortion model must be a string')
        if model == '' and d.size == 0:
            model = 'plumb_bob'
        supported_sizes = {
            'plumb_bob': (0, 4, 5),
            'rational_polynomial': (8, 12, 14),
            'equidistant': (4,),
        }
        if (
            model not in supported_sizes
            or d.size not in supported_sizes[model]
        ):
            raise ValueError(
                'Unsupported camera distortion model or coefficient count'
            )
        k.setflags(write=False)
        d.setflags(write=False)
        object.__setattr__(self, 'k', k)
        object.__setattr__(self, 'd', d)
        object.__setattr__(self, 'distortion_model', model)


def _finite_vector(values, size, name):
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f'{name} must contain {size} finite values')
    return vector.copy()


def check_camera_view(
    button_position,
    surface_normal,
    tool_position,
    tool_orientation,
    tool_to_camera_translation,
    tool_to_camera_quaternion,
    camera_model,
    *,
    button_radius_m,
    position_uncertainty_m,
    image_margin_ratio,
    minimum_depth_m,
    maximum_depth_m,
    maximum_tilt_rad,
    maximum_roll_rad,
):
    """Check a mounted camera's button footprint, panel tilt and image roll.

    Poses and the inward panel normal are expressed in the base frame. The
    mount transform maps camera optical coordinates into tool coordinates.
    This validates one pose's geometric view, not occlusion or path visibility.
    """
    try:
        if not isinstance(camera_model, CameraModel):
            raise ValueError('A validated CameraModel is required')
        scalars = (
            button_radius_m, position_uncertainty_m, image_margin_ratio,
            minimum_depth_m, maximum_depth_m,
            maximum_tilt_rad, maximum_roll_rad,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError('Camera-view limits must be finite')
        if button_radius_m <= 0.0 or position_uncertainty_m < 0.0:
            raise ValueError(
                'Button radius must be positive and uncertainty non-negative'
            )
        if not 0.0 <= image_margin_ratio < 0.5:
            raise ValueError('Image margin ratio must be in [0, 0.5)')
        if not 0.0 < minimum_depth_m < maximum_depth_m:
            raise ValueError(
                'Camera depth limits must be positive and increasing'
            )
        if not 0.0 <= maximum_tilt_rad <= math.pi / 2.0:
            raise ValueError('Maximum camera tilt must be in [0, pi/2]')
        if not 0.0 <= maximum_roll_rad <= math.pi:
            raise ValueError('Maximum camera roll must be in [0, pi]')
        button = _finite_vector(button_position, 3, 'Button position')
        normal = _finite_vector(surface_normal, 3, 'Surface normal')
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm < 1.0e-9:
            raise ValueError('Surface normal must be non-zero')
        normal /= normal_norm
        tool = _finite_vector(tool_position, 3, 'Tool position')
        mount_position = _finite_vector(
            tool_to_camera_translation, 3, 'Camera mount translation',
        )
        tool_rotation = quaternion_to_matrix(
            _finite_vector(tool_orientation, 4, 'Tool orientation'),
        )
        mount_rotation = quaternion_to_matrix(
            _finite_vector(
                tool_to_camera_quaternion, 4, 'Camera mount orientation',
            ),
        )
        camera_position = tool + tool_rotation @ mount_position
        camera_rotation = tool_rotation @ mount_rotation
        tilt = math.acos(float(np.clip(
            np.dot(camera_rotation[:, 2], normal), -1.0, 1.0,
        )))
        if tilt > maximum_tilt_rad + 1.0e-9:
            return False, (
                f'Camera tilt {math.degrees(tilt):.1f} deg exceeds limit'
            )
        roll = camera_level_roll_error(
            matrix_to_quaternion(camera_rotation),
            normal.copy(), [0.0, 0.0, 1.0],
        )
        if not math.isfinite(roll):
            return False, 'Panel normal has no defined image-up direction'
        if abs(roll) > maximum_roll_rad + 1.0e-9:
            return False, (
                f'Camera roll {math.degrees(roll):.1f} deg exceeds limit'
            )

        # An inflated square in the panel plane contains the button and its
        # tangential localization uncertainty, including all four corners.
        tangent_x = camera_rotation[:, 0] - np.dot(
            camera_rotation[:, 0], normal,
        ) * normal
        tangent_norm = float(np.linalg.norm(tangent_x))
        if tangent_norm < 1.0e-9:
            return False, 'Camera axes cannot define the button footprint'
        tangent_x /= tangent_norm
        tangent_y = np.cross(normal, tangent_x)
        radius = float(button_radius_m + position_uncertainty_m)
        footprint = np.asarray([
            button + radius * (sx * tangent_x + sy * tangent_y)
            for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0))
        ])
        camera_points = (footprint - camera_position) @ camera_rotation
        if not np.all(np.isfinite(camera_points)):
            return False, 'Button footprint has non-finite camera coordinates'
        depths = camera_points[:, 2]
        if (
            np.any(depths < minimum_depth_m)
            or np.any(depths > maximum_depth_m)
        ):
            return False, (
                'Button footprint is outside the valid camera depth range'
            )
        points = camera_points.reshape(-1, 1, 3)
        if camera_model.distortion_model == 'equidistant':
            pixels, _ = cv2.fisheye.projectPoints(
                points, np.zeros(3), np.zeros(3), camera_model.k,
                camera_model.d.reshape(4, 1),
            )
        else:
            pixels, _ = cv2.projectPoints(
                points, np.zeros(3), np.zeros(3), camera_model.k,
                camera_model.d if camera_model.d.size else None,
            )
        pixels = pixels.reshape(-1, 2)
        if not np.all(np.isfinite(pixels)):
            return False, 'Button footprint has non-finite image coordinates'
        lower = image_margin_ratio * np.array([
            camera_model.width, camera_model.height,
        ])
        upper = np.array([
            camera_model.width - 1, camera_model.height - 1,
        ]) - lower
        if np.any(pixels < lower) or np.any(pixels > upper):
            return False, 'Button footprint leaves the image safety margin'
        return True, (
            f'Button visible; tilt={math.degrees(tilt):.1f} deg, '
            f'roll={math.degrees(roll):.1f} deg, '
            f'depth={depths[-1]:.3f} m'
        )
    except (TypeError, ValueError, OverflowError, cv2.error) as exc:
        return False, f'Invalid camera-view geometry: {exc}'


def parse_arm_joint_limits(robot_description_xml, joint_names):
    """Read finite bounded revolute joint limits from a URDF document."""
    try:
        names = list(joint_names)
    except TypeError as exc:
        raise ValueError('Arm joint names must be a sequence') from exc
    if (
        isinstance(joint_names, str)
        or not all(isinstance(name, str) and name for name in names)
    ):
        raise ValueError('Arm joint names must be non-empty strings')
    if not names or len(names) != len(set(names)):
        raise ValueError('Arm joint names must be non-empty and unique')
    try:
        root = ET.fromstring(robot_description_xml)
    except (ET.ParseError, TypeError) as exc:
        raise ValueError('Robot description is not valid URDF XML') from exc
    if root.tag != 'robot':
        raise ValueError('Robot description must have a robot root')
    joints = {}
    for joint in root.findall('joint'):
        name = joint.get('name')
        if name in names:
            if name in joints:
                raise ValueError(f'Duplicate arm joint in URDF: {name}')
            joints[name] = joint
    result = {}
    for name in names:
        joint = joints.get(name)
        if joint is None or joint.get('type') != 'revolute':
            raise ValueError(
                f'Arm joint {name} must be a bounded revolute joint'
            )
        limit = joint.find('limit')
        try:
            lower = float(limit.get('lower'))
            upper = float(limit.get('upper'))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                f'Missing or invalid limits for arm joint {name}'
            ) from exc
        if (
            not math.isfinite(lower) or not math.isfinite(upper)
            or lower >= upper
        ):
            raise ValueError(f'Invalid bounded limits for arm joint {name}')
        result[name] = (lower, upper)
    return result


def joint_configuration_is_safe(
    positions, limits, margin, wrist_joint, minimum_abs_wrist_bend,
):
    """Check each arm joint's limit margin and a minimum wrist bend."""
    try:
        if (
            not isinstance(positions, Mapping)
            or not isinstance(limits, Mapping)
        ):
            raise ValueError('Joint positions and limits must be mappings')
        if not limits:
            raise ValueError('Arm joint limits are empty')
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError('Joint margin must be finite and non-negative')
        if (
            not math.isfinite(minimum_abs_wrist_bend)
            or minimum_abs_wrist_bend < 0.0
        ):
            raise ValueError(
                'Minimum wrist bend must be finite and non-negative'
            )
        if wrist_joint not in limits:
            raise ValueError('Wrist joint is missing from arm limits')
        for name, bounds in limits.items():
            lower, upper = bounds
            value = positions[name]
            if not all(math.isfinite(v) for v in (lower, upper, value)):
                raise ValueError(
                    f'Non-finite joint position or limits: {name}'
                )
            if lower >= upper or lower + margin > upper - margin:
                raise ValueError(
                    f'Invalid joint limits or excessive margin: {name}'
                )
            if value < lower + margin or value > upper - margin:
                return False, (
                    f'Joint {name} is outside its limit margin: '
                    f'value={value:+.9g}rad, required='
                    f'[{lower + margin:+.9g}, {upper - margin:+.9g}]rad, '
                    f'margin={margin:.6g}rad'
                )
        if abs(positions[wrist_joint]) < minimum_abs_wrist_bend:
            return False, (
                f'Joint {wrist_joint} is too close to a straight wrist: '
                f'value={positions[wrist_joint]:+.9g}rad, '
                f'minimum_abs={minimum_abs_wrist_bend:.6g}rad'
            )
        return True, 'All arm joints retain limit margin and wrist bend'
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return False, f'Invalid joint configuration: {exc}'


def joint_configuration_cost(positions, current, limits):
    """Rank configurations by motion from current and joint limit clearance."""
    if (
        not all(isinstance(value, Mapping)
                for value in (positions, current, limits))
        or not limits
    ):
        return math.inf
    cost = 0.0
    try:
        for name in sorted(limits):
            lower, upper = limits[name]
            target = positions[name]
            start = current[name]
            if not all(
                math.isfinite(v) for v in (lower, upper, target, start)
            ):
                return math.inf
            if lower >= upper or not lower <= target <= upper:
                return math.inf
            span = upper - lower
            travel = (target - start) / span
            clearance = min(target - lower, upper - target) / span
            cost += travel * travel + 0.002 / max(clearance, 1.0e-6)
    except (KeyError, TypeError, ValueError, OverflowError):
        return math.inf
    return float(cost)
