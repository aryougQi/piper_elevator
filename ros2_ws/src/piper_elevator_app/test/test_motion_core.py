import math

import numpy as np
import pytest

from piper_elevator_app.motion_core import position_in_workspace
from piper_elevator_app.motion_core import quaternion_to_matrix
from piper_elevator_app.motion_core import transform_button_to_approach


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
