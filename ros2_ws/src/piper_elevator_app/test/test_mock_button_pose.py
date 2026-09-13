"""Synthetic optics are explicit and geometrically consistent."""

import math

import pytest

from piper_elevator_app.mock_button_pose import simulated_camera_info


def test_simulated_intrinsics_match_declared_image_and_field_of_view():
    message = simulated_camera_info(848, 480, 1.518436)
    assert (message.width, message.height) == (848, 480)
    assert message.k[2] == 423.5
    assert message.k[5] == 239.5
    assert message.k[0] == message.k[4] > 0.0
    assert 2.0 * math.atan(message.width / (2.0 * message.k[0])) == (
        pytest.approx(1.518436)
    )
    assert message.p[0] == message.k[0]
    assert message.p[2] == message.k[2]
    assert message.p[5] == message.k[4]
    assert message.p[6] == message.k[5]
    assert list(message.d) == [0.0] * 5


@pytest.mark.parametrize('width,height,fov', [
    (0, 480, 1.5),
    (848, -1, 1.5),
    (848, 480, 0.0),
    (848, 480, math.pi),
    (848, 480, math.nan),
])
def test_simulated_intrinsics_reject_invalid_geometry(width, height, fov):
    with pytest.raises(ValueError):
        simulated_camera_info(width, height, fov)
