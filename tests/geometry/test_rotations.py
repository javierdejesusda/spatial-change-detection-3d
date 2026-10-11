import numpy as np
import pytest
from rotations import random_rotation, rotation_angle_deg, rotation_z


@pytest.mark.parametrize("deg", [0.0, 30.0, 90.0, 179.0])
def test_angle_of_a_known_rotation(deg: float) -> None:
    assert rotation_angle_deg(np.eye(3), rotation_z(deg)) == pytest.approx(deg)


def test_angle_is_symmetric_and_zero_for_equal_rotations() -> None:
    rng = np.random.default_rng(0)
    a, b = random_rotation(rng), random_rotation(rng)

    assert rotation_angle_deg(a, a) == pytest.approx(0.0, abs=1e-3)
    assert rotation_angle_deg(a, b) == pytest.approx(rotation_angle_deg(b, a))
