import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from rotations import perturbation, rotation_z

from spatial_change.geometry import rotation_angle_deg
from spatial_change.geometry.rotation import rotation_angle_rad


def test_the_identity_has_no_rotation() -> None:
    assert rotation_angle_deg(np.eye(3)) == 0.0


@pytest.mark.parametrize("deg", [1.0, 10.0, 90.0, 135.0, 179.0])
def test_measures_a_turn_about_z(deg: float) -> None:
    assert rotation_angle_deg(rotation_z(deg)) == pytest.approx(deg)
    assert rotation_angle_deg(rotation_z(-deg)) == pytest.approx(deg)


def test_a_half_turn_is_180() -> None:
    assert rotation_angle_deg(rotation_z(180.0)) == pytest.approx(180.0)


def test_a_tiny_angle_keeps_its_precision() -> None:
    assert rotation_angle_deg(rotation_z(1e-4)) == pytest.approx(1e-4, rel=1e-6)


@settings(max_examples=100, deadline=None)
@given(
    st.lists(st.floats(-1.0, 1.0), min_size=3, max_size=3).filter(
        lambda v: sum(x * x for x in v) > 1e-3
    ),
    st.floats(0.0, 180.0),
)
def test_recovers_an_axis_angle_rotation(axis: list[float], deg: float) -> None:
    rotation = perturbation(axis, deg, (0.0, 0.0, 0.0)).rotation

    assert rotation_angle_deg(rotation) == pytest.approx(deg, abs=1e-6)


@pytest.mark.parametrize("deg", [0.0, 1.0, 90.0, 179.0, 180.0])
def test_the_unchecked_angle_is_in_radians(deg: float) -> None:
    assert rotation_angle_rad(rotation_z(deg)) == pytest.approx(np.radians(deg))


def test_the_unchecked_angle_lets_a_non_finite_matrix_through() -> None:
    matrix = np.eye(3)
    matrix[0, 0] = np.nan

    assert np.isnan(rotation_angle_rad(matrix))


def test_rejects_a_matrix_that_is_not_3x3() -> None:
    with pytest.raises(ValueError):
        rotation_angle_deg(np.eye(4))
