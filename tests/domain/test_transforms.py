import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray
from rotations import random_rotation, rotation_z

from spatial_change.domain import SE3, Sim3
from spatial_change.domain.transforms import (
    as_finite_points,
    check_array,
    rotation_to_quaternion,
)

_seeded_rotations = st.integers(0, 2**32 - 1).map(
    lambda seed: random_rotation(np.random.default_rng(seed))
)
_cardinal_rotations = st.sampled_from([0.0, 90.0, 180.0, 270.0]).map(rotation_z)
rotations = st.one_of(_seeded_rotations, _cardinal_rotations)
translations = st.lists(
    st.floats(-10.0, 10.0, allow_nan=False), min_size=3, max_size=3
).map(np.array)
scales = st.floats(0.1, 10.0, allow_nan=False)
se3s = st.builds(SE3, rotations, translations)
sim3s = st.builds(Sim3, scales, rotations, translations)
points = st.integers(0, 2**32 - 1).map(
    lambda seed: np.random.default_rng(seed).normal(size=(7, 3))
)


@given(se3s)
def test_se3_inverse_compose_is_identity(t: SE3) -> None:
    np.testing.assert_allclose(
        t.inverse().compose(t).matrix, np.eye(4), atol=1e-9
    )


@given(sim3s)
def test_sim3_inverse_compose_is_identity(t: Sim3) -> None:
    np.testing.assert_allclose(
        t.inverse().compose(t).matrix, np.eye(4), atol=1e-9
    )


@given(se3s, se3s, se3s)
def test_se3_compose_is_associative(a: SE3, b: SE3, c: SE3) -> None:
    np.testing.assert_allclose(
        a.compose(b).compose(c).matrix,
        a.compose(b.compose(c)).matrix,
        atol=1e-9,
    )


@given(sim3s, sim3s, sim3s)
def test_sim3_compose_is_associative(a: Sim3, b: Sim3, c: Sim3) -> None:
    np.testing.assert_allclose(
        a.compose(b).compose(c).matrix,
        a.compose(b.compose(c)).matrix,
        rtol=1e-9,
        atol=1e-9,
    )


@given(se3s, se3s, points)
def test_se3_compose_applies_other_first(
    a: SE3, b: SE3, p: NDArray[np.float64]
) -> None:
    np.testing.assert_allclose(
        a.compose(b).apply(p), a.apply(b.apply(p)), atol=1e-9
    )


@given(sim3s, sim3s, points)
def test_sim3_compose_applies_other_first(
    a: Sim3, b: Sim3, p: NDArray[np.float64]
) -> None:
    np.testing.assert_allclose(
        a.compose(b).apply(p), a.apply(b.apply(p)), rtol=1e-9, atol=1e-8
    )


@given(se3s, points)
def test_se3_apply_agrees_with_matrix(t: SE3, p: NDArray[np.float64]) -> None:
    homogeneous = np.hstack([p, np.ones((len(p), 1))])
    expected = (t.matrix @ homogeneous.T).T[:, :3]
    np.testing.assert_allclose(t.apply(p), expected, atol=1e-9)


@given(se3s, points)
def test_se3_apply_inverse_undoes_apply(t: SE3, p: NDArray[np.float64]) -> None:
    np.testing.assert_allclose(t.apply_inverse(t.apply(p)), p, atol=1e-9)
    np.testing.assert_allclose(
        t.apply_inverse(p), t.inverse().apply(p), atol=1e-9
    )


@given(sim3s, points)
def test_sim3_apply_agrees_with_matrix(t: Sim3, p: NDArray[np.float64]) -> None:
    homogeneous = np.hstack([p, np.ones((len(p), 1))])
    expected = (t.matrix @ homogeneous.T).T[:, :3]
    np.testing.assert_allclose(t.apply(p), expected, atol=1e-9)


@given(se3s)
def test_se3_from_matrix_round_trips(t: SE3) -> None:
    back = SE3.from_matrix(t.matrix)
    np.testing.assert_allclose(back.rotation, t.rotation, atol=1e-12)
    np.testing.assert_allclose(back.translation, t.translation, atol=1e-12)


@given(sim3s)
def test_sim3_from_matrix_round_trips(t: Sim3) -> None:
    back = Sim3.from_matrix(t.matrix)
    assert back.scale == pytest.approx(t.scale, rel=1e-12)
    np.testing.assert_allclose(back.rotation, t.rotation, atol=1e-9)
    np.testing.assert_allclose(back.translation, t.translation, atol=1e-12)


def test_identity_leaves_points_unchanged() -> None:
    p = np.arange(6, dtype=np.float64).reshape(2, 3)
    np.testing.assert_array_equal(SE3.identity().apply(p), p)
    np.testing.assert_array_equal(Sim3.identity().apply(p), p)


@pytest.mark.parametrize(
    "rotation",
    [
        np.diag([1.0, 1.0, 1.1]),
        np.diag([1.0 + 4e-6, 1.0 - 4e-6, 1.0]),
        np.diag([1.0, 1.0, -1.0]),
        np.zeros((3, 3)),
        np.eye(4),
    ],
    ids=["scaled", "near-tolerance", "reflection", "zero", "wrong-shape"],
)
def test_invalid_rotation_raises(rotation: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError):
        SE3(rotation, np.zeros(3))
    with pytest.raises(ValueError):
        Sim3(1.0, rotation, np.zeros(3))


def test_wrong_translation_shape_raises() -> None:
    with pytest.raises(ValueError):
        SE3(np.eye(3), np.zeros(4))


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_sim3_scale_raises(scale: float) -> None:
    with pytest.raises(ValueError):
        Sim3(scale, np.eye(3), np.zeros(3))


@given(sim3s)
def test_from_matrix_tolerates_round_off_in_bottom_row(t: Sim3) -> None:
    m = t.matrix
    m[3] += [1e-15, -1e-15, 0.0, 1e-15]

    assert Sim3.from_matrix(m).scale == pytest.approx(t.scale)


def test_from_matrix_rejects_non_affine_bottom_row() -> None:
    m = np.eye(4)
    m[3, 0] = 1.0
    with pytest.raises(ValueError):
        SE3.from_matrix(m)
    with pytest.raises(ValueError):
        Sim3.from_matrix(m)


def test_arrays_are_read_only_copies() -> None:
    rotation = np.eye(3)
    translation = np.zeros(3)
    t = SE3(rotation, translation)
    s = Sim3(2.0, rotation, translation)

    rotation[0, 0] = 5.0
    assert t.rotation[0, 0] == 1.0
    for array in (t.rotation, t.translation, s.rotation, s.translation):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[0] = 1.0


@given(sim3s, se3s)
def test_transform_pose_keeps_rotation_and_maps_camera_center(
    sim: Sim3, pose: SE3
) -> None:
    moved = sim.transform_pose(pose)

    r = moved.rotation
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(
        moved.translation, sim.apply(pose.translation[None])[0], atol=1e-9
    )
    np.testing.assert_allclose(r, sim.rotation @ pose.rotation, atol=1e-9)


@given(sim3s, se3s, points)
def test_transform_pose_maps_camera_points_consistently(
    sim: Sim3, pose: SE3, p: NDArray[np.float64]
) -> None:
    # Camera-frame points scaled into the new world's units must land where
    # the similarity sends their old world positions.
    moved = sim.transform_pose(pose)
    np.testing.assert_allclose(
        moved.apply(sim.scale * p), sim.apply(pose.apply(p)), atol=1e-8
    )


@given(se3s)
def test_sim3_from_se3_has_unit_scale(t: SE3) -> None:
    s = Sim3.from_se3(t)
    assert s.scale == 1.0
    np.testing.assert_allclose(s.matrix, t.matrix, atol=0.0)


@pytest.mark.parametrize(
    ("array", "shape", "dtype"),
    [
        (np.zeros((3, 4), np.float32), (3, 4), np.float32),
        (np.zeros((5, 3)), (None, 3), np.float64),
        (np.zeros(0, np.uint8), (None,), np.uint8),
    ],
    ids=["fixed", "any-rows", "any-length"],
)
def test_check_array_accepts_matching_array(
    array: NDArray[np.generic],
    shape: tuple[int | None, ...],
    dtype: type[np.generic],
) -> None:
    check_array(array, shape, dtype, "thing")


@pytest.mark.parametrize(
    ("array", "shape", "dtype"),
    [
        (np.zeros((3, 4)), (3, 4), np.float32),
        (np.zeros((3, 5), np.float32), (3, 4), np.float32),
        (np.zeros((5, 2)), (None, 3), np.float64),
        (np.zeros((5, 3, 1)), (None, 3), np.float64),
        (np.zeros((5, 3)), (None,), np.float64),
    ],
    ids=["dtype", "fixed-size", "columns", "extra-axis", "missing-axis"],
)
def test_check_array_rejects_mismatch(
    array: NDArray[np.generic],
    shape: tuple[int | None, ...],
    dtype: type[np.generic],
) -> None:
    with pytest.raises(ValueError):
        check_array(array, shape, dtype, "thing")


def test_check_array_message_names_field_expected_and_actual() -> None:
    with pytest.raises(ValueError) as info:
        check_array(
            np.zeros((5, 2), np.float32), (None, 3), np.float64, "points"
        )

    assert str(info.value) == (
        "points must be float64 with shape (N, 3), "
        "got float32 with shape (5, 2)"
    )
    with pytest.raises(ValueError) as info:
        check_array(np.zeros((2, 2)), (None,), np.float64, "confidence")

    assert str(info.value) == (
        "confidence must be float64 with shape (N,), "
        "got float64 with shape (2, 2)"
    )


def test_as_finite_points_returns_a_float64_array() -> None:
    out = as_finite_points([[1, 2, 3]] * 4, "cloud", 3)

    assert out.dtype == np.float64
    assert out.shape == (4, 3)


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (np.zeros((5, 2)), "shape"),
        (np.zeros(9), "shape"),
        (np.zeros((2, 3)), "at least 3"),
        (np.full((5, 3), np.nan), "finite"),
        (np.full((5, 3), np.inf), "finite"),
    ],
)
def test_as_finite_points_rejects_bad_clouds_naming_the_field(
    value: NDArray[np.float64], match: str
) -> None:
    with pytest.raises(ValueError, match=rf"^cloud.*{match}"):
        as_finite_points(value, "cloud", 3)


def test_as_finite_points_honours_min_count() -> None:
    as_finite_points(np.zeros((2, 3)), "cloud", 2)

    with pytest.raises(ValueError, match="at least 2"):
        as_finite_points(np.zeros((1, 3)), "cloud", 2)


def _matrix_of(q: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_the_identity_has_the_unit_quaternion() -> None:
    np.testing.assert_allclose(
        rotation_to_quaternion(np.eye(3)), [0.0, 0.0, 0.0, 1.0]
    )


def test_a_quarter_turn_about_z() -> None:
    root = np.sqrt(0.5)

    np.testing.assert_allclose(
        rotation_to_quaternion(rotation_z(90.0)), [0.0, 0.0, root, root]
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_half_turn_about_each_axis_round_trips(axis: int) -> None:
    rotation = np.diag([-1.0, -1.0, -1.0])
    rotation[axis, axis] = 1.0

    q = rotation_to_quaternion(rotation)

    assert np.linalg.norm(q) == pytest.approx(1.0)
    np.testing.assert_allclose(_matrix_of(q), rotation, atol=1e-12)


@given(rotations)
def test_the_quaternion_is_a_unit_one_that_rebuilds_the_rotation(
    rotation: NDArray[np.float64],
) -> None:
    q = rotation_to_quaternion(rotation)

    assert np.linalg.norm(q) == pytest.approx(1.0)
    np.testing.assert_allclose(_matrix_of(q), rotation, atol=1e-9)
