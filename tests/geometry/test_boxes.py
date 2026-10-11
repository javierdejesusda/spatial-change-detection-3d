import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray
from rotations import rotation_z

import spatial_change.geometry as geometry
from spatial_change.domain import OrientedBox
from spatial_change.geometry import fit_upright_box, yaw_rotation


def surface_grid(box: OrientedBox, steps: int = 9) -> NDArray[np.float64]:
    """Samples the six faces of a box on a grid that is symmetric per axis."""
    half = box.extent / 2.0
    axis = np.linspace(-1.0, 1.0, steps)
    faces = []
    for fixed in range(3):
        a, b = [i for i in range(3) if i != fixed]
        for sign in (-1.0, 1.0):
            u, v = np.meshgrid(axis, axis)
            local = np.zeros((u.size, 3))
            local[:, a] = u.ravel() * half[a]
            local[:, b] = v.ravel() * half[b]
            local[:, fixed] = sign * half[fixed]
            faces.append(local)
    return np.concatenate(faces) @ box.rotation.T + box.center


def same_corners(a: OrientedBox, b: OrientedBox, atol: float = 1e-9) -> bool:
    """Tells whether two boxes have the same corners, in any order."""
    gap = np.linalg.norm(
        a.corners()[:, None, :] - b.corners()[None, :, :], axis=2
    )
    return bool(gap.min(axis=1).max() < atol and gap.min(axis=0).max() < atol)


def yawed_box(
    center: tuple[float, float, float],
    extent: tuple[float, float, float],
    yaw_deg: float,
) -> OrientedBox:
    return OrientedBox(np.array(center), np.array(extent), rotation_z(yaw_deg))


def test_package_exports_the_box_fit() -> None:
    assert geometry.__all__ == sorted(geometry.__all__)
    assert {"fit_upright_box", "yaw_rotation"} <= set(geometry.__all__)


def test_a_quarter_turn_about_z_takes_x_to_y() -> None:
    rotation = yaw_rotation(math.pi / 2.0)

    np.testing.assert_allclose(rotation @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(rotation @ [0, 0, 1], [0, 0, 1], atol=1e-12)


@pytest.mark.parametrize(
    ("center", "extent", "yaw_deg"),
    [
        pytest.param((1.0, -2.0, 0.6), (1.2, 0.5, 0.9), yaw, id=f"yaw-{yaw:g}")
        for yaw in (0.0, 33.0, -70.0, 120.0, 179.0)
    ]
    + [
        pytest.param(
            (0.0, 0.5, 1.0), (0.4, 1.5, 0.8), 25.0, id="short-side-first"
        )
    ],
)
def test_recovers_a_yawed_box_from_its_surface(
    center: tuple[float, float, float],
    extent: tuple[float, float, float],
    yaw_deg: float,
) -> None:
    truth = yawed_box(center, extent, yaw_deg)

    box = fit_upright_box(surface_grid(truth), 0.01)

    assert same_corners(box, truth)


@pytest.mark.parametrize(
    ("truth_extent", "rotation"),
    [
        pytest.param((1.0, 0.4, 0.8), np.eye(3), id="x-elongated"),
        pytest.param((0.4, 1.0, 0.8), rotation_z(90.0), id="y-elongated"),
    ],
)
def test_an_axis_aligned_footprint_gets_its_canonical_rotation(
    truth_extent: tuple[float, float, float], rotation: NDArray[np.float64]
) -> None:
    rng = np.random.default_rng(11)
    truth = yawed_box((2.0, 1.0, 0.5), truth_extent, 0.0)
    surface = surface_grid(truth)
    shift = np.array([5.0, -3.0, 1.0])
    shuffled = surface[rng.permutation(len(surface))] + shift

    for cloud, offset in ((surface, 0.0), (shuffled, shift)):
        box = fit_upright_box(cloud, 0.01)

        np.testing.assert_allclose(box.rotation, rotation, atol=1e-12)
        np.testing.assert_allclose(
            box.center, truth.center + offset, atol=1e-12
        )
        np.testing.assert_allclose(box.extent, [1.0, 0.4, 0.8], atol=1e-12)
    assert same_corners(fit_upright_box(surface, 0.01), truth)


def test_a_single_point_gives_a_minimum_box() -> None:
    point = np.array([[1.5, -0.5, 0.25]])

    box = fit_upright_box(point, 0.05)

    np.testing.assert_allclose(box.center, point[0])
    np.testing.assert_allclose(box.extent, [0.05, 0.05, 0.05])
    np.testing.assert_allclose(box.rotation, np.eye(3))
    assert box.contains(point).all()


def test_a_flat_patch_gives_the_minimum_extent_in_height() -> None:
    truth = yawed_box((0.0, 0.0, 0.3), (1.0, 0.5, 0.2), 40.0)
    patch = surface_grid(truth)
    patch = patch[np.isclose(patch[:, 2], 0.4)]

    box = fit_upright_box(patch, 0.05)

    np.testing.assert_allclose(box.extent[2], 0.05)
    np.testing.assert_allclose(box.center[2], 0.4)
    np.testing.assert_allclose(box.extent[:2], [1.0, 0.5], atol=1e-9)
    assert box.contains(patch).all()


def test_collinear_points_follow_the_line() -> None:
    direction = rotation_z(40.0)[:2, 0]
    t = np.linspace(-1.0, 3.0, 25)
    points = np.column_stack(
        [t * direction[0] + 1.0, t * direction[1] - 2.0, np.full_like(t, 0.7)]
    )

    box = fit_upright_box(points, 0.02)

    np.testing.assert_allclose(box.extent, [4.0, 0.02, 0.02], atol=1e-9)
    np.testing.assert_allclose(np.abs(box.rotation[:2, 0]), np.abs(direction))
    assert box.contains(points).all()


def test_an_isotropic_footprint_keeps_the_identity_rotation() -> None:
    square = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
    )

    box = fit_upright_box(square, 0.01)

    np.testing.assert_allclose(box.rotation, np.eye(3))
    np.testing.assert_allclose(box.extent, [1.0, 1.0, 1.0])


def test_every_extent_is_at_least_the_minimum() -> None:
    truth = yawed_box((0.0, 0.0, 0.5), (1.0, 0.01, 0.02), 15.0)

    surface = surface_grid(truth)

    box = fit_upright_box(surface, 0.1)

    np.testing.assert_allclose(box.extent, [1.0, 0.1, 0.1], atol=1e-9)
    assert box.contains(surface).all()


def test_the_rotation_is_upright() -> None:
    rng = np.random.default_rng(3)
    points = rng.normal(size=(40, 3)) * [2.0, 0.5, 1.0]

    box = fit_upright_box(points, 0.05)

    np.testing.assert_allclose(box.rotation[:, 2], [0, 0, 1], atol=1e-12)


@given(st.integers(0, 2**32 - 1), st.integers(1, 60), st.floats(0.0, 1e4))
def test_the_box_contains_every_input_point(
    seed: int, count: int, offset: float
) -> None:
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(count, 3)) * rng.uniform(0.01, 3.0, 3) + offset

    box = fit_upright_box(points, 0.05)

    assert box.contains(points).all()


@pytest.mark.parametrize("seed", range(8))
def test_a_square_footprint_is_fitted_at_its_own_area(seed: int) -> None:
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0.0, 2.0 * math.pi)
    turn = rotation_z(math.degrees(angle))[:2, :2]
    xy = rng.uniform(-0.25, 0.25, size=(2000, 2)) @ turn.T
    points = np.column_stack([xy, rng.uniform(0.0, 0.4, len(xy))])

    box = fit_upright_box(points, 0.01)

    assert box.extent[0] * box.extent[1] <= 0.25 * (1.0 + 1e-12)
    assert box.contains(points).all()


@given(st.integers(0, 2**32 - 1), st.integers(1, 60))
def test_no_bounding_rectangle_of_any_whole_degree_is_smaller(
    seed: int, count: int
) -> None:
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(count, 3)) * rng.uniform(0.01, 3.0, 3)
    min_extent = 1e-9

    box = fit_upright_box(points, min_extent)

    scale = max(1.0, float(np.abs(points).max()))
    for degrees in range(180):
        turn = rotation_z(float(degrees))[:2, :2]
        along = points[:, :2] @ turn[:, 0]
        across = points[:, :2] @ turn[:, 1]
        area = max(np.ptp(along), min_extent) * max(np.ptp(across), min_extent)
        assert box.extent[0] * box.extent[1] <= area * (1.0 + 1e-9) + (
            1e-12 * scale**2
        )


@pytest.mark.parametrize(("yaw_deg", "expected"), [(30.0, 30.0), (60.0, -30.0)])
def test_a_square_footprint_takes_the_yaw_within_45_degrees(
    yaw_deg: float, expected: float
) -> None:
    truth = yawed_box((0.5, -0.5, 0.5), (1.0, 1.0, 1.0), yaw_deg)

    box = fit_upright_box(surface_grid(truth), 0.01)

    np.testing.assert_allclose(box.rotation, rotation_z(expected), atol=1e-9)
    np.testing.assert_allclose(box.extent, [1.0, 1.0, 1.0], atol=1e-9)


def test_the_point_order_does_not_change_the_box() -> None:
    rng = np.random.default_rng(7)
    points = rng.normal(size=(50, 3)) * [1.5, 0.4, 0.8]

    box = fit_upright_box(points, 0.05)
    shuffled = fit_upright_box(points[rng.permutation(50)], 0.05)

    assert same_corners(box, shuffled, atol=1e-9)


@pytest.mark.parametrize(
    "points",
    [
        np.zeros((0, 3)),
        np.zeros((3, 2)),
        np.zeros(3),
        np.array([[0.0, 0.0, math.nan]]),
        np.array([[0.0, math.inf, 0.0]]),
    ],
)
def test_invalid_points_are_rejected(points: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError, match="points"):
        fit_upright_box(points, 0.05)


@pytest.mark.parametrize("min_extent", [0.0, -0.1, math.nan, math.inf])
def test_invalid_minimum_extent_is_rejected(min_extent: float) -> None:
    with pytest.raises(ValueError, match="min_extent must be"):
        fit_upright_box(np.zeros((1, 3)), min_extent)
