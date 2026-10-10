import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from rotations import random_rotation

from spatial_change.domain import SE3, Intrinsics
from spatial_change.geometry import in_image, project, unproject_depth

_INTRINSICS = Intrinsics(
    fx=40.0, fy=45.0, cx=15.5, cy=11.5, width=32, height=24
)


def _random_pose(rng: np.random.Generator) -> SE3:
    return SE3(random_rotation(rng), rng.uniform(-5.0, 5.0, size=3))


@given(st.integers(0, 2**32 - 1))
def test_unproject_then_project_round_trips(seed: int) -> None:
    rng = np.random.default_rng(seed)
    intrinsics = Intrinsics(
        fx=rng.uniform(20.0, 80.0),
        fy=rng.uniform(20.0, 80.0),
        cx=rng.uniform(5.0, 15.0),
        cy=rng.uniform(5.0, 10.0),
        width=int(rng.integers(4, 20)),
        height=int(rng.integers(4, 16)),
    )
    pose = _random_pose(rng)
    depth = rng.uniform(0.1, 10.0, size=(intrinsics.height, intrinsics.width))
    depth = depth.astype(np.float32)

    points = unproject_depth(depth, intrinsics, pose)
    uv, z = project(points, intrinsics, pose)

    rows, cols = np.mgrid[: intrinsics.height, : intrinsics.width]
    np.testing.assert_allclose(uv[:, 0], cols.ravel(), atol=1e-6)
    np.testing.assert_allclose(uv[:, 1], rows.ravel(), atol=1e-6)
    np.testing.assert_allclose(z, depth.ravel(), atol=1e-6)
    assert in_image(uv, z, intrinsics).all()


def test_project_follows_pinhole_convention() -> None:
    # Camera at (0, 0, -2) looking along world +z (identity rotation).
    pose = SE3(np.eye(3), np.array([0.0, 0.0, -2.0]))
    points = np.array([[0.0, 0.0, 0.0], [0.5, -0.25, 2.0]])

    uv, depth = project(points, _INTRINSICS, pose)

    np.testing.assert_allclose(depth, [2.0, 4.0])
    np.testing.assert_allclose(uv[0], [15.5, 11.5])
    np.testing.assert_allclose(uv[1], [15.5 + 40.0 * 0.125, 11.5 - 45.0 / 16])


def test_project_applies_camera_rotation() -> None:
    # Camera axes in world: x -> -z, y -> +y, optical axis z -> +x.
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    pose = SE3(rotation, np.zeros(3))
    points = np.array([[2.0, 0.5, 0.25]])

    uv, depth = project(points, _INTRINSICS, pose)

    # Camera-frame point is (-0.25, 0.5, 2.0).
    np.testing.assert_allclose(depth, [2.0])
    np.testing.assert_allclose(uv[0], [15.5 - 40.0 / 8, 11.5 + 45.0 / 4])
    np.testing.assert_allclose(
        unproject_depth(
            np.full((1, 1), 2.0, dtype=np.float32),
            Intrinsics(fx=40.0, fy=45.0, cx=0.0, cy=0.0, width=1, height=1),
            pose,
        ),
        [[2.0, 0.0, 0.0]],
    )


def test_point_behind_camera_is_out() -> None:
    pose = SE3.identity()
    points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]])

    uv, depth = project(points, _INTRINSICS, pose)

    assert depth[1] < 0.0
    np.testing.assert_array_equal(
        in_image(uv, depth, _INTRINSICS), [True, False, False]
    )


@pytest.mark.parametrize(
    ("u", "v", "inside"),
    [
        (-0.5, -0.5, True),
        (31.49, 23.49, True),
        (-0.51, 5.0, False),
        (31.5, 5.0, False),
        (5.0, 23.5, False),
        (np.nan, 5.0, False),
    ],
)
def test_in_image_bounds(u: float, v: float, inside: bool) -> None:
    # Pixel centers are integers, so the image spans [-0.5, W - 0.5).
    mask = in_image(np.array([[u, v]]), np.array([1.0]), _INTRINSICS)

    assert mask.tolist() == [inside]


def test_unproject_skips_invalid_pixels_in_row_major_order() -> None:
    intrinsics = Intrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=3, height=2)
    depth = np.array([[1.0, 0.0, np.nan], [2.0, np.inf, 3.0]], dtype=np.float32)

    points = unproject_depth(depth, intrinsics, SE3.identity())

    np.testing.assert_allclose(
        points, [[0.0, 0.0, 1.0], [0.0, 2.0, 2.0], [6.0, 3.0, 3.0]]
    )


def test_unproject_respects_mask() -> None:
    intrinsics = Intrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=2, height=2)
    depth = np.ones((2, 2), np.float32)
    mask = np.array([[False, True], [True, False]])

    points = unproject_depth(depth, intrinsics, SE3.identity(), mask=mask)

    np.testing.assert_allclose(points, [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])


def test_unproject_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        unproject_depth(
            np.ones((3, 3), np.float32), _INTRINSICS, SE3.identity()
        )
    with pytest.raises(ValueError):
        unproject_depth(
            np.ones((24, 32), np.float32),
            _INTRINSICS,
            SE3.identity(),
            mask=np.ones((2, 2), bool),
        )


def test_unproject_rejects_wrong_dtypes() -> None:
    with pytest.raises(ValueError):
        unproject_depth(
            np.ones((24, 32), np.float64), _INTRINSICS, SE3.identity()
        )
    with pytest.raises(ValueError):
        unproject_depth(
            np.ones((24, 32), np.float32),
            _INTRINSICS,
            SE3.identity(),
            mask=np.ones((24, 32), np.uint8),
        )
