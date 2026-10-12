import dataclasses

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from rotations import random_rotation

from spatial_change.domain import SE3, Epoch, Frame, Intrinsics
from spatial_change.geometry import (
    VisiblePoints,
    depth_residual,
    in_image,
    project,
    unproject_depth,
    usable_pixels,
    visible_points,
)

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


def _wall_frame(
    depth_value: float = 2.0,
    confidence: np.ndarray | None = None,
    depth: np.ndarray | None = None,
) -> Frame:
    if depth is None:
        depth = np.full((24, 32), depth_value, np.float32)
    return Frame(
        epoch=Epoch.BEFORE,
        index=0,
        timestamp_s=0.0,
        intrinsics=_INTRINSICS,
        world_from_camera=SE3.identity(),
        depth=depth,
        confidence=confidence,
    )


def test_usable_pixels_without_confidence_need_only_a_valid_depth() -> None:
    depth = np.full((24, 32), 2.0, np.float32)
    depth[0, 0] = 0.0
    depth[0, 1] = np.nan
    depth[0, 2] = -1.0

    usable = usable_pixels(_wall_frame(depth=depth), 0.9)

    assert usable.dtype == np.bool_
    assert not usable[0, :3].any()
    assert usable.sum() == 24 * 32 - 3


def test_usable_pixels_apply_the_confidence_threshold_inclusively() -> None:
    confidence = np.full((24, 32), 0.5, np.float32)
    confidence[1, 1] = 0.49
    confidence[2, 2] = 0.51

    usable = usable_pixels(_wall_frame(confidence=confidence), 0.5)

    assert not usable[1, 1]
    assert usable[2, 2] and usable[0, 0]


def test_usable_pixels_need_a_depth_map() -> None:
    frame = dataclasses.replace(_wall_frame(), depth=None)

    with pytest.raises(ValueError, match="depth"):
        usable_pixels(frame, 0.1)


def _visible(
    points: list[list[float]],
    frame: Frame | None = None,
    abs_tolerance: float = 0.05,
    rel_tolerance: float = 0.02,
) -> VisiblePoints:
    frame = frame or _wall_frame()
    usable = usable_pixels(frame, 0.1)
    return visible_points(
        np.array(points, np.float64).reshape(-1, 3),
        frame,
        usable,
        abs_tolerance,
        rel_tolerance,
    )


def test_a_point_on_the_depth_surface_is_visible_with_its_pixel() -> None:
    result = _visible([[0.0, 0.0, 2.0], [0.5, -0.25, 2.0]])

    np.testing.assert_array_equal(result.indices, [0, 1])
    np.testing.assert_allclose(
        result.uv, [[15.5, 11.5], [15.5 + 40.0 * 0.25, 11.5 - 45.0 * 0.125]]
    )
    np.testing.assert_array_equal(result.pixels, [[16, 12], [26, 6]])
    assert result.pixels.dtype == np.intp


def test_points_off_the_surface_are_not_visible() -> None:
    inside = [0.0, 0.0, 2.0]
    behind_the_wall = [0.0, 0.0, 3.0]
    in_front_of_the_wall = [0.0, 0.0, 1.0]

    result = _visible([behind_the_wall, inside, in_front_of_the_wall])

    np.testing.assert_array_equal(result.indices, [1])


def test_points_behind_the_camera_or_outside_the_image_are_not_visible() -> (
    None
):
    result = _visible([[0.0, 0.0, -2.0], [9.0, 0.0, 2.0], [0.0, 0.0, 2.0]])

    np.testing.assert_array_equal(result.indices, [2])


def test_the_depth_tolerance_is_absolute_plus_relative() -> None:
    # |z - 2| <= 0.125 + 0.5 * z holds up to z = 4.25.
    just_in = [0.0, 0.0, 4.25 - 1e-3]
    just_out = [0.0, 0.0, 4.25 + 1e-3]

    result = _visible(
        [just_out, just_in], abs_tolerance=0.125, rel_tolerance=0.5
    )

    np.testing.assert_array_equal(result.indices, [1])


def test_points_on_unusable_pixels_are_not_visible() -> None:
    confidence = np.ones((24, 32), np.float32)
    confidence[12, 16] = 0.0
    frame = _wall_frame(confidence=confidence)

    result = _visible([[0.0, 0.0, 2.0], [0.5, 0.0, 2.0]], frame)

    np.testing.assert_array_equal(result.indices, [1])


def test_no_points_give_empty_results() -> None:
    result = _visible([])

    assert result.indices.shape == (0,)
    assert result.uv.shape == (0, 2)
    assert result.pixels.shape == (0, 2)


def test_depth_residual_is_the_signed_difference_to_the_observed_depth() -> (
    None
):
    frame = _wall_frame()
    usable = usable_pixels(frame, 0.1)
    points = np.array(
        [
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0],
            [9.0, 0.0, 2.0],
            [0.0, 0.0, -2.0],
        ],
        np.float64,
    )

    result = depth_residual(points, frame, usable)

    np.testing.assert_array_equal(result.indices, [0, 1, 2])
    np.testing.assert_allclose(result.z, [3.0, 2.0, 1.0])
    np.testing.assert_allclose(result.residual, [1.0, 0.0, -1.0])
    np.testing.assert_array_equal(result.pixels, [[16, 12]] * 3)
    np.testing.assert_allclose(result.uv, [[15.5, 11.5]] * 3)


def test_depth_residual_skips_unusable_pixels() -> None:
    frame = _wall_frame()
    usable = np.zeros_like(usable_pixels(frame, 0.1))

    result = depth_residual(np.array([[0.0, 0.0, 2.0]]), frame, usable)

    assert len(result.indices) == 0
    assert result.residual.shape == (0,)


def test_depth_residual_needs_a_depth_map() -> None:
    frame = dataclasses.replace(_wall_frame(), depth=None)

    with pytest.raises(ValueError, match="depth"):
        depth_residual(np.zeros((1, 3)), frame, np.ones((24, 32), bool))
