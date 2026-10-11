import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from rotations import random_rotation

from spatial_change.domain import SE3, Epoch, Frame, Intrinsics, Sim3
from spatial_change.geometry.projection import unproject_depth


def _intrinsics(width: int = 64, height: int = 48) -> Intrinsics:
    return Intrinsics(
        fx=50.0, fy=60.0, cx=31.5, cy=23.5, width=width, height=height
    )


def test_matrix_layout() -> None:
    k = Intrinsics(fx=1.0, fy=2.0, cx=3.0, cy=4.0, width=8, height=6).matrix

    np.testing.assert_array_equal(
        k, [[1.0, 0.0, 3.0], [0.0, 2.0, 4.0], [0.0, 0.0, 1.0]]
    )


def test_scaled_maps_pixel_centers() -> None:
    original = _intrinsics(width=640, height=480)
    half = original.scaled(0.5, 0.5)

    assert (half.width, half.height) == (320, 240)
    # Output pixel (0, 0) covers input pixels 0..1, whose centers average to
    # 0.5, so a ray hitting input (0.5, 0.5) must hit output (0, 0).
    ray = np.linalg.solve(original.matrix, [0.5, 0.5, 1.0])
    u, v, w = half.matrix @ ray
    assert (u / w, v / w) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert half.fx == pytest.approx(original.fx * 0.5)
    assert half.cy == pytest.approx(0.5 * (original.cy + 0.5) - 0.5)


def test_scaled_identity_is_unchanged() -> None:
    original = _intrinsics()

    assert original.scaled(1.0, 1.0) == original


@pytest.mark.parametrize(("sx", "sy"), [(0.0, 1.0), (1.0, -0.5), (0.3, 1.0)])
def test_scaled_rejects_bad_factors(sx: float, sy: float) -> None:
    # 64 * 0.3 = 19.2 is not a whole pixel count.
    with pytest.raises(ValueError):
        _intrinsics().scaled(sx, sy)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fx": 0.0},
        {"fy": -1.0},
        {"width": 0},
        {"height": -2},
        {"cx": float("nan")},
    ],
)
def test_invalid_intrinsics_raise(kwargs: dict[str, float]) -> None:
    params: dict[str, float] = {
        "fx": 1.0,
        "fy": 1.0,
        "cx": 0.0,
        "cy": 0.0,
        "width": 4,
        "height": 3,
    }
    params.update(kwargs)
    with pytest.raises(ValueError):
        Intrinsics(**params)  # type: ignore[arg-type]


def test_epoch_values() -> None:
    assert Epoch("before") is Epoch.BEFORE
    assert Epoch.AFTER == "after"


def test_frame_accepts_matching_arrays() -> None:
    intrinsics = _intrinsics()
    frame = Frame(
        epoch=Epoch.BEFORE,
        index=3,
        timestamp_s=0.1,
        intrinsics=intrinsics,
        world_from_camera=SE3.identity(),
        image=np.zeros((48, 64, 3), np.uint8),
        depth=np.ones((48, 64), np.float32),
        confidence=np.ones((48, 64), np.float32),
    )

    assert frame.image is not None
    assert frame.image.shape == (48, 64, 3)


@pytest.mark.parametrize(
    "arrays",
    [
        {"image": np.zeros((64, 48, 3), np.uint8)},
        {"image": np.zeros((48, 64), np.uint8)},
        {"image": np.zeros((48, 64, 3), np.float32)},
        {"depth": np.ones((48, 63), np.float32)},
        {"depth": np.ones((48, 64), np.float64)},
        {"confidence": np.ones((47, 64), np.float32)},
    ],
    ids=[
        "image-transposed",
        "image-gray",
        "image-dtype",
        "depth-shape",
        "depth-dtype",
        "confidence-shape",
    ],
)
def test_frame_shape_mismatch_raises(arrays: dict[str, np.ndarray]) -> None:
    with pytest.raises(ValueError):
        Frame(
            epoch=Epoch.AFTER,
            index=0,
            timestamp_s=0.0,
            intrinsics=_intrinsics(),
            world_from_camera=SE3.identity(),
            **arrays,
        )


def _random_sim3(rng: np.random.Generator, scale: float) -> Sim3:
    return Sim3(scale, random_rotation(rng), rng.normal(scale=2.0, size=3))


def test_transformed_moves_pose_and_scales_depth() -> None:
    rng = np.random.default_rng(0)
    intrinsics = Intrinsics(fx=10.0, fy=11.0, cx=3.5, cy=2.5, width=8, height=6)
    depth = rng.uniform(0.5, 4.0, size=(6, 8)).astype(np.float32)
    frame = Frame(
        epoch=Epoch.AFTER,
        index=7,
        timestamp_s=0.7,
        intrinsics=intrinsics,
        world_from_camera=SE3(random_rotation(rng), rng.normal(size=3)),
        image=np.zeros((6, 8, 3), np.uint8),
        depth=depth,
        confidence=np.ones((6, 8), np.float32),
    )
    sim = _random_sim3(rng, 2.5)

    moved = frame.transformed(sim)

    assert moved.depth is not None
    np.testing.assert_allclose(moved.depth, depth * 2.5, rtol=1e-6)
    assert (moved.epoch, moved.index, moved.timestamp_s) == (
        Epoch.AFTER,
        7,
        0.7,
    )
    assert moved.intrinsics is intrinsics
    assert moved.image is frame.image
    assert moved.confidence is frame.confidence
    np.testing.assert_allclose(
        moved.world_from_camera.matrix,
        sim.transform_pose(frame.world_from_camera).matrix,
    )
    np.testing.assert_array_equal(frame.depth, depth)


def test_transformed_without_depth_only_moves_the_pose() -> None:
    frame = Frame(
        Epoch.BEFORE,
        0,
        0.0,
        Intrinsics(5.0, 5.0, 1.0, 1.0, 3, 3),
        SE3.identity(),
    )

    moved = frame.transformed(Sim3(2.0, np.eye(3), np.ones(3)))

    assert moved.depth is None
    np.testing.assert_allclose(moved.world_from_camera.translation, np.ones(3))


@given(
    seed=st.integers(0, 2**32 - 1),
    scale=st.floats(0.1, 10.0),
)
def test_transformed_unprojection_equals_transformed_points(
    seed: int, scale: float
) -> None:
    rng = np.random.default_rng(seed)
    intrinsics = Intrinsics(fx=9.0, fy=8.0, cx=2.0, cy=1.5, width=5, height=4)
    frame = Frame(
        Epoch.BEFORE,
        0,
        0.0,
        intrinsics,
        SE3(random_rotation(rng), rng.normal(size=3)),
        depth=rng.uniform(0.2, 5.0, size=(4, 5)).astype(np.float32),
    )
    sim = _random_sim3(rng, scale)
    assert frame.depth is not None

    moved = frame.transformed(sim)

    assert moved.depth is not None
    expected = sim.apply(
        unproject_depth(frame.depth, intrinsics, frame.world_from_camera)
    )
    actual = unproject_depth(moved.depth, intrinsics, moved.world_from_camera)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
