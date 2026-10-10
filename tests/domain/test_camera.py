import numpy as np
import pytest

from spatial_change.domain import SE3, Epoch, Frame, Intrinsics


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
