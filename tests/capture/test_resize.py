import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from spatial_change.capture.resize import ResizePolicy, resize_for_backbone
from spatial_change.domain import Intrinsics

POLICY = ResizePolicy(long_side=518, multiple_of=14)


def _image(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


@given(
    long=st.integers(min_value=64, max_value=4000),
    ratio=st.floats(min_value=0.4, max_value=1.0),
    portrait=st.booleans(),
)
def test_shape_constraints(long: int, ratio: float, portrait: bool) -> None:
    short = max(16, int(long * ratio))
    height, width = (long, short) if portrait else (short, long)

    resized, sx, sy = resize_for_backbone(_image(height, width), POLICY)

    new_h, new_w = resized.shape[:2]
    assert new_h % 14 == 0
    assert new_w % 14 == 0
    assert min(new_h, new_w) >= 14
    assert max(new_h, new_w) <= 518
    assert resized.dtype == np.uint8
    assert sx == pytest.approx(new_w / width)
    assert sy == pytest.approx(new_h / height)
    assert POLICY.target_size(height, width) == (new_h, new_w)
    new_long, new_short = max(new_h, new_w), min(new_h, new_w)
    assert abs(new_long * short / long - new_short) <= 14


def test_portrait_phone_frame() -> None:
    resized, sx, sy = resize_for_backbone(_image(1920, 1080), POLICY)

    assert resized.shape == (518, 294, 3)
    assert sx == pytest.approx(294 / 1080)
    assert sy == pytest.approx(518 / 1920)


def test_small_image_is_not_upscaled_beyond_rounding() -> None:
    resized, _, _ = resize_for_backbone(_image(100, 60), POLICY)

    assert resized.shape[:2] == (98, 56)


def test_tiny_image_gets_minimum_side() -> None:
    resized, _, _ = resize_for_backbone(_image(5, 40), POLICY)

    assert resized.shape[:2] == (14, 42)


def test_policy_rejects_incompatible_values() -> None:
    with pytest.raises(ValueError):
        ResizePolicy(long_side=500, multiple_of=14)


@pytest.mark.parametrize("long_side", [0, -14])
def test_policy_rejects_non_positive_long_side(long_side: int) -> None:
    with pytest.raises(ValueError):
        ResizePolicy(long_side=long_side, multiple_of=14)


def test_resize_lands_a_point_where_scaled_intrinsics_project_it() -> None:
    height, width = 720, 1280
    intrinsics = Intrinsics(
        fx=1000.0, fy=1000.0, cx=639.5, cy=359.5, width=width, height=height
    )
    ray = np.array([0.1605, -0.1605, 1.0])
    u, v, _ = intrinsics.matrix @ ray
    col, row = round(u), round(v)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[row - 4 : row + 5, col - 4 : col + 5] = 255

    new_h, new_w = POLICY.target_size(height, width)
    resized, _, _ = resize_for_backbone(image, POLICY)
    scaled = intrinsics.scaled(new_w / width, new_h / height)
    expected_u, expected_v, _ = scaled.matrix @ ray

    assert resized.shape[:2] == (new_h, new_w) == (scaled.height, scaled.width)
    weights = resized[..., 0].astype(np.float64)
    rows, cols = np.indices(weights.shape)
    assert (weights * cols).sum() / weights.sum() == pytest.approx(
        expected_u, abs=1.0
    )
    assert (weights * rows).sum() / weights.sum() == pytest.approx(
        expected_v, abs=1.0
    )


def test_target_size_follows_the_policy() -> None:
    assert POLICY.target_size(1920, 1080) == (518, 294)
    assert POLICY.target_size(100, 60) == (98, 56)
    assert POLICY.target_size(5, 40) == (14, 42)
