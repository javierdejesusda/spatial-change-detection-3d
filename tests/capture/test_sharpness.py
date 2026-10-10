from collections.abc import Callable

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from spatial_change.capture.sharpness import (
    fit_long_side,
    sharp_mask,
    sharpness,
    working_gray,
)

TextureFactory = Callable[..., NDArray[np.uint8]]


def test_blurred_copy_scores_lower(texture: TextureFactory) -> None:
    image = texture(640, 480, seed=7, sigma=1.5)
    blurred = cv2.GaussianBlur(image, (0, 0), 3.0)

    assert sharpness(blurred) < 0.5 * sharpness(image)


def test_score_is_roughly_scale_invariant(texture: TextureFactory) -> None:
    image = texture(640, 480, seed=7, sigma=1.5)
    doubled = cv2.resize(image, (1280, 960), interpolation=cv2.INTER_LINEAR)

    ratio = sharpness(doubled) / sharpness(image)

    assert abs(ratio - 1.0) < 0.25


def test_sharp_mask_rejects_injected_blurred_frames(
    texture: TextureFactory,
) -> None:
    base = texture(320, 240, seed=7, sigma=1.5)
    blurred = cv2.GaussianBlur(base, (0, 0), 3.0)
    blurred_at = {3, 7, 11}
    scores = [
        sharpness(blurred if i in blurred_at else base) for i in range(20)
    ]

    mask = sharp_mask(scores)

    assert mask.dtype == np.bool_
    assert not mask[sorted(blurred_at)].any()
    assert mask.sum() >= 12


def test_sharp_mask_keeps_everything_for_few_frames() -> None:
    assert sharp_mask([5.0, 0.1]).all()
    assert sharp_mask([]).shape == (0,)


def test_sharp_mask_floor_rejects_far_below_median() -> None:
    scores = [100.0] * 9 + [10.0]

    mask = sharp_mask(scores, keep_quantile=0.0, floor_ratio=0.5)

    assert mask.tolist() == [True] * 9 + [False]


def test_gray_input_scores_like_rgb(texture: TextureFactory) -> None:
    rgb = texture(320, 240, seed=7, sigma=1.5)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    assert sharpness(gray) == pytest.approx(sharpness(rgb))


@pytest.mark.parametrize("upscale", [False, True])
def test_fit_long_side_shrinks_to_the_long_side(upscale: bool) -> None:
    gray = np.zeros((100, 200), dtype=np.uint8)

    fitted = fit_long_side(gray, 50, upscale=upscale)

    assert fitted.shape == (25, 50)
    assert fitted.dtype == np.uint8


def test_fit_long_side_enlarges_only_when_asked() -> None:
    gray = np.zeros((100, 200), dtype=np.uint8)

    assert fit_long_side(gray, 400, upscale=True).shape == (200, 400)
    assert fit_long_side(gray, 400, upscale=False).shape == (100, 200)


def test_fit_long_side_keeps_at_least_one_pixel() -> None:
    gray = np.zeros((1, 1000), dtype=np.uint8)

    assert fit_long_side(gray, 10, upscale=False).shape == (1, 10)
    assert fit_long_side(gray.T, 10, upscale=False).shape == (10, 1)


@pytest.mark.parametrize("shape", [(480, 320), (1080, 1920), (100, 100)])
def test_working_gray_has_a_640_long_side(
    texture: TextureFactory, shape: tuple[int, int]
) -> None:
    rgb = texture(shape[1], shape[0], seed=7, sigma=1.5)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    for image in (rgb, gray):
        working = working_gray(image)
        assert working.ndim == 2
        assert working.dtype == np.uint8
        assert max(working.shape) == 640
