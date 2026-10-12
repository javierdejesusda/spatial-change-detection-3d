from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from spatial_change.perception.padding import check_image
from spatial_change.perception.prompts import check_prompts


def _boxes(count: int = 1) -> NDArray[np.float64]:
    return np.tile(np.array([0.0, 0.0, 4.0, 4.0]), (count, 1))


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((8, 8), np.uint8),
        np.zeros((8, 8, 4), np.uint8),
        np.zeros((8, 8, 3), np.float32),
        np.zeros((0, 8, 3), np.uint8),
    ],
)
def test_invalid_images_are_rejected(image: NDArray[Any]) -> None:
    with pytest.raises(ValueError, match="image"):
        check_image(image)


def test_a_valid_image_is_accepted() -> None:
    check_image(np.zeros((8, 6, 3), np.uint8))


@pytest.mark.parametrize(
    "boxes",
    [
        np.zeros((2, 3), np.float64),
        np.zeros((2, 4), np.float32),
        np.array([[0.0, 0.0, np.nan, 4.0]]),
        np.array([[0.0, 0.0, np.inf, 4.0]]),
        np.array([[5.0, 0.0, 4.0, 4.0]]),
        np.array([[0.0, 5.0, 4.0, 4.0]]),
    ],
)
def test_invalid_boxes_are_rejected(boxes: NDArray[Any]) -> None:
    with pytest.raises(ValueError, match="boxes"):
        check_prompts(boxes, None)


@pytest.mark.parametrize(
    "points",
    [
        np.zeros((1, 2), np.float64),
        np.zeros((2, 3), np.float64),
        np.zeros((2, 2), np.float32),
        np.array([[1.0, np.nan], [1.0, 1.0]]),
    ],
)
def test_invalid_points_are_rejected(points: NDArray[Any]) -> None:
    with pytest.raises(ValueError, match="points"):
        check_prompts(_boxes(2), points)


@pytest.mark.parametrize("count", [0, 1, 3])
def test_valid_prompts_are_accepted(count: int) -> None:
    check_prompts(_boxes(count), None)
    check_prompts(_boxes(count), np.zeros((count, 2)))
