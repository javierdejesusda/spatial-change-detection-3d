"""Blur scoring that is comparable across input resolutions."""

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

_WORKING_LONG_SIDE = 640


def fit_long_side(
    gray: NDArray[np.uint8], long_side: int, *, upscale: bool
) -> NDArray[np.uint8]:
    """Resizes a grayscale image so its long side is ``long_side`` pixels.

    The aspect ratio is kept and each side is at least one pixel. Shrinking
    uses area averaging and enlarging uses bilinear interpolation.

    Args:
        gray: Grayscale image of shape (H, W), dtype uint8.
        long_side: Target length of the longer side, in pixels.
        upscale: Whether to enlarge images whose long side is shorter than
            ``long_side``. If False, such images are returned unchanged.

    Returns:
        The resized image, or ``gray`` itself when no resize is needed.
    """
    height, width = gray.shape
    scale = long_side / max(height, width)
    if scale == 1.0 or (scale > 1.0 and not upscale):
        return gray
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        gray,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=interpolation,
    )
    return np.asarray(resized, dtype=np.uint8)


def working_gray(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Converts an image to the grayscale used to score and compare frames.

    Args:
        image: RGB image of shape (H, W, 3) or grayscale image of shape
            (H, W), dtype uint8.

    Returns:
        A grayscale image of shape (H, W) whose long side is 640 px.
    """
    gray = image
    if image.ndim == 3:
        gray = np.asarray(
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), dtype=np.uint8
        )
    return fit_long_side(gray, _WORKING_LONG_SIDE, upscale=True)


def sharpness(image: NDArray[np.uint8]) -> float:
    """Scores focus as the variance of the Laplacian of the gray image.

    The image is first resized so its long side is 640 px, which keeps scores
    comparable between 720p, 1080p and low-resolution inputs.

    Args:
        image: RGB image of shape (H, W, 3) or grayscale image of shape
            (H, W), dtype uint8.

    Returns:
        The variance of the Laplacian. Higher means sharper.
    """
    return float(cv2.Laplacian(working_gray(image), cv2.CV_64F).var())


def sharp_mask(
    scores: Sequence[float],
    keep_quantile: float = 0.25,
    floor_ratio: float = 0.5,
) -> NDArray[np.bool_]:
    """Flags frames that are sharp relative to the video's own scores.

    Phones differ too much for an absolute threshold, so a frame is rejected
    when it falls below the ``keep_quantile`` quantile of the scores, or below
    ``floor_ratio`` times the median score.

    Args:
        scores: One sharpness score per frame.
        keep_quantile: Quantile of the scores used as the lower cut.
        floor_ratio: Fraction of the median score used as a second cut.

    Returns:
        A boolean array, True for frames to keep. With fewer than 3 frames,
        all frames are kept.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.size < 3:
        return np.ones(values.shape, dtype=np.bool_)
    threshold = max(
        float(np.quantile(values, keep_quantile)),
        floor_ratio * float(np.median(values)),
    )
    return values >= threshold
