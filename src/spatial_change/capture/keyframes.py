"""Keyframe selection from decoded video frames."""

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from spatial_change.capture.sharpness import (
    fit_long_side,
    sharp_mask,
    sharpness,
    working_gray,
)
from spatial_change.capture.video import VideoFrame

_MIN_OVERLAP_PIXELS = 4


@dataclass(frozen=True)
class KeyframeConfig:
    """Keyframe selection settings.

    Attributes:
        max_keyframes: Upper bound on the output size, which bounds the
            memory use of the reconstruction backbone.
        min_keyframes: Target lower bound when enough sharp frames exist.
        overlap_threshold: A new keyframe is taken when the overlap with the
            previous keyframe falls below this value.
        thumb_long_side: Long side of the thumbnails used to measure overlap.
        blur_keep_quantile: Frames scoring below this quantile of the video's
            sharpness scores are dropped as blurry.
        blur_floor_ratio: Frames scoring below this fraction of the median
            sharpness score are dropped as blurry.

    Raises:
        ValueError: If the budget or threshold values are inconsistent.
    """

    max_keyframes: int = 48
    min_keyframes: int = 8
    overlap_threshold: float = 0.7
    thumb_long_side: int = 160
    blur_keep_quantile: float = 0.25
    blur_floor_ratio: float = 0.5

    def __post_init__(self) -> None:
        """Validates the configuration."""
        if not 2 <= self.min_keyframes <= self.max_keyframes:
            raise ValueError(
                "require 2 <= min_keyframes <= max_keyframes, got "
                f"{self.min_keyframes} and {self.max_keyframes}"
            )
        if not 0.0 < self.overlap_threshold <= 1.0:
            raise ValueError("overlap_threshold must be in (0, 1]")
        if self.thumb_long_side < 8:
            raise ValueError("thumb_long_side must be >= 8")


def _crop(image: NDArray[np.float64], dx: int, dy: int) -> NDArray[np.float64]:
    """Returns the part of ``image`` that stays in view after shifting."""
    height, width = image.shape
    return image[
        max(0, dy) : height + min(0, dy), max(0, dx) : width + min(0, dx)
    ]


def overlap(a_thumb: NDArray[np.uint8], b_thumb: NDArray[np.uint8]) -> float:
    """Estimates how much two views overlap, from grayscale thumbnails.

    This is a cheap image-space proxy that needs no camera poses. Phase
    correlation (with a Hanning window) gives the 2D shift between the
    thumbnails. The score is the area fraction shared after that shift,
    multiplied by the normalized cross-correlation of the shared parts,
    clipped to [0, 1]. It assumes mostly translational motion, so rotation
    or strong parallax lowers the score, which errs toward more keyframes.
    Shifts beyond half the thumbnail size alias through the FFT, so call it
    on temporally close frames, as the greedy pass does.

    Args:
        a_thumb: First grayscale thumbnail of shape (H, W).
        b_thumb: Second grayscale thumbnail with the same shape.

    Returns:
        The overlap estimate in [0, 1].

    Raises:
        ValueError: If the thumbnails have different shapes.
    """
    if a_thumb.shape != b_thumb.shape:
        raise ValueError(f"shape mismatch: {a_thumb.shape} vs {b_thumb.shape}")
    height, width = a_thumb.shape
    a = a_thumb.astype(np.float64)
    b = b_thumb.astype(np.float64)
    window = cv2.createHanningWindow((width, height), cv2.CV_64F)
    # phaseCorrelate applies the window to its inputs in place.
    (shift_x, shift_y), _ = cv2.phaseCorrelate(a.copy(), b.copy(), window)
    dx, dy = round(shift_x), round(shift_y)
    area = max(0.0, 1.0 - abs(dx) / width) * max(0.0, 1.0 - abs(dy) / height)
    if area == 0.0:
        return 0.0
    # phaseCorrelate reports b as a displaced by (dx, dy), so a's top-left
    # part matches b's bottom-right part for a positive shift.
    a_part = _crop(a, -dx, -dy)
    b_part = _crop(b, dx, dy)
    if a_part.size < _MIN_OVERLAP_PIXELS:
        return 0.0
    a_zero = a_part - a_part.mean()
    b_zero = b_part - b_part.mean()
    denominator = float(np.sqrt((a_zero**2).sum() * (b_zero**2).sum()))
    if denominator < 1e-6:
        return area if np.array_equal(a_part, b_part) else 0.0
    ncc = float((a_zero * b_zero).sum()) / denominator
    return float(area * np.clip(ncc, 0.0, 1.0))


def _uniform(items: list[int], count: int) -> list[int]:
    """Picks ``count`` items spread evenly, keeping the first and last."""
    positions = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[p] for p in positions]


def select_keyframes(
    frames: Iterable[VideoFrame], config: KeyframeConfig
) -> list[int]:
    """Selects sharp, well-spread keyframes.

    Frames are consumed in a single pass and only a small grayscale
    thumbnail and a sharpness score are kept for each one, so a decoding
    iterator such as ``read_frames(path)`` can be passed directly without
    holding the full-resolution frames in memory.

    Blurry frames are dropped first. A greedy pass then takes the first sharp
    frame and every frame whose overlap with the last keyframe falls below
    ``config.overlap_threshold``. The last sharp frame is always included.
    The result is thinned uniformly to ``config.max_keyframes``, or topped up
    with the sharp frames farthest from any keyframe, one at a time, to
    ``config.min_keyframes``.

    Args:
        frames: Decoded frames in temporal order.
        config: Selection settings.

    Returns:
        Sorted positions of the keyframes in the order ``frames`` yielded
        them. Empty if no frame is sharp.
    """
    scores: list[float] = []
    thumbs: list[NDArray[np.uint8]] = []
    for frame in frames:
        gray = working_gray(frame.image)
        scores.append(sharpness(gray))
        thumbs.append(
            fit_long_side(gray, config.thumb_long_side, upscale=False)
        )
    mask = sharp_mask(
        scores, config.blur_keep_quantile, config.blur_floor_ratio
    )
    sharp = np.flatnonzero(mask).tolist()
    if len(sharp) < 2:
        return sharp

    selected = [sharp[0]]
    for i in sharp[1:]:
        if overlap(thumbs[selected[-1]], thumbs[i]) < config.overlap_threshold:
            selected.append(i)
    if selected[-1] != sharp[-1]:
        selected.append(sharp[-1])

    if len(selected) > config.max_keyframes:
        selected = _uniform(selected, config.max_keyframes)
    target = min(config.min_keyframes, len(sharp))
    candidates = [i for i in sharp if i not in selected]
    while len(selected) < target:
        chosen = np.asarray(selected)
        farthest = max(candidates, key=lambda i: np.abs(chosen - i).min())
        candidates.remove(farthest)
        selected = sorted([*selected, farthest])
    return selected
