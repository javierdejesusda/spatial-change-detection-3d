"""Deterministic stand-ins for the perception ports, for tests."""

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from spatial_change.perception.padding import check_image, pad_to_patches
from spatial_change.perception.prompts import (
    check_labeling_inputs,
    check_prompts,
)


class FakeFeatureEncoder:
    """Encodes each patch as the unit direction of its mean color.

    One is added to the mean RGB before normalizing, so black has a
    defined direction. The cosine of two features therefore compares hue
    and saturation and ignores brightness, which is the invariance the
    appearance cue needs from a real encoder.

    Attributes:
        encode_count: How many times ``encode`` was called.
        unload_count: How many times ``unload`` was called.
    """

    def __init__(self, patch_size: int = 14) -> None:
        """Creates the fake encoder.

        Args:
            patch_size: Patch side in pixels, at least 1.

        Raises:
            ValueError: If ``patch_size`` is below 1.
        """
        if patch_size < 1:
            raise ValueError("patch_size must be >= 1")
        self._patch_size = patch_size
        self.encode_count = 0
        self.unload_count = 0

    @property
    def patch_size(self) -> int:
        """Side length in input pixels of the square patch per feature."""
        return self._patch_size

    def encode(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Encodes an image into a grid of color-direction features.

        Args:
            image: ``(H, W, 3)`` uint8 RGB image.

        Returns:
            ``(ceil(H / p), ceil(W / p), 3)`` float32 unit vectors.

        Raises:
            ValueError: If ``image`` is not a non-empty RGB uint8 array.
        """
        check_image(image)
        self.encode_count += 1
        p = self._patch_size
        padded = pad_to_patches(image, p)
        rows, cols = padded.shape[0] // p, padded.shape[1] // p
        mean = padded.reshape(rows, p, cols, p, 3).mean(axis=(1, 3)) + 1.0
        norm = np.linalg.norm(mean, axis=2, keepdims=True)
        return np.asarray(mean / norm, dtype=np.float32)

    def unload(self) -> None:
        """Counts the call; the fake holds no resources."""
        self.unload_count += 1


class FakeMaskSegmenter:
    """Segments each box as the rectangle of pixels it contains.

    A pixel belongs to the mask of a box when its center, at integer
    coordinates, lies inside the box (inclusive), so the masks include any
    background in the box. Point prompts are validated and recorded but do
    not change the masks.

    Attributes:
        unload_count: How many times ``unload`` was called.
        calls: The ``(image, boxes_xyxy, points_xy)`` of each ``segment``
            call, in order.
    """

    def __init__(self) -> None:
        """Creates the fake segmenter."""
        self.unload_count = 0
        self.calls: list[
            tuple[
                NDArray[np.uint8],
                NDArray[np.float64],
                NDArray[np.float64] | None,
            ]
        ] = []

    def segment(
        self,
        image: NDArray[np.uint8],
        boxes_xyxy: NDArray[np.float64],
        points_xy: NDArray[np.float64] | None,
    ) -> NDArray[np.bool_]:
        """Returns the rectangle of pixel centers inside each box.

        Args:
            image: ``(H, W, 3)`` uint8 RGB image.
            boxes_xyxy: ``(K, 4)`` float64 boxes ``(x0, y0, x1, y1)``.
            points_xy: Optional ``(K, 2)`` float64 points, ignored.

        Returns:
            ``(K, H, W)`` boolean masks in prompt order.

        Raises:
            ValueError: If an input has the wrong shape, dtype, or values.
        """
        check_image(image)
        check_prompts(boxes_xyxy, points_xy)
        self.calls.append((image, boxes_xyxy, points_xy))
        height, width = image.shape[:2]
        columns = np.arange(width)
        rows = np.arange(height)
        x0, y0, x1, y1 = (boxes_xyxy[:, i, None] for i in range(4))
        in_columns = (columns >= x0) & (columns <= x1)
        in_rows = (rows >= y0) & (rows <= y1)
        return np.asarray(
            in_rows[:, :, None] & in_columns[:, None, :], dtype=np.bool_
        )

    def unload(self) -> None:
        """Counts the call; the fake holds no resources."""
        self.unload_count += 1


class FakeLabeler:
    """Labels a crop with the entry whose known hue most of its pixels have.

    Each pixel votes for the vocabulary entry with a known color whose RGB
    direction is nearest to its own, when the cosine of the two is at least
    ``min_cosine``; a black pixel or one that matches no color does not vote.
    Brightness does not matter, so a shaded face keeps the hue of its
    class. The label is the entry with the most votes, the earlier
    vocabulary entry on a tie, and its score is its share of the votes. A
    crop without votes, or a vocabulary without known colors, gives the
    first entry with score 0.

    Attributes:
        unload_count: How many times ``unload`` was called.
    """

    def __init__(
        self,
        colors: Mapping[str, tuple[int, int, int]],
        min_cosine: float = 0.98,
    ) -> None:
        """Creates the fake labeler.

        Args:
            colors: Known RGB color, 0 to 255, by label.
            min_cosine: Smallest cosine between a pixel and a color for the
                pixel to vote, in ``(0, 1]``.

        Raises:
            ValueError: If ``min_cosine`` is outside ``(0, 1]``.
        """
        if not 0.0 < min_cosine <= 1.0:
            raise ValueError("min_cosine must be within (0, 1]")
        self._colors = dict(colors)
        self._min_cosine = min_cosine
        self.unload_count = 0

    def label(
        self, crops: Sequence[NDArray[np.uint8]], vocabulary: Sequence[str]
    ) -> list[tuple[str, float]]:
        """Labels each crop by the hue of its pixels.

        Args:
            crops: ``(H, W, 3)`` uint8 RGB crops.
            vocabulary: Non-empty, unique label names.

        Returns:
            One ``(label, score)`` per crop in order.

        Raises:
            ValueError: If a crop or the vocabulary is invalid.
        """
        check_labeling_inputs(crops, vocabulary)
        known = [name for name in vocabulary if name in self._colors]
        directions = np.array(
            [self._colors[n] for n in known], np.float64
        ).reshape(-1, 3)
        directions /= np.maximum(
            np.linalg.norm(directions, axis=1, keepdims=True), 1e-12
        )
        results = []
        for crop in crops:
            votes = np.zeros(len(known), np.int64)
            if known:
                pixels = crop.reshape(-1, 3).astype(np.float64)
                norms = np.linalg.norm(pixels, axis=1)
                unit = pixels / np.maximum(norms, 1e-12)[:, None]
                cosine = unit @ directions.T
                nearest = np.argmax(cosine, axis=1)
                voting = (norms > 0) & (cosine.max(axis=1) >= self._min_cosine)
                votes = np.bincount(nearest[voting], minlength=len(known))
            if not votes.any():
                results.append((vocabulary[0], 0.0))
                continue
            best = int(np.argmax(votes))
            results.append((known[best], float(votes[best] / votes.sum())))
        return results

    def unload(self) -> None:
        """Counts the call; the fake holds no resources."""
        self.unload_count += 1
