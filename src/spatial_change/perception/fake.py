"""A deterministic stand-in for a ``FeatureEncoder``, for tests."""

import numpy as np
from numpy.typing import NDArray

from spatial_change.perception.padding import check_image, pad_to_patches
from spatial_change.perception.prompts import check_prompts


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
