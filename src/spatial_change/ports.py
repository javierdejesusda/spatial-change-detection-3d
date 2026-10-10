"""Interfaces for the learned models the pipeline depends on.

Core logic talks to models only through these protocols, so adapters for
specific networks stay swappable and tests can use small fakes. Every port
extends ``Unloadable`` because models run one at a time on a GPU with about
8 GB of memory: an adapter loads its weights lazily on first use and frees
them when unloaded, and the pipeline unloads each model before the next one
runs.

Conventions follow ``spatial_change.domain``: images are ``(H, W, 3)``
uint8 RGB, pixel coordinates use the OpenCV convention with pixel centers
at integer coordinates, and poses are ``world_from_camera``.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain import SE3, Intrinsics


@dataclass(frozen=True, eq=False)
class ReconstructionResult:
    """Per-view geometry predicted by a ``Reconstructor``.

    Entry ``i`` of every list describes input view ``i``. Depth and
    confidence maps may be at the model's working resolution rather than
    the input resolution; ``intrinsics[i]`` always matches ``depth[i]``.

    Attributes:
        intrinsics: Intrinsics of each view at its depth map resolution.
        world_from_camera: Camera-to-world pose of each view.
        depth: ``(h, w)`` float32 depth along the camera z axis per view;
            0 or non-finite means invalid.
        confidence: ``(h, w)`` float32 confidence per view, higher is more
            reliable.
    """

    intrinsics: list[Intrinsics]
    world_from_camera: list[SE3]
    depth: list[NDArray[np.float32]]
    confidence: list[NDArray[np.float32]]

    def __post_init__(self) -> None:
        """Checks that all lists describe the same views.

        Raises:
            ValueError: If the lists have different lengths.
        """
        lengths = {
            len(self.intrinsics),
            len(self.world_from_camera),
            len(self.depth),
            len(self.confidence),
        }
        if len(lengths) != 1:
            raise ValueError("all per-view lists must have the same length")


@runtime_checkable
class Unloadable(Protocol):
    """A model adapter that can free the memory its weights occupy."""

    def unload(self) -> None:
        """Releases model weights and GPU memory."""
        ...


@runtime_checkable
class Reconstructor(Unloadable, Protocol):
    """Predicts camera poses, intrinsics, and depth from a set of images."""

    def reconstruct(
        self, images: Sequence[NDArray[np.uint8]]
    ) -> ReconstructionResult:
        """Reconstructs a set of views of one scene.

        Args:
            images: ``(H, W, 3)`` uint8 RGB images of the same scene.

        Returns:
            One entry per image, in input order. All poses share one world
            frame; its origin, orientation, and scale are model-defined
            (metric only if the model predicts metric depth).
        """
        ...


@runtime_checkable
class MaskSegmenter(Unloadable, Protocol):
    """Segments objects in an image from box (and optional point) prompts."""

    def segment(
        self,
        image: NDArray[np.uint8],
        boxes_xyxy: NDArray[np.float64],
        points_xy: NDArray[np.float64] | None,
    ) -> NDArray[np.bool_]:
        """Predicts one mask per box prompt.

        Args:
            image: ``(H, W, 3)`` uint8 RGB image.
            boxes_xyxy: ``(K, 4)`` boxes as ``(x0, y0, x1, y1)`` in pixels.
            points_xy: Optional ``(K, 2)`` positive point prompts in pixels,
                one per box, as ``(x, y)``.

        Returns:
            ``(K, H, W)`` boolean masks, one per box, in prompt order.
        """
        ...


@runtime_checkable
class FeatureEncoder(Unloadable, Protocol):
    """Computes dense patch features for appearance comparison."""

    @property
    def patch_size(self) -> int:
        """Side length in input pixels of the square patch per feature."""
        ...

    def encode(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Encodes an image into a grid of patch features.

        Args:
            image: ``(H, W, 3)`` uint8 RGB image.

        Returns:
            ``(h, w, C)`` float32 features, where feature ``(i, j)`` covers
            the input patch starting at pixel row ``i * patch_size`` and
            column ``j * patch_size``.
        """
        ...


@runtime_checkable
class Labeler(Unloadable, Protocol):
    """Assigns open-vocabulary labels to image crops."""

    def label(
        self,
        crops: Sequence[NDArray[np.uint8]],
        vocabulary: Sequence[str],
    ) -> list[tuple[str, float]]:
        """Picks the best vocabulary entry for each crop.

        Args:
            crops: ``(h, w, 3)`` uint8 RGB crops, each of any size.
            vocabulary: Candidate class names.

        Returns:
            One ``(label, score)`` per crop, in input order, where ``label``
            is an entry of ``vocabulary`` and ``score`` is in [0, 1].
        """
        ...
