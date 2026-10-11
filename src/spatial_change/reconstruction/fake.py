"""A deterministic stand-in for a ``Reconstructor``, for tests."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain import SE3, Intrinsics
from spatial_change.ports import ReconstructionResult


class FakeReconstructor:
    """Reconstructor with known geometry that records how it is used.

    View ``i`` of every call sits at ``(i * baseline_m, 0, 0)`` with the
    identity rotation and sees a fronto-parallel plane at ``depth_m`` with
    confidence 1. The intrinsics are isotropic with the focal length equal
    to the depth map width and the principal point at the center.

    Attributes:
        calls: The images passed to each ``reconstruct`` call, in order.
        unload_count: How many times ``unload`` was called.
    """

    def __init__(
        self,
        depth_size: tuple[int, int] | None = None,
        depth_m: float = 2.0,
        baseline_m: float = 0.1,
    ) -> None:
        """Configures the geometry of the fake reconstruction.

        Args:
            depth_size: Depth map ``(width, height)``; None keeps the size
                of each input image.
            depth_m: Depth of every pixel in meters.
            baseline_m: Distance between consecutive views in meters.
        """
        self._depth_size = depth_size
        self._depth_m = depth_m
        self._baseline_m = baseline_m
        self.calls: list[list[NDArray[np.uint8]]] = []
        self.unload_count = 0

    def reconstruct(
        self, images: Sequence[NDArray[np.uint8]]
    ) -> ReconstructionResult:
        """Returns the known geometry for ``images``.

        Args:
            images: ``(H, W, 3)`` uint8 RGB images.

        Returns:
            One view per image, in input order.

        Raises:
            ValueError: If ``images`` is empty.
        """
        if not images:
            raise ValueError("need at least one image")
        self.calls.append(list(images))
        intrinsics, poses, depth, confidence = [], [], [], []
        for i, image in enumerate(images):
            width, height = self._depth_size or (
                image.shape[1],
                image.shape[0],
            )
            intrinsics.append(
                Intrinsics(
                    fx=float(width),
                    fy=float(width),
                    cx=(width - 1) / 2,
                    cy=(height - 1) / 2,
                    width=width,
                    height=height,
                )
            )
            poses.append(
                SE3(np.eye(3), np.array([i * self._baseline_m, 0.0, 0.0]))
            )
            depth.append(np.full((height, width), self._depth_m, np.float32))
            confidence.append(np.ones((height, width), np.float32))
        return ReconstructionResult(intrinsics, poses, depth, confidence)

    def unload(self) -> None:
        """Counts the call; the fake holds no resources."""
        self.unload_count += 1
