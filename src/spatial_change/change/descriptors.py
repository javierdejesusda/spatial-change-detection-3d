"""Appearance descriptors of lifted object instances.

An instance is described by the mean of the unit patch features that the
frames of its own epoch see on its surface. Every frame is encoded once for
all the instances, and each occupied voxel of an instance counts once, so a
densely sampled side does not outweigh a sparse one.

Importing this module does not load torch.
"""

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain import EpochReconstruction, ObjectInstance
from spatial_change.geometry import voxel_cells
from spatial_change.geometry.projection import (
    DEFAULT_MIN_CONFIDENCE,
    usable_pixels,
    visible_points,
)
from spatial_change.perception.sampling import sample_features
from spatial_change.ports import FeatureEncoder

_MIN_NORM = 1e-12


@dataclass(frozen=True)
class DescriptorConfig:
    """Parameters of the instance descriptors. The defaults are provisional.

    They are placeholders until T22 calibrates them on real captures.
    Lengths are in the units of the world frame, meters after the alignment
    stage.

    Attributes:
        voxel_size: Edge of the voxels that each count once, positive and
            finite.
        abs_tolerance: Depth tolerance at zero depth of the visibility
            test, positive and finite.
        rel_tolerance: Extra tolerance per unit of depth, at least 0 and
            finite.
        min_confidence: Pixels with a lower confidence give no view, in
            ``[0, 1]``.
    """

    voxel_size: float = 0.05
    abs_tolerance: float = 0.05
    rel_tolerance: float = 0.02
    min_confidence: float = DEFAULT_MIN_CONFIDENCE

    def __post_init__(self) -> None:
        """Validates the parameters.

        Raises:
            ValueError: If a length is out of range or not finite, or the
                confidence is outside ``[0, 1]``.
        """
        if not 0.0 < self.voxel_size < math.inf:
            raise ValueError("voxel_size must be positive and finite")
        if not 0.0 < self.abs_tolerance < math.inf:
            raise ValueError("abs_tolerance must be positive and finite")
        if not 0.0 <= self.rel_tolerance < math.inf:
            raise ValueError("rel_tolerance must be >= 0 and finite")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be within [0, 1]")


_DEFAULT_CONFIG = DescriptorConfig()


def _voxel_centers(
    points: NDArray[np.float64], voxel_size: float
) -> NDArray[np.float64]:
    """Represents each occupied voxel by the mean of its points.

    Args:
        points: ``(N, 3)`` world points.
        voxel_size: Voxel edge.

    Returns:
        ``(V, 3)`` voxel centers in the order of the voxel keys.
    """
    cells = voxel_cells(points, voxel_size)
    occupied, inverse = np.unique(cells, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    count = len(occupied)
    sizes = np.bincount(inverse, minlength=count)
    return np.stack(
        [
            np.bincount(inverse, weights=points[:, k], minlength=count) / sizes
            for k in range(3)
        ],
        axis=1,
    )


def describe_instances(
    reconstruction: EpochReconstruction,
    instances: tuple[ObjectInstance, ...],
    encoder: FeatureEncoder,
    config: DescriptorConfig = _DEFAULT_CONFIG,
) -> tuple[ObjectInstance, ...]:
    """Gives each instance the appearance feature of its surface.

    Each instance's points are binned into voxels, each represented by its
    mean point. Every frame of the reconstruction with an image and a depth
    map that sees at least one instance is encoded once; a frame that sees
    none is not encoded. The feature at the projection of each voxel the
    frame sees on its depth surface is normalized and added up, and the
    instance's feature is the sum, normalized again.

    Args:
        reconstruction: The epoch's reconstruction, whose frames carry the
            images and depth maps.
        instances: Instances of that epoch.
        encoder: Feature encoder; it is not unloaded.
        config: Voxel size and depth test.

    Returns:
        Copies of the instances in the same order with ``feature`` set, or
        None for an instance that no frame sees.

    Raises:
        ValueError: If an instance belongs to another epoch.
    """
    for item in instances:
        if item.epoch != reconstruction.epoch:
            raise ValueError(
                f"instance {item.object_id} is from epoch "
                f"{item.epoch.value}, the reconstruction from epoch "
                f"{reconstruction.epoch.value}"
            )
    if not instances:
        return ()
    centers = [_voxel_centers(i.points, config.voxel_size) for i in instances]
    totals: list[NDArray[np.float32] | None] = [None] * len(instances)
    for frame in reconstruction.frames:
        if frame.image is None or frame.depth is None:
            continue
        usable = usable_pixels(frame, config.min_confidence)
        views = [
            visible_points(
                voxels,
                frame,
                usable,
                config.abs_tolerance,
                config.rel_tolerance,
            )
            for voxels in centers
        ]
        if not any(view.indices.size for view in views):
            continue
        features = encoder.encode(frame.image)
        for index, seen in enumerate(views):
            if seen.indices.size == 0:
                continue
            sampled = sample_features(features, seen.uv, encoder.patch_size)
            norms = np.sqrt(np.einsum("vc,vc->v", sampled, sampled))
            sampled /= np.maximum(norms, _MIN_NORM)[:, None]
            total = sampled.sum(axis=0)
            running = totals[index]
            totals[index] = total if running is None else running + total
    return tuple(
        dataclasses.replace(item, feature=_normalized(total))
        for item, total in zip(instances, totals, strict=True)
    )


def _normalized(
    total: NDArray[np.float32] | None,
) -> NDArray[np.float32] | None:
    """Scales a summed feature to unit length.

    Args:
        total: The sum, or None when nothing was added.

    Returns:
        The unit vector, or None when there is no sum or it is zero.
    """
    if total is None:
        return None
    norm = float(np.linalg.norm(total))
    if norm < _MIN_NORM:
        return None
    return np.asarray(total / norm, dtype=np.float32)
