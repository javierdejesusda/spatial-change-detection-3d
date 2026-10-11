"""Fusion of posed frames into one point cloud per epoch."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain import EpochReconstruction, Frame
from spatial_change.geometry.projection import (
    DEFAULT_MIN_CONFIDENCE,
    confident_pixels,
    unproject_depth,
)


@dataclass(frozen=True, eq=False)
class _VoxelTable:
    """Per-voxel running sums, ordered by voxel key.

    Attributes:
        keys: ``(n, 3)`` int64 voxel indices, lexicographically sorted and
            unique.
        counts: ``(n,)`` number of points in each voxel.
        point_sums: ``(n, 3)`` float64 sum of the point positions.
        color_sums: ``(n, 3)`` int64 sum of the point colors.
        peak: ``(n,)`` float32 maximum confidence.
    """

    keys: NDArray[np.int64]
    counts: NDArray[np.int64]
    point_sums: NDArray[np.float64]
    color_sums: NDArray[np.int64]
    peak: NDArray[np.float32]


def fuse_frames(
    frames: Sequence[Frame],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    voxel_size: float | None = None,
) -> EpochReconstruction:
    """Fuses the frames of one epoch into a single point cloud.

    Pixels with a valid depth and a confidence at or above
    ``min_confidence`` are unprojected with each frame's pose. Colors and
    confidences come from the same pixels, in the row-major order of
    ``unproject_depth``.

    Args:
        frames: Posed frames of one epoch, each with depth, confidence, and
            image.
        min_confidence: Confidence threshold in ``[0, 1]``.
        voxel_size: Edge length of the voxel grid used to thin the cloud, or
            None to keep every point. With a grid, each occupied voxel
            yields its mean position, mean color rounded to uint8, and
            maximum confidence, ordered by voxel key. Each frame is reduced
            to its voxels as soon as it is unprojected, so memory follows
            the number of occupied voxels, not the number of raw points.

    Returns:
        The fused reconstruction with float64 points, uint8 colors, and
        float32 confidences.

    Raises:
        ValueError: If there are no frames, the frames belong to different
            epochs or lack depth, confidence, or image, ``min_confidence``
            is outside ``[0, 1]``, or ``voxel_size`` is not positive.
    """
    if not frames:
        raise ValueError("need at least one frame to fuse")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be within [0, 1]")
    if voxel_size is not None and not voxel_size > 0.0:
        raise ValueError("voxel_size must be positive")
    epoch = frames[0].epoch
    if any(frame.epoch != epoch for frame in frames):
        raise ValueError("all frames must belong to one epoch")

    points, colors, confidence, tables = [], [], [], []
    for frame in frames:
        if (
            frame.depth is None
            or frame.confidence is None
            or frame.image is None
        ):
            raise ValueError("every frame needs depth, confidence, and image")
        keep = confident_pixels(frame.depth, frame.confidence, min_confidence)
        frame_points = unproject_depth(
            frame.depth, frame.intrinsics, frame.world_from_camera, keep
        )
        frame_colors = frame.image[keep]
        frame_confidence = frame.confidence[keep]
        if voxel_size is None:
            points.append(frame_points)
            colors.append(frame_colors)
            confidence.append(frame_confidence)
        else:
            tables.append(
                _frame_table(
                    frame_points, frame_colors, frame_confidence, voxel_size
                )
            )

    if voxel_size is None:
        fused_points = np.concatenate(points)
        fused_colors = np.concatenate(colors)
        fused_confidence = np.concatenate(confidence)
    else:
        table = _merge(tables)
        counts = table.counts[:, None]
        fused_points = table.point_sums / counts
        fused_colors = np.rint(table.color_sums / counts).astype(np.uint8)
        fused_confidence = table.peak
    return EpochReconstruction(
        epoch=epoch,
        frames=list(frames),
        points=fused_points,
        colors=fused_colors,
        confidence=fused_confidence,
    )


def _frame_table(
    points: NDArray[np.float64],
    colors: NDArray[np.uint8],
    confidence: NDArray[np.float32],
    voxel_size: float,
) -> _VoxelTable:
    keys = np.floor(points / voxel_size).astype(np.int64)
    return _group(
        keys,
        np.ones(len(keys), np.int64),
        points,
        colors.astype(np.int64),
        confidence,
    )


def _merge(tables: Sequence[_VoxelTable]) -> _VoxelTable:
    return _group(
        np.concatenate([t.keys for t in tables]),
        np.concatenate([t.counts for t in tables]),
        np.concatenate([t.point_sums for t in tables]),
        np.concatenate([t.color_sums for t in tables]),
        np.concatenate([t.peak for t in tables]),
    )


def _group(
    keys: NDArray[np.int64],
    counts: NDArray[np.int64],
    point_sums: NDArray[np.float64],
    color_sums: NDArray[np.int64],
    peak: NDArray[np.float32],
) -> _VoxelTable:
    """Combines the rows that share a voxel key, in lexicographic key order."""
    if len(keys) == 0:
        return _VoxelTable(keys, counts, point_sums, color_sums, peak)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    starts = np.flatnonzero(
        np.concatenate(
            [[True], np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)]
        )
    )
    return _VoxelTable(
        keys=sorted_keys[starts],
        counts=np.add.reduceat(counts[order], starts),
        point_sums=np.add.reduceat(point_sums[order], starts, axis=0),
        color_sums=np.add.reduceat(color_sums[order], starts, axis=0),
        peak=np.maximum.reduceat(peak[order], starts),
    )
