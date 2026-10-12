"""Regions that only one capture observed, which cannot be judged.

A point of one epoch is voted by every frame of the other epoch. When nearly
every vote is UNOBSERVED (the point is outside the frustum, or the depth is
unusable) the other capture never looked there, so a missing counterpart
says nothing about change. Such points are grouped into connected regions on
a voxel grid, like the change candidates, and reported apart from them.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from spatial_change.domain import (
    BoxModel,
    Epoch,
    EpochReconstruction,
    OrientedBox,
    UnobservedRegion,
)
from spatial_change.domain.transforms import as_finite_points
from spatial_change.geometry import (
    fit_upright_box,
    label_voxel_cells,
    voxel_cells,
)
from spatial_change.visibility import CrossEvidence, PointEvidence, PointState


@dataclass(frozen=True)
class UnobservedConfig:
    """Parameters of the unobserved-region extraction.

    The defaults suit a room reconstruction in meters: a region must hold
    about a third of a square meter of surface (30 voxels of 10 cm) seen as
    at least 200 points, which drops the specks along the edge of a
    frustum, and a point counts as unobserved when 90 % of its votes were
    cast from outside the other capture's view.

    Attributes:
        voxel_size: Edge of a grid voxel, positive and finite, in the units
            of the point cloud.
        link_radius: Two occupied voxels are linked when their keys differ
            by at most this on every axis; at least 1.
        min_points: Points a region needs, at least 1.
        min_voxels: Occupied voxels a region needs, at least 1.
        min_unobserved_fraction: Share of a point's votes that must be
            UNOBSERVED for the point to count, in ``(0, 1]``.
    """

    voxel_size: float = 0.1
    link_radius: int = 1
    min_points: int = 200
    min_voxels: int = 30
    min_unobserved_fraction: float = 0.9

    def __post_init__(self) -> None:
        """Validates the parameters.

        Raises:
            ValueError: If the voxel size is not positive and finite, a
                count is out of range, or the fraction is outside
                ``(0, 1]``.
        """
        if not 0.0 < self.voxel_size < math.inf:
            raise ValueError("voxel_size must be positive and finite")
        if self.link_radius < 1:
            raise ValueError("link_radius must be >= 1")
        if self.min_points < 1:
            raise ValueError("min_points must be >= 1")
        if self.min_voxels < 1:
            raise ValueError("min_voxels must be >= 1")
        if not 0.0 < self.min_unobserved_fraction <= 1.0:
            raise ValueError("min_unobserved_fraction must be within (0, 1]")


_DEFAULT_CONFIG = UnobservedConfig()


def _unobserved_mask(
    evidence: PointEvidence, config: UnobservedConfig
) -> NDArray[np.bool_]:
    """Marks the undecided points that the other capture never looked at."""
    votes = evidence.counts.sum(axis=1)
    undecided = evidence.states == PointState.UNKNOWN
    unobserved = evidence.unobserved >= config.min_unobserved_fraction * votes
    return np.asarray(undecided & (votes > 0) & unobserved, dtype=np.bool_)


def extract_unobserved(
    points: ArrayLike,
    evidence: PointEvidence,
    config: UnobservedConfig = _DEFAULT_CONFIG,
) -> tuple[OrientedBox, ...]:
    """Groups the unobserved points of one epoch into regions.

    The undecided points whose votes are almost all UNOBSERVED are binned
    into voxels of ``config.voxel_size`` and the occupied voxels are linked
    within ``config.link_radius``. A component is kept when it has enough
    points and voxels, and gets an upright box. The result is ordered by
    decreasing point count, ties by the smallest voxel key, so it does not
    depend on the order of the input points.

    Args:
        points: World points of the epoch, shape ``(N, 3)``; may be empty.
        evidence: Votes and tri-state of those points.
        config: Voxel size, linking, and the filters.

    Returns:
        The boxes of the regions, possibly none.

    Raises:
        ValueError: If ``points`` is not a finite ``(N, 3)`` array, the
            evidence has another number of rows, or the points span more
            voxels than a 64-bit key can index.
    """
    cloud = as_finite_points(points, "points", 0)
    if len(evidence.states) != len(cloud) or len(evidence.counts) != len(cloud):
        raise ValueError(
            f"evidence has {len(evidence.states)} rows for {len(cloud)} points"
        )
    members = cloud[_unobserved_mask(evidence, config)]
    if len(members) == 0:
        return ()
    cells = voxel_cells(members, config.voxel_size)
    labels = label_voxel_cells(cells, config.link_radius)
    components, component_of_point = np.unique(labels, return_inverse=True)
    point_counts = np.bincount(component_of_point, minlength=len(components))
    voxels = np.unique(
        np.column_stack([cells, component_of_point.reshape(-1, 1)]), axis=0
    )
    voxel_counts = np.bincount(voxels[:, 3], minlength=len(components))
    kept = np.flatnonzero(
        (point_counts >= config.min_points)
        & (voxel_counts >= config.min_voxels)
    )
    ranked = kept[np.argsort(-point_counts[kept], kind="stable")]
    return tuple(
        fit_upright_box(
            members[component_of_point == component], config.voxel_size
        )
        for component in ranked.tolist()
    )


def find_unobserved_regions(
    before: EpochReconstruction,
    after: EpochReconstruction,
    evidence: CrossEvidence,
    config: UnobservedConfig = _DEFAULT_CONFIG,
) -> tuple[UnobservedRegion, ...]:
    """Finds the regions that only one of the two captures observed.

    A region of BEFORE points was seen only by the BEFORE capture, so its
    ``epoch_seen`` is BEFORE; likewise for AFTER. A region holds only
    UNKNOWN points while a change candidate holds only CHANGED ones, so
    the two never share a point and an unobserved region is never reported
    as a change; a region box may still overlap a change box.

    Args:
        before: The BEFORE reconstruction, whose points the evidence votes.
        after: The AFTER reconstruction, whose points the evidence votes.
        evidence: The votes of both epochs, as returned by ``vote_epochs``.
        config: Voxel size, linking, and the filters.

    Returns:
        The BEFORE regions then the AFTER regions, each by decreasing point
        count, with ids ``region-0``, ``region-1``, and so on.

    Raises:
        ValueError: If ``before`` is not the BEFORE epoch, ``after`` is not
            the AFTER epoch, or ``extract_unobserved`` rejects an epoch.
    """
    if before.epoch != Epoch.BEFORE:
        raise ValueError("before must be the BEFORE reconstruction")
    if after.epoch != Epoch.AFTER:
        raise ValueError("after must be the AFTER reconstruction")
    found = [
        (
            Epoch.BEFORE,
            extract_unobserved(before.points, evidence.before, config),
        ),
        (Epoch.AFTER, extract_unobserved(after.points, evidence.after, config)),
    ]
    boxes = [(epoch, box) for epoch, group in found for box in group]
    return tuple(
        UnobservedRegion(
            region_id=f"region-{rank}",
            box=BoxModel.from_box(box),
            epoch_seen=epoch,
        )
        for rank, (epoch, box) in enumerate(boxes)
    )
