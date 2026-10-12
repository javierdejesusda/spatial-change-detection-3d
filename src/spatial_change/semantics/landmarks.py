"""Static landmarks extracted from the BEFORE reconstruction.

A landmark is a piece of furniture that the AFTER epoch confirms is still
there. The points that both epochs agree on lose the floor, the ceiling, and
the vertical walls, the rest is grouped into clusters, and a cluster whose
open-vocabulary label is a furniture name becomes a landmark.

Importing this module does not load torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain import Epoch, EpochReconstruction, OrientedBox
from spatial_change.geometry import (
    fit_upright_box,
    label_voxel_cells,
    voxel_cells,
)
from spatial_change.geometry.views import usable_frames
from spatial_change.geometry.voxels import cube_offsets, lookup_keys, voxel_grid
from spatial_change.geometry.walls import WallConfig, find_walls
from spatial_change.ports import Labeler
from spatial_change.semantics.labeling import LabelConfig, label_point_sets
from spatial_change.semantics.relations import Landmark
from spatial_change.semantics.vocabulary import FURNITURE, STRUCTURE
from spatial_change.visibility import CrossEvidence, PointState


@dataclass(frozen=True)
class LandmarkConfig:
    """Parameters of the landmark extraction. The defaults are provisional.

    They are placeholders until T22 calibrates them on real captures.
    Lengths are in meters of the z-up room frame, with the floor at
    ``z = 0``.

    Attributes:
        voxel_size: Edge of the voxels of the wall search and of the
            clusters, positive and finite.
        floor_clearance: Points lower than this are floor, at least 0 and
            finite.
        max_height: Points higher than this are dropped, which removes the
            ceiling; above ``floor_clearance`` and finite.
        walls: The search for the vertical walls that are removed first.
        link_radius: Voxels of a cluster are linked when their keys
            differ by at most this on every axis; at least 1.
        min_points: Points a cluster needs, at least 1.
        min_height: A cluster whose highest point is lower is dropped, so
            floor residue is not furniture; at least 0 and finite.
        min_extent: A cluster whose largest box extent is smaller is
            dropped, positive and finite.
        max_extent: A cluster whose largest box extent is larger is
            dropped (a missed wall or a merged room); at least
            ``min_extent`` and finite.
        max_changed_share: A cluster is dropped when the changed points
            within ``link_radius`` voxels of it make up at least this share
            of them and the cluster's own points, which removes the
            confirmed fringe of a changed object; in ``(0, 1]``.
        min_score: Smallest fused score of a furniture label that makes a
            landmark, in ``[0, 1]``; separate from the object threshold in
            ``labeling`` because the landmark vocabulary is smaller, so the
            same match scores higher.
        labeling: Views and crops of the labeling; its ``min_score`` is
            not used, ``min_score`` is.
    """

    voxel_size: float = 0.05
    floor_clearance: float = 0.05
    max_height: float = 2.2
    walls: WallConfig = WallConfig()
    link_radius: int = 2
    min_points: int = 50
    min_height: float = 0.2
    min_extent: float = 0.25
    max_extent: float = 3.0
    max_changed_share: float = 0.5
    min_score: float = 0.3
    labeling: LabelConfig = field(default_factory=LabelConfig)

    def __post_init__(self) -> None:
        """Validates the parameters.

        Raises:
            ValueError: If a length is out of range or not finite, or a
                count is out of range.
        """
        for name in (
            "voxel_size",
            "min_extent",
        ):
            if not 0.0 < getattr(self, name) < math.inf:
                raise ValueError(f"{name} must be positive and finite")
        if not 0.0 <= self.floor_clearance < math.inf:
            raise ValueError("floor_clearance must be >= 0 and finite")
        if not self.floor_clearance < self.max_height < math.inf:
            raise ValueError(
                "max_height must be above floor_clearance and finite"
            )
        if self.link_radius < 1:
            raise ValueError("link_radius must be >= 1")
        if self.min_points < 1:
            raise ValueError("min_points must be >= 1")
        if not 0.0 <= self.min_height < math.inf:
            raise ValueError("min_height must be >= 0 and finite")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be within [0, 1]")
        if not 0.0 < self.max_changed_share <= 1.0:
            raise ValueError("max_changed_share must be within (0, 1]")
        if not self.min_extent <= self.max_extent < math.inf:
            raise ValueError(
                "max_extent must be at least min_extent and finite"
            )


_DEFAULT_CONFIG = LandmarkConfig()


def _changed_near(
    cells: NDArray[np.int64],
    owner: NDArray[np.intp],
    changed: NDArray[np.int64],
    count: int,
    radius: int,
) -> NDArray[np.int64]:
    """Counts the changed points within some voxels of each cluster.

    Args:
        cells: ``(N, 3)`` voxel key of every clustered point, at least one.
        owner: ``(N,)`` cluster of every point; a voxel belongs to one
            cluster.
        changed: ``(M, 3)`` voxel key of each changed point.
        count: Number of clusters.
        radius: Voxels around a cluster that count.

    Returns:
        ``(count,)`` changed points that lie in a voxel within ``radius``
        voxels of the cluster's voxels on every axis. Changed points
        outside the bounds of the clusters' neighbourhood are skipped
        first, so a far point cannot blow up the grid.
    """
    origin, _, dims = voxel_grid(cells, radius)
    low = np.array(origin, np.int64)
    high = low + np.array(dims, np.int64) - 1
    changed = changed[np.all((changed >= low) & (changed <= high), axis=1)]
    counts = np.zeros(count, np.int64)
    if len(changed) == 0:
        return counts
    packed, first = np.unique(
        np.ravel_multi_index((cells - low).T, dims), return_index=True
    )
    cluster_of_voxel = owner[first]
    shifted = changed - low
    bounds = np.array(dims, np.int64)
    pairs = []
    for offset in cube_offsets(radius):
        near = shifted + offset
        inside = np.flatnonzero(np.all((near >= 0) & (near < bounds), axis=1))
        slot, found = lookup_keys(
            packed, np.ravel_multi_index(near[inside].T, dims)
        )
        pairs.append(
            inside[found] * count + cluster_of_voxel[slot[found]],
        )
    once = np.unique(np.concatenate(pairs))
    return np.bincount(once % count, minlength=count)


def _clusters(
    points: NDArray[np.float64],
    changed: NDArray[np.float64],
    config: LandmarkConfig,
) -> list[tuple[NDArray[np.float64], OrientedBox]]:
    """Groups points into connected voxel clusters of a plausible kind.

    Args:
        points: ``(N, 3)`` static points.
        changed: ``(M, 3)`` points of the changed surfaces.
        config: Voxel size, link radius, and cluster limits.

    Returns:
        The points and the upright box of each kept cluster.
    """
    if len(points) == 0:
        return []
    cells = voxel_cells(points, config.voxel_size)
    labels = label_voxel_cells(cells, config.link_radius)
    sizes = np.bincount(labels)
    big = np.flatnonzero(sizes >= config.min_points)
    if len(big) == 0:
        return []
    in_big = np.flatnonzero(sizes[labels] >= config.min_points)
    ordered = in_big[np.argsort(labels[in_big], kind="stable")]
    groups = np.split(ordered, np.cumsum(sizes[big])[:-1])
    kept = []
    for group in groups:
        members = points[group]
        if members[:, 2].max() < config.min_height:
            continue
        box = fit_upright_box(members, config.voxel_size)
        if config.min_extent <= box.extent.max() <= config.max_extent:
            kept.append((group, members, box))
    if not kept:
        return []
    near = _changed_near(
        cells[np.concatenate([group for group, _, _ in kept])],
        np.repeat(np.arange(len(kept)), [len(group) for group, _, _ in kept]),
        voxel_cells(changed, config.voxel_size),
        len(kept),
        config.link_radius,
    )
    return [
        (members, box)
        for (_, members, box), outside in zip(kept, near, strict=True)
        if outside < config.max_changed_share * (outside + len(members))
    ]


def extract_landmarks(
    reconstruction: EpochReconstruction,
    evidence: CrossEvidence,
    labeler: Labeler,
    config: LandmarkConfig = _DEFAULT_CONFIG,
) -> tuple[Landmark, ...]:
    """Finds the static pieces of furniture of the BEFORE epoch.

    Only the points that the AFTER epoch confirms (``PointState.PRESENT``)
    are used, and a cluster that sits among changed points is dropped, so a
    changed object is not a landmark. The floor (lower
    than ``floor_clearance``) and everything above ``max_height`` go, then
    the vertical walls found by RANSAC, and the rest is grouped into voxel
    clusters of a plausible size. Each cluster gets an upright box and is
    labeled over furniture and room structure from its best views; it
    becomes a landmark when the fused label is furniture.

    Args:
        reconstruction: The BEFORE reconstruction in the z-up room frame
            with the floor at ``z = 0``, whose frames carry the images and
            depth maps.
        evidence: The cross-epoch votes; the BEFORE side is used.
        labeler: Open-vocabulary labeler; it is not unloaded.
        config: Filtering, wall, cluster, and labeling settings.

    Returns:
        The landmarks, numbered ``landmark-0``, ``landmark-1``, ... by box
        center x and then y.

    Raises:
        ValueError: If the reconstruction is not the BEFORE epoch or the
            evidence does not cover its points.
    """
    if reconstruction.epoch != Epoch.BEFORE:
        raise ValueError("reconstruction must be the BEFORE epoch")
    states = evidence.before.states
    if len(states) != len(reconstruction.points):
        raise ValueError(
            f"evidence has {len(states)} points, the reconstruction "
            f"{len(reconstruction.points)}"
        )
    height = reconstruction.points[:, 2]
    between = (height >= config.floor_clearance) & (height <= config.max_height)
    points = reconstruction.points[(states == PointState.PRESENT) & between]
    changed = reconstruction.points[(states == PointState.CHANGED) & between]
    for wall in find_walls(points, config.voxel_size, config.walls):
        points = points[~wall.contains(points, config.walls.plane_distance)]
    clusters = _clusters(points, changed, config)
    if not clusters:
        return ()
    vocabulary = FURNITURE + STRUCTURE
    labels = label_point_sets(
        usable_frames(reconstruction, config.labeling.views.min_confidence),
        [members for members, _ in clusters],
        labeler,
        vocabulary,
        config.labeling,
        config.min_score,
    )
    found = [
        (box, label)
        for (_, box), (label, _) in zip(clusters, labels, strict=True)
        if label in FURNITURE
    ]
    found.sort(key=lambda item: (item[0].center[0], item[0].center[1]))
    return tuple(
        Landmark(f"landmark-{rank}", str(label), box)
        for rank, (box, label) in enumerate(found)
    )
