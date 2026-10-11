"""Candidate change regions grouped from visibility evidence.

The CHANGED points of one epoch (T10 votes) are grouped into connected
regions on a voxel grid. A region of BEFORE points is a surface that
disappeared; a region of AFTER points is a surface that appeared. Each
candidate keeps its points, an upright box, and the summed votes that
support it.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from spatial_change.domain import (
    Epoch,
    EpochReconstruction,
    EvidenceCounts,
    OrientedBox,
)
from spatial_change.domain.transforms import as_finite_points
from spatial_change.geometry import fit_upright_box
from spatial_change.visibility import CrossEvidence, PointEvidence, PointState


@dataclass(frozen=True)
class CandidateConfig:
    """Parameters of the candidate extraction.

    Attributes:
        voxel_size: Edge of a grid voxel, positive and finite, in the units
            of the point cloud (meters after the alignment stage).
        link_radius: Two occupied voxels are linked when their keys differ
            by at most this on every axis, so 1 is the 26-neighbourhood and
            2 bridges a gap of one empty voxel; at least 1.
        min_points: CHANGED points a candidate needs, at least 1.
        min_voxels: Occupied voxels a candidate needs, at least 1.
        min_changed_fraction: Share of the epoch's CHANGED and PRESENT points
            inside the candidate's voxels that must be CHANGED, in
            ``[0, 1]``. UNKNOWN points do not count: the other epoch never
            saw them. It rejects speckles of false CHANGED on a surface
            confirmed PRESENT.
    """

    voxel_size: float = 0.05
    link_radius: int = 2
    min_points: int = 10
    min_voxels: int = 3
    min_changed_fraction: float = 0.3

    def __post_init__(self) -> None:
        """Validates the parameters.

        Raises:
            ValueError: If the voxel size is not positive and finite, a
                count is out of range, or the fraction is outside
                ``[0, 1]``.
        """
        if not 0.0 < self.voxel_size < math.inf:
            raise ValueError("voxel_size must be positive and finite")
        if self.link_radius < 1:
            raise ValueError("link_radius must be >= 1")
        if self.min_points < 1:
            raise ValueError("min_points must be >= 1")
        if self.min_voxels < 1:
            raise ValueError("min_voxels must be >= 1")
        if not 0.0 <= self.min_changed_fraction <= 1.0:
            raise ValueError("min_changed_fraction must be within [0, 1]")


_DEFAULT_CONFIG = CandidateConfig()
# Voxels the linear key of the grid can index, as a float to compare spans
# that are too large for an int64.
_MAX_VOXELS = 2.0**63


@dataclass(frozen=True, eq=False)
class Candidate:
    """A connected region of CHANGED points in one epoch.

    Attributes:
        candidate_id: ``"<epoch>-<rank>"``, the rank counting from 0 by
            point count, for example ``"before-0"``.
        epoch: BEFORE if the surface disappeared, AFTER if it appeared.
        point_indices: Sorted rows of the CHANGED points in that epoch's
            points, which are also the rows of its ``PointEvidence``.
        points: The CHANGED points, shape ``(N, 3)``.
        box: Upright box around ``points``.
        evidence: Vote counts summed over the points.
        voxel_count: Occupied voxels of the region.
        changed_fraction: CHANGED points divided by the CHANGED and PRESENT
            points of the epoch that fall in the region's voxels, UNKNOWN
            points excluded: near 1 for a vanished object, low for speckles
            on a confirmed static surface.
    """

    candidate_id: str
    epoch: Epoch
    point_indices: NDArray[np.int64]
    points: NDArray[np.float64]
    box: OrientedBox
    evidence: EvidenceCounts
    voxel_count: int
    changed_fraction: float

    @property
    def point_count(self) -> int:
        """Number of CHANGED points."""
        return len(self.point_indices)


@dataclass(frozen=True, eq=False)
class ChangeCandidates:
    """The candidates of both directions.

    Attributes:
        disappeared: Regions of BEFORE points, by decreasing point count.
        appeared: Regions of AFTER points, by decreasing point count.
    """

    disappeared: tuple[Candidate, ...]
    appeared: tuple[Candidate, ...]


def _forward_offsets(radius: int) -> NDArray[np.int64]:
    """Lists the neighbour offsets that come after zero in key order.

    Args:
        radius: Largest absolute offset per axis.

    Returns:
        Array of shape ``(((2r + 1)^3 - 1) / 2, 3)``: one offset of every
        opposite pair, so each neighbouring pair is visited once.
    """
    span = np.arange(-radius, radius + 1)
    grid = np.stack(np.meshgrid(span, span, span, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3)[(2 * radius + 1) ** 3 // 2 + 1 :]


def _lookup(
    packed: NDArray[np.int64], keys: ArrayLike
) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
    """Finds linear voxel keys among the occupied ones.

    Args:
        packed: Sorted unique linear voxel keys, shape ``(V,)``, at least
            one.
        keys: Linear keys to find, shape ``(K,)``.

    Returns:
        The rows of ``packed`` and a mask of shape ``(K,)``. A row holds its
        key where the mask is true; elsewhere it is only a valid index.
    """
    slots = np.minimum(np.searchsorted(packed, keys), len(packed) - 1)
    return slots, packed[slots] == keys


def _label_voxels(
    packed: NDArray[np.int64], dims: tuple[int, ...], radius: int
) -> NDArray[np.int64]:
    """Labels occupied voxels by connected component.

    Every neighbouring pair is found by a binary search of the shifted
    key. The components then grow by hooking each edge's roots onto the
    smaller one and halving the trees by pointer jumping until no edge
    joins two trees.

    Args:
        packed: Sorted unique linear voxel keys, shape ``(V,)``.
        dims: Grid size the linear keys are packed with.
        radius: Neighbourhood radius in voxels.

    Returns:
        For every voxel the row of the first voxel of its component.
    """
    strides = np.array([dims[1] * dims[2], dims[2], 1], dtype=np.int64)
    heads, tails = [], []
    for offset in _forward_offsets(radius):
        slot, linked = _lookup(packed, packed + offset @ strides)
        heads.append(np.flatnonzero(linked))
        tails.append(slot[linked])
    head, tail = np.concatenate(heads), np.concatenate(tails)
    parent = np.arange(len(packed))
    while head.size:
        root_head, root_tail = parent[head], parent[tail]
        lower = np.minimum(root_head, root_tail)
        np.minimum.at(parent, root_head, lower)
        np.minimum.at(parent, root_tail, lower)
        while True:
            jumped = parent[parent]
            if np.array_equal(jumped, parent):
                break
            parent = jumped
        open_edges = parent[head] != parent[tail]
        head, tail = head[open_edges], tail[open_edges]
    return parent


def _points_per_component(
    keys: NDArray[np.int64],
    dims: tuple[int, ...],
    packed: NDArray[np.int64],
    component_of_voxel: NDArray[np.int64],
    components: int,
) -> NDArray[np.int64]:
    """Counts the points whose voxel belongs to each component.

    Args:
        keys: Grid key of every point to count, shape ``(N, 3)``. Keys
            outside ``[0, dims)`` lie off the grid and count for nobody.
        dims: Grid size the linear keys are packed with.
        packed: Sorted unique linear keys of the occupied voxels.
        component_of_voxel: Component of every occupied voxel.
        components: Number of components.

    Returns:
        Array of shape ``(components,)``.
    """
    inside = np.all((keys >= 0) & (keys < dims), axis=1)
    linear = np.ravel_multi_index(keys[inside].T, dims)
    slot, occupied = _lookup(packed, linear)
    return np.bincount(component_of_voxel[slot[occupied]], minlength=components)


def extract_candidates(
    points: ArrayLike,
    evidence: PointEvidence,
    epoch: Epoch,
    config: CandidateConfig = _DEFAULT_CONFIG,
) -> tuple[Candidate, ...]:
    """Groups the CHANGED points of one epoch into candidate regions.

    The CHANGED points are binned into voxels of ``config.voxel_size`` and
    the occupied voxels are linked within ``config.link_radius``. A
    connected component is kept when it has enough points and voxels and
    its ``changed_fraction`` reaches ``config.min_changed_fraction``: the
    CHANGED points over the CHANGED and PRESENT points of the epoch in its
    voxels, so UNKNOWN points count neither way. The result is ordered by
    decreasing point count, ties by the smallest voxel key, so it does not
    depend on the order of the input points.

    Args:
        points: World points of the epoch, shape ``(N, 3)``; may be empty.
        evidence: Votes and tri-state of those points.
        epoch: The epoch of ``points``.
        config: Voxel size, linking, and the filters.

    Returns:
        The candidates, possibly none.

    Raises:
        ValueError: If ``points`` is not a finite ``(N, 3)`` array, the
            evidence has another number of rows, or the CHANGED points
            span more voxels than a 64-bit key can index.
    """
    cloud = as_finite_points(points, "points", 0)
    if len(evidence.states) != len(cloud) or len(evidence.counts) != len(cloud):
        raise ValueError(
            f"evidence has {len(evidence.states)} rows for {len(cloud)} points"
        )
    changed = np.flatnonzero(evidence.states == PointState.CHANGED)
    if changed.size == 0:
        return ()
    radius = config.link_radius
    with np.errstate(over="ignore", invalid="ignore"):
        cells = np.floor(cloud / config.voxel_size)
        changed_cells = cells[changed]
        origin = changed_cells.min(axis=0) - radius
        spans = changed_cells.max(axis=0) + radius - origin + 1
    # Negated so that a NaN span, which no comparison accepts, is rejected.
    if not math.prod(spans.tolist()) < _MAX_VOXELS:
        raise ValueError(
            "the CHANGED points span too many voxels; increase voxel_size"
        )
    dims = tuple(spans.astype(np.int64).tolist())
    # Points off the grid are clipped to the voxel just outside it, so the
    # cast to int64 never overflows.
    keys = np.clip(cells - origin, -1.0, spans).astype(np.int64)
    packed, voxel_of_point = np.unique(
        np.ravel_multi_index(keys[changed].T, dims), return_inverse=True
    )
    roots = _label_voxels(packed, dims, radius)
    firsts, component_of_voxel = np.unique(roots, return_inverse=True)
    component_of_point = component_of_voxel[voxel_of_point]
    count = len(firsts)
    point_counts = np.bincount(component_of_point, minlength=count)
    voxel_counts = np.bincount(component_of_voxel, minlength=count)
    present = _points_per_component(
        keys[evidence.states == PointState.PRESENT],
        dims,
        packed,
        component_of_voxel,
        count,
    )
    fractions = point_counts / (point_counts + present)
    kept = np.flatnonzero(
        (point_counts >= config.min_points)
        & (voxel_counts >= config.min_voxels)
        & (fractions >= config.min_changed_fraction)
    )
    ranked = kept[np.argsort(-point_counts[kept], kind="stable")]
    grouped = changed[np.argsort(component_of_point, kind="stable")]
    groups = np.split(grouped, np.cumsum(point_counts)[:-1])
    candidates = []
    for rank, component in enumerate(ranked.tolist()):
        indices = groups[component]
        members = cloud[indices]
        candidates.append(
            Candidate(
                candidate_id=f"{epoch.value}-{rank}",
                epoch=epoch,
                point_indices=indices,
                points=members,
                box=fit_upright_box(members, config.voxel_size),
                evidence=evidence.total(indices),
                voxel_count=int(voxel_counts[component]),
                changed_fraction=float(fractions[component]),
            )
        )
    return tuple(candidates)


def find_candidates(
    before: EpochReconstruction,
    after: EpochReconstruction,
    evidence: CrossEvidence,
    config: CandidateConfig = _DEFAULT_CONFIG,
) -> ChangeCandidates:
    """Finds the regions that disappeared and the regions that appeared.

    Args:
        before: The BEFORE reconstruction, whose points the evidence votes.
        after: The AFTER reconstruction, whose points the evidence votes.
        evidence: The votes of both epochs, as returned by ``vote_epochs``.
        config: Voxel size, linking, and the filters.

    Returns:
        The BEFORE candidates as ``disappeared`` and the AFTER candidates as
        ``appeared``.

    Raises:
        ValueError: If ``before`` is not the BEFORE epoch, ``after`` is not
            the AFTER epoch, or ``extract_candidates`` rejects an epoch.
    """
    if before.epoch != Epoch.BEFORE:
        raise ValueError("before must be the BEFORE reconstruction")
    if after.epoch != Epoch.AFTER:
        raise ValueError("after must be the AFTER reconstruction")
    return ChangeCandidates(
        disappeared=extract_candidates(
            before.points, evidence.before, Epoch.BEFORE, config
        ),
        appeared=extract_candidates(
            after.points, evidence.after, Epoch.AFTER, config
        ),
    )
