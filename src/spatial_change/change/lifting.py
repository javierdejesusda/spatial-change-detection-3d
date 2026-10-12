"""Lifting of change candidates to object instances with promptable masks.

A candidate is the CHANGED part of an object, and candidates fragment: one
object can give several. Each candidate is projected into the frames of its
own epoch that see it best, and a promptable segmenter turns a box and a
point there into a mask. The masks lift to world points through the frames'
depth, voxels the views disagree on are dropped, voxels that are not linked
to the candidate's own points are dropped, and fragments whose masks cover
each other's points are merged into one object.

Importing this module does not load torch.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spatial_change.change.candidates import Candidate, ChangeCandidates
from spatial_change.domain import (
    Epoch,
    EpochReconstruction,
    Frame,
    ObjectInstance,
)
from spatial_change.geometry import (
    fit_upright_box,
    label_voxel_cells,
    voxel_cells,
)
from spatial_change.geometry.projection import (
    VisiblePoints,
    unproject_depth,
    visible_points,
)
from spatial_change.geometry.views import (
    UsableFrame,
    ViewConfig,
    pixel_box,
    select_views,
    usable_frames,
)
from spatial_change.geometry.voxels import lookup_keys, voxel_grid
from spatial_change.ports import MaskSegmenter


@dataclass(frozen=True)
class LiftConfig:
    """Parameters of the lifting. The thresholds are provisional.

    They are placeholders until T22 calibrates them on real captures.
    Lengths are in the units of the world frame, meters after the alignment
    stage.

    Attributes:
        views: How the frames of a candidate are picked.
        box_margin_px: Pixels added on every side of the box prompt, at
            least 0 and finite.
        voxel_size: Edge of the voxels of the consensus and of the merge,
            positive and finite.
        min_agreement: Share of the views that see a voxel and did not lift
            any of its points whose mask must contain it for the voxel to
            stay, in ``[0, 1]``.
        min_points: Lifted points an instance needs; with fewer the
            candidate's own points are kept, at least 1.
        merge_overlap: Share of the own voxels of one candidate that must
            lie in the agreed voxels of another for the two to be merged,
            in ``(0, 1]``.
        link_radius: Agreed voxels stay only when linked to the voxels of
            the candidate's own points through voxels whose keys differ by
            at most this on every axis, so 2 bridges a gap of one empty
            voxel; at least 1.
        floor_clearance: Lifted points lower than this above the floor
            (``z = 0`` of the room frame) are dropped before the consensus
            unless they lie in a voxel of the candidate's own points, which
            keeps the object's floor contact; at least 0 and finite, and 0
            still drops the points below the floor.
    """

    views: ViewConfig = ViewConfig()
    box_margin_px: float = 4.0
    voxel_size: float = 0.05
    min_agreement: float = 0.5
    min_points: int = 10
    merge_overlap: float = 0.5
    link_radius: int = 2
    floor_clearance: float = 0.02

    def __post_init__(self) -> None:
        """Validates the parameters.

        Raises:
            ValueError: If a count is below 1, a length is out of range or
                not finite, or a share is outside its interval.
        """
        if not 0.0 <= self.box_margin_px < math.inf:
            raise ValueError("box_margin_px must be >= 0 and finite")
        if not 0.0 < self.voxel_size < math.inf:
            raise ValueError("voxel_size must be positive and finite")
        if not 0.0 <= self.min_agreement <= 1.0:
            raise ValueError("min_agreement must be within [0, 1]")
        if self.min_points < 1:
            raise ValueError("min_points must be >= 1")
        if not 0.0 < self.merge_overlap <= 1.0:
            raise ValueError("merge_overlap must be within (0, 1]")
        if self.link_radius < 1:
            raise ValueError("link_radius must be >= 1")
        if not 0.0 <= self.floor_clearance < math.inf:
            raise ValueError("floor_clearance must be >= 0 and finite")


_DEFAULT_CONFIG = LiftConfig()


@dataclass(frozen=True, eq=False)
class LiftedObjects:
    """The object instances of both directions.

    Attributes:
        disappeared: Instances of the BEFORE epoch.
        appeared: Instances of the AFTER epoch.
    """

    disappeared: tuple[ObjectInstance, ...]
    appeared: tuple[ObjectInstance, ...]


@dataclass(frozen=True, eq=False)
class _View:
    """One frame's prompt for one candidate.

    Attributes:
        position: Index of the frame in the reconstruction.
        box: ``(4,)`` box prompt ``(x0, y0, x1, y1)``.
        point: ``(2,)`` point prompt ``(x, y)``.
    """

    position: int
    box: NDArray[np.float64]
    point: NDArray[np.float64]


def _seen(
    points: NDArray[np.float64], item: UsableFrame, config: LiftConfig
) -> VisiblePoints:
    """Finds the points a frame sees on its depth surface.

    Args:
        points: ``(N, 3)`` world points.
        item: The frame.
        config: Supplies the depth tolerances.

    Returns:
        The visible points of ``points`` in the frame.
    """
    return visible_points(
        points,
        item.frame,
        item.usable,
        config.views.abs_tolerance,
        config.views.rel_tolerance,
    )


def _prompt(
    position: int,
    pixels: NDArray[np.intp],
    frame: Frame,
    config: LiftConfig,
) -> _View:
    """Builds the box and the point prompt from the visible pixels.

    Args:
        position: Index of the frame in the reconstruction.
        pixels: ``(M, 2)`` visible candidate pixels as (column, row), at
            least one.
        frame: The frame.
        config: Supplies ``box_margin_px``.

    Returns:
        The box of the pixels grown by the margin and clipped to the image,
        and the pixel nearest to their median.
    """
    box = pixel_box(
        pixels,
        config.box_margin_px,
        frame.intrinsics.width,
        frame.intrinsics.height,
    )
    distance = np.linalg.norm(pixels - np.median(pixels, axis=0), axis=1)
    return _View(position, box, pixels[np.argmin(distance)].astype(np.float64))


def _select_views(
    candidate: Candidate,
    frames: dict[int, UsableFrame],
    config: LiftConfig,
) -> list[_View]:
    """Picks the frames that see a candidate best and prompts them.

    Args:
        candidate: The candidate.
        frames: The usable frames of the candidate's epoch.
        config: Supplies the view limits, tolerances, and margin.

    Returns:
        Up to ``max_views`` views in frame order, the frames that see the
        most candidate points first chosen, ties to the earlier frame.
    """
    selected = select_views(candidate.points, frames, config.views)
    return [
        _prompt(
            view.position, view.seen.pixels, frames[view.position].frame, config
        )
        for view in selected
    ]


def _segment_views(
    frames: dict[int, UsableFrame],
    views: list[list[_View]],
    segmenter: MaskSegmenter,
) -> dict[tuple[int, int], NDArray[np.bool_]]:
    """Segments every frame once with the prompts of all candidates.

    Args:
        frames: The usable frames by position.
        views: The views of each candidate.
        segmenter: The mask segmenter.

    Returns:
        The mask of each view, keyed by the candidate's and the view's
        index.
    """
    by_frame: dict[int, list[tuple[int, int]]] = {}
    for candidate, candidate_views in enumerate(views):
        for index, view in enumerate(candidate_views):
            by_frame.setdefault(view.position, []).append((candidate, index))
    masks: dict[tuple[int, int], NDArray[np.bool_]] = {}
    for position in sorted(by_frame):
        keys = by_frame[position]
        boxes = np.stack([views[c][i].box for c, i in keys])
        points = np.stack([views[c][i].point for c, i in keys])
        result = segmenter.segment(frames[position].image, boxes, points)
        for key, mask in zip(keys, result, strict=True):
            masks[key] = mask
    return masks


def _check_epoch(
    reconstruction: EpochReconstruction, candidates: tuple[Candidate, ...]
) -> None:
    """Checks that every candidate belongs to the reconstruction's epoch.

    Args:
        reconstruction: The epoch's reconstruction.
        candidates: The candidates.

    Raises:
        ValueError: If a candidate is from another epoch.
    """
    for candidate in candidates:
        if candidate.epoch != reconstruction.epoch:
            raise ValueError(
                f"candidate {candidate.candidate_id} is from epoch "
                f"{candidate.epoch.value}, the reconstruction from epoch "
                f"{reconstruction.epoch.value}"
            )


@dataclass(frozen=True, eq=False)
class _Lifted:
    """The world points of one candidate and the voxels they occupy.

    Attributes:
        points: ``(N, 3)`` world points.
        voxels: ``(V, 3)`` unique int64 grid keys of the voxels the views
            agree on, before the link to the own voxels is required.
        own: ``(U, 3)`` unique int64 grid keys of the candidate's own
            points.
    """

    points: NDArray[np.float64]
    voxels: NDArray[np.int64]
    own: NDArray[np.int64]


def _anchored(
    voxels: NDArray[np.int64], own: NDArray[np.int64], radius: int
) -> NDArray[np.bool_]:
    """Tells which voxels are linked to the candidate's own voxels.

    Args:
        voxels: ``(V, 3)`` voxel keys to test.
        own: ``(U, 3)`` voxel keys of the candidate's points, at least one.
        radius: Link radius in voxels.

    Returns:
        Mask of shape ``(V,)``: true where the voxel and an own voxel are
        in one connected component of ``voxels`` and ``own`` together.
    """
    labels = label_voxel_cells(np.concatenate([voxels, own]), radius)
    return np.isin(labels[: len(voxels)], labels[len(voxels) :])


def _agreed_voxels(
    points: NDArray[np.float64],
    owner: NDArray[np.intp],
    views: list[_View],
    masks: list[NDArray[np.bool_]],
    frames: dict[int, UsableFrame],
    own: NDArray[np.int64],
    config: LiftConfig,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Keeps the lifted points whose voxels the views agree on.

    A view sees a voxel when its frame sees one of the voxel's points on
    its depth surface, and its mask includes the voxel when it contains the
    pixel of such a point. A view that lifted a point of the voxel does not
    vote on it, since its mask put the voxel there. The voxel stays when at
    least ``min_agreement`` of the views that see it and lifted none of its
    points include it. A voxel no such view sees is kept, and the link to
    the candidate's own voxels then bounds it. Of the voxels that stay, only
    those linked to the voxels of the candidate's own points are kept, so
    spill far from the object goes even if one view alone includes it.

    Args:
        points: ``(N, 3)`` lifted points of all the views, at least one.
        owner: ``(N,)`` index in ``views`` of the view that lifted each
            point.
        views: The candidate's views.
        masks: The mask of each view.
        frames: The usable frames by position.
        own: ``(U, 3)`` voxel keys of the candidate's own points.
        config: Supplies the voxel size, tolerances, agreement, and link
            radius.

    Returns:
        The points in the linked voxels and the agreed voxels.
    """
    cells = voxel_cells(points, config.voxel_size)
    occupied, inverse = np.unique(cells, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    count = len(occupied)
    visible = np.zeros(count, np.int64)
    hits = np.zeros(count, np.int64)
    for index, (view, mask) in enumerate(zip(views, masks, strict=True)):
        item = frames[view.position]
        lifted_by = np.bincount(inverse[owner == index], minlength=count) > 0
        seen = _seen(points, item, config)
        voting = ~lifted_by[inverse[seen.indices]]
        voxels = inverse[seen.indices][voting]
        inside = mask[seen.pixels[voting, 1], seen.pixels[voting, 0]]
        visible += np.bincount(voxels, minlength=count) > 0
        hits += np.bincount(voxels[inside], minlength=count) > 0
    agreed = hits >= config.min_agreement * visible
    linked = agreed.copy()
    linked[agreed] = _anchored(occupied[agreed], own, config.link_radius)
    return points[linked[inverse]], occupied[agreed]


def _off_floor(
    points: NDArray[np.float64],
    own: NDArray[np.int64],
    config: LiftConfig,
) -> NDArray[np.bool_]:
    """Tells which lifted points are not floor spill.

    Args:
        points: ``(N, 3)`` lifted points.
        own: ``(U, 3)`` voxel keys of the candidate's own points, at least
            one.
        config: Supplies ``floor_clearance`` and the voxel size.

    Returns:
        Mask of shape ``(N,)``: true for points at least ``floor_clearance``
        above the floor and for lower points in an own voxel.
    """
    keep = points[:, 2] >= config.floor_clearance
    low = ~keep
    cells, packed = _pack([voxel_cells(points[low], config.voxel_size), own])
    keep[low] = _contained(cells, packed)
    return keep


def _lift(
    candidate: Candidate,
    views: list[_View],
    masks: list[NDArray[np.bool_]],
    frames: dict[int, UsableFrame],
    config: LiftConfig,
) -> _Lifted:
    """Lifts the masks of a candidate and keeps what the views agree on.

    Args:
        candidate: The candidate.
        views: Its views.
        masks: The mask of each view.
        frames: The usable frames by position.
        config: Lifting settings.

    Returns:
        The agreed points when there are at least ``min_points``, else the
        candidate's own points.
    """
    parts = []
    for view, mask in zip(views, masks, strict=True):
        item = frames[view.position]
        parts.append(
            unproject_depth(
                item.depth,
                item.frame.intrinsics,
                item.frame.world_from_camera,
                mask & item.usable,
            )
        )
    lifted = np.concatenate(parts) if parts else np.empty((0, 3))
    owner = np.repeat(np.arange(len(parts)), [len(part) for part in parts])
    own = np.unique(voxel_cells(candidate.points, config.voxel_size), axis=0)
    keep = _off_floor(lifted, own, config)
    lifted, owner = lifted[keep], owner[keep]
    if len(lifted):
        points, voxels = _agreed_voxels(
            lifted, owner, views, masks, frames, own, config
        )
        if len(points) >= config.min_points:
            return _Lifted(points, voxels, own)
    return _Lifted(candidate.points.copy(), own, own)


def _merge_groups(lifted: list[_Lifted], config: LiftConfig) -> list[list[int]]:
    """Groups the instances that cover each other's candidates.

    Two instances are linked when the own voxels of one lie, for at least
    ``merge_overlap`` of them, inside the agreed voxels of the other. The
    mask of one fragment of an object covers the other fragments, while
    background that the masks of two objects both leak onto is in neither
    object's own voxels. The groups are the connected components.

    Args:
        lifted: The lifted candidates in candidate order.
        config: Supplies ``merge_overlap``.

    Returns:
        Groups of indices, each ascending, ordered by their first index.
    """
    count = len(lifted)
    packed = _pack(
        [item.own for item in lifted] + [item.voxels for item in lifted]
    )
    owns, voxels = packed[:count], packed[count:]
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first in range(count):
        for second in range(first + 1, count):
            cover = max(
                _covered(owns[first], voxels[second]),
                _covered(owns[second], voxels[first]),
            )
            if cover >= config.merge_overlap:
                low, high = sorted((find(first), find(second)))
                parent[high] = low
    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def _pack(groups: list[NDArray[np.int64]]) -> list[NDArray[np.int64]]:
    """Packs voxel keys into linear keys of one grid that spans them all.

    Args:
        groups: ``(K_i, 3)`` voxel keys, at least one in all. A group of
            sorted unique rows packs into sorted unique keys.

    Returns:
        The ``(K_i,)`` linear keys of each group.

    Raises:
        ValueError: If the keys span more voxels than a 64-bit key can
            index.
    """
    origin, _, dims = voxel_grid(np.concatenate(groups), 0)
    shift = np.array(origin, np.int64)
    return [np.ravel_multi_index((g - shift).T, dims) for g in groups]


def _contained(
    keys: NDArray[np.int64], packed: NDArray[np.int64]
) -> NDArray[np.bool_]:
    """Tells which linear voxel keys are in a set of voxels.

    Args:
        keys: ``(K,)`` linear keys.
        packed: ``(V,)`` sorted unique linear keys of the same grid.

    Returns:
        Mask of shape ``(K,)``.
    """
    if len(packed) == 0:
        return np.zeros(len(keys), np.bool_)
    return lookup_keys(packed, keys)[1]


def _covered(own: NDArray[np.int64], packed: NDArray[np.int64]) -> float:
    """Measures how much of a set of voxels lies in another.

    Args:
        own: ``(U,)`` linear keys, at least one.
        packed: ``(V,)`` sorted unique linear keys of the same grid.

    Returns:
        The share of ``own`` that is in ``packed``.
    """
    return float(_contained(own, packed).mean())


def _union(parts: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Joins point sets, dropping repeats and keeping the order.

    The same pixel of one frame lifts to the same point for every candidate
    that uses the frame, so the sets share exact points.

    Args:
        parts: ``(N_i, 3)`` point sets.

    Returns:
        The points of the first set, then the new points of each next set,
        each in its original order.
    """
    points = np.concatenate(parts)
    _, first = np.unique(points, axis=0, return_index=True)
    return points[np.sort(first)]


def lift_candidates(
    reconstruction: EpochReconstruction,
    candidates: tuple[Candidate, ...],
    segmenter: MaskSegmenter,
    config: LiftConfig = _DEFAULT_CONFIG,
) -> tuple[ObjectInstance, ...]:
    """Turns the candidates of one epoch into object instances.

    Each candidate is prompted in up to ``max_views`` frames that see it,
    every frame being segmented once for all the candidates that chose it.
    The masks lift to world points through the frames' depth, the voxels
    the views disagree on or that are not linked to the candidate's own
    points are dropped, and instances whose masks cover each other's
    candidates are merged (see ``LiftConfig``).

    Args:
        reconstruction: The epoch's reconstruction, whose frames carry the
            images and depth maps.
        candidates: Candidates of that epoch.
        segmenter: Promptable mask segmenter; it is not unloaded.
        config: View selection, consensus, and merge settings.

    Returns:
        The instances in candidate order; a merged instance takes the id of
        its first candidate and the union of the points.

    Raises:
        ValueError: If a candidate belongs to another epoch.
    """
    _check_epoch(reconstruction, candidates)
    if not candidates:
        return ()
    frames = usable_frames(reconstruction, config.views.min_confidence)
    views = [_select_views(c, frames, config) for c in candidates]
    masks = _segment_views(frames, views, segmenter)
    lifted = [
        _lift(
            candidate,
            views[index],
            [masks[index, view] for view in range(len(views[index]))],
            frames,
            config,
        )
        for index, candidate in enumerate(candidates)
    ]
    instances = []
    for group in _merge_groups(lifted, config):
        points = (
            lifted[group[0]].points
            if len(group) == 1
            else _union([lifted[i].points for i in group])
        )
        instances.append(
            ObjectInstance(
                object_id=candidates[group[0]].candidate_id,
                epoch=reconstruction.epoch,
                points=points,
                box=fit_upright_box(points, config.voxel_size),
            )
        )
    return tuple(instances)


def lift_change_candidates(
    before: EpochReconstruction,
    after: EpochReconstruction,
    candidates: ChangeCandidates,
    segmenter: MaskSegmenter,
    config: LiftConfig = _DEFAULT_CONFIG,
) -> LiftedObjects:
    """Lifts the candidates of both directions to object instances.

    Args:
        before: The BEFORE reconstruction.
        after: The AFTER reconstruction.
        candidates: The candidates, as returned by ``find_candidates``.
        segmenter: Promptable mask segmenter; it is not unloaded.
        config: View selection, consensus, and merge settings.

    Returns:
        The BEFORE instances as ``disappeared`` and the AFTER instances as
        ``appeared``.

    Raises:
        ValueError: If ``before`` is not the BEFORE epoch, ``after`` is not
            the AFTER epoch, or a candidate belongs to another epoch.
    """
    if before.epoch != Epoch.BEFORE:
        raise ValueError("before must be the BEFORE reconstruction")
    if after.epoch != Epoch.AFTER:
        raise ValueError("after must be the AFTER reconstruction")
    return LiftedObjects(
        disappeared=lift_candidates(
            before, candidates.disappeared, segmenter, config
        ),
        appeared=lift_candidates(after, candidates.appeared, segmenter, config),
    )
