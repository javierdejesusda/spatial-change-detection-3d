"""Scoring of a run against the exact truth of a synthetic scene pair.

A synthetic pair knows which objects moved, appeared, or vanished, which
unchanged one is hidden behind another, and which sits in a zone no camera of
the other epoch sees. ``score_changes`` matches the changes of a run to the
changed objects and counts what is missed and what is wrong, and counts apart
the wrong changes that sit where the run could not have known better. Nothing
here reads a file or loads a model.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain import Change, ChangeType, OrientedBox, Sim3
from spatial_change.pipeline import PipelineResult
from spatial_change.synthetic.scene import SyntheticObject, SyntheticScenePair

CENTER_TOLERANCE_M = 0.35
"""How far a box center may be from the truth, in meters of the room frame."""


@dataclass(frozen=True)
class SyntheticScore:
    """What a run found of the changes of a synthetic pair, in counts.

    Attributes:
        tp: Changes that found a changed object.
        fp: Changes that found none, uncertain ones included.
        fn: Changed objects that no change found.
        fp_occluded: False positives at an unchanged object that the
            other epoch sees only from behind another.
        fp_unobserved: False positives, not counted in ``fp_occluded``, at
            an unchanged object in the unobserved zone or inside the zone.
        n_uncertain: Uncertain changes, which find nothing.
    """

    tp: int
    fp: int
    fn: int
    fp_occluded: int
    fp_unobserved: int
    n_uncertain: int

    @property
    def precision(self) -> float:
        """Share of the changes that are right, 1 when there are none."""
        found = self.tp + self.fp
        return self.tp / found if found else 1.0

    @property
    def recall(self) -> float:
        """Share of the changed objects that were found, 1 if none changed."""
        wanted = self.tp + self.fn
        return self.tp / wanted if wanted else 1.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall.

        It is 1 when nothing was found and nothing was there to find, and 0
        when something was wrong or missed and nothing was right.
        """
        total = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / total if total else 1.0


def _in_room(box: OrientedBox, room_from_world: Sim3) -> OrientedBox:
    return OrientedBox(
        room_from_world.apply(box.center[None])[0],
        box.extent * room_from_world.scale,
        room_from_world.rotation @ box.rotation,
    )


def _change_centers(change: Change) -> list[NDArray[np.float64]]:
    boxes = (change.before_box, change.after_box)
    return [b.to_box().center for b in boxes if b is not None]


def _truth_centers(
    truth: SyntheticObject, room_from_world: Sim3
) -> list[NDArray[np.float64]]:
    boxes = (truth.box_before, truth.box_after)
    return [_in_room(b, room_from_world).center for b in boxes if b is not None]


def _gap(
    change: Change, truth: SyntheticObject, room_from_world: Sim3
) -> float | None:
    """Gives the largest gap between the boxes of a change and of an object.

    Returns:
        The largest distance between a true box center and the matching
        center of the change, or ``None`` if the change is of another type
        or lacks a box the object has.
    """
    if change.type is not truth.status.change_type:
        return None
    gaps = []
    for true_box, box in (
        (truth.box_before, change.before_box),
        (truth.box_after, change.after_box),
    ):
        if true_box is None:
            continue
        if box is None:
            return None
        center = _in_room(true_box, room_from_world).center
        gaps.append(float(np.linalg.norm(box.to_box().center - center)))
    return max(gaps) if gaps else None


def _matches(
    changes: Sequence[Change],
    truths: Sequence[SyntheticObject],
    room_from_world: Sim3,
    tolerance: float,
) -> set[int]:
    """Matches changes to objects greedily, the smallest gap first.

    Returns:
        The positions of the changes that found an object.
    """
    candidates = []
    for i, change in enumerate(changes):
        for j, truth in enumerate(truths):
            gap = _gap(change, truth, room_from_world)
            if gap is not None and gap <= tolerance:
                candidates.append((gap, i, j))
    found: set[int] = set()
    taken: set[int] = set()
    for _, i, j in sorted(candidates):
        if i not in found and j not in taken:
            found.add(i)
            taken.add(j)
    return found


def _is_near(
    change: Change,
    objects: Iterable[SyntheticObject],
    room_from_world: Sim3,
    tolerance: float,
) -> bool:
    centers = _change_centers(change)
    return any(
        np.linalg.norm(center - target) <= tolerance
        for obj in objects
        for target in _truth_centers(obj, room_from_world)
        for center in centers
    )


def _is_inside(change: Change, zone: OrientedBox) -> bool:
    centers = _change_centers(change)
    return bool(centers) and bool(zone.contains(np.array(centers)).any())


def score_changes(
    changes: Sequence[Change],
    room_from_world: Sim3,
    pair: SyntheticScenePair,
    tolerance: float = CENTER_TOLERANCE_M,
) -> SyntheticScore:
    """Scores the changes of a run against a synthetic pair.

    A change finds a changed object (moved, added, or removed) when both are
    of the same type, and every box the object has in the pair is matched by
    the change's box of that epoch, whose center is within ``tolerance`` of
    the true one. Pairs are taken smallest gap first, the gap of a pair being
    the largest of its center distances, and an object or a change is used
    once. An uncertain change finds nothing. A change that found nothing is
    counted in ``fp_occluded`` if one of its boxes is within ``tolerance``
    of an unchanged object hidden behind another, else in ``fp_unobserved``
    if one of its boxes is within ``tolerance`` of an unchanged object in
    the unobserved zone or has its center inside the zone. Unobserved
    regions that a run reports apart from its changes are not scored.

    Args:
        changes: The changes of the run, in the room frame.
        room_from_world: The map from the world frame of the pair to the room
            frame, as ``PipelineResult.room_from_world``.
        pair: The pair the run was made on.
        tolerance: The largest center distance of a match, in meters of the
            room frame.

    Returns:
        The counts.
    """
    truths = [o for o in pair.objects if o.status.change_type is not None]
    found = _matches(changes, truths, room_from_world, tolerance)
    occluded = [pair.object_by_id(i) for i in pair.occluded_object_ids]
    in_zone = [pair.object_by_id(i) for i in pair.zone_object_ids]
    zone = _in_room(pair.unobserved_zone, room_from_world)
    fp_occluded = fp_unobserved = 0
    for i, change in enumerate(changes):
        if i in found:
            continue
        if _is_near(change, occluded, room_from_world, tolerance):
            fp_occluded += 1
        elif _is_near(change, in_zone, room_from_world, tolerance) or (
            _is_inside(change, zone)
        ):
            fp_unobserved += 1
    return SyntheticScore(
        tp=len(found),
        fp=len(changes) - len(found),
        fn=len(truths) - len(found),
        fp_occluded=fp_occluded,
        fp_unobserved=fp_unobserved,
        n_uncertain=sum(c.type is ChangeType.UNCERTAIN for c in changes),
    )


def score_result(
    result: PipelineResult,
    pair: SyntheticScenePair,
    tolerance: float = CENTER_TOLERANCE_M,
) -> SyntheticScore:
    """Scores a run against the synthetic pair it was made on.

    Args:
        result: What the run produced.
        pair: The pair the run was made on.
        tolerance: The largest center distance of a match, in meters of the
            room frame.

    Returns:
        The counts, as ``score_changes`` gives them.
    """
    return score_changes(
        result.report.changes, result.room_from_world, pair, tolerance
    )


def pool_scores(scores: Iterable[SyntheticScore]) -> SyntheticScore:
    """Adds the counts of several runs.

    The precision, recall, and F1 of the sum are those of all the changes
    taken together, not the mean of the ones of each run.

    Args:
        scores: The scores to pool; there may be none.

    Returns:
        The summed counts, which for no score are all zero.
    """
    pooled = SyntheticScore(0, 0, 0, 0, 0, 0)
    for score in scores:
        pooled = SyntheticScore(
            pooled.tp + score.tp,
            pooled.fp + score.fp,
            pooled.fn + score.fn,
            pooled.fp_occluded + score.fp_occluded,
            pooled.fp_unobserved + score.fp_unobserved,
            pooled.n_uncertain + score.n_uncertain,
        )
    return pooled
