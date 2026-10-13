import dataclasses

import numpy as np
import pytest
from rotations import rotation_z

from spatial_change.domain import (
    BoxModel,
    Change,
    ChangeType,
    EvidenceCounts,
    OrientedBox,
    Sim3,
)
from spatial_change.evaluation.synthetic import (
    CENTER_TOLERANCE_M,
    SyntheticScore,
    pool_scores,
    score_changes,
)
from spatial_change.synthetic.scene import (
    ObjectStatus,
    SceneConfig,
    SyntheticObject,
    SyntheticScenePair,
    generate_scene_pair,
)

ROOM_FROM_WORLD = Sim3(
    1.5, rotation_z(40.0), np.array([1.0, -2.0, 0.5], dtype=np.float64)
)
"""A frame that is neither the identity nor unit scale."""
TOLERANCE = CENTER_TOLERANCE_M
"""The tolerance is in meters of the room frame."""


@pytest.fixture(scope="module")
def pair() -> SyntheticScenePair:
    return generate_scene_pair(
        0, SceneConfig(width=64, height=48, frames_per_epoch=4)
    )


def one(pair: SyntheticScenePair, status: ObjectStatus) -> SyntheticObject:
    (found,) = [o for o in pair.objects if o.status is status]
    return found


def room_box(box: OrientedBox, shift: tuple[float, float, float]) -> BoxModel:
    """Gives a world box in the room frame, moved by ``shift`` there."""
    return BoxModel.from_box(
        OrientedBox(
            ROOM_FROM_WORLD.apply(box.center[None])[0] + np.array(shift),
            box.extent * ROOM_FROM_WORLD.scale,
            ROOM_FROM_WORLD.rotation @ box.rotation,
        )
    )


def change_at(
    kind: ChangeType,
    before: OrientedBox | None,
    after: OrientedBox | None,
    *,
    before_shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    after_shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Change:
    return Change(
        change_id="c",
        type=kind,
        label="box",
        description="d",
        confidence=0.9,
        before_box=room_box(before, before_shift) if before else None,
        after_box=room_box(after, after_shift) if after else None,
        displacement_m=1.0 if kind is ChangeType.MOVED else None,
        rotation_deg=0.0 if kind is ChangeType.MOVED else None,
        evidence=EvidenceCounts(consistent=0, free=3, occluded=0, unobserved=0),
    )


def exact(pair: SyntheticScenePair, status: ObjectStatus) -> Change:
    truth = one(pair, status)
    assert truth.status.change_type is not None
    return change_at(
        truth.status.change_type, truth.box_before, truth.box_after
    )


def true_changes(pair: SyntheticScenePair) -> list[Change]:
    return [
        exact(pair, s)
        for s in (ObjectStatus.MOVED, ObjectStatus.ADDED, ObjectStatus.REMOVED)
    ]


def score(pair: SyntheticScenePair, changes: list[Change]) -> SyntheticScore:
    return score_changes(changes, ROOM_FROM_WORLD, pair)


def hidden_box(pair: SyntheticScenePair, object_id: str) -> OrientedBox:
    box = pair.object_by_id(object_id).box_before
    assert box is not None
    return box


def test_the_tolerance_is_the_one_of_the_pipeline_checks() -> None:
    assert CENTER_TOLERANCE_M == 0.35


def test_the_changes_that_are_there_are_all_found(
    pair: SyntheticScenePair,
) -> None:
    got = score(pair, true_changes(pair))

    assert (got.tp, got.fp, got.fn) == (3, 0, 0)
    assert (got.fp_occluded, got.fp_unobserved, got.n_uncertain) == (0, 0, 0)
    assert (got.precision, got.recall, got.f1) == (1.0, 1.0, 1.0)


def test_a_change_within_the_tolerance_is_found_and_beyond_it_is_not(
    pair: SyntheticScenePair,
) -> None:
    added = one(pair, ObjectStatus.ADDED)
    near = (TOLERANCE - 0.02, 0.0, 0.0)
    far = (TOLERANCE + 0.02, 0.0, 0.0)

    inside = score(
        pair,
        [change_at(ChangeType.ADDED, None, added.box_after, after_shift=near)],
    )
    outside = score(
        pair,
        [change_at(ChangeType.ADDED, None, added.box_after, after_shift=far)],
    )

    assert (inside.tp, inside.fp, inside.fn) == (1, 0, 2)
    assert (outside.tp, outside.fp, outside.fn) == (0, 1, 3)


def test_the_tolerance_can_be_given(pair: SyntheticScenePair) -> None:
    added = one(pair, ObjectStatus.ADDED)
    off = change_at(
        ChangeType.ADDED, None, added.box_after, after_shift=(0.2, 0.0, 0.0)
    )

    loose = score_changes([off], ROOM_FROM_WORLD, pair, tolerance=0.3)
    strict = score_changes([off], ROOM_FROM_WORLD, pair, tolerance=0.1)

    assert (loose.tp, loose.fp) == (1, 0)
    assert (strict.tp, strict.fp) == (0, 1)


def test_a_change_of_another_type_does_not_find_the_object(
    pair: SyntheticScenePair,
) -> None:
    added = one(pair, ObjectStatus.ADDED)
    wrong = change_at(ChangeType.REMOVED, added.box_after, None)

    got = score(pair, [wrong])

    assert (got.tp, got.fp, got.fn) == (0, 1, 3)


def test_a_moved_change_needs_both_of_its_boxes_in_place(
    pair: SyntheticScenePair,
) -> None:
    moved = one(pair, ObjectStatus.MOVED)
    far = (TOLERANCE + 0.1, 0.0, 0.0)

    late = change_at(
        ChangeType.MOVED, moved.box_before, moved.box_after, after_shift=far
    )
    early = change_at(
        ChangeType.MOVED, moved.box_before, moved.box_after, before_shift=far
    )

    for wrong in (late, early):
        got = score(pair, [wrong])
        assert (got.tp, got.fp, got.fn) == (0, 1, 3)


def test_an_uncertain_change_finds_nothing_and_is_a_false_positive(
    pair: SyntheticScenePair,
) -> None:
    moved = one(pair, ObjectStatus.MOVED)
    unsure = change_at(ChangeType.UNCERTAIN, moved.box_before, moved.box_after)

    got = score(pair, [unsure])

    assert (got.tp, got.fp, got.fn, got.n_uncertain) == (0, 1, 3, 1)


def test_the_smallest_gap_is_matched_first_whatever_the_order(
    pair: SyntheticScenePair,
) -> None:
    first = one(pair, ObjectStatus.ADDED)
    assert first.box_after is not None
    step = np.array([0.2, 0.0, 0.0])
    second = dataclasses.replace(
        first,
        object_id="added-again",
        box_after=OrientedBox(
            first.box_after.center + step,
            first.box_after.extent,
            first.box_after.rotation,
        ),
    )
    crowded = dataclasses.replace(pair, objects=(*pair.objects, second))
    apart = ROOM_FROM_WORLD.scale * ROOM_FROM_WORLD.rotation @ step
    direction = apart / np.linalg.norm(apart)
    on_second = change_at(
        ChangeType.ADDED,
        None,
        first.box_after,
        after_shift=tuple(apart),
    )
    beside_first = change_at(
        ChangeType.ADDED,
        None,
        first.box_after,
        after_shift=tuple(-0.2 * direction),
    )

    for changes in ([on_second, beside_first], [beside_first, on_second]):
        got = score(crowded, changes)
        assert (got.tp, got.fp, got.fn) == (2, 0, 2)


def test_a_change_where_an_object_is_hidden_is_counted_as_such(
    pair: SyntheticScenePair,
) -> None:
    (stool,) = pair.occluded_object_ids
    box = hidden_box(pair, stool)
    ghost = change_at(
        ChangeType.ADDED, None, box, after_shift=(TOLERANCE - 0.05, 0.0, 0.0)
    )

    got = score(pair, [*true_changes(pair), ghost])

    assert (got.tp, got.fp, got.fn) == (3, 1, 0)
    assert (got.fp_occluded, got.fp_unobserved) == (1, 0)
    assert got.precision == pytest.approx(0.75)


def test_a_change_where_an_object_is_unobserved_is_counted_as_such(
    pair: SyntheticScenePair,
) -> None:
    (crate,) = pair.zone_object_ids
    box = hidden_box(pair, crate)
    ghost = change_at(ChangeType.REMOVED, box, None)

    got = score(pair, [*true_changes(pair), ghost])

    assert (got.tp, got.fp, got.fn) == (3, 1, 0)
    assert (got.fp_occluded, got.fp_unobserved) == (0, 1)


def test_a_change_inside_the_unobserved_zone_is_counted_as_such(
    pair: SyntheticScenePair,
) -> None:
    zone = pair.unobserved_zone
    inside = zone.center + zone.rotation @ (zone.extent * 0.4)
    stray = OrientedBox(inside, np.array([0.2, 0.2, 0.2]), zone.rotation)
    (crate,) = pair.zone_object_ids
    crate_center = hidden_box(pair, crate).center
    assert np.linalg.norm(inside - crate_center) * ROOM_FROM_WORLD.scale > (
        TOLERANCE
    )

    got = score(pair, [change_at(ChangeType.ADDED, None, stray)])

    assert (got.fp, got.fp_unobserved, got.fp_occluded) == (1, 1, 0)


def test_a_change_far_from_everything_is_only_a_false_positive(
    pair: SyntheticScenePair,
) -> None:
    far = OrientedBox(
        np.array([50.0, 50.0, 0.5]), np.array([0.3, 0.3, 0.3]), np.eye(3)
    )

    got = score(pair, [change_at(ChangeType.ADDED, None, far)])

    assert (got.fp, got.fp_occluded, got.fp_unobserved) == (1, 0, 0)


def test_an_uncertain_change_at_a_hidden_object_is_counted_there_too(
    pair: SyntheticScenePair,
) -> None:
    (stool,) = pair.occluded_object_ids
    unsure = change_at(
        ChangeType.UNCERTAIN, hidden_box(pair, stool), hidden_box(pair, stool)
    )

    got = score(pair, [unsure])

    assert (got.fp, got.n_uncertain, got.fp_occluded) == (1, 1, 1)


def test_a_run_that_finds_nothing_misses_everything(
    pair: SyntheticScenePair,
) -> None:
    got = score(pair, [])

    assert (got.tp, got.fp, got.fn) == (0, 0, 3)
    assert (got.precision, got.recall, got.f1) == (1.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("tp", "fp", "fn", "precision", "recall", "f1"),
    [
        (0, 0, 0, 1.0, 1.0, 1.0),
        (0, 2, 0, 0.0, 1.0, 0.0),
        (0, 0, 2, 1.0, 0.0, 0.0),
        (0, 2, 2, 0.0, 0.0, 0.0),
        (3, 1, 1, 0.75, 0.75, 0.75),
        (6, 2, 0, 0.75, 1.0, 12 / 14),
    ],
)
def test_the_ratios_and_their_edge_cases(
    tp: int,
    fp: int,
    fn: int,
    precision: float,
    recall: float,
    f1: float,
) -> None:
    got = SyntheticScore(tp, fp, fn, 0, 0, 0)

    assert got.precision == pytest.approx(precision)
    assert got.recall == pytest.approx(recall)
    assert got.f1 == pytest.approx(f1)


def test_pooling_adds_the_counts_and_then_takes_the_ratios() -> None:
    small = SyntheticScore(1, 0, 0, 0, 0, 0)
    large = SyntheticScore(3, 3, 1, 1, 2, 1)

    pooled = pool_scores([small, large])

    assert pooled == SyntheticScore(4, 3, 1, 1, 2, 1)
    assert pooled.precision == pytest.approx(4 / 7)
    assert pooled.f1 != pytest.approx((small.f1 + large.f1) / 2)


def test_pooling_nothing_is_a_perfect_empty_score() -> None:
    pooled = pool_scores([])

    assert pooled == SyntheticScore(0, 0, 0, 0, 0, 0)
    assert pooled.f1 == 1.0


def test_pooling_takes_any_iterable() -> None:
    scores = (SyntheticScore(1, 1, 1, 0, 0, 0) for _ in range(3))

    assert pool_scores(scores).tp == 3
