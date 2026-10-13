import functools
from collections.abc import Sequence

import numpy as np
import pytest
from matching_scenes import (
    EVIDENCE,
    box_surface,
    candidate_for,
    chair,
    chair_pair,
    instance,
    l_surface,
    unit,
)
from numpy.typing import NDArray

from spatial_change.change import (
    ChangeCandidates,
    LiftedObjects,
    MatchConfig,
    associate,
    associate_with_sources,
    matching,
)
from spatial_change.domain import (
    BoxModel,
    Change,
    ChangeType,
    Epoch,
    EvidenceCounts,
    ObjectInstance,
)

RED = unit(1.0, 0.0, 0.0, 0.0)
GREEN = unit(0.0, 1.0, 0.0, 0.0)
# A look that shares half of its direction with RED.
FADED_RED = unit(1.0, 1.0, 0.0, 0.0)
FEW = EvidenceCounts(consistent=0, free=5, occluded=0, unobserved=0)
PLENTY = EvidenceCounts(consistent=0, free=10000, occluded=0, unobserved=0)


def _instance(
    epoch: Epoch,
    points: NDArray[np.float64],
    index: int = 0,
    feature: NDArray[np.float32] | None = RED,
    label: str | None = None,
    label_score: float | None = None,
) -> ObjectInstance:
    return instance(
        points, f"{epoch.value}-{index}", epoch, label, feature, label_score
    )


_before = functools.partial(_instance, Epoch.BEFORE)
_after = functools.partial(_instance, Epoch.AFTER)


def _inputs(
    before: Sequence[ObjectInstance],
    after: Sequence[ObjectInstance],
    evidence: EvidenceCounts = EVIDENCE,
) -> tuple[LiftedObjects, ChangeCandidates]:
    lifted = LiftedObjects(tuple(before), tuple(after))
    candidates = ChangeCandidates(
        tuple(candidate_for(i, evidence) for i in before),
        tuple(candidate_for(i, evidence) for i in after),
    )
    return lifted, candidates


def _associate(
    before: Sequence[ObjectInstance],
    after: Sequence[ObjectInstance],
    config: MatchConfig | None = None,
    evidence: EvidenceCounts = EVIDENCE,
) -> tuple[Change, ...]:
    lifted, candidates = _inputs(before, after, evidence)
    if config is None:
        return associate(lifted, candidates)
    return associate(lifted, candidates, config)


def _center(box: BoxModel) -> NDArray[np.float64]:
    return np.array(box.center)


def test_identical_twins_that_both_changed_pair_with_the_nearer_one() -> None:
    first = chair(seed=1)
    second = chair(2.0, seed=2)
    moved_first = chair(0.15, seed=3)
    moved_second = chair(2.15, seed=4)

    changes = _associate(
        [_before(first, 0), _before(second, 1)],
        [_after(moved_second, 1), _after(moved_first, 0)],
    )

    assert [c.type for c in changes] == [ChangeType.MOVED] * 2
    for change in changes:
        assert change.before_box is not None
        assert change.after_box is not None
        gap = _center(change.after_box) - _center(change.before_box)
        assert np.allclose(gap, [0.15, 0.0, 0.0], atol=0.03)
        assert change.displacement_m == pytest.approx(0.15, abs=0.03)


def test_a_twin_that_moved_is_not_paired_with_one_that_appeared_far() -> None:
    original, moved = chair_pair(0.3)
    newcomer = chair(4.0, 1.0, seed=3)

    changes = _associate(
        [_before(original)], [_after(newcomer, 1), _after(moved, 0)]
    )

    by_type = {c.type: c for c in changes}
    assert sorted(c.type for c in changes) == [
        ChangeType.ADDED,
        ChangeType.MOVED,
    ]
    assert by_type[ChangeType.MOVED].displacement_m == pytest.approx(
        0.3, abs=0.03
    )
    added = by_type[ChangeType.ADDED].after_box
    assert added is not None
    assert _center(added)[0] == pytest.approx(4.0, abs=0.05)


def test_a_small_move_is_found() -> None:
    before, after = chair_pair(0.08)

    (change,) = _associate([_before(before)], [_after(after)])

    assert change.type is ChangeType.MOVED
    assert change.displacement_m == pytest.approx(0.08, abs=0.02)
    assert change.rotation_deg is not None
    assert change.rotation_deg < 5.0


def test_a_turn_in_place_is_a_move_with_no_displacement() -> None:
    before = l_surface((1.0, 2.0, 0.2), seed=1)
    after = l_surface((1.0, 2.0, 0.2), yaw_deg=90.0, seed=2)

    (change,) = _associate([_before(before)], [_after(after)])

    assert change.type is ChangeType.MOVED
    assert change.rotation_deg == pytest.approx(90.0, abs=4.0)
    assert change.displacement_m is not None
    assert change.displacement_m < 0.03


def test_a_pure_translation_of_a_cuboid_has_no_rotation() -> None:
    before, after = chair_pair(0.6, 0.2)

    (change,) = _associate([_before(before)], [_after(after)])

    assert change.type is ChangeType.MOVED
    assert change.rotation_deg is not None
    assert change.rotation_deg < 3.0
    assert change.displacement_m == pytest.approx(np.hypot(0.6, 0.2), abs=0.03)


def test_two_different_objects_swapped_at_one_spot_are_removed_and_added() -> (
    None
):
    cube = box_surface((0.0, 0.0, 0.3), (0.6, 0.6, 0.6), seed=1)
    slab = box_surface((0.0, 0.0, 0.05), (1.2, 0.3, 0.1), seed=2)

    changes = _associate(
        [_before(cube, feature=RED)], [_after(slab, feature=GREEN)]
    )

    assert [c.type for c in changes] == [ChangeType.REMOVED, ChangeType.ADDED]
    removed, added = changes
    assert removed.before_box is not None and removed.after_box is None
    assert added.after_box is not None and added.before_box is None
    assert removed.displacement_m is None
    assert added.rotation_deg is None


def test_objects_of_one_look_but_unlike_shapes_are_not_paired() -> None:
    cube = box_surface((0.0, 0.0, 0.3), (0.6, 0.6, 0.6), seed=1)
    slab = box_surface((0.0, 0.0, 0.05), (1.2, 0.3, 0.1), seed=2)

    changes = _associate(
        [_before(cube, feature=RED)], [_after(slab, feature=RED)]
    )

    assert [c.type for c in changes] == [ChangeType.REMOVED, ChangeType.ADDED]


def test_a_smaller_box_inside_a_larger_one_is_not_the_same_object() -> None:
    stool = box_surface((0.0, 0.0, 0.25), (0.4, 0.4, 0.5), seed=2)

    changes = _associate([_before(chair())], [_after(stool)])

    assert [c.type for c in changes] == [ChangeType.REMOVED, ChangeType.ADDED]


@pytest.mark.parametrize("seed", range(3))
def test_a_thin_slab_of_an_object_seen_after_matches_the_whole(
    seed: int,
) -> None:
    whole = chair(seed=seed + 1)
    moved = chair(x=0.5, seed=seed + 100)
    slab = moved[moved[:, 2] > 0.83]

    (change,) = _associate([_before(whole)], [_after(slab)])

    assert change.type is ChangeType.MOVED
    assert change.rotation_deg is not None
    assert change.rotation_deg < 5.0


def test_a_small_object_that_fits_on_a_large_one_is_not_matched() -> None:
    slab = box_surface((0.0, 0.0, 0.05), (1.2, 0.3, 0.1), seed=1)
    tile = box_surface((-0.45, 0.0, 0.05), (0.3, 0.3, 0.1), seed=2)

    changes = _associate([_before(tile)], [_after(slab)])

    assert [c.type for c in changes] == [ChangeType.REMOVED, ChangeType.ADDED]


def test_a_small_object_that_moved_and_turned_is_moved() -> None:
    size = (0.07, 0.05, 0.06)
    before = box_surface(
        (0.0, 0.0, 0.03), size, seed=1, spacing=0.004, noise=0.0005
    )
    after = box_surface(
        (0.8, 0.6, 0.03),
        size,
        yaw_deg=30.0,
        seed=2,
        spacing=0.004,
        noise=0.0005,
    )

    (change,) = _associate([_before(before)], [_after(after)])

    assert change.type is ChangeType.MOVED
    assert change.rotation_deg == pytest.approx(30.0, abs=5.0)
    assert change.displacement_m == pytest.approx(1.0, abs=0.03)


def test_a_change_of_look_in_place_is_uncertain() -> None:
    before = l_surface((1.0, 2.0, 0.2), seed=1)
    after = l_surface((1.0, 2.0, 0.2), seed=2)

    (change,) = _associate(
        [_before(before, feature=RED)], [_after(after, feature=FADED_RED)]
    )

    assert change.type is ChangeType.UNCERTAIN
    assert change.before_box is not None
    assert change.after_box is not None
    assert change.displacement_m == pytest.approx(0.0, abs=0.03)
    assert change.rotation_deg is None


def test_an_object_seen_from_another_side_is_not_moved() -> None:
    shelf = l_surface((1.0, 2.0, 0.2), seed=1)
    again = l_surface((1.0, 2.0, 0.2), seed=2)
    before = shelf[shelf[:, 0] < 1.24]
    after = again[again[:, 0] > 0.76]

    (change,) = _associate(
        [_before(before, feature=RED)], [_after(after, feature=FADED_RED)]
    )

    assert change.displacement_m is not None
    assert change.displacement_m > 0.05
    assert change.type is ChangeType.UNCERTAIN
    assert change.rotation_deg is None


def test_a_pair_whose_registration_failed_is_uncertain() -> None:
    tiny = np.random.default_rng(0).normal(size=(6, 3)) * 0.05

    (change,) = _associate([_before(tiny)], [_after(tiny + [0.01, 0.0, 0.0])])

    assert change.type is ChangeType.UNCERTAIN
    assert change.rotation_deg is None


def test_a_half_lifted_object_still_matches_the_whole() -> None:
    before = l_surface((0.0, 0.0, 0.2), seed=1)
    moved = l_surface((0.4, 0.0, 0.2), seed=2)
    partial = moved[moved[:, 0] > 0.4]

    (change,) = _associate([_before(before)], [_after(partial)])

    assert change.type is ChangeType.MOVED
    assert change.rotation_deg is not None
    assert change.rotation_deg < 10.0


def test_far_and_dissimilar_objects_stay_unmatched() -> None:
    cube = box_surface((0.0, 0.0, 0.3), (0.6, 0.6, 0.6), seed=1)
    slab = box_surface((5.0, 3.0, 0.05), (1.2, 0.3, 0.1), seed=2)

    changes = _associate(
        [_before(cube, feature=RED)], [_after(slab, feature=GREEN)]
    )

    assert [c.type for c in changes] == [ChangeType.REMOVED, ChangeType.ADDED]


def test_a_pair_over_the_cost_limit_is_not_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    original = matching.register_objects

    def spy(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(matching, "register_objects", spy)
    cube = box_surface((0.0, 0.0, 0.3), (0.6, 0.6, 0.6), seed=1)
    opposite = box_surface((0.0, 0.0, 0.3), (0.6, 0.6, 0.6), seed=2)
    tiny = box_surface((0.0, 0.0, 0.05), (0.1, 0.1, 0.1), seed=3)

    changes = _associate(
        [_before(cube, feature=RED), _before(cube, 1, feature=RED)],
        [_after(opposite, feature=-RED), _after(tiny, 1, feature=RED)],
    )

    assert not calls
    assert (
        sorted(c.type for c in changes)
        == [ChangeType.ADDED] * 2 + [ChangeType.REMOVED] * 2
    )


def test_features_that_are_missing_count_as_unknown() -> None:
    before, after = chair_pair(0.4)

    (change,) = _associate(
        [_before(before, feature=None)], [_after(after, feature=RED)]
    )

    assert change.type is ChangeType.MOVED


def test_two_runs_give_the_same_changes() -> None:
    a = l_surface((0.0, 0.0, 0.2), seed=1)
    b = chair(2.0, 1.0, seed=2)
    a2 = l_surface((0.5, 0.3, 0.2), yaw_deg=60.0, seed=3)
    c = chair(-2.0, 1.0, seed=4)
    before = [_before(a, 0), _before(b, 1, feature=GREEN)]
    after = [_after(a2, 0), _after(c, 1, feature=-GREEN)]

    first = _associate(before, after)
    second = _associate(before, after)

    assert [c.model_dump() for c in first] == [c.model_dump() for c in second]


def test_nothing_lifted_gives_no_changes() -> None:
    assert _associate([], []) == ()


def test_unmatched_sides_are_reported_alone() -> None:
    cube = box_surface((0.0, 0.0, 0.3), (0.6, 0.6, 0.6), seed=1)

    removed = _associate([_before(cube)], [])
    added = _associate([], [_after(cube)])

    assert [c.type for c in removed] == [ChangeType.REMOVED]
    assert [c.type for c in added] == [ChangeType.ADDED]


def test_ids_are_sequential_with_pairs_then_removed_then_added() -> None:
    stayer, moved = chair_pair(0.4)
    gone = box_surface((-3.0, 2.0, 0.3), (0.6, 0.6, 0.6), seed=3)
    new = box_surface((3.0, -2.0, 0.05), (1.2, 0.3, 0.1), seed=4)

    changes = _associate(
        [_before(gone, 1, feature=GREEN), _before(stayer, 0)],
        [_after(new, 1, feature=-GREEN), _after(moved, 0)],
    )

    assert [c.change_id for c in changes] == [f"change-{i}" for i in range(3)]
    assert [c.type for c in changes] == [
        ChangeType.MOVED,
        ChangeType.REMOVED,
        ChangeType.ADDED,
    ]


def test_the_label_comes_from_the_instance() -> None:
    original, moved = chair_pair(0.4)
    gone = box_surface((-3.0, 2.0, 0.3), (0.6, 0.6, 0.6), seed=3)

    named = _associate(
        [_before(original, label="chair")], [_after(moved, label=None)]
    )
    fallback = _associate([_before(gone)], [])
    after_only = _associate(
        [_before(original, label=None)], [_after(moved, label="stool")]
    )

    assert named[0].label == "chair"
    assert fallback[0].label == "object"
    assert after_only[0].label == "stool"
    assert named[0].description == ""


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        (("chair", 0.4), ("sofa", 0.8), "sofa"),
        (("chair", 0.8), ("sofa", 0.4), "chair"),
        (("chair", 0.5), ("sofa", 0.5), "chair"),
        (("chair", None), ("sofa", 0.2), "sofa"),
        (("chair", 0.2), ("sofa", None), "chair"),
        (("chair", None), ("sofa", None), "chair"),
        ((None, None), ("sofa", 0.1), "sofa"),
        (("chair", 0.1), (None, None), "chair"),
        ((None, 0.9), (None, 0.9), "object"),
    ],
)
def test_a_moved_object_takes_the_label_with_the_higher_score(
    before: tuple[str | None, float | None],
    after: tuple[str | None, float | None],
    expected: str,
) -> None:
    original, moved = chair_pair(0.4)

    changes = _associate(
        [_before(original, label=before[0], label_score=before[1])],
        [_after(moved, label=after[0], label_score=after[1])],
    )

    assert changes[0].type is ChangeType.MOVED
    assert changes[0].label == expected


def test_the_evidence_sums_the_candidates_inside_each_object() -> None:
    original, moved = chair_pair(0.4)
    before = _before(original)
    after = _after(moved)
    lifted = LiftedObjects((before,), (after,))
    low = original[original[:, 2] < 0.45]
    high = original[original[:, 2] >= 0.45]
    fragments = tuple(
        candidate_for(
            instance(points, f"before-{k}", Epoch.BEFORE),
            EvidenceCounts(consistent=k, free=10, occluded=1, unobserved=0),
        )
        for k, points in enumerate((low, high), start=1)
    )
    elsewhere = candidate_for(
        instance(
            box_surface((5.0, 5.0, 0.3), (0.6, 0.6, 0.6)),
            "before-9",
            Epoch.BEFORE,
        ),
        EvidenceCounts(consistent=100, free=100, occluded=100, unobserved=100),
    )
    candidates = ChangeCandidates(
        fragments + (elsewhere,),
        (
            candidate_for(
                after,
                EvidenceCounts(consistent=7, free=1, occluded=0, unobserved=2),
            ),
        ),
    )

    (change,) = associate(lifted, candidates)

    assert change.evidence == EvidenceCounts(
        consistent=1 + 2 + 7, free=10 + 10 + 1, occluded=1 + 1, unobserved=2
    )


def test_an_object_with_no_candidate_has_empty_evidence() -> None:
    lifted = LiftedObjects((_before(chair()),), ())

    (change,) = associate(lifted, ChangeCandidates((), ()))

    assert change.evidence == EvidenceCounts(
        consistent=0, free=0, occluded=0, unobserved=0
    )


def test_confidence_is_a_share_and_a_clean_move_beats_a_doubtful_one() -> None:
    original, moved = chair_pair(0.4)
    lone = box_surface((-3.0, 2.0, 0.3), (0.6, 0.6, 0.6), seed=3)
    shelf = l_surface((1.0, 2.0, 0.2), seed=4)
    repainted = l_surface((1.0, 2.0, 0.2), seed=5)

    clean = _associate([_before(original)], [_after(moved)], evidence=PLENTY)[0]
    doubtful = _associate(
        [_before(shelf, feature=RED)],
        [_after(repainted, feature=FADED_RED)],
        evidence=PLENTY,
    )[0]
    removed = _associate([_before(lone)], [], evidence=PLENTY)[0]

    for change in (clean, doubtful, removed):
        assert 0.0 <= change.confidence <= 1.0
    assert clean.confidence > doubtful.confidence
    assert removed.confidence == pytest.approx(0.9, abs=0.01)


def test_a_change_with_little_evidence_has_low_confidence() -> None:
    lone = box_surface((-3.0, 2.0, 0.3), (0.6, 0.6, 0.6), seed=3)
    original, moved = chair_pair(0.4)

    removed = _associate([_before(lone)], [], evidence=FEW)[0]
    pair = _associate([_before(original)], [_after(moved)], evidence=FEW)[0]
    backed = _associate([_before(original)], [_after(moved)], evidence=PLENTY)[
        0
    ]

    assert removed.confidence < 0.1
    assert pair.type is ChangeType.MOVED
    assert pair.confidence < 0.2 * backed.confidence


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("w_appearance", -1.0),
        ("w_shape", float("nan")),
        ("w_distance", float("inf")),
        ("distance_scale", 0.0),
        ("unknown_cost", 1.5),
        ("max_extent_ratio", 0.5),
        ("max_cost", 0.0),
        ("min_fitness", 1.5),
        ("min_displacement_m", -0.1),
        ("min_rotation_deg", 181.0),
        ("candidate_overlap", 0.0),
        ("support_scale", 0.0),
    ],
)
def test_config_rejects_invalid_values(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        MatchConfig(**{field: value})


def test_instances_of_the_wrong_epoch_are_rejected() -> None:
    lifted = LiftedObjects((_after(chair()),), ())

    with pytest.raises(ValueError, match="disappeared"):
        associate(lifted, ChangeCandidates((), ()))


def test_every_change_comes_with_the_objects_it_was_made_from() -> None:
    blue = unit(0.0, 0.0, 1.0, 0.0)
    faded_blue = unit(0.0, 0.0, 1.0, 1.0)
    stayer, moved = chair_pair(0.4)
    in_place = l_surface((6.0, 6.0, 0.2), seed=5)
    again = l_surface((6.0, 6.0, 0.2), seed=6)
    gone = box_surface((-3.0, 2.0, 0.3), (0.6, 0.6, 0.6), seed=3)
    new = box_surface((3.0, -2.0, 0.05), (1.2, 0.3, 0.1), seed=4)
    before = [
        _before(stayer, 0),
        _before(in_place, 1, feature=blue),
        _before(gone, 2, feature=GREEN),
    ]
    after = [
        _after(moved, 0),
        _after(again, 1, feature=faded_blue),
        _after(new, 2, feature=-GREEN),
    ]
    lifted, candidates = _inputs(before, after)

    found = associate_with_sources(lifted, candidates)

    assert tuple(change for change, _ in found) == associate(lifted, candidates)
    assert [change.type for change, _ in found] == [
        ChangeType.MOVED,
        ChangeType.UNCERTAIN,
        ChangeType.REMOVED,
        ChangeType.ADDED,
    ]
    expected = [
        (before[0], after[0]),
        (before[1], after[1]),
        (before[2], None),
        (None, after[2]),
    ]
    for (change, sources), (was, now) in zip(found, expected, strict=True):
        assert sources.before is was
        assert sources.after is now
        for source, box in (
            (sources.before, change.before_box),
            (sources.after, change.after_box),
        ):
            if source is None:
                assert box is None
            else:
                assert box == BoxModel.from_box(source.box)


def test_without_changes_there_are_no_sources() -> None:
    lifted, candidates = _inputs([], [])

    assert associate_with_sources(lifted, candidates) == ()
