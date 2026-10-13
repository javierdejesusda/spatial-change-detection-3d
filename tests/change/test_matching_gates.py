import functools
import math
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pytest
from matching_scenes import (
    UNGATED,
    box_surface,
    candidate_for,
    chair_pair,
    instance,
    l_surface,
    unit,
)
from numpy.typing import NDArray

from spatial_change.change import (
    ChangeCandidates,
    ChangeSources,
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
BLUE = unit(0.0, 0.0, 1.0, 0.0)
FADED_BLUE = unit(0.0, 0.0, 1.0, 1.0)
# The confidence a change that the matcher abstains from never exceeds.
CEILING = 0.5

Side = tuple[ObjectInstance, EvidenceCounts]

# Where the chair, the box, and the slab of ``_scene`` are in the changes.
CHAIR, BOX, SLAB = 0, 1, 2


def _votes(
    free: int = 10000,
    occluded: int = 0,
    consistent: int = 0,
    unobserved: int = 0,
) -> EvidenceCounts:
    return EvidenceCounts(
        consistent=consistent,
        free=free,
        occluded=occluded,
        unobserved=unobserved,
    )


SEEN = _votes()


def _instance(
    epoch: Epoch,
    points: NDArray[np.float64],
    index: int,
    feature: NDArray[np.float32],
) -> ObjectInstance:
    return instance(points, f"{epoch.value}-{index}", epoch, None, feature)


_before = functools.partial(_instance, Epoch.BEFORE)
_after = functools.partial(_instance, Epoch.AFTER)


def _gone(
    extent: tuple[float, float, float] = (0.6, 0.6, 0.6),
) -> NDArray[np.float64]:
    return box_surface((-3.0, 2.0, extent[2] / 2.0), extent, seed=3)


def _scene(
    moved: EvidenceCounts = SEEN,
    removed: EvidenceCounts = SEEN,
    added: EvidenceCounts = SEEN,
    removed_extent: tuple[float, float, float] = (0.6, 0.6, 0.6),
) -> tuple[list[Side], list[Side]]:
    """Builds a moved chair, a removed box, and an added slab.

    The changes come out in that order: ``MOVED``, ``REMOVED``, ``ADDED``.
    """
    original, shifted = chair_pair(0.4)
    new = box_surface((3.0, -2.0, 0.05), (1.2, 0.3, 0.1), seed=4)
    before = [
        (_before(original, 0, RED), moved),
        (_before(_gone(removed_extent), 1, GREEN), removed),
    ]
    after = [(_after(shifted, 0, RED), moved), (_after(new, 1, -GREEN), added)]
    return before, after


def _inputs(
    scene: tuple[Sequence[Side], Sequence[Side]],
) -> tuple[LiftedObjects, ChangeCandidates]:
    before, after = scene
    lifted = LiftedObjects(
        tuple(item for item, _ in before), tuple(item for item, _ in after)
    )
    candidates = ChangeCandidates(
        tuple(candidate_for(item, votes) for item, votes in before),
        tuple(candidate_for(item, votes) for item, votes in after),
    )
    return lifted, candidates


def _run(
    scene: tuple[Sequence[Side], Sequence[Side]],
    config: MatchConfig = UNGATED,
) -> tuple[Change, ...]:
    lifted, candidates = _inputs(scene)
    return associate(lifted, candidates, config)


def _boxed(extent: tuple[float, float, float] | None) -> BoxModel | None:
    if extent is None:
        return None
    return BoxModel(
        center=(0.0, 0.0, 0.0),
        extent=extent,
        rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )


def _change(
    kind: ChangeType,
    before: tuple[float, float, float] | None = None,
    after: tuple[float, float, float] | None = None,
    confidence: float = 0.9,
) -> Change:
    """Builds a change whose boxes have the given extents."""
    return Change(
        change_id="change-0",
        type=kind,
        label="object",
        description="",
        confidence=confidence,
        before_box=_boxed(before),
        after_box=_boxed(after),
        displacement_m=None,
        rotation_deg=None,
        evidence=SEEN,
    )


def _gated(change: Change, config: MatchConfig) -> Change:
    """Runs the gates on a change that the matcher did not build."""
    ((gated, _),) = matching._apply_gates(
        [(change, ChangeSources(None, None))], config
    )
    return gated


def _assert_demoted(gated: Change, base: Change) -> None:
    assert gated.type is ChangeType.UNCERTAIN
    assert gated.confidence == pytest.approx(min(base.confidence, CEILING))
    restored = gated.model_copy(
        update={"type": base.type, "confidence": base.confidence}
    )
    assert restored == base


def _assert_only_the_box_is_demoted(
    gated: tuple[Change, ...], base: tuple[Change, ...]
) -> None:
    assert gated[CHAIR] == base[CHAIR]
    assert gated[SLAB] == base[SLAB]
    _assert_demoted(gated[BOX], base[BOX])


def test_the_occluded_and_volume_gates_are_on_by_default() -> None:
    config = MatchConfig()

    assert config.max_occluded_fraction == 0.001
    assert config.max_box_volume_m3 == 1.0
    assert config.min_change_confidence is None


def test_the_default_gates_demote_a_change_and_none_turns_them_off() -> None:
    scene = _scene(
        removed=_votes(free=9000, occluded=1000),
        added=_votes(free=9000, occluded=1000),
        removed_extent=(2.0, 1.0, 2.0),
    )

    base = _run(scene)
    gated = _run(scene, MatchConfig())

    assert [c.type for c in base] == [
        ChangeType.MOVED,
        ChangeType.REMOVED,
        ChangeType.ADDED,
    ]
    assert gated[CHAIR] == base[CHAIR]
    _assert_demoted(gated[BOX], base[BOX])
    _assert_demoted(gated[SLAB], base[SLAB])
    only_volume = _run(scene, MatchConfig(max_occluded_fraction=None))
    only_share = _run(scene, MatchConfig(max_box_volume_m3=None))
    _assert_demoted(only_volume[BOX], base[BOX])
    assert only_volume[SLAB] == base[SLAB]
    _assert_demoted(only_share[BOX], base[BOX])
    _assert_demoted(only_share[SLAB], base[SLAB])


def test_a_change_with_too_many_occluded_votes_is_demoted() -> None:
    scene = _scene(
        removed=_votes(free=6000, occluded=4000),
        added=_votes(free=9000, occluded=1000),
    )
    base = _run(scene)

    gated = _run(scene, replace(UNGATED, max_occluded_fraction=0.3))

    assert [c.type for c in base] == [
        ChangeType.MOVED,
        ChangeType.REMOVED,
        ChangeType.ADDED,
    ]
    # Above the ceiling, so the demotion caps the confidence.
    assert base[BOX].confidence > CEILING
    _assert_only_the_box_is_demoted(gated, base)


@pytest.mark.parametrize(
    ("votes", "limit", "demoted"),
    [
        (_votes(free=6000, occluded=4000), 0.4, False),
        (_votes(free=6000, occluded=4000), 0.39, True),
        (_votes(free=0, consistent=10000, occluded=5000), 0.5, False),
        (_votes(free=0, consistent=10000, occluded=5000), 0.3, True),
        (_votes(free=9999, occluded=1, unobserved=100000), 5e-5, True),
        (_votes(free=0, occluded=0, unobserved=500), 0.0, False),
        (_votes(free=10000, occluded=1), 0.0, True),
    ],
)
def test_the_occluded_share_counts_the_votes_that_saw_the_location(
    votes: EvidenceCounts, limit: float, demoted: bool
) -> None:
    scene = ([(_before(_gone(), 0, GREEN), votes)], [])

    (change,) = _run(scene, replace(UNGATED, max_occluded_fraction=limit))

    expected = ChangeType.UNCERTAIN if demoted else ChangeType.REMOVED
    assert change.type is expected


def test_a_change_with_a_very_large_box_is_demoted() -> None:
    # As built, the chair is 0.18 m3, the slab 0.036 m3, and the wall 0.8 m3.
    scene = _scene(removed_extent=(2.0, 0.2, 2.0))
    base = _run(scene)

    gated = _run(scene, replace(UNGATED, max_box_volume_m3=0.3))

    _assert_only_the_box_is_demoted(gated, base)


def test_a_volume_equal_to_the_limit_is_kept() -> None:
    removed = _change(ChangeType.REMOVED, before=(1.0, 2.0, 0.5))

    kept = _gated(removed, replace(UNGATED, max_box_volume_m3=1.0))
    over = _gated(
        removed, replace(UNGATED, max_box_volume_m3=math.nextafter(1.0, 0.0))
    )

    assert kept == removed
    assert over.type is ChangeType.UNCERTAIN


def test_a_change_without_a_box_has_no_volume() -> None:
    boxless = _change(ChangeType.UNCERTAIN, confidence=0.3)

    assert _gated(boxless, replace(UNGATED, max_box_volume_m3=0.1)) == boxless


@pytest.mark.parametrize("bigger_before", [True, False])
def test_the_larger_of_the_two_boxes_of_a_pair_decides_the_volume(
    bigger_before: bool,
) -> None:
    whole = l_surface((0.0, 0.0, 0.2), seed=1)
    moved = l_surface((0.4, 0.0, 0.2), seed=2)
    if bigger_before:
        before, after = whole, moved[moved[:, 0] > 0.4]
    else:
        before, after = whole[whole[:, 0] > 0.0], moved
    scene = (
        [(_before(before, 0, RED), SEEN)],
        [(_after(after, 0, RED), SEEN)],
    )
    (base,) = _run(scene)
    assert base.type is ChangeType.MOVED

    # The whole L is about 0.15 m3 and its half about 0.03 m3.
    kept = _gated(base, replace(UNGATED, max_box_volume_m3=0.2))
    gated = _gated(base, replace(UNGATED, max_box_volume_m3=0.1))

    assert kept == base
    _assert_demoted(gated, base)


def test_a_change_with_a_low_confidence_is_demoted() -> None:
    scene = _scene(removed=_votes(free=5))
    base = _run(scene)
    assert base[BOX].confidence < 0.1

    gated = _run(scene, replace(UNGATED, min_change_confidence=0.1))

    _assert_only_the_box_is_demoted(gated, base)


def test_a_confidence_equal_to_the_limit_is_kept() -> None:
    scene = _scene(removed=_votes(free=5))
    base = _run(scene)
    limit = base[BOX].confidence

    kept = _gated(base[BOX], replace(UNGATED, min_change_confidence=limit))
    under = _gated(
        base[BOX],
        replace(UNGATED, min_change_confidence=math.nextafter(limit, 1.0)),
    )

    assert kept == base[BOX]
    assert under.type is ChangeType.UNCERTAIN


def test_a_demoted_move_keeps_its_boxes_and_its_motion() -> None:
    scene = _scene(moved=_votes(free=5000, occluded=5000))
    base = _run(scene)

    gated = _run(scene, replace(UNGATED, max_occluded_fraction=0.3))

    assert base[CHAIR].type is ChangeType.MOVED
    assert gated[CHAIR].before_box is not None
    assert gated[CHAIR].after_box is not None
    assert gated[CHAIR].displacement_m is not None
    assert gated[CHAIR].rotation_deg is not None
    _assert_demoted(gated[CHAIR], base[CHAIR])
    assert gated[BOX] == base[BOX]
    assert gated[SLAB] == base[SLAB]


def test_a_change_that_trips_several_gates_is_demoted_once() -> None:
    # The wall is occluded by a third of its votes, 0.8 m3, and at a
    # confidence of 0.78, so it is past all three gates.
    scene = _scene(
        removed=_votes(free=1000, occluded=500), removed_extent=(2.0, 0.2, 2.0)
    )
    base = _run(scene)
    assert CEILING < base[BOX].confidence < 0.8
    once = base[BOX].model_copy(
        update={"type": ChangeType.UNCERTAIN, "confidence": CEILING}
    )
    gates = [
        replace(UNGATED, max_occluded_fraction=0.3),
        replace(UNGATED, max_box_volume_m3=0.3),
        replace(UNGATED, min_change_confidence=0.8),
    ]
    together = replace(
        UNGATED,
        max_occluded_fraction=0.3,
        max_box_volume_m3=0.3,
        min_change_confidence=0.8,
    )

    for config in gates:
        assert _gated(base[BOX], config) == once
    gated = _run(scene, together)
    assert gated[BOX] == once
    assert gated[CHAIR] == base[CHAIR]
    assert gated[SLAB] == base[SLAB]


def test_the_gates_leave_the_pairs_the_ids_and_the_sources_alone() -> None:
    shelf = l_surface((6.0, 6.0, 0.2), seed=5)
    repainted = l_surface((6.0, 6.0, 0.2), seed=6)
    before, after = _scene()
    before.append((_before(shelf, 2, BLUE), SEEN))
    after.append((_after(repainted, 2, FADED_BLUE), SEEN))
    lifted, candidates = _inputs((before, after))
    # Every confidence is below 1, so this demotes every change.
    demote_all = replace(UNGATED, min_change_confidence=1.0)

    base = associate_with_sources(lifted, candidates, UNGATED)
    gated = associate_with_sources(lifted, candidates, demote_all)

    assert [c.type for c, _ in base] == [
        ChangeType.MOVED,
        ChangeType.UNCERTAIN,
        ChangeType.REMOVED,
        ChangeType.ADDED,
    ]
    assert [c.type for c, _ in gated] == [ChangeType.UNCERTAIN] * 4
    assert [c.change_id for c, _ in gated] == [c.change_id for c, _ in base]
    for (_, gated_sources), (_, base_sources) in zip(gated, base, strict=True):
        assert gated_sources.before is base_sources.before
        assert gated_sources.after is base_sources.after
    assert gated[1][0] == base[1][0]
    for position in (0, 2, 3):
        _assert_demoted(gated[position][0], base[position][0])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_occluded_fraction", 0.0),
        ("max_occluded_fraction", 1.0),
        ("max_box_volume_m3", 1e-9),
        ("min_change_confidence", 0.0),
        ("min_change_confidence", 1.0),
    ],
)
def test_config_accepts_the_edges_of_the_gates(
    field: str, value: float
) -> None:
    assert getattr(MatchConfig(**{field: value}), field) == value
