from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray
from visibility_scenes import (
    SEEDS,
    LabelledCloud,
    hidden_object_indices,
    scene_evidence,
)

from spatial_change.domain import Epoch
from spatial_change.synthetic.scene import (
    ObjectStatus,
    SyntheticObject,
    SyntheticScenePair,
)
from spatial_change.visibility import (
    PointEvidence,
    PointState,
    VisibilityConfig,
)

SENSITIVE = VisibilityConfig(min_free_votes=1)

# Rates measured over seeds 0 to 2 (see the assertions below): free votes
# on changed objects at least 0.544, CHANGED at least 0.472, PRESENT on
# changed objects at most 0.345 (contact with the floor and the overlap of
# a short move), PRESENT among observed static points at least 0.709, and
# zero false CHANGED on static structure, also at min_free_votes=1.
MIN_FREE_RATE = 0.45
MIN_CHANGED_RATE = 0.35
MAX_STATIC_CHANGED_RATE = 0.002
MIN_STATIC_PRESENT_RATE = 0.6
MAX_CHANGED_PRESENT_RATE = 0.45


@dataclass(frozen=True, eq=False)
class SceneVotes:
    """Per epoch: the cloud and its votes from the other epoch's frames."""

    pair: SyntheticScenePair
    clouds: dict[Epoch, LabelledCloud]
    evidence: dict[Epoch, PointEvidence]
    sensitive: dict[Epoch, PointEvidence]


@pytest.fixture(scope="module", params=SEEDS, ids=[f"seed{s}" for s in SEEDS])
def votes(request: pytest.FixtureRequest) -> SceneVotes:
    default = scene_evidence(request.param, VisibilityConfig())
    sensitive = scene_evidence(request.param, SENSITIVE)
    return SceneVotes(
        default.pair, default.clouds, default.evidence, sensitive.evidence
    )


def of_status(
    pair: SyntheticScenePair, status: ObjectStatus
) -> SyntheticObject:
    (found,) = [o for o in pair.objects if o.status == status]
    return found


def rate(mask: NDArray[np.bool_], selected: NDArray[np.bool_]) -> float:
    assert selected.any()
    return float(mask[selected].mean())


def changed_object_cases(
    votes: SceneVotes,
) -> list[tuple[str, NDArray[np.bool_], PointEvidence]]:
    """Selections of changed-object points with their default votes."""
    pair = votes.pair
    removed = of_status(pair, ObjectStatus.REMOVED).index
    added = of_status(pair, ObjectStatus.ADDED).index
    moved = of_status(pair, ObjectStatus.MOVED).index
    before, after = Epoch.BEFORE, Epoch.AFTER
    clouds, evidence = votes.clouds, votes.evidence
    return [
        ("removed", clouds[before].labels == removed, evidence[before]),
        ("added", clouds[after].labels == added, evidence[after]),
        (
            "moved, before pose",
            clouds[before].labels == moved,
            evidence[before],
        ),
        ("moved, after pose", clouds[after].labels == moved, evidence[after]),
    ]


def test_changed_objects_collect_free_votes(votes: SceneVotes) -> None:
    for name, selected, evidence in changed_object_cases(votes):
        assert rate(evidence.free > 0, selected) >= MIN_FREE_RATE, name


def test_changed_objects_are_changed(votes: SceneVotes) -> None:
    for name, selected, evidence in changed_object_cases(votes):
        changed = rate(evidence.states == PointState.CHANGED, selected)
        assert changed >= MIN_CHANGED_RATE, name


def test_a_changed_object_is_not_confirmed_present(votes: SceneVotes) -> None:
    for name, selected, evidence in changed_object_cases(votes):
        present = rate(evidence.states == PointState.PRESENT, selected)
        assert present <= MAX_CHANGED_PRESENT_RATE, name


def test_the_occluded_object_collects_no_free_votes(votes: SceneVotes) -> None:
    ids = votes.pair.occluded_object_ids
    assert ids
    selected = np.isin(
        votes.clouds[Epoch.BEFORE].labels,
        [votes.pair.object_by_id(i).index for i in ids],
    )
    assert selected.any()

    for evidence in (
        votes.evidence[Epoch.BEFORE],
        votes.sensitive[Epoch.BEFORE],
    ):
        assert evidence.free[selected].sum() == 0
        assert evidence.occluded[selected].sum() > 0
        assert not np.any(evidence.states[selected] == PointState.CHANGED)


def test_the_unobserved_zone_is_only_unobserved(votes: SceneVotes) -> None:
    selected = votes.pair.unobserved_zone.contains(
        votes.clouds[Epoch.BEFORE].points
    )
    assert selected.any()

    for evidence in (
        votes.evidence[Epoch.BEFORE],
        votes.sensitive[Epoch.BEFORE],
    ):
        all_votes = evidence.counts.sum(axis=1)
        assert np.all(evidence.unobserved[selected] == all_votes[selected])
        assert np.all(evidence.states[selected] == PointState.UNKNOWN)


def static_selection(votes: SceneVotes, epoch: Epoch) -> NDArray[np.bool_]:
    """Room shell and unchanged objects, minus the occluded and zone ones."""
    pair = votes.pair
    cloud = votes.clouds[epoch]
    special = hidden_object_indices(pair)
    unchanged = [
        o.index
        for o in pair.objects
        if o.status == ObjectStatus.UNCHANGED and o.index not in special
    ]
    selected = (cloud.labels == -1) | np.isin(cloud.labels, unchanged)
    return selected & ~pair.unobserved_zone.contains(cloud.points)


@pytest.mark.parametrize("epoch", list(Epoch))
def test_static_structure_is_rarely_changed_and_mostly_present(
    votes: SceneVotes, epoch: Epoch
) -> None:
    selected = static_selection(votes, epoch)
    evidence = votes.evidence[epoch]

    for checked in (evidence, votes.sensitive[epoch]):
        assert rate(checked.states == PointState.CHANGED, selected) <= (
            MAX_STATIC_CHANGED_RATE
        )
    observed = evidence.unobserved < evidence.counts.sum(axis=1)
    assert rate(evidence.states == PointState.PRESENT, selected & observed) >= (
        MIN_STATIC_PRESENT_RATE
    )
