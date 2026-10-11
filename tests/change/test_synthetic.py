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

from spatial_change.change import Candidate, extract_candidates
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

# Measured over seeds 0 to 2 with the default CandidateConfig, with seeds 3
# to 9 in parentheses. Every candidate is pure: the majority label holds all
# of its points (1.000, 1.000), and no candidate is room shell or an
# unchanged object. The candidate holding most of a changed object's CHANGED
# points has that object as its majority label. Its share of those points,
# and the share of them inside its box, is at least 1.000 (0.946) for a
# removed or added object and at least 0.644 (0.553) for the moved chair,
# whose legs and back split into several fragments. The distance from its
# box center to the true box center is at most 0.073 m (0.232 m) for a
# removed or added object and 0.213 m (0.290 m) for the moved chair.
MIN_PURITY = 0.95
MIN_SOLID_SHARE = 0.85
MAX_SOLID_CENTER_ERROR = 0.25
MIN_MOVED_SHARE = 0.5
MAX_MOVED_CENTER_ERROR = 0.3


@dataclass(frozen=True, eq=False)
class SceneCandidates:
    """Per epoch: the cloud, its evidence, and its default candidates."""

    pair: SyntheticScenePair
    clouds: dict[Epoch, LabelledCloud]
    evidence: dict[Epoch, PointEvidence]
    candidates: dict[Epoch, tuple[Candidate, ...]]


@pytest.fixture(scope="module", params=SEEDS, ids=[f"seed{s}" for s in SEEDS])
def found(request: pytest.FixtureRequest) -> SceneCandidates:
    votes = scene_evidence(request.param, VisibilityConfig())
    candidates = {
        epoch: extract_candidates(
            votes.clouds[epoch].points, votes.evidence[epoch], epoch
        )
        for epoch in Epoch
    }
    return SceneCandidates(votes.pair, votes.clouds, votes.evidence, candidates)


def changed_objects(
    found: SceneCandidates, epoch: Epoch
) -> list[SyntheticObject]:
    """Lists the changed objects that exist in an epoch."""
    return [
        o
        for o in found.pair.objects
        if o.status != ObjectStatus.UNCHANGED and o.exists_in(epoch)
    ]


def majority(
    found: SceneCandidates, epoch: Epoch, candidate: Candidate
) -> tuple[int, float]:
    """Returns the commonest label of a candidate and its share."""
    labels = found.clouds[epoch].labels[candidate.point_indices]
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.argmax(counts)]), float(counts.max() / counts.sum())


def changed_points_of(
    found: SceneCandidates, epoch: Epoch, obj: SyntheticObject
) -> NDArray[np.bool_]:
    cloud = found.clouds[epoch]
    return (found.evidence[epoch].states == PointState.CHANGED) & (
        cloud.labels == obj.index
    )


def dominant(
    found: SceneCandidates, epoch: Epoch, obj: SyntheticObject
) -> tuple[Candidate, float]:
    """Returns the candidate with most of the object's CHANGED points."""
    mine = changed_points_of(found, epoch, obj)
    assert mine.any(), obj.object_id
    assert found.candidates[epoch], f"no candidates in {epoch.value}"
    held = [int(mine[c.point_indices].sum()) for c in found.candidates[epoch]]
    best = int(np.argmax(held))
    return found.candidates[epoch][best], held[best] / int(mine.sum())


def bounds(obj: SyntheticObject) -> tuple[float, float]:
    """Returns the minimum share and the maximum center error of an object."""
    if obj.status == ObjectStatus.MOVED:
        return MIN_MOVED_SHARE, MAX_MOVED_CENTER_ERROR
    return MIN_SOLID_SHARE, MAX_SOLID_CENTER_ERROR


def test_the_scene_has_a_removed_an_added_and_a_moved_object(
    found: SceneCandidates,
) -> None:
    statuses = [o.status for o in changed_objects(found, Epoch.BEFORE)]
    statuses += [o.status for o in changed_objects(found, Epoch.AFTER)]

    assert sorted(statuses) == sorted(
        [
            ObjectStatus.REMOVED,
            ObjectStatus.MOVED,
            ObjectStatus.MOVED,
            ObjectStatus.ADDED,
        ]
    )


@pytest.mark.parametrize("epoch", list(Epoch))
def test_every_changed_object_has_a_dominant_candidate(
    found: SceneCandidates, epoch: Epoch
) -> None:
    for obj in changed_objects(found, epoch):
        candidate, share = dominant(found, epoch, obj)
        label, _ = majority(found, epoch, candidate)
        min_share, _ = bounds(obj)

        assert label == obj.index, obj.object_id
        assert candidate.epoch == epoch
        assert share >= min_share, obj.object_id


@pytest.mark.parametrize("epoch", list(Epoch))
def test_every_candidate_is_pure(found: SceneCandidates, epoch: Epoch) -> None:
    assert found.candidates[epoch]

    for candidate in found.candidates[epoch]:
        _, purity = majority(found, epoch, candidate)

        assert purity >= MIN_PURITY, candidate.candidate_id


@pytest.mark.parametrize("epoch", list(Epoch))
def test_no_candidate_is_unchanged_structure(
    found: SceneCandidates, epoch: Epoch
) -> None:
    hidden = hidden_object_indices(found.pair)
    changed = {o.index for o in changed_objects(found, epoch)}

    for candidate in found.candidates[epoch]:
        label, _ = majority(found, epoch, candidate)

        assert label != -1, candidate.candidate_id
        assert label not in hidden, candidate.candidate_id
        assert label in changed, candidate.candidate_id


@pytest.mark.parametrize("epoch", list(Epoch))
def test_the_dominant_box_sits_on_the_object(
    found: SceneCandidates, epoch: Epoch
) -> None:
    cloud = found.clouds[epoch]
    for obj in changed_objects(found, epoch):
        candidate, _ = dominant(found, epoch, obj)
        min_share, max_error = bounds(obj)
        mine = cloud.points[changed_points_of(found, epoch, obj)]
        error = float(
            np.linalg.norm(candidate.box.center - obj.box(epoch).center)
        )

        assert error <= max_error, obj.object_id
        assert candidate.box.contains(mine).mean() >= min_share, obj.object_id
