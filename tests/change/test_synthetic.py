from dataclasses import dataclass

import numpy as np
import pytest
from visibility_scenes import (
    SEEDS,
    LabelledCloud,
    changed_objects,
    changed_points_of,
    dominant,
    hidden_object_indices,
    majority,
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


def bounds(obj: SyntheticObject) -> tuple[float, float]:
    """Returns the minimum share and the maximum center error of an object."""
    if obj.status == ObjectStatus.MOVED:
        return MIN_MOVED_SHARE, MAX_MOVED_CENTER_ERROR
    return MIN_SOLID_SHARE, MAX_SOLID_CENTER_ERROR


def test_the_scene_has_a_removed_an_added_and_a_moved_object(
    found: SceneCandidates,
) -> None:
    statuses = [o.status for o in changed_objects(found.pair, Epoch.BEFORE)]
    statuses += [o.status for o in changed_objects(found.pair, Epoch.AFTER)]

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
    cloud = found.clouds[epoch]
    for obj in changed_objects(found.pair, epoch):
        candidate, share = dominant(
            cloud,
            found.evidence[epoch].states,
            obj,
            found.candidates[epoch],
        )
        label, _ = majority(cloud, candidate.point_indices)
        min_share, _ = bounds(obj)

        assert label == obj.index, obj.object_id
        assert candidate.epoch == epoch
        assert share >= min_share, obj.object_id


@pytest.mark.parametrize("epoch", list(Epoch))
def test_every_candidate_is_pure(found: SceneCandidates, epoch: Epoch) -> None:
    assert found.candidates[epoch]

    for candidate in found.candidates[epoch]:
        _, purity = majority(found.clouds[epoch], candidate.point_indices)

        assert purity >= MIN_PURITY, candidate.candidate_id


@pytest.mark.parametrize("epoch", list(Epoch))
def test_no_candidate_is_unchanged_structure(
    found: SceneCandidates, epoch: Epoch
) -> None:
    hidden = hidden_object_indices(found.pair)
    changed = {o.index for o in changed_objects(found.pair, epoch)}

    for candidate in found.candidates[epoch]:
        label, _ = majority(found.clouds[epoch], candidate.point_indices)

        assert label != -1, candidate.candidate_id
        assert label not in hidden, candidate.candidate_id
        assert label in changed, candidate.candidate_id


@pytest.mark.parametrize("epoch", list(Epoch))
def test_the_dominant_box_sits_on_the_object(
    found: SceneCandidates, epoch: Epoch
) -> None:
    cloud = found.clouds[epoch]
    states = found.evidence[epoch].states
    for obj in changed_objects(found.pair, epoch):
        candidate, _ = dominant(cloud, states, obj, found.candidates[epoch])
        min_share, max_error = bounds(obj)
        mine = cloud.points[changed_points_of(cloud, states, obj)]
        error = float(
            np.linalg.norm(candidate.box.center - obj.box(epoch).center)
        )

        assert error <= max_error, obj.object_id
        assert candidate.box.contains(mine).mean() >= min_share, obj.object_id
