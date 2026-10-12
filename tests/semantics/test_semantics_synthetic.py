import dataclasses
import functools

import numpy as np
import pytest
from appearance_scenes import reconstruction
from matching_pipeline import oracle_segmenter, run_pipeline
from visibility_scenes import SEEDS, scene_evidence

from spatial_change.change import (
    ChangeCandidates,
    LiftedObjects,
    associate,
    extract_candidates,
)
from spatial_change.domain import Change, ChangeType, Epoch, ObjectInstance
from spatial_change.perception import FakeFeatureEncoder, FakeLabeler
from spatial_change.semantics import (
    FURNITURE,
    Landmark,
    extract_landmarks,
    label_instances,
)
from spatial_change.synthetic.objects import CLASS_COLORS
from spatial_change.synthetic.scene import ObjectStatus, SyntheticObject
from spatial_change.visibility import CrossEvidence, VisibilityConfig

# Measured over seeds 0 to 2 on the 128x96 scenes with the oracle segmenter
# and FakeLabeler over the class colors of the synthetic objects.
#
# * The room shell is gray, as is the shelf, so a labeler that sees hues
#   cannot tell them apart: with the shelf color known, every crop that
#   contains shell labels as a shelf. The changed objects are therefore
#   labeled with the colors of the other classes, and each gets its class
#   (the chair twice, the crate, and the drum).
# * The AFTER frames confirm only the shelf (the occluder), the walls, and,
#   in seed 1, the table among the BEFORE points; the stool, the bottle,
#   the other crate, and the table of seeds 0 and 2 stay UNKNOWN behind the
#   occluder or in the unobserved zone, so they are not landmarks.
# * With the shelf color known the landmarks are the shelf, whose box center
#   lies within 0.2 m of the true one, fragments of the walls (about 1.3 to
#   1.5 m wide, 2.1 m tall, 0.05 m thick) that a hue cannot tell from a
#   shelf, and in seed 1 the table, which the gray around it outvotes.
SHELF_CENTER_TOLERANCE = 0.3
CHANGED_COLORS = {k: v for k, v in CLASS_COLORS.items() if k != "shelf"}


@dataclasses.dataclass(frozen=True, eq=False)
class Outcome:
    """Labels, changes, and landmarks of one scene."""

    labeled: dict[Epoch, tuple[ObjectInstance, ...]]
    changes: tuple[Change, ...]
    landmarks: tuple[Landmark, ...]
    objects: list[SyntheticObject]
    shelf: SyntheticObject


@functools.cache
def outcome(seed: int) -> Outcome:
    """Runs the labeling and the landmark extraction on a scene.

    Args:
        seed: Scene seed.

    Returns:
        The labeled instances, the changes, and the landmarks.
    """
    votes = scene_evidence(seed, VisibilityConfig())
    run = run_pipeline(
        votes, oracle_segmenter(votes.pair), FakeFeatureEncoder(8)
    )
    recs = {
        epoch: reconstruction(
            epoch,
            votes.pair.epoch_data(epoch).frames,
            votes.clouds[epoch].points,
        )
        for epoch in Epoch
    }
    labeler = FakeLabeler(CHANGED_COLORS)
    labeled = {
        Epoch.BEFORE: label_instances(
            recs[Epoch.BEFORE], run.lifted.disappeared, labeler
        ),
        Epoch.AFTER: label_instances(
            recs[Epoch.AFTER], run.lifted.appeared, labeler
        ),
    }
    found = {
        epoch: extract_candidates(
            votes.clouds[epoch].points, votes.evidence[epoch], epoch
        )
        for epoch in Epoch
    }
    changes = associate(
        LiftedObjects(labeled[Epoch.BEFORE], labeled[Epoch.AFTER]),
        ChangeCandidates(found[Epoch.BEFORE], found[Epoch.AFTER]),
    )
    cross = CrossEvidence(
        votes.evidence[Epoch.BEFORE], votes.evidence[Epoch.AFTER]
    )
    landmarks = extract_landmarks(
        recs[Epoch.BEFORE], cross, FakeLabeler(CLASS_COLORS)
    )
    (shelf,) = [o for o in votes.pair.objects if o.label == "shelf"]
    return Outcome(labeled, changes, landmarks, list(votes.pair.objects), shelf)


def _nearest(
    item: ObjectInstance, objects: list[SyntheticObject]
) -> SyntheticObject:
    present = [o for o in objects if o.exists_in(item.epoch)]
    return min(
        present,
        key=lambda o: float(
            np.linalg.norm(item.box.center - o.box(item.epoch).center)
        ),
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_every_changed_instance_gets_its_class_label(seed: int) -> None:
    result = outcome(seed)

    for items in result.labeled.values():
        assert items
        for item in items:
            truth = _nearest(item, result.objects)
            assert truth.status is not ObjectStatus.UNCHANGED
            assert item.label == truth.label
            assert item.label_score is not None
            assert item.label_score >= 0.3


@pytest.mark.parametrize("seed", SEEDS)
def test_the_changes_carry_the_class_labels(seed: int) -> None:
    changes = {c.type: c.label for c in outcome(seed).changes}

    assert changes == {
        ChangeType.MOVED: "chair",
        ChangeType.REMOVED: "crate",
        ChangeType.ADDED: "drum",
    }


@pytest.mark.parametrize("seed", SEEDS)
def test_the_static_shelf_is_a_landmark(seed: int) -> None:
    result = outcome(seed)
    center = result.shelf.box(Epoch.BEFORE).center

    shelves = [
        item
        for item in result.landmarks
        if item.label == "shelf"
        and np.linalg.norm(item.box.center[:2] - center[:2])
        <= SHELF_CENTER_TOLERANCE
    ]

    assert shelves


@pytest.mark.parametrize("seed", SEEDS)
def test_landmarks_are_furniture_and_never_a_changed_object(seed: int) -> None:
    result = outcome(seed)
    changed = [
        o
        for o in result.objects
        if o.status is not ObjectStatus.UNCHANGED and o.exists_in(Epoch.BEFORE)
    ]

    assert result.landmarks
    assert [i.landmark_id for i in result.landmarks] == [
        f"landmark-{k}" for k in range(len(result.landmarks))
    ]
    for item in result.landmarks:
        assert item.label in FURNITURE
        for obj in changed:
            gap = np.linalg.norm(
                item.box.center[:2] - obj.box(Epoch.BEFORE).center[:2]
            )
            assert gap > 0.5, (item.landmark_id, obj.object_id)
