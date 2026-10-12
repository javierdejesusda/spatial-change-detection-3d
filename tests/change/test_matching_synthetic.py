import dataclasses
import functools

import numpy as np
import pytest
from matching_pipeline import oracle_segmenter, run_pipeline
from visibility_scenes import SEEDS, scene_evidence

from spatial_change.domain import Change, ChangeType
from spatial_change.geometry import rotation_angle_deg
from spatial_change.perception import FakeFeatureEncoder
from spatial_change.synthetic.scene import ObjectStatus, SyntheticObject
from spatial_change.visibility import VisibilityConfig

# Measured over seeds 0 to 2 on the 128x96 scenes with the oracle segmenter,
# the fake color encoder, and the default configs. Every seed gives exactly
# three changes: the chair MOVED, the crate REMOVED, and the drum ADDED.
#
# * The chair's displacement is within 0.0003 m of the distance between its
#   true box centers (0.323, 0.565, and 0.361 m) and its rotation within
#   0.44 degrees of the angle of its motion (0.0, 39.1, and 68.3 degrees).
# * The crate and the drum have a box center within 0.03 m of the true one.
# * The fake encoder gives the crate and the drum nearly the same feature
#   (cosine 0.89 to 0.99), so only the shape keeps them apart: their best
#   registration fits 0.79 or worse, the chair's fits 0.93 or better.
MAX_DISPLACEMENT_ERROR = 0.02
MAX_ROTATION_ERROR = 1.5
MAX_CENTER_ERROR = 0.1
PATCH = 8


@dataclasses.dataclass(frozen=True, eq=False)
class Outcome:
    """The changes of one scene with the true objects behind them."""

    changes: tuple[Change, ...]
    objects: list[SyntheticObject]


@functools.cache
def outcome(seed: int) -> Outcome:
    """Runs votes, candidates, lifting, description, and association.

    Args:
        seed: Scene seed.

    Returns:
        The changes and the objects that changed.
    """
    votes = scene_evidence(seed, VisibilityConfig())
    run = run_pipeline(
        votes, oracle_segmenter(votes.pair), FakeFeatureEncoder(PATCH)
    )
    return Outcome(
        run.changes,
        [
            o
            for o in votes.pair.objects
            if o.status is not ObjectStatus.UNCHANGED
        ],
    )


def _only(changes: tuple[Change, ...], kind: ChangeType) -> Change:
    (change,) = [c for c in changes if c.type is kind]
    return change


@pytest.mark.parametrize("seed", SEEDS)
def test_the_three_changed_objects_give_three_changes(seed: int) -> None:
    result = outcome(seed)

    assert sorted(c.type for c in result.changes) == [
        ChangeType.ADDED,
        ChangeType.MOVED,
        ChangeType.REMOVED,
    ]


@pytest.mark.parametrize("seed", SEEDS)
def test_the_moved_chair_has_its_true_displacement_and_rotation(
    seed: int,
) -> None:
    result = outcome(seed)
    chair = next(o for o in result.objects if o.status is ObjectStatus.MOVED)
    assert chair.box_before is not None and chair.box_after is not None
    assert chair.motion is not None
    moved = _only(result.changes, ChangeType.MOVED)

    truth_distance = float(
        np.linalg.norm(chair.box_after.center - chair.box_before.center)
    )
    truth_angle = rotation_angle_deg(chair.motion.rotation)
    assert moved.displacement_m == pytest.approx(
        truth_distance, abs=MAX_DISPLACEMENT_ERROR
    )
    assert moved.rotation_deg == pytest.approx(
        truth_angle, abs=MAX_ROTATION_ERROR
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_the_removed_and_added_objects_sit_where_they_were(
    seed: int,
) -> None:
    result = outcome(seed)
    crate = next(o for o in result.objects if o.status is ObjectStatus.REMOVED)
    drum = next(o for o in result.objects if o.status is ObjectStatus.ADDED)
    assert crate.box_before is not None and drum.box_after is not None
    removed = _only(result.changes, ChangeType.REMOVED)
    added = _only(result.changes, ChangeType.ADDED)

    assert removed.before_box is not None and added.after_box is not None
    assert (
        np.linalg.norm(
            np.array(removed.before_box.center) - crate.box_before.center
        )
        < MAX_CENTER_ERROR
    )
    assert (
        np.linalg.norm(np.array(added.after_box.center) - drum.box_after.center)
        < MAX_CENTER_ERROR
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_every_change_has_evidence_and_a_confidence(seed: int) -> None:
    for change in outcome(seed).changes:
        counts = change.evidence
        assert counts.consistent + counts.free + counts.occluded > 0
        assert 0.0 < change.confidence <= 1.0
