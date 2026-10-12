import dataclasses
import functools

import lifting_metrics
import numpy as np
import pytest
from appearance_scenes import reconstruction
from lifting_metrics import purity_and_recall
from lifting_scenes import OracleSegmenter
from visibility_scenes import (
    SEEDS,
    LabelledCloud,
    changed_objects,
    scene_evidence,
)

from spatial_change.change import (
    Candidate,
    LiftConfig,
    extract_candidates,
    lift_candidates,
)
from spatial_change.domain import Epoch, ObjectInstance
from spatial_change.perception import FakeMaskSegmenter
from spatial_change.ports import MaskSegmenter
from spatial_change.synthetic.scene import SyntheticObject
from spatial_change.visibility import VisibilityConfig

# Measured over seeds 0 to 2 on the 128x96 scenes with the default
# LiftConfig and the candidates of the default CandidateConfig.
#
# With the oracle segmenter (the exact instance under the prompt point) the
# 2 to 6 candidates of an epoch merge into one instance per changed object
# (the moved chair, the removed crate, the added drum). Every instance has
# purity 1.000 and recall of at least 0.933, and its box center lies at most
# 0.030 m from the true one.
#
# With FakeMaskSegmenter, whose rectangles include the background around an
# object, most of that background is floor, which the floor clearance
# removes: the mean share of an instance's points on its candidate's object
# is 0.944 at min_agreement 0, 0.5, and 0.8 alike (it was 0.332, 0.390, and
# 0.433 before the clearance), and the mean recall of that object is 0.965,
# 0.964, and 0.964. The remaining background is not floor and a rectangle
# that covers it in one view usually covers it in the others too, so
# consensus recovers little.
MIN_ORACLE_PURITY = 0.99
MIN_ORACLE_RECALL = 0.9
MAX_ORACLE_CENTER_ERROR = 0.1
MIN_RECTANGLE_RECALL = 0.75
MIN_RECTANGLE_MEAN_RECALL = 0.9
MIN_STRICT_GAIN = 0.0


@dataclasses.dataclass(frozen=True, eq=False)
class EpochLift:
    """The inputs and outputs of lifting one epoch of a scene."""

    seed: int
    epoch: Epoch
    cloud: LabelledCloud
    candidates: tuple[Candidate, ...]
    instances: tuple[ObjectInstance, ...]
    segmenter: MaskSegmenter
    objects: list[SyntheticObject]


@functools.cache
def lifted(seed: int, kind: str, min_agreement: float = 0.5) -> list[EpochLift]:
    """Lifts the candidates of both epochs of a scene.

    Args:
        seed: Scene seed.
        kind: ``"oracle"`` or ``"rectangle"``.
        min_agreement: The consensus threshold.

    Returns:
        One result per epoch.
    """
    votes = scene_evidence(seed, VisibilityConfig())
    results = []
    for epoch in Epoch:
        cloud = votes.clouds[epoch]
        data = votes.pair.epoch_data(epoch)
        candidates = extract_candidates(
            cloud.points, votes.evidence[epoch], epoch
        )
        segmenter: MaskSegmenter
        if kind == "oracle":
            segmenter = OracleSegmenter(
                {
                    id(f.image): m
                    for f, m in zip(
                        data.frames, data.instance_maps, strict=True
                    )
                }
            )
        else:
            segmenter = FakeMaskSegmenter()
        instances = lift_candidates(
            reconstruction(epoch, data.frames, cloud.points),
            candidates,
            segmenter,
            LiftConfig(min_agreement=min_agreement),
        )
        results.append(
            EpochLift(
                seed,
                epoch,
                cloud,
                candidates,
                instances,
                segmenter,
                changed_objects(votes.pair, epoch),
            )
        )
    return results


def target_share(
    item: EpochLift, instance: ObjectInstance
) -> tuple[float, float]:
    """Gives the purity and recall of an instance for its candidate's object."""
    return purity_and_recall(
        item.cloud, item.candidates, instance, LiftConfig().voxel_size
    )


def candidate_object(item: EpochLift, instance: ObjectInstance) -> int:
    """Gives the object that an instance's first candidate mostly covers."""
    return lifting_metrics.candidate_object(
        item.cloud, item.candidates, instance
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("epoch", list(Epoch))
def test_the_oracle_gives_one_instance_per_changed_object(
    seed: int, epoch: Epoch
) -> None:
    (item,) = [i for i in lifted(seed, "oracle") if i.epoch == epoch]

    objects = sorted(candidate_object(item, i) for i in item.instances)

    assert objects == sorted(o.index for o in item.objects)
    assert len(item.candidates) >= len(item.instances)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("epoch", list(Epoch))
def test_the_oracle_instances_are_pure_complete_and_well_placed(
    seed: int, epoch: Epoch
) -> None:
    (item,) = [i for i in lifted(seed, "oracle") if i.epoch == epoch]

    for instance in item.instances:
        index = candidate_object(item, instance)
        obj = next(o for o in item.objects if o.index == index)
        purity, recall = target_share(item, instance)
        error = float(
            np.linalg.norm(instance.box.center - obj.box(epoch).center)
        )

        assert purity >= MIN_ORACLE_PURITY, obj.object_id
        assert recall >= MIN_ORACLE_RECALL, obj.object_id
        assert error <= MAX_ORACLE_CENTER_ERROR, obj.object_id


@pytest.mark.parametrize("seed", SEEDS)
def test_each_frame_is_segmented_at_most_once(seed: int) -> None:
    for item in lifted(seed, "oracle"):
        assert isinstance(item.segmenter, OracleSegmenter)
        images = [id(image) for image, _, _ in item.segmenter.calls]

        assert images
        assert len(images) == len(set(images))


def _mean(values: list[float]) -> float:
    return float(np.mean(values))


def _rectangle_scores(min_agreement: float) -> list[tuple[float, float]]:
    return [
        target_share(item, instance)
        for seed in SEEDS
        for item in lifted(seed, "rectangle", min_agreement)
        for instance in item.instances
    ]


def test_consensus_does_not_lower_the_purity_of_rectangle_masks() -> None:
    without = _rectangle_scores(0.0)
    default = _rectangle_scores(0.5)

    assert _mean([p for p, _ in default]) >= _mean([p for p, _ in without])
    assert min(r for _, r in default) >= MIN_RECTANGLE_RECALL
    assert _mean([r for _, r in default]) >= MIN_RECTANGLE_MEAN_RECALL


def test_a_stricter_consensus_does_not_lower_purity() -> None:
    without = _rectangle_scores(0.0)
    strict = _rectangle_scores(0.8)

    gain = _mean([p for p, _ in strict]) - _mean([p for p, _ in without])
    assert gain >= MIN_STRICT_GAIN
