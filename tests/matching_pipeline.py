"""The association pipeline on a rendered scene pair, for the matching tests.

The synthetic test runs it with the oracle segmenter and the fake encoder,
the GPU test with the real models.
"""

import time
from dataclasses import dataclass

import numpy as np
from appearance_scenes import reconstruction
from lifting_scenes import OracleSegmenter
from matching_scenes import UNGATED
from numpy.typing import NDArray
from visibility_scenes import SceneEvidence

from spatial_change.change import (
    ChangeCandidates,
    LiftedObjects,
    associate,
    describe_instances,
    extract_candidates,
    lift_change_candidates,
)
from spatial_change.domain import Change, Epoch, EpochReconstruction
from spatial_change.ports import FeatureEncoder, MaskSegmenter
from spatial_change.synthetic.scene import SyntheticScenePair


@dataclass(frozen=True, eq=False)
class PipelineRun:
    """The outcome of the pipeline.

    Attributes:
        lifted: The lifted instances with their features.
        changes: The associated changes.
        seconds: Wall time of the ``lift``, ``describe``, and ``associate``
            stages.
    """

    lifted: LiftedObjects
    changes: tuple[Change, ...]
    seconds: dict[str, float]


def oracle_segmenter(pair: SyntheticScenePair) -> OracleSegmenter:
    """Builds the segmenter that returns the true object under a prompt.

    Args:
        pair: The scene pair.

    Returns:
        An oracle over the instance maps of every frame of both epochs.
    """
    maps: dict[int, NDArray[np.int32]] = {}
    for epoch in Epoch:
        data = pair.epoch_data(epoch)
        for frame, instances in zip(
            data.frames, data.instance_maps, strict=True
        ):
            maps[id(frame.image)] = instances
    return OracleSegmenter(maps)


def run_pipeline(
    votes: SceneEvidence, segmenter: MaskSegmenter, encoder: FeatureEncoder
) -> PipelineRun:
    """Extracts candidates, lifts, describes, and associates.

    Args:
        votes: The scene pair with its clouds and votes.
        segmenter: Segmenter of the lifting.
        encoder: Feature encoder of the descriptors.

    Returns:
        The described instances, the changes, and the stage times.
    """
    reconstructions: dict[Epoch, EpochReconstruction] = {
        epoch: reconstruction(
            epoch, votes.pair.epoch_data(epoch).frames, cloud.points
        )
        for epoch, cloud in votes.clouds.items()
    }
    found = {
        epoch: extract_candidates(
            votes.clouds[epoch].points, votes.evidence[epoch], epoch
        )
        for epoch in Epoch
    }
    candidates = ChangeCandidates(found[Epoch.BEFORE], found[Epoch.AFTER])
    started = time.perf_counter()
    lifted = lift_change_candidates(
        reconstructions[Epoch.BEFORE],
        reconstructions[Epoch.AFTER],
        candidates,
        segmenter,
    )
    lift_s = time.perf_counter() - started
    started = time.perf_counter()
    described = LiftedObjects(
        describe_instances(
            reconstructions[Epoch.BEFORE], lifted.disappeared, encoder
        ),
        describe_instances(
            reconstructions[Epoch.AFTER], lifted.appeared, encoder
        ),
    )
    describe_s = time.perf_counter() - started
    started = time.perf_counter()
    changes = associate(described, candidates, UNGATED)
    associate_s = time.perf_counter() - started
    seconds = {
        "lift": lift_s,
        "describe": describe_s,
        "associate": associate_s,
    }
    return PipelineRun(described, changes, seconds)
