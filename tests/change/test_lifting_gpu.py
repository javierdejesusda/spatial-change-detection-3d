import dataclasses
import time
from collections.abc import Iterator

import numpy as np
import pytest
from appearance_scenes import reconstruction
from lifting_metrics import candidate_object, purity_and_recall
from numpy.typing import NDArray
from visibility_scenes import LabelledCloud, changed_objects, labelled_cloud

from spatial_change.change import (
    Candidate,
    ChangeCandidates,
    LiftConfig,
    extract_candidates,
    lift_change_candidates,
)
from spatial_change.domain import Epoch, EpochReconstruction, ObjectInstance
from spatial_change.perception import SamSegmenter
from spatial_change.synthetic.scene import (
    SceneConfig,
    SyntheticScenePair,
    generate_scene_pair,
)
from spatial_change.visibility import VisibilityConfig, vote_points

pytestmark = pytest.mark.gpu

WIDTH, HEIGHT = 518, 294
FRAMES = 6
# Measured with SAM 3 on seed 0 at 518x294 with 6 frames per epoch: BEFORE
# gives 2 candidates and 2 instances, AFTER 4 candidates and 3 instances
# (the chair stays split in a large and a 124-point instance). Purity is
# 1.000 for the largest instance of each changed object. The box center of
# that instance lies 0.010 to 0.088 m from the true one, except the chair at
# 0.214 m, whose instance misses its backrest (recall 0.42) so its box is
# 0.46 m tall instead of 0.93 m. Before the link to the candidate's own
# voxels, 1 to 2 percent of the points were mask spill onto the background
# up to 3 m behind the object, and the center error was 0.4 to 1.0 m. Before
# the floor clearance, 110 floor points at the chair's base stretched its box
# to 1.09 m and its center error to 0.398 m. Recall of the object's points
# is 0.03 to 1.00.
MIN_PURITY = 0.8
MIN_MAIN_PURITY = 0.9
MAX_MAIN_CENTER_ERROR = 0.35


@dataclasses.dataclass(frozen=True, eq=False)
class Setup:
    """A rendered pair with its clouds, reconstructions, and candidates."""

    pair: SyntheticScenePair
    clouds: dict[Epoch, LabelledCloud]
    reconstructions: dict[Epoch, EpochReconstruction]
    candidates: ChangeCandidates


@pytest.fixture(scope="module")
def segmenter() -> Iterator[SamSegmenter]:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    sam = SamSegmenter()
    started = time.perf_counter()
    try:
        sam.load()
    except (OSError, ValueError) as error:
        pytest.skip(f"no SAM model can be loaded: {error}")
    print(
        f"loaded {sam.loaded_model_id} in {time.perf_counter() - started:.1f} s"
    )
    yield sam
    sam.unload()


@pytest.fixture(scope="module")
def setup() -> Setup:
    pair = generate_scene_pair(
        0, SceneConfig(width=WIDTH, height=HEIGHT, frames_per_epoch=6)
    )
    clouds = {e: labelled_cloud(pair.epoch_data(e)) for e in Epoch}
    voters = {Epoch.BEFORE: pair.after.frames, Epoch.AFTER: pair.before.frames}
    candidates = {}
    for epoch in Epoch:
        evidence = vote_points(
            clouds[epoch].points, voters[epoch], VisibilityConfig()
        )
        candidates[epoch] = extract_candidates(
            clouds[epoch].points, evidence, epoch
        )
    reconstructions = {
        e: reconstruction(e, pair.epoch_data(e).frames, clouds[e].points)
        for e in Epoch
    }
    return Setup(
        pair,
        clouds,
        reconstructions,
        ChangeCandidates(
            disappeared=candidates[Epoch.BEFORE],
            appeared=candidates[Epoch.AFTER],
        ),
    )


def test_real_masks_give_pure_instances_for_every_changed_object(
    segmenter: SamSegmenter, setup: Setup
) -> None:
    timings: list[float] = []
    original = segmenter.segment

    def timed(
        image: NDArray[np.uint8],
        boxes: NDArray[np.float64],
        points: NDArray[np.float64] | None,
    ) -> NDArray[np.bool_]:
        started = time.perf_counter()
        masks = original(image, boxes, points)
        timings.append(time.perf_counter() - started)
        return masks

    segmenter.segment = timed  # type: ignore[method-assign]
    started = time.perf_counter()
    try:
        lifted = lift_change_candidates(
            setup.reconstructions[Epoch.BEFORE],
            setup.reconstructions[Epoch.AFTER],
            setup.candidates,
            segmenter,
            LiftConfig(),
        )
    finally:
        del segmenter.segment
    print(
        f"{segmenter.loaded_model_id}: {len(timings)} segment calls, "
        f"{time.perf_counter() - started:.1f} s in total, "
        f"{np.mean(timings):.2f} s per call"
    )
    assert len(timings) <= 2 * FRAMES
    sides = {
        Epoch.BEFORE: (lifted.disappeared, setup.candidates.disappeared),
        Epoch.AFTER: (lifted.appeared, setup.candidates.appeared),
    }
    for epoch, (instances, candidates) in sides.items():
        report(setup, epoch, instances, candidates)
        assert len(instances) <= len(candidates)


def report(
    setup: Setup,
    epoch: Epoch,
    instances: tuple[ObjectInstance, ...],
    candidates: tuple[Candidate, ...],
) -> None:
    """Prints and checks the instances of one epoch."""
    cloud = setup.clouds[epoch]
    objects = changed_objects(setup.pair, epoch)
    size = LiftConfig().voxel_size
    best: dict[int, tuple[int, float, float]] = {}
    print(f"{epoch.value}: {len(candidates)} candidates -> {len(instances)}")
    for instance in instances:
        purity, recall = purity_and_recall(cloud, candidates, instance, size)
        index = candidate_object(cloud, candidates, instance)
        obj = next(o for o in objects if o.index == index)
        error = float(
            np.linalg.norm(instance.box.center - obj.box(epoch).center)
        )
        print(
            f"  {instance.object_id} object {index}: "
            f"{len(instance.points)} points, purity {purity:.3f}, "
            f"recall {recall:.3f}, box center error {error:.3f} m"
        )
        assert purity >= MIN_PURITY, instance.object_id
        if len(instance.points) > best.get(index, (0, 0.0, 0.0))[0]:
            best[index] = (len(instance.points), purity, error)
    assert sorted(best) == sorted(o.index for o in objects)
    assert all(p >= MIN_MAIN_PURITY for _, p, _ in best.values())
    assert all(e <= MAX_MAIN_CENTER_ERROR for _, _, e in best.values())
