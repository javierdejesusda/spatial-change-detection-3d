import dataclasses
from collections.abc import Iterator

import numpy as np
import pytest
from matching_pipeline import run_pipeline
from visibility_scenes import labelled_cloud, vote_pair

from spatial_change.domain import (
    BoxModel,
    Change,
    ChangeType,
    Epoch,
    OrientedBox,
)
from spatial_change.geometry import rotation_angle_deg
from spatial_change.perception import DinoFeatureEncoder, SamSegmenter
from spatial_change.synthetic.scene import (
    ObjectStatus,
    SceneConfig,
    SyntheticObject,
    SyntheticScenePair,
    generate_scene_pair,
)
from spatial_change.visibility import VisibilityConfig

pytestmark = pytest.mark.gpu

WIDTH, HEIGHT = 518, 294
FRAMES = 6
# Measured with SAM 3 and DINOv3 on seed 0 at 518x294 with 6 frames per
# epoch: the lifting gives 2 BEFORE and 3 AFTER instances, the third being a
# 124-point fragment of the chair that stayed apart, and the association
# gives the chair MOVED, the crate REMOVED, the drum ADDED, and that
# fragment ADDED with confidence 0.06 (42 free observations; the real
# changes have 7,000 to 18,000). Pair fitness is 0.947 for the chair
# (0.745 and below for the other pairs, so none of them is made), and the
# registration takes 0.1 to 0.2 s a pair. The box centers are 0.010 m (chair
# before), 0.214 m (chair after, its instance misses the backrest), 0.088 m
# (crate), and 0.010 m (drum) from the true ones, and the chair's
# displacement is 0.390 m against a true 0.332 m. Its rotation is 20.8
# degrees against a true 69.0, so it is not pinned: the partial AFTER lift
# fits the true motion at 0.709 and a wrong turn at 0.947.
MAX_FULL_CENTER_ERROR = 0.05
MAX_PARTIAL_CENTER_ERROR = 0.35
MAX_CRATE_CENTER_ERROR = 0.2
MAX_DISPLACEMENT_ERROR = 0.15
MAX_FRAGMENT_CONFIDENCE = 0.5
MIN_REAL_CONFIDENCE = 0.5


@dataclasses.dataclass(frozen=True, eq=False)
class Run:
    """The changes found on the rendered pair with real models."""

    pair: SyntheticScenePair
    changes: tuple[Change, ...]


def _require_cuda() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


@pytest.fixture(scope="module")
def segmenter() -> Iterator[SamSegmenter]:
    _require_cuda()
    sam = SamSegmenter()
    try:
        sam.load()
    except (OSError, ValueError) as error:
        pytest.skip(f"no SAM model can be loaded: {error}")
    yield sam
    sam.unload()


@pytest.fixture(scope="module")
def encoder() -> Iterator[DinoFeatureEncoder]:
    _require_cuda()
    dino = DinoFeatureEncoder()
    try:
        dino.load()
    except OSError as error:
        pytest.skip(f"no DINO model can be loaded: {error}")
    yield dino
    dino.unload()


@pytest.fixture(scope="module")
def run(segmenter: SamSegmenter, encoder: DinoFeatureEncoder) -> Run:
    pair = generate_scene_pair(
        0, SceneConfig(width=WIDTH, height=HEIGHT, frames_per_epoch=FRAMES)
    )
    clouds = {e: labelled_cloud(pair.epoch_data(e)) for e in Epoch}
    votes = vote_pair(pair, clouds, VisibilityConfig())
    result = run_pipeline(votes, segmenter, encoder)
    seconds = result.seconds
    print(
        f"lift {seconds['lift']:.1f} s, describe "
        f"{seconds['describe']:.2f} s for {2 * FRAMES} frames, associate "
        f"{seconds['associate']:.2f} s for "
        f"{len(result.lifted.disappeared)} x {len(result.lifted.appeared)} "
        "objects"
    )
    return Run(pair, result.changes)


def _truth(pair: SyntheticScenePair, status: ObjectStatus) -> SyntheticObject:
    return next(o for o in pair.objects if o.status is status)


def _distance(box: BoxModel | None, truth: OrientedBox | None) -> float:
    assert box is not None and truth is not None
    return float(np.linalg.norm(np.array(box.center) - truth.center))


def test_real_models_find_the_three_changed_objects(run: Run) -> None:
    chair = _truth(run.pair, ObjectStatus.MOVED)
    crate = _truth(run.pair, ObjectStatus.REMOVED)
    drum = _truth(run.pair, ObjectStatus.ADDED)
    assert chair.box_before is not None and chair.box_after is not None
    assert chair.motion is not None
    true_distance = float(
        np.linalg.norm(chair.box_after.center - chair.box_before.center)
    )
    print(
        f"truth chair: displacement {true_distance:.3f} m, rotation "
        f"{rotation_angle_deg(chair.motion.rotation):.1f} deg"
    )
    for change in run.changes:
        print(
            f"{change.change_id} {change.type.value} "
            f"confidence {change.confidence:.2f} "
            f"displacement {change.displacement_m} "
            f"rotation {change.rotation_deg} evidence {change.evidence}"
        )
    kinds = [c.type for c in run.changes]
    assert kinds.count(ChangeType.MOVED) == 1
    assert kinds.count(ChangeType.REMOVED) == 1
    assert kinds.count(ChangeType.ADDED) >= 1

    (moved,) = [c for c in run.changes if c.type is ChangeType.MOVED]
    (removed,) = [c for c in run.changes if c.type is ChangeType.REMOVED]
    added = min(
        (c for c in run.changes if c.type is ChangeType.ADDED),
        key=lambda c: _distance(c.after_box, drum.box_after),
    )
    errors = {
        "chair before": _distance(moved.before_box, chair.box_before),
        "chair after": _distance(moved.after_box, chair.box_after),
        "crate": _distance(removed.before_box, crate.box_before),
        "drum": _distance(added.after_box, drum.box_after),
    }
    print(f"box center errors: {errors}")
    assert errors["chair before"] <= MAX_FULL_CENTER_ERROR
    assert errors["chair after"] <= MAX_PARTIAL_CENTER_ERROR
    assert errors["crate"] <= MAX_CRATE_CENTER_ERROR
    assert errors["drum"] <= MAX_FULL_CENTER_ERROR
    for change in run.changes:
        if change.type is ChangeType.ADDED and change is not added:
            assert change.confidence < MAX_FRAGMENT_CONFIDENCE
    assert added.confidence >= MIN_REAL_CONFIDENCE
    assert moved.displacement_m is not None
    assert abs(moved.displacement_m - true_distance) <= MAX_DISPLACEMENT_ERROR
