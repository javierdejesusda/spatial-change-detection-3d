import dataclasses
import time
from collections.abc import Iterator

import numpy as np
import pytest
from appearance_scenes import reconstruction
from matching_pipeline import run_pipeline
from visibility_scenes import labelled_cloud, vote_pair

from spatial_change.change import (
    ChangeCandidates,
    LiftedObjects,
    associate,
    extract_candidates,
)
from spatial_change.domain import Change, ChangeType, Epoch, ObjectInstance
from spatial_change.perception import (
    DinoFeatureEncoder,
    SamSegmenter,
    SiglipLabeler,
)
from spatial_change.semantics import (
    FURNITURE,
    INDOOR_OBJECTS,
    Landmark,
    extract_landmarks,
    label_instances,
)
from spatial_change.synthetic.scene import (
    ObjectStatus,
    SceneConfig,
    SyntheticObject,
    SyntheticScenePair,
    generate_scene_pair,
)
from spatial_change.visibility import CrossEvidence, VisibilityConfig

pytestmark = pytest.mark.gpu

WIDTH, HEIGHT = 518, 294
FRAMES = 6
MIN_CHANGED_GAP = 0.5


@dataclasses.dataclass(frozen=True, eq=False)
class Run:
    """The labels, changes, and landmarks found with real models."""

    pair: SyntheticScenePair
    labeled: dict[Epoch, tuple[ObjectInstance, ...]]
    changes: tuple[Change, ...]
    landmarks: tuple[Landmark, ...]


def _require_cuda() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


@pytest.fixture(scope="module")
def run() -> Iterator[Run]:
    _require_cuda()
    import torch

    pair = generate_scene_pair(
        0, SceneConfig(width=WIDTH, height=HEIGHT, frames_per_epoch=FRAMES)
    )
    clouds = {e: labelled_cloud(pair.epoch_data(e)) for e in Epoch}
    votes = vote_pair(pair, clouds, VisibilityConfig())
    sam, dino = SamSegmenter(), DinoFeatureEncoder()
    try:
        sam.load()
        dino.load()
    except (OSError, ValueError) as error:
        pytest.skip(f"a model cannot be loaded: {error}")
    result = run_pipeline(votes, sam, dino)
    sam.unload()
    dino.unload()

    recs = {
        e: reconstruction(e, pair.epoch_data(e).frames, clouds[e].points)
        for e in Epoch
    }
    torch.cuda.reset_peak_memory_stats()
    labeler = SiglipLabeler()
    try:
        labeler.load()
        started = time.perf_counter()
        labeled = {
            Epoch.BEFORE: label_instances(
                recs[Epoch.BEFORE], result.lifted.disappeared, labeler
            ),
            Epoch.AFTER: label_instances(
                recs[Epoch.AFTER], result.lifted.appeared, labeler
            ),
        }
        label_s = time.perf_counter() - started
        found = {
            e: extract_candidates(clouds[e].points, votes.evidence[e], e)
            for e in Epoch
        }
        changes = associate(
            LiftedObjects(labeled[Epoch.BEFORE], labeled[Epoch.AFTER]),
            ChangeCandidates(found[Epoch.BEFORE], found[Epoch.AFTER]),
        )
        started = time.perf_counter()
        landmarks = extract_landmarks(
            recs[Epoch.BEFORE],
            CrossEvidence(
                votes.evidence[Epoch.BEFORE], votes.evidence[Epoch.AFTER]
            ),
            labeler,
        )
        landmark_s = time.perf_counter() - started
        print(
            f"labeling {label_s:.2f} s, landmarks {landmark_s:.2f} s, "
            f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB"
        )
    finally:
        labeler.unload()
    yield Run(pair, labeled, changes, landmarks)


def _changed(pair: SyntheticScenePair) -> list[SyntheticObject]:
    return [o for o in pair.objects if o.status is not ObjectStatus.UNCHANGED]


def test_instances_get_vocabulary_labels_and_scores(run: Run) -> None:
    for items in run.labeled.values():
        assert items
        for item in items:
            print(f"{item.object_id}: {item.label} {item.label_score}")
            assert (item.label is None) == (item.label_score is None)
            if item.label is not None and item.label_score is not None:
                assert item.label in INDOOR_OBJECTS
                assert 0.0 < item.label_score <= 1.0


def test_every_change_has_a_label(run: Run) -> None:
    for change in run.changes:
        print(f"{change.change_id} {change.type.value}: {change.label}")
        assert change.label
    assert {c.type for c in run.changes} >= {
        ChangeType.MOVED,
        ChangeType.REMOVED,
        ChangeType.ADDED,
    }


def test_landmarks_are_numbered_furniture_apart_from_changed_objects(
    run: Run,
) -> None:
    for item in run.landmarks:
        print(
            f"{item.landmark_id}: {item.label} center "
            f"{np.round(item.box.center, 2)} extent "
            f"{np.round(item.box.extent, 2)}"
        )
    assert [i.landmark_id for i in run.landmarks] == [
        f"landmark-{k}" for k in range(len(run.landmarks))
    ]
    for item in run.landmarks:
        assert item.label in FURNITURE
        for obj in _changed(run.pair):
            if obj.exists_in(Epoch.BEFORE):
                gap = np.linalg.norm(
                    item.box.center[:2] - obj.box(Epoch.BEFORE).center[:2]
                )
                assert gap > MIN_CHANGED_GAP, (item.landmark_id, obj.object_id)
