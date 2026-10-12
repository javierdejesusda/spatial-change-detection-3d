import time
from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray

from spatial_change.perception import (
    SAM21_MODEL_ID,
    SamConfig,
    SamSegmenter,
)
from spatial_change.synthetic.scene import (
    SceneConfig,
    SyntheticEpoch,
    generate_scene_pair,
)

pytestmark = pytest.mark.gpu

WIDTH, HEIGHT = 518, 294
MIN_OBJECT_PIXELS = 400
# Measured on four rendered frames: SAM 3 mean 0.975, SAM 2.1 mean 0.950.
MIN_MEAN_IOU = 0.85


def _load(config: SamConfig) -> Iterator[SamSegmenter]:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    segmenter = SamSegmenter(config)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        segmenter.load()
    except (OSError, ValueError) as error:
        pytest.skip(f"no SAM model can be loaded: {error}")
    print(
        f"loaded {segmenter.loaded_model_id} in "
        f"{time.perf_counter() - started:.1f} s"
    )
    yield segmenter
    segmenter.unload()


@pytest.fixture(scope="module")
def default_segmenter() -> Iterator[SamSegmenter]:
    yield from _load(SamConfig())


@pytest.fixture(scope="module")
def open_segmenter() -> Iterator[SamSegmenter]:
    yield from _load(SamConfig(model_id=SAM21_MODEL_ID, fallback_model_id=None))


@pytest.fixture(scope="module")
def epoch() -> SyntheticEpoch:
    pair = generate_scene_pair(
        0, SceneConfig(width=WIDTH, height=HEIGHT, frames_per_epoch=4)
    )
    return pair.before


def _prompts(
    instance_map: NDArray[np.int32],
) -> tuple[list[int], NDArray[np.float64], NDArray[np.float64]]:
    """Builds a ground-truth box and an interior point per visible object."""
    labels, boxes, points = [], [], []
    for label in np.unique(instance_map[instance_map >= 0]):
        rows, cols = np.nonzero(instance_map == label)
        if len(rows) < MIN_OBJECT_PIXELS:
            continue
        distance = (rows - np.median(rows)) ** 2 + (cols - np.median(cols)) ** 2
        nearest = int(np.argmin(distance))
        labels.append(int(label))
        boxes.append([cols.min(), rows.min(), cols.max(), rows.max()])
        points.append([cols[nearest], rows[nearest]])
    return labels, np.array(boxes, np.float64), np.array(points, np.float64)


def _ious(segmenter: SamSegmenter, epoch: SyntheticEpoch) -> list[float]:
    """Prompts every large object of every frame and scores its mask."""
    torch = pytest.importorskip("torch")
    scores: list[float] = []
    seconds: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    for image, instance_map in zip(
        (frame.image for frame in epoch.frames),
        epoch.instance_maps,
        strict=True,
    ):
        assert image is not None
        labels, boxes, points = _prompts(instance_map)
        if not labels:
            continue
        torch.cuda.synchronize()
        started = time.perf_counter()
        masks = segmenter.segment(image, boxes, points)
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - started)
        assert masks.shape == (len(labels), HEIGHT, WIDTH)
        assert masks.dtype == np.bool_
        for label, mask in zip(labels, masks, strict=True):
            truth = instance_map == label
            scores.append(float((mask & truth).sum() / (mask | truth).sum()))
    assert scores, "no frame shows an object large enough to prompt"
    print(
        f"{segmenter.loaded_model_id}: {len(scores)} objects in "
        f"{len(seconds)} frames, {np.mean(seconds):.3f} s per frame "
        f"(first {seconds[0]:.3f} s), peak "
        f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MiB, IoU mean "
        f"{np.mean(scores):.3f} min {np.min(scores):.3f}"
    )
    return scores


def test_the_default_model_segments_prompted_objects(
    default_segmenter: SamSegmenter, epoch: SyntheticEpoch
) -> None:
    assert np.mean(_ious(default_segmenter, epoch)) >= MIN_MEAN_IOU


def test_the_open_model_segments_prompted_objects(
    open_segmenter: SamSegmenter, epoch: SyntheticEpoch
) -> None:
    assert np.mean(_ious(open_segmenter, epoch)) >= MIN_MEAN_IOU
