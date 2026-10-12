import time
from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray

from spatial_change.domain import Epoch
from spatial_change.perception import SiglipLabeler
from spatial_change.semantics.vocabulary import INDOOR_OBJECTS
from spatial_change.synthetic.scene import (
    SceneConfig,
    SyntheticScenePair,
    generate_scene_pair,
)

pytestmark = pytest.mark.gpu

WIDTH, HEIGHT = 518, 294
MARGIN = 0.1
# Measured with SigLIP 2 base on the best view of every object of seeds 0 to
# 2 (28 crops of at least 400 pixels): 17 ms per crop, 1.5 GiB peak, and 11
# right over the 48 indoor names. Flat-colored primitives are hard: chairs
# 6 of 6 and stools 3 of 3 are right, tables 1 of 4, drums 1 of 3, crates 0
# of 6, and shelves 0 of 6.
MIN_ACCURACY = 0.25


@pytest.fixture(scope="module")
def labeler() -> Iterator[SiglipLabeler]:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    siglip = SiglipLabeler()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        siglip.load()
    except OSError as error:
        pytest.skip(f"no SigLIP model can be loaded: {error}")
    print(
        f"loaded {siglip.loaded_model_id} in "
        f"{time.perf_counter() - started:.1f} s"
    )
    yield siglip
    siglip.unload()


@pytest.fixture(scope="module")
def crops() -> list[tuple[str, NDArray[np.uint8]]]:
    """Crops the best view of every object, labeled with its true class."""
    found = []
    for seed in range(3):
        pair = generate_scene_pair(
            seed,
            SceneConfig(width=WIDTH, height=HEIGHT, frames_per_epoch=6),
        )
        found.extend(_object_crops(pair))
    return found


def _object_crops(
    pair: SyntheticScenePair,
) -> list[tuple[str, NDArray[np.uint8]]]:
    result = []
    for epoch in Epoch:
        data = pair.epoch_data(epoch)
        for obj in pair.objects_in(epoch):
            counts = [int((m == obj.index).sum()) for m in data.instance_maps]
            best = int(np.argmax(counts))
            rows, cols = np.nonzero(data.instance_maps[best] == obj.index)
            if counts[best] < 400:
                continue
            image = data.frames[best].image
            assert image is not None
            pad_r = int(MARGIN * (rows.max() - rows.min() + 1))
            pad_c = int(MARGIN * (cols.max() - cols.min() + 1))
            crop = image[
                max(rows.min() - pad_r, 0) : rows.max() + pad_r + 1,
                max(cols.min() - pad_c, 0) : cols.max() + pad_c + 1,
            ]
            result.append((obj.label, np.ascontiguousarray(crop)))
    return result


def test_every_crop_gets_a_label_of_the_vocabulary_and_a_score(
    labeler: SiglipLabeler, crops: list[tuple[str, NDArray[np.uint8]]]
) -> None:
    images = [crop for _, crop in crops]

    started = time.perf_counter()
    results = labeler.label(images, INDOOR_OBJECTS)
    elapsed = time.perf_counter() - started

    assert len(results) == len(images)
    for label, score in results:
        assert label in INDOOR_OBJECTS
        assert 0.0 <= score <= 1.0
    hits: dict[str, list[bool]] = {}
    for (truth, _), (label, _) in zip(crops, results, strict=True):
        hits.setdefault(truth, []).append(label == truth)
    import torch

    print(
        f"{len(images)} crops in {elapsed:.3f} s "
        f"({elapsed / len(images) * 1000:.1f} ms per crop), "
        f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB"
    )
    for truth, flags in sorted(hits.items()):
        print(f"{truth}: {sum(flags)}/{len(flags)}")
    right = sum(sum(flags) for flags in hits.values())
    assert right / len(images) >= MIN_ACCURACY


def test_the_same_crops_give_the_same_labels(
    labeler: SiglipLabeler, crops: list[tuple[str, NDArray[np.uint8]]]
) -> None:
    images = [crop for _, crop in crops][:8]

    first = labeler.label(images, INDOOR_OBJECTS)
    second = labeler.label(images, INDOOR_OBJECTS)

    assert [a for a, _ in first] == [b for b, _ in second]
    np.testing.assert_allclose(
        [s for _, s in first], [s for _, s in second], atol=1e-3
    )


def test_a_batch_gives_the_labels_of_the_crops_one_by_one(
    labeler: SiglipLabeler, crops: list[tuple[str, NDArray[np.uint8]]]
) -> None:
    images = [crop for _, crop in crops][:6]

    together = labeler.label(images, INDOOR_OBJECTS)
    alone = [labeler.label([i], INDOOR_OBJECTS)[0] for i in images]

    assert [a for a, _ in together] == [b for b, _ in alone]


def test_a_two_entry_vocabulary_scores_sum_to_the_softmax(
    labeler: SiglipLabeler, crops: list[tuple[str, NDArray[np.uint8]]]
) -> None:
    truth, image = crops[0]
    other = next(n for n in INDOOR_OBJECTS if n != truth)

    first = labeler.label([image], [truth, other])[0]
    swapped = labeler.label([image], [other, truth])[0]

    assert first[0] == swapped[0]
    assert first[1] == pytest.approx(swapped[1], abs=1e-3)
    assert first[1] >= 0.5
