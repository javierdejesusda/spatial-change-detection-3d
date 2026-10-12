import time
from collections.abc import Iterator

import numpy as np
import pytest
from facts import make_fact, moved_fact, near
from numpy.typing import NDArray

from spatial_change.domain import ChangeType
from spatial_change.perception import QwenVlm
from spatial_change.ports import ChangeCrops
from spatial_change.semantics import (
    LocalVlmDescriber,
    TemplateDescriber,
    template_sentence,
    validate_description,
)
from spatial_change.semantics.relations import ChangeFact
from spatial_change.synthetic.scene import (
    ObjectStatus,
    SceneConfig,
    SyntheticScenePair,
    generate_scene_pair,
)

pytestmark = pytest.mark.gpu

CONFIG = SceneConfig(width=256, height=192, frames_per_epoch=4)
MIB = 1024 * 1024
MAX_SEEDS = 40


@pytest.fixture(scope="module")
def vlm() -> Iterator[QwenVlm]:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    model = QwenVlm()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        model.load()
    except OSError as error:
        pytest.skip(f"the Qwen model cannot be loaded: {error}")
    print(
        f"loaded {model.loaded_model_id} in "
        f"{time.perf_counter() - started:.1f} s"
    )
    yield model
    model.unload()


def _object_crops(
    pair: SyntheticScenePair, index: int
) -> tuple[ChangeCrops, ChangeCrops]:
    """Crops the object from the frame that shows it largest."""
    sides = []
    for epoch in (pair.before, pair.after):
        best, best_count = None, 0
        for frame, instances in zip(
            epoch.frames, epoch.instance_maps, strict=True
        ):
            count = int((instances == index).sum())
            if count > best_count and frame.image is not None:
                best, best_count = (frame.image, instances), count
        if best is None:
            sides.append(())
            continue
        image, instances = best
        rows, cols = np.nonzero(instances == index)
        crop = image[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
        sides.append((np.ascontiguousarray(crop),))
    return ChangeCrops(before=sides[0]), ChangeCrops(after=sides[1])


def _case(status: ObjectStatus) -> tuple[str, ChangeCrops]:
    """Finds a synthetic object of a status and its label and crops."""
    for seed in range(MAX_SEEDS):
        pair = generate_scene_pair(seed, CONFIG)
        for obj in pair.objects:
            if obj.status is not status:
                continue
            before, after = _object_crops(pair, obj.index)
            crops = ChangeCrops(before=before.before, after=after.after)
            if status is ObjectStatus.ADDED and not crops.after:
                continue
            if status is ObjectStatus.REMOVED and not crops.before:
                continue
            return obj.label, crops
    pytest.skip(f"no {status.value} object in seeds 0 to {MAX_SEEDS - 1}")


def _facts(label: str, change_type: ChangeType) -> ChangeFact:
    if change_type is ChangeType.MOVED:
        return moved_fact(
            label=label,
            near_after=[near("table", "next to", 0.2)],
        )
    if change_type is ChangeType.ADDED:
        return make_fact(
            change_type,
            label=label,
            support_after="floor",
            near_after=[near("shelf", "next to", 0.3)],
        )
    if change_type is ChangeType.REMOVED:
        return make_fact(
            change_type,
            label=label,
            support_before="floor",
            near_before=[near("shelf", "next to", 0.3)],
        )
    return make_fact(change_type, label=label)


@pytest.fixture(scope="module")
def describer(vlm: QwenVlm) -> LocalVlmDescriber:
    return LocalVlmDescriber(vlm, TemplateDescriber())


@pytest.mark.parametrize(
    ("change_type", "status"),
    [
        (ChangeType.MOVED, ObjectStatus.MOVED),
        (ChangeType.ADDED, ObjectStatus.ADDED),
        (ChangeType.REMOVED, ObjectStatus.REMOVED),
        (ChangeType.UNCERTAIN, ObjectStatus.MOVED),
    ],
)
def test_each_change_type_gets_one_valid_repeatable_sentence(
    describer: LocalVlmDescriber,
    change_type: ChangeType,
    status: ObjectStatus,
) -> None:
    import torch

    label, crops = _case(status)
    fact = _facts(label, change_type)

    describer.describe(fact, crops)
    torch.cuda.synchronize()
    started = time.perf_counter()
    text = describer.describe(fact, crops)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    again = describer.describe(fact, crops)

    source = "template" if text == template_sentence(fact) else "model"
    print(
        f"{change_type.value} {label}: {seconds:.2f} s, "
        f"peak {torch.cuda.max_memory_allocated() / MIB:.0f} MiB, "
        f"{source}: {text}"
    )
    assert text.strip() == text
    assert text
    assert "\n" not in text
    assert validate_description(text, fact).ok
    assert again == text


def test_unload_frees_the_gpu_memory(vlm: QwenVlm) -> None:
    import torch

    vlm.unload()

    allocated = torch.cuda.memory_allocated() / MIB
    print(f"allocated after unload: {allocated:.0f} MiB")
    assert vlm.loaded_model_id is None
    assert allocated < 200


def test_the_raw_model_answers_in_json(vlm: QwenVlm) -> None:
    image: NDArray[np.uint8] = np.full((64, 64, 3), 200, np.uint8)

    reply = vlm.generate(
        [image],
        'Reply with strict JSON only: {"description": "<one sentence>"}. '
        "Describe the image.",
    )

    print(f"raw reply: {reply!r}")
    assert reply
