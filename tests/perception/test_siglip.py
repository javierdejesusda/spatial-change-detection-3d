import dataclasses
import gc
import weakref
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from torch import nn

from spatial_change.perception.siglip import (
    SIGLIP2_MODEL_ID,
    SIGLIP_MODEL_ID,
    SiglipConfig,
    SiglipLabeler,
)
from spatial_change.ports import Labeler

SCALE = 10.0
DIRECTIONS = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "twin": (1.0, 0.0, 0.0),
}
_CPU = SiglipConfig(device="cpu")


class FakeSiglip(nn.Module):
    """Stand-in for a Hugging Face SigLIP model.

    A text feature is the direction its prompt was given by the processor
    and an image feature is the mean color of the crop in ``[0, 1]``, so a
    test can predict every logit.
    """

    instances: ClassVar[list["FakeSiglip"]] = []
    requested: ClassVar[list[str]] = []
    failing: ClassVar[dict[str, Exception]] = {}
    load_kwargs: ClassVar[list[dict[str, Any]]] = []

    def __init__(self) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(SCALE))))
        self.eval_called = False
        self.text_calls = 0
        self.image_batches: list[int] = []
        self.modes: list[bool] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> "FakeSiglip":
        cls.requested.append(model_id)
        cls.load_kwargs.append(kwargs)
        if model_id in cls.failing:
            raise cls.failing[model_id]
        model = cls()
        cls.instances.append(model)
        return model

    def eval(self) -> "FakeSiglip":
        self.eval_called = True
        return self

    def get_text_features(self, input_ids: torch.Tensor) -> SimpleNamespace:
        self.text_calls += 1
        self.modes.append(torch.is_inference_mode_enabled())
        return SimpleNamespace(pooler_output=input_ids * 3.0)

    def get_image_features(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        self.image_batches.append(len(pixel_values))
        self.modes.append(torch.is_inference_mode_enabled())
        return SimpleNamespace(pooler_output=pixel_values.mean(dim=(2, 3)) * 2)


class FakeProcessor:
    """Stand-in for the processor: records calls and builds tensors."""

    instances: ClassVar[list["FakeProcessor"]] = []
    requested: ClassVar[list[str]] = []
    failing: ClassVar[dict[str, Exception]] = {}

    def __init__(self) -> None:
        self.text_calls: list[tuple[list[str], dict[str, Any]]] = []
        self.image_calls: list[list[NDArray[np.uint8]]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> "FakeProcessor":
        cls.requested.append(model_id)
        if model_id in cls.failing:
            raise cls.failing[model_id]
        processor = cls()
        cls.instances.append(processor)
        return processor

    def __call__(
        self,
        text: list[str] | None = None,
        images: list[NDArray[np.uint8]] | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if text is not None:
            self.text_calls.append((list(text), kwargs))
            names = [
                t.removeprefix("this is a ").removesuffix(".") for t in text
            ]
            return {
                "input_ids": torch.tensor([DIRECTIONS[n] for n in names]),
            }
        assert images is not None
        self.image_calls.append(list(images))
        pixels = [
            torch.from_numpy(np.asarray(i, np.float32)).mean(dim=(0, 1)) / 255
            for i in images
        ]
        return {
            "pixel_values": torch.stack(pixels)[:, :, None, None],
        }


@pytest.fixture
def fake_siglip(monkeypatch: pytest.MonkeyPatch) -> type[FakeSiglip]:
    import transformers

    for fake in (FakeSiglip, FakeProcessor):
        fake.instances.clear()
        fake.requested.clear()
        fake.failing.clear()
    FakeSiglip.load_kwargs.clear()
    monkeypatch.setattr(transformers, "AutoModel", FakeSiglip)
    monkeypatch.setattr(transformers, "AutoProcessor", FakeProcessor)
    return FakeSiglip


_TEMPLATE = SiglipConfig(device="cpu", prompt="this is a {}.")


def _crop(rgb: tuple[int, int, int], height: int = 6, width: int = 9) -> Any:
    return np.full((height, width, 3), rgb, np.uint8)


def _expected(
    rgb: tuple[int, int, int], vocabulary: Sequence[str]
) -> NDArray[np.float64]:
    image = np.array(rgb, np.float64) / 255 * 2
    image /= np.linalg.norm(image)
    logits = []
    for name in vocabulary:
        text = np.array(DIRECTIONS[name]) * 3.0
        text /= np.linalg.norm(text)
        logits.append(SCALE * float(image @ text))
    shifted = np.exp(np.array(logits) - max(logits))
    return shifted / shifted.sum()


def test_config_defaults() -> None:
    config = SiglipConfig()

    assert config.model_id == SIGLIP2_MODEL_ID
    assert config.fallback_model_id == SIGLIP_MODEL_ID
    assert config.device == "cuda"
    assert config.prompt == "a photo of a {}."
    assert config.batch_size == 16
    assert SIGLIP2_MODEL_ID == "google/siglip2-base-patch16-224"
    assert SIGLIP_MODEL_ID == "google/siglip-base-patch16-224"


def test_config_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        SiglipConfig().device = "cpu"  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_id": ""},
        {"device": ""},
        {"fallback_model_id": ""},
        {"prompt": ""},
        {"prompt": "a photo."},
        {"prompt": "{} {}"},
        {"batch_size": 0},
    ],
)
def test_config_rejects_invalid_values(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SiglipConfig(**overrides)


def test_config_allows_no_fallback() -> None:
    assert SiglipConfig(fallback_model_id=None).fallback_model_id is None


def test_labeler_satisfies_the_port() -> None:
    assert isinstance(SiglipLabeler(), Labeler)


@pytest.fixture
def model_must_not_load(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(self: SiglipLabeler) -> None:
        raise AssertionError("the model must not load for invalid input")

    monkeypatch.setattr(SiglipLabeler, "load", fail)


@pytest.mark.usefixtures("model_must_not_load")
@pytest.mark.parametrize(
    "crop",
    [
        np.zeros((8, 8), np.uint8),
        np.zeros((8, 8, 4), np.uint8),
        np.zeros((8, 8, 3), np.float32),
        np.zeros((0, 8, 3), np.uint8),
    ],
)
def test_invalid_crops_are_rejected_without_loading(crop: Any) -> None:
    with pytest.raises(ValueError, match="image"):
        SiglipLabeler().label([_crop((1, 2, 3)), crop], ["red"])


@pytest.mark.usefixtures("model_must_not_load")
@pytest.mark.parametrize(
    "vocabulary", [[], ["red", "red"], ["red", ""], ["red", 3]]
)
def test_invalid_vocabularies_are_rejected_without_loading(
    vocabulary: list[Any],
) -> None:
    with pytest.raises(ValueError, match="vocabulary"):
        SiglipLabeler().label([_crop((1, 2, 3))], vocabulary)


@pytest.mark.usefixtures("model_must_not_load")
def test_no_crops_give_no_labels_without_loading() -> None:
    assert SiglipLabeler().label([], ["red"]) == []


def test_nothing_is_loaded_before_first_use(
    fake_siglip: type[FakeSiglip],
) -> None:
    labeler = SiglipLabeler(_CPU)

    assert labeler.loaded_model_id is None
    assert fake_siglip.requested == []


def test_load_requests_model_and_processor_in_float32(
    fake_siglip: type[FakeSiglip],
) -> None:
    labeler = SiglipLabeler(SiglipConfig(model_id="org/m", device="cpu"))

    labeler.load()

    assert fake_siglip.requested == ["org/m"]
    assert FakeProcessor.requested == ["org/m"]
    assert labeler.loaded_model_id == "org/m"
    assert fake_siglip.instances[0].eval_called
    assert fake_siglip.load_kwargs[0]["dtype"] is torch.float32


def test_an_unavailable_model_falls_back_with_its_processor(
    fake_siglip: type[FakeSiglip],
) -> None:
    fake_siglip.failing["org/m"] = OSError("gated")
    labeler = SiglipLabeler(
        SiglipConfig(model_id="org/m", fallback_model_id="org/f", device="cpu")
    )

    labeler.load()

    assert fake_siglip.requested == ["org/m", "org/f"]
    assert FakeProcessor.requested == ["org/f"]
    assert labeler.loaded_model_id == "org/f"


def test_an_unavailable_processor_falls_back_too(
    fake_siglip: type[FakeSiglip],
) -> None:
    FakeProcessor.failing["org/m"] = OSError("no tokenizer")
    labeler = SiglipLabeler(
        SiglipConfig(model_id="org/m", fallback_model_id="org/f", device="cpu")
    )

    labeler.load()

    assert labeler.loaded_model_id == "org/f"


def test_without_a_fallback_the_error_propagates(
    fake_siglip: type[FakeSiglip],
) -> None:
    fake_siglip.failing["org/m"] = OSError("gated")
    labeler = SiglipLabeler(
        SiglipConfig(model_id="org/m", fallback_model_id=None, device="cpu")
    )

    with pytest.raises(OSError, match="gated"):
        labeler.load()
    assert labeler.loaded_model_id is None


def test_prompts_are_the_template_filled_with_each_entry(
    fake_siglip: type[FakeSiglip],
) -> None:
    labeler = SiglipLabeler(_TEMPLATE)

    labeler.label([_crop((255, 0, 0))], ["red", "blue"])

    ((prompts, kwargs),) = FakeProcessor.instances[0].text_calls
    assert prompts == ["this is a red.", "this is a blue."]
    assert kwargs["padding"] == "max_length"
    assert kwargs["max_length"] == 64


def test_scores_are_the_softmax_of_the_model_logits(
    fake_siglip: type[FakeSiglip],
) -> None:
    labeler = SiglipLabeler(_TEMPLATE)
    vocabulary = ["red", "green", "blue"]
    colors = [(200, 40, 10), (10, 90, 30), (30, 20, 250)]

    results = labeler.label([_crop(c) for c in colors], vocabulary)

    assert [label for label, _ in results] == ["red", "green", "blue"]
    for rgb, (_, score) in zip(colors, results, strict=True):
        assert score == pytest.approx(
            _expected(rgb, vocabulary).max(), abs=1e-5
        )
        assert 0.0 <= score <= 1.0


def test_a_tie_goes_to_the_earlier_entry(
    fake_siglip: type[FakeSiglip],
) -> None:
    labeler = SiglipLabeler(_TEMPLATE)

    first = labeler.label([_crop((255, 0, 0))], ["red", "twin"])
    second = labeler.label([_crop((255, 0, 0))], ["twin", "red"])

    assert first[0][0] == "red"
    assert second[0][0] == "twin"
    assert first[0][1] == pytest.approx(0.5)


def test_a_single_entry_scores_one(fake_siglip: type[FakeSiglip]) -> None:
    results = SiglipLabeler(_TEMPLATE).label([_crop((9, 9, 9))], ["red"])

    assert results == [("red", pytest.approx(1.0))]


def test_crops_are_encoded_in_batches_and_keep_their_order(
    fake_siglip: type[FakeSiglip],
) -> None:
    labeler = SiglipLabeler(dataclasses.replace(_TEMPLATE, batch_size=2))
    palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    crops = [_crop(palette[i % 3]) for i in range(5)]

    results = labeler.label(crops, ["red", "green", "blue"])

    assert fake_siglip.instances[0].image_batches == [2, 2, 1]
    assert [label for label, _ in results] == [
        "red",
        "green",
        "blue",
        "red",
        "green",
    ]


def test_crops_of_any_size_are_passed_to_the_processor(
    fake_siglip: type[FakeSiglip],
) -> None:
    crops = [_crop((255, 0, 0), 3, 5), _crop((0, 255, 0), 40, 7)]

    SiglipLabeler(_TEMPLATE).label(crops, ["red", "green"])

    (sent,) = FakeProcessor.instances[0].image_calls
    assert [c.shape for c in sent] == [(3, 5, 3), (40, 7, 3)]


def test_the_text_features_are_cached_for_the_last_vocabulary(
    fake_siglip: type[FakeSiglip],
) -> None:
    labeler = SiglipLabeler(_TEMPLATE)
    model = fake_siglip

    labeler.label([_crop((255, 0, 0))], ["red", "green"])
    labeler.label([_crop((0, 255, 0))], ["red", "green"])
    assert model.instances[0].text_calls == 1

    labeler.label([_crop((0, 255, 0))], ["green", "red"])
    assert model.instances[0].text_calls == 2

    labeler.label([_crop((0, 255, 0))], ["green", "red"])
    assert model.instances[0].text_calls == 2

    labeler.label([_crop((0, 255, 0))], ["red", "green"])
    assert model.instances[0].text_calls == 3


def test_inference_runs_without_gradients(
    fake_siglip: type[FakeSiglip],
) -> None:
    SiglipLabeler(_TEMPLATE).label([_crop((255, 0, 0))], ["red", "green"])

    assert all(fake_siglip.instances[0].modes)


def test_unload_releases_the_model_and_clears_the_cache(
    fake_siglip: type[FakeSiglip],
) -> None:
    labeler = SiglipLabeler(_TEMPLATE)
    labeler.label([_crop((255, 0, 0))], ["red", "green"])
    reference = weakref.ref(fake_siglip.instances[0])
    fake_siglip.instances.clear()

    labeler.unload()
    gc.collect()

    assert labeler.loaded_model_id is None
    assert reference() is None
    labeler.label([_crop((255, 0, 0))], ["red", "green"])
    assert fake_siglip.instances[0].text_calls == 1


def test_unload_without_a_model_does_nothing(
    fake_siglip: type[FakeSiglip],
) -> None:
    SiglipLabeler(_TEMPLATE).unload()

    assert fake_siglip.requested == []
