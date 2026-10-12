import dataclasses
import gc
import sys
import weakref
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from spatial_change.perception.sam import (
    SAM3_MODEL_ID,
    SAM21_MODEL_ID,
    SamConfig,
    SamSegmenter,
)
from spatial_change.ports import MaskSegmenter

LOW_H, LOW_W = 4, 6
_CPU = SamConfig(device="cpu")


class FakeEncoding(dict[str, Any]):
    """Stand-in for the processor's batch encoding."""

    def to(self, device: str) -> "FakeEncoding":
        self.device = device
        return self


class Hub:
    """Records what the segmenter asks of the Hugging Face classes."""

    def __init__(self) -> None:
        self.model_types: dict[str, str] = {
            SAM3_MODEL_ID: "sam3",
            SAM21_MODEL_ID: "sam2",
        }
        self.failing: dict[str, Exception] = {}
        self.requested: list[tuple[str, str]] = []
        self.model_kwargs: list[dict[str, Any]] = []
        self.models: list[Any] = []
        self.processors: list[Any] = []


def _classes(hub: Hub, kind: str) -> tuple[type, type]:
    class Model:
        eval_called = False
        device: str | None = None

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.modes: list[tuple[bool, bool]] = []

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> "Model":
            hub.requested.append((kind, model_id))
            hub.model_kwargs.append(kwargs)
            if model_id in hub.failing:
                raise hub.failing[model_id]
            model = cls()
            hub.models.append(model)
            return model

        def eval(self) -> "Model":
            self.eval_called = True
            return self

        def to(self, device: str) -> "Model":
            self.device = device
            return self

        def __call__(self, **kwargs: Any) -> SimpleNamespace:
            self.calls.append(kwargs)
            self.modes.append(
                (
                    torch.is_inference_mode_enabled(),
                    torch.is_autocast_enabled("cpu"),
                )
            )
            count = kwargs["input_boxes"].shape[1]
            logits = torch.full((1, count, 1, LOW_H, LOW_W), -1.0)
            logits[:, 1::2] = 1.0
            return SimpleNamespace(pred_masks=logits)

    class Processor:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.post_calls: list[dict[str, Any]] = []

        @classmethod
        def from_pretrained(cls, model_id: str) -> "Processor":
            processor = cls()
            hub.processors.append(processor)
            return processor

        def __call__(self, **kwargs: Any) -> FakeEncoding:
            self.calls.append(kwargs)
            image = kwargs["images"]
            return FakeEncoding(
                pixel_values=torch.zeros(1, 3, 8, 8),
                original_sizes=torch.tensor([image.shape[:2]]),
                input_boxes=torch.tensor(kwargs["input_boxes"]),
            )

        def post_process_masks(
            self,
            masks: torch.Tensor,
            original_sizes: torch.Tensor,
            mask_threshold: float = 0.0,
            binarize: bool = True,
        ) -> list[torch.Tensor]:
            self.post_calls.append(
                {"mask_threshold": mask_threshold, "binarize": binarize}
            )
            height, width = (int(v) for v in original_sizes[0])
            resized = torch.nn.functional.interpolate(
                masks[0].float(), size=(height, width), mode="nearest"
            )
            return [resized > mask_threshold]

    return Model, Processor


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> Hub:
    import transformers

    state = Hub()

    class AutoConfig:
        @staticmethod
        def from_pretrained(model_id: str) -> SimpleNamespace:
            state.requested.append(("config", model_id))
            if model_id in state.failing:
                raise state.failing[model_id]
            return SimpleNamespace(model_type=state.model_types[model_id])

    monkeypatch.setattr(transformers, "AutoConfig", AutoConfig)
    for kind, model_name, processor_name in [
        ("tracker", "Sam3TrackerModel", "Sam3TrackerProcessor"),
        ("sam2", "Sam2Model", "Sam2Processor"),
    ]:
        model, processor = _classes(state, kind)
        monkeypatch.setattr(transformers, model_name, model)
        monkeypatch.setattr(transformers, processor_name, processor)
    return state


def _image(height: int = 10, width: int = 12) -> NDArray[np.uint8]:
    return np.zeros((height, width, 3), np.uint8)


def _boxes(count: int) -> NDArray[np.float64]:
    return np.tile(np.array([[1.0, 2.0, 7.0, 8.0]]), (count, 1))


def test_config_defaults() -> None:
    config = SamConfig()

    assert config.model_id == SAM3_MODEL_ID
    assert config.fallback_model_id == SAM21_MODEL_ID
    assert config.device == "cuda"
    assert config.mask_threshold == 0.0
    assert SAM3_MODEL_ID == "facebook/sam3"
    assert SAM21_MODEL_ID == "facebook/sam2.1-hiera-large"


def test_config_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        SamConfig().device = "cpu"  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_id": ""},
        {"device": ""},
        {"fallback_model_id": ""},
        {"mask_threshold": float("nan")},
        {"mask_threshold": float("inf")},
    ],
)
def test_config_rejects_invalid_values(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SamConfig(**overrides)


def test_config_allows_no_fallback() -> None:
    assert SamConfig(fallback_model_id=None).fallback_model_id is None


def test_segmenter_satisfies_the_port() -> None:
    assert isinstance(SamSegmenter(), MaskSegmenter)


@pytest.fixture
def model_must_not_load(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(self: SamSegmenter) -> None:
        raise AssertionError("the model must not load")

    monkeypatch.setattr(SamSegmenter, "load", fail)


@pytest.fixture
def no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)


@pytest.mark.usefixtures("model_must_not_load", "no_torch")
def test_no_boxes_need_no_torch() -> None:
    masks = SamSegmenter(_CPU).segment(
        _image(10, 12), np.zeros((0, 4), np.float64), None
    )

    assert masks.shape == (0, 10, 12)
    assert masks.dtype == np.bool_


@pytest.mark.usefixtures("model_must_not_load", "no_torch")
def test_invalid_prompts_are_rejected_before_torch_is_needed() -> None:
    with pytest.raises(ValueError, match="boxes"):
        SamSegmenter(_CPU).segment(
            _image(), np.array([[5.0, 0.0, 4.0, 4.0]]), None
        )
    with pytest.raises(ValueError, match="points"):
        SamSegmenter(_CPU).segment(_image(), _boxes(2), np.zeros((1, 2)))
    with pytest.raises(ValueError, match="image"):
        SamSegmenter(_CPU).segment(np.zeros((8, 8), np.uint8), _boxes(1), None)


def test_nothing_is_loaded_before_first_use(hub: Hub) -> None:
    segmenter = SamSegmenter(_CPU)

    assert segmenter.loaded_model_id is None
    assert hub.requested == []


def test_load_requests_the_configured_model_in_float32(hub: Hub) -> None:
    segmenter = SamSegmenter(SamConfig(model_id=SAM3_MODEL_ID, device="cpu"))

    segmenter.load()

    assert segmenter.loaded_model_id == SAM3_MODEL_ID
    assert hub.model_kwargs == [{"dtype": torch.float32}]
    assert hub.models[0].eval_called
    assert hub.models[0].device == "cpu"


@pytest.mark.parametrize(
    ("model_type", "kind"),
    [
        ("sam3", "tracker"),
        ("sam3_tracker", "tracker"),
        ("sam3_video", "tracker"),
        ("sam2", "sam2"),
        ("sam2_video", "sam2"),
    ],
)
def test_the_classes_follow_the_checkpoint_model_type(
    hub: Hub, model_type: str, kind: str
) -> None:
    hub.model_types[SAM3_MODEL_ID] = model_type

    SamSegmenter(_CPU).load()

    assert hub.requested == [("config", SAM3_MODEL_ID), (kind, SAM3_MODEL_ID)]
    assert len(hub.processors) == 1


def test_an_unknown_model_type_is_rejected_without_the_fallback(
    hub: Hub,
) -> None:
    hub.model_types[SAM3_MODEL_ID] = "vit"

    with pytest.raises(ValueError, match="vit"):
        SamSegmenter(_CPU).load()

    assert hub.models == []
    assert ("config", SAM21_MODEL_ID) not in hub.requested


def test_the_fallback_is_used_when_the_primary_cannot_be_loaded(
    hub: Hub,
) -> None:
    hub.failing[SAM3_MODEL_ID] = OSError("gated repo")
    segmenter = SamSegmenter(_CPU)

    segmenter.load()

    assert segmenter.loaded_model_id == SAM21_MODEL_ID
    assert ("sam2", SAM21_MODEL_ID) in hub.requested


def test_the_error_is_raised_without_a_fallback(hub: Hub) -> None:
    hub.failing[SAM3_MODEL_ID] = OSError("gated repo")
    segmenter = SamSegmenter(SamConfig(device="cpu", fallback_model_id=None))

    with pytest.raises(OSError, match="gated repo"):
        segmenter.load()

    assert segmenter.loaded_model_id is None


def test_both_errors_are_chained_when_the_fallback_fails_too(hub: Hub) -> None:
    primary = OSError("gated repo")
    secondary = OSError("offline")
    hub.failing[SAM3_MODEL_ID] = primary
    hub.failing[SAM21_MODEL_ID] = secondary

    with pytest.raises(OSError) as raised:
        SamSegmenter(_CPU).load()

    assert raised.value.__cause__ is secondary
    assert secondary.__context__ is primary
    assert SAM3_MODEL_ID in str(raised.value)
    assert SAM21_MODEL_ID in str(raised.value)


def test_errors_other_than_os_error_do_not_trigger_the_fallback(
    hub: Hub,
) -> None:
    hub.failing[SAM3_MODEL_ID] = RuntimeError("corrupt")

    with pytest.raises(RuntimeError, match="corrupt"):
        SamSegmenter(_CPU).load()

    assert ("config", SAM21_MODEL_ID) not in hub.requested


def test_the_model_loads_once(hub: Hub) -> None:
    segmenter = SamSegmenter(_CPU)

    segmenter.segment(_image(), _boxes(1), None)
    segmenter.segment(_image(), _boxes(1), None)
    segmenter.load()

    assert len(hub.models) == 1


def test_boxes_and_points_are_nested_as_the_processor_expects(
    hub: Hub,
) -> None:
    image = _image()
    boxes = np.array([[1.0, 2.0, 7.0, 8.0], [0.5, 0.5, 3.5, 4.5]])
    points = np.array([[3.0, 4.0], [2.0, 2.5]])

    SamSegmenter(_CPU).segment(image, boxes, points)

    (call,) = hub.processors[0].calls
    assert call["images"].shape == image.shape
    assert call["input_boxes"] == [[[1.0, 2.0, 7.0, 8.0], [0.5, 0.5, 3.5, 4.5]]]
    assert call["input_points"] == [[[[3.0, 4.0]], [[2.0, 2.5]]]]
    assert call["input_labels"] == [[[1], [1]]]
    assert call["return_tensors"] == "pt"


def test_points_are_optional(hub: Hub) -> None:
    SamSegmenter(_CPU).segment(_image(), _boxes(2), None)

    (call,) = hub.processors[0].calls
    assert call["input_points"] is None
    assert call["input_labels"] is None
    assert len(call["input_boxes"][0]) == 2


def test_the_model_is_called_once_with_a_single_mask_per_prompt(
    hub: Hub,
) -> None:
    SamSegmenter(_CPU).segment(_image(), _boxes(3), None)

    (call,) = hub.models[0].calls
    assert call["multimask_output"] is False
    assert hub.models[0].modes == [(True, False)]


def test_masks_are_returned_at_image_size_in_prompt_order(hub: Hub) -> None:
    masks = SamSegmenter(_CPU).segment(_image(10, 12), _boxes(3), None)

    assert masks.shape == (3, 10, 12)
    assert masks.dtype == np.bool_
    assert [bool(m.all()) for m in masks] == [False, True, False]
    assert [bool(m.any()) for m in masks] == [False, True, False]


def test_the_mask_threshold_is_passed_to_the_post_processing(hub: Hub) -> None:
    config = SamConfig(device="cpu", mask_threshold=-0.5)

    SamSegmenter(config).segment(_image(), _boxes(1), None)

    assert hub.processors[0].post_calls == [
        {"mask_threshold": -0.5, "binarize": True}
    ]


def test_unload_releases_the_weights_and_a_later_call_reloads(
    hub: Hub,
) -> None:
    segmenter = SamSegmenter(_CPU)
    segmenter.load()
    ref = weakref.ref(hub.models[0])
    processor_ref = weakref.ref(hub.processors[0])
    hub.models.clear()
    hub.processors.clear()

    segmenter.unload()
    gc.collect()

    assert ref() is None
    assert processor_ref() is None
    assert segmenter.loaded_model_id is None
    segmenter.load()
    assert len(hub.models) == 1


def test_unload_before_load_does_nothing(hub: Hub) -> None:
    segmenter = SamSegmenter(_CPU)

    segmenter.unload()

    assert hub.requested == []
