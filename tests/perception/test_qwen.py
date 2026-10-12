import dataclasses
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from PIL import Image

from spatial_change.perception.qwen import (
    QWEN_MODEL_ID,
    QwenVlm,
    QwenVlmConfig,
)
from spatial_change.ports import VisionLanguageModel

_CPU = QwenVlmConfig(device="cpu")
_REPLY_TOKENS = [11, 12, 13]
PROMPT_TOKENS = 5


class FakeEncoding(dict[str, Any]):
    """Stand-in for the processor's batch encoding."""

    def to(self, device: str) -> "FakeEncoding":
        self.device = device
        return self


class Hub:
    """Records what the adapter asks of the Hugging Face classes."""

    def __init__(self) -> None:
        self.failing: dict[str, Exception] = {}
        self.requested: list[str] = []
        self.model_kwargs: list[dict[str, Any]] = []
        self.models: list[Any] = []
        self.processors: list[Any] = []
        self.decoded: list[Any] = []
        self.reply = "  a reply \n"


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> Hub:
    import transformers

    state = Hub()

    class Model:
        def __init__(self) -> None:
            self.eval_called = False
            self.generate_calls: list[dict[str, Any]] = []
            self.modes: list[bool] = []

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> "Model":
            state.requested.append(model_id)
            state.model_kwargs.append(kwargs)
            if model_id in state.failing:
                raise state.failing[model_id]
            model = cls()
            state.models.append(model)
            return model

        def eval(self) -> "Model":
            self.eval_called = True
            return self

        def to(self, device: str) -> "Model":
            self.device = device
            return self

        def generate(self, **kwargs: Any) -> torch.Tensor:
            self.generate_calls.append(kwargs)
            self.modes.append(torch.is_inference_mode_enabled())
            prompt = kwargs["input_ids"]
            reply = torch.tensor([_REPLY_TOKENS])
            return torch.cat([prompt, reply], dim=1)

    class Processor:
        def __init__(self) -> None:
            self.conversations: list[Any] = []
            self.template_kwargs: list[dict[str, Any]] = []

        @classmethod
        def from_pretrained(cls, model_id: str) -> "Processor":
            processor = cls()
            state.processors.append(processor)
            return processor

        def apply_chat_template(
            self, conversation: Any, **kwargs: Any
        ) -> FakeEncoding:
            self.conversations.append(conversation)
            self.template_kwargs.append(kwargs)
            return FakeEncoding(input_ids=torch.zeros((1, 5), dtype=torch.long))

        def batch_decode(
            self, tokens: torch.Tensor, **kwargs: Any
        ) -> list[str]:
            state.decoded.append((tokens.tolist(), kwargs))
            return [state.reply]

    monkeypatch.setattr(transformers, "Qwen3_5ForConditionalGeneration", Model)
    monkeypatch.setattr(transformers, "AutoProcessor", Processor)
    return state


def _image(height: int = 6, width: int = 8) -> NDArray[np.uint8]:
    return np.zeros((height, width, 3), np.uint8)


def test_config_defaults() -> None:
    config = QwenVlmConfig()

    assert config.model_id == QWEN_MODEL_ID == "Qwen/Qwen3.5-2B"
    assert config.device == "cuda"
    assert not hasattr(config, "dtype")
    assert config.max_new_tokens == 64
    assert config.repetition_penalty == 1.05


def test_config_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        QwenVlmConfig().device = "cpu"  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_id": ""},
        {"device": ""},
        {"max_new_tokens": 0},
        {"repetition_penalty": 0.9},
        {"repetition_penalty": float("nan")},
        {"repetition_penalty": float("inf")},
    ],
)
def test_config_rejects_invalid_values(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        QwenVlmConfig(**overrides)


def test_the_adapter_satisfies_the_port() -> None:
    assert isinstance(QwenVlm(), VisionLanguageModel)


def test_invalid_images_are_rejected_before_the_model_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(self: QwenVlm) -> None:
        raise AssertionError("the model must not load")

    monkeypatch.setattr(QwenVlm, "load", fail)

    with pytest.raises(ValueError, match="image"):
        QwenVlm(_CPU).generate([np.zeros((4, 4), np.uint8)], "p")


def test_nothing_is_loaded_before_first_use(hub: Hub) -> None:
    vlm = QwenVlm(_CPU)

    assert vlm.loaded_model_id is None
    assert hub.requested == []


def test_load_requests_the_configured_model_and_moves_it_to_the_device(
    hub: Hub,
) -> None:
    vlm = QwenVlm(QwenVlmConfig(device="cpu"))

    vlm.load()

    assert vlm.loaded_model_id == QWEN_MODEL_ID
    assert hub.requested == [QWEN_MODEL_ID]
    assert hub.model_kwargs == [{"dtype": torch.bfloat16}]
    assert hub.models[0].eval_called
    assert hub.models[0].device == "cpu"


def test_loading_twice_loads_once(hub: Hub) -> None:
    vlm = QwenVlm(_CPU)

    vlm.load()
    vlm.load()

    assert len(hub.models) == 1
    assert len(hub.processors) == 1


def test_a_load_error_is_raised_and_leaves_the_adapter_unloaded(
    hub: Hub,
) -> None:
    hub.failing[QWEN_MODEL_ID] = OSError("offline")
    vlm = QwenVlm(_CPU)

    with pytest.raises(OSError, match="offline"):
        vlm.load()

    assert vlm.loaded_model_id is None
    assert hub.requested == [QWEN_MODEL_ID]


def test_generate_returns_the_stripped_reply_of_the_new_tokens(
    hub: Hub,
) -> None:
    vlm = QwenVlm(_CPU)

    assert vlm.generate([_image()], "describe") == "a reply"

    assert hub.decoded == [([_REPLY_TOKENS], {"skip_special_tokens": True})]


def test_the_conversation_holds_the_images_then_the_prompt(hub: Hub) -> None:
    vlm = QwenVlm(_CPU)

    vlm.generate([_image(6, 8), _image(3, 5)], "describe")

    processor = hub.processors[0]
    (conversation,) = processor.conversations
    (message,) = conversation
    assert message["role"] == "user"
    first, second, text = message["content"]
    assert first["type"] == second["type"] == "image"
    assert isinstance(first["image"], Image.Image)
    assert first["image"].size == (8, 6)
    assert second["image"].size == (5, 3)
    assert text == {"type": "text", "text": "describe"}
    kwargs = processor.template_kwargs[0]
    assert kwargs["add_generation_prompt"] is True
    assert kwargs["tokenize"] is True
    assert kwargs["return_dict"] is True
    assert kwargs["enable_thinking"] is False


def test_no_images_gives_a_text_only_conversation(hub: Hub) -> None:
    vlm = QwenVlm(_CPU)

    vlm.generate([], "just text")

    (message,) = hub.processors[0].conversations[0]
    assert message["content"] == [{"type": "text", "text": "just text"}]


def test_generation_is_greedy_and_bounded_without_gradients(hub: Hub) -> None:
    config = QwenVlmConfig(
        device="cpu",
        max_new_tokens=17,
        repetition_penalty=1.3,
    )
    vlm = QwenVlm(config)

    vlm.generate([_image()], "describe")

    model = hub.models[0]
    call = model.generate_calls[0]
    assert call["max_new_tokens"] == 17
    assert call["do_sample"] is False
    assert "repetition_penalty" not in call
    assert model.modes == [True]


def test_the_repetition_penalty_ignores_the_prompt_tokens(hub: Hub) -> None:
    from transformers import RepetitionPenaltyLogitsProcessor

    vlm = QwenVlm(QwenVlmConfig(device="cpu"))

    vlm.generate([_image()], "describe")

    (processor,) = hub.models[0].generate_calls[0]["logits_processor"]
    assert isinstance(processor, RepetitionPenaltyLogitsProcessor)
    assert processor.penalty == 1.05
    assert processor.prompt_ignore_length == PROMPT_TOKENS


def test_no_penalty_means_no_processor(hub: Hub) -> None:
    config = QwenVlmConfig(device="cpu", repetition_penalty=1.0)

    QwenVlm(config).generate([], "describe")

    assert list(hub.models[0].generate_calls[0]["logits_processor"]) == []


def test_unload_releases_the_model_and_a_later_call_reloads(hub: Hub) -> None:
    vlm = QwenVlm(_CPU)
    vlm.load()

    vlm.unload()

    assert vlm.loaded_model_id is None
    vlm.generate([], "again")
    assert len(hub.models) == 2


def test_unload_without_a_model_does_nothing(hub: Hub) -> None:
    vlm = QwenVlm(_CPU)

    vlm.unload()

    assert vlm.loaded_model_id is None
    assert hub.requested == []


def test_the_unloaded_model_can_be_collected(hub: Hub) -> None:
    import gc
    import weakref

    vlm = QwenVlm(_CPU)
    vlm.load()
    reference = weakref.ref(hub.models[0])
    hub.models.clear()

    vlm.unload()
    gc.collect()

    assert reference() is None


def test_the_weights_dtype_is_not_configurable() -> None:
    with pytest.raises(TypeError):
        QwenVlmConfig(dtype="float32")  # type: ignore[call-arg]
