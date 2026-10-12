"""Qwen3.5 vision-language model behind the ``VisionLanguageModel`` port.

torch and transformers are imported inside the methods that need them, so
importing this module stays cheap and the configuration can be built anywhere.
"""

from __future__ import annotations

import gc
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spatial_change.perception.loading import inference_context
from spatial_change.perception.padding import check_image

QWEN_MODEL_ID = "Qwen/Qwen3.5-2B"
"""Default checkpoint: Qwen3.5-2B, an ungated Apache-2.0 repository."""


@dataclass(frozen=True)
class QwenVlmConfig:
    """Settings of the Qwen vision-language model.

    The weights are bfloat16 (about 4.2 GB): that is the dtype the inference
    context runs in, and float32 would not fit an 8 GB GPU.

    Decoding is greedy. Without a repetition penalty greedy decoding can
    loop on one token until the length limit, so the default penalty is
    above 1. The penalty applies to the generated tokens only: the prompt
    holds the labels and numbers the reply should repeat.

    Attributes:
        model_id: Hugging Face repository of the weights.
        device: torch device the model runs on.
        max_new_tokens: Longest reply, in tokens.
        repetition_penalty: Penalty of repeated tokens, at least 1.

    Raises:
        ValueError: If the repository id or the device is empty, the token
            limit is below 1, or the penalty is below 1 or not finite.
    """

    model_id: str = QWEN_MODEL_ID
    device: str = "cuda"
    max_new_tokens: int = 64
    repetition_penalty: float = 1.05

    def __post_init__(self) -> None:
        """Validates the settings."""
        if not self.model_id or not self.device:
            raise ValueError("model_id and device must not be empty")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if (
            not math.isfinite(self.repetition_penalty)
            or self.repetition_penalty < 1.0
        ):
            raise ValueError("repetition_penalty must be finite and >= 1")


class QwenVlm:
    """Answers a prompt about a few images with Qwen3.5.

    The weights (about 4.2 GB in bfloat16) load on the first ``generate`` or
    ``load`` call and are freed by ``unload``; a later call loads them
    again. Loading is the caller's responsibility to schedule: on an 8 GB
    GPU it fits only when every other model has been unloaded. Thinking is
    switched off in the chat template.
    """

    def __init__(self, config: QwenVlmConfig | None = None) -> None:
        """Creates the adapter without loading any weights.

        Args:
            config: Model and decoding settings; the defaults if None.
        """
        self._config = config or QwenVlmConfig()
        self._model: Any = None
        self._processor: Any = None
        self._loaded_id: str | None = None

    @property
    def loaded_model_id(self) -> str | None:
        """Repository of the loaded weights, None before loading."""
        return self._loaded_id

    def generate(self, images: Sequence[NDArray[np.uint8]], prompt: str) -> str:
        """Generates a reply to a prompt about images.

        Args:
            images: ``(h, w, 3)`` uint8 RGB images, each of any size; may
                be empty. They precede the prompt in the conversation.
            prompt: The instruction.

        Returns:
            The decoded reply, special tokens removed and whitespace
            stripped.

        Raises:
            ValueError: If an image is not a non-empty RGB uint8 array.
            OSError: If the model cannot be loaded.
        """
        import torch
        from PIL import Image
        from transformers import (
            LogitsProcessorList,
            RepetitionPenaltyLogitsProcessor,
        )

        for image in images:
            check_image(image)
        self.load()
        device = torch.device(self._config.device)
        content: list[dict[str, Any]] = [
            {"type": "image", "image": Image.fromarray(image)}
            for image in images
        ]
        content.append({"type": "text", "text": prompt})
        inputs = self._processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(device)
        prompt_length = int(inputs["input_ids"].shape[1])
        penalties = LogitsProcessorList()
        if self._config.repetition_penalty > 1.0:
            penalties.append(
                RepetitionPenaltyLogitsProcessor(
                    self._config.repetition_penalty,
                    prompt_ignore_length=prompt_length,
                )
            )
        with inference_context(device):
            output = self._model.generate(
                **inputs,
                max_new_tokens=self._config.max_new_tokens,
                logits_processor=penalties,
                do_sample=False,
            )
        new_tokens = output[:, prompt_length:]
        reply = self._processor.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0]
        return str(reply).strip()

    def load(self) -> None:
        """Loads the weights if they are not loaded yet.

        ``generate`` calls this on first use; call it directly to keep the
        load time out of a measurement. Calling it again does nothing.

        Raises:
            OSError: If the model cannot be loaded (missing, gated, or
                offline).
        """
        if self._model is not None:
            return
        import torch
        import transformers

        model_id = self._config.model_id
        processor_class: Any = transformers.AutoProcessor
        model_class: Any = transformers.Qwen3_5ForConditionalGeneration
        processor = processor_class.from_pretrained(model_id)
        model = model_class.from_pretrained(model_id, dtype=torch.bfloat16)
        self._processor = processor
        self._model = model.eval().to(self._config.device)
        self._loaded_id = model_id

    def unload(self) -> None:
        """Releases the model weights and cached GPU memory.

        Does nothing if the model is not loaded. A later call loads the
        weights again.
        """
        if self._model is None:
            return
        import torch

        self._model = None
        self._processor = None
        self._loaded_id = None
        gc.collect()
        torch.cuda.empty_cache()
