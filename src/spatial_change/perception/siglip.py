"""SigLIP open-vocabulary labels behind the ``Labeler`` port.

torch and transformers are imported inside the methods that need them, so
importing this module stays cheap and the configuration can be built anywhere.
"""

from __future__ import annotations

import gc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spatial_change.perception.loading import (
    inference_context,
    load_with_fallback,
)
from spatial_change.perception.prompts import check_labeling_inputs

SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-224"
"""Default model: SigLIP 2 base with 224 px inputs."""

SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
"""Model used when the default cannot be loaded."""

_TEXT_LENGTH = 64
"""Token length the SigLIP checkpoints were trained with."""


@dataclass(frozen=True)
class SiglipConfig:
    """Settings of the SigLIP labeler.

    Attributes:
        model_id: Hugging Face repository of the weights.
        fallback_model_id: Repository tried when ``model_id`` raises an
            ``OSError`` (gated, missing, or offline); None disables the
            fallback.
        device: torch device the model runs on.
        prompt: Template that turns a vocabulary entry into the text the
            model sees; it holds one ``{}`` for the entry.
        batch_size: Crops encoded per forward pass, at least 1.

    Raises:
        ValueError: If a repository id, the device, or the prompt is empty,
            the prompt does not hold exactly one ``{}``, or the batch size
            is below 1.
    """

    model_id: str = SIGLIP2_MODEL_ID
    fallback_model_id: str | None = SIGLIP_MODEL_ID
    device: str = "cuda"
    prompt: str = "a photo of a {}."
    batch_size: int = 16

    def __post_init__(self) -> None:
        """Validates the settings."""
        if not self.model_id or not self.device:
            raise ValueError("model_id and device must not be empty")
        if self.fallback_model_id == "":
            raise ValueError("fallback_model_id must not be empty")
        if self.prompt.count("{}") != 1:
            raise ValueError("prompt must hold exactly one {}")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")


class SiglipLabeler:
    """Labels image crops with the closest entry of a vocabulary.

    The score of a label is its share of the softmax over the vocabulary
    of the scaled cosines, ``exp(logit_scale) * cosine``, so the scores of
    one crop sum to 1 and their scale depends on the vocabulary size: the
    same match scores lower among more entries. The model's ``logit_bias``
    shifts every logit alike and cancels in the softmax. The text features
    of a vocabulary are computed once and reused while the same vocabulary
    comes back.

    The weights load on the first ``label`` call and are freed by
    ``unload``; a later call loads them again.
    """

    def __init__(self, config: SiglipConfig | None = None) -> None:
        """Creates a labeler without loading any weights.

        Args:
            config: Model, device, and prompt settings; the defaults if
                None.
        """
        self._config = config or SiglipConfig()
        self._model: Any = None
        self._processor: Any = None
        self._loaded_id: str | None = None
        self._text: tuple[tuple[str, ...], Any] | None = None

    @property
    def loaded_model_id(self) -> str | None:
        """Repository of the loaded weights, None before loading."""
        return self._loaded_id

    def label(
        self, crops: Sequence[NDArray[np.uint8]], vocabulary: Sequence[str]
    ) -> list[tuple[str, float]]:
        """Labels each crop with its best vocabulary entry.

        Args:
            crops: ``(H, W, 3)`` uint8 RGB crops of any size.
            vocabulary: Non-empty, unique label names.

        Returns:
            One ``(label, score)`` per crop in order, the label from the
            vocabulary (the earlier entry on a tie) and the score in
            ``[0, 1]``. Empty when there are no crops.

        Raises:
            ValueError: If a crop or the vocabulary is invalid.
            OSError: If neither the model nor its fallback can be loaded.
        """
        check_labeling_inputs(crops, vocabulary)
        if not crops:
            return []
        import torch

        self.load()
        device = torch.device(self._config.device)
        names = tuple(vocabulary)
        text = self._text_features(names, device)
        scale = self._model.logit_scale.detach().exp().float()
        size = self._config.batch_size
        results: list[tuple[str, float]] = []
        for start in range(0, len(crops), size):
            batch = [
                np.ascontiguousarray(c) for c in crops[start : start + size]
            ]
            inputs = self._processor(images=batch, return_tensors="pt")
            pixels = inputs["pixel_values"].to(device)
            with inference_context(device):
                image = self._model.get_image_features(
                    pixel_values=pixels
                ).pooler_output
            image = torch.nn.functional.normalize(image.float(), dim=-1)
            scores = (scale * image @ text.T).softmax(dim=-1)
            for row in scores.cpu().numpy():
                best = int(np.argmax(row))
                results.append((names[best], float(row[best])))
        return results

    def load(self) -> None:
        """Loads the weights if they are not loaded yet.

        ``label`` calls this on first use; call it directly to keep the
        load time out of a measurement.

        Raises:
            OSError: If the model cannot be loaded and there is no
                fallback, or the fallback fails too (chained to its error,
                with the first error as its context).
        """
        if self._model is not None:
            return
        (model, processor), loaded_id = load_with_fallback(
            self._config.model_id,
            self._config.fallback_model_id,
            self._from_pretrained,
        )
        self._model = model
        self._processor = processor
        self._loaded_id = loaded_id

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
        self._text = None
        gc.collect()
        torch.cuda.empty_cache()

    def _text_features(self, names: tuple[str, ...], device: Any) -> Any:
        """Gives the unit text features of a vocabulary, computing them once.

        Args:
            names: The vocabulary.
            device: The ``torch.device`` of the model.

        Returns:
            A ``(V, C)`` float32 tensor of unit vectors in vocabulary order.
        """
        import torch

        if self._text is not None and self._text[0] == names:
            return self._text[1]
        prompts = [self._config.prompt.format(name) for name in names]
        tokens = self._processor(
            text=prompts,
            padding="max_length",
            max_length=_TEXT_LENGTH,
            return_tensors="pt",
        )
        with inference_context(device):
            features = self._model.get_text_features(
                input_ids=tokens["input_ids"].to(device)
            ).pooler_output
        unit = torch.nn.functional.normalize(features.float(), dim=-1)
        self._text = (names, unit)
        return unit

    def _from_pretrained(self, model_id: str) -> tuple[Any, Any]:
        """Loads one repository onto the configured device in eval mode.

        The weights load in float32 whatever dtype the checkpoint stores,
        so CPU inference matches the float32 input and CUDA runs under
        autocast.

        Args:
            model_id: Hugging Face repository to load.

        Returns:
            The model and its processor.
        """
        import torch
        from transformers import AutoModel, AutoProcessor

        model = AutoModel.from_pretrained(model_id, dtype=torch.float32)
        processor_class: Any = AutoProcessor
        processor = processor_class.from_pretrained(model_id)
        return model.eval().to(self._config.device), processor
