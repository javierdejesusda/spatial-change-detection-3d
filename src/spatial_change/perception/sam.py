"""Promptable object masks behind the ``MaskSegmenter`` port.

torch and transformers are imported inside the methods that need them, so
importing this module stays cheap and the configuration can be built anywhere.
"""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spatial_change.perception.loading import (
    inference_context,
    load_with_fallback,
)
from spatial_change.perception.padding import check_image
from spatial_change.perception.prompts import check_prompts

SAM3_MODEL_ID = "facebook/sam3"
"""Default checkpoint: SAM 3, a gated repository."""

SAM21_MODEL_ID = "facebook/sam2.1-hiera-large"
"""Open checkpoint used when the default cannot be loaded."""

_CLASS_PREFIX_BY_MODEL_TYPE = {
    "sam3": "Sam3Tracker",
    "sam3_tracker": "Sam3Tracker",
    "sam3_video": "Sam3Tracker",
    "sam2": "Sam2",
    "sam2_video": "Sam2",
}
"""Prefix of the transformers classes that run the prompts, by model type.

The model class is the prefix and ``Model``, the processor the prefix and
``Processor``.

The hub checkpoints declare their video variant (``sam3_video``,
``sam2_video``); the image classes load the same weights.
"""

_POSITIVE_LABEL = 1


@dataclass(frozen=True)
class SamConfig:
    """Settings of the SAM segmenter.

    Attributes:
        model_id: Hugging Face repository of the weights.
        fallback_model_id: Repository tried when ``model_id`` raises an
            ``OSError`` (gated, missing, or offline); None disables the
            fallback.
        device: torch device the model runs on.
        mask_threshold: Logit above which a pixel belongs to a mask.

    Raises:
        ValueError: If a repository id or the device is empty, or the
            threshold is not finite.
    """

    model_id: str = SAM3_MODEL_ID
    fallback_model_id: str | None = SAM21_MODEL_ID
    device: str = "cuda"
    mask_threshold: float = 0.0

    def __post_init__(self) -> None:
        """Validates the settings."""
        if not self.model_id or not self.device:
            raise ValueError("model_id and device must not be empty")
        if self.fallback_model_id == "":
            raise ValueError("fallback_model_id must not be empty")
        if not math.isfinite(self.mask_threshold):
            raise ValueError("mask_threshold must be finite")


class SamSegmenter:
    """Segments objects from box and point prompts with SAM.

    The weights load on the first ``segment`` or ``load`` call and are freed
    by ``unload``; a later call loads them again. The checkpoint's model type
    selects the transformers classes, so a SAM 3 and a SAM 2.1 checkpoint run
    through the same code.
    """

    def __init__(self, config: SamConfig | None = None) -> None:
        """Creates a segmenter without loading any weights.

        Args:
            config: Model and device settings; the defaults if None.
        """
        self._config = config or SamConfig()
        self._model: Any = None
        self._processor: Any = None
        self._loaded_id: str | None = None

    @property
    def loaded_model_id(self) -> str | None:
        """Repository of the loaded weights, None before loading."""
        return self._loaded_id

    def segment(
        self,
        image: NDArray[np.uint8],
        boxes_xyxy: NDArray[np.float64],
        points_xy: NDArray[np.float64] | None,
    ) -> NDArray[np.bool_]:
        """Predicts one mask per box prompt.

        Args:
            image: ``(H, W, 3)`` uint8 RGB image.
            boxes_xyxy: ``(K, 4)`` float64 boxes ``(x0, y0, x1, y1)`` in
                pixels, with ``x0 <= x1`` and ``y0 <= y1``.
            points_xy: Optional ``(K, 2)`` float64 positive point prompts in
                pixels, one per box, as ``(x, y)``.

        Returns:
            ``(K, H, W)`` boolean masks in prompt order; ``(0, H, W)`` for
            no boxes, without loading the model.

        Raises:
            ValueError: If an input has the wrong shape, dtype, or values.
            OSError: If neither the model nor its fallback can be loaded.
        """
        check_image(image)
        check_prompts(boxes_xyxy, points_xy)
        height, width = image.shape[:2]
        count = len(boxes_xyxy)
        if count == 0:
            return np.zeros((0, height, width), np.bool_)
        import torch

        self.load()
        device = torch.device(self._config.device)
        prompts: dict[str, Any] = {
            "input_boxes": [boxes_xyxy.tolist()],
            "input_points": None,
            "input_labels": None,
        }
        if points_xy is not None:
            prompts["input_points"] = [
                [[point] for point in points_xy.tolist()]
            ]
            prompts["input_labels"] = [[[_POSITIVE_LABEL]] * count]
        inputs = self._processor(
            images=np.ascontiguousarray(image),
            return_tensors="pt",
            **prompts,
        ).to(device)
        with inference_context(device):
            outputs = self._model(**inputs, multimask_output=False)
        masks = self._processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            mask_threshold=self._config.mask_threshold,
            binarize=True,
        )
        return np.asarray(masks[0][:, 0].cpu().numpy(), dtype=np.bool_)

    def load(self) -> None:
        """Loads the weights if they are not loaded yet.

        ``segment`` calls this on first use; call it directly to keep the
        load time out of a measurement.

        Raises:
            OSError: If the model cannot be loaded and there is no
                fallback, or the fallback fails too (chained to its error,
                with the first error as its context).
            ValueError: If the checkpoint is not a SAM 2 or SAM 3 model.
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
        gc.collect()
        torch.cuda.empty_cache()

    def _from_pretrained(self, model_id: str) -> tuple[Any, Any]:
        """Loads one repository onto the configured device in eval mode.

        The weights load in float32 whatever dtype the checkpoint stores, so
        CPU inference matches the float32 input and CUDA runs under
        autocast.

        Args:
            model_id: Hugging Face repository to load.

        Returns:
            The model and its processor.

        Raises:
            ValueError: If the checkpoint's model type is not supported.
        """
        import torch
        import transformers

        model_type = transformers.AutoConfig.from_pretrained(
            model_id
        ).model_type
        prefix = _CLASS_PREFIX_BY_MODEL_TYPE.get(model_type)
        if prefix is None:
            raise ValueError(
                f"{model_id} has unsupported model type {model_type!r}; "
                f"expected one of {sorted(_CLASS_PREFIX_BY_MODEL_TYPE)}"
            )
        model_class = getattr(transformers, prefix + "Model")
        processor_class = getattr(transformers, prefix + "Processor")
        model = model_class.from_pretrained(model_id, dtype=torch.float32)
        processor = processor_class.from_pretrained(model_id)
        return model.eval().to(self._config.device), processor
