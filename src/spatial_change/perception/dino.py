"""DINO patch features behind the ``FeatureEncoder`` port.

torch and transformers are imported inside the methods that need them, so
importing this module stays cheap and the configuration can be built anywhere.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spatial_change.perception.loading import (
    inference_context,
    load_with_fallback,
)
from spatial_change.perception.padding import check_image, pad_to_patches

DINOV3_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
"""Default backbone: DINOv3 ViT-S/16, a gated repository."""

DINOV2_MODEL_ID = "facebook/dinov2-base"
"""Open backbone used when the default cannot be loaded."""

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DinoConfig:
    """Settings of the DINO feature encoder.

    Attributes:
        model_id: Hugging Face repository of the weights.
        fallback_model_id: Repository tried when ``model_id`` raises an
            ``OSError`` (gated, missing, or offline); None disables the
            fallback.
        device: torch device the model runs on.

    Raises:
        ValueError: If a repository id or the device is empty.
    """

    model_id: str = DINOV3_MODEL_ID
    fallback_model_id: str | None = DINOV2_MODEL_ID
    device: str = "cuda"

    def __post_init__(self) -> None:
        """Validates the settings."""
        if not self.model_id or not self.device:
            raise ValueError("model_id and device must not be empty")
        if self.fallback_model_id == "":
            raise ValueError("fallback_model_id must not be empty")


class DinoFeatureEncoder:
    """Encodes images into grids of DINO patch features.

    The weights load on the first ``encode`` or ``patch_size`` call and are
    freed by ``unload``; a later call loads them again. Images are padded at
    the bottom and right by edge replication to whole patches, so feature
    ``(i, j)`` covers input rows ``i * p`` to ``(i + 1) * p - 1`` and the
    matching columns, as the port requires.
    """

    def __init__(self, config: DinoConfig | None = None) -> None:
        """Creates an encoder without loading any weights.

        Args:
            config: Model and device settings; the defaults if None.
        """
        self._config = config or DinoConfig()
        self._model: Any = None
        self._loaded_id: str | None = None

    @property
    def loaded_model_id(self) -> str | None:
        """Repository of the loaded weights, None before loading."""
        return self._loaded_id

    @property
    def patch_size(self) -> int:
        """Side length in input pixels of the square patch per feature.

        Loads the model if needed, because the patch size is a property of
        the weights that were loaded.
        """
        self.load()
        return int(self._model.config.patch_size)

    def encode(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Encodes an image into a grid of patch features.

        Args:
            image: ``(H, W, 3)`` uint8 RGB image with ``H, W >= 1``.

        Returns:
            ``(ceil(H / p), ceil(W / p), C)`` float32 features, the class
            and register tokens dropped.

        Raises:
            ValueError: If ``image`` is not a non-empty RGB uint8 array.
            OSError: If neither the model nor its fallback can be loaded.
        """
        import torch

        check_image(image)
        self.load()
        model = self._model
        device = torch.device(self._config.device)
        patch = int(model.config.patch_size)
        padded = pad_to_patches(image, patch)
        rows, cols = padded.shape[0] // patch, padded.shape[1] // patch
        pixels = (
            torch.from_numpy(np.ascontiguousarray(padded))
            .to(device)
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
        )
        mean = torch.tensor(_IMAGENET_MEAN, device=device).view(3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=device).view(3, 1, 1)
        batch = ((pixels - mean) / std).unsqueeze(0)
        with inference_context(device):
            tokens = model(pixel_values=batch).last_hidden_state
        prefix = 1 + int(getattr(model.config, "num_register_tokens", 0))
        grid = tokens[0, prefix:].float().reshape(rows, cols, -1)
        return np.asarray(grid.cpu().numpy(), dtype=np.float32)

    def load(self) -> None:
        """Loads the weights if they are not loaded yet.

        ``encode`` calls this on first use; call it directly to keep the
        load time out of a measurement.

        Raises:
            OSError: If the model cannot be loaded and there is no
                fallback, or the fallback fails too (chained to its error,
                with the first error as its context).
        """
        if self._model is not None:
            return
        model, loaded_id = load_with_fallback(
            self._config.model_id,
            self._config.fallback_model_id,
            self._from_pretrained,
        )
        self._model = model
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
        self._loaded_id = None
        gc.collect()
        torch.cuda.empty_cache()

    def _from_pretrained(self, model_id: str) -> Any:
        """Loads one repository onto the configured device in eval mode.

        The weights load in float32 whatever dtype the checkpoint stores,
        so CPU inference matches the float32 input and CUDA runs under
        autocast.

        Args:
            model_id: Hugging Face repository to load.

        Returns:
            The model.
        """
        import torch
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_id, dtype=torch.float32)
        return model.eval().to(self._config.device)
