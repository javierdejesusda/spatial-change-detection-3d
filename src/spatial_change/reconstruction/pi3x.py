"""Pi3X, the default reconstruction backbone, behind the ``Reconstructor`` port.

torch and pi3 are imported inside the methods that need them, so importing
this module stays cheap and the configuration can be built anywhere.
"""

from __future__ import annotations

import functools
import gc
import types
import warnings
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from spatial_change.capture.resize import resize_image
from spatial_change.domain.transforms import check_array
from spatial_change.ports import ReconstructionResult
from spatial_change.reconstruction.conversion import (
    PATCH_SIZE,
    to_reconstruction_result,
    working_size,
)

if TYPE_CHECKING:
    import torch

_MIN_PIXEL_LIMIT = PATCH_SIZE * PATCH_SIZE
_BFLOAT16_MODULES = (
    "encoder",
    "decoder",
    "point_decoder",
    "camera_decoder",
    "conf_decoder",
    "metric_decoder",
)


@dataclass(frozen=True)
class Pi3XConfig:
    """Settings of the Pi3X adapter.

    Attributes:
        model_id: Hugging Face repository of the weights.
        pixel_limit: Pixel count the input images are resized to, keeping
            the aspect ratio. The default gives 518x294 for 16:9 input.
        max_frames: Most views one ``reconstruct`` call accepts. All views
            attend to each other, so memory grows with this ceiling.
        head_chunk_size: Views the convolutional heads process at once.
            Smaller values trade speed for lower peak memory.
        device: torch device the model runs on.
        cap_allocator_at_free_vram: On CUDA, limit the process to the VRAM
            that is free at load time plus what the process already holds.
            Windows lets the driver spill to system RAM instead of failing,
            which makes a run extremely slow; the cap turns that into an
            out-of-memory error. The fraction is process-wide: it also
            limits every other model in this process until ``unload`` or a
            failed load sets it back to 1.0, which in turn lifts a cap that
            another adapter set.

    Raises:
        ValueError: If a field is empty or below its minimum.
    """

    model_id: str = "yyfz233/Pi3X"
    pixel_limit: int = 153_000
    max_frames: int = 128
    head_chunk_size: int = 8
    device: str = "cuda"
    cap_allocator_at_free_vram: bool = True

    def __post_init__(self) -> None:
        """Validates the settings."""
        if not self.model_id or not self.device:
            raise ValueError("model_id and device must not be empty")
        if self.pixel_limit < _MIN_PIXEL_LIMIT:
            raise ValueError(
                f"pixel_limit must be at least {_MIN_PIXEL_LIMIT}, "
                f"got {self.pixel_limit}"
            )
        if self.max_frames < 1 or self.head_chunk_size < 1:
            raise ValueError("max_frames and head_chunk_size must be positive")


class Pi3XReconstructor:
    """Predicts metric camera poses, intrinsics, and depth with Pi3X.

    The weights load on the first ``reconstruct`` call and are freed by
    ``unload``; a later call loads them again. All views of one call share a
    world frame whose scale is metric.
    """

    def __init__(self, config: Pi3XConfig | None = None) -> None:
        """Creates an adapter without loading any weights.

        Args:
            config: Model and memory settings; the defaults if None.
        """
        self._config = config or Pi3XConfig()
        self._model: Any = None
        self._device: torch.device | None = None
        self._capped_device: torch.device | None = None

    def reconstruct(
        self, images: Sequence[NDArray[np.uint8]]
    ) -> ReconstructionResult:
        """Reconstructs a set of views of one scene.

        Args:
            images: ``(H, W, 3)`` uint8 RGB images, all the same size.

        Returns:
            One entry per image at the working resolution; the intrinsics
            match the depth maps.

        Raises:
            ValueError: If there are no images, more than ``max_frames``,
                or they are not same-sized RGB uint8 arrays.
            torch.cuda.OutOfMemoryError: If the views do not fit in the VRAM
                that was free at load time.
        """
        import torch

        self._validate(images)
        self.load()
        config = self._config
        device = self._device
        assert device is not None
        height, width = images[0].shape[:2]
        work_width, work_height = working_size(
            width, height, config.pixel_limit
        )
        batch = np.stack(
            [resize_image(image, work_width, work_height) for image in images]
        )
        views = (
            torch.from_numpy(batch)
            .to(device)
            .permute(0, 3, 1, 2)
            .contiguous()
            .float()
            .div_(255.0)
            .unsqueeze(0)
        )
        with (
            torch.inference_mode(),
            torch.autocast(device.type, dtype=torch.bfloat16),
        ):
            out = self._model(views)
        return to_reconstruction_result(
            out["camera_poses"][0].float().cpu().numpy(),
            out["local_points"][0].float().cpu().numpy(),
            out["conf"][0].float().cpu().numpy(),
        )

    def load(self) -> None:
        """Loads the weights if they are not loaded yet.

        ``reconstruct`` calls this on first use; call it directly to keep
        the load time out of a measurement. If the load fails, the VRAM cap
        is lifted again before the error propagates.
        """
        if self._model is not None:
            return
        device = _resolve_device(self._config.device)
        if self._config.cap_allocator_at_free_vram and device.type == "cuda":
            self._cap_allocator(device)
        try:
            self._model = self._build_model(device)
        except BaseException:
            self._lift_allocator_cap()
            raise
        self._device = device

    def unload(self) -> None:
        """Releases the model weights, cached GPU memory, and the VRAM cap.

        Does nothing if the model is not loaded. A later ``reconstruct``
        loads the weights again.
        """
        if self._model is None:
            return
        import torch

        self._model = None
        gc.collect()
        torch.cuda.empty_cache()
        self._lift_allocator_cap()

    def _cap_allocator(self, device: torch.device) -> None:
        import torch

        free, total = torch.cuda.mem_get_info(device)
        reserved = torch.cuda.memory_reserved(device)
        torch.cuda.set_per_process_memory_fraction(
            (free + reserved) / total, device
        )
        self._capped_device = device

    def _lift_allocator_cap(self) -> None:
        if self._capped_device is None:
            return
        import torch

        torch.cuda.set_per_process_memory_fraction(1.0, self._capped_device)
        self._capped_device = None

    def _build_model(self, device: torch.device) -> Any:
        import torch
        from pi3.models.pi3x import Pi3X

        config = self._config
        _use_working_attention_kernel(device)
        model = Pi3X.from_pretrained(config.model_id).eval()
        model.disable_multimodal(free_cuda_cache=False)
        for name in _BFLOAT16_MODULES:
            getattr(model, name).to(torch.bfloat16)
        model.register_token.data = model.register_token.data.to(torch.bfloat16)
        model.metric_token.data = model.metric_token.data.to(torch.bfloat16)
        model.to(device)
        # A weak proxy keeps the model out of a reference cycle, so dropping
        # the last reference frees the weights without a garbage collection.
        model._chunked_conv_head = functools.partial(
            type(model)._chunked_conv_head,
            weakref.proxy(model),
            chunk_size=config.head_chunk_size,
        )
        return model

    def _validate(self, images: Sequence[NDArray[np.uint8]]) -> None:
        if not images:
            raise ValueError("need at least one image")
        if len(images) > self._config.max_frames:
            raise ValueError(
                f"got {len(images)} images, above "
                f"max_frames={self._config.max_frames}"
            )
        for image in images:
            check_array(image, (None, None, 3), np.uint8, "image")
        if len({image.shape for image in images}) != 1:
            raise ValueError("all images must have the same size")


def _use_working_attention_kernel(device: torch.device) -> None:
    """Makes Pi3's bfloat16 attention use a kernel that runs on ``device``.

    Pi3 forces the flash kernel for bfloat16. Whether that kernel runs
    depends on the device, not only on the torch build: a CPU runs it
    regardless of the build flag, while a GPU older than sm80 fails even on
    a wheel built with flash attention. So each kernel is probed on the
    device, flash first and then memory-efficient. Pi3's flash name is
    pointed at the first kernel that ran, or at the math kernel if none did.
    Nothing changes when flash runs.

    Only the flash name is redirected, because Pi3's bfloat16 branch asks for
    flash alone; its float32 branch allows the math and memory-efficient
    kernels, which keep their own names. Only the backend names inside Pi3's
    attention module are swapped; torch itself is untouched. The swap is
    process-wide for Pi3: every Pi3 model in this process uses it from then
    on, whatever device it loads on.

    Args:
        device: Device the model will run on.
    """
    from torch.nn.attention import SDPBackend

    if _kernel_runs(SDPBackend.FLASH_ATTENTION, device):
        return
    fallback = (
        SDPBackend.EFFICIENT_ATTENTION
        if _kernel_runs(SDPBackend.EFFICIENT_ATTENTION, device)
        else SDPBackend.MATH
    )
    import pi3.models.layers.attention as pi3_attention

    pi3_attention.SDPBackend = types.SimpleNamespace(
        **{**SDPBackend.__members__, "FLASH_ATTENTION": fallback}
    )


def _kernel_runs(backend: Any, device: torch.device) -> bool:
    """Returns whether a tiny bfloat16 attention runs on ``backend``.

    Args:
        backend: The ``torch.nn.attention.SDPBackend`` to restrict to.
        device: Device the probe tensors are created on.

    Returns:
        True if the call ran, False if torch raised a ``RuntimeError``
        because ``backend`` has no usable kernel here.
    """
    import torch
    from torch.nn.attention import sdpa_kernel

    query = torch.zeros(1, 1, 8, 64, dtype=torch.bfloat16, device=device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            with sdpa_kernel(backend):
                torch.nn.functional.scaled_dot_product_attention(
                    query, query, query
                )
        except RuntimeError:
            return False
    return True


def _resolve_device(name: str) -> torch.device:
    """Returns the device for ``name``, with the index of a bare CUDA device.

    torch's memory queries reject ``cuda`` without an index.
    """
    import torch

    device = torch.device(name)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device
