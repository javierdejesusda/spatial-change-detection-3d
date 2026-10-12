"""Model loading and inference settings shared by the torch adapters.

torch is imported inside the function that needs it, so importing this module
stays cheap.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any


def load_with_fallback[T](
    model_id: str, fallback_id: str | None, build: Callable[[str], T]
) -> tuple[T, str]:
    """Builds a model from a repository, or from the fallback on ``OSError``.

    Args:
        model_id: Hugging Face repository tried first.
        fallback_id: Repository tried when ``model_id`` raises an
            ``OSError`` (gated, missing, or offline); None disables the
            fallback.
        build: Loads the model of one repository.

    Returns:
        The model and the repository it came from.

    Raises:
        OSError: If the model cannot be loaded and there is no fallback, or
            the fallback fails too (chained to its error, with the first
            error as its context).
    """
    try:
        return build(model_id), model_id
    except OSError:
        if fallback_id is None:
            raise
        try:
            return build(fallback_id), fallback_id
        except OSError as error:
            raise OSError(
                f"could not load {model_id} or the fallback {fallback_id}"
            ) from error


@contextmanager
def inference_context(device: Any) -> Iterator[None]:
    """Runs inference without gradients, in bf16 autocast on CUDA.

    Args:
        device: The ``torch.device`` the model runs on.

    Yields:
        Nothing; the settings hold inside the block.
    """
    import torch

    with (
        torch.inference_mode(),
        torch.autocast(
            device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ),
    ):
        yield
