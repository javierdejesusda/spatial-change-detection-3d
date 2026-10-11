"""Helpers that turn synthetic scenes into appearance-cue inputs."""

import dataclasses
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain import Epoch, EpochReconstruction, Frame

_LUMA = np.array([0.299, 0.587, 0.114])
_BLUE = np.array([0.0, 0.1, 1.0])


def recolor_blue(
    image: NDArray[np.uint8], mask: NDArray[np.bool_]
) -> NDArray[np.uint8]:
    """Recolors the masked pixels blue, keeping their luminance.

    Pixels that cannot reach their luminance in saturated blue clip.

    Args:
        image: ``(H, W, 3)`` uint8 RGB image.
        mask: ``(H, W)`` pixels to recolor.

    Returns:
        A recolored copy of the image.
    """
    luminance = image.astype(np.float64) @ _LUMA
    scale = luminance / (_BLUE @ _LUMA)
    blue = np.clip(scale[..., None] * _BLUE, 0, 255)
    out = image.copy()
    out[mask] = np.rint(blue[mask]).astype(np.uint8)
    return out


def recolored_frames(
    frames: Sequence[Frame],
    instance_maps: Sequence[NDArray[np.int32]],
    index: int,
) -> list[Frame]:
    """Recolors one object blue in every frame, leaving the geometry alone.

    Args:
        frames: Frames with images.
        instance_maps: Per-frame ``(H, W)`` object index maps.
        index: Object index to recolor.

    Returns:
        Copies of the frames with the object's pixels recolored.
    """
    result = []
    for frame, instances in zip(frames, instance_maps, strict=True):
        assert frame.image is not None
        image = recolor_blue(frame.image, instances == index)
        result.append(dataclasses.replace(frame, image=image))
    return result


def reconstruction(
    epoch: Epoch, frames: Sequence[Frame], points: NDArray[np.float64]
) -> EpochReconstruction:
    """Wraps frames and points in a reconstruction with neutral colors.

    Args:
        epoch: The epoch of the frames.
        frames: The frames.
        points: World points of shape ``(N, 3)``.

    Returns:
        The reconstruction.
    """
    count = len(points)
    return EpochReconstruction(
        epoch=epoch,
        frames=list(frames),
        points=points,
        colors=np.zeros((count, 3), np.uint8),
        confidence=np.ones(count, np.float32),
    )
