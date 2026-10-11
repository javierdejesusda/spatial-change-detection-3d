"""Alignment of epochs: gravity and scale, then change-aware registration."""

from spatial_change.alignment.refine import (
    RefineConfig,
    Refinement,
    refine_alignment,
)
from spatial_change.alignment.scale import (
    RoomAlignment,
    ScalePolicy,
    estimate_scale,
    gravity_align,
)

__all__ = [
    "RefineConfig",
    "Refinement",
    "RoomAlignment",
    "ScalePolicy",
    "estimate_scale",
    "gravity_align",
    "refine_alignment",
]
