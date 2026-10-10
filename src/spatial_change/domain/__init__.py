"""Core domain types for spatial change detection.

See the module docstrings for the shared conventions: OpenCV pinhole
cameras, ``world_from_camera`` (camera-to-world) poses, and array layouts.
"""

from spatial_change.domain.camera import Epoch, Frame, Intrinsics
from spatial_change.domain.report import (
    BoxModel,
    Change,
    ChangeReport,
    ChangeType,
    EvidenceCounts,
    UnobservedRegion,
)
from spatial_change.domain.scene import (
    EpochReconstruction,
    ObjectInstance,
    OrientedBox,
)
from spatial_change.domain.transforms import SE3, Sim3

__all__ = [
    "SE3",
    "BoxModel",
    "Change",
    "ChangeReport",
    "ChangeType",
    "Epoch",
    "EpochReconstruction",
    "EvidenceCounts",
    "Frame",
    "Intrinsics",
    "ObjectInstance",
    "OrientedBox",
    "Sim3",
    "UnobservedRegion",
]
