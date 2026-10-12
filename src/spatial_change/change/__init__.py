"""Candidate change regions from visibility evidence, lifted to objects.

Importing the package does not load torch, open3d, or cv2.
"""

from spatial_change.change.appearance import (
    AppearanceConfig,
    CrossAppearance,
    PointAppearance,
    compare_appearance,
    fuse_appearance,
)
from spatial_change.change.candidates import (
    Candidate,
    CandidateConfig,
    ChangeCandidates,
    extract_candidates,
    find_candidates,
)
from spatial_change.change.lifting import (
    LiftConfig,
    LiftedObjects,
    lift_candidates,
    lift_change_candidates,
)

__all__ = [
    "AppearanceConfig",
    "Candidate",
    "CandidateConfig",
    "ChangeCandidates",
    "CrossAppearance",
    "LiftConfig",
    "LiftedObjects",
    "PointAppearance",
    "compare_appearance",
    "extract_candidates",
    "find_candidates",
    "fuse_appearance",
    "lift_candidates",
    "lift_change_candidates",
]
