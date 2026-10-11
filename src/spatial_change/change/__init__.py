"""Candidate change regions from visibility evidence.

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

__all__ = [
    "AppearanceConfig",
    "Candidate",
    "CandidateConfig",
    "ChangeCandidates",
    "CrossAppearance",
    "PointAppearance",
    "compare_appearance",
    "extract_candidates",
    "find_candidates",
    "fuse_appearance",
]
