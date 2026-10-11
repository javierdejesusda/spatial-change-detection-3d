"""Candidate change regions from visibility evidence.

Importing the package does not load torch, open3d, or cv2.
"""

from spatial_change.change.candidates import (
    Candidate,
    CandidateConfig,
    ChangeCandidates,
    extract_candidates,
    find_candidates,
)

__all__ = [
    "Candidate",
    "CandidateConfig",
    "ChangeCandidates",
    "extract_candidates",
    "find_candidates",
]
