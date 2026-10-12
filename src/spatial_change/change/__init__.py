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
from spatial_change.change.assignment import (
    gated_assignment,
    linear_assignment,
)
from spatial_change.change.candidates import (
    Candidate,
    CandidateConfig,
    ChangeCandidates,
    extract_candidates,
    find_candidates,
)
from spatial_change.change.descriptors import (
    DescriptorConfig,
    describe_instances,
)
from spatial_change.change.lifting import (
    LiftConfig,
    LiftedObjects,
    lift_candidates,
    lift_change_candidates,
)
from spatial_change.change.matching import MatchConfig, associate

__all__ = [
    "AppearanceConfig",
    "Candidate",
    "CandidateConfig",
    "ChangeCandidates",
    "CrossAppearance",
    "DescriptorConfig",
    "LiftConfig",
    "LiftedObjects",
    "MatchConfig",
    "PointAppearance",
    "associate",
    "compare_appearance",
    "describe_instances",
    "extract_candidates",
    "find_candidates",
    "fuse_appearance",
    "gated_assignment",
    "lift_candidates",
    "lift_change_candidates",
    "linear_assignment",
]
