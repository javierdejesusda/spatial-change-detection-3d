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
from spatial_change.change.unobserved import (
    UnobservedConfig,
    extract_unobserved,
    find_unobserved_regions,
)

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
    "UnobservedConfig",
    "associate",
    "compare_appearance",
    "describe_instances",
    "extract_candidates",
    "extract_unobserved",
    "find_candidates",
    "find_unobserved_regions",
    "fuse_appearance",
    "gated_assignment",
    "lift_candidates",
    "lift_change_candidates",
    "linear_assignment",
]
