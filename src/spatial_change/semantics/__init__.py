"""Spatial semantics: labels, landmarks, facts, and sentences about changes.

Importing the package does not load torch or transformers; the labeler that
the functions take does.
"""

from spatial_change.semantics.labeling import LabelConfig, label_instances
from spatial_change.semantics.landmarks import (
    LandmarkConfig,
    extract_landmarks,
)
from spatial_change.semantics.relations import (
    ChangeFact,
    Direction,
    Landmark,
    LandmarkRelation,
    RelationConfig,
    build_change_fact,
    describe_direction,
    distance_to_box,
    format_distance,
    format_rotation,
    nearest_landmarks,
    supporting_surface,
)
from spatial_change.semantics.summary import (
    summarize_changes,
    summarize_report,
)
from spatial_change.semantics.template import template_sentence
from spatial_change.semantics.validator import (
    ValidationResult,
    repair_description,
    validate_description,
)
from spatial_change.semantics.vocabulary import (
    FURNITURE,
    INDOOR_OBJECTS,
    STRUCTURE,
)

__all__ = [
    "ChangeFact",
    "Direction",
    "FURNITURE",
    "INDOOR_OBJECTS",
    "LabelConfig",
    "Landmark",
    "LandmarkConfig",
    "LandmarkRelation",
    "RelationConfig",
    "STRUCTURE",
    "ValidationResult",
    "build_change_fact",
    "describe_direction",
    "distance_to_box",
    "extract_landmarks",
    "format_distance",
    "format_rotation",
    "label_instances",
    "nearest_landmarks",
    "repair_description",
    "summarize_changes",
    "summarize_report",
    "supporting_surface",
    "template_sentence",
    "validate_description",
]
