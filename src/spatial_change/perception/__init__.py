"""Dense patch features for appearance comparison.

Importing the package does not load torch or transformers; the DINO encoder
imports them when its model loads.
"""

from spatial_change.perception.dino import (
    DINOV2_MODEL_ID,
    DINOV3_MODEL_ID,
    DinoConfig,
    DinoFeatureEncoder,
)
from spatial_change.perception.fake import FakeFeatureEncoder
from spatial_change.perception.sampling import sample_features

__all__ = [
    "DINOV2_MODEL_ID",
    "DINOV3_MODEL_ID",
    "DinoConfig",
    "DinoFeatureEncoder",
    "FakeFeatureEncoder",
    "sample_features",
]
