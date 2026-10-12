"""Dense patch features and promptable object masks.

Importing the package does not load torch or transformers; the DINO encoder
and the SAM segmenter import them when their model loads.
"""

from spatial_change.perception.dino import (
    DINOV2_MODEL_ID,
    DINOV3_MODEL_ID,
    DinoConfig,
    DinoFeatureEncoder,
)
from spatial_change.perception.fake import (
    FakeFeatureEncoder,
    FakeMaskSegmenter,
)
from spatial_change.perception.sam import (
    SAM3_MODEL_ID,
    SAM21_MODEL_ID,
    SamConfig,
    SamSegmenter,
)
from spatial_change.perception.sampling import sample_features

__all__ = [
    "DINOV2_MODEL_ID",
    "DINOV3_MODEL_ID",
    "DinoConfig",
    "DinoFeatureEncoder",
    "FakeFeatureEncoder",
    "FakeMaskSegmenter",
    "SAM21_MODEL_ID",
    "SAM3_MODEL_ID",
    "SamConfig",
    "SamSegmenter",
    "sample_features",
]
