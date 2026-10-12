"""Dense patch features, promptable object masks, and open-vocabulary labels.

Importing the package does not load torch or transformers; the DINO encoder,
the SAM segmenter, and the SigLIP labeler import them when their model loads.
"""

from spatial_change.perception.dino import (
    DINOV2_MODEL_ID,
    DINOV3_MODEL_ID,
    DinoConfig,
    DinoFeatureEncoder,
)
from spatial_change.perception.fake import (
    FakeFeatureEncoder,
    FakeLabeler,
    FakeMaskSegmenter,
    FakeVlm,
)
from spatial_change.perception.sam import (
    SAM3_MODEL_ID,
    SAM21_MODEL_ID,
    SamConfig,
    SamSegmenter,
)
from spatial_change.perception.sampling import sample_features
from spatial_change.perception.siglip import (
    SIGLIP2_MODEL_ID,
    SIGLIP_MODEL_ID,
    SiglipConfig,
    SiglipLabeler,
)

__all__ = [
    "DINOV2_MODEL_ID",
    "DINOV3_MODEL_ID",
    "DinoConfig",
    "DinoFeatureEncoder",
    "FakeFeatureEncoder",
    "FakeLabeler",
    "FakeMaskSegmenter",
    "FakeVlm",
    "SAM21_MODEL_ID",
    "SAM3_MODEL_ID",
    "SIGLIP2_MODEL_ID",
    "SIGLIP_MODEL_ID",
    "SamConfig",
    "SamSegmenter",
    "SiglipConfig",
    "SiglipLabeler",
    "sample_features",
]
