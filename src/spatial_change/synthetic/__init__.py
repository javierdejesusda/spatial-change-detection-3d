"""Procedural synthetic rooms with exact ground truth for testing."""

from spatial_change.synthetic.scene import (
    ObjectStatus,
    SceneConfig,
    SyntheticEpoch,
    SyntheticObject,
    SyntheticScenePair,
    generate_scene_pair,
)
from spatial_change.synthetic.video import write_video

__all__ = [
    "ObjectStatus",
    "SceneConfig",
    "SyntheticEpoch",
    "SyntheticObject",
    "SyntheticScenePair",
    "generate_scene_pair",
    "write_video",
]
