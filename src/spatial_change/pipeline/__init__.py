"""The end-to-end pipeline from two videos to a change report.

Importing the package does not load torch, transformers, open3d, or cv2.
"""

from spatial_change.pipeline.config import RunConfig
from spatial_change.pipeline.models import Models, default_models
from spatial_change.pipeline.progress import (
    STAGES,
    ProgressCallback,
    StageEvent,
    format_event,
)

__all__ = [
    "Models",
    "ProgressCallback",
    "RunConfig",
    "STAGES",
    "StageEvent",
    "default_models",
    "format_event",
]
