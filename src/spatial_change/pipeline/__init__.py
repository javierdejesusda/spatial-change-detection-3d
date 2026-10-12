"""The end-to-end pipeline from two videos to a change report.

Importing the package does not load torch, transformers, open3d, or cv2.
"""

from spatial_change.pipeline.config import RunConfig
from spatial_change.pipeline.models import Models, default_models
from spatial_change.pipeline.outputs import OutputFiles
from spatial_change.pipeline.progress import (
    STAGES,
    ProgressCallback,
    StageEvent,
    format_event,
)
from spatial_change.pipeline.run import PipelineResult, run_frames, run_videos

__all__ = [
    "Models",
    "OutputFiles",
    "PipelineResult",
    "ProgressCallback",
    "RunConfig",
    "STAGES",
    "StageEvent",
    "default_models",
    "format_event",
    "run_frames",
    "run_videos",
]
