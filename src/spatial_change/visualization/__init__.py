"""Rerun visualization of a change report.

Importing the package does not load rerun, torch, transformers, open3d, or
cv2; they load when a recording is built.
"""

from spatial_change.visualization.config import RerunConfig
from spatial_change.visualization.recording import (
    SceneVisualization,
    build_recording,
    save_recording,
)

__all__ = [
    "RerunConfig",
    "SceneVisualization",
    "build_recording",
    "save_recording",
]
