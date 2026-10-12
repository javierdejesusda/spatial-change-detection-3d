"""Rerun recording and GLB export of a change report.

Importing the package does not load rerun, torch, transformers, open3d, or
cv2; rerun loads when a recording is built.
"""

from spatial_change.visualization.config import RerunConfig
from spatial_change.visualization.glb import GlbConfig, export_glb, write_glb
from spatial_change.visualization.palette import (
    CHANGE_COLORS,
    UNOBSERVED_COLOR,
    change_color_hex,
)
from spatial_change.visualization.points import ScenePoints
from spatial_change.visualization.recording import (
    SceneVisualization,
    build_recording,
    save_recording,
)

__all__ = [
    "CHANGE_COLORS",
    "GlbConfig",
    "RerunConfig",
    "ScenePoints",
    "SceneVisualization",
    "UNOBSERVED_COLOR",
    "build_recording",
    "change_color_hex",
    "export_glb",
    "save_recording",
    "write_glb",
]
