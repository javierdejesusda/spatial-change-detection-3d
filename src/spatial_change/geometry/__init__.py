"""Pure geometry on domain types: projection, alignment, floor, and ICP.

Importing this package does not load open3d: ``TargetIndex`` imports it
only when an index is built.
"""

from spatial_change.geometry.alignment import umeyama
from spatial_change.geometry.floor import (
    FloorConfig,
    FloorPlane,
    camera_up_hint,
    fit_floor,
    floor_frame,
)
from spatial_change.geometry.icp import (
    IcpConfig,
    IcpResult,
    TargetIndex,
    trimmed_icp,
)
from spatial_change.geometry.projection import (
    in_image,
    project,
    unproject_depth,
)

__all__ = [
    "FloorConfig",
    "FloorPlane",
    "IcpConfig",
    "IcpResult",
    "TargetIndex",
    "camera_up_hint",
    "fit_floor",
    "floor_frame",
    "in_image",
    "project",
    "trimmed_icp",
    "umeyama",
    "unproject_depth",
]
