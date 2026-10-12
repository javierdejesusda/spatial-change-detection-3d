"""Pure geometry on domain types: projection, alignment, floor, boxes, ICP.

Importing this package does not load open3d: ``TargetIndex`` imports it
only when an index is built.
"""

from spatial_change.geometry.alignment import umeyama
from spatial_change.geometry.boxes import fit_upright_box, yaw_rotation
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
    VisiblePoints,
    in_image,
    project,
    unproject_depth,
    usable_pixels,
    visible_points,
)
from spatial_change.geometry.voxels import label_voxel_cells, voxel_cells

__all__ = [
    "FloorConfig",
    "FloorPlane",
    "IcpConfig",
    "IcpResult",
    "TargetIndex",
    "VisiblePoints",
    "camera_up_hint",
    "fit_floor",
    "fit_upright_box",
    "floor_frame",
    "in_image",
    "label_voxel_cells",
    "project",
    "trimmed_icp",
    "umeyama",
    "unproject_depth",
    "usable_pixels",
    "visible_points",
    "voxel_cells",
    "yaw_rotation",
]
