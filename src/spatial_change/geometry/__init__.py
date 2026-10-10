"""Pure geometry on domain types: projection and point set alignment."""

from spatial_change.geometry.alignment import umeyama
from spatial_change.geometry.projection import (
    in_image,
    project,
    unproject_depth,
)

__all__ = ["in_image", "project", "umeyama", "unproject_depth"]
