"""Parameters of the Rerun recording."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from spatial_change.domain import ChangeType
from spatial_change.visualization.palette import CHANGE_COLORS, RGB


@dataclass(frozen=True)
class RerunConfig:
    """Parameters of the Rerun recording.

    Lengths are in the units of the world frame. The two alphas are
    fractions in the open interval ``(0, 1)``, so a ghost or an unknown
    region is always see-through yet never invisible.

    Attributes:
        application_id: Application id stored in the recording.
        point_radius: Radius of the scene points, positive and finite.
        line_radius: Radius of box edges, arrows, and trajectories,
            positive and finite.
        max_points: Points drawn per epoch; a larger cloud is subsampled
            with the seed, at least 1.
        max_unobserved_points: Never-observed points drawn per epoch, at
            least 0; 0 draws none.
        seed: Seed of the subsampling, at least 0.
        change_colors: Box color of each change type, as three int bytes;
            stored as a read-only copy.
        ghost_alpha: Opacity of the ghost of a moved object's old box.
        unknown_alpha: Opacity of the unobserved regions and points.
        image_downscale: Integer factor by which the image panels shrink,
            at least 1.
        max_image_panels: Image panels per epoch, at least 0; 0 draws none.
        panel_points: Points of one changed object drawn on a panel, at
            least 1.
    """

    application_id: str = "spatial-change"
    point_radius: float = 0.01
    line_radius: float = 0.005
    max_points: int = 200_000
    max_unobserved_points: int = 20_000
    seed: int = 0
    change_colors: Mapping[ChangeType, RGB] = field(
        default=CHANGE_COLORS, hash=False
    )
    ghost_alpha: float = 0.25
    unknown_alpha: float = 0.2
    image_downscale: int = 2
    max_image_panels: int = 8
    panel_points: int = 2_000

    def __post_init__(self) -> None:
        """Validates the parameters.

        Raises:
            ValueError: If the application id is empty, a radius is not
                positive and finite, a count or the seed is out of range,
                an alpha is outside ``(0, 1)``, or ``change_colors`` lacks
                a change type or holds something other than three int
                bytes.
        """
        if not self.application_id:
            raise ValueError("application_id must be non-empty")
        for name in ("point_radius", "line_radius"):
            if not 0.0 < getattr(self, name) < math.inf:
                raise ValueError(f"{name} must be positive and finite")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be an int >= 0")
        if self.max_points < 1:
            raise ValueError("max_points must be >= 1")
        if self.max_unobserved_points < 0:
            raise ValueError("max_unobserved_points must be >= 0")
        for name in ("ghost_alpha", "unknown_alpha"):
            if not 0.0 < getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be within (0, 1)")
        if self.image_downscale < 1:
            raise ValueError("image_downscale must be >= 1")
        if self.max_image_panels < 0:
            raise ValueError("max_image_panels must be >= 0")
        if self.panel_points < 1:
            raise ValueError("panel_points must be >= 1")
        self._check_colors()

    def _check_colors(self) -> None:
        if set(self.change_colors) != set(ChangeType):
            raise ValueError("change_colors must have every change type")
        frozen = {}
        for kind, color in self.change_colors.items():
            if not (
                isinstance(color, tuple | list)
                and len(color) == 3
                and all(type(c) is int and 0 <= c <= 255 for c in color)
            ):
                raise ValueError(
                    f"change_colors must hold three int bytes, got {color!r}"
                )
            frozen[kind] = (color[0], color[1], color[2])
        object.__setattr__(self, "change_colors", MappingProxyType(frozen))
