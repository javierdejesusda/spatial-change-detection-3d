"""Image crops of the changed objects, for the describers and the report.

A change has a box in each capture it exists in. The reconstruction points
inside a box are the object, and the frames of that capture that see them
best give the crops.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain import BoxModel, Change, EpochReconstruction
from spatial_change.geometry.views import (
    UsableFrame,
    ViewConfig,
    crop_view,
    select_views,
    usable_frames,
)
from spatial_change.ports import ChangeCrops as DescriberCrops
from spatial_change.reporting import ChangeCrops as ReportCrops

_DEFAULT_VIEWS = ViewConfig(max_views=3)


@dataclass(frozen=True)
class CropConfig:
    """How the crops of an object are cut.

    Attributes:
        views: Which frames give a crop and how many.
        margin: Share of the object's pixel extent added on every side,
            at least 0 and finite.
        min_margin_px: Smallest margin in pixels, at least 0 and finite.
        min_crop_px: A crop whose shorter side is smaller is left out, at
            least 1.
    """

    views: ViewConfig = _DEFAULT_VIEWS
    margin: float = 0.15
    min_margin_px: float = 4.0
    min_crop_px: int = 8

    def __post_init__(self) -> None:
        """Validates the parameters.

        Raises:
            ValueError: If a margin is negative or not finite, or the
                minimum crop size is below 1.
        """
        if not 0.0 <= self.margin < math.inf:
            raise ValueError("margin must be >= 0 and finite")
        if not 0.0 <= self.min_margin_px < math.inf:
            raise ValueError("min_margin_px must be >= 0 and finite")
        if self.min_crop_px < 1:
            raise ValueError("min_crop_px must be >= 1")


_DEFAULT_CONFIG = CropConfig()


@dataclass(frozen=True, eq=False)
class ChangeViews:
    """The crops of one change in both forms.

    Attributes:
        describer: Every crop of each capture, the one that sees the object
            most first, for the describers.
        report: The first crop of each capture, or ``None`` where there is
            none, for the report page.
    """

    describer: DescriberCrops
    report: ReportCrops


def _crop_side(
    box: BoxModel | None,
    reconstruction: EpochReconstruction,
    frames: Mapping[int, UsableFrame],
    config: CropConfig,
) -> list[NDArray[np.uint8]]:
    """Crops the object of a box out of the frames that see it best.

    Args:
        box: The object's box in this capture, or ``None`` if it is not
            there.
        reconstruction: The capture's reconstruction.
        frames: The usable frames of that capture.
        config: View limits, margin, and minimum crop size.

    Returns:
        The crops, the view with the most visible points first; empty
        without a box, without points in it, or without a view.
    """
    if box is None:
        return []
    points = reconstruction.points[box.to_box().contains(reconstruction.points)]
    if len(points) == 0:
        return []
    views = select_views(points, frames, config.views)
    views.sort(key=lambda view: -len(view.seen.indices))
    crops = (
        crop_view(
            frames[view.position],
            view,
            config.margin,
            config.min_margin_px,
            config.min_crop_px,
        )
        for view in views
    )
    return [crop for crop in crops if crop is not None]


def crops_for_changes(
    changes: Sequence[Change],
    before: EpochReconstruction,
    after: EpochReconstruction,
    config: CropConfig = _DEFAULT_CONFIG,
) -> dict[str, ChangeViews]:
    """Cuts the crops of every change out of the frames of both captures.

    The BEFORE box of a change is cut out of the BEFORE frames and its
    AFTER box out of the AFTER frames.

    Args:
        changes: The changes, with boxes in the world frame of the
            reconstructions.
        before: The BEFORE reconstruction, whose frames carry images and
            depth maps.
        after: The AFTER reconstruction.
        config: View limits, margin, and minimum crop size.

    Returns:
        The crops by change id; a side with no box or no view is empty.
    """
    sides = (
        (before, usable_frames(before, config.views.min_confidence)),
        (after, usable_frames(after, config.views.min_confidence)),
    )
    result = {}
    for change in changes:
        found = [
            _crop_side(box, reconstruction, frames, config)
            for box, (reconstruction, frames) in zip(
                (change.before_box, change.after_box), sides, strict=True
            )
        ]
        result[change.change_id] = ChangeViews(
            describer=DescriberCrops(
                before=tuple(found[0]), after=tuple(found[1])
            ),
            report=ReportCrops(
                before=found[0][0] if found[0] else None,
                after=found[1][0] if found[1] else None,
            ),
        )
    return result
