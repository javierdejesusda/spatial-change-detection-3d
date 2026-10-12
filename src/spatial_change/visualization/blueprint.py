"""The viewer layout saved with the recording."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from spatial_change.visualization.layers import EPOCH_TIMELINE
from spatial_change.visualization.panels import Panel

if TYPE_CHECKING:
    import rerun.blueprint as rrb

_PANEL_COLUMNS = 2
_SCENE_SHARE = 2.0
_PANEL_SHARE = 1.0


def build_blueprint(panels: Sequence[Panel]) -> rrb.Blueprint:
    """Lays out one 3D view of the scene and a grid of image panels.

    The time panel follows the ``epoch`` timeline. Without panels the 3D view
    fills the viewer.

    Args:
        panels: The image panels, in display order.

    Returns:
        The blueprint.
    """
    import rerun.blueprint as rrb

    scene = rrb.Spatial3DView(origin="world", name="Scene")
    body: rrb.Spatial3DView | rrb.Horizontal = scene
    if panels:
        grid = rrb.Grid(
            *(rrb.Spatial2DView(origin=p.origin, name=p.name) for p in panels),
            grid_columns=_PANEL_COLUMNS,
        )
        body = rrb.Horizontal(
            scene, grid, column_shares=[_SCENE_SHARE, _PANEL_SHARE]
        )
    return rrb.Blueprint(body, rrb.TimePanel(timeline=EPOCH_TIMELINE))
