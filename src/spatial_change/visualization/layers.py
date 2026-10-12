"""The 3D layers of the recording: clouds, trajectories, and changes.

Time model: one sequence timeline, ``epoch``, with step 0 for the BEFORE
capture and step 1 for the AFTER capture. Data of an epoch is logged at its
step and cleared at the next one when it must not stay on screen, so each step
shows one coherent scene. The camera trajectories have no step: they are
static.

Entity tree::

    world/<epoch>/points                  scene points of the epoch
    world/<epoch>/unobserved              points no frame of the other
                                          epoch observed (needs evidence)
    world/<epoch>/trajectory              camera path of the epoch (static)
    world/changes/<id>/box                box, colored by change type
    world/changes/<id>/ghost              old box of a moved object
    world/changes/<id>/arrow              old center to new center
    world/unknown/<id>                    region only one epoch observed
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain import (
    BoxModel,
    Change,
    ChangeType,
    Epoch,
    EpochReconstruction,
    OrientedBox,
    UnobservedRegion,
)
from spatial_change.domain.transforms import rotation_to_quaternion
from spatial_change.visibility import PointEvidence, Vote
from spatial_change.visualization.config import RGB, RerunConfig

if TYPE_CHECKING:
    import rerun as rr
    from rerun.components import FillMode

STEP = {Epoch.BEFORE: 0, Epoch.AFTER: 1}
"""Timeline step of each epoch."""

EPOCH_TIMELINE = "epoch"
TRAJECTORY_COLORS: dict[Epoch, RGB] = {
    Epoch.BEFORE: (120, 200, 255),
    Epoch.AFTER: (255, 170, 90),
}
"""Camera path colors: light blue before, orange after."""

_UNOBSERVED_COLOR: RGB = (160, 160, 160)
_OPAQUE = 255


def epoch_path(epoch: Epoch, name: str) -> str:
    """Builds the entity path of an epoch's layer.

    Args:
        epoch: The epoch.
        name: Layer name below the epoch.

    Returns:
        The entity path.
    """
    return f"world/{epoch.value}/{name}"


def change_path(change_id: str, name: str) -> str:
    """Builds the entity path of a layer of one change.

    Args:
        change_id: Identifier of the change.
        name: Layer name below the change.

    Returns:
        The entity path, with the identifier escaped.
    """
    import rerun as rr

    return f"world/changes/{rr.escape_entity_path_part(change_id)}/{name}"


def region_path(region_id: str) -> str:
    """Builds the entity path of an unobserved region.

    Args:
        region_id: Identifier of the region.

    Returns:
        The entity path, with the identifier escaped.
    """
    import rerun as rr

    return f"world/unknown/{rr.escape_entity_path_part(region_id)}"


def at_step(stream: rr.RecordingStream, epoch: Epoch) -> None:
    """Moves the timeline of a stream to the step of an epoch.

    Args:
        stream: The recording stream.
        epoch: The epoch to move to.
    """
    stream.set_time(EPOCH_TIMELINE, sequence=STEP[epoch])


def clear_at(
    stream: rr.RecordingStream,
    path: str,
    epoch: Epoch,
    recursive: bool = False,
) -> None:
    """Clears an entity at the step of an epoch.

    Args:
        stream: The recording stream.
        path: Entity path.
        epoch: The epoch whose step the clear happens at.
        recursive: Whether to clear the whole subtree.
    """
    import rerun as rr

    at_step(stream, epoch)
    stream.log(path, rr.Clear(recursive=recursive))


def rgba(color: RGB, alpha: float = 1.0) -> NDArray[np.uint8]:
    """Appends an alpha byte to a color.

    Args:
        color: RGB bytes.
        alpha: Opacity in ``[0, 1]``.

    Returns:
        The color as an RGBA row of shape ``(4,)``.
    """
    return np.array([*color, round(alpha * _OPAQUE)], np.uint8)


def subsample(count: int, limit: int, seed: int) -> NDArray[np.intp]:
    """Picks which of ``count`` items to draw.

    Args:
        count: Number of items.
        limit: Largest number to draw.
        seed: Seed of the draw.

    Returns:
        All indices when ``count <= limit``, else ``limit`` distinct indices
        in ascending order, the same for the same arguments.
    """
    if count <= limit:
        return np.arange(count)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, limit, replace=False))


def log_points(
    stream: rr.RecordingStream,
    recon: EpochReconstruction,
    config: RerunConfig,
) -> None:
    """Logs the points of an epoch at its step.

    Args:
        stream: The recording stream.
        recon: The epoch's reconstruction.
        config: Point radius, cap, and seed.
    """
    import rerun as rr

    keep = subsample(len(recon.points), config.max_points, config.seed)
    at_step(stream, recon.epoch)
    stream.log(
        epoch_path(recon.epoch, "points"),
        rr.Points3D(
            recon.points[keep],
            colors=recon.colors[keep],
            radii=config.point_radius,
        ),
    )


def log_unobserved_points(
    stream: rr.RecordingStream,
    recon: EpochReconstruction,
    evidence: PointEvidence,
    config: RerunConfig,
) -> None:
    """Logs the points no frame of the other epoch observed, translucent.

    A point counts as never observed when every vote on it is an
    ``UNOBSERVED`` one and it got at least one. Nothing is logged if there
    are none.

    Args:
        stream: The recording stream.
        recon: The epoch's reconstruction.
        evidence: The votes on its points.
        config: Cap, seed, radius, and opacity.
    """
    import rerun as rr

    counts = evidence.counts
    seen = counts[:, : Vote.UNOBSERVED].sum(axis=1)
    never = np.flatnonzero((seen == 0) & (counts[:, Vote.UNOBSERVED] > 0))
    if config.max_unobserved_points == 0 or len(never) == 0:
        return
    keep = never[
        subsample(len(never), config.max_unobserved_points, config.seed)
    ]
    at_step(stream, recon.epoch)
    stream.log(
        epoch_path(recon.epoch, "unobserved"),
        rr.Points3D(
            recon.points[keep],
            colors=rgba(_UNOBSERVED_COLOR, config.unknown_alpha),
            radii=config.point_radius,
        ),
    )


def log_trajectory(
    stream: rr.RecordingStream, recon: EpochReconstruction, config: RerunConfig
) -> None:
    """Logs the camera path of an epoch as a static line strip.

    Nothing is logged for fewer than two frames.

    Args:
        stream: The recording stream.
        recon: The epoch's reconstruction.
        config: Line radius.
    """
    import rerun as rr

    if len(recon.frames) < 2:
        return
    centers = np.stack([f.world_from_camera.translation for f in recon.frames])
    stream.log(
        epoch_path(recon.epoch, "trajectory"),
        rr.LineStrips3D(
            [centers],
            colors=rgba(TRAJECTORY_COLORS[recon.epoch]),
            radii=config.line_radius,
        ),
        static=True,
    )


def _boxes(
    box: BoxModel | OrientedBox,
    color: RGB,
    alpha: float,
    config: RerunConfig,
    labels: list[str] | None = None,
    fill_mode: FillMode | None = None,
) -> rr.Boxes3D:
    import rerun as rr

    if isinstance(box, BoxModel):
        box = box.to_box()
    return rr.Boxes3D(
        centers=[box.center],
        half_sizes=[box.extent / 2.0],
        quaternions=[rotation_to_quaternion(box.rotation)],
        colors=[rgba(color, alpha)],
        radii=config.line_radius,
        labels=labels,
        show_labels=True if labels else None,
        fill_mode=fill_mode,
    )


def log_change(
    stream: rr.RecordingStream, change: Change, config: RerunConfig
) -> None:
    """Logs one change: its boxes, and the ghost and arrow if it moved.

    The before box shows at the BEFORE step and is cleared at the AFTER step
    unless the change has an after box, which replaces it. A moved object
    also gets a translucent ghost of its old box and an arrow from the old
    center to the new one at the AFTER step.

    Args:
        stream: The recording stream.
        change: The change.
        config: Colors, opacity, and radii.
    """
    color = config.change_colors[change.type]
    path = change_path(change.change_id, "box")
    labels = [change.label]
    if change.before_box is not None:
        at_step(stream, Epoch.BEFORE)
        stream.log(path, _boxes(change.before_box, color, 1.0, config, labels))
    if change.after_box is not None:
        at_step(stream, Epoch.AFTER)
        stream.log(path, _boxes(change.after_box, color, 1.0, config, labels))
    elif change.before_box is not None:
        clear_at(stream, path, Epoch.AFTER)
    if change.type is ChangeType.MOVED:
        _log_move(stream, change, color, config)


def _log_move(
    stream: rr.RecordingStream,
    change: Change,
    color: RGB,
    config: RerunConfig,
) -> None:
    import rerun as rr

    if change.before_box is None or change.after_box is None:
        return
    at_step(stream, Epoch.AFTER)
    stream.log(
        change_path(change.change_id, "ghost"),
        _boxes(
            change.before_box,
            color,
            config.ghost_alpha,
            config,
            fill_mode=rr.components.FillMode.Solid,
        ),
    )
    origin = np.array(change.before_box.center)
    stream.log(
        change_path(change.change_id, "arrow"),
        rr.Arrows3D(
            origins=[origin],
            vectors=[np.array(change.after_box.center) - origin],
            colors=[rgba(color)],
            radii=config.line_radius,
        ),
    )


def log_unobserved_region(
    stream: rr.RecordingStream,
    region: UnobservedRegion,
    config: RerunConfig,
) -> None:
    """Logs an unobserved region as a translucent solid box.

    Only one capture saw the region, so its change is unknown in the other
    capture: the box shows at that step and is cleared at the next one if it
    is not the last.

    Args:
        stream: The recording stream.
        region: The region.
        config: Opacity and radii.
    """
    import rerun as rr

    unknown = Epoch.AFTER if region.epoch_seen is Epoch.BEFORE else Epoch.BEFORE
    path = region_path(region.region_id)
    at_step(stream, unknown)
    stream.log(
        path,
        _boxes(
            region.box,
            _UNOBSERVED_COLOR,
            config.unknown_alpha,
            config,
            fill_mode=rr.components.FillMode.Solid,
        ),
    )
    if unknown is Epoch.BEFORE:
        clear_at(stream, path, Epoch.AFTER)
