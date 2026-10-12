"""Builds the Rerun recording of a change report.

``rerun`` is imported inside the functions, so importing this module loads
neither it nor any model library.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from spatial_change.domain import (
    ChangeReport,
    Epoch,
    EpochReconstruction,
)
from spatial_change.visibility import CrossEvidence
from spatial_change.visualization import layers
from spatial_change.visualization.blueprint import build_blueprint
from spatial_change.visualization.config import RerunConfig
from spatial_change.visualization.panels import log_panels

if TYPE_CHECKING:
    import rerun as rr

_DEFAULT_CONFIG = RerunConfig()


@dataclass(frozen=True, eq=False)
class SceneVisualization:
    """Everything the recording shows.

    Attributes:
        before: The BEFORE reconstruction.
        after: The AFTER reconstruction, in the same world frame.
        report: The detected changes and unobserved regions.
        evidence: Optional visibility evidence of both clouds; it adds the
            never-observed points.
    """

    before: EpochReconstruction
    after: EpochReconstruction
    report: ChangeReport
    evidence: CrossEvidence | None = None

    def __post_init__(self) -> None:
        """Checks the epochs and the evidence.

        Raises:
            ValueError: If ``before`` is not the BEFORE reconstruction,
                ``after`` is not the AFTER one, or the evidence does not
                have one row per point of its cloud.
        """
        if self.before.epoch != Epoch.BEFORE:
            raise ValueError("before must be the BEFORE reconstruction")
        if self.after.epoch != Epoch.AFTER:
            raise ValueError("after must be the AFTER reconstruction")
        if self.evidence is not None and (
            len(self.evidence.before.counts) != len(self.before.points)
            or len(self.evidence.after.counts) != len(self.after.points)
        ):
            raise ValueError("evidence must have one row per point")


def build_recording(
    scene: SceneVisualization, config: RerunConfig = _DEFAULT_CONFIG
) -> rr.RecordingStream:
    """Logs a scene to a new recording stream.

    The data is buffered in memory until a sink is attached to the stream.

    Args:
        scene: The scene to show.
        config: Appearance and size limits.

    Returns:
        The stream holding the recording.
    """
    import rerun as rr

    stream = rr.RecordingStream(config.application_id)
    _record(stream, scene, config)
    return stream


def _record(
    stream: rr.RecordingStream, scene: SceneVisualization, config: RerunConfig
) -> None:
    import rerun as rr

    stream.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    cross = scene.evidence
    for recon, votes in (
        (scene.before, None if cross is None else cross.before),
        (scene.after, None if cross is None else cross.after),
    ):
        layers.log_points(stream, recon, config)
        if votes is not None:
            layers.log_unobserved_points(stream, recon, votes, config)
        layers.log_trajectory(stream, recon, config)
    layers.clear_at(stream, "world/before", Epoch.AFTER, recursive=True)
    for change in scene.report.changes:
        layers.log_change(stream, change, config)
    for region in scene.report.unobserved_regions:
        layers.log_unobserved_region(stream, region, config)
    panels = [
        panel
        for recon in (scene.before, scene.after)
        for panel in log_panels(stream, recon, scene.report.changes, config)
    ]
    stream.send_blueprint(build_blueprint(panels))


def save_recording(
    scene: SceneVisualization,
    path: Path,
    config: RerunConfig = _DEFAULT_CONFIG,
) -> None:
    """Writes a scene to a ``.rrd`` file.

    The file sink is attached before anything is logged, so the data streams
    to disk, and the stream is disconnected even if writing fails.

    Args:
        scene: The scene to show.
        path: Output file; missing parent directories are created.
        config: Appearance and size limits.
    """
    import rerun as rr

    stream = rr.RecordingStream(config.application_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        stream.save(path)
        _record(stream, scene, config)
        stream.flush()
    finally:
        stream.disconnect()
