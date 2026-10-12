"""The files a run writes and the folder they go in.

Importing this module does not load rerun, torch, open3d, or cv2.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from spatial_change.domain import ChangeReport, Epoch, EpochReconstruction
from spatial_change.reporting import ChangeCrops, ReportInputs, write_report
from spatial_change.semantics import summarize_report
from spatial_change.visibility import CrossEvidence
from spatial_change.visualization import (
    RerunConfig,
    ScenePoints,
    SceneVisualization,
    save_recording,
    write_glb,
)

OUTPUT_FILES: tuple[str, ...] = (
    "changes.json",
    "report.html",
    "scene.rrd",
    "scene.glb",
)
"""Names of the files a run writes into its output folder."""


@dataclass(frozen=True)
class OutputFiles:
    """The files of a finished run.

    Attributes:
        changes: The ``changes.json`` report.
        report: The self-contained ``report.html`` page.
        recording: The ``scene.rrd`` Rerun recording.
        scene: The ``scene.glb`` annotated point cloud.
    """

    changes: Path
    report: Path
    recording: Path
    scene: Path

    @property
    def all(self) -> tuple[Path, ...]:
        """The four files, in the order of ``OUTPUT_FILES``."""
        return (self.changes, self.report, self.recording, self.scene)


def prepare_out_dir(out_dir: Path, overwrite: bool) -> None:
    """Makes sure a run may write into a folder.

    Args:
        out_dir: The output folder; created with its parents if missing.
        overwrite: Whether a folder that already holds files is accepted.
            The four output files of an earlier run are deleted at once,
            so a failed run never leaves results that look like its own;
            other files are left alone.

    Raises:
        FileExistsError: If the folder holds files and ``overwrite`` is
            false.
        NotADirectoryError: If ``out_dir`` is a file.
    """
    if out_dir.exists() and not out_dir.is_dir():
        raise NotADirectoryError(f"not a folder: {out_dir}")
    if not overwrite and out_dir.is_dir() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} is not empty; pass --overwrite to write into it"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILES:
        (out_dir / name).unlink(missing_ok=True)


def write_outputs(
    out_dir: Path,
    report: ChangeReport,
    before: EpochReconstruction,
    after: EpochReconstruction,
    evidence: CrossEvidence,
    crops: Mapping[str, ChangeCrops],
    rerun: RerunConfig,
) -> OutputFiles:
    """Writes the report, the page, the recording, and the scene.

    Args:
        out_dir: The output folder, which must exist.
        report: The detected changes and unobserved regions.
        before: The BEFORE reconstruction in the room frame.
        after: The AFTER reconstruction in the same frame.
        evidence: The visibility evidence of both clouds.
        crops: The report-page crop of each side of each change, by id.
        rerun: Settings of the recording.

    Returns:
        The paths written.
    """
    files = OutputFiles(*(out_dir / name for name in OUTPUT_FILES))
    clouds = {
        Epoch.BEFORE: ScenePoints(before.points, before.colors),
        Epoch.AFTER: ScenePoints(after.points, after.colors),
    }
    files.changes.write_text(report.to_json(), encoding="utf-8")
    write_report(
        ReportInputs(report, summarize_report(report), crops, clouds),
        files.report,
    )
    save_recording(
        SceneVisualization(before, after, report, evidence),
        files.recording,
        rerun,
    )
    write_glb(clouds, report, files.scene)
    return files
