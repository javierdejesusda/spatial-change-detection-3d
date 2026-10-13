"""The steps behind the web app, free of any web framework.

They check the two uploads, give a run its own folder, run the pipeline
through an injected runner, and shape what it wrote for display. The page in
``ui`` only wires them to its components.

Importing this module does not load gradio, torch, transformers, open3d, or
cv2.
"""

import gc
import shutil
import tempfile
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from spatial_change import pipeline
from spatial_change.diagnostics.errors import first_line
from spatial_change.domain import ChangeReport
from spatial_change.pipeline import (
    OutputFiles,
    PipelineResult,
    ProgressCallback,
    RunConfig,
    StageEvent,
)
from spatial_change.semantics import summarize_report

VIDEO_SUFFIXES: tuple[str, ...] = (".mp4", ".mov", ".m4v")
"""The file types an upload may have, in lower case."""

STAGE_LABELS: dict[str, str] = {
    "keyframes": "Reading the videos",
    "reconstruction": "Rebuilding both captures in 3D",
    "alignment": "Aligning the two captures",
    "voting": "Checking what each capture could see",
    "appearance": "Comparing how surfaces look",
    "candidates": "Finding places that differ",
    "lifting": "Lifting those places into 3D",
    "descriptors": "Describing the look of each object",
    "labeling": "Naming the objects",
    "matching": "Pairing objects across the captures",
    "describing": "Writing the descriptions",
    "outputs": "Writing the report and the 3D scene",
}
"""What the user reads while each stage of ``pipeline.STAGES`` runs."""

CHANGE_COLUMNS: tuple[str, ...] = (
    "ID",
    "Type",
    "Object",
    "Confidence",
    "Moved by (m)",
    "Description",
)
"""The headers of the change table."""

PipelineRunner = Callable[[Path, Path, Path, ProgressCallback], PipelineResult]
"""Runs the pipeline on a BEFORE video, an AFTER video, and an output folder."""

ProgressSink = Callable[[float, str], None]
"""Receives how far a run is, from 0 to 1, and what it is doing."""

TableRow = list[str | float | None]


class UploadError(ValueError):
    """An upload cannot be analyzed; the message tells the user what to do."""


class AnalysisError(RuntimeError):
    """A run failed; the message is the first line of the cause."""


@dataclass(frozen=True)
class Analysis:
    """What a finished run shows.

    Attributes:
        run_dir: The folder the run wrote into.
        summary: One paragraph about the changes and the unobserved regions.
        rows: The change table, one row per change, in ``CHANGE_COLUMNS``.
        report_html: The text of the self-contained ``report.html``.
        files: The four files the run wrote.
    """

    run_dir: Path
    summary: str
    rows: list[TableRow]
    report_html: str
    files: OutputFiles


def _check_upload(side: str, upload: str | Path | None) -> Path:
    """Checks one upload.

    Args:
        side: ``before`` or ``after``, as the user knows it.
        upload: The path the browser upload was stored at, if any.

    Returns:
        The path of a video file.

    Raises:
        UploadError: If nothing was uploaded, the file is gone, or its type
            is not one of ``VIDEO_SUFFIXES``.
    """
    if upload is None:
        raise UploadError(f"Choose the {side} video.")
    path = Path(upload)
    if not path.is_file():
        raise UploadError(
            f"The {side} video could not be found: {path.name}. "
            "Upload it again."
        )
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        raise UploadError(
            f"{path.name} is not a supported video. "
            f"Use one of {', '.join(VIDEO_SUFFIXES)}."
        )
    return path


def validate_uploads(
    before: str | Path | None, after: str | Path | None
) -> tuple[Path, Path]:
    """Checks the two uploads of a run.

    Args:
        before: The path of the BEFORE upload, if any.
        after: The path of the AFTER upload, if any.

    Returns:
        The two video paths.

    Raises:
        UploadError: If either upload is missing, gone, or not a video.
    """
    return _check_upload("before", before), _check_upload("after", after)


def default_runs_dir() -> Path:
    """Names the folder that holds the runs when the user names none.

    Returns:
        ``.cache/spatial-change/app`` in the home folder of the user, which
        no other user can create first.
    """
    return Path.home() / ".cache" / "spatial-change" / "app"


def make_runs_root(root: Path) -> None:
    """Creates the folder that holds the runs, for its owner only.

    Args:
        root: The folder; created with its parents if missing. A folder that
            exists keeps its mode.
    """
    root.mkdir(mode=0o700, parents=True, exist_ok=True)


def new_run_dir(root: Path) -> Path:
    """Creates the empty folder one run writes into.

    Args:
        root: The folder that holds the folders of all runs; created with
            its parents if missing.

    Returns:
        A new empty folder under ``root`` that no other call returns.
    """
    make_runs_root(root)
    stamp = datetime.now().strftime("run-%Y%m%d-%H%M%S-")
    return Path(tempfile.mkdtemp(prefix=stamp, dir=root))


def stage_progress(event: StageEvent) -> tuple[float, str]:
    """Words a stage event for a progress bar.

    Args:
        event: The start or the end of a stage.

    Returns:
        How far the run is, from 0 to 1, and a line such as
        ``Step 4 of 12: Checking what each capture could see``. A start
        event fills the bar through the stages before it, an end event
        through the stage itself.
    """
    done = event.index if event.seconds is not None else event.index - 1
    label = STAGE_LABELS[event.name]
    return done / event.total, f"Step {event.index} of {event.total}: {label}"


def change_rows(report: ChangeReport) -> list[TableRow]:
    """Lists the changes of a report as table rows.

    Args:
        report: The report.

    Returns:
        One row per change, in report order, with the columns of
        ``CHANGE_COLUMNS``; the distance is empty unless the object moved.
    """
    return [
        [
            change.change_id,
            change.type.value,
            change.label,
            round(change.confidence, 2),
            None
            if change.displacement_m is None
            else round(change.displacement_m, 2),
            change.description,
        ]
        for change in report.changes
    ]


def analyze(
    before: str | Path | None,
    after: str | Path | None,
    *,
    runner: PipelineRunner,
    runs_root: Path,
    on_progress: ProgressSink | None = None,
) -> Analysis:
    """Runs the pipeline on two uploads and collects what it wrote.

    Args:
        before: The path of the BEFORE upload, if any.
        after: The path of the AFTER upload, if any.
        runner: Runs the pipeline.
        runs_root: The folder that holds the folder of this run.
        on_progress: Receives the progress of every stage start and end.

    Returns:
        The summary, the table, the report page, and the files.

    Raises:
        UploadError: If an upload is missing, gone, or not a video; nothing
            runs and no folder is made.
        AnalysisError: If the run fails; its folder is removed.
    """
    before_path, after_path = validate_uploads(before, after)
    run_dir = new_run_dir(runs_root)

    def forward(event: StageEvent) -> None:
        if on_progress is not None:
            on_progress(*stage_progress(event))

    try:
        result = runner(before_path, after_path, run_dir, forward)
    except Exception as error:
        # The error keeps the frames of the stage that failed, with its
        # models and tensors. Empty them, then collect what was in a cycle.
        traceback.clear_frames(error.__traceback__)
        shutil.rmtree(run_dir, ignore_errors=True)
        gc.collect()
        raise AnalysisError(first_line(error)) from error
    report = result.report
    return Analysis(
        run_dir,
        summarize_report(report),
        change_rows(report),
        result.files.report.read_text(encoding="utf-8"),
        result.files,
    )


def pipeline_runner(
    config: RunConfig, cache_dir: Path | None = None
) -> PipelineRunner:
    """Makes the runner that calls the pipeline, as ``run`` does.

    Args:
        config: The settings of every run.
        cache_dir: Folder that keeps the reconstruction and the appearance
            comparison between runs, or ``None`` to compute both every
            time.

    Returns:
        A runner for every run of the app. The model factories are built
        here, once; no weights load until a run uses them.
    """
    models = pipeline.default_models(config)

    def run(
        before: Path,
        after: Path,
        out_dir: Path,
        progress: ProgressCallback,
    ) -> PipelineResult:
        return pipeline.run_videos(
            before,
            after,
            out_dir,
            config,
            models,
            progress,
            cache_dir=cache_dir,
        )

    return run
