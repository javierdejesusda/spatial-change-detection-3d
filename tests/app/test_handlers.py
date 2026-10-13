import stat
import sys
import weakref
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock

import pytest
from app_runner import REPORT_PAGE, FakeRunner
from report_scenes import empty_report, one_of_each_report
from run_spy import install_run_spy

from spatial_change import pipeline
from spatial_change.app import handlers
from spatial_change.pipeline import RunConfig, StageEvent
from spatial_change.pipeline.outputs import OUTPUT_FILES


def test_two_existing_videos_are_accepted_as_paths(
    videos: tuple[Path, Path],
) -> None:
    before, after = videos

    assert handlers.validate_uploads(str(before), after) == (before, after)


@pytest.mark.parametrize("missing", ["before", "after"])
def test_a_missing_upload_names_the_side_to_fill(
    videos: tuple[Path, Path], missing: str
) -> None:
    before, after = videos
    args = {"before": before, "after": after, missing: None}

    with pytest.raises(handlers.UploadError, match=f"{missing} video"):
        handlers.validate_uploads(args["before"], args["after"])


def test_an_upload_that_is_not_on_disk_is_named(
    videos: tuple[Path, Path], tmp_path: Path
) -> None:
    before, _ = videos

    with pytest.raises(handlers.UploadError, match="gone.mp4"):
        handlers.validate_uploads(before, tmp_path / "gone.mp4")


def test_a_file_that_is_not_a_video_lists_the_accepted_types(
    videos: tuple[Path, Path], tmp_path: Path
) -> None:
    before, _ = videos
    notes = tmp_path / "notes.txt"
    notes.write_text("x", encoding="utf-8")

    with pytest.raises(handlers.UploadError, match=r"notes\.txt.*\.mp4"):
        handlers.validate_uploads(before, notes)


@pytest.mark.parametrize("suffix", [".avi", ".mkv", ".webm"])
def test_a_container_the_decoders_do_not_read_is_refused(
    tmp_path: Path, suffix: str
) -> None:
    clip = tmp_path / f"clip{suffix}"
    clip.write_bytes(b"v")

    with pytest.raises(handlers.UploadError) as excinfo:
        handlers.validate_uploads(clip, clip)

    assert str(excinfo.value) == (
        f"clip{suffix} is not a supported video. Use one of .mp4, .mov, .m4v."
    )


def test_a_video_suffix_is_accepted_in_any_case(tmp_path: Path) -> None:
    upper = tmp_path / "CLIP.MOV"
    upper.write_bytes(b"v")

    assert handlers.validate_uploads(upper, upper) == (upper, upper)


def test_a_run_folder_is_new_and_empty_under_the_root(tmp_path: Path) -> None:
    root = tmp_path / "runs"

    first = handlers.new_run_dir(root)
    second = handlers.new_run_dir(root)

    assert first != second
    for folder in (first, second):
        assert folder.parent == root
        assert folder.is_dir()
        assert list(folder.iterdir()) == []


def test_the_default_runs_folder_belongs_to_the_user(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    assert handlers.default_runs_dir() == (
        tmp_path / ".cache" / "spatial-change" / "app"
    )


def test_the_runs_folder_is_made_for_its_owner_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    modes: dict[Path, int] = {}
    real_mkdir = Path.mkdir

    def spy(
        self: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        modes[self] = mode
        real_mkdir(self, mode, parents, exist_ok)

    monkeypatch.setattr(Path, "mkdir", spy)
    root = tmp_path / "runs"

    handlers.new_run_dir(root)

    assert modes[root] == 0o700


@pytest.mark.skipif(sys.platform == "win32", reason="no POSIX modes")
def test_other_users_cannot_enter_a_new_runs_folder(tmp_path: Path) -> None:
    root = tmp_path / "runs"

    handlers.new_run_dir(root)

    assert stat.S_IMODE(root.stat().st_mode) == 0o700


def test_every_stage_has_a_plain_label() -> None:
    assert set(handlers.STAGE_LABELS) == set(pipeline.STAGES)
    for label in handlers.STAGE_LABELS.values():
        assert label
        assert label[0].isupper()
        assert "\u2014" not in label


def test_a_stage_start_fills_the_bar_up_to_the_stage_before() -> None:
    fraction, text = handlers.stage_progress(StageEvent("voting", 4, 12, None))

    assert fraction == pytest.approx(3 / 12)
    assert text == f"Step 4 of 12: {handlers.STAGE_LABELS['voting']}"


def test_a_stage_end_fills_the_bar_through_the_stage() -> None:
    fraction, text = handlers.stage_progress(StageEvent("voting", 4, 12, 4.2))

    assert fraction == pytest.approx(4 / 12)
    assert text == f"Step 4 of 12: {handlers.STAGE_LABELS['voting']}"


def test_the_change_table_has_one_row_per_change() -> None:
    rows = handlers.change_rows(one_of_each_report("A chair moved."))

    assert rows[0] == ["c1", "moved", "chair", 0.87, 1.25, "A chair moved."]
    assert [row[:3] for row in rows[1:]] == [
        ["c2", "added", "lamp"],
        ["c3", "removed", "crate"],
        ["c4", "uncertain", "rug"],
    ]
    assert all(row[4] is None for row in rows[1:])


def test_a_report_without_changes_has_no_rows() -> None:
    assert handlers.change_rows(empty_report()) == []


def test_analyze_runs_the_pipeline_on_a_fresh_folder_and_collects_outputs(
    videos: tuple[Path, Path], tmp_path: Path
) -> None:
    runner = FakeRunner()
    root = tmp_path / "runs"

    analysis = handlers.analyze(*videos, runner=runner, runs_root=root)

    ((before, after, out_dir),) = runner.calls
    assert (before, after) == videos
    assert out_dir.parent == root
    assert analysis.run_dir == out_dir
    assert analysis.files.all == tuple(out_dir / n for n in OUTPUT_FILES)
    assert all(path.is_file() for path in analysis.files.all)
    assert analysis.report_html == REPORT_PAGE
    assert analysis.rows == handlers.change_rows(runner.report)
    assert analysis.summary.startswith("4 changes:")


def test_analyze_says_when_nothing_changed(
    videos: tuple[Path, Path], tmp_path: Path
) -> None:
    runner = FakeRunner(empty_report())

    analysis = handlers.analyze(*videos, runner=runner, runs_root=tmp_path)

    assert analysis.summary == "No changes were found."
    assert analysis.rows == []


def test_analyze_forwards_the_stage_progress(
    videos: tuple[Path, Path], tmp_path: Path
) -> None:
    seen: list[tuple[float, str]] = []

    handlers.analyze(
        *videos,
        runner=FakeRunner(),
        runs_root=tmp_path,
        on_progress=lambda fraction, text: seen.append((fraction, text)),
    )

    label = handlers.STAGE_LABELS["keyframes"]
    assert seen == [
        (0.0, f"Step 1 of 12: {label}"),
        (pytest.approx(1 / 12), f"Step 1 of 12: {label}"),
    ]


def test_analyze_rejects_bad_uploads_before_it_runs_anything(
    tmp_path: Path,
) -> None:
    runner = FakeRunner()

    with pytest.raises(handlers.UploadError):
        handlers.analyze(None, None, runner=runner, runs_root=tmp_path)

    assert runner.calls == []
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("video not found: a.mp4"),
        ValueError("no sharp frame in video: b.mp4\nsecond line"),
        RuntimeError("CUDA out of memory"),
    ],
)
def test_a_failed_run_becomes_one_line_for_the_user(
    videos: tuple[Path, Path], tmp_path: Path, error: Exception
) -> None:
    runner = FakeRunner(error=error)

    with pytest.raises(handlers.AnalysisError) as excinfo:
        handlers.analyze(*videos, runner=runner, runs_root=tmp_path)

    assert str(excinfo.value) == str(error).splitlines()[0]
    assert excinfo.value.__cause__ is error


def test_a_failed_run_leaves_no_folder_behind(
    videos: tuple[Path, Path], tmp_path: Path
) -> None:
    runner = FakeRunner(error=RuntimeError("CUDA out of memory"))
    root = tmp_path / "runs"

    with pytest.raises(handlers.AnalysisError):
        handlers.analyze(*videos, runner=runner, runs_root=root)

    ((_, _, out_dir),) = runner.calls
    assert not out_dir.exists()
    assert list(root.iterdir()) == []


def test_a_finished_run_keeps_its_folder_and_files(
    videos: tuple[Path, Path], tmp_path: Path
) -> None:
    analysis = handlers.analyze(
        *videos, runner=FakeRunner(), runs_root=tmp_path / "runs"
    )

    assert analysis.run_dir.is_dir()
    assert all(path.is_file() for path in analysis.files.all)


class Hoard:
    """Points at itself, so only the garbage collector can free it."""

    def __init__(self) -> None:
        self.me = self


def test_a_failed_run_does_not_keep_what_its_stage_held(
    videos: tuple[Path, Path], tmp_path: Path
) -> None:
    ghosts: list[weakref.ref[Hoard]] = []

    def runner(
        before: Path,
        after: Path,
        out_dir: Path,
        progress: Callable[[StageEvent], None],
    ) -> None:
        hoard = Hoard()
        ghosts.append(weakref.ref(hoard))
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(handlers.AnalysisError) as excinfo:
        handlers.analyze(*videos, runner=runner, runs_root=tmp_path)

    assert excinfo.value.__cause__ is not None
    (ghost,) = ghosts
    assert ghost() is None


def test_an_interrupt_is_not_turned_into_an_analysis_error(
    videos: tuple[Path, Path], tmp_path: Path
) -> None:
    runner = FakeRunner(error=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        handlers.analyze(*videos, runner=runner, runs_root=tmp_path)


def test_the_pipeline_runner_calls_the_pipeline_entry_point(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spy = install_run_spy(monkeypatch)
    config = RunConfig(describer="template")
    cache = tmp_path / "cache"
    shown: list[StageEvent] = []

    runner = handlers.pipeline_runner(config, cache)
    result = runner(Path("a.mp4"), Path("b.mp4"), tmp_path, shown.append)

    assert result.files.all == tuple(tmp_path / n for n in OUTPUT_FILES)
    assert spy.calls == [
        {
            "before": Path("a.mp4"),
            "after": Path("b.mp4"),
            "out": tmp_path,
            "config": config,
            "models": "models",
            "overwrite": False,
            "cache_dir": cache,
        }
    ]
    assert shown == [
        StageEvent("voting", 4, 12, None),
        StageEvent("voting", 4, 12, 4.23),
    ]


def test_the_pipeline_runner_builds_the_models_once_for_all_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spy = install_run_spy(monkeypatch)
    default_models = Mock(return_value="models")
    monkeypatch.setattr(pipeline, "default_models", default_models)
    config = RunConfig(describer="template")
    shown: list[StageEvent] = []

    runner = handlers.pipeline_runner(config)
    runner(Path("a.mp4"), Path("b.mp4"), tmp_path, shown.append)
    runner(Path("c.mp4"), Path("d.mp4"), tmp_path, shown.append)

    default_models.assert_called_once_with(config)
    assert [call["models"] for call in spy.calls] == ["models", "models"]
