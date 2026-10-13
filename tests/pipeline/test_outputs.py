from pathlib import Path
from typing import Any

import pytest
from appearance_scenes import reconstruction
from report_pipeline import synthetic_report
from visibility_scenes import scene_evidence

from spatial_change.domain import ChangeReport, Epoch
from spatial_change.pipeline import outputs as outputs_module
from spatial_change.pipeline.outputs import (
    OUTPUT_FILES,
    OutputFiles,
    prepare_out_dir,
    write_outputs,
)
from spatial_change.visibility import CrossEvidence, VisibilityConfig
from spatial_change.visualization import RerunConfig

SEED = 0


def write_synthetic(out_dir: Path) -> tuple[OutputFiles, ChangeReport]:
    """Writes the outputs of a synthetic scene and gives the report."""
    votes = scene_evidence(SEED, VisibilityConfig())
    pair = votes.pair
    inputs = synthetic_report(SEED).inputs
    before = reconstruction(
        Epoch.BEFORE, pair.before.frames, votes.clouds[Epoch.BEFORE].points
    )
    after = reconstruction(
        Epoch.AFTER, pair.after.frames, votes.clouds[Epoch.AFTER].points
    )
    evidence = CrossEvidence(
        votes.evidence[Epoch.BEFORE], votes.evidence[Epoch.AFTER]
    )

    files = write_outputs(
        out_dir,
        inputs.report,
        before,
        after,
        evidence,
        inputs.crops,
        RerunConfig(),
    )
    return files, inputs.report


def test_all_four_files_are_written_and_the_report_reads_back(
    tmp_path: Path,
) -> None:
    files, report = write_synthetic(tmp_path)

    names = {path.name for path in files.all}
    assert names == set(OUTPUT_FILES)
    for path in files.all:
        assert path.parent == tmp_path
        assert path.stat().st_size > 0
    assert ChangeReport.from_json(files.changes.read_text("utf-8")) == report
    page = files.report.read_text("utf-8")
    assert page.startswith("<!doctype html>")
    assert files.scene.read_bytes()[:4] == b"glTF"


def fail_after_a_partial_write(*args: Any) -> None:
    """Stands in for a writer: starts its file, then fails."""
    next(arg for arg in args if isinstance(arg, Path)).write_bytes(b"part")
    raise RuntimeError("disk full")


@pytest.mark.parametrize("writer", ["save_recording", "write_glb"])
def test_a_failed_write_leaves_none_of_the_four_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: str
) -> None:
    (tmp_path / "notes.txt").write_text("keep", encoding="utf-8")

    monkeypatch.setattr(outputs_module, writer, fail_after_a_partial_write)

    with pytest.raises(RuntimeError, match="disk full"):
        write_synthetic(tmp_path)

    assert [p.name for p in tmp_path.iterdir()] == ["notes.txt"]


def test_a_file_that_cannot_be_removed_keeps_the_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unlink = Path.unlink

    def locked_unlink(path: Path, missing_ok: bool = False) -> None:
        if path.name == "report.html":
            raise PermissionError("the file is open in another program")
        unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(outputs_module, "write_glb", fail_after_a_partial_write)
    monkeypatch.setattr(Path, "unlink", locked_unlink)

    with pytest.raises(RuntimeError, match="disk full"):
        write_synthetic(tmp_path)

    assert [p.name for p in tmp_path.iterdir()] == ["report.html"]


def test_a_missing_out_dir_is_created(tmp_path: Path) -> None:
    target = tmp_path / "runs" / "one"

    prepare_out_dir(target, overwrite=False)

    assert target.is_dir()


def test_an_empty_out_dir_is_accepted(tmp_path: Path) -> None:
    prepare_out_dir(tmp_path, overwrite=False)


def test_a_non_empty_out_dir_is_an_error_unless_overwriting(
    tmp_path: Path,
) -> None:
    (tmp_path / "old.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        prepare_out_dir(tmp_path, overwrite=False)
    prepare_out_dir(tmp_path, overwrite=True)

    assert (tmp_path / "old.txt").read_text(encoding="utf-8") == "keep"


def test_an_out_dir_that_is_a_file_is_an_error(tmp_path: Path) -> None:
    target = tmp_path / "file"
    target.write_text("x", encoding="utf-8")

    with pytest.raises(NotADirectoryError, match="file"):
        prepare_out_dir(target, overwrite=True)


def test_overwriting_removes_the_old_output_files_only(tmp_path: Path) -> None:
    for name in (*OUTPUT_FILES, "notes.txt"):
        (tmp_path / name).write_text("old", encoding="utf-8")

    prepare_out_dir(tmp_path, overwrite=True)

    assert [p.name for p in tmp_path.iterdir()] == ["notes.txt"]
