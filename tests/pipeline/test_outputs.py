from pathlib import Path

import pytest
from appearance_scenes import reconstruction
from report_pipeline import synthetic_report
from visibility_scenes import scene_evidence

from spatial_change.domain import ChangeReport, Epoch
from spatial_change.pipeline.outputs import (
    OUTPUT_FILES,
    prepare_out_dir,
    write_outputs,
)
from spatial_change.visibility import CrossEvidence, VisibilityConfig
from spatial_change.visualization import RerunConfig

SEED = 0


def test_all_four_files_are_written_and_the_report_reads_back(
    tmp_path: Path,
) -> None:
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
        tmp_path,
        inputs.report,
        before,
        after,
        evidence,
        inputs.crops,
        RerunConfig(),
    )

    names = {path.name for path in files.all}
    assert names == set(OUTPUT_FILES)
    for path in files.all:
        assert path.parent == tmp_path
        assert path.stat().st_size > 0
    assert ChangeReport.from_json(files.changes.read_text("utf-8")) == (
        inputs.report
    )
    page = files.report.read_text("utf-8")
    assert page.startswith("<!doctype html>")
    assert files.scene.read_bytes()[:4] == b"glTF"


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
