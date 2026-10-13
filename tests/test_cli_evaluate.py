import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from spatial_change import pipeline
from spatial_change.cli import main
from spatial_change.evaluation import batch, scenediff_fetch
from spatial_change.evaluation.batch import (
    BatchSummary,
    PairEvent,
    PairRecord,
)
from spatial_change.evaluation.metrics import Metrics
from spatial_change.evaluation.scene import PairEvalConfig
from spatial_change.pipeline import RunConfig
from spatial_change.pipeline.config import build_stage_configs

METRICS = Metrics(0.5, 0.25, 0.75, 0.5, 3, 1, 2, 30, 10, 20)


def record(
    pair: str,
    status: str = "ok",
    *,
    seconds: float = 61.4,
    stride: int = 30,
    hits: tuple[str, ...] = ("reconstruction", "appearance"),
) -> PairRecord:
    failed = status == "failed"
    return PairRecord(
        pair=pair,
        status=status,  # type: ignore[arg-type]
        settings="digest",
        stride=stride,
        max_length=64,
        seconds=seconds,
        stage_seconds={},
        cache_hits=() if failed else hits,
        n_gt=2,
        n_changes=0,
        n_uncertain=0,
        uncertain_as="drop",
        n_scored=0,
        changes_by_type={},
        floor_observed=None,
        camera_height_m=None,
        after_cropped=None,
        frames_before=None,
        frames_after=None,
        gt_matches_video=None,
        metrics=None if failed else METRICS,
        error_type="RuntimeError" if failed else None,
        error="CUDA out of memory" if failed else None,
        failed_stage="lifting" if failed else None,
    )


def summary(
    total: int = 2,
    ok: int = 2,
    failed: int = 0,
    pairs: tuple[PairRecord, ...] = (),
) -> BatchSummary:
    return BatchSummary(
        total=total,
        ok=ok,
        failed=failed,
        pending=total - ok - failed,
        uncertain_as="drop",
        failure_rate=failed / total if total else 0.0,
        metrics=METRICS,
        metrics_ok_only=METRICS if ok else None,
        floor_unobserved=(),
        after_cropped=(),
        gt_mismatch=(),
        changes_by_type={},
        pairs=pairs,
    )


class Spy:
    """Stands in for ``run_batch`` and ``summarize``."""

    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []
        self.summaries: list[dict[str, Any]] = []
        self.events: list[PairEvent] = []
        self.error: BaseException | None = None
        self.result = summary()

    def run_batch(
        self,
        pairs: Sequence[str],
        data_dir: Path,
        out_dir: Path,
        config: RunConfig,
        models: object,
        *,
        eval_config: PairEvalConfig,
        cache_dir: Path | None = None,
        on_pair: Callable[[PairEvent], None] | None = None,
    ) -> BatchSummary:
        self.runs.append(
            {
                "pairs": list(pairs),
                "data": data_dir,
                "out": out_dir,
                "config": config,
                "models": models,
                "eval": eval_config,
                "cache_dir": cache_dir,
            }
        )
        if on_pair is not None:
            for event in self.events:
                on_pair(event)
        if self.error is not None:
            raise self.error
        return self.result

    def summarize(
        self,
        pairs: Sequence[str],
        data_dir: Path,
        out_dir: Path,
        eval_config: PairEvalConfig,
    ) -> BatchSummary:
        self.summaries.append(
            {
                "pairs": list(pairs),
                "data": data_dir,
                "out": out_dir,
                "eval": eval_config,
            }
        )
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture(name="spy")
def spy_fixture(monkeypatch: pytest.MonkeyPatch) -> Spy:
    spy = Spy()
    monkeypatch.setattr(batch, "run_batch", spy.run_batch)
    monkeypatch.setattr(batch, "summarize", spy.summarize)
    monkeypatch.setattr(pipeline, "default_models", lambda config: "models")
    return spy


@pytest.fixture(name="data")
def data_fixture(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    for name in ("b_pair", "a_pair"):
        (data / name).mkdir(parents=True)
        (data / name / "segments.pkl").write_bytes(b"")
    (data / "no_annotations").mkdir()
    (data / "notes.txt").write_text("x", encoding="utf-8")
    return data


def test_without_a_list_the_pairs_are_the_folders_with_annotations(
    spy: Spy, data: Path, tmp_path: Path
) -> None:
    code = main(["evaluate", str(data), "--out", str(tmp_path / "out")])

    assert code == 0
    (call,) = spy.runs
    assert call["pairs"] == ["a_pair", "b_pair"]
    assert call["data"] == data
    assert call["out"] == tmp_path / "out"
    assert call["models"] == "models"


def test_the_defaults_describe_with_the_template_and_score_every_30th_frame(
    spy: Spy, data: Path, tmp_path: Path
) -> None:
    main(["evaluate", str(data), "--out", str(tmp_path)])

    (call,) = spy.runs
    assert call["config"] == RunConfig(describer="template")
    assert call["eval"] == PairEvalConfig(stride=30, uncertain_as="drop")


def test_the_stage_cache_defaults_to_a_folder_of_the_results(
    spy: Spy, data: Path, tmp_path: Path
) -> None:
    main(["evaluate", str(data), "--out", str(tmp_path / "out")])

    assert spy.runs[0]["cache_dir"] == tmp_path / "out" / "cache"


def test_the_flags_reach_the_batch(
    spy: Spy, data: Path, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"

    code = main(
        [
            "evaluate",
            str(data),
            "--out",
            str(tmp_path / "out"),
            "--pairs",
            "z_pair",
            "a_pair",
            "--cache-dir",
            str(cache),
            "--device",
            "cpu",
            "--describer",
            "vlm",
            "--stride",
            "15",
            "--uncertain-as",
            "moved",
        ]
    )

    assert code == 0
    (call,) = spy.runs
    assert call["pairs"] == ["z_pair", "a_pair"]
    assert call["cache_dir"] == cache
    assert call["config"] == RunConfig(describer="vlm", device="cpu")
    assert call["eval"] == PairEvalConfig(stride=15, uncertain_as="moved")


def test_a_config_file_is_read_and_its_describer_is_kept(
    spy: Spy, data: Path, tmp_path: Path
) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"seed": 5, "describer": "vlm", "max_keyframes": 20}),
        encoding="utf-8",
    )

    main(["evaluate", str(data), "--out", str(tmp_path), "--config", str(path)])

    assert spy.runs[0]["config"] == RunConfig(
        seed=5, describer="vlm", max_keyframes=20
    )


def test_the_stage_overrides_of_a_config_file_reach_the_batch(
    spy: Spy, data: Path, tmp_path: Path
) -> None:
    stages = {
        "appearance": {"changed_below": 0.3},
        "lift": {"views": {"max_views": 7}},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"stages": stages}), encoding="utf-8")

    main(["evaluate", str(data), "--out", str(tmp_path), "--config", str(path)])

    config = spy.runs[0]["config"]
    assert config == RunConfig(describer="template", stages=stages)
    built = build_stage_configs(config, "cpu")
    assert built.appearance.changed_below == 0.3
    assert built.lift.views.max_views == 7


def test_the_describer_flag_wins_over_the_config_file(
    spy: Spy, data: Path, tmp_path: Path
) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"describer": "vlm"}', encoding="utf-8")

    main(
        [
            "evaluate",
            str(data),
            "--out",
            str(tmp_path),
            "--config",
            str(path),
            "--describer",
            "template",
        ]
    )

    assert spy.runs[0]["config"].describer == "template"


def test_fetch_downloads_the_split_and_runs_the_pairs_it_returns(
    spy: Spy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    fetched: list[tuple[str, Path]] = []

    def fetch(split: str, dest: Path) -> list[Path]:
        fetched.append((split, dest))
        return [dest / "p_1", dest / "p_2"]

    monkeypatch.setattr(scenediff_fetch, "fetch_indoor_split", fetch)

    code = main(
        ["evaluate", str(data), "--out", str(tmp_path), "--fetch", "val"]
    )

    assert code == 0
    assert fetched == [("val", data)]
    assert spy.runs[0]["pairs"] == ["p_1", "p_2"]
    assert spy.runs[0]["data"] == data


def test_a_failed_download_is_one_line_and_exit_1(
    spy: Spy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fetch(split: str, dest: Path) -> list[Path]:
        raise RuntimeError("HTTP 503 for https://example.invalid\nmore")

    monkeypatch.setattr(scenediff_fetch, "fetch_indoor_split", fetch)

    code = main(
        [
            "evaluate",
            str(tmp_path / "d"),
            "--out",
            str(tmp_path),
            "--fetch",
            "test",
        ]
    )

    assert code == 1
    err = capsys.readouterr().err
    assert [line for line in err.splitlines() if line.startswith("error")] == [
        "error: HTTP 503 for https://example.invalid"
    ]
    assert spy.runs == []


def test_pairs_and_fetch_exclude_each_other(
    spy: Spy, data: Path, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "evaluate",
                str(data),
                "--out",
                str(tmp_path),
                "--pairs",
                "a_pair",
                "--fetch",
                "val",
            ]
        )

    assert excinfo.value.code == 2
    assert spy.runs == []


def test_summary_only_never_downloads(
    spy: Spy,
    data: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fetch(split: str, dest: Path) -> list[Path]:
        raise AssertionError("--summary-only must not download")

    monkeypatch.setattr(scenediff_fetch, "fetch_indoor_split", fetch)

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "evaluate",
                str(data),
                "--out",
                str(tmp_path),
                "--fetch",
                "val",
                "--summary-only",
            ]
        )

    assert excinfo.value.code == 2
    assert "--fetch" in capsys.readouterr().err
    assert spy.summaries == []


def test_summary_only_builds_no_models_and_runs_nothing(
    spy: Spy,
    data: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden(config: RunConfig) -> object:
        raise AssertionError("--summary-only must not build models")

    monkeypatch.setattr(pipeline, "default_models", forbidden)

    code = main(
        [
            "evaluate",
            str(data),
            "--out",
            str(tmp_path / "out"),
            "--summary-only",
            "--stride",
            "15",
        ]
    )

    assert code == 0
    assert spy.runs == []
    (call,) = spy.summaries
    assert call["pairs"] == ["a_pair", "b_pair"]
    assert call["out"] == tmp_path / "out"
    assert call["eval"] == PairEvalConfig(stride=15)
    assert str(tmp_path / "out" / "summary.json") in capsys.readouterr().out


def test_summary_only_with_other_settings_is_an_error_and_writes_nothing(
    data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    folder = out / "pairs" / "a_pair"
    folder.mkdir(parents=True)
    (folder / "result.json").write_text(
        record("a_pair", "failed", stride=30).model_dump_json(),
        encoding="utf-8",
    )

    code = main(
        [
            "evaluate",
            str(data),
            "--out",
            str(out),
            "--summary-only",
            "--stride",
            "15",
        ]
    )

    assert code == 1
    err = capsys.readouterr().err
    failure = [line for line in err.splitlines() if line.startswith("error")]
    assert len(failure) == 1
    assert "stride 30" in failure[0] and "stride 15" in failure[0]
    assert "Traceback" not in err
    assert not (out / "summary.json").exists()


def test_every_pair_gets_one_progress_line(
    spy: Spy,
    data: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spy.events = [
        PairEvent(3, 28, record("living_room_3_living_room_4"), False),
        PairEvent(4, 28, record("closet_1_closet_2", hits=()), False),
        PairEvent(5, 28, record("bedroom_28_bedroom_29", "failed"), False),
        PairEvent(6, 28, record("hallway_1_hallway_2", seconds=9.0), True),
    ]

    main(["evaluate", str(data), "--out", str(tmp_path)])

    assert capsys.readouterr().err.splitlines() == [
        "[3/28] living_room_3_living_room_4: ok 61 s "
        "(cache: reconstruction,appearance)",
        "[4/28] closet_1_closet_2: ok 61 s",
        "[5/28] bedroom_28_bedroom_29: failed at lifting "
        "(RuntimeError: CUDA out of memory)",
        "[6/28] hallway_1_hallway_2: ok (resumed)",
    ]


def test_finished_with_failed_pairs_is_still_success_and_says_where_it_wrote(
    spy: Spy,
    data: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spy.result = summary(total=28, ok=25, failed=3)
    spy.events = [
        PairEvent(1, 28, record("a_pair"), True),
        PairEvent(2, 28, record("b_pair"), False),
        PairEvent(3, 28, record("c_pair"), True),
    ]

    code = main(["evaluate", str(data), "--out", str(tmp_path / "out")])

    assert code == 0
    out = capsys.readouterr().out
    assert "28 pairs: 25 ok, 3 failed, 0 pending (2 resumed)" in out
    assert "pixel IoU 0.500" in out
    assert "AP per scene 0.750" in out
    assert f"wrote {tmp_path / 'out' / 'summary.json'}" in out


def test_the_failed_pairs_are_listed_after_the_counts(
    spy: Spy, data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spy.result = summary(
        total=3,
        ok=2,
        failed=1,
        pairs=(
            record("a_pair"),
            record("bedroom_28_bedroom_29", "failed"),
            record("c_pair"),
        ),
    )

    main(["evaluate", str(data), "--out", str(tmp_path / "out")])

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("3 pairs: 2 ok, 1 failed")
    assert (
        "failed: bedroom_28_bedroom_29 at lifting "
        "(RuntimeError: CUDA out of memory)"
    ) in lines


@pytest.mark.parametrize(
    ("where", "extra"),
    [
        ("missing", []),
        ("empty", []),
        ("missing", ["--pairs", "a", "b"]),
        ("missing", ["--summary-only", "--pairs", "a"]),
    ],
)
def test_unusable_input_is_one_line_and_exit_1(
    spy: Spy,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    where: str,
    extra: list[str],
) -> None:
    data = tmp_path / "data"
    if where == "empty":
        data.mkdir()

    code = main(["evaluate", str(data), "--out", str(tmp_path / "out"), *extra])

    assert code == 1
    err = capsys.readouterr().err
    failure = [line for line in err.splitlines() if line.startswith("error")]
    assert len(failure) == 1
    assert str(data) in failure[0]
    assert "Traceback" not in err
    assert spy.runs == [] and spy.summaries == []
    assert not (tmp_path / "out").exists()


def test_a_batch_that_cannot_start_is_one_line_and_exit_1(
    spy: Spy,
    data: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spy.error = ValueError("invalid pair name: 'a b'")

    code = main(["evaluate", str(data), "--out", str(tmp_path)])

    assert code == 1
    err = capsys.readouterr().err
    assert "error: invalid pair name: 'a b'" in err.splitlines()
    assert "Traceback" not in err


def test_debug_lets_the_error_through(
    spy: Spy, data: Path, tmp_path: Path
) -> None:
    spy.error = ValueError("invalid pair name: 'a b'")

    with pytest.raises(ValueError, match="invalid pair name"):
        main(["evaluate", str(data), "--out", str(tmp_path), "--debug"])


def test_an_interrupt_says_how_to_resume(
    spy: Spy,
    data: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spy.error = KeyboardInterrupt()

    code = main(["evaluate", str(data), "--out", str(tmp_path)])

    assert code == 130
    assert "rerun the same command to resume" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments",
    [
        ["evaluate"],
        ["evaluate", "data"],
        ["evaluate", "data", "--out", "o", "--fetch", "all"],
        ["evaluate", "data", "--out", "o", "--uncertain-as", "static"],
        ["evaluate", "data", "--out", "o", "--stride", "x"],
        ["evaluate", "data", "--out", "o", "--describer", "gpt"],
    ],
)
def test_usage_errors_exit_with_2(
    spy: Spy, capsys: pytest.CaptureFixture[str], arguments: list[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(arguments)

    assert excinfo.value.code == 2
    assert "Traceback" not in capsys.readouterr().err
    assert spy.runs == []


@pytest.mark.parametrize(
    ("arguments", "fragment"),
    [
        (["--stride", "0"], "stride must be at least 1"),
        (["--config", "missing.json"], "missing.json"),
    ],
)
def test_bad_settings_are_a_clean_usage_error(
    spy: Spy,
    data: Path,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    fragment: str,
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["evaluate", str(data), "--out", "o", *arguments])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert fragment in err
    assert "Traceback" not in err
    assert spy.runs == []


def test_a_bad_stage_override_is_a_clean_usage_error(
    spy: Spy, data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"stages": {"appearance": {"changed_belw": 0.3}}}', encoding="utf-8"
    )

    with pytest.raises(SystemExit) as excinfo:
        main(["evaluate", str(data), "--out", "o", "--config", str(path)])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "appearance.changed_belw" in err
    assert "Traceback" not in err
    assert spy.runs == []


def test_evaluate_help_exits_with_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["evaluate", "--help"])

    assert excinfo.value.code == 0
    assert "--summary-only" in capsys.readouterr().out
