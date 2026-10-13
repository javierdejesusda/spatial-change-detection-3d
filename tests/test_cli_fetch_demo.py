import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from eval_pairs import write_pair
from run_spy import RunSpy, install_run_spy

from spatial_change import pipeline
from spatial_change.cli import main
from spatial_change.evaluation import scenediff_fetch
from spatial_change.pipeline import RunConfig

PAIRS = scenediff_fetch.DEMO_PAIRS
VIDEOS = ("video1.mp4", "video2.mov")


class FetchSpy:
    """Stands in for ``fetch_pairs`` and writes the folders it would fetch."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], Path]] = []
        self.error: Exception | None = None
        self.without_videos: set[str] = set()

    def __call__(self, pairs: Sequence[str], dest_root: Path) -> list[Path]:
        self.calls.append((tuple(pairs), dest_root))
        if self.error is not None:
            raise self.error
        return [
            write_pair(
                dest_root,
                pair,
                {},
                videos=() if pair in self.without_videos else VIDEOS,
            )
            for pair in pairs
        ]


@pytest.fixture(name="fetch")
def fetch_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FetchSpy:
    monkeypatch.chdir(tmp_path)
    spy = FetchSpy()
    monkeypatch.setattr(scenediff_fetch, "fetch_pairs", spy)
    return spy


@pytest.fixture(name="spy")
def spy_fixture(monkeypatch: pytest.MonkeyPatch) -> RunSpy:
    return install_run_spy(monkeypatch)


def test_fetch_demo_downloads_the_demo_pairs_into_data_demo(
    fetch: FetchSpy, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["fetch-demo"])

    assert code == 0
    assert fetch.calls == [(PAIRS, Path("data/demo"))]
    assert capsys.readouterr().out.splitlines() == [
        str(Path("data/demo") / pair) for pair in PAIRS
    ]


def test_the_data_folder_is_the_first_argument(
    fetch: FetchSpy, tmp_path: Path
) -> None:
    data = tmp_path / "elsewhere"

    code = main(["fetch-demo", str(data)])

    assert code == 0
    assert fetch.calls == [(PAIRS, data)]


def test_without_run_no_pair_is_run(fetch: FetchSpy, spy: RunSpy) -> None:
    assert main(["fetch-demo"]) == 0
    assert spy.calls == []


def test_a_failed_download_is_one_line_and_exit_1(
    fetch: FetchSpy, capsys: pytest.CaptureFixture[str]
) -> None:
    fetch.error = RuntimeError("HTTP 503 for https://example.invalid\nmore")

    code = main(["fetch-demo"])

    assert code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.splitlines() == [
        "error: HTTP 503 for https://example.invalid"
    ]


def test_run_runs_every_pair_into_a_folder_named_after_it(
    fetch: FetchSpy, spy: RunSpy, tmp_path: Path
) -> None:
    data = tmp_path / "data"
    out = tmp_path / "out"

    code = main(["fetch-demo", str(data), "--run", str(out)])

    assert code == 0
    assert [(c["before"], c["after"], c["out"]) for c in spy.calls] == [
        (data / pair / "video1.mp4", data / pair / "video2.mov", out / pair)
        for pair in PAIRS
    ]
    assert all(call["models"] == "models" for call in spy.calls)
    assert all(call["config"] == RunConfig() for call in spy.calls)
    assert all(call["cache_dir"] == out / "cache" for call in spy.calls)
    assert all(call["overwrite"] is False for call in spy.calls)


def test_run_names_each_pair_and_tallies_them(
    fetch: FetchSpy,
    spy: RunSpy,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    main(["fetch-demo", str(tmp_path / "data"), "--run", str(tmp_path / "o")])

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert f"{PAIRS[0]}: 0 changes, 0 unobserved regions" in lines
    assert lines[-1] == "5 pairs: 5 ok, 0 failed"
    assert f"[2/5] {PAIRS[1]}: running" in captured.err.splitlines()
    assert "[4/12] voting: 4.2 s" in captured.err


def test_the_settings_flags_reach_every_pair(
    fetch: FetchSpy, spy: RunSpy, tmp_path: Path
) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"sample_fps": 2.0, "max_keyframes": 20, "seed": 5}),
        encoding="utf-8",
    )
    cache = tmp_path / "cache"

    code = main(
        [
            "fetch-demo",
            str(tmp_path / "data"),
            "--run",
            str(tmp_path / "out"),
            "--config",
            str(config),
            "--describer",
            "template",
            "--device",
            "cpu",
            "--max-keyframes",
            "12",
            "--overwrite",
            "--cache-dir",
            str(cache),
        ]
    )

    assert code == 0
    assert len(spy.calls) == len(PAIRS)
    for call in spy.calls:
        assert call["config"] == RunConfig(
            sample_fps=2.0,
            seed=5,
            describer="template",
            device="cpu",
            max_keyframes=12,
        )
        assert call["overwrite"] is True
        assert call["cache_dir"] == cache


def test_the_models_are_built_once_for_all_the_pairs(
    fetch: FetchSpy,
    spy: RunSpy,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models = object()
    built: list[RunConfig] = []

    def build(config: RunConfig) -> object:
        built.append(config)
        return models

    monkeypatch.setattr(pipeline, "default_models", build)

    code = main(
        ["fetch-demo", str(tmp_path / "d"), "--run", str(tmp_path / "o")]
    )

    assert code == 0
    assert built == [RunConfig()]
    assert all(call["models"] is models for call in spy.calls)


def test_models_that_cannot_be_built_stop_the_command_before_the_download(
    fetch: FetchSpy,
    spy: RunSpy,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    def build(config: RunConfig) -> object:
        raise RuntimeError("no CUDA device\nmore")

    monkeypatch.setattr(pipeline, "default_models", build)

    code = main(
        ["fetch-demo", str(tmp_path / "d"), "--run", str(tmp_path / "o")]
    )

    assert code == 1
    assert capsys.readouterr().err.splitlines() == ["error: no CUDA device"]
    assert fetch.calls == []
    assert spy.calls == []


def test_a_bad_config_is_a_usage_error_before_anything_is_downloaded(
    fetch: FetchSpy,
    spy: RunSpy,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"max_keyframes": 1}', encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        main(["fetch-demo", "--run", str(tmp_path), "--config", str(config)])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "max_keyframes" in err
    assert "Traceback" not in err
    assert fetch.calls == []
    assert spy.calls == []


@pytest.mark.parametrize(
    "arguments", [["fetch-demo", "--run"], ["fetch-demo", "a", "b"]]
)
def test_usage_errors_exit_with_2(
    fetch: FetchSpy,
    spy: RunSpy,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(arguments)

    assert excinfo.value.code == 2
    assert "Traceback" not in capsys.readouterr().err
    assert fetch.calls == []
    assert spy.calls == []


def test_a_failing_pair_is_reported_and_the_others_still_run(
    fetch: FetchSpy,
    spy: RunSpy,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    spy.errors[PAIRS[1]] = ValueError("no sharp frame in video: v\nmore")

    code = main(
        ["fetch-demo", str(tmp_path / "d"), "--run", str(tmp_path / "o")]
    )

    assert code == 1
    assert [call["out"].name for call in spy.calls] == list(PAIRS)
    captured = capsys.readouterr()
    assert (
        f"error: {PAIRS[1]}: no sharp frame in video: v"
        in captured.err.splitlines()
    )
    assert "Traceback" not in captured.err
    lines = captured.out.splitlines()
    assert not any(line.startswith(f"{PAIRS[1]}:") for line in lines)
    assert f"{PAIRS[2]}: 0 changes, 0 unobserved regions" in lines
    assert lines[-1] == "5 pairs: 4 ok, 1 failed"


def test_a_pair_without_its_videos_is_reported_and_the_others_still_run(
    fetch: FetchSpy,
    spy: RunSpy,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    fetch.without_videos = {PAIRS[0]}

    code = main(
        ["fetch-demo", str(tmp_path / "d"), "--run", str(tmp_path / "o")]
    )

    assert code == 1
    assert [call["out"].name for call in spy.calls] == list(PAIRS[1:])
    captured = capsys.readouterr()
    (failure,) = [
        line for line in captured.err.splitlines() if line.startswith("error")
    ]
    assert failure.startswith(f"error: {PAIRS[0]}: no video1.mp4 or video1.mov")
    assert captured.out.splitlines()[-1] == "5 pairs: 4 ok, 1 failed"


def test_debug_lets_a_failing_pair_through_with_its_traceback(
    fetch: FetchSpy, spy: RunSpy, tmp_path: Path
) -> None:
    spy.errors[PAIRS[0]] = RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        main(
            [
                "fetch-demo",
                str(tmp_path / "d"),
                "--run",
                str(tmp_path / "o"),
                "--debug",
            ]
        )


def test_debug_lets_a_failed_download_through_with_its_traceback(
    fetch: FetchSpy,
) -> None:
    fetch.error = RuntimeError("HTTP 503")

    with pytest.raises(RuntimeError, match="HTTP 503"):
        main(["fetch-demo", "--debug"])


@pytest.mark.parametrize(
    ("arguments", "flags"),
    [
        (["--config", "c.json"], ["--config"]),
        (["--describer", "template"], ["--describer"]),
        (["--device", "cpu"], ["--device"]),
        (["--max-keyframes", "12"], ["--max-keyframes"]),
        (["--overwrite"], ["--overwrite"]),
        (["--cache-dir", "c"], ["--cache-dir"]),
        (
            ["--device", "cpu", "--overwrite", "--cache-dir", "c"],
            ["--device", "--overwrite", "--cache-dir"],
        ),
    ],
)
def test_settings_flags_without_run_are_a_usage_error(
    fetch: FetchSpy,
    spy: RunSpy,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    flags: list[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["fetch-demo", *arguments])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert f"{', '.join(flags)} can only be used with --run" in err
    assert "Traceback" not in err
    assert fetch.calls == []
    assert spy.calls == []


def test_the_help_lists_the_flags_and_what_a_rerun_needs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["fetch-demo", "--help"])

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--run", "--config", "--cache-dir", "--overwrite", "--debug"):
        assert flag in out
    assert "data/demo" in out
    assert "settings of the run (need --run)" in out
    unwrapped = "".join(out.split())
    assert "alreadyholdsresultsneeds--overwrite" in unwrapped
    assert "(default:cache,insidethe--runfolder)" in unwrapped
