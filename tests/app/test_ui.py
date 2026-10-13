import html
import inspect
import re
import socket
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from app_runner import FakeRunner

gr = pytest.importorskip("gradio")
rerun_component = pytest.importorskip("gradio_rerun")

from spatial_change.app import handlers, ui  # noqa: E402


def _blocks(demo: Any, kind: type) -> list[Any]:
    return [block for block in demo.blocks.values() if isinstance(block, kind)]


def _inputs(function: Any, videos: tuple[Path, Path]) -> list[Any]:
    uploads = iter(map(str, videos))
    return [
        {} if isinstance(block, gr.State) else next(uploads)
        for block in function.inputs
    ]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture(name="demo", scope="module")
def demo_fixture(tmp_path_factory: pytest.TempPathFactory) -> gr.Blocks:
    return ui.build_app(FakeRunner(), tmp_path_factory.mktemp("runs"))


def test_the_page_has_two_videos_one_button_and_four_result_views(
    demo: gr.Blocks,
) -> None:
    videos = _blocks(demo, gr.Video)
    assert [video.label for video in videos] == [
        "Before video (up to 1 GB)",
        "After video (up to 1 GB)",
    ]
    (button,) = _blocks(demo, gr.Button)
    assert button.value == "Find changes"
    (table,) = _blocks(demo, gr.Dataframe)
    assert tuple(table.headers) == handlers.CHANGE_COLUMNS
    assert len(_blocks(demo, rerun_component.Rerun)) == 1
    assert len(_blocks(demo, gr.File)) == 1
    tabs = [tab.label for tab in _blocks(demo, gr.Tab)]
    assert tabs == ["Report", "Changes", "3D scene", "Downloads"]


def test_one_pipeline_job_runs_at_a_time(demo: gr.Blocks) -> None:
    jobs = [
        function
        for function in demo.fns.values()
        if function.concurrency_id == "pipeline"
    ]

    assert len(jobs) == 1
    assert jobs[0].concurrency_limit == 1


def test_at_most_four_runs_wait_for_the_one_that_runs(demo: gr.Blocks) -> None:
    assert demo._queue.max_size == 4


def test_old_copies_of_uploads_and_outputs_are_swept(demo: gr.Blocks) -> None:
    assert demo.delete_cache == (3600, 86400)


def test_every_event_returns_one_value_per_output(
    videos: tuple[Path, Path], tmp_path: Path
) -> None:
    demo = ui.build_app(FakeRunner(), tmp_path)

    assert len(demo.fns) == 4
    for function in demo.fns.values():
        returned = function.fn(*_inputs(function, videos))

        assert len(returned) == len(function.outputs), function.name


def test_a_run_locks_the_button_and_clears_the_views() -> None:
    button, summary, report, rows, _, _, files = ui._start()

    assert button["interactive"] is False
    assert summary == ""
    assert report == ui.EMPTY_REPORT
    assert rows == []
    assert files is None


def test_a_failed_run_is_an_error_that_ends_with_its_cause_and_a_retry(
    videos: tuple[Path, Path], tmp_path: Path
) -> None:
    runner = FakeRunner(error=RuntimeError("CUDA out of memory\nsecond line"))
    demo = ui.build_app(runner, tmp_path)
    (run,) = (f for f in demo.fns.values() if f.concurrency_id == "pipeline")
    (failed,) = (f for f in demo.fns.values() if f.trigger_only_on_failure)
    inputs = _inputs(run, videos)

    with pytest.raises(gr.Error) as excinfo:
        run.fn(*inputs)
    summary, button = failed.fn(inputs[-1])

    assert excinfo.value.message == "CUDA out of memory"
    assert "CUDA out of memory" in summary
    assert "second line" not in summary
    assert button["interactive"] is True
    assert button["value"] == "Find changes"


def test_a_bad_upload_is_an_error_that_ends_with_the_words_to_fix(
    tmp_path: Path,
) -> None:
    runner = FakeRunner()
    failure: dict[str, str] = {}
    find_changes = ui.find_changes_callback(runner, tmp_path)

    with pytest.raises(gr.Error) as excinfo:
        find_changes(None, None, failure)

    summary, _ = ui._failed(failure)
    assert excinfo.value.message == "Choose the before video."
    assert "Choose the before video." in summary
    assert runner.calls == []


def test_the_cause_in_the_status_cannot_add_markup() -> None:
    summary, _ = ui._failed({"cause": "expected <class 'str'>"})

    assert "&lt;class 'str'&gt;" in summary
    assert "<class" not in summary


def test_a_failure_without_a_known_cause_still_ends_the_run() -> None:
    summary, button = ui._failed({})

    assert summary.strip("*") == ui.FAILED
    assert button["interactive"] is True


def test_the_page_cannot_be_told_to_listen_elsewhere() -> None:
    assert "host" not in inspect.signature(ui.launch_app).parameters


def test_a_finished_run_fills_every_view(
    videos: tuple[Path, Path], tmp_path: Path
) -> None:
    runner = FakeRunner()
    find_changes = ui.find_changes_callback(runner, tmp_path / "runs")

    summary, frame, rows, recording, files = find_changes(*map(str, videos), {})

    assert summary.startswith("4 changes:")
    assert rows == handlers.change_rows(runner.report)
    assert recording.endswith("scene.rrd")
    assert [Path(path).name for path in files] == [
        "changes.json",
        "report.html",
        "scene.rrd",
        "scene.glb",
    ]
    assert all(Path(path).is_file() for path in files)
    assert 'sandbox="allow-scripts"' in frame
    assert 'title="Change report"' in frame


def test_the_report_cannot_close_its_frame() -> None:
    page = '"><script>alert(1)</script><b a="'

    frame = ui.report_frame(page)

    (srcdoc,) = re.findall(r'srcdoc="([^"]*)"', frame)
    assert html.unescape(srcdoc) == page
    assert "<script>" not in frame


def test_the_empty_report_frame_invites_a_run() -> None:
    assert "Find changes" in ui.EMPTY_REPORT


def test_the_app_listens_on_this_computer_and_answers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "127.0.0.1")
    port = _free_port()

    demo = ui.launch_app(
        runner=FakeRunner(),
        runs_root=tmp_path,
        port=port,
        prevent_thread_lock=True,
    )
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with direct.open(f"http://127.0.0.1:{port}/") as response:
            status = response.status
            page = response.read().decode("utf-8")
    finally:
        demo.close()

    assert status == 200
    assert "Spatial change" in page
    assert demo.server_name == "127.0.0.1"
    assert demo.share is False
    assert demo.max_file_size == 1024**3
    assert demo.allowed_paths == [str(tmp_path)]
