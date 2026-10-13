"""The web page of the app, built with gradio.

The page only wires its components to ``handlers`` and shows their errors.
It needs the ``app`` extra, so nothing outside the ``app`` subcommand
imports it.
"""

import html
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gradio as gr
from gradio_rerun import Rerun

from spatial_change.app import handlers
from spatial_change.app.handlers import (
    AnalysisError,
    PipelineRunner,
    TableRow,
    UploadError,
)

TITLE = "Spatial change"
INTRO = (
    "Upload two videos of the same room, recorded at different times. "
    "The app finds what was added, removed, or moved between them."
)
PRIVACY = "Your videos are processed on this computer and are never uploaded."
NO_RESULTS = "No results yet."
FAILED = "The run did not finish."
HOST = "127.0.0.1"
"""The only address the page listens on, so only this computer reaches it."""
MAX_UPLOAD_GB = 1
MAX_UPLOAD = f"{MAX_UPLOAD_GB}gb"
"""The largest file an upload may be."""
MAX_WAITING = 4
"""How many runs may wait for the one that is running; more are refused."""
DELETE_CACHE = (3600, 86400)
"""How gradio sweeps the copies it makes of uploads and outputs.

It looks every hour (3600 seconds) and deletes those over a day (86400
seconds) old.
"""

EMPTY_REPORT = (
    '<div class="empty"><p>The report appears here after a run.</p>'
    "<p>Add both videos, then select Find changes.</p></div>"
)
EMPTY_SCENE = (
    '<div class="empty"><p>The 3D scene appears here after a run.</p></div>'
)

RunOutputs = tuple[str, str, list[TableRow], str, list[str]]
"""The summary, the report frame, the table rows, the recording path, and the
output file paths of a finished run.
"""

_BEFORE_LABEL = f"Before video (up to {MAX_UPLOAD_GB} GB)"
_AFTER_LABEL = f"After video (up to {MAX_UPLOAD_GB} GB)"
_IDLE_BUTTON = "Find changes"
_BUSY_BUTTON = "Finding changes"
_VIDEO_HEIGHT = 300
_VIEWER_HEIGHT = 640

_PRIMARY = gr.themes.Color(
    c50="#eef5fc",
    c100="#d9e8f8",
    c200="#b3d1f1",
    c300="#84b4e6",
    c400="#4d90d6",
    c500="#2273c3",
    c600="#0b5cad",
    c700="#094b8e",
    c800="#083c72",
    c900="#062e57",
    c950="#041f3c",
    name="survey",
)
_FONTS = ("ui-sans-serif", "system-ui", "Segoe UI", "Roboto", "sans-serif")
THEME = gr.themes.Base(
    primary_hue=_PRIMARY,
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_md,
    font=_FONTS,
    font_mono=_FONTS,
).set(
    body_text_color_subdued="*neutral_600",
    body_text_color_subdued_dark="*neutral_300",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_background_fill_dark="*primary_600",
    button_primary_background_fill_hover_dark="*primary_500",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
)
"""System fonts only, so the page fetches nothing, and the report's blue."""

CSS = """
.gradio-container {
  width: 100%;
  max-width: 1200px !important;
  margin-inline: auto !important;
  -webkit-font-smoothing: antialiased;
}
.app-header h1 {
  margin: 0 0 0.375rem;
  font-size: 1.625rem;
  font-weight: 650;
  letter-spacing: -0.01em;
  text-wrap: balance;
}
.app-header p {
  max-width: 64ch;
  margin: 0 0 0.25rem;
  color: var(--body-text-color-subdued);
  text-wrap: pretty;
}
#run-button {
  min-height: 2.75rem;
}
.gradio-container table {
  font-variant-numeric: tabular-nums;
}
#summary {
  min-height: 3.5rem;
}
#summary p {
  font-size: 1.125rem;
  line-height: 1.5;
  font-variant-numeric: tabular-nums;
  text-wrap: pretty;
}
.report-frame {
  display: block;
  width: 100%;
  height: 78vh;
  min-height: 520px;
  border: 0;
  border-radius: var(--radius-lg);
  box-shadow: 0 0 0 1px var(--border-color-primary);
}
.empty {
  display: grid;
  place-content: center;
  min-height: 16rem;
  padding: 1rem;
  text-align: center;
  color: var(--body-text-color-subdued);
}
.empty p {
  margin: 0.125rem 0;
}
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
"""


def report_frame(page: str) -> str:
    """Wraps a report page in a frame that keeps its script to itself.

    Args:
        page: The text of ``report.html``.

    Returns:
        An iframe element holding the page. Its sandbox lets the page run
        its own script but gives it no access to this one.
    """
    return (
        '<iframe class="report-frame" title="Change report" '
        f'sandbox="allow-scripts" srcdoc="{html.escape(page, quote=True)}">'
        "</iframe>"
    )


def find_changes_callback(
    runner: PipelineRunner, runs_root: Path
) -> Callable[..., RunOutputs]:
    """Makes the function that the Find changes button calls.

    Args:
        runner: Runs the pipeline.
        runs_root: The folder that holds the folder of each run.

    Returns:
        A function from the two uploads and the failure record of the
        session to the summary, the report frame, the table rows, the
        recording path, and the output file paths. It raises ``gr.Error``
        when an upload is not usable or the run fails, and leaves the words
        of the problem in the record for ``_failed``, which gradio calls
        next.
    """

    def find_changes(
        before: str | None,
        after: str | None,
        failure: dict[str, str],
        progress: gr.Progress = gr.Progress(),  # noqa: B008
    ) -> RunOutputs:
        try:
            analysis = handlers.analyze(
                before,
                after,
                runner=runner,
                runs_root=runs_root,
                on_progress=lambda fraction, text: progress(
                    fraction, desc=text
                ),
            )
        except UploadError as error:
            failure["cause"] = str(error)
            raise gr.Error(
                str(error),
                duration=None,
                title="Check the videos",
                print_exception=False,
            ) from error
        except AnalysisError as error:
            failure["cause"] = str(error)
            raise gr.Error(
                str(error), duration=None, title="The run failed"
            ) from error
        files = analysis.files
        return (
            analysis.summary,
            report_frame(analysis.report_html),
            analysis.rows,
            str(files.recording),
            [str(path) for path in files.all],
        )

    return find_changes


def _start() -> tuple[Any, ...]:
    """Clears the results and locks the button while a run goes on."""
    return (
        gr.update(value=_BUSY_BUTTON, interactive=False),
        "",
        EMPTY_REPORT,
        [],
        gr.update(value=None, visible=False),
        gr.update(visible=True),
        None,
    )


def _succeeded() -> tuple[Any, ...]:
    """Shows the viewer in place of its empty state and frees the button."""
    return gr.update(visible=True), gr.update(visible=False), _idle_button()


def _failed(failure: dict[str, str]) -> tuple[str, Any]:
    """Says in the page that the run ended without results, and why.

    Args:
        failure: The session's record of the problem; the words are taken
            out of it.

    Returns:
        The status text and the button, free to start another run.
    """
    text = html.escape(failure.pop("cause", ""), quote=False)
    return f"**{FAILED}** {text}".rstrip(), _idle_button()


def _idle_button() -> Any:
    return gr.update(value=_IDLE_BUTTON, interactive=True)


def build_app(runner: PipelineRunner, runs_root: Path) -> gr.Blocks:
    """Builds the page.

    Args:
        runner: Runs the pipeline.
        runs_root: The folder that holds the folder of each run.

    Returns:
        The page, with its queue set to run one job at a time and to hold
        at most ``MAX_WAITING`` more.
    """
    demo = gr.Blocks(
        title=TITLE, analytics_enabled=False, delete_cache=DELETE_CACHE
    )
    with demo:
        with gr.Column(elem_classes="app-header"):
            gr.HTML(f"<h1>{TITLE}</h1><p>{INTRO}</p><p>{PRIVACY}</p>")
        with gr.Row():
            before = gr.Video(
                label=_BEFORE_LABEL,
                sources=["upload"],
                height=_VIDEO_HEIGHT,
            )
            after = gr.Video(
                label=_AFTER_LABEL,
                sources=["upload"],
                height=_VIDEO_HEIGHT,
            )
        with gr.Row():
            button = gr.Button(
                _IDLE_BUTTON,
                variant="primary",
                size="lg",
                scale=0,
                min_width=220,
                elem_id="run-button",
            )
        summary = gr.Markdown(NO_RESULTS, elem_id="summary")
        with gr.Tabs():
            with gr.Tab("Report"):
                report = gr.HTML(EMPTY_REPORT)
            with gr.Tab("Changes"):
                table = gr.Dataframe(
                    headers=list(handlers.CHANGE_COLUMNS),
                    datatype=["str", "str", "str", "number", "number", "str"],
                    value=[],
                    interactive=False,
                    wrap=True,
                    show_label=False,
                )
            with gr.Tab("3D scene"):
                scene_empty = gr.HTML(EMPTY_SCENE)
                viewer = Rerun(
                    height=_VIEWER_HEIGHT, visible=False, show_label=False
                )
            with gr.Tab("Downloads"):
                files = gr.File(
                    file_count="multiple",
                    interactive=False,
                    show_label=False,
                )

        failure = gr.State({})
        outputs = [summary, report, table, viewer, files]
        start = button.click(
            _start,
            None,
            [button, summary, report, table, viewer, scene_empty, files],
            queue=False,
            api_visibility="private",
        )
        run = start.then(
            find_changes_callback(runner, runs_root),
            [before, after, failure],
            outputs,
            concurrency_limit=1,
            concurrency_id="pipeline",
            show_progress_on=summary,
            api_visibility="private",
        )
        run.success(
            _succeeded,
            None,
            [viewer, scene_empty, button],
            queue=False,
            api_visibility="private",
        )
        run.failure(
            _failed,
            failure,
            [summary, button],
            queue=False,
            api_visibility="private",
        )
    demo.queue(default_concurrency_limit=1, max_size=MAX_WAITING)
    return demo


def launch_app(
    *,
    runner: PipelineRunner,
    runs_root: Path,
    port: int | None = None,
    prevent_thread_lock: bool = False,
) -> gr.Blocks:
    """Serves the page.

    It listens on ``HOST`` only, is never shared through a public link, and
    sends no usage data.

    Args:
        runner: Runs the pipeline.
        runs_root: The folder that holds the folder of each run.
        port: The port to listen on, or ``None`` for 7860 or the next free
            one.
        prevent_thread_lock: Whether to return at once with the server
            running in a thread, instead of waiting until it stops.

    Returns:
        The page, which is closed when the server stops.

    Raises:
        OSError: If the runs folder cannot be made or no port is free.
    """
    handlers.make_runs_root(runs_root)
    demo = build_app(runner, runs_root)
    demo.launch(
        server_name=HOST,
        server_port=port,
        share=False,
        prevent_thread_lock=prevent_thread_lock,
        max_file_size=MAX_UPLOAD,
        allowed_paths=[str(runs_root)],
        footer_links=["settings", "gradio"],
        theme=THEME,
        css=CSS,
    )
    return demo
