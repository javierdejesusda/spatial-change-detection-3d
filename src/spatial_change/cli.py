"""Command-line interface for spatial-change."""

import argparse
import gc
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from spatial_change import __version__
from spatial_change.diagnostics.errors import first_line
from spatial_change.evaluation.scenediff import DEFAULT_FRAME_STRIDE

if TYPE_CHECKING:
    from spatial_change import pipeline
    from spatial_change.evaluation import batch
    from spatial_change.evaluation.metrics import Metrics

_EXIT_FAILURE = 1
_EXIT_INTERRUPTED = 130
# Destinations of the flags whose value wins over the config file.
_CONFIG_FLAGS = ("describer", "device", "max_keyframes")
_NO_FFPROBE_NOTE = (
    "note: ffprobe is not on PATH; frame slots use the time between frames, "
    "so videos with dropped frames may be off by one slot"
)


def build_parser() -> argparse.ArgumentParser:
    """Builds the top-level argument parser.

    Returns:
        The configured parser with the ``run``, ``evaluate``, and
        ``fetch-demo`` subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="spatial-change",
        description=(
            "Detect and describe 3D scene changes between two videos "
            "of the same area."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    commands = parser.add_subparsers(dest="command")
    run = commands.add_parser(
        "run",
        help="detect the changes between two videos",
        description=(
            "Run the whole system on two videos and write changes.json, "
            "report.html, scene.rrd, and scene.glb into the output folder."
        ),
    )
    run.add_argument("before", type=Path, help="the video of the first capture")
    run.add_argument("after", type=Path, help="the video of the second capture")
    run.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output folder; it must be empty unless --overwrite is given",
    )
    _add_run_settings(run, "settings of the run")
    run.set_defaults(run_parser=run)
    _add_evaluate(commands)
    _add_fetch_demo(commands)
    return parser


def _add_run_settings(
    parser: argparse.ArgumentParser,
    title: str,
    cache_default: str | None = None,
) -> list[argparse.Action]:
    """Adds the flags that set how the pipeline runs.

    Args:
        parser: The parser of a subcommand that runs the pipeline.
        title: The title of the group that holds the settings in the help.
        cache_default: Where ``--cache-dir`` defaults to, if anywhere.

    Returns:
        The settings, which are all the flags added but ``--debug``.
    """
    group = parser.add_argument_group(title)
    cache_help = (
        "folder that keeps the reconstruction and the appearance "
        "comparison, so a rerun with the same videos skips them"
    )
    if cache_default is not None:
        cache_help += f" (default: {cache_default})"
    settings = [
        group.add_argument(
            "--config",
            type=Path,
            help="JSON file with the settings of the run",
        ),
        group.add_argument(
            "--describer",
            choices=("auto", "vlm", "template"),
            help="how changes are described (default: auto)",
        ),
        group.add_argument(
            "--device",
            help=(
                "torch device of the models, such as cuda or cpu "
                "(default: auto)"
            ),
        ),
        group.add_argument(
            "--max-keyframes",
            type=int,
            help="keyframes kept per video (default: 48)",
        ),
        group.add_argument(
            "--overwrite",
            action="store_true",
            help="write into an output folder that already holds files",
        ),
        group.add_argument("--cache-dir", type=Path, help=cache_help),
    ]
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show the full traceback of a failed run",
    )
    return settings


def _add_evaluate(
    commands: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    """Adds the ``evaluate`` subcommand.

    Args:
        commands: The subparsers of the top-level parser.
    """
    evaluate = commands.add_parser(
        "evaluate",
        help="run and score SceneDiff pairs, resuming after a crash",
        description=(
            "Run the system on SceneDiff pairs, score each against its "
            "annotations, and pool the scores into summary.json. Pairs "
            "finished by an earlier call with the same settings are not run "
            "again. A pair that fails is recorded, the batch goes on, and "
            "the next call tries it again."
        ),
    )
    evaluate.add_argument(
        "data_dir",
        type=Path,
        help="folder with one subfolder per pair (segments.pkl and videos)",
    )
    evaluate.add_argument(
        "--out",
        type=Path,
        required=True,
        help="results folder; run it again with the same value to resume",
    )
    which = evaluate.add_mutually_exclusive_group()
    which.add_argument(
        "--pairs",
        nargs="+",
        metavar="NAME",
        help="pairs to evaluate (default: every subfolder with segments.pkl)",
    )
    which.add_argument(
        "--fetch",
        choices=("val", "test"),
        help="download the indoor pairs of a split into data_dir first",
    )
    evaluate.add_argument(
        "--cache-dir",
        type=Path,
        help=(
            "folder that keeps the reconstruction and appearance stages "
            "(default: cache, inside --out)"
        ),
    )
    evaluate.add_argument(
        "--config",
        type=Path,
        help="JSON file with the settings of the run",
    )
    evaluate.add_argument(
        "--device",
        help="torch device of the models, such as cuda or cpu (default: auto)",
    )
    evaluate.add_argument(
        "--describer",
        choices=("auto", "vlm", "template"),
        help=(
            "how changes are described; descriptions are not scored, so the "
            "default is template unless the config file says otherwise"
        ),
    )
    evaluate.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_FRAME_STRIDE,
        help=(
            "frames of the 30 fps annotation timeline whose index is a "
            "multiple of this are run and scored "
            f"(default: {DEFAULT_FRAME_STRIDE})"
        ),
    )
    evaluate.add_argument(
        "--uncertain-as",
        choices=("drop", "moved"),
        default="drop",
        help=(
            "leave uncertain changes out or score them as drawn: moved, "
            "or added or removed when they have a single box"
        ),
    )
    evaluate.add_argument(
        "--summary-only",
        action="store_true",
        help="rebuild summary.json from the results on disk; run nothing",
    )
    evaluate.add_argument(
        "--debug",
        action="store_true",
        help="show the full traceback of a failed batch",
    )
    evaluate.set_defaults(evaluate_parser=evaluate)


def _add_fetch_demo(
    commands: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    """Adds the ``fetch-demo`` subcommand.

    Args:
        commands: The subparsers of the top-level parser.
    """
    fetch_demo = commands.add_parser(
        "fetch-demo",
        help="download the demo pairs, and optionally run them",
        description=(
            "Download the demo pairs of the SceneDiff benchmark into a data "
            "folder, one subfolder per pair. With --run, also run the whole "
            "system on each pair."
        ),
    )
    fetch_demo.add_argument(
        "data_dir",
        type=Path,
        nargs="?",
        default=Path("data/demo"),
        help="folder that receives one subfolder per pair (default: data/demo)",
    )
    fetch_demo.add_argument(
        "--run",
        type=Path,
        metavar="OUT",
        help=(
            "after the download, run every pair and write its results into "
            "a subfolder of OUT named after the pair; a subfolder that "
            "already holds results needs --overwrite, and the stage cache "
            "in OUT/cache lets a rerun skip the reconstruction"
        ),
    )
    settings = _add_run_settings(
        fetch_demo,
        "settings of the run (need --run)",
        cache_default="cache, inside the --run folder",
    )
    fetch_demo.set_defaults(fetch_demo_parser=fetch_demo, run_settings=settings)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _config_error_text(error: Exception) -> str:
    """Words an invalid configuration on one line.

    Args:
        error: What loading the settings raised.

    Returns:
        The path and message of each invalid field if pydantic found them,
        else the message of the error.
    """
    from pydantic import ValidationError

    if isinstance(error, ValidationError):
        return "; ".join(
            f"{'.'.join(map(str, item['loc']))}: {item['msg']}"
            for item in error.errors()
        )
    return str(error)


def _load_run_config(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> "pipeline.RunConfig":
    """Reads the settings of a run from the config file and the flags.

    Args:
        args: The parsed arguments.
        parser: The parser of the subcommand, which reports invalid settings.

    Returns:
        The settings, where a flag wins over the config file.
    """
    from spatial_change import pipeline

    overrides = {
        dest: getattr(args, dest)
        for dest in _CONFIG_FLAGS
        if getattr(args, dest) is not None
    }
    try:
        return pipeline.RunConfig.load(args.config, **overrides)
    except (OSError, ValueError) as error:
        parser.error(f"invalid configuration: {_config_error_text(error)}")


def _run_videos(
    config: "pipeline.RunConfig",
    models: "pipeline.Models",
    before: Path,
    after: Path,
    out: Path,
    *,
    overwrite: bool,
    cache_dir: Path | None,
) -> "pipeline.PipelineResult":
    """Runs the pipeline on two videos, showing the stages on stderr.

    Args:
        config: The settings of the run.
        models: The factories of the models.
        before: The video of the first capture.
        after: The video of the second capture.
        out: The output folder.
        overwrite: Whether an output folder that holds files is accepted.
        cache_dir: The folder of the stage cache, or ``None`` for no cache.

    Returns:
        What the pipeline produced.
    """
    from spatial_change import pipeline

    def show(event: pipeline.StageEvent) -> None:
        line = pipeline.format_event(event)
        if line is not None:
            print(line, file=sys.stderr, flush=True)

    return pipeline.run_videos(
        before,
        after,
        out,
        config,
        models,
        show,
        overwrite=overwrite,
        cache_dir=cache_dir,
    )


def _changes_line(result: "pipeline.PipelineResult") -> str:
    """Words what a run found as one line.

    Args:
        result: What the pipeline produced.

    Returns:
        A line such as ``2 changes (1 added, 1 moved), 0 unobserved regions``.
    """
    changes = result.report.changes
    kinds = Counter(change.type.value for change in changes)
    tally = ", ".join(f"{n} {kind}" for kind, n in sorted(kinds.items()))
    return (
        _plural(len(changes), "change")
        + (f" ({tally})" if tally else "")
        + ", "
        + _plural(len(result.report.unobserved_regions), "unobserved region")
    )


def _run_command(args: argparse.Namespace) -> int:
    """Runs the pipeline for the ``run`` subcommand.

    Args:
        args: The parsed arguments.

    Returns:
        0 on success, 1 if the run failed.
    """
    from spatial_change import pipeline

    config = _load_run_config(args, args.run_parser)
    try:
        result = _run_videos(
            config,
            pipeline.default_models(config),
            args.before,
            args.after,
            args.out,
            overwrite=args.overwrite,
            cache_dir=args.cache_dir,
        )
    except Exception as error:
        if args.debug:
            raise
        print(f"error: {first_line(error)}", file=sys.stderr)
        return _EXIT_FAILURE

    print(_changes_line(result))
    for path in result.files.all:
        print(f"wrote {path}")
    return 0


def _failure_place(record: "batch.PairRecord") -> str:
    return f"at {record.failed_stage} ({record.error_type}: {record.error})"


def _pair_line(event: "batch.PairEvent") -> str:
    """Words the end of a pair as one plain line.

    Args:
        event: The finished pair.

    Returns:
        A line such as ``[3/28] name: ok 61 s (cache: reconstruction)``.
    """
    record = event.record
    head = f"[{event.position}/{event.total}] {record.pair}:"
    if record.status == "failed":
        line = f"{head} failed {_failure_place(record)}"
    elif event.resumed:
        line = f"{head} ok"
    else:
        line = f"{head} ok {record.seconds:.0f} s"
        if record.cache_hits:
            line += f" (cache: {','.join(record.cache_hits)})"
    return line + (" (resumed)" if event.resumed else "")


def _metrics_line(label: str, metrics: "Metrics | None") -> str | None:
    if metrics is None:
        return None
    return (
        f"{label}: pixel IoU {metrics.pixel_iou:.3f}, "
        f"AP per image {metrics.ap_per_image:.3f}, "
        f"AP per scene {metrics.ap_per_scene:.3f}, "
        f"typed AP per scene {metrics.ap_per_scene_typed:.3f}"
    )


def _evaluate_pairs(args: argparse.Namespace) -> list[str]:
    """Lists the pairs to evaluate, downloading them if asked.

    Args:
        args: The parsed arguments.

    Returns:
        The pair names.

    Raises:
        FileNotFoundError: If there is no data folder to read, even when the
            pairs are named.
        ValueError: If no pair is found.
    """
    data_dir: Path = args.data_dir
    from spatial_change.evaluation import scenediff, scenediff_fetch

    if args.fetch is not None:
        return [
            folder.name
            for folder in scenediff_fetch.fetch_indoor_split(
                args.fetch, data_dir
            )
        ]
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data folder not found: {data_dir}")
    if args.pairs is not None:
        return list(args.pairs)
    names = scenediff.list_pairs(data_dir)
    if not names:
        raise ValueError(
            f"no pairs found in {data_dir} (a pair is a folder with "
            "segments.pkl)"
        )
    return names


def _evaluate_command(args: argparse.Namespace) -> int:
    """Runs the batch for the ``evaluate`` subcommand.

    Args:
        args: The parsed arguments.

    Returns:
        0 when the batch completed, even with failed pairs, 1 if it could
        not start or crashed, 130 if it was interrupted.
    """
    from spatial_change import pipeline
    from spatial_change.evaluation import batch
    from spatial_change.evaluation.scene import PairEvalConfig

    evaluate_parser: argparse.ArgumentParser = args.evaluate_parser
    if args.summary_only and args.fetch is not None:
        evaluate_parser.error(
            "--summary-only reads the results on disk and never downloads; "
            "drop --fetch"
        )
    overrides: dict[str, object] = {}
    if args.device is not None:
        overrides["device"] = args.device
    if args.describer is not None:
        overrides["describer"] = args.describer
    try:
        config = pipeline.RunConfig.load(args.config, **overrides)
        eval_config = PairEvalConfig(
            stride=args.stride,
            uncertain_as=args.uncertain_as,
        )
    except (OSError, ValueError) as error:
        evaluate_parser.error(
            f"invalid configuration: {_config_error_text(error)}"
        )
    if args.describer is None and config.describer == "auto":
        config = config.model_copy(update={"describer": "template"})

    if not args.summary_only:
        from spatial_change.capture import durations

        if durations.ffprobe_path() is None:
            print(_NO_FFPROBE_NOTE, file=sys.stderr, flush=True)

    resumed = 0

    def show(event: "batch.PairEvent") -> None:
        nonlocal resumed
        resumed += event.resumed
        print(_pair_line(event), file=sys.stderr, flush=True)

    cache_dir: Path = (
        args.out / "cache" if args.cache_dir is None else args.cache_dir
    )
    try:
        names = _evaluate_pairs(args)
        if args.summary_only:
            result = batch.summarize(
                names, args.data_dir, args.out, eval_config
            )
        else:
            result = batch.run_batch(
                names,
                args.data_dir,
                args.out,
                config,
                pipeline.default_models(config),
                eval_config=eval_config,
                cache_dir=cache_dir,
                on_pair=show,
            )
    except KeyboardInterrupt:
        print("interrupted: rerun the same command to resume", file=sys.stderr)
        return _EXIT_INTERRUPTED
    except Exception as error:
        if args.debug:
            raise
        print(f"error: {first_line(error)}", file=sys.stderr)
        return _EXIT_FAILURE

    print(
        f"{_plural(result.total, 'pair')}: {result.ok} ok, "
        f"{result.failed} failed, {result.pending} pending"
        + (f" ({resumed} resumed)" if resumed else "")
    )
    for record in result.pairs:
        if record.status == "failed":
            print(f"failed: {record.pair} {_failure_place(record)}")
    for label, metrics in (
        ("all pairs, failed ones scored as missed", result.metrics),
        ("ok pairs only", result.metrics_ok_only),
    ):
        line = _metrics_line(label, metrics)
        if line is not None:
            print(line)
    print(f"wrote {args.out / batch.SUMMARY_FILE}")
    return 0


def _reject_settings_without_run(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """Reports the settings flags given when no run was asked for.

    Args:
        args: The parsed arguments.
        parser: The parser of ``fetch-demo``, which reports the usage error.
    """
    given = [
        action.option_strings[0]
        for action in args.run_settings
        if getattr(args, action.dest) != action.default
    ]
    if given:
        parser.error(f"{', '.join(given)} can only be used with --run")


def _run_demo_pairs(
    args: argparse.Namespace,
    config: "pipeline.RunConfig",
    models: "pipeline.Models",
    folders: list[Path],
) -> int:
    """Runs the pipeline on each downloaded pair, whatever happens to the rest.

    Args:
        args: The parsed arguments.
        config: The settings of the runs.
        models: The factories of the models.
        folders: The folder of each pair, holding its two videos.

    Returns:
        0 when every pair was run, 1 if any failed.
    """
    from spatial_change.evaluation import scenediff

    cache_dir = args.cache_dir or args.run / "cache"
    failed = 0
    for position, folder in enumerate(folders, start=1):
        name = folder.name
        print(
            f"[{position}/{len(folders)}] {name}: running",
            file=sys.stderr,
            flush=True,
        )
        result: pipeline.PipelineResult | None
        try:
            before, after = scenediff.find_pair_videos(folder)
            result = _run_videos(
                config,
                models,
                before,
                after,
                args.run / name,
                overwrite=args.overwrite,
                cache_dir=cache_dir,
            )
        except Exception as error:
            if args.debug:
                raise
            print(f"error: {name}: {first_line(error)}", file=sys.stderr)
            failed += 1
            result = None
        if result is None:
            # Out of the except block the traceback no longer holds the
            # tensors of the stage that failed.
            gc.collect()
        else:
            print(f"{name}: {_changes_line(result)}")
    print(
        f"{_plural(len(folders), 'pair')}: {len(folders) - failed} ok, "
        f"{failed} failed"
    )
    return _EXIT_FAILURE if failed else 0


def _fetch_demo_command(args: argparse.Namespace) -> int:
    """Downloads the demo pairs and, with ``--run``, runs them.

    Args:
        args: The parsed arguments.

    Returns:
        0 when every pair was downloaded and, with ``--run``, run; 1 if the
        models could not be built, the download failed, or any pair failed
        to run.
    """
    from spatial_change import pipeline
    from spatial_change.evaluation import scenediff_fetch

    parser: argparse.ArgumentParser = args.fetch_demo_parser
    config: pipeline.RunConfig | None = None
    models: pipeline.Models | None = None
    if args.run is None:
        _reject_settings_without_run(args, parser)
    else:
        config = _load_run_config(args, parser)
    try:
        if config is not None:
            models = pipeline.default_models(config)
        folders = scenediff_fetch.fetch_pairs(
            scenediff_fetch.DEMO_PAIRS, args.data_dir
        )
    except Exception as error:
        if args.debug:
            raise
        print(f"error: {first_line(error)}", file=sys.stderr)
        return _EXIT_FAILURE
    for folder in folders:
        print(folder)
    if config is None or models is None:
        return 0
    return _run_demo_pairs(args, config, models, folders)


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the command-line interface.

    Args:
        argv: Arguments to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        The process exit code: 0 on success, 1 if a run or a download failed,
        130 if an evaluation was interrupted. Invalid arguments or settings
        exit with 2 through argparse.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "evaluate":
        return _evaluate_command(args)
    if args.command == "fetch-demo":
        return _fetch_demo_command(args)
    return _run_command(args)
