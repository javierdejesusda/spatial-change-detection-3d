"""Command-line interface for spatial-change."""

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from spatial_change import __version__

_EXIT_FAILURE = 1


def build_parser() -> argparse.ArgumentParser:
    """Builds the top-level argument parser.

    Returns:
        The configured parser with the ``run`` subcommand.
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
    run.add_argument(
        "--config",
        type=Path,
        help="JSON file with the settings of the run",
    )
    run.add_argument(
        "--describer",
        choices=("auto", "vlm", "template"),
        help="how changes are described (default: auto)",
    )
    run.add_argument(
        "--device",
        help="torch device of the models, such as cuda or cpu (default: auto)",
    )
    run.add_argument(
        "--max-keyframes",
        type=int,
        help="keyframes kept per video (default: 48)",
    )
    run.add_argument(
        "--overwrite",
        action="store_true",
        help="write into an output folder that already holds files",
    )
    run.add_argument(
        "--debug",
        action="store_true",
        help="show the full traceback of a failed run",
    )
    run.set_defaults(run_parser=run)
    return parser


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _run_command(args: argparse.Namespace) -> int:
    """Runs the pipeline for the ``run`` subcommand.

    Args:
        args: The parsed arguments.

    Returns:
        0 on success, 1 if the run failed.
    """
    from pydantic import ValidationError

    from spatial_change import pipeline

    run_parser: argparse.ArgumentParser = args.run_parser
    overrides = {
        key: value
        for key, value in (
            ("describer", args.describer),
            ("device", args.device),
            ("max_keyframes", args.max_keyframes),
        )
        if value is not None
    }
    try:
        config = pipeline.RunConfig.load(args.config, **overrides)
    except (OSError, ValueError) as error:
        if isinstance(error, ValidationError):
            error_text = "; ".join(
                f"{'.'.join(map(str, item['loc']))}: {item['msg']}"
                for item in error.errors()
            )
        else:
            error_text = str(error)
        run_parser.error(f"invalid configuration: {error_text}")

    def show(event: pipeline.StageEvent) -> None:
        line = pipeline.format_event(event)
        if line is not None:
            print(line, file=sys.stderr, flush=True)

    try:
        result = pipeline.run_videos(
            args.before,
            args.after,
            args.out,
            config,
            pipeline.default_models(config),
            show,
            overwrite=args.overwrite,
        )
    except Exception as error:
        if args.debug:
            raise
        text = str(error).strip()
        first = text.splitlines()[0] if text else type(error).__name__
        print(f"error: {first}", file=sys.stderr)
        return _EXIT_FAILURE

    changes = result.report.changes
    kinds = Counter(change.type.value for change in changes)
    tally = ", ".join(f"{n} {kind}" for kind, n in sorted(kinds.items()))
    print(
        _plural(len(changes), "change")
        + (f" ({tally})" if tally else "")
        + ", "
        + _plural(len(result.report.unobserved_regions), "unobserved region")
    )
    for path in result.files.all:
        print(f"wrote {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the command-line interface.

    Args:
        argv: Arguments to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        The process exit code: 0 on success, 1 if a run failed. Invalid
        arguments or settings exit with 2 through argparse.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return _run_command(args)
