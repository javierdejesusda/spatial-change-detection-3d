"""Command-line interface for spatial-change."""

import argparse
from collections.abc import Sequence

from spatial_change import __version__


def build_parser() -> argparse.ArgumentParser:
    """Builds the top-level argument parser.

    Returns:
        The configured parser. Subcommands are registered on it by later
        pipeline stages.
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
    parser.add_subparsers(dest="command")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the command-line interface.

    Args:
        argv: Arguments to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        The process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
    return 0
