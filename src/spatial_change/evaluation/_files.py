"""Writing a file so a reader never sees half of it."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """Gives a sibling of ``path`` to write, then moves it over ``path``.

    The move happens only if the block ends well, so a failed or interrupted
    write keeps the old file and leaves no partial one. The sibling is
    removed whatever happens.

    Args:
        path: The file to write.

    Yields:
        The path to write instead, in the same folder.
    """
    temporary = path.with_name(path.name + ".tmp")
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
