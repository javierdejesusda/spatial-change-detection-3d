from pathlib import Path

import pytest


@pytest.fixture(name="videos")
def videos_fixture(tmp_path: Path) -> tuple[Path, Path]:
    before = tmp_path / "before.mp4"
    after = tmp_path / "after.mov"
    before.write_bytes(b"v")
    after.write_bytes(b"v")
    return before, after
