import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
@pytest.mark.skipif(
    shutil.which("uv") is None or shutil.which("git") is None,
    reason="needs uv and git",
)
def test_sdist_ships_only_files_git_does_not_ignore(tmp_path: Path) -> None:
    subprocess.run(
        ["uv", "build", "--sdist", "--out-dir", str(tmp_path)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    visible = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    (sdist,) = tmp_path.glob("*.tar.gz")

    with tarfile.open(sdist) as archive:
        shipped = {name.split("/", 1)[1] for name in archive.getnames()}

    assert shipped - {"PKG-INFO"} <= set(visible)
