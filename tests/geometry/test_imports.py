import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "spatial_change.geometry",
        "spatial_change.alignment",
        "spatial_change.alignment.scale",
    ],
)
def test_importing_the_package_does_not_load_open3d(module: str) -> None:
    code = f"import sys\nimport {module}\nassert 'open3d' not in sys.modules\n"

    subprocess.run([sys.executable, "-I", "-c", code], check=True)
