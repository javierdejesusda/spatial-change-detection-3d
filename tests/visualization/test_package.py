import subprocess
import sys

import pytest

import spatial_change.visualization as visualization


def test_package_exports_the_public_names() -> None:
    assert visualization.__all__ == sorted(visualization.__all__)
    assert set(visualization.__all__) == {
        "CHANGE_COLORS",
        "GlbConfig",
        "RerunConfig",
        "ScenePoints",
        "SceneVisualization",
        "UNOBSERVED_COLOR",
        "build_recording",
        "change_color_hex",
        "export_glb",
        "save_recording",
        "write_glb",
    }
    for name in visualization.__all__:
        assert hasattr(visualization, name)


@pytest.mark.parametrize(
    "package", ["spatial_change.visualization", "spatial_change.reporting"]
)
def test_importing_a_package_loads_no_heavy_library(package: str) -> None:
    code = (
        "import sys\n"
        f"import {package}\n"
        "heavy = ('rerun', 'torch', 'transformers', 'open3d', 'cv2', 'PIL')\n"
        "loaded = [name for name in heavy if name in sys.modules]\n"
        "assert not loaded, loaded\n"
    )

    subprocess.run([sys.executable, "-I", "-c", code], check=True)
