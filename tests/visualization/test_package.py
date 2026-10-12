import subprocess
import sys

import spatial_change.visualization as visualization


def test_package_exports_the_public_names() -> None:
    assert visualization.__all__ == sorted(visualization.__all__)
    assert set(visualization.__all__) == {
        "RerunConfig",
        "SceneVisualization",
        "build_recording",
        "save_recording",
    }
    for name in visualization.__all__:
        assert hasattr(visualization, name)


def test_importing_the_package_loads_no_heavy_library() -> None:
    code = (
        "import sys\n"
        "import spatial_change.visualization\n"
        "heavy = ('rerun', 'torch', 'transformers', 'open3d', 'cv2')\n"
        "loaded = [name for name in heavy if name in sys.modules]\n"
        "assert not loaded, loaded\n"
    )

    subprocess.run([sys.executable, "-I", "-c", code], check=True)
