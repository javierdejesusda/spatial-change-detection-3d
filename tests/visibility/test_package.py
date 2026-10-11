import subprocess
import sys

import spatial_change.visibility as visibility


def test_package_exports_the_public_names() -> None:
    assert visibility.__all__ == sorted(visibility.__all__)
    assert set(visibility.__all__) == {
        "CrossEvidence",
        "PointEvidence",
        "PointState",
        "VisibilityConfig",
        "Vote",
        "vote_epochs",
        "vote_points",
    }
    for name in visibility.__all__:
        assert hasattr(visibility, name)


def test_importing_the_package_does_not_load_torch() -> None:
    code = (
        "import sys\n"
        "import spatial_change.visibility\n"
        "assert 'torch' not in sys.modules\n"
    )

    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def test_importing_the_package_does_not_load_the_reconstruction_stack() -> None:
    code = (
        "import sys\n"
        "import spatial_change.visibility\n"
        "assert 'cv2' not in sys.modules\n"
        "loaded = [\n"
        "    name for name in sys.modules\n"
        "    if name == 'spatial_change.reconstruction'\n"
        "    or name.startswith('spatial_change.reconstruction.')\n"
        "]\n"
        "assert not loaded, loaded\n"
    )

    subprocess.run([sys.executable, "-I", "-c", code], check=True)
