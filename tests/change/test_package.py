import subprocess
import sys

import spatial_change.change as change


def test_package_exports_the_public_names() -> None:
    assert change.__all__ == sorted(change.__all__)
    assert set(change.__all__) == {
        "Candidate",
        "CandidateConfig",
        "ChangeCandidates",
        "extract_candidates",
        "find_candidates",
    }
    for name in change.__all__:
        assert hasattr(change, name)


def test_importing_the_package_does_not_load_the_heavy_stacks() -> None:
    code = (
        "import sys\n"
        "import spatial_change.change\n"
        "loaded = [m for m in ('torch', 'open3d', 'cv2') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )

    subprocess.run([sys.executable, "-I", "-c", code], check=True)
