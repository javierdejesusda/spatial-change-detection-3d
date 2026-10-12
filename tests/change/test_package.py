import subprocess
import sys

import spatial_change.change as change


def test_package_exports_the_public_names() -> None:
    assert change.__all__ == sorted(change.__all__)
    assert set(change.__all__) == {
        "AppearanceConfig",
        "Candidate",
        "CandidateConfig",
        "ChangeCandidates",
        "CrossAppearance",
        "DescriptorConfig",
        "LiftConfig",
        "LiftedObjects",
        "MatchConfig",
        "PointAppearance",
        "UnobservedConfig",
        "associate",
        "compare_appearance",
        "describe_instances",
        "extract_candidates",
        "extract_unobserved",
        "find_candidates",
        "find_unobserved_regions",
        "fuse_appearance",
        "gated_assignment",
        "lift_candidates",
        "lift_change_candidates",
        "linear_assignment",
    }
    for name in change.__all__:
        assert hasattr(change, name)


def test_importing_the_package_does_not_load_the_heavy_stacks() -> None:
    code = (
        "import sys\n"
        "import spatial_change.change\n"
        "heavy = ('torch', 'transformers', 'open3d', 'cv2')\n"
        "loaded = [m for m in heavy if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )

    subprocess.run([sys.executable, "-I", "-c", code], check=True)
