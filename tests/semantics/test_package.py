import subprocess
import sys

import spatial_change.semantics as semantics


def test_package_exports_the_public_names() -> None:
    assert semantics.__all__ == sorted(semantics.__all__)
    assert set(semantics.__all__) == {
        "FURNITURE",
        "INDOOR_OBJECTS",
        "STRUCTURE",
        "ChangeFact",
        "Direction",
        "LabelConfig",
        "Landmark",
        "LandmarkConfig",
        "LandmarkRelation",
        "RelationConfig",
        "build_change_fact",
        "describe_direction",
        "distance_to_box",
        "extract_landmarks",
        "format_distance",
        "format_rotation",
        "label_instances",
        "nearest_landmarks",
        "supporting_surface",
    }
    for name in semantics.__all__:
        assert hasattr(semantics, name)


def test_importing_the_package_does_not_load_the_heavy_stacks() -> None:
    code = (
        "import sys\n"
        "import spatial_change.semantics\n"
        "heavy = ('torch', 'transformers', 'open3d', 'cv2')\n"
        "loaded = [m for m in heavy if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )

    subprocess.run([sys.executable, "-I", "-c", code], check=True)
