import subprocess
import sys

import spatial_change.perception as perception


def test_package_exports_the_public_names() -> None:
    assert perception.__all__ == sorted(perception.__all__)
    assert set(perception.__all__) == {
        "DINOV2_MODEL_ID",
        "DINOV3_MODEL_ID",
        "DinoConfig",
        "DinoFeatureEncoder",
        "FakeFeatureEncoder",
        "FakeMaskSegmenter",
        "SAM21_MODEL_ID",
        "SAM3_MODEL_ID",
        "SamConfig",
        "SamSegmenter",
        "sample_features",
    }
    for name in perception.__all__:
        assert hasattr(perception, name)


def test_importing_the_package_does_not_load_torch_or_transformers() -> None:
    code = (
        "import sys\n"
        "import spatial_change.perception\n"
        "loaded = [m for m in ('torch', 'transformers') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )

    subprocess.run([sys.executable, "-I", "-c", code], check=True)
