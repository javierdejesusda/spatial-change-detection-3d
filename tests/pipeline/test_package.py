import subprocess
import sys

import pytest

import spatial_change.pipeline as pipeline


def test_package_exports_the_public_names() -> None:
    assert pipeline.__all__ == sorted(pipeline.__all__)
    assert set(pipeline.__all__) == {
        "STAGES",
        "Models",
        "ProgressCallback",
        "RunConfig",
        "StageEvent",
        "default_models",
        "format_event",
    }
    for name in pipeline.__all__:
        assert hasattr(pipeline, name)


@pytest.mark.parametrize(
    "module", ["spatial_change.pipeline", "spatial_change.cli"]
)
def test_importing_does_not_load_the_heavy_stacks(module: str) -> None:
    code = (
        "import sys\n"
        f"import {module}\n"
        "heavy = ('torch', 'transformers', 'open3d', 'cv2')\n"
        "loaded = [m for m in heavy if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )

    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def test_the_stages_are_the_pinned_twelve() -> None:
    assert pipeline.STAGES == (
        "keyframes",
        "reconstruction",
        "alignment",
        "voting",
        "appearance",
        "candidates",
        "lifting",
        "descriptors",
        "labeling",
        "matching",
        "describing",
        "outputs",
    )
