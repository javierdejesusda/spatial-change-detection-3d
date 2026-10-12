import json
import subprocess
import sys

import pytest

from spatial_change.perception import (
    DinoFeatureEncoder,
    FakeFeatureEncoder,
    QwenVlm,
    SamSegmenter,
    SiglipLabeler,
)
from spatial_change.pipeline import RunConfig, default_models
from spatial_change.pipeline.models import loaded
from spatial_change.reconstruction import Pi3XReconstructor


def test_the_default_factories_build_the_local_models_without_loading() -> None:
    models = default_models(RunConfig(device="cpu"))

    assert isinstance(models.reconstructor(), Pi3XReconstructor)
    assert isinstance(models.encoder(), DinoFeatureEncoder)
    assert isinstance(models.segmenter(), SamSegmenter)
    assert isinstance(models.labeler(), SiglipLabeler)
    assert isinstance(models.vlm(), QwenVlm)
    assert models.max_frames == 128
    assert set(models.ids) == {
        "reconstruction",
        "features",
        "segmentation",
        "labels",
        "descriptions",
    }


def test_the_default_models_expose_every_adapter_setting() -> None:
    models = default_models(RunConfig(device="cpu"))

    reconstruction = models.settings["reconstruction"]
    features = models.settings["features"]
    assert reconstruction["max_frames"] == models.max_frames
    assert reconstruction["device"] == "cpu"
    assert features["model_id"] == models.ids["features"]
    assert features["fallback_model_id"]
    json.dumps(models.settings)


def test_building_the_default_factories_does_not_import_torch() -> None:
    code = (
        "import sys\n"
        "from spatial_change.pipeline import RunConfig, default_models\n"
        "default_models(RunConfig(device='cpu'))\n"
        "assert 'torch' not in sys.modules\n"
    )

    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def test_a_loaded_model_is_unloaded_when_the_block_ends() -> None:
    encoder = FakeFeatureEncoder()

    with loaded(lambda: encoder) as model:
        assert model is encoder
        assert encoder.unload_count == 0

    assert encoder.unload_count == 1


def test_a_loaded_model_is_unloaded_when_the_block_fails() -> None:
    encoder = FakeFeatureEncoder()

    with pytest.raises(RuntimeError, match="boom"), loaded(lambda: encoder):
        raise RuntimeError("boom")

    assert encoder.unload_count == 1
