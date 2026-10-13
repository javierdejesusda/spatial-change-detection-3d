from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from spatial_change.alignment import RefineConfig
from spatial_change.change import (
    AppearanceConfig,
    CandidateConfig,
    DescriptorConfig,
    LiftConfig,
    MatchConfig,
    UnobservedConfig,
)
from spatial_change.pipeline import RunConfig
from spatial_change.pipeline.config import build_stage_configs
from spatial_change.semantics import (
    LabelConfig,
    LandmarkConfig,
    RelationConfig,
)
from spatial_change.visibility import VisibilityConfig


def test_the_defaults_are_valid_and_leave_the_stage_configs_alone() -> None:
    stages = build_stage_configs(RunConfig(), "cpu")

    assert stages.appearance == AppearanceConfig()
    assert stages.candidates == CandidateConfig()
    assert stages.unobserved == UnobservedConfig()
    assert stages.lift == LiftConfig()
    assert stages.descriptors == DescriptorConfig()
    assert stages.label == LabelConfig()
    assert stages.relation == RelationConfig()
    assert stages.refine == RefineConfig()


def test_the_knobs_reach_the_stage_configs() -> None:
    config = RunConfig(
        sample_fps=2.0,
        max_keyframes=20,
        backbone_long_side=252,
        voxel_size=0.04,
        seed=7,
    )

    stages = build_stage_configs(config, "cuda")

    assert stages.sample_fps == 2.0
    assert stages.max_keyframes == 20
    assert stages.backbone_long_side == 252
    assert stages.fusion_voxel_size == 0.04
    assert stages.visibility == VisibilityConfig(device="cuda")
    assert stages.floor.seed == 7
    assert stages.landmarks.walls.seed == 7
    assert stages.match.registration.seed == 7
    assert stages.rerun.seed == 7


def test_the_frame_stride_is_off_by_default_and_reaches_the_stages() -> None:
    assert RunConfig().frame_stride is None
    assert build_stage_configs(RunConfig(), "cpu").frame_stride is None

    stages = build_stage_configs(RunConfig(frame_stride=30), "cpu")

    assert stages.frame_stride == 30


def test_the_frame_stride_loads_from_a_json_file(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"frame_stride": 15}', encoding="utf-8")

    assert RunConfig.load(path).frame_stride == 15


def test_every_geometric_stage_inherits_the_other_defaults() -> None:
    stages = build_stage_configs(RunConfig(), "cpu")

    assert stages.landmarks == LandmarkConfig(walls=stages.landmarks.walls)
    assert stages.match.min_fitness == MatchConfig().min_fitness


@pytest.mark.parametrize(
    "field",
    [
        {"sample_fps": 0.0},
        {"sample_fps": -1.0},
        {"max_keyframes": 1},
        {"max_keyframes": 65},
        {"backbone_long_side": 500},
        {"backbone_long_side": 0},
        {"voxel_size": 0.0},
        {"describer": "gpt"},
        {"device": ""},
        {"seed": -1},
        {"frame_stride": 0},
        {"frame_stride": -30},
    ],
)
def test_invalid_values_are_rejected(field: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RunConfig(**field)


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        RunConfig.model_validate({"sample_fpz": 3})


def test_the_config_is_frozen() -> None:
    config = RunConfig()

    with pytest.raises(ValidationError):
        config.seed = 1  # type: ignore[misc]


def test_a_voxel_size_of_none_keeps_every_point() -> None:
    stages = build_stage_configs(RunConfig(voxel_size=None), "cpu")

    assert stages.fusion_voxel_size is None


def test_a_json_file_round_trips(tmp_path: Path) -> None:
    config = RunConfig(
        sample_fps=3.0, max_keyframes=30, describer="template", seed=3
    )
    path = tmp_path / "config.json"
    path.write_text(config.model_dump_json(), encoding="utf-8")

    assert RunConfig.load(path) == config


def test_a_file_with_a_bad_value_is_a_validation_error(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"max_keyframes": 1}', encoding="utf-8")

    with pytest.raises(ValidationError, match="max_keyframes"):
        RunConfig.load(path)


def test_a_missing_file_is_reported_by_name(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing.json"):
        RunConfig.load(tmp_path / "missing.json")


def test_without_a_file_the_defaults_are_loaded() -> None:
    assert RunConfig.load(None) == RunConfig()


def test_overrides_win_over_the_file_and_the_rest_is_kept(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"seed": 3, "describer": "vlm", "max_keyframes": 30}',
        encoding="utf-8",
    )

    config = RunConfig.load(path, describer="template", device="cpu")

    assert (config.seed, config.max_keyframes) == (3, 30)
    assert (config.describer, config.device) == ("template", "cpu")


def test_an_invalid_override_is_a_validation_error() -> None:
    with pytest.raises(ValidationError, match="max_keyframes"):
        RunConfig.load(None, max_keyframes=1)


def test_a_file_that_is_not_an_object_is_a_value_error(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("[1, 2]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        RunConfig.load(path)
