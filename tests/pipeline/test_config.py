import json
from dataclasses import dataclass, replace
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
from spatial_change.pipeline.config import (
    build_stage_configs,
    override_fields,
)
from spatial_change.semantics import (
    LabelConfig,
    LandmarkConfig,
    RelationConfig,
)
from spatial_change.visibility import VisibilityConfig


@dataclass(frozen=True)
class Inner:
    depth: int = 1


@dataclass(frozen=True)
class Outer:
    name: str = "a"
    ratio: float = 0.5
    enabled: bool = True
    count: int = 3
    corner: tuple[int, int] = (0, 0)
    inner: Inner = Inner()


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


def test_no_stage_is_overridden_by_default() -> None:
    assert RunConfig().stages == {}


def test_a_stage_override_replaces_only_the_fields_it_names() -> None:
    config = RunConfig(
        seed=7,
        stages={
            "appearance": {"changed_below": 0.3},
            "match": {"min_fitness": 0.8},
        },
    )

    stages = build_stage_configs(config, "cpu")

    assert stages.appearance == replace(AppearanceConfig(), changed_below=0.3)
    assert stages.match.min_fitness == 0.8
    assert stages.match.max_cost == MatchConfig().max_cost
    assert stages.match.registration.seed == 7
    assert stages.candidates == CandidateConfig()


def test_a_nested_override_replaces_fields_of_the_nested_config() -> None:
    config = RunConfig(
        seed=7,
        stages={
            "lift": {"views": {"max_views": 7}},
            "match": {"registration": {"voxel_size": 0.05}},
            "landmarks": {"walls": {"min_width": 2.0}},
        },
    )

    stages = build_stage_configs(config, "cpu")

    assert stages.lift.views == replace(LiftConfig().views, max_views=7)
    assert stages.match.registration.voxel_size == 0.05
    assert stages.match.registration.seed == 7
    assert stages.landmarks.walls.min_width == 2.0
    assert stages.landmarks.walls.seed == 7


def test_an_override_wins_over_what_the_run_injects() -> None:
    config = RunConfig(seed=7, stages={"floor": {"seed": 99}})

    assert build_stage_configs(config, "cpu").floor.seed == 99


def test_the_overrides_load_from_a_json_file(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "seed": 3,
                "stages": {
                    "appearance": {"changed_below": 0.3},
                    "lift": {"views": {"max_views": 7}},
                },
            }
        ),
        encoding="utf-8",
    )

    config = RunConfig.load(path)

    stages = build_stage_configs(config, "cpu")
    assert stages.appearance.changed_below == 0.3
    assert stages.lift.views.max_views == 7
    assert RunConfig.load(path) == config
    path.write_text(config.model_dump_json(), encoding="utf-8")
    assert RunConfig.load(path) == config


def test_an_integer_is_a_valid_value_for_a_float_field() -> None:
    stages = build_stage_configs(
        RunConfig(stages={"visibility": {"abs_tolerance": 1}}), "cpu"
    )

    assert stages.visibility.abs_tolerance == 1.0
    assert isinstance(stages.visibility.abs_tolerance, float)


@pytest.mark.parametrize(
    ("stages", "fragments"),
    [
        ({"nope": {"x": 1}}, ["nope", "unknown stage"]),
        ({"sample_fps": {"x": 1}}, ["sample_fps", "unknown stage"]),
        (
            {"appearance": {"changed_belw": 0.3}},
            ["appearance", "changed_belw", "unknown field"],
        ),
        (
            {"lift": {"views": {"max_viewz": 7}}},
            ["lift.views", "max_viewz", "unknown field"],
        ),
        ({"visibility": {"device": "cuda"}}, ["visibility.device", "device"]),
        ({"lift": {"views": 7}}, ["lift.views", "object"]),
        (
            {"appearance": {"changed_below": 2.0}},
            ["appearance", "changed_below"],
        ),
        ({"lift": {"views": {"max_views": 0}}}, ["lift.views", "max_views"]),
        (
            {"appearance": {"changed_below": "0.3"}},
            ["appearance.changed_below"],
        ),
        ({"appearance": {"min_views": 2.5}}, ["appearance.min_views"]),
        ({"appearance": {"min_views": True}}, ["appearance.min_views"]),
        ({"appearance": {"voxel_size": None}}, ["appearance.voxel_size"]),
        (
            {"rerun": {"change_colors": {"added": [1, 2, 3]}}},
            ["rerun.change_colors"],
        ),
    ],
)
def test_a_bad_override_fails_when_the_config_is_built(
    stages: dict[str, Any], fragments: list[str]
) -> None:
    with pytest.raises(ValidationError) as excinfo:
        RunConfig(stages=stages)

    message = str(excinfo.value)
    for fragment in fragments:
        assert fragment in message


def test_a_stage_given_as_a_number_is_a_validation_error() -> None:
    with pytest.raises(ValidationError, match="stages"):
        RunConfig.model_validate({"stages": {"appearance": 0.3}})


def test_a_json_list_becomes_a_tuple_where_the_field_is_a_tuple() -> None:
    changed = override_fields(Outer(), {"corner": [3, 4]}, "outer")

    assert changed.corner == (3, 4)
    assert isinstance(changed.corner, tuple)


def test_a_list_of_the_wrong_length_or_type_is_refused_for_a_tuple() -> None:
    with pytest.raises(ValueError, match="outer.corner"):
        override_fields(Outer(), {"corner": [3, 4, 5]}, "outer")
    with pytest.raises(ValueError, match="outer.corner"):
        override_fields(Outer(), {"corner": 3}, "outer")


def test_override_fields_keeps_what_it_is_not_given() -> None:
    changed = override_fields(
        Outer(), {"ratio": 1, "inner": {"depth": 2}}, "outer"
    )

    assert changed == Outer(ratio=1.0, inner=Inner(depth=2))
    assert override_fields(Outer(), {}, "outer") == Outer()


def test_the_visibility_veto_is_switched_off_from_the_config() -> None:
    off = RunConfig(stages={"visibility": {"veto": False}})

    assert build_stage_configs(RunConfig(), "cuda").visibility.veto is True
    assert build_stage_configs(off, "cuda").visibility == VisibilityConfig(
        device="cuda", veto=False
    )


def test_the_visibility_veto_is_a_boolean_in_the_config() -> None:
    with pytest.raises(ValidationError, match=r"visibility\.veto"):
        RunConfig(stages={"visibility": {"veto": "false"}})
