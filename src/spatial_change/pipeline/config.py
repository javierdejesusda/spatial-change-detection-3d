"""The settings of a run and the stage configs they stand for.

A ``RunConfig`` holds the few knobs a user may turn, and the overrides of any
field of any stage config. ``build_stage_configs`` turns it into the frozen
config of every stage; whatever the config does not name keeps the default
of its stage.

Importing this module does not load torch, transformers, open3d, or cv2.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from spatial_change.alignment import RefineConfig
from spatial_change.change import (
    AppearanceConfig,
    CandidateConfig,
    DescriptorConfig,
    LiftConfig,
    MatchConfig,
    UnobservedConfig,
)
from spatial_change.geometry import FloorConfig
from spatial_change.geometry.projection import DEFAULT_MIN_CONFIDENCE
from spatial_change.geometry.walls import WallConfig
from spatial_change.semantics import (
    LabelConfig,
    LandmarkConfig,
    RelationConfig,
)
from spatial_change.visibility import VisibilityConfig
from spatial_change.visualization import RerunConfig

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

BACKBONE_MULTIPLE = 14
"""Both sides of a frame are rounded to a multiple of the ViT patch size."""
MAX_VIDEO_KEYFRAMES = 64
"""Half the views the reconstruction accepts, so both videos fit in one."""
MIN_KEYFRAMES = 8
"""Keyframes a video is topped up to when enough sharp frames exist."""


class RunConfig(BaseModel):
    """The knobs of a run.

    Attributes:
        sample_fps: Frames per second taken from each video before the
            keyframes are chosen, positive.
        max_keyframes: Keyframes kept per video, from 2 to
            ``MAX_VIDEO_KEYFRAMES``.
        backbone_long_side: Longest side in pixels of the frames given to
            the reconstruction, a multiple of 14.
        voxel_size: Edge in meters of the voxels that thin each epoch's
            fused cloud, or ``None`` to keep every point.
        describer: ``vlm`` writes the descriptions with the local
            vision-language model, ``template`` with the deterministic
            template, and ``auto`` picks ``vlm`` on a CUDA device and
            ``template`` otherwise.
        device: The torch device of the models, or ``auto`` for CUDA when
            there is one and the CPU otherwise.
        frame_stride: When set, each video contributes every frame whose
            index is a multiple of this stride instead of the keyframes the
            sharpness and overlap rules pick: no frame is dropped for being
            blurry or redundant. The index is the slot on the 30 fps timeline
            that ``ffmpeg -r 30`` writes, the one annotations count, not the
            position in the stream; they differ for a video that is not 30
            fps, where a frame can fill no slot or several. Only above the
            keyframe cap (the smaller of ``max_keyframes`` and half the
            views the reconstruction accepts) are the frames thinned,
            evenly, keeping the first and the last; the run metadata then
            says how many frames were on the stride. At least 1, or
            ``None`` for the usual keyframes.
        seed: Seed of every random search and subsampling, at least 0.
        stages: Overrides of the stage configs, by stage name and then by
            field name, for example ``{"appearance": {"changed_below":
            0.3}}``. An object on a field that holds a config replaces
            fields of that nested config, for example ``{"lift": {"views":
            {"max_views": 7}}}``. They are applied on top of the defaults,
            the seed, and the device, so they win. The stage names are the
            config fields of ``StageConfigs``, the field names those of the
            stage config, except ``visibility.device``, which the run
            resolves from ``device``. Every override is checked when the
            config is built.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_fps: float = Field(default=5.0, gt=0.0, allow_inf_nan=False)
    max_keyframes: int = Field(default=48, ge=2, le=MAX_VIDEO_KEYFRAMES)
    backbone_long_side: int = Field(default=518, ge=BACKBONE_MULTIPLE)
    voxel_size: float | None = Field(default=0.02, gt=0.0, allow_inf_nan=False)
    describer: Literal["auto", "vlm", "template"] = "auto"
    device: str = Field(default="auto", min_length=1)
    frame_stride: int | None = Field(default=None, ge=1)
    seed: int = Field(default=0, ge=0)
    stages: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator("backbone_long_side")
    @classmethod
    def _multiple_of_patch(cls, value: int) -> int:
        if value % BACKBONE_MULTIPLE:
            raise ValueError(
                f"backbone_long_side must be a multiple of {BACKBONE_MULTIPLE}"
            )
        return value

    @field_validator("stages")
    @classmethod
    def _stages_build(
        cls, value: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        build_stage_configs(cls.model_construct(stages=value), "cpu")
        return value

    @classmethod
    def load(cls, path: Path | None = None, **overrides: Any) -> Self:
        """Reads a JSON config file and applies overrides on top of it.

        Args:
            path: The file, or ``None`` for the defaults.
            **overrides: Field values that replace those of the file.

        Returns:
            The validated config.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a valid JSON object.
            pydantic.ValidationError: If a value is invalid or a key is
                unknown.
        """
        data: dict[str, Any] = {}
        if path is not None:
            if not path.is_file():
                raise FileNotFoundError(f"config file not found: {path}")
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path} is not valid JSON: {error}"
                ) from error
            if not isinstance(data, dict):
                raise ValueError(f"{path} must hold a JSON object")
        return cls.model_validate({**data, **overrides})


def resolve_device(config: RunConfig) -> str:
    """Names the torch device a run uses.

    Args:
        config: The run settings.

    Returns:
        ``config.device``, or for ``auto`` ``cuda`` when torch sees a CUDA
        device and ``cpu`` otherwise.
    """
    if config.device != "auto":
        return config.device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class StageConfigs:
    """The config of every stage of a run.

    Attributes:
        sample_fps: Frames per second taken from each video.
        max_keyframes: Keyframes kept per video.
        backbone_long_side: Longest side of the frames given to the
            reconstruction.
        frame_stride: Stride, in slots of the 30 fps timeline, of the frames
            each video contributes, thinned evenly only above the keyframe
            cap and then recorded in the run metadata, or ``None`` for the
            usual keyframes.
        fusion_voxel_size: Voxel edge that thins a fused cloud, or ``None``.
        min_confidence: Pixels with a lower confidence stay out of the
            fused clouds.
        floor: Floor search of the gravity alignment.
        refine: Registration of the AFTER epoch onto the BEFORE epoch.
        visibility: The visibility vote.
        appearance: The appearance comparison.
        candidates: The change candidates.
        unobserved: The unobserved regions.
        lift: The lifting of the candidates to objects.
        descriptors: The appearance descriptors of the objects.
        label: The labeling of the objects.
        landmarks: The landmark search.
        match: The association of the objects.
        relation: The spatial facts of the changes.
        rerun: The Rerun recording.
    """

    sample_fps: float
    max_keyframes: int
    backbone_long_side: int
    frame_stride: int | None
    fusion_voxel_size: float | None
    min_confidence: float
    floor: FloorConfig
    refine: RefineConfig
    visibility: VisibilityConfig
    appearance: AppearanceConfig
    candidates: CandidateConfig
    unobserved: UnobservedConfig
    lift: LiftConfig
    descriptors: DescriptorConfig
    label: LabelConfig
    landmarks: LandmarkConfig
    match: MatchConfig
    relation: RelationConfig
    rerun: RerunConfig


def _new_value(current: object, value: Any, path: str) -> object:
    """Gives the new value of a field from the plain JSON value for it.

    Raises:
        ValueError: If the value does not fit the field.
    """
    if is_dataclass(current) and not isinstance(current, type):
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}: expected an object of fields")
        return override_fields(current, value, path)
    if isinstance(current, tuple):
        if not isinstance(value, list | tuple) or len(value) != len(current):
            raise ValueError(f"{path}: expected a list of {len(current)}")
        return tuple(
            _new_value(item, new, f"{path}[{i}]")
            for i, (item, new) in enumerate(zip(current, value, strict=True))
        )
    kind = type(current)
    if kind not in (bool, int, float, str):
        raise ValueError(f"{path}: is not a plain setting")
    if kind is float and type(value) is int:
        return float(value)
    if type(value) is not kind:
        raise ValueError(
            f"{path}: expected {kind.__name__}, got {type(value).__name__}"
        )
    return value


def override_fields[T: DataclassInstance](
    config: T, overrides: Mapping[str, Any], where: str
) -> T:
    """Replaces fields of a frozen dataclass with plain JSON values.

    An object on a field that holds a dataclass replaces fields of that
    nested dataclass, recursively. A list on a tuple field becomes a tuple
    and an integer on a float field a float. Any other value must be of the
    type of the field.

    Args:
        config: The dataclass to start from; it is not changed.
        overrides: Field names and their new values.
        where: Name of ``config`` in the messages, such as ``appearance``.

    Returns:
        A copy of ``config`` with the fields replaced.

    Raises:
        ValueError: If a name is not a field, a value does not fit its
            field, or ``config`` refuses the result. The message starts
            with the dotted path of the field or of the config.
    """
    known = [f.name for f in fields(config)]
    changes = {}
    for name, value in overrides.items():
        path = f"{where}.{name}"
        if name not in known:
            raise ValueError(
                f"{path}: unknown field, expected one of {', '.join(known)}"
            )
        changes[name] = _new_value(getattr(config, name), value, path)
    try:
        return replace(config, **changes)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{where}: {error}") from error


_RESOLVED_FIELDS = {"visibility": "device"}
"""Fields a stage override may not set, as the run resolves them."""


def _with_overrides(
    stages: StageConfigs, overrides: Mapping[str, Mapping[str, Any]]
) -> StageConfigs:
    """Applies the stage overrides of a run.

    Raises:
        ValueError: If a name is not a stage, an override sets a field the
            run resolves, or a stage config refuses its overrides.
    """
    known = [
        f.name for f in fields(stages) if is_dataclass(getattr(stages, f.name))
    ]
    changes = {}
    for stage, values in overrides.items():
        if stage not in known:
            raise ValueError(
                f"{stage}: unknown stage, expected one of {', '.join(known)}"
            )
        resolved = _RESOLVED_FIELDS.get(stage)
        if resolved in values:
            raise ValueError(
                f"{stage}.{resolved}: the run resolves the {resolved}, "
                "it cannot be overridden"
            )
        changes[stage] = override_fields(getattr(stages, stage), values, stage)
    return replace(stages, **changes)


def build_stage_configs(config: RunConfig, device: str) -> StageConfigs:
    """Builds the config of every stage from the knobs of a run.

    Args:
        config: The run settings, with the stage overrides they hold.
        device: The torch device of the visibility vote, as resolved by
            ``resolve_device``.

    Returns:
        The stage configs.

    Raises:
        ValueError: If a stage override is not valid. A ``RunConfig`` that
            was built has none.
    """
    seed = config.seed
    landmarks = LandmarkConfig()
    match = MatchConfig()
    stages = StageConfigs(
        sample_fps=config.sample_fps,
        max_keyframes=config.max_keyframes,
        backbone_long_side=config.backbone_long_side,
        frame_stride=config.frame_stride,
        fusion_voxel_size=config.voxel_size,
        min_confidence=DEFAULT_MIN_CONFIDENCE,
        floor=replace(FloorConfig(), seed=seed),
        refine=RefineConfig(),
        visibility=VisibilityConfig(device=device),
        appearance=AppearanceConfig(),
        candidates=CandidateConfig(),
        unobserved=UnobservedConfig(),
        lift=LiftConfig(),
        descriptors=DescriptorConfig(),
        label=LabelConfig(),
        landmarks=replace(landmarks, walls=replace(WallConfig(), seed=seed)),
        match=replace(
            match, registration=replace(match.registration, seed=seed)
        ),
        relation=RelationConfig(),
        rerun=RerunConfig(seed=seed),
    )
    return _with_overrides(stages, config.stages)
