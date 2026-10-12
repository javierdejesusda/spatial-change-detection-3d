"""The ``changes.json`` report, schema version 1.

The report is the public output of the pipeline: a list of detected changes
between the ``before`` and ``after`` captures, plus regions that only one
capture observed and that therefore cannot be judged. Boxes are expressed in
the shared world frame of the aligned reconstructions, in meters when metric
scale is known.

All models forbid unknown fields and are immutable, so a parsed report is
exactly what the schema describes.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    model_validator,
)

from spatial_change.domain.camera import Epoch
from spatial_change.domain.scene import OrientedBox

_Vec3 = tuple[float, float, float]


class _ReportModel(BaseModel):
    # JSON has no NaN or infinity, so reject them instead of writing a
    # report that cannot be read back.
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ChangeType(StrEnum):
    """Kind of change detected for one object."""

    ADDED = "added"
    REMOVED = "removed"
    MOVED = "moved"
    UNCERTAIN = "uncertain"


class BoxModel(_ReportModel):
    """Serializable form of an ``OrientedBox``.

    Attributes:
        center: Box center in world coordinates.
        extent: Full side lengths along the box axes, all positive.
        rotation: Row-major 3x3 rotation whose columns are the box axes.
    """

    center: _Vec3
    extent: _Vec3
    rotation: tuple[_Vec3, _Vec3, _Vec3]

    @model_validator(mode="after")
    def _check_box(self) -> BoxModel:
        self.to_box()
        return self

    @classmethod
    def from_box(cls, box: OrientedBox) -> BoxModel:
        """Converts a domain box.

        Args:
            box: The box to serialize.

        Returns:
            The equivalent model.
        """
        return cls(
            center=box.center.tolist(),
            extent=box.extent.tolist(),
            rotation=box.rotation.tolist(),
        )

    def to_box(self) -> OrientedBox:
        """Converts to a domain box.

        Returns:
            The equivalent ``OrientedBox``.

        Raises:
            ValueError: If the fields do not describe a valid box.
        """
        return OrientedBox(
            center=np.array(self.center),
            extent=np.array(self.extent),
            rotation=np.array(self.rotation),
        )


class EvidenceCounts(_ReportModel):
    """Visibility evidence gathered for a change, in observation counts.

    Attributes:
        consistent: Observations that agree with the claimed state.
        free: Observations that saw through the location (free space).
        occluded: Observations where something nearer blocked the view.
        unobserved: Views where the location was outside the frustum.
    """

    consistent: NonNegativeInt
    free: NonNegativeInt
    occluded: NonNegativeInt
    unobserved: NonNegativeInt


class Change(_ReportModel):
    """One detected change.

    Attributes:
        change_id: Identifier, unique within the report.
        type: Kind of change.
        label: Object class name.
        description: Natural-language description of the change.
        confidence: Confidence in [0, 1].
        before_box: Object box in the ``before`` capture, if it was there.
        after_box: Object box in the ``after`` capture, if it is there.
        displacement_m: Distance between box centers, for moved objects.
        rotation_deg: Magnitude of the relative rotation in [0, 180], for
            moved objects.
        evidence: Visibility evidence behind the decision.
    """

    change_id: str
    type: ChangeType
    label: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    before_box: BoxModel | None
    after_box: BoxModel | None
    displacement_m: float | None = Field(ge=0.0)
    rotation_deg: float | None = Field(ge=0.0, le=180.0)
    evidence: EvidenceCounts

    @model_validator(mode="after")
    def _check_type_fields(self) -> Change:
        has_before = self.before_box is not None
        has_after = self.after_box is not None
        if self.type is ChangeType.ADDED and (has_before or not has_after):
            raise ValueError("added requires after_box and no before_box")
        if self.type is ChangeType.REMOVED and (has_after or not has_before):
            raise ValueError("removed requires before_box and no after_box")
        if self.type is ChangeType.MOVED and not (
            has_before
            and has_after
            and self.displacement_m is not None
            and self.rotation_deg is not None
        ):
            raise ValueError(
                "moved requires before_box, after_box, displacement_m, "
                "and rotation_deg"
            )
        return self


class UnobservedRegion(_ReportModel):
    """A region only one capture observed, so its change is unknown.

    Attributes:
        region_id: Identifier, unique within the report.
        box: Region bounds in world coordinates.
        epoch_seen: The capture that did observe the region.
    """

    region_id: str
    box: BoxModel
    epoch_seen: Epoch


class ChangeReport(_ReportModel):
    """The full ``changes.json`` document.

    Attributes:
        schema_version: Report schema version, always ``"1"``.
        changes: Detected changes, with distinct ids.
        unobserved_regions: Regions that cannot be judged, with distinct
            ids.
        metadata: Flat run information such as frame counts and settings.
    """

    schema_version: Literal["1"] = "1"
    changes: list[Change]
    unobserved_regions: list[UnobservedRegion]
    metadata: dict[str, str | int | float | bool]

    @model_validator(mode="after")
    def _check_unique_ids(self) -> ChangeReport:
        for kind, ids in (
            ("change", [c.change_id for c in self.changes]),
            ("region", [r.region_id for r in self.unobserved_regions]),
        ):
            repeated = sorted({i for i in ids if ids.count(i) > 1})
            if repeated:
                raise ValueError(f"duplicate {kind} ids: {repeated}")
        return self

    def to_json(self) -> str:
        """Serializes the report as JSON indented by 2 spaces."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str | bytes) -> ChangeReport:
        """Parses and validates a report.

        Args:
            text: JSON document.

        Returns:
            The parsed report.

        Raises:
            pydantic.ValidationError: If the document does not match the
                schema.
        """
        return cls.model_validate_json(text)

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Returns the JSON Schema of the report."""
        return cls.model_json_schema()
