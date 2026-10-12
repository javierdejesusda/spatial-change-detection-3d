import json
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from rotations import rotation_z

from spatial_change.domain import (
    BoxModel,
    Change,
    ChangeReport,
    ChangeType,
    EvidenceCounts,
    OrientedBox,
    UnobservedRegion,
)

_BOX = BoxModel(
    center=(0.0, 0.0, 0.5),
    extent=(0.2, 0.3, 0.4),
    rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
)
_EVIDENCE = EvidenceCounts(consistent=10, free=2, occluded=1, unobserved=0)


def _change(**overrides: Any) -> Change:
    fields: dict[str, Any] = {
        "change_id": "c0",
        "type": ChangeType.MOVED,
        "label": "chair",
        "description": "The chair moved toward the window.",
        "confidence": 0.8,
        "before_box": _BOX,
        "after_box": _BOX,
        "displacement_m": 1.2,
        "rotation_deg": 15.0,
        "evidence": _EVIDENCE,
    }
    fields.update(overrides)
    return Change(**fields)


def _report() -> ChangeReport:
    return ChangeReport(
        changes=[
            _change(),
            _change(
                change_id="c1",
                type=ChangeType.ADDED,
                before_box=None,
                displacement_m=None,
                rotation_deg=None,
            ),
            _change(change_id="c2", type=ChangeType.UNCERTAIN, after_box=None),
        ],
        unobserved_regions=[
            UnobservedRegion(region_id="u0", box=_BOX, epoch_seen="before")
        ],
        metadata={"frames": 120, "aligned": True, "rmse_m": 0.01, "v": "x"},
    )


def test_json_round_trip_is_equal() -> None:
    report = _report()
    text = report.to_json()

    assert ChangeReport.from_json(text) == report
    assert json.loads(text)["schema_version"] == "1"
    assert text.startswith("{\n  ")


def test_metadata_keeps_value_types() -> None:
    back = ChangeReport.from_json(_report().to_json())

    assert back.metadata["aligned"] is True
    assert isinstance(back.metadata["frames"], int)


def test_extra_field_is_rejected() -> None:
    data = json.loads(_report().to_json())
    data["extra"] = 1
    with pytest.raises(ValidationError):
        ChangeReport.model_validate(data)

    data = json.loads(_report().to_json())
    data["changes"][0]["note"] = "x"
    with pytest.raises(ValidationError):
        ChangeReport.model_validate(data)


def test_models_are_frozen() -> None:
    change = _change()
    with pytest.raises(ValidationError):
        change.label = "table"  # type: ignore[misc]


def test_schema_version_is_fixed() -> None:
    data = json.loads(_report().to_json())
    data["schema_version"] = "2"
    with pytest.raises(ValidationError):
        ChangeReport.model_validate(data)


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "type": ChangeType.ADDED,
            "before_box": None,
            "after_box": None,
        },
        {
            "type": ChangeType.ADDED,
            "before_box": _BOX,
            "after_box": _BOX,
        },
        {
            "type": ChangeType.REMOVED,
            "before_box": None,
            "after_box": None,
        },
        {
            "type": ChangeType.REMOVED,
            "before_box": _BOX,
            "after_box": _BOX,
        },
        {"type": ChangeType.MOVED, "before_box": None},
        {"type": ChangeType.MOVED, "after_box": None},
        {"type": ChangeType.MOVED, "displacement_m": None},
        {"type": ChangeType.MOVED, "rotation_deg": None},
    ],
    ids=[
        "added-without-after",
        "added-with-before",
        "removed-without-before",
        "removed-with-after",
        "moved-without-before",
        "moved-without-after",
        "moved-without-displacement",
        "moved-without-rotation",
    ],
)
def test_type_specific_box_rules(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _change(**overrides)


def test_valid_type_specific_changes() -> None:
    removed = _change(type=ChangeType.REMOVED, after_box=None)
    uncertain = _change(
        type=ChangeType.UNCERTAIN,
        before_box=None,
        after_box=None,
        displacement_m=None,
        rotation_deg=None,
    )

    assert removed.type is ChangeType.REMOVED
    assert uncertain.before_box is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"confidence": 1.1},
        {"confidence": -0.1},
        {"displacement_m": -0.5},
        {"rotation_deg": 181.0},
        {
            "evidence": {
                "consistent": -1,
                "free": 0,
                "occluded": 0,
                "unobserved": 0,
            }
        },
    ],
)
def test_out_of_range_values_raise(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _change(**overrides)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_floats_are_rejected(value: float) -> None:
    # JSON has no NaN or infinity, so accepting them would write a report
    # that cannot be read back.
    with pytest.raises(ValidationError):
        ChangeReport(changes=[], unobserved_regions=[], metadata={"x": value})
    with pytest.raises(ValidationError):
        _change(displacement_m=value)


def test_unobserved_region_epoch_is_constrained() -> None:
    with pytest.raises(ValidationError):
        UnobservedRegion(region_id="u", box=_BOX, epoch_seen="during")  # type: ignore[arg-type]


def test_json_schema_has_schema_version() -> None:
    schema = ChangeReport.json_schema()

    assert "schema_version" in schema["properties"]
    assert schema["additionalProperties"] is False


def test_box_model_round_trips_through_oriented_box() -> None:
    box = OrientedBox(
        center=np.array([1.0, -2.0, 0.5]),
        extent=np.array([0.4, 0.5, 0.6]),
        rotation=rotation_z(17.0),
    )

    model = BoxModel.from_box(box)
    back = BoxModel.model_validate_json(model.model_dump_json()).to_box()

    np.testing.assert_array_equal(back.center, box.center)
    np.testing.assert_array_equal(back.extent, box.extent)
    np.testing.assert_array_equal(back.rotation, box.rotation)
    assert all(isinstance(v, float) for v in model.center)


def test_box_model_rejects_invalid_box() -> None:
    with pytest.raises(ValidationError):
        BoxModel(
            center=(0.0, 0.0, 0.0),
            extent=(1.0, 0.0, 1.0),
            rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )


def test_duplicate_change_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match="c0"):
        ChangeReport(
            changes=[_change(), _change()],
            unobserved_regions=[],
            metadata={},
        )


def test_duplicate_region_ids_are_rejected() -> None:
    region = UnobservedRegion(region_id="u0", box=_BOX, epoch_seen="before")

    with pytest.raises(ValidationError, match="u0"):
        ChangeReport(
            changes=[], unobserved_regions=[region, region], metadata={}
        )
