from spatial_change.domain import ChangeType
from spatial_change.semantics.relations import ChangeFact, LandmarkRelation


def near(
    label: str, relation: str = "near", distance_m: float = 0.4
) -> LandmarkRelation:
    return LandmarkRelation(
        landmark_id=f"{label}-1",
        label=label,
        distance_m=distance_m,
        relation=relation,  # type: ignore[arg-type]
    )


def make_fact(
    change_type: ChangeType = ChangeType.MOVED,
    label: str = "chair",
    support_before: str | None = None,
    support_after: str | None = None,
    near_before: list[LandmarkRelation] | None = None,
    near_after: list[LandmarkRelation] | None = None,
    direction: str | None = None,
    toward: str | None = None,
    displacement_m: float | None = None,
    displacement_text: str | None = None,
    rotation_deg: float | None = None,
    rotation_text: str | None = None,
) -> ChangeFact:
    return ChangeFact(
        change_id="c1",
        type=change_type,
        label=label,
        support_before=support_before,
        support_after=support_after,
        near_before=near_before or [],
        near_after=near_after or [],
        direction=direction,
        toward=toward,
        displacement_m=displacement_m,
        displacement_text=displacement_text,
        rotation_deg=rotation_deg,
        rotation_text=rotation_text,
    )


def moved_fact(**overrides: object) -> ChangeFact:
    values: dict[str, object] = {
        "direction": "to the left",
        "toward": "table",
        "displacement_m": 0.3,
        "displacement_text": "about 30 cm",
        "rotation_deg": 70.0,
        "rotation_text": "about 70 degrees",
    }
    values.update(overrides)
    return make_fact(ChangeType.MOVED, **values)  # type: ignore[arg-type]
