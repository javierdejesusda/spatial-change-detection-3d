import numpy as np
import pytest
from appearance_scenes import reconstruction
from report_scenes import change
from visibility_scenes import scene_clouds

from spatial_change.domain import (
    BoxModel,
    ChangeType,
    Epoch,
    EpochReconstruction,
    OrientedBox,
)
from spatial_change.geometry.views import ViewConfig
from spatial_change.pipeline.crops import CropConfig, crops_for_changes
from spatial_change.ports import ChangeCrops as DescriberCrops
from spatial_change.reporting import ChangeCrops as ReportCrops
from spatial_change.synthetic.scene import ObjectStatus

SEED = 0


def model(box: OrientedBox | None) -> BoxModel | None:
    return None if box is None else BoxModel.from_box(box)


def epochs() -> tuple[EpochReconstruction, EpochReconstruction]:
    pair, before, after = scene_clouds(SEED)
    return (
        reconstruction(Epoch.BEFORE, pair.before.frames, before.points),
        reconstruction(Epoch.AFTER, pair.after.frames, after.points),
    )


def changed(status: ObjectStatus) -> OrientedBox | tuple[OrientedBox, ...]:
    pair, _, _ = scene_clouds(SEED)
    (obj,) = [o for o in pair.objects if o.status is status]
    if status is ObjectStatus.MOVED:
        assert obj.box_before is not None and obj.box_after is not None
        return obj.box_before, obj.box_after
    box = obj.box_before or obj.box_after
    assert box is not None
    return box


def test_a_removed_object_has_crops_of_the_before_capture_only() -> None:
    before, after = epochs()
    box = changed(ObjectStatus.REMOVED)
    assert isinstance(box, OrientedBox)
    item = change("change-0", ChangeType.REMOVED, "chair", model(box), None)

    views = crops_for_changes([item], before, after)["change-0"]

    assert isinstance(views.describer, DescriberCrops)
    assert isinstance(views.report, ReportCrops)
    assert 1 <= len(views.describer.before) <= CropConfig().views.max_views
    assert views.describer.after == ()
    assert views.report.before is not None
    assert views.report.after is None
    for crop in views.describer.before:
        assert crop.dtype == np.uint8
        assert crop.ndim == 3
        assert min(crop.shape[:2]) >= CropConfig().min_crop_px


def test_an_added_object_has_crops_of_the_after_capture_only() -> None:
    before, after = epochs()
    box = changed(ObjectStatus.ADDED)
    assert isinstance(box, OrientedBox)
    item = change("change-0", ChangeType.ADDED, "chair", None, model(box))

    views = crops_for_changes([item], before, after)["change-0"]

    assert views.describer.before == ()
    assert len(views.describer.after) >= 1
    assert views.report.before is None
    assert views.report.after is not None


def test_a_moved_object_has_crops_on_both_sides() -> None:
    before, after = epochs()
    boxes = changed(ObjectStatus.MOVED)
    assert isinstance(boxes, tuple)
    item = change(
        "change-0",
        ChangeType.MOVED,
        "chair",
        model(boxes[0]),
        model(boxes[1]),
    )

    views = crops_for_changes([item], before, after)["change-0"]

    assert views.describer.before and views.describer.after
    assert views.report.before is not None and views.report.after is not None


def test_the_report_crop_is_the_view_that_sees_the_most() -> None:
    before, after = epochs()
    box = changed(ObjectStatus.REMOVED)
    assert isinstance(box, OrientedBox)
    item = change("change-0", ChangeType.REMOVED, "chair", model(box), None)

    views = crops_for_changes([item], before, after)["change-0"]

    assert views.report.before is views.describer.before[0]


def test_a_box_that_no_frame_sees_gives_no_crop() -> None:
    before, after = epochs()
    far = OrientedBox(np.array([100.0, 100.0, 1.0]), np.ones(3), np.eye(3))
    item = change("change-0", ChangeType.REMOVED, "chair", model(far), None)

    views = crops_for_changes([item], before, after)["change-0"]

    assert views.describer.before == ()
    assert views.report.before is None


def test_every_change_gets_an_entry_and_the_views_are_capped() -> None:
    before, after = epochs()
    box = changed(ObjectStatus.REMOVED)
    assert isinstance(box, OrientedBox)
    changes = [
        change(f"change-{i}", ChangeType.REMOVED, "chair", model(box), None)
        for i in range(2)
    ]
    config = CropConfig(views=ViewConfig(max_views=1))

    views = crops_for_changes(changes, before, after, config)

    assert set(views) == {"change-0", "change-1"}
    assert len(views["change-0"].describer.before) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [("margin", -0.1), ("min_margin_px", -1.0), ("min_crop_px", 0)],
)
def test_the_config_rejects_invalid_values(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        CropConfig(**{field: value})
