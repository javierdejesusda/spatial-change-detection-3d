import dataclasses

import numpy as np
import pytest
from appearance_scenes import recolor_blue, reconstruction
from lifting_scenes import Cuboid, HandScene, hand_scene, ring_poses
from matching_scenes import instance

from spatial_change.change import DescriptorConfig, describe_instances
from spatial_change.domain import Epoch, EpochReconstruction
from spatial_change.perception import FakeFeatureEncoder

PATCH = 8
BLUE_BOX = Cuboid((-1.2, -0.4, 0.0), (-0.6, 0.2, 0.6))
GRAY_BOX = Cuboid((0.6, -0.3, 0.0), (1.2, 0.3, 0.9))


def _scene(epoch: Epoch = Epoch.BEFORE) -> HandScene:
    """Renders a blue and a gray box with a ring of eight cameras."""
    scene = hand_scene(
        [BLUE_BOX, GRAY_BOX], ring_poses(8, radius=2.2, height=1.8), epoch
    )
    frames = []
    for frame, instances in zip(scene.frames, scene.instance_maps, strict=True):
        assert frame.image is not None
        image = recolor_blue(frame.image, instances == 0)
        frames.append(dataclasses.replace(frame, image=image))
    return HandScene(frames, scene.instance_maps, epoch)


def _reconstruction(scene: HandScene) -> EpochReconstruction:
    return reconstruction(scene.epoch, scene.frames, scene.object_points(0))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_the_feature_is_a_unit_vector_of_the_objects_look() -> None:
    scene = _scene()
    blue = instance(scene.object_points(0), "blue", Epoch.BEFORE)

    (described,) = describe_instances(
        _reconstruction(scene), (blue,), FakeFeatureEncoder(PATCH)
    )

    assert described.feature is not None
    assert described.feature.dtype == np.float32
    assert np.linalg.norm(described.feature) == pytest.approx(1.0, abs=1e-5)


def test_objects_that_look_different_get_different_features() -> None:
    scene = _scene()
    blue = instance(scene.object_points(0), "blue", Epoch.BEFORE)
    gray = instance(scene.object_points(1), "gray", Epoch.BEFORE)

    described = describe_instances(
        _reconstruction(scene), (blue, gray), FakeFeatureEncoder(PATCH)
    )

    assert described[0].feature is not None
    assert described[1].feature is not None
    # The fake encoder gives pure blue and gray a cosine of 0.76; patches
    # on the silhouette mix in the background.
    assert _cosine(described[0].feature, described[1].feature) < 0.9


def test_two_views_of_one_object_describe_it_alike() -> None:
    scene = _scene()
    points = scene.object_points(0)
    half = len(points) // 2
    first = instance(points[:half], "first", Epoch.BEFORE)
    second = instance(points[half:], "second", Epoch.BEFORE)

    described = describe_instances(
        _reconstruction(scene), (first, second), FakeFeatureEncoder(PATCH)
    )

    assert described[0].feature is not None
    assert described[1].feature is not None
    assert _cosine(described[0].feature, described[1].feature) > 0.99


def test_each_frame_is_encoded_once_for_all_the_instances() -> None:
    scene = _scene()
    instances = tuple(
        instance(scene.object_points(i), f"object-{i}", Epoch.BEFORE)
        for i in range(2)
    )
    encoder = FakeFeatureEncoder(PATCH)

    describe_instances(_reconstruction(scene), instances, encoder)

    assert encoder.encode_count == len(scene.frames)


def test_every_voxel_counts_once_whatever_its_point_count() -> None:
    scene = _scene()
    blue = scene.object_points(0)[:1]
    gray = scene.object_points(1)[:1]
    crowded = np.concatenate([np.repeat(blue, 500, axis=0), gray])
    even = np.concatenate([blue, gray])
    encoder = FakeFeatureEncoder(PATCH)

    (many,) = describe_instances(
        _reconstruction(scene),
        (instance(crowded, "many", Epoch.BEFORE),),
        encoder,
    )
    (one,) = describe_instances(
        _reconstruction(scene),
        (instance(even, "one", Epoch.BEFORE),),
        encoder,
    )

    assert many.feature is not None
    assert one.feature is not None
    assert np.allclose(many.feature, one.feature, atol=1e-5)


def test_an_instance_no_frame_sees_gets_no_feature() -> None:
    scene = _scene()
    hidden = instance(
        np.array([[0.0, 0.0, -5.0], [0.1, 0.0, -5.0], [0.0, 0.1, -5.0]]),
        "hidden",
        Epoch.BEFORE,
    )

    (described,) = describe_instances(
        _reconstruction(scene), (hidden,), FakeFeatureEncoder(PATCH)
    )

    assert described.feature is None


def test_the_other_fields_are_kept_and_the_input_is_not_changed() -> None:
    scene = _scene()
    original = instance(
        scene.object_points(0), "blue", Epoch.BEFORE, label="chair"
    )

    (described,) = describe_instances(
        _reconstruction(scene), (original,), FakeFeatureEncoder(PATCH)
    )

    assert described is not original
    assert original.feature is None
    assert described.object_id == "blue"
    assert described.label == "chair"
    assert described.points is original.points
    assert described.box is original.box


def test_no_instances_encode_nothing() -> None:
    encoder = FakeFeatureEncoder(PATCH)

    result = describe_instances(_reconstruction(_scene()), (), encoder)

    assert result == ()
    assert encoder.encode_count == 0


def test_an_instance_of_the_other_epoch_is_rejected() -> None:
    scene = _scene()
    stranger = instance(scene.object_points(0), "late", Epoch.AFTER)

    with pytest.raises(ValueError, match="late"):
        describe_instances(
            _reconstruction(scene), (stranger,), FakeFeatureEncoder(PATCH)
        )


def test_frames_without_depth_are_skipped() -> None:
    scene = _scene()
    frames = [dataclasses.replace(scene.frames[0], depth=None)]
    frames += scene.frames[1:]
    blue = instance(scene.object_points(0), "blue", Epoch.BEFORE)
    encoder = FakeFeatureEncoder(PATCH)

    (described,) = describe_instances(
        reconstruction(Epoch.BEFORE, frames, scene.object_points(0)),
        (blue,),
        encoder,
    )

    assert described.feature is not None
    assert encoder.encode_count == len(frames) - 1


def test_a_frame_that_sees_no_instance_is_not_encoded() -> None:
    scene = _scene()
    blind = dataclasses.replace(
        scene.frames[0], depth=np.full_like(scene.frames[0].depth, 50.0)
    )
    frames = [blind] + scene.frames[1:]
    blue = instance(scene.object_points(0), "blue", Epoch.BEFORE)
    encoder = FakeFeatureEncoder(PATCH)

    (described,) = describe_instances(
        reconstruction(Epoch.BEFORE, frames, scene.object_points(0)),
        (blue,),
        encoder,
    )

    assert described.feature is not None
    assert encoder.encode_count == len(frames) - 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("voxel_size", 0.0),
        ("abs_tolerance", -1.0),
        ("rel_tolerance", float("inf")),
        ("min_confidence", 1.5),
    ],
)
def test_config_rejects_invalid_values(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        DescriptorConfig(**{field: value})
