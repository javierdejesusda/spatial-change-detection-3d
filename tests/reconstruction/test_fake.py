import numpy as np
import pytest
from numpy.typing import NDArray

from spatial_change.ports import Reconstructor, Unloadable
from spatial_change.reconstruction.fake import FakeReconstructor


def _images(
    n: int, width: int = 32, height: int = 24
) -> list[NDArray[np.uint8]]:
    return [np.full((height, width, 3), i, np.uint8) for i in range(n)]


def test_fake_satisfies_the_reconstructor_protocol() -> None:
    fake = FakeReconstructor()

    assert isinstance(fake, Reconstructor)
    assert isinstance(fake, Unloadable)


def test_fake_returns_one_view_per_image_at_the_image_resolution() -> None:
    result = FakeReconstructor().reconstruct(_images(3, width=32, height=24))

    assert len(result.depth) == len(result.confidence) == 3
    for k, depth, confidence in zip(
        result.intrinsics, result.depth, result.confidence, strict=True
    ):
        assert (k.width, k.height) == (32, 24)
        assert depth.shape == confidence.shape == (24, 32)
        assert depth.dtype == confidence.dtype == np.float32


def test_fake_uses_the_requested_depth_resolution() -> None:
    fake = FakeReconstructor(depth_size=(16, 12))

    result = fake.reconstruct(_images(2, width=32, height=24))

    assert (result.intrinsics[0].width, result.intrinsics[0].height) == (16, 12)
    assert result.depth[0].shape == (12, 16)


def test_fake_places_views_along_the_x_axis_with_known_depth() -> None:
    fake = FakeReconstructor(depth_m=3.0, baseline_m=0.5)

    result = fake.reconstruct(_images(3))

    for i, pose in enumerate(result.world_from_camera):
        np.testing.assert_array_equal(pose.rotation, np.eye(3))
        np.testing.assert_array_equal(pose.translation, [0.5 * i, 0.0, 0.0])
    assert np.all(result.depth[0] == 3.0)
    assert np.all(result.confidence[0] == 1.0)


def test_fake_intrinsics_are_centered_and_isotropic() -> None:
    k = FakeReconstructor().reconstruct(_images(1)).intrinsics[0]

    assert k.fx == k.fy
    assert (k.cx, k.cy) == (15.5, 11.5)


def test_fake_records_calls_and_unloads() -> None:
    fake = FakeReconstructor()
    first, second = _images(2), _images(3)

    fake.reconstruct(first)
    fake.reconstruct(second)
    fake.unload()

    assert [len(call) for call in fake.calls] == [2, 3]
    assert fake.calls[0][0] is first[0]
    assert fake.calls[1][2] is second[2]
    assert fake.unload_count == 1


def test_fake_rejects_an_empty_image_list() -> None:
    with pytest.raises(ValueError, match="at least one image"):
        FakeReconstructor().reconstruct([])
