from collections.abc import Sequence

import numpy as np
import pytest
from fakes import ShortReconstructor

from spatial_change.capture import VideoFrame
from spatial_change.domain import Epoch
from spatial_change.reconstruction.fake import FakeReconstructor
from spatial_change.reconstruction.joint import reconstruct_joint


def _frames(
    indices: Sequence[int], fill: int, width: int = 32, height: int = 24
) -> list[VideoFrame]:
    return [
        VideoFrame(
            index=i,
            timestamp_s=i / 30.0,
            image=np.full((height, width, 3), fill, np.uint8),
        )
        for i in indices
    ]


def test_joint_pass_runs_one_reconstruction_on_both_epochs() -> None:
    fake = FakeReconstructor()
    before = _frames([0, 10], fill=10)
    after = _frames([5, 15, 25], fill=20)

    reconstruct_joint(fake, before, after)

    assert len(fake.calls) == 1
    images = fake.calls[0]
    assert len(images) == 5
    for image, frame in zip(images, [*before, *after], strict=True):
        assert image is frame.image


def test_joint_pass_splits_the_result_by_epoch() -> None:
    fake = FakeReconstructor(baseline_m=0.5)
    before = _frames([0, 10], fill=10)
    after = _frames([5, 15, 25], fill=20)

    joint = reconstruct_joint(fake, before, after)

    assert [f.epoch for f in joint.before] == [Epoch.BEFORE] * 2
    assert [f.epoch for f in joint.after] == [Epoch.AFTER] * 3
    assert [f.index for f in joint.before] == [0, 10]
    assert [f.index for f in joint.after] == [5, 15, 25]
    assert [f.timestamp_s for f in joint.after] == [5 / 30, 15 / 30, 25 / 30]


def test_joint_pass_keeps_both_epochs_in_one_world_frame() -> None:
    fake = FakeReconstructor(baseline_m=0.5)

    joint = reconstruct_joint(
        fake, _frames([0, 1], fill=1), _frames([0, 1, 2], fill=2)
    )

    x = [f.world_from_camera.translation[0] for f in joint.before + joint.after]
    assert x == [0.0, 0.5, 1.0, 1.5, 2.0]


def test_joint_pass_attaches_geometry_and_colors_at_depth_resolution() -> None:
    fake = FakeReconstructor(depth_size=(16, 12), depth_m=3.0)

    joint = reconstruct_joint(
        fake, _frames([0], fill=10), _frames([0], fill=20)
    )

    for frame, fill in zip(joint.before + joint.after, (10, 20), strict=True):
        assert (frame.intrinsics.width, frame.intrinsics.height) == (16, 12)
        assert frame.image is not None
        assert frame.image.shape == (12, 16, 3)
        assert np.all(frame.image == fill)
        assert frame.depth is not None
        assert np.all(frame.depth == 3.0)
        assert frame.confidence is not None
        assert frame.confidence.shape == (12, 16)


def test_joint_pass_reuses_images_already_at_depth_resolution() -> None:
    before = _frames([0], fill=10)

    joint = reconstruct_joint(FakeReconstructor(), before, _frames([0], 20))

    assert joint.before[0].image is before[0].image


def test_joint_pass_accepts_any_sequence() -> None:
    joint = reconstruct_joint(
        FakeReconstructor(), tuple(_frames([0], 1)), tuple(_frames([0], 2))
    )

    assert isinstance(joint.before, list)
    assert isinstance(joint.after, list)


def test_joint_pass_does_not_unload_the_reconstructor() -> None:
    fake = FakeReconstructor()

    reconstruct_joint(fake, _frames([0], 1), _frames([0], 2))

    assert fake.unload_count == 0


@pytest.mark.parametrize(("n_before", "n_after"), [(0, 2), (2, 0), (0, 0)])
def test_joint_pass_rejects_an_empty_epoch(n_before: int, n_after: int) -> None:
    fake = FakeReconstructor()

    with pytest.raises(ValueError, match="epoch"):
        reconstruct_joint(
            fake, _frames(range(n_before), 1), _frames(range(n_after), 2)
        )
    assert fake.calls == []


def test_joint_pass_rejects_a_result_of_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="4 views for 5 images"):
        reconstruct_joint(
            ShortReconstructor(), _frames([0, 1], 1), _frames([0, 1, 2], 2)
        )
