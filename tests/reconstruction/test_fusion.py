from dataclasses import replace

import numpy as np
import pytest

from spatial_change.domain import SE3, Epoch, Frame, Intrinsics
from spatial_change.reconstruction.fusion import fuse_frames


def _tiny_frame(
    depth: list[list[float]],
    confidence: list[list[float]],
    colors: list[list[tuple[int, int, int]]],
    epoch: Epoch = Epoch.BEFORE,
    pose: SE3 | None = None,
) -> Frame:
    """A 2x2 frame with fx = fy = 1 and the principal point at the origin."""
    return Frame(
        epoch=epoch,
        index=0,
        timestamp_s=0.0,
        intrinsics=Intrinsics(1.0, 1.0, 0.0, 0.0, 2, 2),
        world_from_camera=pose or SE3.identity(),
        image=np.array(colors, np.uint8),
        depth=np.array(depth, np.float32),
        confidence=np.array(confidence, np.float32),
    )


def _fusion_frames() -> list[Frame]:
    first = _tiny_frame(
        depth=[[1.0, 1.0], [1.0, 1.0]],
        confidence=[[0.5, 0.9], [0.3, 0.05]],
        colors=[[(10, 0, 255), (11, 1, 255)], [(12, 5, 254), (99, 99, 99)]],
    )
    second = _tiny_frame(
        depth=[[5.0, 0.0], [np.nan, -1.0]],
        confidence=[[0.7, 1.0], [1.0, 1.0]],
        colors=[[(1, 2, 3), (0, 0, 0)], [(0, 0, 0), (0, 0, 0)]],
    )
    return [first, second]


def test_fuse_keeps_confident_valid_pixels_in_row_major_order() -> None:
    fused = fuse_frames(_fusion_frames(), min_confidence=0.1)

    np.testing.assert_allclose(
        fused.points,
        [[0, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, 5]],
    )
    np.testing.assert_array_equal(
        fused.colors,
        [[10, 0, 255], [11, 1, 255], [12, 5, 254], [1, 2, 3]],
    )
    np.testing.assert_allclose(fused.confidence, [0.5, 0.9, 0.3, 0.7])
    assert fused.epoch == Epoch.BEFORE


def test_fuse_threshold_is_inclusive() -> None:
    fused = fuse_frames(_fusion_frames(), min_confidence=0.3)

    np.testing.assert_allclose(fused.confidence, [0.5, 0.9, 0.3, 0.7])
    assert len(fuse_frames(_fusion_frames(), min_confidence=0.31).points) == 3


def test_fuse_voxels_average_positions_and_colors() -> None:
    fused = fuse_frames(_fusion_frames(), min_confidence=0.1, voxel_size=2.0)

    np.testing.assert_allclose(
        fused.points, [[1 / 3, 1 / 3, 1.0], [0.0, 0.0, 5.0]]
    )
    np.testing.assert_array_equal(fused.colors, [[11, 2, 255], [1, 2, 3]])
    np.testing.assert_allclose(fused.confidence, [0.9, 0.7])


def test_fuse_voxel_order_does_not_depend_on_frame_order() -> None:
    frames = _fusion_frames()

    forward = fuse_frames(frames, voxel_size=2.0)
    backward = fuse_frames(frames[::-1], voxel_size=2.0)

    np.testing.assert_array_equal(forward.points, backward.points)
    np.testing.assert_array_equal(forward.colors, backward.colors)
    np.testing.assert_array_equal(forward.confidence, backward.confidence)


def test_fuse_voxels_match_a_per_voxel_reference_in_key_order() -> None:
    rng = np.random.default_rng(4)
    height, width = 20, 30
    frames = [
        Frame(
            Epoch.BEFORE,
            index,
            0.0,
            Intrinsics(25.0, 25.0, width / 2, height / 2, width, height),
            SE3(np.eye(3), np.array([0.3 * index, 0.0, 0.0])),
            image=rng.integers(0, 256, (height, width, 3), dtype=np.uint8),
            depth=rng.uniform(0.5, 6.0, (height, width)).astype(np.float32),
            confidence=rng.random((height, width)).astype(np.float32),
        )
        for index in range(3)
    ]
    size = 0.4

    fused = fuse_frames(frames, min_confidence=0.0, voxel_size=size)

    raw = fuse_frames(frames, min_confidence=0.0)
    keys = np.floor(raw.points / size).astype(np.int64)
    expected = sorted({tuple(key) for key in keys})
    assert len(fused.points) == len(expected)
    for row, key in enumerate(expected):
        members = np.all(keys == key, axis=1)
        np.testing.assert_allclose(
            fused.points[row], raw.points[members].mean(axis=0), atol=1e-12
        )
        np.testing.assert_array_equal(
            fused.colors[row],
            np.rint(raw.colors[members].mean(axis=0)).astype(np.uint8),
        )
        assert fused.confidence[row] == raw.confidence[members].max()


def test_fuse_uses_the_frame_poses_and_returns_the_frames() -> None:
    moved = _tiny_frame(
        depth=[[1.0, 0.0], [0.0, 0.0]],
        confidence=[[1.0, 1.0], [1.0, 1.0]],
        colors=[[(0, 0, 0)] * 2] * 2,
        pose=SE3(np.eye(3), np.array([10.0, 20.0, 30.0])),
    )

    fused = fuse_frames([moved])

    np.testing.assert_allclose(fused.points, [[10.0, 20.0, 31.0]])
    assert fused.frames == [moved]


def test_fuse_with_nothing_confident_returns_empty_arrays() -> None:
    fused = fuse_frames(_fusion_frames(), min_confidence=1.0, voxel_size=1.0)

    assert fused.points.shape == (0, 3)
    assert fused.colors.shape == (0, 3)
    assert fused.confidence.shape == (0,)


def test_fuse_rejects_bad_input() -> None:
    good = _fusion_frames()
    no_depth = replace(good[0], depth=None)
    no_confidence = replace(good[0], confidence=None)
    no_image = replace(good[0], image=None)
    other_epoch = replace(good[1], epoch=Epoch.AFTER)

    with pytest.raises(ValueError, match="at least one"):
        fuse_frames([])
    for broken in (no_depth, no_confidence, no_image):
        with pytest.raises(ValueError, match="depth, confidence, and image"):
            fuse_frames([broken])
    with pytest.raises(ValueError, match="epoch"):
        fuse_frames([good[0], other_epoch])
    with pytest.raises(ValueError, match="voxel_size"):
        fuse_frames(good, voxel_size=0.0)
    with pytest.raises(ValueError, match="min_confidence"):
        fuse_frames(good, min_confidence=2.0)


def test_fuse_voxels_merge_points_from_different_frames() -> None:
    near = _tiny_frame(
        depth=[[1.0, 0.0], [0.0, 0.0]],
        confidence=[[0.2, 0.0], [0.0, 0.0]],
        colors=[[(10, 20, 30)] * 2] * 2,
    )
    far = _tiny_frame(
        depth=[[1.0, 0.0], [0.0, 0.0]],
        confidence=[[0.8, 0.0], [0.0, 0.0]],
        colors=[[(20, 40, 61)] * 2] * 2,
        pose=SE3(np.eye(3), np.array([0.5, 0.0, 0.0])),
    )

    fused = fuse_frames([near, far], voxel_size=2.0)

    np.testing.assert_allclose(fused.points, [[0.25, 0.0, 1.0]])
    np.testing.assert_array_equal(fused.colors, [[15, 30, 46]])
    np.testing.assert_allclose(fused.confidence, [0.8])


def test_low_confidence_pixels_are_excluded_from_fusion() -> None:
    confidence = np.ones((2, 2), np.float32)
    confidence[:, 0] = 0.05
    half = _tiny_frame(
        depth=[[1.0, 1.0], [1.0, 1.0]],
        confidence=confidence.tolist(),
        colors=[[(0, 0, 0)] * 2] * 2,
    )

    fused = fuse_frames([half])

    assert len(fused.points) == 2
    assert fused.confidence.min() >= 0.1
