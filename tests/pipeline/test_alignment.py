import dataclasses
from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray
from rotations import perturbation
from visibility_scenes import scene_clouds

from spatial_change.domain import Frame, Sim3
from spatial_change.geometry import TargetIndex
from spatial_change.pipeline import RunConfig
from spatial_change.pipeline.alignment import _thin_points, align_epochs
from spatial_change.pipeline.config import build_stage_configs
from spatial_change.reconstruction import fuse_frames

SEED = 0
STAGES = build_stage_configs(RunConfig(voxel_size=0.05), "cpu")


def with_confidence(frames: Sequence[Frame]) -> list[Frame]:
    assert all(f.depth is not None for f in frames)
    return [
        dataclasses.replace(
            f,
            confidence=np.ones(f.depth.shape, np.float32),  # type: ignore[union-attr]
        )
        for f in frames
    ]


def tilt() -> Sim3:
    pose = perturbation((1.0, 2.0, 2.0), 25.0, (0.4, -0.3, 0.2))
    return Sim3(1.0, pose.rotation, pose.translation)


def median_gap(
    source: NDArray[np.float64], target: NDArray[np.float64]
) -> float:
    sample = source[:: max(1, len(source) // 3000)]
    return float(np.median(TargetIndex(target).nearest(sample)[1]))


def test_a_tilted_world_ends_with_the_floor_at_zero_and_z_up() -> None:
    pair, original_before, original_after = scene_clouds(SEED)
    world = tilt()
    before = [f.transformed(world) for f in with_confidence(pair.before.frames)]
    after = [f.transformed(world) for f in with_confidence(pair.after.frames)]

    aligned = align_epochs(before, after, STAGES)

    for recon, original in (
        (aligned.before, original_before),
        (aligned.after, original_after),
    ):
        height = recon.points[:, 2]
        assert abs(height.min()) < 0.05
        assert abs(height.max() - original.points[:, 2].max()) < 0.05
    assert aligned.room_from_world.scale == 1.0
    assert aligned.floor_observed


def test_one_alignment_serves_both_epochs() -> None:
    pair, _, _ = scene_clouds(SEED)
    before = with_confidence(pair.before.frames)
    after = with_confidence(pair.after.frames)

    aligned = align_epochs(before, after, STAGES)

    unaligned = fuse_frames(after, voxel_size=0.05).points
    moved = aligned.room_from_world.apply(unaligned)
    assert median_gap(moved, aligned.before.points) < 0.02


def test_a_small_offset_of_the_after_epoch_is_registered_away() -> None:
    pair, _, _ = scene_clouds(SEED)
    offset = perturbation((0.0, 0.0, 1.0), 2.0, (0.06, -0.05, 0.0))
    shift = Sim3.from_se3(offset)
    before = with_confidence(pair.before.frames)
    after = [f.transformed(shift) for f in with_confidence(pair.after.frames)]
    crooked = fuse_frames(after, voxel_size=0.05)

    aligned = align_epochs(before, after, STAGES)

    room = aligned.room_from_world
    assert median_gap(room.apply(crooked.points), aligned.before.points) > 0.04
    assert median_gap(aligned.after.points, aligned.before.points) < 0.02


def test_epochs_without_confident_points_are_a_clear_error() -> None:
    pair, _, _ = scene_clouds(SEED)
    blind = [
        dataclasses.replace(f, confidence=np.zeros(f.depth.shape, np.float32))  # type: ignore[union-attr]
        for f in pair.before.frames
    ]
    other = [
        dataclasses.replace(f, confidence=np.zeros(f.depth.shape, np.float32))  # type: ignore[union-attr]
        for f in pair.after.frames
    ]

    with pytest.raises(ValueError, match="no confident points"):
        align_epochs(blind, other, STAGES)


def test_thinning_keeps_everything_below_the_limit() -> None:
    points = np.arange(30, dtype=np.float64).reshape(10, 3)

    assert _thin_points(points, 10, seed=0) is points


def test_thinning_draws_without_replacement_and_repeats_per_seed() -> None:
    points = np.arange(3000, dtype=np.float64).reshape(1000, 3)

    first = _thin_points(points, 100, seed=1)
    again = _thin_points(points, 100, seed=1)
    other = _thin_points(points, 100, seed=2)

    assert first.shape == (100, 3)
    assert len({tuple(row) for row in first}) == 100
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_thinning_is_not_a_fixed_stride() -> None:
    # Fused points come in voxel-key order, so a fixed stride can miss a
    # whole plane that sits at the positions it skips.
    count, limit = 10_000, 1_000
    points = np.zeros((count, 3))
    points[:, 0] = np.arange(count)
    floor = np.arange(count) % (count // limit) == 1

    kept = _thin_points(points, limit, seed=0)

    on_floor = floor[kept[:, 0].astype(int)]
    assert 0.03 < on_floor.mean() < 0.2
