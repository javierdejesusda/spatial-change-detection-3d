from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from spatial_change.capture import VideoFrame, probe_video, read_frames
from spatial_change.domain import Frame
from spatial_change.geometry import unproject_depth
from spatial_change.ports import ReconstructionResult
from spatial_change.reconstruction import (
    Pi3XReconstructor,
    reconstruct_joint,
)

pytestmark = pytest.mark.gpu

Box = tuple[NDArray[np.float64], NDArray[np.float64]]

MIB = 1 << 20
GIB = 1 << 30
MIN_EPOCH_OVERLAP_IOU = 0.5
SCENE_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "scenediff"
    / "living_room_3_living_room_4"
)


@pytest.fixture
def adapter() -> Iterator[Pi3XReconstructor]:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.cuda.reset_peak_memory_stats()
    reconstructor = Pi3XReconstructor()
    yield reconstructor
    reconstructor.unload()


def _textured_views(
    n_frames: int, width: int = 640, height: int = 360, shift_px: int = 6
) -> list[NDArray[np.uint8]]:
    """Crops of one procedural scene, shifted a few pixels per view."""
    rng = np.random.default_rng(0)
    canvas_width = width + shift_px * n_frames
    noise = rng.integers(0, 256, (height, canvas_width, 3), dtype=np.uint8)
    canvas = cv2.GaussianBlur(noise, (0, 0), 3.0)
    canvas = cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX)
    for _ in range(60):
        x = int(rng.integers(0, canvas_width - 40))
        y = int(rng.integers(0, height - 40))
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        if rng.random() < 0.5:
            cv2.rectangle(canvas, (x, y), (x + 40, y + 30), color, -1)
        else:
            cv2.circle(canvas, (x, y), 20, color, -1)
    return [
        np.ascontiguousarray(canvas[:, i * shift_px : i * shift_px + width])
        for i in range(n_frames)
    ]


def _assert_valid_views(
    result: ReconstructionResult, n_views: int, size: tuple[int, int]
) -> None:
    width, height = size
    assert len(result.depth) == n_views
    for k, pose, depth, confidence in zip(
        result.intrinsics,
        result.world_from_camera,
        result.depth,
        result.confidence,
        strict=True,
    ):
        assert (k.width, k.height) == (width, height)
        assert depth.shape == confidence.shape == (height, width)
        assert depth.dtype == confidence.dtype == np.float32
        assert np.isfinite(pose.translation).all()
        assert np.isfinite(pose.rotation).all()
        assert np.isfinite(depth).all()
        assert (depth > 0).any()
        assert confidence.min() >= 0.0
        assert confidence.max() <= 1.0


def test_reconstructs_a_procedural_scene_within_the_vram_cap(
    adapter: Pi3XReconstructor,
) -> None:
    import torch

    free_bytes, _ = torch.cuda.mem_get_info()

    result = adapter.reconstruct(_textured_views(8))

    _assert_valid_views(result, 8, (518, 294))
    assert torch.cuda.max_memory_reserved() <= free_bytes


def test_the_cap_turns_an_allocation_over_the_budget_into_an_error(
    adapter: Pi3XReconstructor,
) -> None:
    import torch

    free_bytes, _ = torch.cuda.mem_get_info()
    budget = free_bytes + torch.cuda.memory_reserved()
    adapter.load()

    with pytest.raises(torch.cuda.OutOfMemoryError):
        torch.empty(budget + GIB, dtype=torch.uint8, device="cuda")


def test_unload_returns_the_gpu_memory_and_a_later_call_reloads(
    adapter: Pi3XReconstructor,
) -> None:
    import torch

    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    views = _textured_views(2)

    adapter.reconstruct(views)
    allocated_loaded = torch.cuda.memory_allocated()
    adapter.unload()

    assert allocated_loaded - allocated_before > 1024 * MIB
    assert torch.cuda.memory_allocated() - allocated_before < 16 * MIB
    assert torch.cuda.memory_reserved() - reserved_before < 64 * MIB
    _assert_valid_views(adapter.reconstruct(views), 2, (518, 294))


def _keyframes(path: Path, count: int) -> list[VideoFrame]:
    stride = max(1, probe_video(path).frame_count // count)
    return list(read_frames(path, stride=stride, max_frames=count))


def _scene_points(frames: list[Frame]) -> NDArray[np.float64]:
    """Confident points of the frames in the world frame, subsampled."""
    return np.concatenate(
        [
            unproject_depth(
                f.depth,
                f.intrinsics,
                f.world_from_camera,
                mask=f.confidence > 0.5,
            )[::16]
            for f in frames
            if f.depth is not None and f.confidence is not None
        ]
    )


def _core_box(points: NDArray[np.float64]) -> Box:
    """Axis-aligned box of the 1st to 99th percentile of each axis."""
    low, high = np.percentile(points, [1, 99], axis=0)
    return low, high


def _scene_extent(frames: list[Frame]) -> float:
    low, high = _core_box(_scene_points(frames))
    return float(np.linalg.norm(high - low))


def _box_iou(first: Box, second: Box) -> float:
    overlap = np.minimum(first[1], second[1]) - np.maximum(first[0], second[0])
    intersection = float(np.prod(np.clip(overlap, 0.0, None)))
    volumes = [float(np.prod(high - low)) for low, high in (first, second)]
    return intersection / (sum(volumes) - intersection)


@pytest.mark.skipif(
    not (
        (SCENE_DIR / "video1.mp4").exists()
        and (SCENE_DIR / "video2.mp4").exists()
    ),
    reason="SceneDiff videos are not downloaded",
)
def test_joint_pass_on_real_frames_shares_one_world_frame(
    adapter: Pi3XReconstructor,
) -> None:
    before = _keyframes(SCENE_DIR / "video1.mp4", 8)
    after = _keyframes(SCENE_DIR / "video2.mp4", 8)
    assert len(before) == len(after) == 8

    joint = reconstruct_joint(adapter, before, after)

    frames = joint.before + joint.after
    assert len(joint.before) == len(joint.after) == 8
    assert all(f.depth is not None and f.confidence is not None for f in frames)
    _assert_valid_views(
        ReconstructionResult(
            intrinsics=[f.intrinsics for f in frames],
            world_from_camera=[f.world_from_camera for f in frames],
            depth=[f.depth for f in frames if f.depth is not None],
            confidence=[
                f.confidence for f in frames if f.confidence is not None
            ],
        ),
        16,
        (294, 518),
    )
    assert all(
        f.image is not None and f.image.shape == (518, 294, 3) for f in frames
    )
    extent = _scene_extent(frames)
    assert extent > 0.0
    for epoch in (joint.before, joint.after):
        centers = np.stack([f.world_from_camera.translation for f in epoch])
        steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        assert np.all(steps < extent)
    overlap = _box_iou(
        _core_box(_scene_points(joint.before)),
        _core_box(_scene_points(joint.after)),
    )
    assert overlap > MIN_EPOCH_OVERLAP_IOU
