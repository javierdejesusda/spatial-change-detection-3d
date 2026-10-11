from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pytest
from fakes import ShortReconstructor
from numpy.typing import NDArray
from rotations import random_rotation

from spatial_change.capture import VideoFrame
from spatial_change.domain import SE3, Epoch, Frame, Sim3
from spatial_change.geometry.projection import unproject_depth
from spatial_change.ports import ReconstructionResult
from spatial_change.reconstruction.densify import (
    ChunkAlignment,
    DensifyConfig,
    densify_epoch,
)
from spatial_change.reconstruction.fake import FakeReconstructor
from spatial_change.reconstruction.fusion import fuse_frames
from spatial_change.synthetic import SceneConfig, generate_scene_pair


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 0},
        {"anchors_per_chunk": 0},
        {"min_confidence": -0.1},
        {"min_confidence": 1.5},
        {"min_confidence": float("nan")},
        {"trim_fraction": -0.1},
        {"trim_fraction": 1.0},
        {"trim_iterations": -1},
        {"ratio_gate": 1.0},
        {"ratio_gate": 0.5},
        {"max_alignment_points": 2},
        {"min_alignment_points": 2},
        {"min_alignment_points": 50, "max_alignment_points": 40},
    ],
)
def test_config_rejects_invalid_values(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        DensifyConfig(**kwargs)  # type: ignore[arg-type]


def test_config_defaults_are_valid() -> None:
    config = DensifyConfig()

    assert config.chunk_size == 32
    assert config.anchors_per_chunk == 4
    assert config.min_confidence == 0.1
    assert config.ratio_gate == 1.5


@pytest.fixture(scope="module")
def truth_frames() -> list[Frame]:
    config = SceneConfig(width=64, height=48, frames_per_epoch=12)
    frames = generate_scene_pair(0, config).before.frames
    assert len({f.image.tobytes() for f in frames if f.image is not None}) == 12
    return frames


class SimilarityReconstructor:
    """Reports ground truth moved into a different world on every call.

    The first ``corrupt_views`` views get a fraction of their depth
    multiplied by factors drawn from ``corrupt_factors`` and a fraction
    replaced by ``invalid_value`` (zero or NaN) at full confidence.

    Call ``k`` returns the poses of the requested frames in the world
    ``chunk_world = transforms[k] (truth_world)`` with depth multiplied by
    that scale, as a real model with its own world frame would.

    Attributes:
        calls: The frame keys of the images passed to each call.
    """

    def __init__(
        self,
        truth: dict[bytes, Frame],
        transforms: Sequence[Sim3],
        corrupt_views: int = 0,
        corrupt_fraction: float = 0.0,
        corrupt_confidence: float = 1.0,
        corrupt_factors: Sequence[float] = (3.0,),
        invalid_fraction: float = 0.0,
        invalid_value: float = 0.0,
    ) -> None:
        self._truth = truth
        self._transforms = transforms
        self._corrupt_views = corrupt_views
        self._corrupt_fraction = corrupt_fraction
        self._corrupt_confidence = corrupt_confidence
        self._corrupt_factors = np.asarray(corrupt_factors, np.float32)
        self._invalid_fraction = invalid_fraction
        self._invalid_value = invalid_value
        self.calls: list[list[bytes]] = []

    def reconstruct(
        self, images: Sequence[NDArray[np.uint8]]
    ) -> ReconstructionResult:
        call = len(self.calls)
        keys = [image.tobytes() for image in images]
        self.calls.append(keys)
        sim = self._transforms[call]
        rng = np.random.default_rng(call)
        result = ReconstructionResult([], [], [], [])
        for view, key in enumerate(keys):
            frame = self._truth[key]
            assert frame.depth is not None
            depth = frame.depth * np.float32(sim.scale)
            confidence = np.ones_like(depth)
            if view < self._corrupt_views:
                bad = rng.random(depth.shape) < self._corrupt_fraction
                factors = rng.choice(self._corrupt_factors, size=depth.shape)
                depth[bad] *= factors[bad]
                confidence[bad] = self._corrupt_confidence
                holes = rng.random(depth.shape) < self._invalid_fraction
                depth[holes] = np.float32(self._invalid_value)
            result.intrinsics.append(frame.intrinsics)
            result.world_from_camera.append(
                sim.transform_pose(frame.world_from_camera)
            )
            result.depth.append(depth)
            result.confidence.append(confidence)
        return result

    def unload(self) -> None:
        raise AssertionError("the caller owns the reconstructor")


@dataclass
class Scene:
    truth: dict[bytes, Frame]
    anchor_videos: list[VideoFrame]
    anchor_frames: list[Frame]
    others: list[VideoFrame]


def _video_frame(frame: Frame) -> VideoFrame:
    assert frame.image is not None
    return VideoFrame(frame.index, frame.timestamp_s, frame.image)


def make_scene(
    truth_frames: list[Frame],
    anchor_indices: Sequence[int],
    other_indices: Sequence[int],
    joint: Sim3 | None = None,
) -> Scene:
    """Assigns ground-truth frames to the requested source indices.

    The anchor frames play the joint pass output: ground truth moved into
    the anchor world ``joint`` (identity by default) with full confidence.
    """
    joint = joint or Sim3.identity()
    indices = [*anchor_indices, *other_indices]
    reindexed = [
        replace(frame, index=index, timestamp_s=index / 10.0)
        for frame, index in zip(truth_frames, indices, strict=False)
    ]
    assert len(reindexed) == len(indices)
    truth = {f.image.tobytes(): f for f in reindexed if f.image is not None}
    n_anchors = len(anchor_indices)
    anchors = [
        replace(
            frame.transformed(joint),
            confidence=np.ones_like(frame.depth),
        )
        for frame in reindexed[:n_anchors]
    ]
    return Scene(
        truth=truth,
        anchor_videos=[_video_frame(f) for f in reindexed[:n_anchors]],
        anchor_frames=anchors,
        others=[_video_frame(f) for f in reindexed[n_anchors:]],
    )


def known_similarities(count: int, seed: int = 0) -> list[Sim3]:
    rng = np.random.default_rng(seed)
    return [
        Sim3(
            float(rng.uniform(0.4, 3.0)),
            random_rotation(rng),
            rng.normal(scale=3.0, size=3),
        )
        for _ in range(count)
    ]


def assert_sim3_close(actual: Sim3, expected: Sim3, atol: float) -> None:
    assert actual.scale == pytest.approx(expected.scale, rel=atol)
    np.testing.assert_allclose(actual.rotation, expected.rotation, atol=atol)
    np.testing.assert_allclose(
        actual.translation, expected.translation, atol=atol * 10
    )


def test_recovers_known_similarities_and_densifies_the_epoch(
    truth_frames: list[Frame],
) -> None:
    joint = Sim3(1.7, random_rotation(np.random.default_rng(99)), np.ones(3))
    scene = make_scene(truth_frames, [0, 11], list(range(1, 11)), joint)
    transforms = known_similarities(2)
    model = SimilarityReconstructor(scene.truth, transforms)
    config = DensifyConfig(chunk_size=5, anchors_per_chunk=2)

    dense = densify_epoch(
        model, scene.anchor_videos, scene.anchor_frames, scene.others, config
    )

    assert len(dense.chunks) == 2
    for chunk, sim in zip(dense.chunks, transforms, strict=True):
        assert_sim3_close(chunk.transform, joint.compose(sim.inverse()), 1e-4)
        assert chunk.rmse < 1e-4
        assert chunk.anchor_indices == [0, 11]
    assert [f.index for f in dense.frames] == list(range(12))
    for frame in dense.frames:
        truth = (
            scene.truth[frame.image.tobytes()]
            if frame.image is not None
            else None
        )
        assert truth is not None and truth.depth is not None
        expected = joint.transform_pose(truth.world_from_camera)
        np.testing.assert_allclose(
            frame.world_from_camera.matrix, expected.matrix, atol=1e-4
        )
        assert frame.depth is not None
        np.testing.assert_allclose(
            frame.depth, truth.depth * np.float32(joint.scale), rtol=1e-4
        )
        assert frame.epoch == truth.epoch
    fused = fuse_frames(dense.frames)
    expected_cloud = np.concatenate(
        [
            joint.apply(
                unproject_depth(t.depth, t.intrinsics, t.world_from_camera)
            )
            for t in sorted(scene.truth.values(), key=lambda f: f.index)
            if t.depth is not None
        ]
    )
    np.testing.assert_allclose(fused.points, expected_cloud, atol=1e-3)
    assert len(model.calls) == 2


def _recover(
    scene: Scene,
    model: SimilarityReconstructor,
    config: DensifyConfig,
) -> list[ChunkAlignment]:
    return densify_epoch(
        model, scene.anchor_videos, scene.anchor_frames, scene.others, config
    ).chunks


def _transform_error(actual: Sim3, expected: Sim3) -> float:
    return max(
        abs(actual.scale / expected.scale - 1.0),
        float(np.abs(actual.rotation - expected.rotation).max()),
        float(np.abs(actual.translation - expected.translation).max()),
    )


def _corrupt_anchor_depth(
    scene: Scene,
    factors: Sequence[float],
    fraction: float,
    invalid_value: float | None = None,
) -> None:
    """Corrupts joint pass depth in place at full confidence.

    A fraction of the pixels is multiplied by factors drawn from
    ``factors``; with ``invalid_value`` set, that fraction is replaced by
    the value instead.
    """
    rng = np.random.default_rng(1)
    pool = np.asarray(factors, np.float32)
    for frame in scene.anchor_frames:
        assert frame.depth is not None
        bad = rng.random(frame.depth.shape) < fraction
        if invalid_value is None:
            picks = rng.choice(pool, size=frame.depth.shape)
            frame.depth[bad] *= picks[bad]
        else:
            frame.depth[bad] = np.float32(invalid_value)


@pytest.mark.parametrize("side", ["chunk", "anchor"])
@pytest.mark.parametrize("fraction", [0.15, 0.3])
@pytest.mark.parametrize("factor", [3.0, 4.0, 10.0])
def test_gross_depth_errors_are_gated_out(
    truth_frames: list[Frame], factor: float, fraction: float, side: str
) -> None:
    scene = make_scene(truth_frames, [0, 11], [1, 2, 3])
    sim = known_similarities(1, seed=5)[0]
    if side == "anchor":
        _corrupt_anchor_depth(scene, (factor,), fraction)
        model = SimilarityReconstructor(scene.truth, [sim])
    else:
        model = SimilarityReconstructor(
            scene.truth,
            [sim],
            corrupt_views=2,
            corrupt_fraction=fraction,
            corrupt_factors=(factor,),
        )
    config = DensifyConfig(anchors_per_chunk=2, max_alignment_points=10**6)

    chunk = _recover(scene, model, config)[0]

    assert _transform_error(chunk.transform, sim.inverse()) < 5e-3


def _in_gate_model(scene: Scene, sim: Sim3) -> SimilarityReconstructor:
    return SimilarityReconstructor(
        scene.truth,
        [sim],
        corrupt_views=2,
        corrupt_fraction=0.15,
        corrupt_factors=(1.3, 0.7),
    )


@pytest.mark.parametrize(
    ("trim_iterations", "error_remains"), [(3, False), (0, True)]
)
def test_trimming_removes_depth_errors_inside_the_ratio_gate(
    truth_frames: list[Frame], trim_iterations: int, error_remains: bool
) -> None:
    scene = make_scene(truth_frames, [0, 11], [1, 2, 3])
    sim = known_similarities(1, seed=5)[0]
    config = DensifyConfig(
        anchors_per_chunk=2,
        trim_iterations=trim_iterations,
        max_alignment_points=10**6,
    )

    chunk = _recover(scene, _in_gate_model(scene, sim), config)[0]

    error = _transform_error(chunk.transform, sim.inverse())
    assert (error > 5e-3) == error_remains


class StretchedPoseReconstructor(SimilarityReconstructor):
    """Keeps the depth ratios but spreads the camera positions."""

    def reconstruct(
        self, images: Sequence[NDArray[np.uint8]]
    ) -> ReconstructionResult:
        result = super().reconstruct(images)
        return replace(
            result,
            world_from_camera=[
                SE3(pose.rotation, pose.translation * 20.0)
                for pose in result.world_from_camera
            ],
        )


def test_a_fit_that_contradicts_the_depth_ratios_is_rejected(
    truth_frames: list[Frame],
) -> None:
    scene = make_scene(truth_frames, [0, 11], [1, 2, 3])
    model = StretchedPoseReconstructor(scene.truth, known_similarities(1))
    config = DensifyConfig(anchors_per_chunk=2)

    with pytest.raises(ValueError, match=r"\[1, 2, 3\].*scale"):
        _recover(scene, model, config)


@pytest.mark.parametrize("value", [0.0, float("nan")])
@pytest.mark.parametrize("side", ["chunk", "anchor"])
def test_invalid_depth_on_one_side_keeps_the_pairs_matched(
    truth_frames: list[Frame], side: str, value: float
) -> None:
    scene = make_scene(truth_frames, [0, 11], [1, 2, 3])
    sim = known_similarities(1, seed=9)[0]
    if side == "anchor":
        _corrupt_anchor_depth(scene, (1.0,), 0.1, invalid_value=value)
        model = SimilarityReconstructor(scene.truth, [sim])
    else:
        model = SimilarityReconstructor(
            scene.truth,
            [sim],
            corrupt_views=2,
            invalid_fraction=0.1,
            invalid_value=value,
        )
    config = DensifyConfig(
        anchors_per_chunk=2, trim_iterations=0, max_alignment_points=10**6
    )

    chunk = _recover(scene, model, config)[0]

    assert_sim3_close(chunk.transform, sim.inverse(), 1e-4)


def test_low_confidence_pixels_are_excluded_from_alignment(
    truth_frames: list[Frame],
) -> None:
    scene = make_scene(truth_frames, [0, 11], [1, 2, 3])
    sim = known_similarities(1, seed=6)[0]
    model = SimilarityReconstructor(
        scene.truth,
        [sim],
        corrupt_views=2,
        corrupt_fraction=0.3,
        corrupt_confidence=0.05,
    )
    config = DensifyConfig(
        anchors_per_chunk=2, trim_iterations=0, max_alignment_points=10**6
    )

    chunk = _recover(scene, model, config)[0]

    assert_sim3_close(chunk.transform, sim.inverse(), 1e-4)


def test_low_joint_pass_confidence_is_excluded_from_alignment(
    truth_frames: list[Frame],
) -> None:
    scene = make_scene(truth_frames, [0, 11], [1, 2, 3])
    rng = np.random.default_rng(3)
    for frame in scene.anchor_frames:
        assert frame.depth is not None and frame.confidence is not None
        bad = rng.random(frame.depth.shape) < 0.3
        frame.depth[bad] *= np.float32(3.0)
        frame.confidence[bad] = 0.05
    sim = known_similarities(1, seed=7)[0]
    model = SimilarityReconstructor(scene.truth, [sim])
    config = DensifyConfig(
        anchors_per_chunk=2, trim_iterations=0, max_alignment_points=10**6
    )

    chunk = _recover(scene, model, config)[0]

    assert_sim3_close(chunk.transform, sim.inverse(), 1e-4)


def test_subsampling_is_capped_and_deterministic(
    truth_frames: list[Frame],
) -> None:
    scene = make_scene(truth_frames, [0, 11], [1, 2, 3])
    sim = known_similarities(1, seed=8)[0]
    config = DensifyConfig(
        anchors_per_chunk=2, max_alignment_points=500, min_alignment_points=100
    )

    first = _recover(scene, SimilarityReconstructor(scene.truth, [sim]), config)
    second = _recover(
        scene, SimilarityReconstructor(scene.truth, [sim]), config
    )

    assert first[0].point_count <= 500
    assert_sim3_close(first[0].transform, sim.inverse(), 1e-4)
    np.testing.assert_array_equal(
        first[0].transform.matrix, second[0].transform.matrix
    )


def _chunk_plan(
    truth_frames: list[Frame],
    anchor_indices: Sequence[int],
    other_indices: Sequence[int],
    config: DensifyConfig,
) -> tuple[Scene, SimilarityReconstructor, list[ChunkAlignment]]:
    scene = make_scene(truth_frames, anchor_indices, other_indices)
    model = SimilarityReconstructor(scene.truth, known_similarities(20))
    chunks = _recover(scene, model, config)
    return scene, model, chunks


def test_chunks_hold_at_most_chunk_size_new_frames_with_nearest_anchors(
    truth_frames: list[Frame],
) -> None:
    config = DensifyConfig(chunk_size=2, anchors_per_chunk=2)

    scene, model, chunks = _chunk_plan(
        truth_frames, [0, 10, 20], [18, 2, 12, 4, 16, 14], config
    )

    assert [c.frame_indices for c in chunks] == [[2, 4], [12, 14], [16, 18]]
    assert [c.anchor_indices for c in chunks] == [[0, 10], [10, 20], [10, 20]]
    anchors = {v.index: v.image.tobytes() for v in scene.anchor_videos}
    others = {v.index: v.image.tobytes() for v in scene.others}
    for call, chunk in zip(model.calls, chunks, strict=True):
        wanted = [anchors[i] for i in chunk.anchor_indices]
        wanted += [others[i] for i in chunk.frame_indices]
        assert call == wanted


@pytest.mark.parametrize(
    ("chunk_size", "anchors_per_chunk", "anchors", "others", "expected"),
    [
        pytest.param(2, 1, [10, 0], [4, 6], [0], id="equidistant-lower-index"),
        pytest.param(3, 1, [0, 5, 20], [4, 6, 7], [5], id="anchor-in-range"),
        pytest.param(
            2, 1, [9, 15, 24, 100], [10, 40], [15], id="whole-group-range"
        ),
        pytest.param(10, 4, [0, 9], [1, 2, 3], [0, 9], id="fewer-than-asked"),
    ],
)
def test_nearest_anchors_are_chosen_by_distance_to_the_group_range(
    truth_frames: list[Frame],
    chunk_size: int,
    anchors_per_chunk: int,
    anchors: list[int],
    others: list[int],
    expected: list[int],
) -> None:
    config = DensifyConfig(
        chunk_size=chunk_size, anchors_per_chunk=anchors_per_chunk
    )

    _, _, chunks = _chunk_plan(truth_frames, anchors, others, config)

    assert len(chunks) == 1
    assert chunks[0].anchor_indices == expected


def test_every_frame_appears_once_and_anchors_are_unchanged(
    truth_frames: list[Frame],
) -> None:
    scene = make_scene(truth_frames, [9, 0], [8, 3, 5, 1, 7])
    model = SimilarityReconstructor(scene.truth, known_similarities(5))

    dense = densify_epoch(
        model,
        scene.anchor_videos,
        scene.anchor_frames,
        scene.others,
        DensifyConfig(chunk_size=2, anchors_per_chunk=1),
    )

    assert [f.index for f in dense.frames] == [0, 1, 3, 5, 7, 8, 9]
    by_index = {f.index: f for f in dense.frames}
    for anchor in scene.anchor_frames:
        assert by_index[anchor.index] is anchor
    assert sorted(i for c in dense.chunks for i in c.frame_indices) == [
        1,
        3,
        5,
        7,
        8,
    ]
    assert all(len(c.frame_indices) <= 2 for c in dense.chunks)
    assert all(f.epoch == Epoch.BEFORE for f in dense.frames)


def test_no_other_frames_returns_the_anchors_without_reconstructing(
    truth_frames: list[Frame],
) -> None:
    scene = make_scene(truth_frames, [3, 1], [])
    model = SimilarityReconstructor(scene.truth, [])

    dense = densify_epoch(model, scene.anchor_videos, scene.anchor_frames, [])

    assert model.calls == []
    assert dense.chunks == []
    assert [f.index for f in dense.frames] == [1, 3]


def test_the_reconstructor_is_not_unloaded(truth_frames: list[Frame]) -> None:
    scene = make_scene(truth_frames, [0], [1, 2])
    fake = FakeReconstructor()

    densify_epoch(fake, scene.anchor_videos, scene.anchor_frames, scene.others)

    assert fake.unload_count == 0


def test_wrong_view_count_is_rejected(truth_frames: list[Frame]) -> None:
    scene = make_scene(truth_frames, [0], [1, 2])

    with pytest.raises(ValueError, match="views"):
        densify_epoch(
            ShortReconstructor(),
            scene.anchor_videos,
            scene.anchor_frames,
            scene.others,
        )


def test_anchor_depth_shape_mismatch_is_rejected(
    truth_frames: list[Frame],
) -> None:
    scene = make_scene(truth_frames, [0], [1, 2])
    half = FakeReconstructor(depth_size=(32, 24))

    with pytest.raises(ValueError, match="shape"):
        densify_epoch(
            half, scene.anchor_videos, scene.anchor_frames, scene.others
        )


def test_too_few_correspondences_are_rejected(
    truth_frames: list[Frame],
) -> None:
    scene = make_scene(truth_frames, [0], [1, 2])
    model = SimilarityReconstructor(scene.truth, known_similarities(1))
    config = DensifyConfig(
        min_alignment_points=10**6, max_alignment_points=10**7
    )

    with pytest.raises(ValueError, match=r"\[1, 2\].*correspondences"):
        _recover(scene, model, config)


def test_a_degenerate_chunk_error_names_the_chunk(
    truth_frames: list[Frame],
) -> None:
    scene = make_scene(truth_frames, [0], [1, 2])
    flat = np.full_like(truth_frames[0].depth, 2.0)  # type: ignore[arg-type]
    scene.truth = {
        key: replace(frame, depth=flat.copy())
        for key, frame in scene.truth.items()
    }
    column = np.zeros_like(flat)
    column[:, 0] = 1.0
    scene.anchor_frames = [
        replace(frame, depth=flat.copy(), confidence=column.copy())
        for frame in scene.anchor_frames
    ]
    model = SimilarityReconstructor(scene.truth, known_similarities(1))
    config = DensifyConfig(min_alignment_points=10)

    with pytest.raises(ValueError, match=r"\[1, 2\].*collinear"):
        _recover(scene, model, config)


def test_all_pixels_below_the_confidence_threshold_are_rejected(
    truth_frames: list[Frame],
) -> None:
    scene = make_scene(truth_frames, [0], [1, 2])
    for frame in scene.anchor_frames:
        assert frame.confidence is not None
        frame.confidence[:] = 0.05
    model = SimilarityReconstructor(scene.truth, known_similarities(1))

    with pytest.raises(ValueError, match="correspondences"):
        _recover(scene, model, DensifyConfig())


@pytest.mark.parametrize(
    "case",
    ["no_anchors", "length", "order", "epoch", "duplicate", "no_depth"],
)
def test_invalid_inputs_are_rejected(
    truth_frames: list[Frame], case: str
) -> None:
    scene = make_scene(truth_frames, [0, 5], [1, 2])
    videos, frames, others = (
        scene.anchor_videos,
        scene.anchor_frames,
        scene.others,
    )
    if case == "no_anchors":
        videos, frames = [], []
    elif case == "length":
        videos = videos[:1]
    elif case == "order":
        videos = videos[::-1]
    elif case == "epoch":
        frames = [frames[0], replace(frames[1], epoch=Epoch.AFTER)]
    elif case == "duplicate":
        others = [others[0], replace(others[1], index=others[0].index)]
    elif case == "no_depth":
        frames = [frames[0], replace(frames[1], depth=None)]
    model = SimilarityReconstructor(scene.truth, known_similarities(1))

    with pytest.raises(ValueError):
        densify_epoch(model, videos, frames, others)
    assert model.calls == []
