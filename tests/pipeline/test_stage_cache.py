import dataclasses
import shutil
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from appearance_scenes import reconstruction
from pipeline_scenes import PATCH, TEMPLATE, Timeline, inert_models, run_pair
from rotations import rotation_z

from spatial_change.cache import CacheKey, StageCache
from spatial_change.capture import VideoFrame
from spatial_change.capture.resize import resize_image
from spatial_change.change import (
    CrossAppearance,
    PointAppearance,
)
from spatial_change.domain import (
    SE3,
    ChangeReport,
    Epoch,
    EpochReconstruction,
    Frame,
    Intrinsics,
)
from spatial_change.perception import FakeFeatureEncoder
from spatial_change.pipeline import (
    Models,
    PipelineResult,
    RunConfig,
    stage_cache,
)
from spatial_change.pipeline import run as run_module
from spatial_change.pipeline.config import StageConfigs, build_stage_configs
from spatial_change.pipeline.stage_cache import (
    appearance_cached,
    appearance_from_entry,
    appearance_key,
    appearance_to_entry,
    joint_from_entry,
    joint_to_entry,
    reconstruct_cached,
    reconstruction_key,
)
from spatial_change.reconstruction.joint import JointReconstruction

DEPTH_W, DEPTH_H = 8, 6
IMAGE_W, IMAGE_H = 16, 12


def with_settings(role: str, **values: int | str) -> Models:
    return dataclasses.replace(inert_models(), settings={role: values})


def stages() -> StageConfigs:
    return build_stage_configs(RunConfig(device="cpu"), "cpu")


def video_frames(
    rng: np.random.Generator, count: int, first_index: int = 0
) -> list[VideoFrame]:
    return [
        VideoFrame(
            first_index + 3 * i,
            0.25 * (first_index + 3 * i),
            rng.integers(0, 256, (IMAGE_H, IMAGE_W, 3), dtype=np.uint8),
        )
        for i in range(count)
    ]


def frame_of(
    rng: np.random.Generator, epoch: Epoch, source: VideoFrame, k: int
) -> Frame:
    return Frame(
        epoch=epoch,
        index=source.index,
        timestamp_s=source.timestamp_s,
        intrinsics=Intrinsics(
            fx=8.5 + k,
            fy=8.25,
            cx=3.5,
            cy=2.5,
            width=DEPTH_W,
            height=DEPTH_H,
        ),
        world_from_camera=SE3(rotation_z(5.0 * (k + 1)), rng.normal(size=3)),
        image=resize_image(source.image, DEPTH_W, DEPTH_H),
        depth=rng.uniform(0.5, 4.0, (DEPTH_H, DEPTH_W)).astype(np.float32),
        confidence=rng.uniform(0.0, 1.0, (DEPTH_H, DEPTH_W)).astype(np.float32),
    )


def joint_of(
    rng: np.random.Generator,
    before: Sequence[VideoFrame],
    after: Sequence[VideoFrame],
) -> JointReconstruction:
    return JointReconstruction(
        before=[
            frame_of(rng, Epoch.BEFORE, v, k) for k, v in enumerate(before)
        ],
        after=[frame_of(rng, Epoch.AFTER, v, k) for k, v in enumerate(after)],
    )


def assert_frames_equal(got: list[Frame], expected: list[Frame]) -> None:
    assert len(got) == len(expected)
    for a, b in zip(got, expected, strict=True):
        assert a.epoch is b.epoch
        assert a.index == b.index
        assert a.timestamp_s == b.timestamp_s
        assert a.intrinsics == b.intrinsics
        np.testing.assert_array_equal(
            a.world_from_camera.rotation, b.world_from_camera.rotation
        )
        np.testing.assert_array_equal(
            a.world_from_camera.translation, b.world_from_camera.translation
        )
        assert a.depth is not None and b.depth is not None
        assert a.confidence is not None and b.confidence is not None
        assert a.depth.dtype == b.depth.dtype
        assert a.confidence.dtype == b.confidence.dtype
        assert a.world_from_camera.rotation.dtype == np.float64
        assert a.world_from_camera.translation.dtype == np.float64
        assert a.image is not None and a.image.dtype == np.uint8
        np.testing.assert_array_equal(a.depth, b.depth)
        np.testing.assert_array_equal(a.confidence, b.confidence)
        assert a.image is not None and b.image is not None
        np.testing.assert_array_equal(a.image, b.image)


def test_a_joint_reconstruction_survives_the_store_with_its_images(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(0)
    before, after = video_frames(rng, 3), video_frames(rng, 2, 100)
    joint = joint_of(rng, before, after)
    cache = StageCache(tmp_path)
    key = reconstruction_key(before, after, inert_models())

    stored = cache.put(key, *joint_to_entry(joint))
    restored = joint_from_entry(stored, before, after)

    assert_frames_equal(restored.before, joint.before)
    assert_frames_equal(restored.after, joint.after)


def test_the_stored_frames_leave_the_images_out() -> None:
    rng = np.random.default_rng(1)
    before, after = video_frames(rng, 2), video_frames(rng, 2, 50)
    joint = joint_of(rng, before, after)

    arrays, _ = joint_to_entry(joint)

    assert not any(a.dtype == np.uint8 for a in arrays.values())


def test_an_entry_of_other_frames_is_refused_on_load(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    before, after = video_frames(rng, 3), video_frames(rng, 2, 100)
    joint = joint_of(rng, before, after)
    stored = StageCache(tmp_path).put(
        reconstruction_key(before, after, inert_models()),
        *joint_to_entry(joint),
    )

    with pytest.raises(ValueError, match="frames"):
        joint_from_entry(stored, before[:2], after)


def appearance_of(rng: np.random.Generator) -> CrossAppearance:
    def side(n: int) -> PointAppearance:
        similarity = rng.uniform(-1.0, 1.0, n).astype(np.float32)
        similarity[::3] = np.nan
        return PointAppearance(
            similarity=similarity,
            own_views=rng.integers(0, 9, n).astype(np.int32),
            other_views=rng.integers(0, 9, n).astype(np.int32),
        )

    return CrossAppearance(before=side(11), after=side(7))


def epochs_of(
    rng: np.random.Generator, n_points: int = 20
) -> tuple[EpochReconstruction, EpochReconstruction]:
    joint = joint_of(rng, video_frames(rng, 2), video_frames(rng, 2, 100))
    return (
        reconstruction(
            Epoch.BEFORE, joint.before, rng.normal(size=(n_points, 3))
        ),
        reconstruction(
            Epoch.AFTER, joint.after, rng.normal(size=(n_points, 3))
        ),
    )


def test_a_cross_appearance_survives_the_store_exactly(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    appearance = appearance_of(rng)
    key = appearance_key(*epochs_of(rng), inert_models(), stages())

    restored = appearance_from_entry(
        StageCache(tmp_path).put(key, *appearance_to_entry(appearance))
    )

    for got, want in (
        (restored.before, appearance.before),
        (restored.after, appearance.after),
    ):
        assert got.similarity.dtype == np.float32
        assert got.own_views.dtype == np.int32
        assert got.other_views.dtype == np.int32
        np.testing.assert_array_equal(got.similarity, want.similarity)
        np.testing.assert_array_equal(got.own_views, want.own_views)
        np.testing.assert_array_equal(got.other_views, want.other_views)


def test_the_reconstruction_key_follows_its_inputs_and_settings() -> None:
    rng = np.random.default_rng(4)
    before, after = video_frames(rng, 3), video_frames(rng, 2, 100)
    base = reconstruction_key(before, after, inert_models())
    pixel = before[1].image.copy()
    pixel[0, 0, 0] ^= 1

    assert base == reconstruction_key(before, after, inert_models())
    changed = {
        "pixel": reconstruction_key(
            [before[0], dataclasses.replace(before[1], image=pixel), before[2]],
            after,
            inert_models(),
        ),
        "index": reconstruction_key(
            [dataclasses.replace(before[0], index=999), *before[1:]],
            after,
            inert_models(),
        ),
        "timestamp": reconstruction_key(
            before,
            [dataclasses.replace(after[0], timestamp_s=77.0), after[1]],
            inert_models(),
        ),
        "split": reconstruction_key(
            before[:2], [before[2], *after], inert_models()
        ),
        "model": reconstruction_key(
            before, after, inert_models(reconstruction="other")
        ),
        "settings": reconstruction_key(
            before, after, with_settings("reconstruction", pixel_limit=7)
        ),
    }
    assert base not in changed.values()
    assert len(set(changed.values())) == len(changed)


def test_the_reconstruction_key_ignores_the_encoder() -> None:
    rng = np.random.default_rng(8)
    before, after = video_frames(rng, 2), video_frames(rng, 2, 100)
    base = reconstruction_key(before, after, inert_models())

    assert base == reconstruction_key(
        before, after, with_settings("features", model_id="x")
    )
    assert base == reconstruction_key(
        before, after, inert_models(features="other")
    )


@pytest.mark.parametrize("version", ["STAGE_VERSION", "__version__"])
def test_both_keys_follow_the_stage_and_package_versions(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    rng = np.random.default_rng(9)
    video = (video_frames(rng, 2), video_frames(rng, 2, 100))
    epochs = epochs_of(rng)
    base = (
        reconstruction_key(*video, inert_models()),
        appearance_key(*epochs, inert_models(), stages()),
    )

    monkeypatch.setattr(stage_cache, version, "bumped")

    assert reconstruction_key(*video, inert_models()) != base[0]
    assert appearance_key(*epochs, inert_models(), stages()) != base[1]


def perturbed(
    epoch: EpochReconstruction, frame: int | None = None, **changes: object
) -> EpochReconstruction:
    """Copies an epoch with some of its fields or one frame's fields changed."""
    if frame is not None:
        frames = list(epoch.frames)
        frames[frame] = dataclasses.replace(frames[frame], **changes)
        return dataclasses.replace(epoch, frames=frames)
    return dataclasses.replace(epoch, **changes)


def test_the_appearance_key_follows_what_the_comparison_reads() -> None:
    rng = np.random.default_rng(6)
    before, after = epochs_of(rng)
    base = appearance_key(before, after, inert_models(), stages())
    points = before.points.copy()
    points[3, 1] += 1e-9
    frame = after.frames[1]
    assert frame.depth is not None and frame.confidence is not None
    assert frame.image is not None
    depth, confidence, image = (
        frame.depth.copy(),
        frame.confidence.copy(),
        frame.image.copy(),
    )
    depth[0, 0] += np.float32(1e-3)
    confidence[0, 0] = 0.0
    image[0, 0, 0] ^= 1
    pose = SE3(frame.world_from_camera.rotation, np.array([0.0, 0.0, 9.0]))
    turn = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    turned = SE3(turn, frame.world_from_camera.translation)
    intrinsics = dataclasses.replace(
        frame.intrinsics, fx=frame.intrinsics.fx + 1
    )

    changed = {
        "point": appearance_key(
            perturbed(before, points=points), after, inert_models(), stages()
        ),
        "depth": appearance_key(
            before, perturbed(after, 1, depth=depth), inert_models(), stages()
        ),
        "confidence": appearance_key(
            before,
            perturbed(after, 1, confidence=confidence),
            inert_models(),
            stages(),
        ),
        "image": appearance_key(
            before, perturbed(after, 1, image=image), inert_models(), stages()
        ),
        "pose": appearance_key(
            before,
            perturbed(after, 1, world_from_camera=pose),
            inert_models(),
            stages(),
        ),
        "rotation": appearance_key(
            before,
            perturbed(after, 1, world_from_camera=turned),
            inert_models(),
            stages(),
        ),
        "intrinsics": appearance_key(
            before,
            perturbed(after, 1, intrinsics=intrinsics),
            inert_models(),
            stages(),
        ),
        "frame_count": appearance_key(
            before,
            perturbed(after, frames=after.frames[:1]),
            inert_models(),
            stages(),
        ),
        "swapped": appearance_key(after, before, inert_models(), stages()),
    }

    assert base == appearance_key(before, after, inert_models(), stages())
    assert base not in changed.values()
    assert len(set(changed.values())) == len(changed)


def test_the_appearance_key_follows_the_encoder_and_its_settings() -> None:
    before, after = epochs_of(np.random.default_rng(7))
    config = stages()

    def with_appearance(**changes: float) -> StageConfigs:
        return dataclasses.replace(
            config, appearance=dataclasses.replace(config.appearance, **changes)
        )

    base = appearance_key(before, after, inert_models(), config)
    changed = [
        appearance_key(before, after, inert_models(features="other"), config),
        appearance_key(
            before,
            after,
            with_settings("features", fallback_model_id="x"),
            config,
        ),
        appearance_key(
            before, after, inert_models(), with_appearance(min_views=3)
        ),
        appearance_key(
            before, after, inert_models(), with_appearance(voxel_size=0.1)
        ),
    ]

    assert base not in changed
    assert len(set(changed)) == len(changed)


def test_the_appearance_key_ignores_what_is_applied_later() -> None:
    before, after = epochs_of(np.random.default_rng(8))
    config = stages()
    base = appearance_key(before, after, inert_models(), config)

    swept = dataclasses.replace(
        config,
        appearance=dataclasses.replace(config.appearance, changed_below=0.9),
        candidates=dataclasses.replace(config.candidates, min_points=99),
    )
    realigned = dataclasses.replace(
        config,
        fusion_voxel_size=0.5,
        min_confidence=0.9,
        floor=dataclasses.replace(config.floor, seed=42),
    )

    assert base == appearance_key(before, after, inert_models(), swept)
    assert base == appearance_key(before, after, inert_models(), realigned)


class FlakyCache(StageCache):
    """A cache whose reads or writes can be made to fail."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.fail_get = False
        self.fail_put = False

    def get(self, key: CacheKey) -> Any:
        if self.fail_get:
            raise PermissionError("locked")
        return super().get(key)

    def put(self, key: CacheKey, arrays: Any, meta: Any) -> Any:
        if self.fail_put:
            raise OSError("disk full")
        return super().put(key, arrays, meta)


class Counted:
    """A stage computation that counts its calls."""

    def __init__(self, value: Any) -> None:
        self.value = value
        self.calls = 0

    def __call__(self) -> Any:
        self.calls += 1
        return self.value


def reconstruction_setup(
    seed: int = 20,
) -> tuple[list[VideoFrame], list[VideoFrame], JointReconstruction]:
    rng = np.random.default_rng(seed)
    before, after = video_frames(rng, 3), video_frames(rng, 2, 100)
    return before, after, joint_of(rng, before, after)


def test_a_miss_computes_once_and_returns_the_computed_value(
    tmp_path: Path,
) -> None:
    before, after, joint = reconstruction_setup()
    compute = Counted(joint)
    cache = StageCache(tmp_path)

    first = reconstruct_cached(cache, inert_models(), before, after, compute)
    second = reconstruct_cached(cache, inert_models(), before, after, compute)

    assert compute.calls == 1
    assert first.value is joint
    assert not first.hit
    assert second.hit
    assert_frames_equal(second.value.before, joint.before)
    assert_frames_equal(second.value.after, joint.after)


def appearance_setup(
    seed: int = 21,
) -> tuple[EpochReconstruction, EpochReconstruction, CrossAppearance]:
    rng = np.random.default_rng(seed)
    before, after = epochs_of(rng)
    return before, after, appearance_of(rng)


def test_without_a_cache_the_stage_just_computes_and_builds_no_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*args: object) -> None:
        raise AssertionError("a key was built")

    monkeypatch.setattr(stage_cache, "reconstruction_key", refuse)
    monkeypatch.setattr(stage_cache, "appearance_key", refuse)
    before, after, joint = reconstruction_setup()
    reconstruct = Counted(joint)
    epoch_before, epoch_after, appearance = appearance_setup()
    compare = Counted((appearance, "dino"))

    reconstructed = reconstruct_cached(
        None, inert_models(), before, after, reconstruct
    )
    compared = appearance_cached(
        None, inert_models(), stages(), epoch_before, epoch_after, compare
    )

    assert (reconstructed.value, reconstructed.hit) == (joint, False)
    assert (compared.value, compared.hit) == (appearance, False)
    assert (reconstruct.calls, compare.calls) == (1, 1)


def test_a_failing_store_write_warns_and_the_stage_goes_on(
    tmp_path: Path,
) -> None:
    before, after, joint = reconstruction_setup()
    cache = FlakyCache(tmp_path)
    cache.fail_put = True
    compute = Counted(joint)

    with pytest.warns(RuntimeWarning, match="disk full"):
        result = reconstruct_cached(
            cache, inert_models(), before, after, compute
        )

    assert result.value is joint
    assert compute.calls == 1


def test_a_value_that_cannot_be_stored_warns_and_the_stage_goes_on(
    tmp_path: Path,
) -> None:
    before, after, joint = reconstruction_setup()
    bare = dataclasses.replace(joint.before[0], depth=None, confidence=None)
    joint = dataclasses.replace(joint, before=[bare, *joint.before[1:]])
    cache = StageCache(tmp_path)
    compute = Counted(joint)

    with pytest.warns(RuntimeWarning, match="depth and confidence"):
        result = reconstruct_cached(
            cache, inert_models(), before, after, compute
        )

    assert result.value is joint
    assert compute.calls == 1
    assert not any(tmp_path.rglob("*.npz"))


def test_a_failing_store_read_warns_and_the_stage_computes(
    tmp_path: Path,
) -> None:
    before, after, joint = reconstruction_setup()
    cache = FlakyCache(tmp_path)
    cache.fail_get = True
    compute = Counted(joint)

    with pytest.warns(RuntimeWarning, match="locked"):
        result = reconstruct_cached(
            cache, inert_models(), before, after, compute
        )

    assert result.value is joint
    assert not result.hit
    assert compute.calls == 1


def test_a_loadable_but_invalid_entry_is_replaced_by_a_fresh_one(
    tmp_path: Path,
) -> None:
    before, after, joint = reconstruction_setup()
    cache = StageCache(tmp_path)
    key = reconstruction_key(before, after, inert_models())
    other = joint_of(np.random.default_rng(1), before[:2], after)
    cache.put(key, *joint_to_entry(other))
    compute = Counted(joint)

    with pytest.warns(RuntimeWarning, match="reconstruction"):
        first = reconstruct_cached(
            cache, inert_models(), before, after, compute
        )
    second = reconstruct_cached(cache, inert_models(), before, after, compute)

    assert first.value is joint
    assert not first.hit
    assert second.hit
    assert compute.calls == 1
    assert_frames_equal(second.value.before, joint.before)


def test_an_entry_with_a_missing_array_is_replaced(tmp_path: Path) -> None:
    before, after, joint = reconstruction_setup()
    cache = StageCache(tmp_path)
    key = reconstruction_key(before, after, inert_models())
    arrays, meta = joint_to_entry(joint)
    del arrays["after_depth"]
    cache.put(key, arrays, meta)
    compute = Counted(joint)

    with pytest.warns(RuntimeWarning, match="reconstruction"):
        result = reconstruct_cached(
            cache, inert_models(), before, after, compute
        )

    assert result.value is joint
    assert compute.calls == 1


def test_an_appearance_miss_computes_once_then_hits(tmp_path: Path) -> None:
    before, after, appearance = appearance_setup()
    compute = Counted((appearance, "dino"))
    cache = StageCache(tmp_path)

    first = appearance_cached(
        cache, inert_models(), stages(), before, after, compute
    )
    second = appearance_cached(
        cache, inert_models(), stages(), before, after, compute
    )

    assert compute.calls == 1
    assert first.value is appearance
    assert not first.hit
    assert second.hit
    np.testing.assert_array_equal(
        second.value.after.similarity, appearance.after.similarity
    )


def test_an_appearance_of_a_fallback_encoder_is_not_stored(
    tmp_path: Path,
) -> None:
    before, after, appearance = appearance_setup()
    cache = StageCache(tmp_path)
    compute = Counted((appearance, "fallback"))

    with pytest.warns(RuntimeWarning, match="fallback"):
        first = appearance_cached(
            cache, inert_models(), stages(), before, after, compute
        )
    with pytest.warns(RuntimeWarning, match="fallback"):
        second = appearance_cached(
            cache, inert_models(), stages(), before, after, compute
        )

    assert first.value is appearance
    assert not second.hit
    assert compute.calls == 2
    assert not list((tmp_path / "appearance").glob("*/*"))


def test_an_encoder_that_never_loaded_gives_the_value_silently(
    tmp_path: Path,
) -> None:
    before, after, appearance = appearance_setup()
    compute = Counted((appearance, None))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = appearance_cached(
            StageCache(tmp_path),
            inert_models(),
            stages(),
            before,
            after,
            compute,
        )

    assert result.value is appearance
    assert not list((tmp_path / "appearance").glob("*/*"))


def test_an_appearance_entry_of_the_wrong_shape_is_replaced(
    tmp_path: Path,
) -> None:
    before, after, appearance = appearance_setup()
    cache = StageCache(tmp_path)
    key = appearance_key(before, after, inert_models(), stages())
    arrays, meta = appearance_to_entry(appearance)
    arrays["after_own_views"] = arrays["after_own_views"][:-1]
    cache.put(key, arrays, meta)
    compute = Counted((appearance, "dino"))

    with pytest.warns(RuntimeWarning, match="appearance"):
        result = appearance_cached(
            cache, inert_models(), stages(), before, after, compute
        )

    assert result.value is appearance
    assert compute.calls == 1


STAGE_DIRS = ("reconstruction", "appearance")
WARM_ROLES = ["segmenter", "encoder", "labeler"]
COLD_ROLES = ["reconstructor", "encoder", "segmenter", "encoder", "labeler"]


def comparable(result: PipelineResult) -> ChangeReport:
    """Gives the report without what differs between two runs of one input."""
    volatile = ("cache_hits", "peak_vram_mb")
    metadata = {
        name: value
        for name, value in result.report.metadata.items()
        if not name.startswith("seconds_") and name not in volatile
    }
    return result.report.model_copy(update={"metadata": metadata})


def entries(cache_dir: Path, stage: str) -> list[Path]:
    return sorted((cache_dir / stage).glob("*/*"))


@dataclasses.dataclass(frozen=True, eq=False)
class Warm:
    """A cold run, then the same run again on the same cache."""

    cache_dir: Path
    cold: PipelineResult
    cold_models: Timeline
    again: PipelineResult
    again_models: Timeline


@pytest.fixture(scope="module")
def warm(tmp_path_factory: pytest.TempPathFactory) -> Warm:
    root = tmp_path_factory.mktemp("warm")
    cold, cold_models = run_pair(root / "cold", cache_dir=root / "cache")
    again, again_models = run_pair(root / "again", cache_dir=root / "cache")
    return Warm(root / "cache", cold, cold_models, again, again_models)


@pytest.fixture
def warm_cache(warm: Warm, tmp_path: Path) -> Path:
    """A copy of the warm cache that a test may add to or damage."""
    return shutil.copytree(warm.cache_dir, tmp_path / "cache")


@pytest.fixture(scope="module")
def uncached(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[PipelineResult, Timeline, Path]:
    root = tmp_path_factory.mktemp("uncached")
    result, timeline = run_pair(root / "out")
    return result, timeline, root


def test_the_first_run_computes_and_stores_both_stages(warm: Warm) -> None:
    assert warm.cold_models.roles() == COLD_ROLES
    assert warm.cold.report.metadata["cache_hits"] == ""
    assert [len(entries(warm.cache_dir, s)) for s in STAGE_DIRS] == [1, 1]


def test_the_second_run_skips_the_reconstructor_and_the_appearance_pass(
    warm: Warm,
) -> None:
    assert warm.again_models.roles() == WARM_ROLES
    assert (
        warm.again.report.metadata["cache_hits"] == "reconstruction,appearance"
    )
    assert set(warm.again.seconds) == set(warm.cold.seconds)


def test_the_second_run_writes_the_same_changes(warm: Warm) -> None:
    assert comparable(warm.again) == comparable(warm.cold)
    got, want = warm.again.room_from_world, warm.cold.room_from_world
    assert got.scale == want.scale
    np.testing.assert_array_equal(got.rotation, want.rotation)
    np.testing.assert_array_equal(got.translation, want.translation)


def sweep(monkeypatch: pytest.MonkeyPatch, **appearance: float) -> None:
    original = run_module.build_stage_configs

    def build(config: RunConfig, device: str) -> StageConfigs:
        stages_ = original(config, device)
        return dataclasses.replace(
            stages_,
            appearance=dataclasses.replace(stages_.appearance, **appearance),
            candidates=dataclasses.replace(
                stages_.candidates, min_points=stages_.candidates.min_points + 1
            ),
        )

    monkeypatch.setattr(run_module, "build_stage_configs", build)


def test_a_later_threshold_reuses_both_entries(
    warm_cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sweep(monkeypatch, changed_below=0.2)

    result, timeline = run_pair(tmp_path / "out", cache_dir=warm_cache)

    assert result.report.metadata["cache_hits"] == "reconstruction,appearance"
    assert timeline.roles() == WARM_ROLES
    assert [len(entries(warm_cache, s)) for s in STAGE_DIRS] == [1, 1]


def test_an_alignment_setting_reuses_the_reconstruction_only(
    warm_cache: Path, tmp_path: Path
) -> None:
    result, timeline = run_pair(
        tmp_path / "out",
        TEMPLATE.model_copy(update={"voxel_size": 0.05}),
        cache_dir=warm_cache,
    )

    assert result.report.metadata["cache_hits"] == "reconstruction"
    assert timeline.roles() == ["encoder", *WARM_ROLES]
    assert [len(entries(warm_cache, s)) for s in STAGE_DIRS] == [1, 2]


def test_a_changed_input_frame_recomputes_the_reconstruction(
    warm_cache: Path, tmp_path: Path
) -> None:
    def mutate(frames: list[VideoFrame]) -> list[VideoFrame]:
        first = dataclasses.replace(
            frames[0], timestamp_s=frames[0].timestamp_s + 0.5
        )
        return [first, *frames[1:]]

    result, timeline = run_pair(
        tmp_path / "out", cache_dir=warm_cache, mutate=mutate
    )

    # The oracle answers the same geometry, so the appearance, which is keyed
    # on the aligned frames and points, still hits.
    assert result.report.metadata["cache_hits"] == "appearance"
    assert timeline.roles() == ["reconstructor", *WARM_ROLES]
    assert [len(entries(warm_cache, s)) for s in STAGE_DIRS] == [2, 1]


def test_without_a_cache_directory_nothing_is_stored(
    uncached: tuple[PipelineResult, Timeline, Path], warm: Warm
) -> None:
    result, timeline, root = uncached

    assert timeline.roles() == COLD_ROLES
    assert result.report.metadata["cache_hits"] == ""
    assert [p.name for p in root.iterdir()] == ["out"]
    assert comparable(result) == comparable(warm.cold)


class FallbackEncoder(FakeFeatureEncoder):
    @property
    def loaded_model_id(self) -> str:
        return "fallback-weights"


def test_an_encoder_that_fell_back_leaves_no_appearance_entry(
    tmp_path: Path,
) -> None:
    def adapt(models_: Models) -> Models:
        return dataclasses.replace(
            models_, encoder=lambda: FallbackEncoder(PATCH)
        )

    cache_dir = tmp_path / "cache"
    with pytest.warns(RuntimeWarning, match="fallback-weights"):
        result, _ = run_pair(tmp_path / "out", cache_dir=cache_dir, adapt=adapt)

    assert result.report.metadata["cache_hits"] == ""
    assert [len(entries(cache_dir, s)) for s in STAGE_DIRS] == [1, 0]
