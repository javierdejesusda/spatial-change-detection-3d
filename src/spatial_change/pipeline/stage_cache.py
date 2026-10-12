"""The two stages of a run whose results are kept between runs.

A batch over many pairs, or a sweep of cheap thresholds over one pair, must
not redo the expensive GPU work, so two stages are cached:

* ``reconstruction``, the joint reconstruction of both epochs, which is the
  longest stage by far. Its key holds a fingerprint of every input frame
  and the reconstructor's id and settings. The images are not stored: the
  decode is cheap, and a hit resizes the images of the input frames again.
* ``appearance``, the cross-epoch feature similarity of the points, which
  is the other long encoder pass. Its key holds a fingerprint of the
  aligned reconstructions the comparison reads (the points, and per frame
  the intrinsics, pose, image, depth, and confidence), the encoder's id and
  settings, and the appearance settings except ``changed_below``, which
  ``fuse_appearance`` applies afterwards, so a sweep over it still hits. It
  does not chain on the reconstruction entry: a recomputed reconstruction
  can differ from the stored one, and the key must follow the points.

Every other stage is recomputed. A key holds a version string, which is
``STAGE_VERSION`` plus the package ``__version__`` so a new release also
starts a fresh cache, and nothing that belongs to the machine, such as a
path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from spatial_change import __version__
from spatial_change.cache import (
    CachedArtifact,
    CacheKey,
    JsonValue,
    StageCache,
    fingerprint_array,
)
from spatial_change.change import CrossAppearance, PointAppearance
from spatial_change.domain import (
    SE3,
    Epoch,
    EpochReconstruction,
    Frame,
    Intrinsics,
)
from spatial_change.ports import ReconstructionResult

if TYPE_CHECKING:
    from spatial_change.capture import VideoFrame
    from spatial_change.pipeline.config import StageConfigs
    from spatial_change.pipeline.models import Models
    from spatial_change.reconstruction.joint import JointReconstruction

STAGE_VERSION = "2"
"""Bump when a serializer here or a cached stage changes its output."""
RECONSTRUCTION = "reconstruction"
APPEARANCE = "appearance"

Entry = tuple[dict[str, NDArray[Any]], dict[str, JsonValue]]


@dataclass(frozen=True, eq=False)
class StageResult[T]:
    """What a cached stage gives back.

    Attributes:
        value: The stage output.
        hit: Whether it was read from the cache.
    """

    value: T
    hit: bool


def _version() -> str:
    """Gives the version string of every key."""
    return f"{STAGE_VERSION}+{__version__}"


def _fingerprint_frames(frames: Sequence[VideoFrame]) -> str:
    """Hashes the images, indices, and timestamps of frames in order."""
    digest = hashlib.sha256(str(len(frames)).encode("ascii"))
    for frame in frames:
        digest.update(fingerprint_array(frame.image).encode("ascii"))
        digest.update(f"{frame.index}:{frame.timestamp_s!r}".encode("ascii"))
    return digest.hexdigest()


def reconstruction_key(
    before: Sequence[VideoFrame],
    after: Sequence[VideoFrame],
    models: Models,
) -> CacheKey:
    """Builds the key of the joint reconstruction.

    Args:
        before: BEFORE keyframes, as given to the reconstructor.
        after: AFTER keyframes.
        models: Supplies the reconstructor's weights id and settings, which
            include its device.

    Returns:
        The key.
    """
    return CacheKey.build(
        RECONSTRUCTION,
        _version(),
        {
            "model": models.ids["reconstruction"],
            "settings": dict(models.settings.get("reconstruction", {})),
        },
        {
            "before": _fingerprint_frames(before),
            "after": _fingerprint_frames(after),
        },
    )


def _fingerprint_epoch(epoch: EpochReconstruction) -> str:
    """Hashes what ``compare_appearance`` reads from an epoch."""
    digest = hashlib.sha256(epoch.epoch.value.encode("ascii"))
    digest.update(fingerprint_array(epoch.points).encode("ascii"))
    digest.update(str(len(epoch.frames)).encode("ascii"))
    for frame in epoch.frames:
        digest.update(repr(dataclasses.astuple(frame.intrinsics)).encode())
        for array in (
            frame.world_from_camera.matrix,
            frame.image,
            frame.depth,
            frame.confidence,
        ):
            digest.update(
                b"-"
                if array is None
                else fingerprint_array(array).encode("ascii")
            )
    return digest.hexdigest()


def appearance_key(
    before: EpochReconstruction,
    after: EpochReconstruction,
    models: Models,
    stages: StageConfigs,
) -> CacheKey:
    """Builds the key of the cross-epoch appearance.

    The key holds a fingerprint of the aligned reconstructions the
    comparison reads, not of how they were made, so it follows whatever
    changes the points or the frames and nothing else.

    Args:
        before: The aligned BEFORE reconstruction.
        after: The aligned AFTER reconstruction.
        models: Supplies the feature encoder's weights id and settings,
            which include its device.
        stages: Supplies the appearance settings except ``changed_below``.

    Returns:
        The key.
    """
    appearance = dataclasses.asdict(stages.appearance)
    del appearance["changed_below"]
    return CacheKey.build(
        APPEARANCE,
        _version(),
        {
            "model": models.ids["features"],
            "settings": dict(models.settings.get("features", {})),
            "appearance": appearance,
        },
        {
            "before": _fingerprint_epoch(before),
            "after": _fingerprint_epoch(after),
        },
    )


def _epoch_arrays(
    name: str, frames: Sequence[Frame]
) -> dict[str, NDArray[Any]]:
    """Stacks what a reconstruction gives per frame, under ``name``."""
    depths = []
    confidences = []
    for frame in frames:
        if frame.depth is None or frame.confidence is None:
            raise ValueError("a reconstructed frame needs depth and confidence")
        depths.append(frame.depth)
        confidences.append(frame.confidence)
    return {
        f"{name}_intrinsics": np.array(
            [dataclasses.astuple(f.intrinsics) for f in frames], np.float64
        ),
        f"{name}_pose": np.stack([f.world_from_camera.matrix for f in frames]),
        f"{name}_depth": np.stack(depths),
        f"{name}_confidence": np.stack(confidences),
    }


def joint_to_entry(joint: JointReconstruction) -> Entry:
    """Turns a joint reconstruction into the arrays and metadata to store.

    Only what the reconstructor gave is kept. The images, indices, and
    timestamps come from the input frames, so ``joint_from_entry`` takes
    them from there.

    Args:
        joint: The reconstruction, every frame with depth and confidence.

    Returns:
        The arrays and an empty metadata.

    Raises:
        ValueError: If a frame has no depth or no confidence, or the frames
            of an epoch differ in size.
    """
    arrays = {
        **_epoch_arrays("before", joint.before),
        **_epoch_arrays("after", joint.after),
    }
    return arrays, {}


def _result_from_entry(
    name: str, artifact: CachedArtifact
) -> ReconstructionResult:
    """Rebuilds what the reconstructor gave for the frames of an epoch."""

    def intrinsics(row: NDArray[np.float64]) -> Intrinsics:
        fx, fy, cx, cy, width, height = row.tolist()
        return Intrinsics(fx, fy, cx, cy, int(width), int(height))

    arrays = artifact.arrays
    return ReconstructionResult(
        intrinsics=[intrinsics(row) for row in arrays[f"{name}_intrinsics"]],
        world_from_camera=[
            SE3.from_matrix(matrix) for matrix in arrays[f"{name}_pose"]
        ],
        depth=list(arrays[f"{name}_depth"]),
        confidence=list(arrays[f"{name}_confidence"]),
    )


def joint_from_entry(
    artifact: CachedArtifact,
    before: Sequence[VideoFrame],
    after: Sequence[VideoFrame],
) -> JointReconstruction:
    """Rebuilds a joint reconstruction from a stored entry.

    Args:
        artifact: The entry written by ``joint_to_entry``.
        before: The BEFORE keyframes the entry was made from.
        after: The AFTER keyframes.

    Returns:
        The reconstruction, built as ``reconstruct_joint`` builds it, so each
        frame has its image resized to its depth map.

    Raises:
        ValueError: If the entry holds another number of frames than given.
    """
    # Loading this module imports cv2 through the capture package, which
    # importing the pipeline must not do.
    from spatial_change.reconstruction.joint import (
        JointReconstruction,
        frame_from_result,
    )

    def frames(
        name: str, epoch: Epoch, sources: Sequence[VideoFrame]
    ) -> list[Frame]:
        result = _result_from_entry(name, artifact)
        if len(result.depth) != len(sources):
            raise ValueError(
                f"the cached {name} frames do not match the {len(sources)} "
                "input frames"
            )
        return [
            frame_from_result(epoch, source, result, i)
            for i, source in enumerate(sources)
        ]

    return JointReconstruction(
        before=frames("before", Epoch.BEFORE, before),
        after=frames("after", Epoch.AFTER, after),
    )


def appearance_to_entry(appearance: CrossAppearance) -> Entry:
    """Turns a cross appearance into the arrays and metadata to store.

    Args:
        appearance: The appearance of both epochs' points.

    Returns:
        The arrays and an empty metadata.
    """
    arrays: dict[str, NDArray[Any]] = {}
    for name, side in (
        ("before", appearance.before),
        ("after", appearance.after),
    ):
        arrays[f"{name}_similarity"] = side.similarity
        arrays[f"{name}_own_views"] = side.own_views
        arrays[f"{name}_other_views"] = side.other_views
    return arrays, {}


def appearance_from_entry(artifact: CachedArtifact) -> CrossAppearance:
    """Rebuilds a cross appearance from a stored entry.

    Args:
        artifact: The entry written by ``appearance_to_entry``.

    Returns:
        The appearance.
    """

    def side(name: str) -> PointAppearance:
        return PointAppearance(
            similarity=artifact.arrays[f"{name}_similarity"],
            own_views=artifact.arrays[f"{name}_own_views"],
            other_views=artifact.arrays[f"{name}_other_views"],
        )

    appearance = CrossAppearance(before=side("before"), after=side("after"))
    for part in (appearance.before, appearance.after):
        if not (
            part.similarity.dtype == np.float32
            and part.own_views.dtype == np.int32
            and part.other_views.dtype == np.int32
            and part.similarity.shape
            == part.own_views.shape
            == part.other_views.shape
        ):
            raise ValueError("the cached appearance arrays do not agree")
    return appearance


def _warn(action: str, stage: str, error: Exception) -> None:
    warnings.warn(
        f"stage cache: could not {action} the {stage} entry ({error}); "
        "continuing without it",
        RuntimeWarning,
        stacklevel=3,
    )


def _cached[T](
    cache: StageCache,
    key: CacheKey,
    compute: Callable[[], tuple[T, bool]],
    encode: Callable[[T], Entry],
    decode: Callable[[CachedArtifact], T],
) -> StageResult[T]:
    """Runs a stage through the cache without letting the cache fail it.

    A problem of the store itself, such as a locked file or a full disk,
    or a value that cannot be encoded, warns and the stage goes on
    uncached. An entry that loads but does not
    decode is removed and recomputed. The stage runs at most once, and on a
    miss the computed value is returned as it is.

    Args:
        cache: The cache.
        key: The entry of the stage.
        compute: Runs the stage and says whether its value may be stored.
        encode: Turns a value into the arrays and metadata to store.
        decode: Turns a stored entry back into a value.

    Returns:
        The value and whether it came from the cache.
    """
    stage = key.stage
    try:
        artifact = cache.get(key)
    except OSError as error:
        _warn("read", stage, error)
        artifact = None
    if artifact is not None:
        try:
            return StageResult(decode(artifact), True)
        except (KeyError, TypeError, ValueError, IndexError) as error:
            _warn("decode", stage, error)
            cache.discard(key)
    value, storable = compute()
    if storable:
        try:
            cache.put(key, *encode(value))
        except (OSError, ValueError) as error:
            _warn("write", stage, error)
    return StageResult(value, False)


def reconstruct_cached(
    cache: StageCache | None,
    models: Models,
    before: Sequence[VideoFrame],
    after: Sequence[VideoFrame],
    compute: Callable[[], JointReconstruction],
) -> StageResult[JointReconstruction]:
    """Gives the joint reconstruction from the cache or from ``compute``.

    Args:
        cache: The cache, or ``None`` to always compute.
        models: Supplies the reconstructor id and settings for the key.
        before: BEFORE keyframes.
        after: AFTER keyframes.
        compute: Runs the reconstructor; not called on a hit.

    Returns:
        The reconstruction and whether it was a hit.
    """
    if cache is None:
        return StageResult(compute(), False)
    return _cached(
        cache,
        reconstruction_key(before, after, models),
        lambda: (compute(), True),
        joint_to_entry,
        lambda artifact: joint_from_entry(artifact, before, after),
    )


def appearance_cached(
    cache: StageCache | None,
    models: Models,
    stages: StageConfigs,
    before: EpochReconstruction,
    after: EpochReconstruction,
    compute: Callable[[], tuple[CrossAppearance, str | None]],
) -> StageResult[CrossAppearance]:
    """Gives the cross appearance from the cache or from ``compute``.

    The result is stored only when the encoder loaded the weights the key
    names. An encoder that fell back to other weights gives similarities
    the thresholds were not calibrated for, and a later run with the
    primary weights must not be served them.

    Args:
        cache: The cache, or ``None`` to always compute.
        models: Supplies the encoder id and settings for the key.
        stages: Supplies the appearance settings.
        before: The aligned BEFORE reconstruction.
        after: The aligned AFTER reconstruction.
        compute: Runs the encoder; returns the appearance and the id of the
            weights that loaded. Not called on a hit.

    Returns:
        The appearance and whether it was a hit.
    """
    if cache is None:
        return StageResult(compute()[0], False)
    requested = models.ids["features"]
    loaded_id: str | None = requested

    def run() -> tuple[CrossAppearance, bool]:
        nonlocal loaded_id
        value, loaded_id = compute()
        return value, loaded_id == requested

    result = _cached(
        cache,
        appearance_key(before, after, models, stages),
        run,
        appearance_to_entry,
        appearance_from_entry,
    )
    if loaded_id not in (None, requested):
        warnings.warn(
            f"stage cache: the appearance came from {loaded_id!r}, not "
            f"{requested!r}, so it was not stored",
            RuntimeWarning,
            stacklevel=2,
        )
    return result
