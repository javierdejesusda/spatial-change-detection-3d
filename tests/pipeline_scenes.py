"""A synthetic scene pair turned into the fakes the pipeline tests run on.

The reconstructor answers with the true depth, intrinsics, and poses of the
rendered frames, the segmenter with the true object under a prompt, and the
labeler by the hue of the true object colors. Every model is wrapped so a
shared timeline sees when it loads and when it unloads.
"""

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from matching_pipeline import oracle_segmenter
from numpy.typing import NDArray
from visibility_scenes import SEEDS, scene_clouds

from spatial_change.capture import VideoFrame
from spatial_change.domain import Epoch, Frame
from spatial_change.perception import FakeFeatureEncoder, FakeLabeler
from spatial_change.perception.fake import FAKE_ENCODER_ID
from spatial_change.pipeline import (
    Models,
    PipelineResult,
    ProgressCallback,
    RunConfig,
    run_frames,
)
from spatial_change.ports import ReconstructionResult
from spatial_change.synthetic.scene import SyntheticScenePair

PATCH = 8
TEMPLATE = RunConfig(device="cpu", describer="template", voxel_size=None)


@dataclass
class Timeline:
    """What happened to the models, in order.

    Attributes:
        events: ``(what, name)`` pairs, ``what`` being ``"load"`` or
            ``"unload"``.
        loaded: Names of the models loaded now.
        most_loaded: Largest number of models loaded at once.
    """

    events: list[tuple[str, str]] = field(default_factory=list)
    loaded: set[str] = field(default_factory=set)
    most_loaded: int = 0

    def touch(self, name: str) -> None:
        if name not in self.loaded:
            self.loaded.add(name)
            self.events.append(("load", name))
            self.most_loaded = max(self.most_loaded, len(self.loaded))

    def release(self, name: str) -> None:
        self.loaded.discard(name)
        self.events.append(("unload", name))

    def roles(self) -> list[str]:
        """Roles in the order their models loaded."""
        return [
            name.split("#")[0] for what, name in self.events if what == "load"
        ]


class Tracked:
    """Forwards to a fake model and reports its use to a timeline."""

    def __init__(self, name: str, inner: Any, timeline: Timeline) -> None:
        self._name = name
        self._inner = inner
        self._timeline = timeline

    def __getattr__(self, attribute: str) -> Any:
        target = getattr(self._inner, attribute)
        if attribute == "unload":

            def unload() -> None:
                self._timeline.release(self._name)
                target()

            return unload
        if not callable(target):
            self._timeline.touch(self._name)
            return target

        def call(*args: Any, **kwargs: Any) -> Any:
            self._timeline.touch(self._name)
            return target(*args, **kwargs)

        return call


class OracleReconstructor:
    """Answers with the true geometry of the rendered frames."""

    def __init__(self, pair: SyntheticScenePair) -> None:
        self._frames: dict[int, Frame] = {
            id(frame.image): frame
            for epoch in Epoch
            for frame in pair.epoch_data(epoch).frames
        }

    def reconstruct(
        self, images: Sequence[NDArray[np.uint8]]
    ) -> ReconstructionResult:
        frames = [self._frames[id(image)] for image in images]
        depths = [f.depth for f in frames]
        assert all(depth is not None for depth in depths)
        return ReconstructionResult(
            intrinsics=[f.intrinsics for f in frames],
            world_from_camera=[f.world_from_camera for f in frames],
            depth=[d for d in depths if d is not None],
            confidence=[
                np.ones(d.shape, np.float32) for d in depths if d is not None
            ],
        )

    def unload(self) -> None:
        pass


class DescribingVlm:
    """A vision-language model that answers with a plain sentence.

    Attributes:
        calls: ``(image count, prompt)`` of every call.
        unload_count: How many times it was unloaded.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []
        self.unload_count = 0

    def generate(self, images: Sequence[NDArray[np.uint8]], prompt: str) -> str:
        self.calls.append((len(images), prompt))
        fields = dict(
            line.split(": ", 1) for line in prompt.splitlines() if ": " in line
        )
        label = fields.get("Object", "object")
        kind = fields.get("Change type", "changed")
        word = {"added": "added", "removed": "removed", "moved": "moved"}.get(
            kind, "changed"
        )
        return f'{{"description": "The {label} was {word}."}}'

    def unload(self) -> None:
        self.unload_count += 1


def video_frames(pair: SyntheticScenePair, epoch: Epoch) -> list[VideoFrame]:
    """Turns the rendered frames of an epoch into video frames.

    Args:
        pair: The scene pair.
        epoch: The epoch.

    Returns:
        Video frames that share the image arrays of the rendered frames.
    """
    frames = []
    for frame in pair.epoch_data(epoch).frames:
        assert frame.image is not None
        frames.append(VideoFrame(frame.index, frame.timestamp_s, frame.image))
    return frames


def synthetic_models(
    pair: SyntheticScenePair,
    timeline: Timeline,
    vlm: Callable[[], Any] = DescribingVlm,
) -> Models:
    """Builds the factories of the fake models for a scene pair.

    Args:
        pair: The scene pair the oracle reconstructor and segmenter know.
        timeline: Receives the loads and unloads of every model built.
        vlm: Factory of the describing model.

    Returns:
        The model factories, each wrapping a fresh fake.
    """
    colors = {o.label: o.model.color for o in pair.objects}
    counter = itertools.count()

    def track(role: str, make: Callable[[], Any]) -> Callable[[], Any]:
        return lambda: Tracked(f"{role}#{next(counter)}", make(), timeline)

    return Models(
        reconstructor=track("reconstructor", lambda: OracleReconstructor(pair)),
        encoder=track("encoder", lambda: FakeFeatureEncoder(PATCH)),
        segmenter=track("segmenter", lambda: oracle_segmenter(pair)),
        labeler=track("labeler", lambda: FakeLabeler(colors)),
        vlm=track("vlm", vlm),
        max_frames=128,
        ids={
            "reconstruction": "oracle",
            "features": FAKE_ENCODER_ID,
            "descriptions": "fake",
        },
    )


def run_pair(
    out_dir: Path,
    config: RunConfig = TEMPLATE,
    progress: ProgressCallback | None = None,
    *,
    seed: int = SEEDS[0],
    cache_dir: Path | None = None,
    vlm: Callable[[], Any] = DescribingVlm,
    mutate: Callable[[list[VideoFrame]], list[VideoFrame]] | None = None,
    adapt: Callable[[Models], Models] | None = None,
) -> tuple[PipelineResult, Timeline]:
    """Runs the pipeline on the keyframes of a scene pair, with fake models.

    Args:
        out_dir: Output folder.
        config: The knobs of the run.
        progress: Receives the start and the end of every stage.
        seed: Seed of the scene pair.
        cache_dir: Folder of the stage cache, or ``None`` for no cache.
        vlm: Factory of the describing model.
        mutate: Changes the BEFORE keyframes before the run.
        adapt: Changes the models before the run.

    Returns:
        The result and the timeline of the models built.
    """
    pair, _, _ = scene_clouds(seed)
    timeline = Timeline()
    models = synthetic_models(pair, timeline, vlm)
    before = video_frames(pair, Epoch.BEFORE)
    result = run_frames(
        mutate(before) if mutate else before,
        video_frames(pair, Epoch.AFTER),
        out_dir,
        config,
        adapt(models) if adapt else models,
        progress,
        cache_dir=cache_dir,
    )
    return result, timeline
