"""The models a run uses, built on demand so tests can pass fakes.

Each role is a factory rather than an instance: the run builds a model when
its stage starts and unloads it when the stage ends, so no two heavy models
share the GPU.

Importing this module does not load torch, transformers, or cv2.
"""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

from spatial_change.cache import JsonValue
from spatial_change.pipeline.config import RunConfig, resolve_device
from spatial_change.ports import (
    FeatureEncoder,
    Labeler,
    MaskSegmenter,
    Reconstructor,
    Unloadable,
    VisionLanguageModel,
)


@dataclass(frozen=True)
class Models:
    """One factory per model role.

    Attributes:
        reconstructor: Builds the joint reconstruction backbone.
        encoder: Builds the patch feature encoder; it is built twice, for
            the appearance comparison and for the object descriptors.
        segmenter: Builds the promptable mask segmenter.
        labeler: Builds the open-vocabulary labeler.
        vlm: Builds the vision-language model of the describer; it is only
            called when the run describes changes with a model.
        max_frames: Views the reconstruction accepts at once, which the
            keyframes of both videos together must not exceed.
        ids: The weights each role asks for, by role, for the report.
        settings: Every setting of the adapters whose results are cached,
            by role (``reconstruction`` and ``features``), so a cache key
            changes with any of them.
    """

    reconstructor: Callable[[], Reconstructor]
    encoder: Callable[[], FeatureEncoder]
    segmenter: Callable[[], MaskSegmenter]
    labeler: Callable[[], Labeler]
    vlm: Callable[[], VisionLanguageModel]
    max_frames: int = 128
    ids: Mapping[str, str] = field(default_factory=dict)
    settings: Mapping[str, Mapping[str, JsonValue]] = field(
        default_factory=dict
    )


@contextmanager
def loaded[T: Unloadable](make: Callable[[], T]) -> Iterator[T]:
    """Builds a model and unloads it when the block ends, even on error.

    Args:
        make: Factory of the model.

    Yields:
        The model.
    """
    model = make()
    try:
        yield model
    finally:
        model.unload()


def default_models(config: RunConfig) -> Models:
    """Builds the factories of the local models.

    No weights are loaded here; each model loads on its first use.

    Args:
        config: Supplies the device.

    Returns:
        Pi3X for the reconstruction, DINO for the features, SAM for the
        masks, SigLIP for the labels, and Qwen for the descriptions.
    """
    from spatial_change.perception import (
        DinoConfig,
        DinoFeatureEncoder,
        QwenVlm,
        QwenVlmConfig,
        SamConfig,
        SamSegmenter,
        SiglipConfig,
        SiglipLabeler,
    )
    from spatial_change.reconstruction import Pi3XConfig, Pi3XReconstructor

    device = resolve_device(config)
    pi3x = Pi3XConfig(device=device)
    dino = DinoConfig(device=device)
    sam = SamConfig(device=device)
    siglip = SiglipConfig(device=device)
    qwen = QwenVlmConfig(device=device)
    return Models(
        reconstructor=lambda: Pi3XReconstructor(pi3x),
        encoder=lambda: DinoFeatureEncoder(dino),
        segmenter=lambda: SamSegmenter(sam),
        labeler=lambda: SiglipLabeler(siglip),
        vlm=lambda: QwenVlm(qwen),
        max_frames=pi3x.max_frames,
        ids={
            "reconstruction": pi3x.model_id,
            "features": dino.model_id,
            "segmentation": sam.model_id,
            "labels": siglip.model_id,
            "descriptions": qwen.model_id,
        },
        settings={"reconstruction": asdict(pi3x), "features": asdict(dino)},
    )
