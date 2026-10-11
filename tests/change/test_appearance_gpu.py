import time
from collections.abc import Iterator

import numpy as np
import pytest
from appearance_scenes import recolored_frames, reconstruction
from numpy.typing import NDArray
from visibility_scenes import labelled_cloud

from spatial_change.change import AppearanceConfig, compare_appearance
from spatial_change.domain import Epoch
from spatial_change.perception import DinoFeatureEncoder
from spatial_change.synthetic.scene import (
    ObjectStatus,
    SceneConfig,
    SyntheticScenePair,
    generate_scene_pair,
)

pytestmark = pytest.mark.gpu

POINT_STRIDE = 6
# Measured with the DINOv3 ViT-S/16 weights on seed 0 at 518x294 with 6
# frames per epoch and every 6th point: static surfaces have a median cosine
# of 0.954 (first percentile 0.82, minimum 0.65), and the recolored flat
# shelf drops from a median of 0.953 to 0.914. The two ranges overlap, so
# the real features do not separate a flat recolor from lighting noise; the
# flip threshold is calibrated on real captures at T22. Only the margins
# below are pinned: static surfaces stay far above the default threshold,
# and the recolor lowers the shelf's similarity.
MIN_STATIC_MEDIAN = 0.9
MAX_STATIC_BELOW_DEFAULT = 0.001
MIN_SHELF_DROP = 0.01


@pytest.fixture(scope="module")
def encoder() -> Iterator[DinoFeatureEncoder]:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    dino = DinoFeatureEncoder()
    try:
        dino.load()
    except OSError as error:
        pytest.skip(f"no DINO model can be loaded: {error}")
    yield dino
    dino.unload()


@pytest.fixture(scope="module")
def pair() -> SyntheticScenePair:
    config = SceneConfig(width=518, height=294, frames_per_epoch=6)
    return generate_scene_pair(0, config)


def similarity(
    encoder: DinoFeatureEncoder, pair: SyntheticScenePair, recolor: bool
) -> dict[Epoch, tuple[NDArray[np.float32], NDArray[np.int32]]]:
    """Compares the scene, optionally with the shelf recolored in AFTER."""
    shelf = next(o for o in pair.objects if o.label == "shelf")
    after_frames = pair.after.frames
    if recolor:
        after_frames = recolored_frames(
            after_frames, pair.after.instance_maps, shelf.index
        )
    clouds = {e: labelled_cloud(pair.epoch_data(e)) for e in Epoch}
    points = {e: c.points[::POINT_STRIDE] for e, c in clouds.items()}
    started = time.perf_counter()
    appearance = compare_appearance(
        reconstruction(Epoch.BEFORE, pair.before.frames, points[Epoch.BEFORE]),
        reconstruction(Epoch.AFTER, after_frames, points[Epoch.AFTER]),
        encoder,
        AppearanceConfig(),
    )
    print(
        f"{encoder.loaded_model_id}: compared 12 frames in "
        f"{time.perf_counter() - started:.1f} s (recolor={recolor})"
    )
    return {
        Epoch.BEFORE: (
            appearance.before.similarity,
            clouds[Epoch.BEFORE].labels[::POINT_STRIDE],
        ),
        Epoch.AFTER: (
            appearance.after.similarity,
            clouds[Epoch.AFTER].labels[::POINT_STRIDE],
        ),
    }


def test_real_features_keep_static_surfaces_similar_and_lower_the_recolor(
    encoder: DinoFeatureEncoder, pair: SyntheticScenePair
) -> None:
    shelf = next(o for o in pair.objects if o.label == "shelf")
    changed = [
        o.index for o in pair.objects if o.status != ObjectStatus.UNCHANGED
    ]
    plain = similarity(encoder, pair, recolor=False)
    swapped = similarity(encoder, pair, recolor=True)

    for epoch in Epoch:
        sim, labels = plain[epoch]
        static = ~np.isin(labels, [shelf.index, *changed]) & ~np.isnan(sim)
        shelf_before = np.median(sim[(labels == shelf.index) & ~np.isnan(sim)])
        sim2, _ = swapped[epoch]
        shelf_after = np.median(sim2[(labels == shelf.index) & ~np.isnan(sim2)])
        print(
            f"{epoch.value}: static median {np.median(sim[static]):.3f} "
            f"p1 {np.percentile(sim[static], 1):.3f} "
            f"min {sim[static].min():.3f}; shelf median "
            f"{shelf_before:.3f} -> {shelf_after:.3f}"
        )
        assert np.median(sim[static]) >= MIN_STATIC_MEDIAN
        assert np.mean(sim[static] < 0.4) <= MAX_STATIC_BELOW_DEFAULT
        assert shelf_after <= shelf_before - MIN_SHELF_DROP
