"""Labelled point clouds of synthetic scene pairs for the visibility tests."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain import Epoch
from spatial_change.geometry.projection import unproject_depth
from spatial_change.synthetic.scene import (
    SceneConfig,
    SyntheticEpoch,
    SyntheticScenePair,
    generate_scene_pair,
)

SCENE_CONFIG = SceneConfig(width=128, height=96, frames_per_epoch=12)
SEEDS = [0, 1, 2]


@dataclass(frozen=True, eq=False)
class LabelledCloud:
    """Surface points of one epoch with their object index.

    Attributes:
        points: World points of shape ``(N, 3)``.
        labels: Object index per point, -1 for the room shell.
    """

    points: NDArray[np.float64]
    labels: NDArray[np.int32]


def labelled_cloud(data: SyntheticEpoch) -> LabelledCloud:
    """Unprojects every valid pixel of every frame of an epoch.

    Args:
        data: Rendered frames and instance maps.

    Returns:
        The points with the instance label of their pixel.
    """
    points, labels = [], []
    for frame, instances in zip(data.frames, data.instance_maps, strict=True):
        assert frame.depth is not None
        valid = np.isfinite(frame.depth) & (frame.depth > 0)
        points.append(
            unproject_depth(
                frame.depth, frame.intrinsics, frame.world_from_camera
            )
        )
        labels.append(instances[valid])
    return LabelledCloud(np.concatenate(points), np.concatenate(labels))


def scene_clouds(
    seed: int,
) -> tuple[SyntheticScenePair, LabelledCloud, LabelledCloud]:
    """Generates a scene pair and the clouds of both epochs.

    Args:
        seed: Scene seed.

    Returns:
        The pair, its BEFORE cloud, and its AFTER cloud.
    """
    pair = generate_scene_pair(seed, SCENE_CONFIG)
    return (
        pair,
        labelled_cloud(pair.epoch_data(Epoch.BEFORE)),
        labelled_cloud(pair.epoch_data(Epoch.AFTER)),
    )
