"""Labelled point clouds and votes of synthetic scene pairs for the tests.

The scenes and votes are cached per process and shared by the test modules,
so callers must not modify the arrays they return.
"""

import functools
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spatial_change.change import Candidate
from spatial_change.domain import Epoch
from spatial_change.geometry.projection import unproject_depth
from spatial_change.synthetic.scene import (
    ObjectStatus,
    SceneConfig,
    SyntheticEpoch,
    SyntheticObject,
    SyntheticScenePair,
    generate_scene_pair,
)
from spatial_change.visibility import (
    PointEvidence,
    PointState,
    VisibilityConfig,
    vote_points,
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


def changed_objects(
    pair: SyntheticScenePair, epoch: Epoch
) -> list[SyntheticObject]:
    """Lists the changed objects that exist in an epoch.

    Args:
        pair: The scene pair.
        epoch: The epoch.

    Returns:
        The objects whose status is not unchanged and that exist in it.
    """
    return [
        o
        for o in pair.objects
        if o.status != ObjectStatus.UNCHANGED and o.exists_in(epoch)
    ]


def majority(
    cloud: LabelledCloud, point_indices: NDArray[np.intp]
) -> tuple[int, float]:
    """Returns the commonest label among some points and its share.

    Args:
        cloud: The labelled cloud.
        point_indices: Indices into the cloud, such as a candidate's.

    Returns:
        The label and the share of the points that carry it.
    """
    values, counts = np.unique(cloud.labels[point_indices], return_counts=True)
    return int(values[np.argmax(counts)]), float(counts.max() / counts.sum())


def changed_points_of(
    cloud: LabelledCloud, states: NDArray[np.int8], obj: SyntheticObject
) -> NDArray[np.bool_]:
    """Marks the CHANGED points of a cloud that lie on an object.

    Args:
        cloud: The labelled cloud.
        states: Point states of the cloud.
        obj: The object.

    Returns:
        ``(N,)`` mask of the CHANGED points with the object's label.
    """
    return (states == PointState.CHANGED) & (cloud.labels == obj.index)


def dominant(
    cloud: LabelledCloud,
    states: NDArray[np.int8],
    obj: SyntheticObject,
    candidates: tuple[Candidate, ...],
) -> tuple[Candidate, float]:
    """Finds the candidate with most of an object's CHANGED points.

    Args:
        cloud: The labelled cloud.
        states: Point states of the cloud.
        obj: The object, which must have CHANGED points.
        candidates: The candidates found on the cloud, at least one.

    Returns:
        The candidate and its share of the object's CHANGED points.
    """
    mine = changed_points_of(cloud, states, obj)
    assert mine.any(), obj.object_id
    assert candidates, f"no candidates for {obj.object_id}"
    held = [int(mine[c.point_indices].sum()) for c in candidates]
    best = int(np.argmax(held))
    return candidates[best], held[best] / int(mine.sum())


@functools.cache
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


@dataclass(frozen=True, eq=False)
class SceneEvidence:
    """A scene pair with the votes on the cloud of each epoch.

    Attributes:
        pair: The scene pair.
        clouds: The labelled cloud of each epoch.
        evidence: The votes on each epoch's cloud from the frames of the
            other epoch.
    """

    pair: SyntheticScenePair
    clouds: dict[Epoch, LabelledCloud]
    evidence: dict[Epoch, PointEvidence]


@functools.cache
def scene_evidence(seed: int, config: VisibilityConfig) -> SceneEvidence:
    """Votes each epoch's cloud of a scene against the other epoch's frames.

    Args:
        seed: Scene seed.
        config: Visibility parameters; equal configs share one result.

    Returns:
        The pair, its clouds, and the votes on both clouds.
    """
    pair, before, after = scene_clouds(seed)
    clouds = {Epoch.BEFORE: before, Epoch.AFTER: after}
    voters = {Epoch.BEFORE: pair.after.frames, Epoch.AFTER: pair.before.frames}
    evidence = {
        epoch: vote_points(cloud.points, voters[epoch], config)
        for epoch, cloud in clouds.items()
    }
    return SceneEvidence(pair, clouds, evidence)


def hidden_object_indices(pair: SyntheticScenePair) -> set[int]:
    """Lists the objects behind the occluder or inside the unobserved zone.

    Args:
        pair: The scene pair.

    Returns:
        The object indices of its occluded and unobserved-zone objects.
    """
    return {
        pair.object_by_id(i).index
        for i in (*pair.occluded_object_ids, *pair.zone_object_ids)
    }
