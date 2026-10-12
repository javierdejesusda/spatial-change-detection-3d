"""Scores of lifted object instances against the labelled synthetic cloud."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from visibility_scenes import LabelledCloud, majority

from spatial_change.change import Candidate
from spatial_change.domain import ObjectInstance
from spatial_change.geometry import voxel_cells


def labels_of(
    cloud: LabelledCloud, points: NDArray[np.float64]
) -> NDArray[np.int32]:
    """Looks up the object label of lifted points in the cloud.

    The lifted points are unprojected from the same pixels and depth as the
    cloud, so every one equals a cloud point exactly.

    Args:
        cloud: The labelled cloud of the epoch.
        points: Lifted points.

    Returns:
        The label of each point.
    """
    table = {
        row.tobytes(): int(label)
        for row, label in zip(cloud.points, cloud.labels, strict=True)
    }
    return np.array([table[row.tobytes()] for row in points], np.int32)


def voxel_recall(
    cloud: LabelledCloud, label: int, points: NDArray[np.float64], size: float
) -> float:
    """Measures how much of an object's cloud lies in the voxels of points.

    Args:
        cloud: The labelled cloud of the epoch.
        label: Object index.
        points: The instance points.
        size: Voxel edge.

    Returns:
        The share of the object's points whose voxel holds an instance
        point.
    """
    cells = voxel_cells(points, size)
    truth = voxel_cells(cloud.points[cloud.labels == label], size)
    _, ids = np.unique(
        np.concatenate([cells, truth]), axis=0, return_inverse=True
    )
    ids = ids.reshape(-1)
    return float(np.mean(np.isin(ids[len(cells) :], ids[: len(cells)])))


def candidate_object(
    cloud: LabelledCloud,
    candidates: Sequence[Candidate],
    instance: ObjectInstance,
) -> int:
    """Gives the object that an instance's first candidate mostly covers.

    Args:
        cloud: The labelled cloud of the epoch.
        candidates: The candidates that were lifted.
        instance: One of the resulting instances.

    Returns:
        The object index.
    """
    candidate = next(
        c for c in candidates if c.candidate_id == instance.object_id
    )
    return majority(cloud, candidate.point_indices)[0]


def purity_and_recall(
    cloud: LabelledCloud,
    candidates: Sequence[Candidate],
    instance: ObjectInstance,
    voxel_size: float,
) -> tuple[float, float]:
    """Scores an instance against the object its first candidate covers.

    Args:
        cloud: The labelled cloud of the epoch.
        candidates: The candidates that were lifted.
        instance: One of the resulting instances.
        voxel_size: Voxel edge of the recall.

    Returns:
        The share of the instance's points on that object, and the share of
        that object's points in the instance's voxels.
    """
    target = candidate_object(cloud, candidates, instance)
    purity = float(np.mean(labels_of(cloud, instance.points) == target))
    recall = voxel_recall(cloud, target, instance.points, voxel_size)
    return purity, recall
