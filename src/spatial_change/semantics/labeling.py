"""Open-vocabulary labels for object instances.

Each object is cropped from the frames of its own epoch that see its points
best, one ``Labeler`` call labels the crops of all the objects, and the
labels of an object's crops are fused into one label with a score.

Importing this module does not load torch.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain import EpochReconstruction, ObjectInstance
from spatial_change.geometry.views import (
    UsableFrame,
    ViewConfig,
    crop_view,
    select_views,
    usable_frames,
)
from spatial_change.ports import Labeler, check_vocabulary
from spatial_change.semantics.vocabulary import INDOOR_OBJECTS


@dataclass(frozen=True)
class LabelConfig:
    """Parameters of the labeling. The defaults are provisional.

    They are placeholders until T22 calibrates them on real captures.

    Attributes:
        views: How the frames cropped for an object are picked.
        margin: Share of the visible extent added on every side of a crop,
            at least 0 and finite.
        min_margin_px: Smallest margin in pixels, at least 0 and finite;
            the margin of a side is the larger of the two.
        min_crop_px: A crop whose shorter side has fewer pixels is
            skipped, at least 1.
        min_score: Smallest fused score that gives a label, in ``[0, 1]``.
    """

    views: ViewConfig = ViewConfig(min_visible_points=20)
    margin: float = 0.1
    min_margin_px: float = 4.0
    min_crop_px: int = 16
    min_score: float = 0.3

    def __post_init__(self) -> None:
        """Validates the parameters.

        Raises:
            ValueError: If a count is below 1, a length or share is out of
                range or not finite, or a score or confidence is outside
                ``[0, 1]``.
        """
        if not 0.0 <= self.margin < math.inf:
            raise ValueError("margin must be >= 0 and finite")
        if not 0.0 <= self.min_margin_px < math.inf:
            raise ValueError("min_margin_px must be >= 0 and finite")
        if self.min_crop_px < 1:
            raise ValueError("min_crop_px must be >= 1")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be within [0, 1]")


_DEFAULT_CONFIG = LabelConfig()


def _crops(
    points: NDArray[np.float64],
    frames: Mapping[int, UsableFrame],
    config: LabelConfig,
) -> list[NDArray[np.uint8]]:
    """Crops the frames that see a set of points best.

    Args:
        points: ``(N, 3)`` world points of one object.
        frames: The usable frames of the points' epoch.
        config: View limits, margin, and minimum crop size.

    Returns:
        The crops in frame order; a view whose crop is smaller than
        ``min_crop_px`` on its shorter side is left out.
    """
    crops = (
        crop_view(
            frames[view.position],
            view,
            config.margin,
            config.min_margin_px,
            config.min_crop_px,
        )
        for view in select_views(points, frames, config.views)
    )
    return [crop for crop in crops if crop is not None]


def _fuse(
    results: Sequence[tuple[str, float]],
    vocabulary: Sequence[str],
    min_score: float,
) -> tuple[str | None, float | None]:
    """Fuses the labels of one object's crops.

    Args:
        results: The ``(label, score)`` of each crop, possibly none.
        vocabulary: The label names, in the order that breaks ties.
        min_score: Smallest fused score that gives a label.

    Returns:
        The label and its score: each label scores the sum of its crop
        scores over the number of crops, and the best label wins, the
        earlier vocabulary entry on a tie. Both are None without crops,
        when the best score is zero, or when it is under ``min_score``.

    Raises:
        ValueError: If a crop's label is not in the vocabulary.
    """
    if not results:
        return None, None
    index = {name: i for i, name in enumerate(vocabulary)}
    totals = np.zeros(len(vocabulary))
    for label, score in results:
        if label not in index:
            raise ValueError(
                f"the labeler returned {label!r}, not in the vocabulary"
            )
        totals[index[label]] += score
    totals /= len(results)
    best = int(np.argmax(totals))
    score = float(totals[best])
    if score <= 0.0 or score < min_score:
        return None, None
    return vocabulary[best], score


def label_point_sets(
    frames: Mapping[int, UsableFrame],
    point_sets: Sequence[NDArray[np.float64]],
    labeler: Labeler,
    vocabulary: Sequence[str],
    config: LabelConfig,
    min_score: float,
) -> list[tuple[str | None, float | None]]:
    """Labels sets of points of one epoch from their best views.

    Args:
        frames: The usable frames of the epoch.
        point_sets: ``(N_i, 3)`` world points of each object.
        labeler: The labeler; it is called once for all the crops and is
            not unloaded.
        vocabulary: Non-empty, unique label names.
        config: View and crop settings.
        min_score: Smallest fused score that gives a label.

    Returns:
        The label and score of each set, as fused by ``label_instances``.
    """
    crops: list[NDArray[np.uint8]] = []
    owner: list[int] = []
    for position, points in enumerate(point_sets):
        found = _crops(points, frames, config)
        crops.extend(found)
        owner.extend([position] * len(found))
    labels = labeler.label(crops, vocabulary) if crops else []
    grouped: list[list[tuple[str, float]]] = [[] for _ in point_sets]
    for position, result in zip(owner, labels, strict=True):
        grouped[position].append(result)
    return [_fuse(group, vocabulary, min_score) for group in grouped]


def label_instances(
    reconstruction: EpochReconstruction,
    instances: tuple[ObjectInstance, ...],
    labeler: Labeler,
    vocabulary: Sequence[str] = INDOOR_OBJECTS,
    config: LabelConfig = _DEFAULT_CONFIG,
) -> tuple[ObjectInstance, ...]:
    """Gives each instance the open-vocabulary label of its look.

    Each instance is cropped from up to ``max_views`` frames of its own
    epoch that see its points, with a margin around the visible pixels.
    One ``labeler`` call labels all the crops of all the instances. The
    score of a label for an instance is the sum of its crop scores over
    the number of crops, and the best label wins, the earlier vocabulary
    entry on a tie.

    Args:
        reconstruction: The epoch's reconstruction, whose frames carry the
            images and depth maps.
        instances: Instances of that epoch.
        labeler: Open-vocabulary labeler; it is not unloaded.
        vocabulary: Non-empty, unique label names.
        config: View, crop, and fusion settings.

    Returns:
        Copies of the instances in the same order with ``label`` and
        ``label_score`` set. Both are None when no crop exists, the best
        fused score is zero, or it is under ``min_score``, so the score is
        always the score of the label.

    Raises:
        ValueError: If the vocabulary is invalid, an instance belongs to
            another epoch, or the labeler rejects a crop or returns a label
            outside the vocabulary.
    """
    check_vocabulary(vocabulary)
    for item in instances:
        if item.epoch != reconstruction.epoch:
            raise ValueError(
                f"instance {item.object_id} is from epoch "
                f"{item.epoch.value}, the reconstruction from epoch "
                f"{reconstruction.epoch.value}"
            )
    if not instances:
        return ()
    fused = label_point_sets(
        usable_frames(reconstruction, config.views.min_confidence),
        [item.points for item in instances],
        labeler,
        vocabulary,
        config,
        config.min_score,
    )
    return tuple(
        dataclasses.replace(item, label=label, label_score=score)
        for item, (label, score) in zip(instances, fused, strict=True)
    )
