"""From a pipeline run to a scene the SceneDiff metrics can score.

A run reports 3D boxes, and the benchmark scores 2D masks, so the objects
behind each change are projected into the frames of the run, and the ground
truth is resized to the same grid. The pieces are the changes as scorable
objects, the grid, the projection of a whole run, and the scene that joins a
projection with the ground truth.

A run is projected once, with its uncertain changes drawn as moved objects
(or as added or removed ones, when a gate of the matcher left them a single
epoch) and flagged. Whether they count is a choice of the scoring, made when
the scene is built, so it can change without running the pipeline again.

Importing this module loads the pipeline package, which does not load torch,
transformers, open3d, or cv2.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from spatial_change.change import ChangeSources
from spatial_change.domain import Change, ChangeType, Epoch
from spatial_change.evaluation.metrics import PredItem, SceneEval
from spatial_change.evaluation.predictions import Predictions
from spatial_change.evaluation.projection import (
    ChangeObject3D,
    ProjectionConfig,
    project_to_pred_items,
)
from spatial_change.evaluation.scenediff import (
    DEFAULT_FRAME_STRIDE,
    DEFAULT_MAX_LENGTH,
    PairAnnotations,
    eval_size,
    gt_items,
    official_eval_hw,
)
from spatial_change.pipeline import MetadataValue, PipelineResult

UncertainAs = Literal["drop", "moved"]
"""What becomes of an uncertain change when a run is scored."""


@dataclass(frozen=True)
class PairEvalConfig:
    """How one pair is turned into a scored scene.

    Attributes:
        stride: Frames whose index is a multiple of this are scored.
        max_length: Length of the longer side of the evaluation grid.
        projection: Settings of the depth-tested projection.
        uncertain_as: ``"drop"`` leaves out the changes the pipeline could
            not classify, and ``"moved"`` scores them at their own
            confidence, as they are drawn: an uncertain change with both
            boxes as a moved object, and one with a single box (demoted by a
            gate of the matcher) in its one epoch, as an added or removed
            object. The metrics know only moved and static objects, so
            there is no honest score for an uncertain one.
    """

    stride: int = DEFAULT_FRAME_STRIDE
    max_length: int = DEFAULT_MAX_LENGTH
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    uncertain_as: UncertainAs = "drop"

    def __post_init__(self) -> None:
        """Validates the settings.

        Raises:
            ValueError: If the stride or the length is below 1, or
                ``uncertain_as`` is neither ``"drop"`` nor ``"moved"``.
        """
        if self.stride < 1:
            raise ValueError("stride must be at least 1")
        if self.max_length < 1:
            raise ValueError("max_length must be at least 1")
        if self.uncertain_as not in ("drop", "moved"):
            raise ValueError(
                'uncertain_as must be "drop" or "moved", got '
                f"{self.uncertain_as!r}"
            )


def _points_by_epoch(
    change: Change, sources: ChangeSources
) -> dict[Epoch, NDArray[np.float64]]:
    """Gathers the object points of a change in the epochs it is present in.

    Raises:
        ValueError: If the sources are not the objects the boxes of the
            change are made from.
    """
    wants_before = change.before_box is not None
    wants_after = change.after_box is not None
    if (sources.before is not None) != wants_before or (
        sources.after is not None
    ) != wants_after:
        raise ValueError(
            f"{change.change_id}: a {change.type.value} change does not come "
            "from the objects it has"
        )
    points: dict[Epoch, NDArray[np.float64]] = {}
    if sources.before is not None:
        points[Epoch.BEFORE] = sources.before.points
    if sources.after is not None:
        points[Epoch.AFTER] = sources.after.points
    return points


def _drawn_kind(change: Change) -> ChangeType:
    """Gives the kind of object an uncertain change is drawn as.

    A pair is drawn as moved, and a change a gate of the matcher demoted
    from added or removed keeps the one epoch it has.
    """
    if change.type is not ChangeType.UNCERTAIN:
        return change.type
    if change.before_box is None:
        return ChangeType.ADDED
    if change.after_box is None:
        return ChangeType.REMOVED
    return ChangeType.MOVED


def change_objects(
    changes: Sequence[Change], sources: Mapping[str, ChangeSources]
) -> list[ChangeObject3D]:
    """Turns the changes of a run into objects the projection can draw.

    An uncertain change is drawn as a moved object when it has the points
    of both epochs, and as an added or removed one when it has only those of
    one; whether it is scored is decided later.

    Args:
        changes: The changes of a report.
        sources: The objects behind them, by change id.

    Returns:
        One object per change, in order. Its id is the position of the
        change in ``changes`` and its score is the confidence of the change.

    Raises:
        ValueError: If a change has no sources or its sources do not fit
            its type.
    """
    objects = []
    for index, change in enumerate(changes):
        source = sources.get(change.change_id)
        if source is None:
            raise ValueError(f"{change.change_id}: the run kept no objects")
        objects.append(
            ChangeObject3D(
                index,
                _drawn_kind(change),
                change.confidence,
                _points_by_epoch(change, source),
            )
        )
    return objects


def eval_grid(
    annotations: PairAnnotations,
    metadata: Mapping[str, MetadataValue],
    max_length: int,
) -> tuple[int, int]:
    """Picks the evaluation grid of a pair.

    Args:
        annotations: The ground truth of the pair.
        metadata: The metadata of the report of the run.
        max_length: Length of the longer side of the grid.

    Returns:
        The grid of the first ground-truth mask. A pair without any mask
        gets the grid its BEFORE video would have, so its predictions still
        count, all as false positives.

    Raises:
        ValueError: If the pair has no mask and the metadata lacks the size
            of the BEFORE video.
    """
    grid = eval_size(annotations, max_length)
    if grid is not None:
        return grid
    try:
        return official_eval_hw(
            int(metadata["video_before_height"]),
            int(metadata["video_before_width"]),
            max_length,
        )
    except KeyError as error:
        raise ValueError(
            f"{annotations.pair}: no ground-truth mask and no size of the "
            "BEFORE video to take the evaluation grid from"
        ) from error


def run_pred_items(
    result: PipelineResult, config: PairEvalConfig, height: int, width: int
) -> list[PredItem]:
    """Projects every change of a run into the scored frames of both videos.

    Args:
        result: The run.
        config: Stride and projection settings.
        height: Grid height.
        width: Grid width.

    Returns:
        One item per change seen in a scored frame, uncertain ones included
        and drawn as ``change_objects`` draws them.

    Raises:
        ValueError: If the sources of the run do not fit its changes.
    """
    scene = result.scene
    return project_to_pred_items(
        change_objects(result.report.changes, scene.sources),
        scene.before_frames,
        scene.after_frames,
        height,
        width,
        config=config.projection,
        stride=config.stride,
        after_window=scene.after_window,
    )


def uncertain_ids(changes: Sequence[Change]) -> frozenset[int]:
    """Gives the object ids of the changes the run could not classify."""
    return frozenset(
        index
        for index, change in enumerate(changes)
        if change.type is ChangeType.UNCERTAIN
    )


def scored_items(
    items: Sequence[PredItem],
    uncertain: Collection[int],
    uncertain_as: UncertainAs,
) -> list[PredItem]:
    """Keeps the objects a scoring counts.

    Args:
        items: Predicted objects.
        uncertain: Ids of those that came from uncertain changes.
        uncertain_as: ``"drop"`` leaves the uncertain ones out, ``"moved"``
            keeps them.

    Returns:
        The kept objects, in order.
    """
    if uncertain_as == "moved":
        return list(items)
    return [item for item in items if item.obj_id not in uncertain]


def project_run(
    result: PipelineResult,
    annotations: PairAnnotations,
    config: PairEvalConfig,
) -> Predictions:
    """Projects a run on the evaluation grid of its pair.

    Args:
        result: The run on the pair.
        annotations: The ground truth of the pair.
        config: Stride, grid, and projection settings.

    Returns:
        The objects seen in a scored frame, and which of them are uncertain.

    Raises:
        ValueError: If the pair has no grid (see ``eval_grid``) or the run's
            sources do not fit its changes.
    """
    height, width = eval_grid(
        annotations, result.report.metadata, config.max_length
    )
    items = run_pred_items(result, config, height, width)
    uncertain = uncertain_ids(result.report.changes)
    return Predictions(
        height,
        width,
        items,
        frozenset(item.obj_id for item in items if item.obj_id in uncertain),
    )


def build_scene(
    name: str,
    annotations: PairAnnotations,
    predictions: Predictions,
    config: PairEvalConfig,
) -> SceneEval:
    """Joins the ground truth of a pair with the objects a run predicted.

    Args:
        name: Name of the scene.
        annotations: The ground truth of the pair.
        predictions: The objects, drawn on their own grid; none for a pair
            that failed.
        config: Stride and the treatment of uncertain objects.

    Returns:
        The scene at the grid of the predictions.
    """
    return SceneEval(
        name=name,
        height=predictions.height,
        width=predictions.width,
        gt=gt_items(
            annotations, predictions.height, predictions.width, config.stride
        ),
        pred=scored_items(
            predictions.items, predictions.uncertain, config.uncertain_as
        ),
    )
