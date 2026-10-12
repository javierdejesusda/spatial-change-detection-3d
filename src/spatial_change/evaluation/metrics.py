"""SceneDiff metrics: pixel IoU and object AP per image and per scene.

Reimplements the definitions of ``scripts/evaluate_multiview.py`` from
https://github.com/yuqunw/scene_diff (MIT license, copyright 2025 Yuqun Wu),
commit 62dcbd285fcfe984bb3db55dc01bdcd1a0055714; see ``NOTICE``. All
frames are scored at a common resolution per scene, and every metric pools
its detections over all scenes before the average precision is taken.

* Pixel IoU: per scene, every predicted mask of a (video, frame) is merged
  into one mask and so is every ground-truth mask; true positive, false
  positive and false negative pixels are summed over all frames and scenes.
* Object AP per image: each predicted mask is a detection ranked by its
  score, matched to the best ground-truth mask of the same (video, frame)
  with IoU above the threshold.
* Object AP per scene: each predicted object is one detection, ranked by the
  mean of its mask scores, matched to the ground-truth object with the best
  IoU above the threshold, where the IoU stacks the masks of both videos.
* Typed AP: as object AP per scene, but a prediction only matches ground
  truth of the same type. The official type has two classes: moved (the
  object has masks in both videos) and static (added or removed).

A ground-truth item or prediction may be matched at most
``duplicate_match_threshold`` times (``per_frame_...`` for masks); the first
match is a true positive, repeats within the threshold are ignored, and
repeats beyond it are false positives.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FrameKey = tuple[int, int]
"""A ``(video, frame)`` pair; ``video`` is 1 or 2."""

_VIDEOS = (1, 2)


def _check_keys(keys: Iterable[FrameKey]) -> None:
    for video, _ in keys:
        if video not in _VIDEOS:
            raise ValueError(f"video must be 1 or 2, got {video}")


@dataclass(frozen=True, eq=False)
class GtItem:
    """One ground-truth changed object.

    Attributes:
        obj_id: Object id.
        masks: Boolean masks keyed by ``(video, frame)``. May be empty: the
            object then counts toward the totals but cannot be matched.
        moved: Whether the object is moved (present in both videos); added
            and removed objects are static.
    """

    obj_id: int
    masks: Mapping[FrameKey, NDArray[np.bool_]]
    moved: bool = False

    def __post_init__(self) -> None:
        """Checks the video indices.

        Raises:
            ValueError: If a key names a video other than 1 or 2.
        """
        _check_keys(self.masks)


@dataclass(frozen=True, eq=False)
class PredItem:
    """One predicted changed object.

    Attributes:
        obj_id: Object id.
        masks: Boolean masks keyed by ``(video, frame)``.
        scores: One confidence per mask, higher is better.
    """

    obj_id: int
    masks: Mapping[FrameKey, NDArray[np.bool_]]
    scores: Mapping[FrameKey, float]

    def __post_init__(self) -> None:
        """Checks the video indices and the scores.

        Raises:
            ValueError: If a key names a video other than 1 or 2, or the
                scores do not cover exactly the masks.
        """
        _check_keys(self.masks)
        if set(self.scores) != set(self.masks):
            raise ValueError("every mask needs exactly one score")

    @property
    def moved(self) -> bool:
        """Whether the object has masks in both videos."""
        return {video for video, _ in self.masks} == set(_VIDEOS)

    @property
    def confidence(self) -> float:
        """Mean score over the object's masks, 0 without masks."""
        if not self.scores:
            return 0.0
        return float(np.mean(list(self.scores.values())))


@dataclass(frozen=True, eq=False)
class SceneEval:
    """Ground truth and predictions of one scene at one resolution.

    Attributes:
        name: Scene name.
        height: Common mask height.
        width: Common mask width.
        gt: Ground-truth objects.
        pred: Predicted objects.
    """

    name: str
    height: int
    width: int
    gt: Sequence[GtItem]
    pred: Sequence[PredItem]


@dataclass(frozen=True)
class EvalConfig:
    """Matching parameters; the defaults are the official ones.

    Attributes:
        iou_threshold: A match needs an IoU strictly above this value.
        duplicate_match_threshold: Matches allowed per ground-truth object.
        per_frame_duplicate_match_threshold: Matches allowed per
            ground-truth mask.
    """

    iou_threshold: float = 0.5
    duplicate_match_threshold: int = 1
    per_frame_duplicate_match_threshold: int = 1

    def __post_init__(self) -> None:
        """Validates the parameters.

        Raises:
            ValueError: If the IoU threshold is outside ``[0, 1)`` or a
                duplicate threshold is below 1.
        """
        if not 0.0 <= self.iou_threshold < 1.0:
            raise ValueError("iou_threshold must be in [0, 1)")
        if (
            self.duplicate_match_threshold < 1
            or self.per_frame_duplicate_match_threshold < 1
        ):
            raise ValueError("duplicate thresholds must be at least 1")


@dataclass(frozen=True)
class Metrics:
    """Pooled SceneDiff metrics.

    Attributes:
        pixel_iou: Pixel IoU of the merged masks over all frames.
        ap_per_image: Object AP per image.
        ap_per_scene: Object AP per scene, ignoring the change type.
        ap_per_scene_typed: Object AP per scene, requiring the type.
        tp: Object-level true positives (type ignored).
        fp: Object-level false positives (type ignored).
        fn: Ground-truth objects not matched (type ignored).
        pixel_tp: Pixel true positives.
        pixel_fp: Pixel false positives.
        pixel_fn: Pixel false negatives.
    """

    pixel_iou: float
    ap_per_image: float
    ap_per_scene: float
    ap_per_scene_typed: float
    tp: int
    fp: int
    fn: int
    pixel_tp: int
    pixel_fp: int
    pixel_fn: int


def voc_ap(
    recalls: NDArray[np.float64], precisions: NDArray[np.float64]
) -> float:
    """Computes VOC all-point interpolated average precision.

    Args:
        recalls: Recall after each ranked detection.
        precisions: Precision after each ranked detection.

    Returns:
        The area under the monotone precision envelope.
    """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


@dataclass
class _Ranked:
    confidence: float
    matched: bool


def _ap(ranked: list[_Ranked], total: int) -> float:
    ranked = sorted(ranked, key=lambda d: d.confidence, reverse=True)
    tp = np.cumsum([d.matched for d in ranked], dtype=np.float64)
    fp = np.cumsum([not d.matched for d in ranked], dtype=np.float64)
    recalls = tp / total if total > 0 else np.zeros_like(tp)
    precisions = tp / (tp + fp) if len(tp) else np.zeros_like(tp)
    return voc_ap(recalls, precisions)


def _check_scene(scene: SceneEval) -> None:
    shape = (scene.height, scene.width)
    items: list[GtItem | PredItem] = [*scene.gt, *scene.pred]
    for item in items:
        for mask in item.masks.values():
            if mask.shape != shape or mask.dtype != np.bool_:
                raise ValueError(
                    f"{scene.name}: masks must be bool with shape {shape}, "
                    f"got {mask.dtype} {mask.shape}"
                )


def _pixel_counts(scene: SceneEval) -> tuple[int, int, int]:
    merged: dict[str, dict[FrameKey, NDArray[np.bool_]]] = {
        "pred": {},
        "gt": {},
    }
    for side, items in (("gt", scene.gt), ("pred", scene.pred)):
        for item in items:
            for key, mask in item.masks.items():
                bucket = merged[side]
                bucket[key] = bucket[key] | mask if key in bucket else mask
    tp = fp = fn = 0
    for key, pred in merged["pred"].items():
        gt = merged["gt"].get(key)
        if gt is None:
            fp += int(np.count_nonzero(pred))
            continue
        inter = int(np.count_nonzero(pred & gt))
        tp += inter
        fp += int(np.count_nonzero(pred)) - inter
        fn += int(np.count_nonzero(gt)) - inter
    for key, gt in merged["gt"].items():
        if key not in merged["pred"]:
            fn += int(np.count_nonzero(gt))
    return tp, fp, fn


def _register(
    matches: dict[tuple[int, ...], int],
    best: tuple[int, ...] | None,
    threshold: int,
) -> tuple[bool, bool]:
    """Applies the duplicate-match rule to one detection.

    The first match of a ground-truth item is a true positive, repeats
    within the threshold are ignored, and repeats beyond it are false
    positives.

    Args:
        matches: Match count per ground-truth item, updated in place.
        best: The ground-truth item the detection matched, if any.
        threshold: Matches allowed per ground-truth item.

    Returns:
        ``(matched, record)``: whether the detection counts as matched and
        whether it enters the ranked list.
    """
    if best is None:
        return False, True
    count = matches.get(best, 0)
    if count == 0:
        matches[best] = 1
        return True, True
    if count < threshold:
        matches[best] = count + 1
        return True, False
    return False, True


class _Overlaps:
    """Mask areas and intersections between a scene's pred and gt masks."""

    def __init__(self, scene: SceneEval) -> None:
        self.scene = scene
        self.gt_by_key: dict[FrameKey, list[int]] = {}
        for gi, gt in enumerate(scene.gt):
            for key in gt.masks:
                self.gt_by_key.setdefault(key, []).append(gi)
        self.pred_area = [_areas(p.masks) for p in scene.pred]
        self.gt_area = [_areas(g.masks) for g in scene.gt]
        self.inter: dict[tuple[int, int], dict[FrameKey, int]] = {}
        for pi, pred in enumerate(scene.pred):
            for key, mask in pred.masks.items():
                for gi in self.gt_by_key.get(key, ()):
                    both = int(np.count_nonzero(mask & scene.gt[gi].masks[key]))
                    self.inter.setdefault((pi, gi), {})[key] = both

    def frame_iou(self, pi: int, gi: int, key: FrameKey) -> float:
        inter = self.inter[(pi, gi)][key]
        union = self.pred_area[pi][key] + self.gt_area[gi][key] - inter
        return inter / union if union else 0.0

    def object_iou(self, pi: int, gi: int) -> float:
        inter = sum(self.inter.get((pi, gi), {}).values())
        union = (
            sum(self.pred_area[pi].values())
            + sum(self.gt_area[gi].values())
            - inter
        )
        return inter / union if union else 0.0


def _areas(masks: Mapping[FrameKey, NDArray[np.bool_]]) -> dict[FrameKey, int]:
    return {key: int(np.count_nonzero(m)) for key, m in masks.items()}


def _match_frames(
    scene: SceneEval,
    overlaps: _Overlaps,
    config: EvalConfig,
    out: list[_Ranked],
) -> int:
    detections = sorted(
        (
            (pi, key, pred.scores[key])
            for pi, pred in enumerate(scene.pred)
            for key in sorted(pred.masks, key=lambda k: k[0])
        ),
        key=lambda d: d[2],
        reverse=True,
    )
    matches: dict[tuple[int, ...], int] = {}
    for pi, key, score in detections:
        best: tuple[int, ...] | None = None
        best_iou = 0.0
        for gi in overlaps.gt_by_key.get(key, ()):
            iou = overlaps.frame_iou(pi, gi, key)
            if iou > config.iou_threshold and iou > best_iou:
                best, best_iou = (gi, key[0], key[1]), iou
        matched, record = _register(
            matches, best, config.per_frame_duplicate_match_threshold
        )
        if record:
            out.append(_Ranked(score, matched))
    return sum(len(g.masks) for g in scene.gt)


def _match_objects(
    scene: SceneEval,
    overlaps: _Overlaps,
    config: EvalConfig,
    untyped: list[_Ranked],
    typed: list[_Ranked],
) -> None:
    gts = [gi for gi, gt in enumerate(scene.gt) if gt.masks]
    preds = sorted(
        (pi for pi, pred in enumerate(scene.pred) if pred.masks),
        key=lambda pi: scene.pred[pi].confidence,
        reverse=True,
    )
    for require_type, out in ((False, untyped), (True, typed)):
        matches: dict[tuple[int, ...], int] = {}
        for pi in preds:
            pred = scene.pred[pi]
            best: tuple[int, ...] | None = None
            best_iou = 0.0
            for gi in gts:
                if require_type and scene.gt[gi].moved != pred.moved:
                    continue
                iou = overlaps.object_iou(pi, gi)
                if iou > best_iou and iou > config.iou_threshold:
                    best, best_iou = (gi,), iou
            matched, record = _register(
                matches, best, config.duplicate_match_threshold
            )
            if record:
                out.append(_Ranked(pred.confidence, matched))


def evaluate(
    scenes: Sequence[SceneEval], config: EvalConfig | None = None
) -> Metrics:
    """Computes the SceneDiff metrics over a set of scenes.

    Args:
        scenes: Scenes with ground truth and predictions. A scene without
            ground truth still contributes its predictions as false
            positives, as in the official evaluation.
        config: Matching parameters; defaults to the official ones.

    Returns:
        The pooled metrics.

    Raises:
        ValueError: If a mask is not a boolean array of its scene's size.
    """
    config = config or EvalConfig()
    frame_dets: list[_Ranked] = []
    object_dets: list[_Ranked] = []
    typed_dets: list[_Ranked] = []
    gt_regions = gt_objects = 0
    pixel = np.zeros(3, dtype=np.int64)
    for scene in scenes:
        _check_scene(scene)
        overlaps = _Overlaps(scene)
        pixel += _pixel_counts(scene)
        gt_regions += _match_frames(scene, overlaps, config, frame_dets)
        _match_objects(scene, overlaps, config, object_dets, typed_dets)
        gt_objects += len(scene.gt)
    tp, fp, fn = (int(v) for v in pixel)
    union = tp + fp + fn
    matched = sum(d.matched for d in object_dets)
    return Metrics(
        pixel_iou=tp / union if union else 0.0,
        ap_per_image=_ap(frame_dets, gt_regions),
        ap_per_scene=_ap(object_dets, gt_objects),
        ap_per_scene_typed=_ap(typed_dets, gt_objects),
        tp=matched,
        fp=len(object_dets) - matched,
        fn=gt_objects - matched,
        pixel_tp=tp,
        pixel_fp=fp,
        pixel_fn=fn,
    )
