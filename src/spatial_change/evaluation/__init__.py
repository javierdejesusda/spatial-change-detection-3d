"""Benchmark data access and evaluation: SceneDiff loader, projection, metrics.

Importing this package loads none of torch, transformers, open3d or cv2.
"""

from spatial_change.evaluation._masks import resize_nearest
from spatial_change.evaluation.metrics import (
    EvalConfig,
    GtItem,
    Metrics,
    PredItem,
    SceneEval,
    evaluate,
    voc_ap,
)
from spatial_change.evaluation.projection import (
    ChangeObject3D,
    ProjectionConfig,
    project_to_pred_items,
)
from spatial_change.evaluation.scenediff import (
    DEFAULT_FRAME_STRIDE,
    DEFAULT_MAX_LENGTH,
    INDOOR_CATEGORIES,
    GtObject,
    PairAnnotations,
    RleMask,
    change_type_from_flags,
    eval_size,
    gt_items,
    indoor_pairs,
    load_pair_annotations,
    official_eval_hw,
    parse_pair_annotations,
    scene_category,
)
from spatial_change.evaluation.scenediff_fetch import fetch_indoor_split

__all__ = [
    "ChangeObject3D",
    "DEFAULT_FRAME_STRIDE",
    "DEFAULT_MAX_LENGTH",
    "EvalConfig",
    "GtItem",
    "GtObject",
    "INDOOR_CATEGORIES",
    "Metrics",
    "PairAnnotations",
    "PredItem",
    "ProjectionConfig",
    "RleMask",
    "SceneEval",
    "change_type_from_flags",
    "eval_size",
    "evaluate",
    "fetch_indoor_split",
    "gt_items",
    "indoor_pairs",
    "load_pair_annotations",
    "official_eval_hw",
    "parse_pair_annotations",
    "project_to_pred_items",
    "resize_nearest",
    "scene_category",
    "voc_ap",
]
