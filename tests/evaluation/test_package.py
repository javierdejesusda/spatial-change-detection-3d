import subprocess
import sys

import spatial_change.evaluation as evaluation


def test_package_exports_the_public_names() -> None:
    assert evaluation.__all__ == sorted(evaluation.__all__)
    assert set(evaluation.__all__) == {
        "DEFAULT_FRAME_STRIDE",
        "DEFAULT_MAX_LENGTH",
        "INDOOR_CATEGORIES",
        "ChangeObject3D",
        "EvalConfig",
        "GtItem",
        "GtObject",
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
    }
    for name in evaluation.__all__:
        assert hasattr(evaluation, name)


def test_importing_the_package_does_not_load_the_heavy_stacks() -> None:
    code = (
        "import sys\n"
        "import spatial_change.evaluation\n"
        "heavy = ('torch', 'transformers', 'open3d', 'cv2')\n"
        "loaded = [m for m in heavy if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )

    subprocess.run([sys.executable, "-I", "-c", code], check=True)
