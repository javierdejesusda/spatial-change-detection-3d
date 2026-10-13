import http.server
import io
import json
import pickle
import threading
import zipfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from rle_helpers import encode_mask

from spatial_change.domain import ChangeType
from spatial_change.evaluation import scenediff
from spatial_change.evaluation.scenediff import (
    SEGMENTS_FILE,
    PairAnnotations,
    change_type_from_flags,
    check_pair,
    find_pair_videos,
    find_video,
    first_mask_shape,
    indoor_pairs,
    list_pairs,
    load_pair_annotations,
    parse_pair_annotations,
    scene_category,
)
from spatial_change.evaluation.scenediff_fetch import fetch_indoor_split

ROOT = "scenediff_benchmark"


def _mask(top: int, left: int) -> np.ndarray:
    mask = np.zeros((6, 4), dtype=bool)
    mask[top : top + 2, left : left + 2] = True
    return mask


def _raw(
    flags: list[tuple[bool, bool]], *, extra: dict[str, object] | None = None
) -> dict[str, object]:
    objects = []
    video1: dict[int, dict[str, dict[str, object]]] = {}
    video2: dict[int, dict[str, dict[str, object]]] = {}
    for idx, (in1, in2) in enumerate(flags):
        meta: dict[str, object] = {
            "label": f"thing_{idx}",
            "in_video1": in1,
            "in_video2": in2,
            "original_obj_idx": idx,
        }
        meta.update(extra or {})
        objects.append(meta)
        if in1:
            video1[idx] = {
                "0": encode_mask(_mask(0, 0)),
                "30": encode_mask(_mask(2, 1)),
            }
        if in2:
            video2[idx] = {"60": encode_mask(_mask(1, 1))}
    return {
        "scene_type": "living_room",
        "objects": objects,
        "video1_objects": video1,
        "video2_objects": video2,
        "comments": "",
        "light_change": False,
    }


@pytest.mark.parametrize(
    ("in1", "in2", "expected"),
    [
        (True, True, ChangeType.MOVED),
        (True, False, ChangeType.REMOVED),
        (False, True, ChangeType.ADDED),
        (False, False, None),
    ],
)
def test_change_type_from_flags(
    in1: bool, in2: bool, expected: ChangeType | None
) -> None:
    assert change_type_from_flags(in1, in2) is expected


def test_parse_builds_typed_objects() -> None:
    ann = parse_pair_annotations(
        "p", _raw([(True, True), (True, False), (False, True)])
    )

    assert isinstance(ann, PairAnnotations)
    assert [o.change_type for o in ann.objects] == [
        ChangeType.MOVED,
        ChangeType.REMOVED,
        ChangeType.ADDED,
    ]
    assert [o.obj_id for o in ann.objects] == [0, 1, 2]
    assert ann.objects[0].label == "thing_0"


def test_parse_keys_frames_by_integer_video_and_frame() -> None:
    ann = parse_pair_annotations("p", _raw([(True, True)]))

    frames = ann.objects[0].frames
    assert sorted(frames[1]) == [0, 30]
    assert sorted(frames[2]) == [60]
    np.testing.assert_array_equal(frames[1][30].decode(), _mask(2, 1))


def test_removed_object_has_no_video2_frames() -> None:
    ann = parse_pair_annotations("p", _raw([(True, False)]))

    assert ann.objects[0].frames[2] == {}


def test_unflagged_objects_are_skipped() -> None:
    ann = parse_pair_annotations("p", _raw([(False, False), (True, True)]))

    assert [o.obj_id for o in ann.objects] == [1]


def test_pair_without_annotated_objects_has_none() -> None:
    assert parse_pair_annotations("p", _raw([])).objects == ()
    assert parse_pair_annotations("p", _raw([(False, False)])).objects == ()


def test_missing_section_fails_loudly() -> None:
    raw = _raw([(True, True)])
    del raw["video2_objects"]

    with pytest.raises(ValueError, match="video2_objects"):
        parse_pair_annotations("p", raw)


def test_flagged_object_without_masks_is_kept_with_empty_frames() -> None:
    raw = _raw([(True, True)])
    raw["video1_objects"] = {}

    ann = parse_pair_annotations("p", raw)

    assert len(ann.objects) == 1
    assert ann.objects[0].frames[1] == {}
    assert sorted(ann.objects[0].frames[2]) == [60]


def test_load_reads_through_the_safe_unpickler(tmp_path: Path) -> None:
    (tmp_path / "segments.pkl").write_bytes(
        pickle.dumps(_raw([(True, False)]), protocol=4)
    )

    ann = load_pair_annotations("p", tmp_path)

    assert ann.objects[0].change_type is ChangeType.REMOVED


def test_load_rejects_a_malicious_pickle(tmp_path: Path) -> None:
    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self) -> tuple[object, tuple[object, ...]]:
            return (marker.write_text, ("x",))

    (tmp_path / "segments.pkl").write_bytes(pickle.dumps(Evil()))

    with pytest.raises(pickle.UnpicklingError, match="forbidden global"):
        load_pair_annotations("p", tmp_path)
    assert not marker.exists()


def test_loader_never_uses_bare_pickle() -> None:
    source = Path(scenediff.__file__).read_text(encoding="utf-8")

    assert "pickle.load" not in source
    assert "safe_load" in source


@pytest.mark.parametrize(
    ("pair", "category"),
    [
        ("living_room_3_living_room_4", "living_room"),
        ("bed_1_bed_2", "bed"),
        ("room_entrance_2_room_entrance_3", "room_entrance"),
        ("P09-20240621-093545_1_P09-20240621-093545_2", "P09-20240621-093545"),
    ],
)
def test_scene_category(pair: str, category: str) -> None:
    assert scene_category(pair) == category


def test_indoor_pairs_keep_only_indoor_categories_of_varied() -> None:
    split = {
        "varied": [
            "living_room_1_living_room_2",
            "street_1_street_2",
            "office_3_office_4",
            "store_1_store_2",
        ],
        "kitchen": ["P01-20240202-110250_1_P01-20240202-110250_2"],
    }

    assert indoor_pairs(split) == [
        "living_room_1_living_room_2",
        "office_3_office_4",
    ]


def test_indoor_pairs_rejects_a_split_without_varied() -> None:
    with pytest.raises(KeyError, match="varied"):
        indoor_pairs({"kitchen": []})


def _split_zip() -> bytes:
    indoor = ["living_room_1_living_room_2", "bedroom_1_bedroom_2"]
    outdoor = ["street_1_street_2"]
    files: dict[str, bytes | str] = {}
    for pair in indoor + outdoor:
        base = f"{ROOT}/data/{pair}"
        files[f"{base}/original_video1.mp4"] = pair + "-v1"
        files[f"{base}/original_video2.mp4"] = pair + "-v2"
        files[f"{base}/segments.pkl"] = pickle.dumps(_raw([(True, True)]))
    split = {"varied": indoor[:1] + outdoor + indoor[1:], "kitchen": []}
    files[f"{ROOT}/splits/val_split.json"] = json.dumps(split)
    files[f"{ROOT}/splits/test_split.json"] = json.dumps(split)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


@pytest.fixture
def served_zip() -> Iterator[str]:
    data = _split_zip()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            start, end = self.headers["Range"].removeprefix("bytes=").split("-")
            lo, hi = int(start), min(int(end), len(data) - 1)
            body = data[lo : hi + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {lo}-{hi}/{len(data)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.01},
        daemon=True,
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/z.zip"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_fetch_indoor_split_downloads_only_the_indoor_pairs(
    served_zip: str, tmp_path: Path
) -> None:
    dirs = fetch_indoor_split("val", tmp_path, url=served_zip)

    assert [d.name for d in dirs] == [
        "living_room_1_living_room_2",
        "bedroom_1_bedroom_2",
    ]
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        d.name for d in dirs
    )
    ann = load_pair_annotations(dirs[0].name, dirs[0])
    assert ann.objects[0].change_type is ChangeType.MOVED


def test_fetch_indoor_split_reports_the_selection_size(
    served_zip: str, tmp_path: Path
) -> None:
    assert len(fetch_indoor_split("test", tmp_path, url=served_zip)) == 2


def test_official_eval_size_swaps_the_stored_size_like_the_reference() -> None:
    # The reference reads the stored (height, width) as (width, height).
    assert scenediff.official_eval_hw(1280, 720, 1024) == (576, 1024)
    assert scenediff.official_eval_hw(8, 4, 4) == (2, 4)


def test_eval_size_comes_from_the_first_annotated_mask() -> None:
    ann = parse_pair_annotations("p", _raw([(True, True)]))

    assert scenediff.eval_size(ann, 3) == (2, 3)


def test_eval_size_is_none_for_an_empty_pair() -> None:
    assert scenediff.eval_size(parse_pair_annotations("p", _raw([])), 3) is None


def test_the_first_mask_shape_is_its_stored_height_and_width() -> None:
    ann = parse_pair_annotations("p", _raw([(True, True)]))

    assert first_mask_shape(ann) == (6, 4)


def test_the_first_mask_shape_is_found_in_the_second_video_too() -> None:
    ann = parse_pair_annotations("p", _raw([(False, True)]))

    assert first_mask_shape(ann) == (6, 4)


def test_a_pair_without_masks_has_no_mask_shape() -> None:
    assert first_mask_shape(parse_pair_annotations("p", _raw([]))) is None


def test_the_pairs_of_a_folder_are_the_subfolders_with_annotations(
    tmp_path: Path,
) -> None:
    for name in ("b", "a"):
        (tmp_path / name).mkdir()
        (tmp_path / name / SEGMENTS_FILE).write_bytes(b"")
    (tmp_path / "without").mkdir()
    (tmp_path / "stray.txt").write_text("", encoding="utf-8")

    assert list_pairs(tmp_path) == ["a", "b"]


def test_the_pairs_of_a_missing_folder_are_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list_pairs(tmp_path / "absent")


@pytest.mark.parametrize(
    "name", ["video1.mp4", "video1.MOV", "VIDEO1.Mp4", "video1.mov"]
)
def test_a_video_is_found_whatever_the_case(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_bytes(b"")
    (tmp_path / "video2.mp4").write_bytes(b"")
    (tmp_path / "video10.mp4").write_bytes(b"")

    assert find_video(tmp_path, "video1") == tmp_path / name


def test_a_missing_video_is_named(tmp_path: Path) -> None:
    (tmp_path / "video1.avi").write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="video1.mp4 or video1.mov"):
        find_video(tmp_path, "video1")


def test_a_folder_that_is_absent_is_named(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="pair folder not found"):
        find_video(tmp_path / "absent", "video1")


def test_two_videos_of_one_kind_are_ambiguous(tmp_path: Path) -> None:
    (tmp_path / "video1.mp4").write_bytes(b"")
    (tmp_path / "video1.mov").write_bytes(b"")

    with pytest.raises(ValueError, match="several video1"):
        find_video(tmp_path, "video1")


def test_the_videos_of_a_pair_are_the_before_and_the_after(
    tmp_path: Path,
) -> None:
    (tmp_path / "video2.MOV").write_bytes(b"")
    (tmp_path / "video1.mp4").write_bytes(b"")

    assert find_pair_videos(tmp_path) == (
        tmp_path / "video1.mp4",
        tmp_path / "video2.MOV",
    )


def test_a_pair_without_its_after_video_is_named(tmp_path: Path) -> None:
    (tmp_path / "video1.mp4").write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="video2.mp4 or video2.mov"):
        find_pair_videos(tmp_path)


@pytest.mark.parametrize("name", ["a", "living_room_3_living_room_4", "A-1_b"])
def test_a_safe_pair_name_passes(name: str) -> None:
    check_pair(name)


@pytest.mark.parametrize(
    "name", ["", "../x", "a/b", "a\\b", "a b", "a.b", "x\n"]
)
def test_an_unsafe_pair_name_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="invalid pair name"):
        check_pair(name)


def test_gt_items_keep_every_stride_th_frame() -> None:
    ann = parse_pair_annotations(
        "p", _raw([(True, True), (True, False), (False, True)])
    )

    items = scenediff.gt_items(ann, 6, 4, stride=30)

    assert [i.moved for i in items] == [True, False, False]
    assert sorted(items[0].masks) == [(1, 0), (1, 1), (2, 2)]
    assert sorted(items[1].masks) == [(1, 0), (1, 1)]
    np.testing.assert_array_equal(items[0].masks[(1, 1)], _mask(2, 1))


def test_gt_items_drop_frames_off_the_stride_but_keep_the_object() -> None:
    ann = parse_pair_annotations("p", _raw([(False, True)]))

    items = scenediff.gt_items(ann, 6, 4, stride=7)

    assert len(items) == 1
    assert items[0].masks == {}


def test_gt_items_resize_with_nearest_sampling() -> None:
    ann = parse_pair_annotations("p", _raw([(True, False)]))

    items = scenediff.gt_items(ann, 12, 8, stride=30)

    expected = np.kron(_mask(0, 0), np.ones((2, 2), bool))
    np.testing.assert_array_equal(items[0].masks[(1, 0)], expected)


def test_missing_object_key_is_a_value_error() -> None:
    raw = _raw([(True, True)])
    del raw["objects"][0]["original_obj_idx"]  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="original_obj_idx"):
        parse_pair_annotations("p", raw)


def test_hostile_annotation_is_rejected_cheaply() -> None:
    raw = _raw([(True, False)])
    raw["video1_objects"] = {  # type: ignore[assignment]
        0: {"0": {"size": [100000, 100000], "counts": [10**10]}}
    }

    with pytest.raises(ValueError, match="size"):
        parse_pair_annotations("p", raw)


def test_gt_items_reject_a_bad_stride_or_grid() -> None:
    ann = parse_pair_annotations("p", _raw([(True, True)]))

    with pytest.raises(ValueError, match="stride"):
        scenediff.gt_items(ann, 6, 4, stride=0)
    with pytest.raises(ValueError, match="grid"):
        scenediff.gt_items(ann, 100_000, 100_000)


def test_gt_type_follows_the_flags_not_the_sampled_masks() -> None:
    # The reference types ground truth from in_video1/in_video2, so a moved
    # object stays moved even if only one video keeps a sampled frame.
    ann = parse_pair_annotations("p", _raw([(True, True)]))

    items = scenediff.gt_items(ann, 6, 4, stride=60)

    assert sorted(items[0].masks) == [(1, 0), (2, 1)]
    one_video = scenediff.gt_items(ann, 6, 4, stride=45)
    assert sorted(one_video[0].masks) == [(1, 0)]
    assert one_video[0].moved
