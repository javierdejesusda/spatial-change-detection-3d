"""SceneDiff ground truth: RLE masks, change types, and pair annotations.

The annotation format and the evaluation protocol follow the SceneDiff
benchmark (https://github.com/yuqunw/scene_diff, MIT license, copyright
2025 Yuqun Wu), commit 62dcbd285fcfe984bb3db55dc01bdcd1a0055714,
``scripts/evaluate_multiview.py``. The code here is an independent
reimplementation; see ``NOTICE`` next to this module.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spatial_change.domain import ChangeType
from spatial_change.evaluation._masks import resize_nearest
from spatial_change.evaluation.metrics import GtItem as GtMetricItem
from spatial_change.evaluation.safe_pickle import safe_load

_CHAR_OFFSET = 48
_PAYLOAD_BITS = 5
_PAYLOAD_MASK = 0x1F
_CONTINUE_BIT = 0x20
_SIGN_BIT = 0x10
_FIRST_DELTA_RUN = 3
_MAX_RUN_CHARS = 7

SEGMENTS_FILE = "segments.pkl"
"""Name of the annotations in a pair folder."""

BEFORE_VIDEO = "video1"
"""Stem of the BEFORE video in a pair folder."""

AFTER_VIDEO = "video2"
"""Stem of the AFTER video in a pair folder."""

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov"})
"""Extensions of the videos of a pair, in lower case."""

_PAIR_RE = re.compile(r"[A-Za-z0-9_-]+")

MAX_SIDE = 16384
MAX_PIXELS = 1 << 28
"""Largest accepted mask side and pixel count, bounding what a hostile
annotation can make the decoder allocate."""


def _parse_counts(counts: str) -> list[int]:
    """Decodes a compressed COCO RLE string into run lengths."""
    runs: list[int] = []
    pos, end = 0, len(counts)
    while pos < end:
        value, shift, more = 0, 0, True
        while more:
            if pos >= end:
                raise ValueError("truncated run in RLE counts")
            c = ord(counts[pos]) - _CHAR_OFFSET
            if not 0 <= c < 64:
                raise ValueError(f"invalid character in RLE counts: {counts!r}")
            if shift >= _MAX_RUN_CHARS:
                raise ValueError("run too long in RLE counts")
            value |= (c & _PAYLOAD_MASK) << (_PAYLOAD_BITS * shift)
            more = bool(c & _CONTINUE_BIT)
            pos += 1
            shift += 1
            if not more and c & _SIGN_BIT:
                value -= 1 << (_PAYLOAD_BITS * shift)
        if len(runs) >= _FIRST_DELTA_RUN:
            value += runs[-2]
        runs.append(value)
    return runs


def _check_size(height: int, width: int, what: str) -> None:
    if not (
        0 < height <= MAX_SIDE
        and 0 < width <= MAX_SIDE
        and height * width <= MAX_PIXELS
    ):
        raise ValueError(
            f"{what} {height}x{width} is outside the accepted range "
            f"(sides up to {MAX_SIDE}, up to {MAX_PIXELS} pixels)"
        )


@dataclass(frozen=True, eq=False)
class RleMask:
    """A binary mask in COCO run-length encoding, decoded on demand.

    Runs alternate between 0 and 1, start with a (possibly empty) run of 0,
    and walk the image in column-major order. The size is validated when the
    mask is built; the counts are parsed and validated in ``decode``, so a
    pair that is only sampled never pays for the frames it skips.

    Attributes:
        height: Mask height in pixels.
        width: Mask width in pixels.
        counts: Compressed COCO counts string, or uncompressed run lengths.
    """

    height: int
    width: int
    counts: str | tuple[int, ...]

    def __post_init__(self) -> None:
        """Validates the size.

        Raises:
            ValueError: If the size is outside ``MAX_SIDE`` and
                ``MAX_PIXELS``.
        """
        _check_size(self.height, self.width, "RLE size")

    @classmethod
    def from_coco(cls, rle: Mapping[str, Any]) -> "RleMask":
        """Builds a mask from a COCO RLE dict.

        Args:
            rle: Dict with ``size = [height, width]`` and ``counts`` as a
                compressed string (str or bytes) or an uncompressed list of
                run lengths.

        Returns:
            The mask.

        Raises:
            ValueError: If a key is missing or the size is out of range.
        """
        if "size" not in rle or "counts" not in rle:
            raise ValueError("RLE needs both size and counts")
        height, width = (int(v) for v in rle["size"])
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("ascii")
        if not isinstance(counts, str):
            counts = tuple(int(run) for run in counts)
        return cls(height, width, counts)

    def decode(self) -> NDArray[np.bool_]:
        """Parses the counts and expands the runs.

        Returns:
            Boolean array of shape ``(height, width)``.

        Raises:
            ValueError: If the counts are malformed, a run is negative or
                longer than the image, or the runs do not cover the image.
        """
        runs = (
            _parse_counts(self.counts)
            if isinstance(self.counts, str)
            else list(self.counts)
        )
        total = self.height * self.width
        if any(not 0 <= run <= total for run in runs):
            raise ValueError("RLE run is negative or longer than the image")
        if sum(runs) != total:
            raise ValueError("RLE runs do not cover the image")
        values = np.arange(len(runs)) % 2 == 1
        flat = np.repeat(values, np.asarray(runs, dtype=np.intp))
        return flat.reshape(self.width, self.height).T.copy()


def change_type_from_flags(
    in_video1: bool, in_video2: bool
) -> ChangeType | None:
    """Derives the change type of an object from its presence flags.

    Args:
        in_video1: Whether the object is in the first video.
        in_video2: Whether the object is in the second video.

    Returns:
        ``MOVED`` when in both, ``REMOVED`` when only in the first,
        ``ADDED`` when only in the second, ``None`` when in neither.
    """
    if in_video1 and in_video2:
        return ChangeType.MOVED
    if in_video1:
        return ChangeType.REMOVED
    if in_video2:
        return ChangeType.ADDED
    return None


@dataclass(frozen=True, eq=False)
class GtObject:
    """One annotated changed object.

    Attributes:
        obj_id: The ``original_obj_idx`` of the object.
        label: Free-text object name.
        change_type: Added, removed, or moved.
        frames: Masks as ``frames[video][frame_index]`` with ``video`` 1 or
            2 and ``frame_index`` the index in the original video.
    """

    obj_id: int
    label: str
    change_type: ChangeType
    frames: Mapping[int, Mapping[int, RleMask]]


@dataclass(frozen=True, eq=False)
class PairAnnotations:
    """Ground truth of one video pair.

    Attributes:
        pair: Pair name.
        objects: Objects present in at least one video, in file order. The
            list is empty for the pairs without annotations.
    """

    pair: str
    objects: tuple[GtObject, ...]


def _object_frames(
    section: Mapping[int, Mapping[str, Mapping[str, Any]]], obj_id: int
) -> dict[int, RleMask]:
    raw = section.get(obj_id, {})
    return {int(k): RleMask.from_coco(v) for k, v in raw.items()}


def parse_pair_annotations(
    pair: str, data: Mapping[str, Any]
) -> PairAnnotations:
    """Builds typed annotations from an unpickled ``segments.pkl``.

    The official evaluation keeps pairs without annotated objects (their
    predictions all count as false positives) and counts every object that
    is flagged present in a video, even one without masks, so this parser
    keeps both. Objects present in neither video are skipped.

    Args:
        pair: Pair name.
        data: The unpickled annotations.

    Returns:
        The annotations.

    Raises:
        ValueError: If ``objects``, ``video1_objects`` or ``video2_objects``
            is missing, or a mask is malformed.
    """
    for key in ("objects", "video1_objects", "video2_objects"):
        if key not in data:
            raise ValueError(f"{pair}: annotations lack {key!r}")
    sections = {1: data["video1_objects"], 2: data["video2_objects"]}
    objects: list[GtObject] = []
    for meta in data["objects"]:
        if "original_obj_idx" not in meta:
            raise ValueError(f"{pair}: object lacks 'original_obj_idx'")
        obj_id = int(meta["original_obj_idx"])
        change_type = change_type_from_flags(
            bool(meta.get("in_video1", False)),
            bool(meta.get("in_video2", False)),
        )
        if change_type is None:
            continue
        frames = {
            1: _object_frames(sections[1], obj_id)
            if change_type is not ChangeType.ADDED
            else {},
            2: _object_frames(sections[2], obj_id)
            if change_type is not ChangeType.REMOVED
            else {},
        }
        objects.append(
            GtObject(
                obj_id=obj_id,
                label=str(meta.get("label", "")),
                change_type=change_type,
                frames=frames,
            )
        )
    return PairAnnotations(pair=pair, objects=tuple(objects))


def load_pair_annotations(pair: str, pair_dir: Path) -> PairAnnotations:
    """Reads ``segments.pkl`` of a fetched pair through the safe unpickler.

    Args:
        pair: Pair name.
        pair_dir: Directory holding ``segments.pkl``.

    Returns:
        The annotations.

    Raises:
        pickle.UnpicklingError: If the file references any global.
        ValueError: If the annotations are malformed.
    """
    return parse_pair_annotations(pair, safe_load(pair_dir / SEGMENTS_FILE))


def first_mask_shape(ann: PairAnnotations) -> tuple[int, int] | None:
    """Gives the stored size of the first annotated mask of a pair.

    Args:
        ann: The pair annotations.

    Returns:
        ``(height, width)`` as stored, looking in the first video before the
        second, or ``None`` if the pair has no mask.
    """
    for video in (1, 2):
        for obj in ann.objects:
            for mask in obj.frames[video].values():
                return mask.height, mask.width
    return None


def check_pair(pair: str) -> None:
    """Checks that a pair name is safe to use as a folder name.

    Args:
        pair: Pair directory name.

    Raises:
        ValueError: If the name is empty or has a character other than a
            letter, a digit, an underscore, or a hyphen.
    """
    if _PAIR_RE.fullmatch(pair) is None:
        raise ValueError(f"invalid pair name: {pair!r}")


def list_pairs(data_dir: Path) -> list[str]:
    """Names the pairs in a folder: its subfolders that hold annotations.

    Args:
        data_dir: Folder with one subfolder per pair.

    Returns:
        The pair names, sorted.

    Raises:
        FileNotFoundError: If the folder does not exist.
    """
    return sorted(
        folder.name
        for folder in data_dir.iterdir()
        if (folder / SEGMENTS_FILE).is_file()
    )


def find_video(pair_dir: Path, stem: str) -> Path:
    """Finds ``<stem>.mp4`` or ``<stem>.mov`` in a pair folder, any case.

    Args:
        pair_dir: The pair folder.
        stem: ``BEFORE_VIDEO`` or ``AFTER_VIDEO``.

    Returns:
        The video file.

    Raises:
        FileNotFoundError: If the folder or the video is missing.
        ValueError: If several files qualify.
    """
    if not pair_dir.is_dir():
        raise FileNotFoundError(f"pair folder not found: {pair_dir}")
    found = sorted(
        path
        for path in pair_dir.iterdir()
        if path.is_file()
        and path.stem.lower() == stem
        and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not found:
        raise FileNotFoundError(f"no {stem}.mp4 or {stem}.mov in {pair_dir}")
    if len(found) > 1:
        raise ValueError(f"several {stem} files in {pair_dir}")
    return found[0]


def find_pair_videos(pair_dir: Path) -> tuple[Path, Path]:
    """Finds the BEFORE and the AFTER video of a pair folder.

    Args:
        pair_dir: The pair folder.

    Returns:
        The BEFORE video and the AFTER video.

    Raises:
        FileNotFoundError: If the folder or either video is missing.
        ValueError: If several files qualify as either video.
    """
    return (
        find_video(pair_dir, BEFORE_VIDEO),
        find_video(pair_dir, AFTER_VIDEO),
    )


INDOOR_CATEGORIES = frozenset(
    {
        "bathroom",
        "bed",
        "bedroom",
        "billiards_room",
        "closet",
        "dinning_hall",
        "game_room",
        "hallway",
        "kitchen",
        "laundry_room",
        "living_room",
        "livingroom",
        "lounge",
        "office",
        "recreation_room",
        "room_entrance",
        "study",
        "workspace",
    }
)
"""Scene categories of the ``varied`` set counted as indoor rooms.

On the pinned split files this selects 28 validation and 92 test pairs.
"""

_CATEGORY_RE = re.compile(r"(.+?)_\d+_")


def scene_category(pair: str) -> str:
    """Returns the scene category of a pair name.

    Args:
        pair: Pair name such as ``living_room_3_living_room_4``.

    Returns:
        The text before the first ``_<number>_``.

    Raises:
        ValueError: If the name does not follow ``<category>_<n>_...``.
    """
    match = _CATEGORY_RE.match(pair)
    if match is None:
        raise ValueError(f"cannot read a scene category from {pair!r}")
    return match[1]


def indoor_pairs(split: Mapping[str, Sequence[str]]) -> list[str]:
    """Selects the indoor pairs of the ``varied`` set from a split file.

    Args:
        split: Split mapping from set name (``varied``, ``kitchen``) to pair
            names, as returned by ``load_split``.

    Returns:
        The ``varied`` pairs whose category is in ``INDOOR_CATEGORIES``, in
        split order.

    Raises:
        KeyError: If the split has no ``varied`` set.
    """
    if "varied" not in split:
        raise KeyError("split has no varied set")
    return [
        pair
        for pair in split["varied"]
        if scene_category(pair) in INDOOR_CATEGORIES
    ]


DEFAULT_FRAME_STRIDE = 30
DEFAULT_MAX_LENGTH = 1024


def official_eval_hw(
    stored_height: int, stored_width: int, max_length: int
) -> tuple[int, int]:
    """Computes the evaluation resolution the way the reference does.

    The reference reads the stored RLE size ``[height, width]`` as
    ``(width, height)`` and scales it so the longer side is ``max_length``.
    For portrait video this yields a landscape grid, which squeezes the
    masks. The same grid is applied to ground truth and predictions, so the
    quirk is reproduced to keep scores comparable with published ones.

    Args:
        stored_height: First number of the RLE ``size``.
        stored_width: Second number of the RLE ``size``.
        max_length: Length of the longer side of the grid.

    Returns:
        The evaluation grid as ``(height, width)``.
    """
    src_w, src_h = stored_height, stored_width
    scale = max_length / max(src_w, src_h)
    return max(1, round(src_h * scale)), max(1, round(src_w * scale))


def eval_size(
    ann: PairAnnotations, max_length: int = DEFAULT_MAX_LENGTH
) -> tuple[int, int] | None:
    """Returns the evaluation grid of a pair from its first annotated mask.

    Args:
        ann: The pair annotations.
        max_length: Length of the longer side of the grid.

    Returns:
        ``(height, width)``, or ``None`` if the pair has no mask.
    """
    shape = first_mask_shape(ann)
    if shape is None:
        return None
    return official_eval_hw(*shape, max_length)


def gt_items(
    ann: PairAnnotations,
    height: int,
    width: int,
    stride: int = DEFAULT_FRAME_STRIDE,
) -> list[GtMetricItem]:
    """Turns annotations into ground truth for the metrics.

    Only frames whose index is a multiple of ``stride`` are kept and are
    re-indexed as ``index // stride``; masks are resized to the evaluation
    grid with nearest sampling. Objects without any kept frame stay in the
    list, because the official totals count them.

    Args:
        ann: The pair annotations.
        height: Evaluation grid height.
        width: Evaluation grid width.
        stride: Frame sampling stride of the evaluation.

    Returns:
        One item per annotated object, in file order. An object is moved
        when its flags say so, as in the reference, even if only one video
        keeps a sampled frame.

    Raises:
        ValueError: If ``stride`` is below 1 or the grid is out of range.
    """
    if stride < 1:
        raise ValueError("stride must be at least 1")
    _check_size(height, width, "evaluation grid")
    items = []
    for obj in ann.objects:
        masks = {
            (video, index // stride): resize_nearest(
                mask.decode(), height, width
            )
            for video in (1, 2)
            for index, mask in obj.frames[video].items()
            if index % stride == 0
        }
        items.append(
            GtMetricItem(
                obj.obj_id, masks, moved=obj.change_type is ChangeType.MOVED
            )
        )
    return items
