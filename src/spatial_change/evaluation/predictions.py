"""Predicted masks on disk, so a pair can be scored again without a run.

A file is a compressed ``.npz`` archive of plain arrays, read with pickling
off. Every mask of every object is flattened into one bit-packed array, and
small arrays say which object, video, and frame each one belongs to, and
which objects came from changes the run could not classify.
"""

import zipfile
import zlib
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from spatial_change.evaluation._files import atomic_path
from spatial_change.evaluation.metrics import FrameKey, PredItem

_FIELDS = (
    "grid",
    "ids",
    "counts",
    "uncertain",
    "video",
    "frame",
    "score",
    "bits",
)


class Predictions(NamedTuple):
    """The objects a run predicted, on the grid their masks were drawn on.

    Attributes:
        height: Height of every mask.
        width: Width of every mask.
        items: The objects, uncertain ones included. An uncertain change
            with both boxes is drawn as a moved object, and one with a
            single box (demoted by a gate of the matcher) in its one epoch,
            as an added or removed object.
        uncertain: Ids of the objects that came from uncertain changes.
    """

    height: int
    width: int
    items: list[PredItem]
    uncertain: frozenset[int]


def save_predictions(path: Path, predictions: Predictions) -> None:
    """Writes predicted objects and the grid of their masks.

    The archive is written next to ``path`` and moved over it, so a reader
    never sees half a file and a failed write keeps the old one.

    Args:
        path: The file to write.
        predictions: The objects; their order and their masks' order are
            kept.

    Raises:
        ValueError: If a mask does not have the shape ``(height, width)``,
            or an uncertain id names no object.
    """
    height, width, items, uncertain = predictions
    stray = uncertain - {item.obj_id for item in items}
    if stray:
        raise ValueError(f"uncertain ids without an object: {sorted(stray)}")
    keys: list[FrameKey] = []
    scores: list[float] = []
    masks: list[NDArray[np.bool_]] = []
    for item in items:
        for key, mask in item.masks.items():
            if mask.shape != (height, width):
                raise ValueError(
                    f"object {item.obj_id} has a {mask.shape} mask, "
                    f"expected {(height, width)}"
                )
            keys.append(key)
            scores.append(item.scores[key])
            masks.append(mask)
    bits = (
        np.packbits(np.stack(masks), axis=None)
        if masks
        else np.zeros(0, dtype=np.uint8)
    )
    arrays: dict[str, Any] = {
        "grid": np.array([height, width], dtype=np.int64),
        "ids": np.array([i.obj_id for i in items], dtype=np.int64),
        "counts": np.array([len(i.masks) for i in items], dtype=np.int64),
        "uncertain": np.array(
            [i.obj_id in uncertain for i in items], dtype=np.bool_
        ),
        "video": np.array([v for v, _ in keys], dtype=np.int64),
        "frame": np.array([f for _, f in keys], dtype=np.int64),
        "score": np.array(scores, dtype=np.float64),
        "bits": bits,
    }
    with atomic_path(path) as temporary, temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def load_predictions(path: Path) -> Predictions:
    """Reads what ``save_predictions`` wrote.

    Args:
        path: The file.

    Returns:
        The objects, the grid of their masks, and which are uncertain.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is damaged, cut short, or not a predictions
            file.
    """
    try:
        with np.load(path, allow_pickle=False) as archive:
            fields = {name: archive[name] for name in _FIELDS}
        height, width = (int(n) for n in fields["grid"])
        ids, counts = fields["ids"], fields["counts"]
        flags = fields["uncertain"]
        video, frame, score = fields["video"], fields["frame"], fields["score"]
        total = int(counts.sum())
        pixels = total * height * width
        if (
            height < 1
            or width < 1
            or not len(ids) == len(counts) == len(flags)
            or (counts < 0).any()
            or not len(video) == len(frame) == len(score) == total
            or fields["bits"].size != -(-pixels // 8)
        ):
            raise ValueError("the arrays disagree with each other")
        masks = np.unpackbits(fields["bits"])[:pixels].reshape(
            total, height, width
        )
        items = []
        start = 0
        for obj_id, count in zip(ids.tolist(), counts.tolist(), strict=True):
            keys = [
                (int(video[i]), int(frame[i]))
                for i in range(start, start + count)
            ]
            items.append(
                PredItem(
                    int(obj_id),
                    {
                        k: masks[start + n].astype(np.bool_)
                        for n, k in enumerate(keys)
                    },
                    {k: float(score[start + n]) for n, k in enumerate(keys)},
                )
            )
            start += count
        uncertain = frozenset(int(i) for i in ids[flags])
        return Predictions(height, width, items, uncertain)
    except FileNotFoundError:
        raise
    except (
        OSError,
        EOFError,
        KeyError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
        zlib.error,
    ) as error:
        raise ValueError(
            f"{path} is not a readable predictions file: {error}"
        ) from error
