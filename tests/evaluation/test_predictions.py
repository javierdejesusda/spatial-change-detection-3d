from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from spatial_change.evaluation.metrics import PredItem
from spatial_change.evaluation.predictions import (
    Predictions,
    load_predictions,
    save_predictions,
)

HEIGHT, WIDTH = 7, 11


def mask(seed: int) -> NDArray[np.bool_]:
    return np.random.default_rng(seed).random((HEIGHT, WIDTH)) < 0.4


def item(obj_id: int, keys: list[tuple[int, int]], seed: int) -> PredItem:
    return PredItem(
        obj_id,
        {key: mask(seed + n) for n, key in enumerate(keys)},
        {key: 0.1 * (n + 1) + seed / 1000 for n, key in enumerate(keys)},
    )


def predictions(
    items: list[PredItem], uncertain: frozenset[int] = frozenset()
) -> Predictions:
    return Predictions(HEIGHT, WIDTH, items, uncertain)


def same(a: PredItem, b: PredItem) -> None:
    assert a.obj_id == b.obj_id
    assert list(a.masks) == list(b.masks)
    assert a.scores == b.scores
    for key, value in a.masks.items():
        assert b.masks[key].dtype == np.bool_
        np.testing.assert_array_equal(b.masks[key], value)


def test_predictions_survive_a_round_trip_exactly(tmp_path: Path) -> None:
    items = [
        item(3, [(1, 0), (1, 2), (2, 5)], seed=1),
        item(0, [(2, 1)], seed=20),
        item(9, [], seed=0),
    ]
    path = tmp_path / "pred.npz"

    save_predictions(path, predictions(items, frozenset({0, 9})))
    loaded = load_predictions(path)

    assert (loaded.height, loaded.width) == (HEIGHT, WIDTH)
    assert loaded.uncertain == {0, 9}
    assert len(loaded.items) == len(items)
    for want, got in zip(items, loaded.items, strict=True):
        same(want, got)


def test_an_empty_list_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "pred.npz"

    save_predictions(path, predictions([]))

    assert load_predictions(path) == predictions([])


def test_an_uncertain_flag_for_an_object_that_is_not_there_is_an_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="uncertain"):
        save_predictions(
            tmp_path / "pred.npz",
            predictions([item(1, [(1, 0)], seed=1)], frozenset({2})),
        )


def test_a_mask_of_another_size_is_an_error(tmp_path: Path) -> None:
    wrong = PredItem(1, {(1, 0): np.zeros((3, 3), dtype=bool)}, {(1, 0): 0.5})

    with pytest.raises(ValueError, match="mask"):
        save_predictions(tmp_path / "pred.npz", predictions([wrong]))


def test_saving_again_replaces_the_file_whole_or_not_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pred.npz"
    save_predictions(path, predictions([item(1, [(1, 0)], seed=1)]))
    save_predictions(path, predictions([item(2, [(2, 0)], seed=2)]))
    assert [p.name for p in tmp_path.iterdir()] == ["pred.npz"]
    assert load_predictions(path).items[0].obj_id == 2
    before = path.read_bytes()

    def broken(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", broken)
    with pytest.raises(OSError, match="disk full"):
        save_predictions(path, predictions([item(3, [(2, 0)], seed=3)]))

    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["pred.npz"]


def test_a_truncated_file_is_a_value_error(tmp_path: Path) -> None:
    path = tmp_path / "pred.npz"
    save_predictions(path, predictions([item(1, [(1, 0), (2, 0)], seed=1)]))
    data = path.read_bytes()

    for keep in (len(data) // 2, len(data) - 30, 10, 0):
        path.write_bytes(data[:keep])
        with pytest.raises(ValueError, match="pred.npz"):
            load_predictions(path)


@pytest.mark.parametrize("kind", ["bytes", "array", "other archive"])
def test_a_file_that_is_not_a_predictions_file_is_a_value_error(
    tmp_path: Path, kind: str
) -> None:
    path = tmp_path / "pred.npz"
    if kind == "bytes":
        path.write_bytes(b"not an archive at all")
    elif kind == "array":
        with path.open("wb") as handle:
            np.save(handle, np.arange(3))
    else:
        np.savez_compressed(path, other=np.arange(3))

    with pytest.raises(ValueError, match="pred.npz"):
        load_predictions(path)


BREAKS = {
    "counts": lambda a: a + 1,
    "ids": lambda a: np.concatenate([a, a]),
    "uncertain": lambda a: np.concatenate([a, a]),
    "grid": lambda a: np.concatenate([a, a]),
    "bits": lambda a: a[:-1],
}


@pytest.mark.parametrize("field", sorted(BREAKS))
def test_an_archive_that_disagrees_with_itself_is_a_value_error(
    tmp_path: Path, field: str
) -> None:
    path = tmp_path / "pred.npz"
    save_predictions(path, predictions([item(1, [(1, 0)], seed=1)]))
    with np.load(path, allow_pickle=False) as archive:
        fields = {name: archive[name] for name in archive.files}
    fields[field] = BREAKS[field](fields[field])
    np.savez_compressed(path, **fields)

    with pytest.raises(ValueError, match="pred.npz"):
        load_predictions(path)


def test_a_missing_file_is_a_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_predictions(tmp_path / "absent.npz")
