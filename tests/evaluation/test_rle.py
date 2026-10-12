import numpy as np
import pytest
from rle_helpers import encode_mask, run_lengths

from spatial_change.evaluation.scenediff import RleMask


def _decode(rle: dict[str, object]) -> np.ndarray:
    return RleMask.from_coco(rle).decode()


def test_all_ones_literal() -> None:
    mask = _decode({"size": [2, 3], "counts": "06"})

    assert mask.dtype == np.bool_
    assert mask.shape == (2, 3)
    assert mask.all()


def test_all_zeros_literal() -> None:
    assert not _decode({"size": [2, 3], "counts": "6"}).any()


def test_runs_are_column_major() -> None:
    mask = _decode({"size": [2, 2], "counts": "121"})

    np.testing.assert_array_equal(mask, [[False, True], [True, False]])


def test_later_counts_are_delta_coded() -> None:
    mask = _decode({"size": [2, 2], "counts": "1110"})

    np.testing.assert_array_equal(mask, [[False, False], [True, True]])


def test_negative_delta_literal() -> None:
    # runs [1, 5, 1, 1]: the fourth is stored as 1 - 5 = -4, character "L".
    mask = _decode({"size": [2, 4], "counts": "151L"})

    assert run_lengths(mask) == [1, 5, 1, 1]


def test_multi_character_run_literal() -> None:
    # 16 is "`0": 0x10 sets the sign bit so a second character follows.
    mask = _decode({"size": [4, 5], "counts": "`04"})

    assert mask.shape == (4, 5)
    assert mask.sum() == 4
    assert not mask.reshape(-1, order="F")[:16].any()


def test_round_trip_random_masks() -> None:
    rng = np.random.default_rng(0)
    for shape in ((1, 1), (7, 5), (40, 33), (64, 3)):
        mask = rng.random(shape) > 0.6

        np.testing.assert_array_equal(_decode(encode_mask(mask)), mask)


def test_round_trip_edge_masks() -> None:
    for mask in (
        np.zeros((9, 4), dtype=bool),
        np.ones((9, 4), dtype=bool),
        np.eye(6, dtype=bool),
    ):
        np.testing.assert_array_equal(_decode(encode_mask(mask)), mask)


def test_long_runs_round_trip() -> None:
    mask = np.zeros((300, 300), dtype=bool)
    mask[100:200, 50:250] = True

    np.testing.assert_array_equal(_decode(encode_mask(mask)), mask)


def test_uncompressed_list_counts() -> None:
    mask = _decode({"size": [2, 2], "counts": [1, 2, 1]})

    np.testing.assert_array_equal(mask, [[False, True], [True, False]])


def test_bytes_counts_match_str_counts() -> None:
    mask = _decode({"size": [2, 2], "counts": b"121"})

    np.testing.assert_array_equal(mask, [[False, True], [True, False]])


def test_counts_that_do_not_cover_the_image_are_rejected_on_decode() -> None:
    mask = RleMask.from_coco({"size": [2, 3], "counts": "05"})

    with pytest.raises(ValueError, match="cover"):
        mask.decode()


def test_invalid_character_is_rejected_on_decode() -> None:
    mask = RleMask.from_coco({"size": [2, 2], "counts": "1\x01"})

    with pytest.raises(ValueError, match="counts"):
        mask.decode()


def test_truncated_run_is_rejected_on_decode() -> None:
    mask = RleMask.from_coco({"size": [2, 2], "counts": "1`"})

    with pytest.raises(ValueError, match="counts"):
        mask.decode()


def test_missing_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="size"):
        RleMask.from_coco({"counts": "1"})


def test_oversized_image_is_rejected_eagerly() -> None:
    with pytest.raises(ValueError, match="size"):
        RleMask.from_coco({"size": [100_000, 100_000], "counts": [10**10]})
    with pytest.raises(ValueError, match="size"):
        RleMask(100_000, 100_000, (1,))


def test_run_longer_than_the_image_is_rejected_on_decode() -> None:
    for counts in ([2**70, 0], [-1, 17]):
        mask = RleMask.from_coco({"size": [4, 4], "counts": counts})
        with pytest.raises(ValueError, match="run"):
            mask.decode()


def test_overlong_compressed_run_is_rejected_on_decode() -> None:
    mask = RleMask.from_coco({"size": [4, 4], "counts": "o" * 200 + "0"})

    with pytest.raises(ValueError, match="counts"):
        mask.decode()


def test_size_limits_are_inclusive() -> None:
    RleMask(16384, 1, (16384,))
    with pytest.raises(ValueError, match="size"):
        RleMask(16385, 1, (16385,))
