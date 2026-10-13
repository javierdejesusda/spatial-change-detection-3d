from pathlib import Path

import pytest

from spatial_change.evaluation._files import atomic_path


def test_a_finished_block_puts_the_file_in_place(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"

    with atomic_path(target) as temporary:
        assert temporary.parent == tmp_path
        assert temporary != target
        temporary.write_text("new", encoding="utf-8")
        assert not target.exists()

    assert target.read_text(encoding="utf-8") == "new"
    assert [p.name for p in tmp_path.iterdir()] == ["out.txt"]


def test_a_finished_block_replaces_an_old_file(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    target.write_text("old", encoding="utf-8")

    with atomic_path(target) as temporary:
        temporary.write_text("new", encoding="utf-8")

    assert target.read_text(encoding="utf-8") == "new"


def test_a_failed_block_keeps_the_old_file_and_leaves_nothing_behind(
    tmp_path: Path,
) -> None:
    target = tmp_path / "out.txt"
    target.write_text("old", encoding="utf-8")

    with pytest.raises(RuntimeError, match="boom"), atomic_path(target) as temp:
        temp.write_text("half", encoding="utf-8")
        raise RuntimeError("boom")

    assert target.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["out.txt"]


def test_a_failed_block_with_no_old_file_leaves_no_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "out.txt"

    with pytest.raises(KeyboardInterrupt), atomic_path(target) as temporary:
        temporary.write_bytes(b"half")
        raise KeyboardInterrupt

    assert list(tmp_path.iterdir()) == []
