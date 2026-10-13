import pytest

from spatial_change.diagnostics.errors import first_line


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("CUDA out of memory\nmore text"), "CUDA out of memory"),
        (ValueError("  padded  \n"), "padded"),
        (KeyError("missing"), "'missing'"),
        (ValueError(), "ValueError"),
        (ValueError("  \n "), "ValueError"),
    ],
)
def test_an_error_is_named_by_the_first_line_of_its_message(
    error: Exception, expected: str
) -> None:
    assert first_line(error) == expected
