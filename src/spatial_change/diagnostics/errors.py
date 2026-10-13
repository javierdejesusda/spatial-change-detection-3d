"""Short, one-line words for errors."""


def first_line(error: BaseException) -> str:
    """Words an error as one line.

    Args:
        error: The error.

    Returns:
        The first line of its message, or the name of its type if the
        message is empty.
    """
    text = str(error).strip()
    return text.splitlines()[0] if text else type(error).__name__
