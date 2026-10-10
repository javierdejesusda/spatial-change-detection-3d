"""Unpickling restricted to builtin container and scalar types.

A pickle downloaded from the internet can execute arbitrary code during
``pickle.load`` through ``GLOBAL`` and ``REDUCE`` opcodes. The benchmark
annotations only need builtin types (dict, list, str, int, float, bool,
None), which pickle builds without resolving any global. Forbidding every
global therefore loads the data while making code execution impossible.
"""

import io
import pickle
from pathlib import Path
from typing import Any


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that refuses to resolve any global."""

    def find_class(self, module: str, name: str) -> Any:
        """Always rejects the requested global.

        Args:
            module: Module name the pickle asks to import.
            name: Attribute name the pickle asks to resolve.

        Raises:
            pickle.UnpicklingError: Always, naming ``module.name``.
        """
        raise pickle.UnpicklingError(
            f"forbidden global in pickle: {module}.{name}"
        )


def safe_loads(data: bytes) -> Any:
    """Deserializes bytes while forbidding every global.

    Args:
        data: Pickle bytes.

    Returns:
        The deserialized object, built from builtin types only.

    Raises:
        pickle.UnpicklingError: If the pickle references any global.
    """
    return RestrictedUnpickler(io.BytesIO(data)).load()


def safe_load(path: Path) -> Any:
    """Deserializes a pickle file while forbidding every global.

    Args:
        path: Path of the pickle file.

    Returns:
        The deserialized object, built from builtin types only.

    Raises:
        pickle.UnpicklingError: If the pickle references any global.
    """
    return safe_loads(path.read_bytes())
