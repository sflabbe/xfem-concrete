"""Utilities for optional Numba acceleration.

The repository must remain usable without Numba. This module provides:

* ``NUMBA_AVAILABLE``: whether ``numba`` can be imported.
* ``njit`` / ``prange`` shims: if Numba is missing, they degrade to no-ops.

This lets the rest of the code import kernels unconditionally.
"""

from __future__ import annotations

from typing import Any, Callable


try:  # pragma: no cover
    import numba as _nb

    NUMBA_AVAILABLE: bool = True
    prange = _nb.prange

    def njit(*args: Any, **kwargs: Any) -> Callable:
        return _nb.njit(*args, **kwargs)

except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def prange(*args: Any, **kwargs: Any):  # type: ignore
        # Fallback: behave like range.
        return range(*args, **kwargs)

    def njit(*args: Any, **kwargs: Any) -> Callable:  # type: ignore
        # Fallback decorator: returns the function unchanged.
        def _decorator(fn: Callable) -> Callable:
            return fn

        # Support both @njit and @njit(...)
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]  # type: ignore
        return _decorator


def numba_available() -> bool:
    return bool(NUMBA_AVAILABLE)
