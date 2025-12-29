"""Numba-accelerated kernels (Phase 2).

This subpackage contains small, *stateless* computational kernels designed to
run in Numba's ``nopython`` mode.

The project intentionally keeps Numba **opt-in**:

* If Numba is not installed, imports still work and the code falls back to
  pure-Python implementations.
* Enable kernels by passing ``--use-numba`` in the runners or by setting
  ``model.use_numba = True``.
"""

from .utils import NUMBA_AVAILABLE, numba_available, njit

__all__ = [
    "NUMBA_AVAILABLE",
    "numba_available",
    "njit",
]
