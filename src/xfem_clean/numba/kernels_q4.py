"""Small Q4 (bilinear quad) kernels.

These are intentionally tiny and stateless so they can be used inside later
Numba-compiled assembly loops (Phase 3).
"""

from __future__ import annotations

import numpy as np

from xfem_clean.numba.utils import njit


@njit(cache=True)
def q4_shape_numba(xi: float, eta: float):
    """Return Q4 shape functions and their derivatives.

    Parameters
    ----------
    xi, eta : float
        Natural coordinates in [-1, 1].

    Returns
    -------
    N : (4,) float
    dN_dxi : (4,) float
    dN_deta : (4,) float
    """
    # Shape functions
    N0 = 0.25 * (1.0 - xi) * (1.0 - eta)
    N1 = 0.25 * (1.0 + xi) * (1.0 - eta)
    N2 = 0.25 * (1.0 + xi) * (1.0 + eta)
    N3 = 0.25 * (1.0 - xi) * (1.0 + eta)

    # Derivatives
    dN0_dxi = -0.25 * (1.0 - eta)
    dN1_dxi = +0.25 * (1.0 - eta)
    dN2_dxi = +0.25 * (1.0 + eta)
    dN3_dxi = -0.25 * (1.0 + eta)

    dN0_deta = -0.25 * (1.0 - xi)
    dN1_deta = -0.25 * (1.0 + xi)
    dN2_deta = +0.25 * (1.0 + xi)
    dN3_deta = +0.25 * (1.0 - xi)

    N = np.array((N0, N1, N2, N3), dtype=np.float64)
    dN_dxi = np.array((dN0_dxi, dN1_dxi, dN2_dxi, dN3_dxi), dtype=np.float64)
    dN_deta = np.array((dN0_deta, dN1_deta, dN2_deta, dN3_deta), dtype=np.float64)
    return N, dN_dxi, dN_deta
