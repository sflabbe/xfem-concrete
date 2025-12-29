"""Q4 shape functions (bilinear quadrilateral)."""

from __future__ import annotations
import numpy as np

def q4_shape(xi: float, eta: float):
    # N1..N4 (counter-clockwise)
    N = 0.25 * np.array(
        [(1 - xi) * (1 - eta),
         (1 + xi) * (1 - eta),
         (1 + xi) * (1 + eta),
         (1 - xi) * (1 + eta)],
        dtype=float,
    )
    dN_dxi = 0.25 * np.array(
        [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)], dtype=float
    )
    dN_deta = 0.25 * np.array(
        [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)], dtype=float
    )
    return N, dN_dxi, dN_deta


