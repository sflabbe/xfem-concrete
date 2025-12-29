"""Q4 kinematics and simple mappings for axis-aligned structured meshes."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from xfem_clean.fem.q4 import q4_shape


def element_x_y(xi: float, eta: float, xe: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Map parent coordinates to global (x,y) plus Q4 shape data."""
    N, dN_dxi, dN_deta = q4_shape(float(xi), float(eta))
    x = float(np.dot(N, xe[:, 0]))
    y = float(np.dot(N, xe[:, 1]))
    return x, y, N, dN_dxi, dN_deta


def element_dN_dxdy(dN_dxi: np.ndarray, dN_deta: np.ndarray, xe: np.ndarray):
    """Compute global shape function gradients and detJ."""
    J = np.zeros((2, 2), dtype=float)
    J[0, 0] = float(np.dot(dN_dxi, xe[:, 0]))
    J[0, 1] = float(np.dot(dN_dxi, xe[:, 1]))
    J[1, 0] = float(np.dot(dN_deta, xe[:, 0]))
    J[1, 1] = float(np.dot(dN_deta, xe[:, 1]))
    detJ = float(np.linalg.det(J))
    invJ = np.linalg.inv(J)
    dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
    dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
    return dN_dx, dN_dy, detJ


def element_bounds(xe: np.ndarray):
    xmin = float(np.min(xe[:, 0])); xmax = float(np.max(xe[:, 0]))
    ymin = float(np.min(xe[:, 1])); ymax = float(np.max(xe[:, 1]))
    return xmin, xmax, ymin, ymax


def element_local_xi_at_x(x0: float, xe: np.ndarray) -> float:
    x_coords = xe[:, 0]
    xmin = float(np.min(x_coords)); xmax = float(np.max(x_coords))
    xc = 0.5 * (xmin + xmax); hx = 0.5 * (xmax - xmin)
    if hx <= 0:
        return 0.0
    xi = (float(x0) - xc) / hx
    return max(-1.0, min(1.0, float(xi)))


def element_local_eta_at_y(y0: float, xe: np.ndarray) -> float:
    y_coords = xe[:, 1]
    ymin = float(np.min(y_coords)); ymax = float(np.max(y_coords))
    yc = 0.5 * (ymin + ymax); hy = 0.5 * (ymax - ymin)
    if hy <= 0:
        return 0.0
    eta = (float(y0) - yc) / hy
    return max(-1.0, min(1.0, float(eta)))


def map_global_to_parent_Q4(x: float, y: float, xe: np.ndarray) -> Tuple[float, float]:
    """Inverse mapping (x,y)->(xi,eta) for axis-aligned structured Q4 elements."""
    xi = element_local_xi_at_x(float(x), xe)
    eta = element_local_eta_at_y(float(y), xe)
    return xi, eta
