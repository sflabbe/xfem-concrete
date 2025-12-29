"""Enriched strain-displacement operator for the single-crack prototype."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from xfem_clean.xfem.geometry import XFEMCrack, branch_F_and_grad
from xfem_clean.xfem.dofs_single import XFEMDofs


def build_B_enriched(
    x: float,
    y: float,
    N: np.ndarray,
    dN_dx: np.ndarray,
    dN_dy: np.ndarray,
    xe: np.ndarray,
    conn: np.ndarray,
    dofs: XFEMDofs,
    crack: XFEMCrack,
    tip_enr_radius: float,
    crack_active: bool,
    heaviside_side: float,
    tip_x: float,
    tip_y: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build enriched B and the corresponding global dof list."""

    edofs = []
    B_cols = []

    H_i = np.array([crack.H(float(xe[a, 0]), float(xe[a, 1])) for a in range(4)], dtype=float)

    F_i = np.zeros((4, 4), dtype=float)
    for a in range(4):
        Fi, _, _ = branch_F_and_grad(float(xe[a, 0]), float(xe[a, 1]), tip_x, tip_y)
        F_i[a, :] = Fi

    Fg, dFdx_g, dFdy_g = branch_F_and_grad(float(x), float(y), tip_x, tip_y)

    G_y = crack.behind_tip(float(x), float(y)) if crack_active else 0.0
    r_tip = math.sqrt((float(x) - tip_x) ** 2 + (float(y) - tip_y) ** 2)
    G_tip = 1.0 if (crack_active and (r_tip <= float(tip_enr_radius))) else 0.0

    # base dofs
    for a in range(4):
        n = int(conn[a])
        B_cols.append(np.array([dN_dx[a], 0.0, dN_dy[a]], dtype=float))
        edofs.append(int(dofs.std[n, 0]))
        B_cols.append(np.array([0.0, dN_dy[a], dN_dx[a]], dtype=float))
        edofs.append(int(dofs.std[n, 1]))

    # Heaviside enrichment
    for a in range(4):
        n = int(conn[a])
        if dofs.H[n, 0] < 0:
            continue
        fac = (float(heaviside_side) - H_i[a]) * float(G_y)
        B_cols.append(np.array([fac * dN_dx[a], 0.0, fac * dN_dy[a]], dtype=float))
        edofs.append(int(dofs.H[n, 0]))
        B_cols.append(np.array([0.0, fac * dN_dy[a], fac * dN_dx[a]], dtype=float))
        edofs.append(int(dofs.H[n, 1]))

    # Tip enrichment
    if G_tip > 0.0:
        for a in range(4):
            n = int(conn[a])
            if dofs.tip[n, 0, 0] < 0:
                continue
            for k in range(4):
                fac = float(Fg[k] - F_i[a, k])
                dphi_dx = dN_dx[a] * fac + float(N[a]) * float(dFdx_g[k])
                dphi_dy = dN_dy[a] * fac + float(N[a]) * float(dFdy_g[k])

                B_cols.append(np.array([dphi_dx, 0.0, dphi_dy], dtype=float) * float(G_tip))
                edofs.append(int(dofs.tip[n, k, 0]))
                B_cols.append(np.array([0.0, dphi_dy, dphi_dx], dtype=float) * float(G_tip))
                edofs.append(int(dofs.tip[n, k, 1]))

    B = np.stack(B_cols, axis=1)
    return B, np.array(edofs, dtype=int)
