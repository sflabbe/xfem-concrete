"""Embedded rebar (truss-line) contribution."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import scipy.sparse as sp

# Optional acceleration via Numba
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap
    def prange(*args):
        return range(*args)


def prepare_rebar_segments(nodes: np.ndarray, cover: float):
    """
    Returns a dense array segs[ns,5] with columns:
      [n1, n2, L0, cx, cy]
    Node ids are stored as floats but cast to int in numba kernels.
    """
    ys = nodes[:, 1]
    y_levels = np.unique(np.round(ys, 12))
    y_bar = y_levels[np.argmin(np.abs(y_levels - cover))]

    bar_nodes = np.where(np.isclose(ys, y_bar))[0]
    bar_nodes = bar_nodes[np.argsort(nodes[bar_nodes, 0])]
    segs = []
    for i in range(len(bar_nodes) - 1):
        n1 = int(bar_nodes[i])
        n2 = int(bar_nodes[i + 1])
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        L0 = math.sqrt(dx * dx + dy * dy)
        if L0 < 1e-12:
            continue
        cx = dx / L0
        cy = dy / L0
        segs.append((float(n1), float(n2), float(L0), float(cx), float(cy)))
    if len(segs) == 0:
        return np.zeros((0,5), dtype=float)
    return np.array(segs, dtype=float)




def steel_bilinear_sigma_tangent(eps: float, E: float, fy: float, Eh: float, fu: float):
    eps_y = fy / E
    if abs(eps) <= eps_y:
        return E * eps, E
    sign = 1.0 if eps >= 0.0 else -1.0
    s = sign * (fy + Eh * (abs(eps) - eps_y))
    if abs(s) > fu:
        return sign * fu, 1e-9 * E
    return s, Eh



@njit(cache=True, parallel=True)
def _rebar_triplets_numba(u, segs, A_total, E, fy, fu, Eh):
    ndof_u = u.shape[0]
    ns = segs.shape[0]
    f = np.zeros(ndof_u, dtype=np.float64)
    rows = np.empty(ns*16, dtype=np.int64)
    cols = np.empty(ns*16, dtype=np.int64)
    data = np.empty(ns*16, dtype=np.float64)

    for i in prange(ns):
        n1 = int(segs[i,0]); n2 = int(segs[i,1])
        L0 = segs[i,2]; cx = segs[i,3]; cy = segs[i,4]

        u1x = u[2*n1]; u1y = u[2*n1+1]
        u2x = u[2*n2]; u2y = u[2*n2+1]
        dux = u2x - u1x
        duy = u2y - u1y
        axial = dux*cx + duy*cy
        eps = axial / L0

        eps_y = fy / E
        if abs(eps) <= eps_y:
            sig = E*eps
            Et = E
        else:
            sign = 1.0 if eps >= 0.0 else -1.0
            sig = sign * (fy + Eh*(abs(eps)-eps_y))
            if abs(sig) > fu:
                sig = sign * fu
                Et = 1e-9 * E
            else:
                Et = Eh

        N = sig * A_total
        fx = N*cx
        fy_ = N*cy

        # internal force vector
        f[2*n1]   -= fx
        f[2*n1+1] -= fy_
        f[2*n2]   += fx
        f[2*n2+1] += fy_

        k = A_total * Et / L0
        # 4x4 axial stiffness in global (u1x,u1y,u2x,u2y)
        tt00 = cx*cx; tt01 = cx*cy; tt11 = cy*cy
        # block [[tt, -tt],[-tt, tt]]
        k00 = k*tt00; k01 = k*tt01; k11 = k*tt11

        dofs = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1], dtype=np.int64)

        base = i*16
        idx = 0
        for a in range(4):
            for b in range(4):
                rows[base+idx] = dofs[a]
                cols[base+idx] = dofs[b]
                idx += 1

        # fill values in same order
        # matrix:
        # [ k*tt, -k*tt;
        #  -k*tt, k*tt ]
        # explicit
        vals = np.empty(16, dtype=np.float64)
        # row0
        vals[0] =  k00; vals[1] =  k01; vals[2] = -k00; vals[3] = -k01
        # row1
        vals[4] =  k01; vals[5] =  k11; vals[6] = -k01; vals[7] = -k11
        # row2
        vals[8] = -k00; vals[9] = -k01; vals[10]=  k00; vals[11]=  k01
        # row3
        vals[12]= -k01; vals[13]= -k11; vals[14]=  k01; vals[15]=  k11

        for j in range(16):
            data[base+j] = vals[j]

    return f, rows, cols, data

def rebar_contrib(nodes: np.ndarray, segs: np.ndarray, u: np.ndarray, A_total: float, E: float, fy: float, fu: float, Eh: float):
    """
    Rebar as truss-line contribution (internal force + tangent).
    Uses numba (if available) to generate sparse triplets.
    """
    ndof_u = u.shape[0]
    if segs.shape[0] == 0:
        return np.zeros(ndof_u, dtype=float), sp.csr_matrix((ndof_u, ndof_u))

    if NUMBA_AVAILABLE:
        f, rows, cols, data = _rebar_triplets_numba(u.astype(np.float64), segs.astype(np.float64),
                                                    float(A_total), float(E), float(fy), float(fu), float(Eh))
        K = sp.coo_matrix((data, (rows, cols)), shape=(ndof_u, ndof_u)).tocsr()
        return np.asarray(f, dtype=float), K

    # fallback pure python
    f = np.zeros(ndof_u, dtype=float)
    rows, cols, data = [], [], []
    A = float(A_total)
    for (n1f, n2f, L0, cx, cy) in segs:
        n1 = int(n1f); n2 = int(n2f)
        u1 = np.array([u[2 * n1], u[2 * n1 + 1]])
        u2 = np.array([u[2 * n2], u[2 * n2 + 1]])
        du = u2 - u1
        axial = float(du[0] * cx + du[1] * cy)
        eps = axial / L0
        sig, Et = steel_bilinear_sigma_tangent(eps, E, fy, Eh, fu)
        N = sig * A
        fx = N * cx
        fy_ = N * cy
        dofs = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
        f[dofs[0]] -= fx; f[dofs[1]] -= fy_; f[dofs[2]] += fx; f[dofs[3]] += fy_
        k = A * Et / L0
        tt = np.array([[cx * cx, cx * cy], [cx * cy, cy * cy]], dtype=float)
        k_local = k * np.block([[tt, -tt], [-tt, tt]])
        for a in range(4):
            for b in range(4):
                rows.append(dofs[a]); cols.append(dofs[b]); data.append(k_local[a, b])
    K = sp.coo_matrix((data, (rows, cols)), shape=(ndof_u, ndof_u)).tocsr()
    return f, K



# -----------------------------
# Mesh + FE assembly
# -----------------------------
