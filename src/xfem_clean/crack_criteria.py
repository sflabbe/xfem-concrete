"""Crack initiation / propagation helpers (Rankine + nonlocal)."""

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np

def principal_max_2d(sig: np.ndarray) -> float:
    sxx, syy, txy = sig
    S2 = np.array([[sxx, txy], [txy, syy]], dtype=float)
    vals = np.linalg.eigvalsh(S2)
    # include sigma_zz=0 as possible maximum in plane stress
    return float(max(vals[1], 0.0))



def principal_min_2d(sig: np.ndarray) -> float:
    sxx, syy, txy = sig
    S2 = np.array([[sxx, txy], [txy, syy]], dtype=float)
    vals = np.linalg.eigvalsh(S2)
    return float(min(vals[0], 0.0))



def candidate_points_bottom(nodes: np.ndarray, L: float, margin: float, spacing: float) -> np.ndarray:
    # y=0 edge
    y0 = float(np.min(nodes[:,1]))
    xs = np.unique(nodes[np.isclose(nodes[:,1], y0), 0])
    xs = xs[(xs >= margin) & (xs <= L - margin)]
    if len(xs) == 0:
        return np.array([[L/2, y0]], dtype=float)
    # subsample by spacing
    cps = []
    last = -1e18
    for x in xs:
        if x - last >= spacing:
            cps.append([float(x), y0])
            last = x
    return np.asarray(cps, dtype=float)

def candidate_points_bottom_edge_midpoints(nodes: np.ndarray,
                                          elems: np.ndarray,
                                          L: float,
                                          margin: float,
                                          window: Tuple[float, float] = (0.0, 1.0),
                                          windows: Optional[Sequence[Tuple[float, float]]] = None,
                                          spacing: Optional[float] = None) -> np.ndarray:
    """
    Candidate points (Gutiérrez-style) for flexural cracks:
    - Only on the bottom edge (tension side in 3-pt bending).
    - Use *element bottom-edge midpoints* (more robust than nodal x-locations).
    - Optionally restrict to a dominant-crack window in x (fractions of L).
    """
    y0 = float(np.min(nodes[:, 1]))
    x_min = float(margin)
    x_max = float(L - margin)
    w0 = float(window[0]) * L
    w1 = float(window[1]) * L

    if windows is None:
        windows = [window]
    win_abs = [(float(a)*L, float(b)*L) for (a,b) in windows]

    cps = []
    for conn in elems:
        xe = nodes[conn, :]
        # bottom-row elements only
        if float(np.min(xe[:, 1])) > y0 + 1e-10:
            continue
        # pick the two nodes with smallest y as bottom edge
        idx = np.argsort(xe[:, 1])[:2]
        p = 0.5 * (xe[idx[0], :] + xe[idx[1], :])
        x = float(p[0])
        if x < x_min or x > x_max:
            continue
        in_any = False
        for (wa, wb) in win_abs:
            if x >= wa and x <= wb:
                in_any = True
                break
        if not in_any:
            continue
        cps.append([x, y0])

    if len(cps) == 0:
        return np.array([[L / 2.0, y0]], dtype=float)

    cps = np.asarray(cps, dtype=float)
    # unique by x (tolerant)
    cps = cps[np.argsort(cps[:, 0])]
    if spacing is not None and spacing > 0.0:
        out = []
        last = -1e18
        for x, y in cps:
            if x - last >= spacing - 1e-12:
                out.append([float(x), float(y)])
                last = x
        cps = np.asarray(out, dtype=float)
    return cps



def nonlocal_sigma1_at_x(aux_gp_pos: np.ndarray, aux_sigma1: np.ndarray, x0: float, y_max: float, rho: float) -> float:
    # use only gp close to bottom
    m = aux_gp_pos[:,1] <= y_max + 1e-12
    if not np.any(m):
        return 0.0
    dx = aux_gp_pos[m,0] - x0
    w = np.exp(-(dx*dx)/(rho*rho))
    num = float(np.sum(w * aux_sigma1[m]))
    den = float(np.sum(w)) + 1e-15
    return num/den



def nonlocal_bar_stress(aux_gp_pos: np.ndarray,
                        aux_gp_sig: np.ndarray,
                        x0: float,
                        y0: float,
                        rho: float,
                        y_max: Optional[float] = None,
                        r_cut: Optional[float] = None) -> np.ndarray:
    """Wells-type nonlocal stress average with Gaussian weights."""
    if aux_gp_pos.size == 0:
        return np.zeros(3, dtype=float)
    m = np.ones(aux_gp_pos.shape[0], dtype=bool)
    if y_max is not None:
        m &= (aux_gp_pos[:, 1] <= float(y_max) + 1e-12)
    dx = aux_gp_pos[m, 0] - float(x0)
    dy = aux_gp_pos[m, 1] - float(y0)
    r2 = dx * dx + dy * dy
    if r_cut is not None:
        m2 = r2 <= float(r_cut) ** 2
        if not np.any(m2):
            return np.zeros(3, dtype=float)
        dx = dx[m2]; dy = dy[m2]; r2 = r2[m2]
        sig = aux_gp_sig[m][m2]
    else:
        sig = aux_gp_sig[m]
    w = np.exp(-r2 / max(1e-18, float(rho) ** 2))
    den = float(np.sum(w)) + 1e-15
    sigbar = (w[:, None] * sig).sum(axis=0) / den
    return sigbar.astype(float)


def principal_max_dir(sig: np.ndarray) -> Tuple[float, np.ndarray]:
    """Return (sigma1>=0, unit eigenvector of sigma1) in plane stress."""
    sxx, syy, txy = map(float, sig)
    S = np.array([[sxx, txy], [txy, syy]], dtype=float)
    vals, vecs = np.linalg.eigh(S)
    # pick max eigenvalue (index 1)
    lam = float(vals[1])
    v = vecs[:, 1].astype(float)
    nrm = float(np.linalg.norm(v)) + 1e-15
    v = v / nrm
    return max(lam, 0.0), v

# -----------------------------
