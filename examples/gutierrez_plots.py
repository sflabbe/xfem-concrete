
"""
Gutierrez-style plotting helpers for the 3-point bending demo.

Goal: produce plots similar to Gutierrez thesis figures:
- Load vs mid-displacement (P-u)
- Final crack pattern over deformed beam outline + rebar line

These utilities are intentionally lightweight and depend only on numpy + matplotlib
(and optionally scipy for KDTree acceleration).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _as_xy(u: np.ndarray, nnode: int) -> np.ndarray:
    """Standard nodal displacements (nnode,2) from global vector `u`.

    Convention in this repo: standard dofs are the first 2*nnode entries.
    """
    if u.ndim != 1:
        u = u.reshape(-1)
    m = 2 * nnode
    if u.shape[0] < m:
        raise ValueError(f"u has size {u.shape[0]} but need at least {m} for standard dofs")
    return u[:m].reshape((nnode, 2)).astype(float)

def _boundary_node_ids(nodes: np.ndarray, L: float, H: float) -> dict[str, np.ndarray]:
    """Return indices for boundary node sets (structured mesh friendly)."""
    x = nodes[:, 0]
    y = nodes[:, 1]
    eps = 1e-12
    bottom = np.where(np.isclose(y, 0.0, atol=eps))[0]
    top    = np.where(np.isclose(y, H,   atol=eps))[0]
    left   = np.where(np.isclose(x, 0.0, atol=eps))[0]
    right  = np.where(np.isclose(x, L,   atol=eps))[0]
    bottom = bottom[np.argsort(x[bottom])]
    top    = top[np.argsort(x[top])]
    left   = left[np.argsort(y[left])]
    right  = right[np.argsort(y[right])]
    return {"bottom": bottom, "top": top, "left": left, "right": right}

def _polyline_from_boundary(nodes_def: np.ndarray, bnd: dict[str, np.ndarray]) -> np.ndarray:
    """Closed polyline around beam boundary in deformed coordinates."""
    bottom = nodes_def[bnd["bottom"]]
    right  = nodes_def[bnd["right"]]
    top    = nodes_def[bnd["top"]][::-1]
    left   = nodes_def[bnd["left"]][::-1]
    poly = np.vstack([bottom, right[1:], top[1:], left[1:], bottom[:1]])
    return poly

def _nearest_node_deformer(nodes: np.ndarray, U: np.ndarray, scale: float):
    """Return a callable p -> p + scale*U(nearest node)."""
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(nodes)
        def deform(p: np.ndarray) -> np.ndarray:
            _, idx = tree.query(p, k=1)
            return p + scale * U[int(idx)]
        return deform
    except Exception:
        # fallback: brute-force (fine for ~O(2e3) nodes)
        def deform(p: np.ndarray) -> np.ndarray:
            d2 = np.sum((nodes - p[None, :])**2, axis=1)
            idx = int(np.argmin(d2))
            return p + scale * U[idx]
        return deform

def plot_pu_gutierrez(results,
                      out_png: str,
                      *,
                      label: str = "Model") -> None:
    """Load vs mid-displacement plot (Gutierrez Fig-style).

    Accepts either:
      - list/tuple of dict rows (multi-crack runner)
      - numpy array with columns:
        [step, u, P, M, kappa, R, tip_x, tip_y, ang_deg, crack_active] (single-crack runner)
    """
    import matplotlib.pyplot as plt

    if results is None:
        raise ValueError("results is None")

    # --- extract u [mm] and P [kN] ---
    if isinstance(results, np.ndarray):
        if results.size == 0:
            raise ValueError("results array is empty")
        arr = np.asarray(results, dtype=float)
        u_mm = arr[:, 1] * 1e3
        P_kN = arr[:, 2] / 1e3
        # crack event: when crack_active switches 0->1
        if arr.shape[1] >= 10:
            act = arr[:, 9].astype(int)
            ev = np.where(np.diff(act, prepend=act[0]) > 0)[0]
        else:
            ev = np.array([], dtype=int)
    else:
        if len(results) == 0:
            raise ValueError("results is empty")
        u_mm = np.array([float(r.get("u", 0.0))*1e3 for r in results], dtype=float)
        P_kN = np.array([float(r.get("P", 0.0))/1e3 for r in results], dtype=float)
        ncr = np.array([int(r.get("ncr", 0)) for r in results], dtype=int)
        ev = np.where(np.diff(ncr, prepend=ncr[0]) > 0)[0]

    plt.rcParams.update({
        "font.family": "serif",
        "axes.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot(u_mm, P_kN, "-", linewidth=1.2, color="black", label=label)

    if ev.size > 0:
        ax.plot(u_mm[ev], P_kN[ev], "o", markersize=4, markerfacecolor="white",
                markeredgecolor="black", markeredgewidth=0.9, linestyle="None",
                label="Crack init/grow")

    ax.set_xlabel("Mid-displacement [mm]")
    ax.set_ylabel("Load [kN]")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.tick_params(direction="in", length=4, width=0.8, top=True, right=True)
    ax.legend(frameon=True, framealpha=1.0, edgecolor="black", fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_crack_pattern_gutierrez(nodes: np.ndarray,
                                 u: np.ndarray,
                                 cracks: Sequence[object],
                                 model: object,
                                 out_png: str,
                                 *,
                                 scale: float | None = None,
                                 show_ids: bool = True,
                                 show_axes: bool = False) -> None:
    """Final crack pattern over deformed outline + rebar line (Gutierrez Fig-style)."""
    import matplotlib.pyplot as plt

    nnode = int(nodes.shape[0])
    U = _as_xy(u, nnode)

    # choose a visual scale so small deflections remain visible
    if scale is None:
        umax = float(np.max(np.abs(U[:, 1]))) + 1e-15
        target = 0.05 * float(getattr(model, "L", float(np.max(nodes[:, 0]) - np.min(nodes[:, 0]))))
        scale = float(np.clip(target / umax, 1.0, 25.0))

    nodes_def = nodes + scale * U

    L = float(getattr(model, "L", float(np.max(nodes[:, 0]) - np.min(nodes[:, 0]))))
    H = float(getattr(model, "H", float(np.max(nodes[:, 1]) - np.min(nodes[:, 1]))))
    cover = float(getattr(model, "cover", 0.0))

    bnd = _boundary_node_ids(nodes, L, H)
    poly = _polyline_from_boundary(nodes_def, bnd)

    deform_pt = _nearest_node_deformer(nodes, U, scale)

    plt.rcParams.update({"font.family": "serif", "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(9.0, 2.4))

    # Outline (deformed)
    ax.plot(poly[:, 0], poly[:, 1], color="black", linewidth=1.0)

    # Rebar line at y ~ cover (if nodes exist there)
    if cover > 0.0:
        ys = nodes[:, 1]
        y_levels = np.unique(np.round(ys, 12))
        y_bar = float(y_levels[np.argmin(np.abs(y_levels - cover))])
        bar_nodes = np.where(np.isclose(ys, y_bar, atol=1e-12))[0]
        bar_nodes = bar_nodes[np.argsort(nodes[bar_nodes, 0])]
        if bar_nodes.size >= 2:
            pts = nodes_def[bar_nodes]
            ax.plot(pts[:, 0], pts[:, 1], color="blue", linewidth=1.0)

    # Cracks: draw shorter ones lighter, longer ones darker
    crack_entries = []
    for i, c in enumerate(cracks, start=1):
        if not getattr(c, "active", True):
            continue
        try:
            x0 = float(getattr(c, "x0")); y0 = float(getattr(c, "y0"))
            xt = float(getattr(c, "tip_x")); yt = float(getattr(c, "tip_y"))
        except Exception:
            continue
        Lc = float(np.hypot(xt - x0, yt - y0))
        crack_entries.append((Lc, i, np.array([x0, y0], float), np.array([xt, yt], float)))

    crack_entries.sort(key=lambda t: t[0])  # shortest first
    Lmax = max([c[0] for c in crack_entries], default=1.0)

    for (Lc, i, p0, pt) in crack_entries:
        p0d = deform_pt(p0)
        ptd = deform_pt(pt)

        # alpha by relative length
        rel = float(Lc / (Lmax + 1e-15))
        a_main = 0.35 + 0.55 * rel

        for lw, a in [(2.8, 0.12*a_main), (1.6, 0.35*a_main), (0.9, a_main)]:
            ax.plot([p0d[0], ptd[0]], [p0d[1], ptd[1]],
                    color="black", linewidth=lw, alpha=a, solid_capstyle="round")

        if show_ids and i <= 20:
            dx = 0.008 * L
            ax.text(ptd[0] + dx, ptd[1], f"C{i}", fontsize=9, color="black",
                    va="center", ha="left")

    ax.set_aspect("equal", adjustable="box")
    if show_axes:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.tick_params(direction="in", length=4, width=0.8, top=True, right=True)
        ax.set_title(f"Computed crack pattern (scale={scale:.1f})", fontsize=10)
    else:
        ax.axis("off")

    fig.tight_layout(pad=0.2)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
