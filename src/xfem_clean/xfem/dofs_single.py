"""DOF mapping for the single-crack XFEM prototype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from xfem_clean.xfem.geometry import XFEMCrack
from xfem_clean.xfem.q4_utils import element_bounds
from xfem_clean.xfem.geometry import clip_segment_to_bbox


@dataclass
class XFEMDofs:
    """Degree-of-freedom mapping for a single straight crack.

    Attributes
    ----------
    std:
        Standard dofs (nnode,2).
    H:
        Heaviside dofs (nnode,2) or -1.
    tip:
        Tip dofs (nnode,4,2) or -1.
    ndof:
        Total number of dofs.
    H_nodes:
        Boolean mask of nodes carrying Heaviside dofs.
    tip_nodes:
        Boolean mask of nodes carrying tip dofs.
    """

    std: np.ndarray
    H: np.ndarray
    tip: np.ndarray
    ndof: int
    H_nodes: np.ndarray
    tip_nodes: np.ndarray


def build_xfem_dofs(
    nodes: np.ndarray,
    elems: np.ndarray,
    crack: XFEMCrack,
    H_region_ymax: float,
    tip_patch: Tuple[float, float, float, float],
) -> XFEMDofs:
    """Build XFEM dof layout for the current crack configuration."""

    nnode = nodes.shape[0]
    std = np.arange(2 * nnode, dtype=int).reshape(nnode, 2)

    xs = nodes[:, 0]
    ys = nodes[:, 1]
    H_nodes = np.zeros(nnode, dtype=bool)
    tip_nodes = np.zeros(nnode, dtype=bool)

    yH = float(min(H_region_ymax, crack.tip_y if crack.active else 0.0))

    if crack.active and yH > 0.0:
        p0 = crack.p0()
        pt = crack.pt()
        for e in range(elems.shape[0]):
            en = elems[e]
            xe = nodes[en]
            xmin, xmax, ymin, ymax = element_bounds(xe)
            if ymin > yH + 1e-12:
                continue
            seg = clip_segment_to_bbox(p0, pt, xmin, xmax, ymin, ymax)
            if seg is None:
                continue
            for a in en:
                if float(nodes[a, 1]) <= yH + 1e-12:
                    H_nodes[int(a)] = True

    xmin, xmax, ymin, ymax = tip_patch
    tip_nodes = (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)

    H = -np.ones((nnode, 2), dtype=int)
    tip = -np.ones((nnode, 4, 2), dtype=int)

    idx = 2 * nnode
    for n in np.where(H_nodes)[0]:
        H[n, 0] = idx
        H[n, 1] = idx + 1
        idx += 2

    for n in np.where(tip_nodes)[0]:
        for k in range(4):
            tip[n, k, 0] = idx
            tip[n, k, 1] = idx + 1
            idx += 2

    return XFEMDofs(std=std, H=H, tip=tip, ndof=int(idx), H_nodes=H_nodes, tip_nodes=tip_nodes)


def transfer_q_between_dofs(q_old: np.ndarray, dofs_old: XFEMDofs, dofs_new: XFEMDofs) -> np.ndarray:
    """Map a solution vector between two XFEM dof layouts."""

    q_new = np.zeros(dofs_new.ndof, dtype=float)
    nnode = dofs_new.std.shape[0]

    for a in range(nnode):
        for d in range(2):
            io = int(dofs_old.std[a, d])
            inew = int(dofs_new.std[a, d])
            if io >= 0 and inew >= 0 and io < len(q_old):
                q_new[inew] = float(q_old[io])

    for a in range(nnode):
        for d in range(2):
            io = int(dofs_old.H[a, d])
            inew = int(dofs_new.H[a, d])
            if io >= 0 and inew >= 0 and io < len(q_old):
                q_new[inew] = float(q_old[io])

    for a in range(nnode):
        for k in range(4):
            for d in range(2):
                io = int(dofs_old.tip[a, k, d])
                inew = int(dofs_new.tip[a, k, d])
                if io >= 0 and inew >= 0 and io < len(q_old):
                    q_new[inew] = float(q_old[io])

    return q_new
