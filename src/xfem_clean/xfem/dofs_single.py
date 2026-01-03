"""DOF mapping for the single-crack XFEM prototype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

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
    steel:
        Steel dofs (nnode,2) or -1. For bond-slip modeling.
    steel_dof_offset:
        First steel DOF index in global vector. For bond-slip modeling.
    steel_nodes:
        Boolean mask of nodes carrying steel dofs. For bond-slip modeling.
    """

    std: np.ndarray
    H: np.ndarray
    tip: np.ndarray
    ndof: int
    H_nodes: np.ndarray
    tip_nodes: np.ndarray
    steel: np.ndarray = None  # Optional: for bond-slip
    steel_dof_offset: int = -1  # Optional: for bond-slip
    steel_nodes: np.ndarray = None  # Optional: for bond-slip


def build_xfem_dofs(
    nodes: np.ndarray,
    elems: np.ndarray,
    crack: XFEMCrack,
    H_region_ymax: float,
    tip_patch: Optional[Tuple[float, float, float, float]] = None,
    rebar_segs: np.ndarray = None,
    enable_bond_slip: bool = False,
    enable_dolbow_removal: bool = False,
    tol_dolbow: float = 1e-4,
) -> XFEMDofs:
    """Build XFEM dof layout for the current crack configuration.

    Parameters
    ----------
    nodes : np.ndarray
        Node coordinates (nnode, 2)
    elems : np.ndarray
        Element connectivity (nelem, 4)
    crack : XFEMCrack
        Current crack geometry
    H_region_ymax : float
        Maximum y for Heaviside enrichment
    tip_patch : tuple
        (xmin, xmax, ymin, ymax) for tip enrichment region
    rebar_segs : np.ndarray, optional
        Rebar segments array [n_seg, 5]: [n1, n2, L0, cx, cy]
    enable_bond_slip : bool, optional
        If True, allocate steel DOFs for bond-slip modeling

    Returns
    -------
    XFEMDofs
        DOF mapping with optional steel DOFs
    """

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

    # Apply Dolbow ill-conditioning node removal (Dissertation Eq. 4.3)
    if enable_dolbow_removal and crack.active:
        from xfem_clean.numerical_aspects import remove_ill_conditioned_nodes
        H_nodes = remove_ill_conditioned_nodes(
            nodes=nodes,
            elems=elems,
            crack=crack,
            enriched_nodes=H_nodes,
            tol_dolbow=tol_dolbow,
        )

    # If tip_patch is None, use empty patch (no tip enrichment)
    if tip_patch is None:
        tip_nodes = np.zeros(nnode, dtype=bool)
    else:
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

    # Bond-slip: allocate steel DOFs
    steel = None
    steel_dof_offset = -1
    steel_nodes = None

    if enable_bond_slip and rebar_segs is not None and len(rebar_segs) > 0:
        steel_dof_offset = idx  # Steel DOFs start after enrichment DOFs
        steel_nodes = np.zeros(nnode, dtype=bool)
        steel = -np.ones((nnode, 2), dtype=int)

        # Identify nodes used by rebar segments
        for seg in rebar_segs:
            n1 = int(seg[0])
            n2 = int(seg[1])
            steel_nodes[n1] = True
            steel_nodes[n2] = True

        # Allocate steel DOFs for rebar nodes
        for n in np.where(steel_nodes)[0]:
            steel[n, 0] = idx
            steel[n, 1] = idx + 1
            idx += 2

    return XFEMDofs(
        std=std,
        H=H,
        tip=tip,
        ndof=int(idx),
        H_nodes=H_nodes,
        tip_nodes=tip_nodes,
        steel=steel,
        steel_dof_offset=steel_dof_offset,
        steel_nodes=steel_nodes,
    )


def transfer_q_between_dofs(q_old: np.ndarray, dofs_old: XFEMDofs, dofs_new: XFEMDofs) -> np.ndarray:
    """Map a solution vector between two XFEM dof layouts.

    Transfers standard, Heaviside, tip, and (optionally) steel DOFs
    from old to new DOF structure.
    """

    q_new = np.zeros(dofs_new.ndof, dtype=float)
    nnode = dofs_new.std.shape[0]

    # Transfer standard DOFs
    for a in range(nnode):
        for d in range(2):
            io = int(dofs_old.std[a, d])
            inew = int(dofs_new.std[a, d])
            if io >= 0 and inew >= 0 and io < len(q_old):
                q_new[inew] = float(q_old[io])

    # Transfer Heaviside DOFs
    for a in range(nnode):
        for d in range(2):
            io = int(dofs_old.H[a, d])
            inew = int(dofs_new.H[a, d])
            if io >= 0 and inew >= 0 and io < len(q_old):
                q_new[inew] = float(q_old[io])

    # Transfer tip DOFs
    for a in range(nnode):
        for k in range(4):
            for d in range(2):
                io = int(dofs_old.tip[a, k, d])
                inew = int(dofs_new.tip[a, k, d])
                if io >= 0 and inew >= 0 and io < len(q_old):
                    q_new[inew] = float(q_old[io])

    # Transfer steel DOFs (if present in both old and new)
    if (dofs_old.steel is not None and dofs_new.steel is not None):
        for a in range(nnode):
            for d in range(2):
                io = int(dofs_old.steel[a, d])
                inew = int(dofs_new.steel[a, d])
                if io >= 0 and inew >= 0 and io < len(q_old):
                    q_new[inew] = float(q_old[io])

    return q_new
