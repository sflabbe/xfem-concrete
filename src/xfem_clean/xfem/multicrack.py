"""Multi-crack (Heaviside + cohesive) XFEM extension.

This module is a direct refactor of the multi-crack section that used to live in
`xfem_xfem.py`. The intent is to keep numerical behavior identical while
isolating the optional multi-crack workflow.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Union, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from xfem_clean.cohesive_laws import CohesiveLaw, CohesiveState, cohesive_update
from xfem_clean.numba.kernels_cohesive import pack_cohesive_law_params, cohesive_update_values_numba
from xfem_clean.numba.kernels_bulk import pack_bulk_params
from xfem_clean.crack_criteria import principal_max_dir, nonlocal_bar_stress
from xfem_clean.fem.q4 import q4_shape
from xfem_clean.fem.mesh import structured_quad_mesh
from xfem_clean.fem.bcs import apply_dirichlet
from xfem_clean.rebar import prepare_rebar_segments, rebar_contrib
from xfem_clean.junction import detect_crack_coalescence, arrest_secondary_crack_at_junction

from xfem_clean.xfem.geometry import XFEMCrack, clip_segment_to_bbox
from xfem_clean.xfem.material import plane_stress_C
from xfem_clean.xfem.model import XFEMModel
from xfem_clean.xfem.q4_utils import map_global_to_parent_Q4
from xfem_clean.xfem.utils import find_nearest_node
from xfem_clean.xfem.state_arrays import CohesiveStateArrays, CohesiveStatePatch, BulkStateArrays, BulkStatePatch
from xfem_clean.material_point import MaterialPoint, mp_default
from xfem_clean.numba.kernels_bulk import (
    elastic_integrate_plane_stress_numba,
    dp_integrate_plane_stress_numba,
    cdp_integrate_plane_stress_numba,
)
from xfem_clean.constitutive import LinearElasticPlaneStress
# Multi-crack (Heaviside + cohesive) extension
# ==========================================================

@dataclass
class MultiXFEMDofs:
    """DOF mapping for multiple Heaviside cracks.

    - Standard dofs are always present (u_x, u_y at every node).
    - For each crack k, additional Heaviside dofs (a_x, a_y) exist only
      at nodes whose support is cut by the current crack segment.

    Notes
    -----
    * Tip enrichment is intentionally disabled for robustness.
    * Dofs are rebuilt whenever a crack is initiated or propagated.
    * Bond-slip support: steel DOFs allocated for rebar nodes (FASE D)
    """
    std: np.ndarray  # (nnode, 2)
    H: list          # list of (nnode, 2) arrays
    H_nodes: list    # list of (nnode,) bool arrays
    ndof: int

    # Bond-slip DOFs (FASE D)
    steel: Optional[np.ndarray] = None  # (nnode, 2) steel DOF map (-1 if no steel)
    steel_dof_offset: int = -1  # Offset where steel DOFs start
    steel_nodes: Optional[np.ndarray] = None  # (nnode,) bool: node has steel


def _node_xy(nodes, idx: int) -> tuple[float, float]:
    """Return node coordinates for both supported node containers.

    The codebase uses `nodes` either as:
      1) numpy array of shape (nnode, 2) with columns [x, y]
      2) list of node-like objects with `.x` and `.y` attributes
    """
    n = nodes[idx]
    if hasattr(n, "x"):
        return float(n.x), float(n.y)
    # numpy row / sequence
    return float(n[0]), float(n[1])


def _node_y(nodes, idx: int) -> float:
    return _node_xy(nodes, idx)[1]


def _node_x(nodes, idx: int) -> float:
    return _node_xy(nodes, idx)[0]


def _nodes_y(nodes) -> np.ndarray:
    """Vector of all y coordinates."""
    if len(nodes) == 0:
        return np.zeros(0, dtype=float)
    if hasattr(nodes[0], "y"):
        return np.array([float(n.y) for n in nodes], dtype=float)
    return np.asarray(nodes, dtype=float)[:, 1]


def build_xfem_dofs_multi(
    nodes,
    elems,
    cracks: list[XFEMCrack],
    ny: int,
    rebar_segs: Optional[np.ndarray] = None,
    enable_bond_slip: bool = False,
) -> MultiXFEMDofs:
    """Build DOF mapping for multicrack XFEM with optional bond-slip.

    Parameters
    ----------
    nodes : array-like
        Node coordinates
    elems : np.ndarray
        Element connectivity
    cracks : list[XFEMCrack]
        Active cracks
    ny : int
        Number of elements in y-direction (for enrichment gating)
    rebar_segs : np.ndarray, optional
        Rebar segments [n_seg, 5]: [n1, n2, L0, cx, cy]
    enable_bond_slip : bool
        Enable bond-slip DOFs (steel nodes)

    Returns
    -------
    dofs : MultiXFEMDofs
        DOF mapping with concrete + crack + steel DOFs
    """
    nnode = len(nodes)

    # Standard dofs
    std = -np.ones((nnode, 2), dtype=int)
    dof = 0
    for a in range(nnode):
        std[a, 0] = dof; dof += 1
        std[a, 1] = dof; dof += 1

    yvals = _nodes_y(nodes)
    dy = (float(np.max(yvals)) - float(np.min(yvals))) / max(1, ny)

    H_list: list[np.ndarray] = []
    Hn_list: list[np.ndarray] = []

    for crack in cracks:
        H = -np.ones((nnode, 2), dtype=int)
        H_nodes = np.zeros(nnode, dtype=bool)

        if crack.active:
            # Mark nodes in elements cut by this crack segment
            for e, conn in enumerate(elems):
                xe = np.array([_node_xy(nodes, i) for i in conn], dtype=float)
                if crack.cuts_element(xe):
                    for i in conn:
                        H_nodes[i] = True

            # Optional: avoid enriching nodes above the current tip too far
            tip_y = crack.tip_y
            for a in range(nnode):
                if _node_y(nodes, a) > tip_y + 1.5 * dy:
                    H_nodes[a] = False

            # Assign dofs
            for a in range(nnode):
                if H_nodes[a]:
                    H[a, 0] = dof; dof += 1
                    H[a, 1] = dof; dof += 1

        H_list.append(H)
        Hn_list.append(H_nodes)

    # Bond-slip: allocate steel DOFs (FASE D)
    steel = None
    steel_dof_offset = -1
    steel_nodes = None

    if enable_bond_slip and rebar_segs is not None and len(rebar_segs) > 0:
        steel_dof_offset = dof  # Steel DOFs start after concrete + crack DOFs
        steel_nodes = np.zeros(nnode, dtype=bool)
        steel = -np.ones((nnode, 2), dtype=int)

        # Identify nodes used by rebar segments
        for seg in rebar_segs:
            n1 = int(seg[0])
            n2 = int(seg[1])
            steel_nodes[n1] = True
            steel_nodes[n2] = True

        # Allocate steel DOFs for rebar nodes
        for a in np.where(steel_nodes)[0]:
            steel[a, 0] = dof; dof += 1
            steel[a, 1] = dof; dof += 1

    return MultiXFEMDofs(
        std=std,
        H=H_list,
        H_nodes=Hn_list,
        ndof=dof,
        steel=steel,
        steel_dof_offset=steel_dof_offset,
        steel_nodes=steel_nodes,
    )


def transfer_q_between_dofs_multi(q_old: np.ndarray, dofs_old: MultiXFEMDofs, dofs_new: MultiXFEMDofs) -> np.ndarray:
    """Transfer solution vector to new DOF numbering after crack updates.

    Also transfers steel DOFs if bond-slip is enabled (FASE D).
    """
    q_new = np.zeros(dofs_new.ndof, dtype=float)

    # Standard dofs always present
    nnode = dofs_new.std.shape[0]
    for a in range(nnode):
        for comp in (0, 1):
            d_old = int(dofs_old.std[a, comp])
            d_new = int(dofs_new.std[a, comp])
            if d_old >= 0 and d_new >= 0 and d_old < len(q_old):
                q_new[d_new] = q_old[d_old]

    # Crack dofs: map by (crack_index, node, comp)
    ncr_old = len(dofs_old.H)
    ncr_new = len(dofs_new.H)
    ncr = min(ncr_old, ncr_new)
    for k in range(ncr):
        Hold = dofs_old.H[k]
        Hnew = dofs_new.H[k]
        for a in range(nnode):
            for comp in (0, 1):
                d_old = int(Hold[a, comp])
                d_new = int(Hnew[a, comp])
                if d_old >= 0 and d_new >= 0 and d_old < len(q_old):
                    q_new[d_new] = q_old[d_old]

    # Bond-slip DOFs: transfer steel displacements (FASE D)
    if (dofs_old.steel is not None and dofs_new.steel is not None):
        for a in range(nnode):
            for comp in (0, 1):
                d_old = int(dofs_old.steel[a, comp])
                d_new = int(dofs_new.steel[a, comp])
                if d_old >= 0 and d_new >= 0 and d_old < len(q_old):
                    q_new[d_new] = q_old[d_old]

    return q_new


def _build_B_enriched_multi(
    N,
    dNdx,
    nodes,
    conn,
    cracks: list[XFEMCrack],
    dofs: MultiXFEMDofs,
    x_gp: float,
    y_gp: float,
    enr_scale: float = 1.0,
):
    """Build strain-displacement matrix B for multi-crack Heaviside enrichment.

    Returns
    -------
    edofs : list[int]
        Global dof indices in the same order as B columns.
    B : (3, ndof_el) ndarray
    """
    nen = len(conn)

    edofs: list[int] = []
    # standard dofs
    for a_local, a in enumerate(conn):
        edofs.append(int(dofs.std[a, 0]))
        edofs.append(int(dofs.std[a, 1]))

    # Heaviside dofs for each crack
    # Keep a mapping from (k, a_local, comp) -> column index
    h_cols = []
    for k, crack in enumerate(cracks):
        if not crack.active:
            continue
        Hmap = dofs.H[k]
        for a_local, a in enumerate(conn):
            for comp in (0, 1):
                d = int(Hmap[a, comp])
                if d >= 0:
                    h_cols.append((k, a_local, comp, len(edofs)))
                    edofs.append(d)

    B = np.zeros((3, len(edofs)), dtype=float)

    # standard part
    for a_local in range(nen):
        B[0, 2*a_local + 0] = dNdx[a_local, 0]
        B[1, 2*a_local + 1] = dNdx[a_local, 1]
        B[2, 2*a_local + 0] = dNdx[a_local, 1]
        B[2, 2*a_local + 1] = dNdx[a_local, 0]

    # enriched part: for each crack
    if h_cols:
        # Precompute node positions
        xy_nodes = np.array([_node_xy(nodes, a) for a in conn], dtype=float)

        for (k, a_local, comp, col) in h_cols:
            crack = cracks[k]
            # Heaviside with behind-tip gating (same idea as single-crack implementation)
            H_gp = crack.H(x_gp, y_gp)
            H_a = crack.H(xy_nodes[a_local, 0], xy_nodes[a_local, 1])
            G_y = crack.behind_tip(x_gp, y_gp)
            # Optional continuation/ramping factor for enrichment activation.
            # When enr_scale < 1, the enrichment is gradually introduced to
            # improve robustness right after crack initiation/growth.
            fac = float(enr_scale) * float(H_gp - H_a) * float(G_y)
            if abs(fac) < 1e-14:
                continue

            if comp == 0:
                B[0, col] = fac * dNdx[a_local, 0]
                B[2, col] = fac * dNdx[a_local, 1]
            else:
                B[1, col] = fac * dNdx[a_local, 1]
                B[2, col] = fac * dNdx[a_local, 0]

    return edofs, B


def assemble_xfem_system_multi(
    nodes,
    elems,
    q,
    dofs: MultiXFEMDofs,
    cracks: list[XFEMCrack],
    model: XFEMModel,
    C,
    rebar_segs,
    coh_states: Union[dict[tuple, CohesiveState], CohesiveStateArrays],
    u_bar: float,
    visc_damp: float,
    law: CohesiveLaw,
    use_numba: bool = False,
    coh_params: Optional[np.ndarray] = None,
    enr_scale: float = 1.0,
    bulk_states: Optional[Union[dict, "BulkStateArrays"]] = None,
    bulk_kind: int = 0,
    bulk_params: Optional[np.ndarray] = None,
    material: Optional["ConstitutiveModel"] = None,
    # Bond-slip parameters (FASE D)
    bond_law: Optional[object] = None,
    bond_layers: Optional[list] = None,
    bond_states_comm: Optional[object] = None,
    enable_bond_slip: bool = False,
    steel_EA: float = 0.0,
    perimeter_total: Optional[float] = None,  # Total perimeter for bond-slip
    bond_gamma: float = 1.0,  # BLOQUE A: Bond-slip continuation parameter
    # Subdomain support (FASE D)
    subdomain_mgr: Optional[object] = None,
):
    """Assemble global stiffness and residual with multiple Heaviside cracks.

    Cohesive history is stored either as a legacy dict keyed by
    ``(crack_id, elem_id, igp)`` or as :class:`~xfem_clean.xfem.state_arrays.CohesiveStateArrays`
    (Phase-1). Assembly returns trial history as a *patch* so committed history
    is never mutated inside Newton iterations.

    Phase 2: Now supports bulk material nonlinearity (CDP/DP) via Numba kernels.
    """
    case_id = getattr(model, "case_id", "unknown")
    ndof = dofs.ndof
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    fint = np.zeros(ndof, dtype=float)
    fext = np.zeros(ndof, dtype=float)
    # external load (point load at load node)
    # It is enforced via Dirichlet (displacement control) in the solver, so fext can be zero.

    # Bulk integration (2x2 Gauss). For cut elements we use a subcell integration
    # identical in spirit to the single-crack solver (uniform nsub x nsub split).
    gp = [-1.0/math.sqrt(3.0), +1.0/math.sqrt(3.0)]
    gw = [1.0, 1.0]
    thickness = float(model.b)

    aux_gp_pos = []
    aux_gp_sig = []

    use_coh_arrays = isinstance(coh_states, CohesiveStateArrays)
    coh_updates = CohesiveStatePatch.empty() if use_coh_arrays else dict(coh_states)

    # Phase 2: Bulk material states
    use_bulk_arrays = isinstance(bulk_states, BulkStateArrays)
    use_bulk_numba = bool(use_numba) and use_bulk_arrays and (bulk_params is not None) and (int(bulk_kind) in (1, 2, 3))
    bulk_updates: Union[dict, BulkStatePatch]
    bulk_updates = BulkStatePatch.empty() if use_bulk_arrays else {}

    # Default material (retrocompat)
    if material is None and not use_bulk_numba:
        material = LinearElasticPlaneStress(E=float(model.E), nu=float(model.nu))
        material.C = np.asarray(C, dtype=float)
    
    nsub_cut = 4  # subcell integration resolution for cut elements
    
    for e, conn in enumerate(elems):
        xe = np.array([_node_xy(nodes, i) for i in conn], dtype=float)

        # Check if element is void (use penalty stiffness) - BLOQUE D
        is_void_elem = subdomain_mgr is not None and subdomain_mgr.is_void(e)

        # Get effective material properties and thickness
        thickness_eff = thickness
        C_eff = C  # Default: use full stiffness
        if subdomain_mgr is not None:
            thickness_eff = subdomain_mgr.get_effective_thickness(e, thickness)

        # BLOQUE D: Void elements - apply penalty stiffness to prevent singularity
        # Instead of skipping, use very small stiffness to keep matrix non-singular
        void_penalty_factor = 1e-9
        if is_void_elem:
            C_eff = C * void_penalty_factor  # Penalty stiffness (very small but non-zero)
            if thickness_eff < 1e-12:
                thickness_eff = thickness * void_penalty_factor  # Minimal thickness to avoid zero volume
            # For Numba path: store penalty factor to scale material parameters
            # (applied in the integration loop below)
        else:
            # Non-void element: skip only if thickness is effectively zero
            if thickness_eff < 1e-12:
                continue

        is_cut = False
        for crack in cracks:
            if crack.active and crack.cuts_element(xe):
                is_cut = True
                break

        if not is_cut:
            subdomains = [(-1.0, 1.0, -1.0, 1.0)]
        else:
            xis = np.linspace(-1.0, 1.0, nsub_cut + 1)
            etas = np.linspace(-1.0, 1.0, nsub_cut + 1)
            subdomains = []
            for i in range(nsub_cut):
                for j in range(nsub_cut):
                    subdomains.append((float(xis[i]), float(xis[i+1]), float(etas[j]), float(etas[j+1])))
    
        for sidx, (xi_a, xi_b, eta_a, eta_b) in enumerate(subdomains):
            for ixi, xi_hat in enumerate(gp):
                xi = 0.5*(xi_b - xi_a)*xi_hat + 0.5*(xi_b + xi_a)
                wx = gw[ixi] * 0.5*(xi_b - xi_a)
                for ieta, eta_hat in enumerate(gp):
                    eta = 0.5*(eta_b - eta_a)*eta_hat + 0.5*(eta_b + eta_a)
                    wy = gw[ieta] * 0.5*(eta_b - eta_a)
                    w = float(wx * wy)
    
                    N, dN_dxi_s, dN_deta_s = q4_shape(xi, eta)
                    dN_dxi = np.column_stack((dN_dxi_s, dN_deta_s))  # (4,2)
    
                    J = xe.T @ dN_dxi
                    detJ = float(np.linalg.det(J))
                    if detJ <= 1e-14:
                        continue
                    invJ = np.linalg.inv(J)
                    dNdx = dN_dxi @ invJ  # (4,2) columns are [dN/dx, dN/dy]
    
                    x_gp = float(N @ xe[:, 0])
                    y_gp = float(N @ xe[:, 1])
    
                    edofs, B = _build_B_enriched_multi(
                        N,
                        dNdx,
                        nodes,
                        conn,
                        cracks,
                        dofs,
                        x_gp,
                        y_gp,
                        enr_scale=float(enr_scale),
                    )
                    q_e = q[np.array(edofs, dtype=int)]
                    eps = B @ q_e

                    # Integration-point id (same logic as single-crack)
                    ip_local = int(ixi + 2 * ieta)
                    ipid = ip_local if (not is_cut) else int(4 + 4 * sidx + ip_local)

                    # Phase 2: Constitutive integration (Numba path or Python material)
                    if use_bulk_numba:
                        assert isinstance(bulk_states, BulkStateArrays)
                        # Load state
                        dt0 = float(bulk_states.damage_t[e, ipid])
                        dc0 = float(bulk_states.damage_c[e, ipid])
                        wpl0 = float(bulk_states.w_plastic[e, ipid])
                        wft0 = float(bulk_states.w_fract_t[e, ipid])
                        wfc0 = float(bulk_states.w_fract_c[e, ipid])
                        eps_p6_old = np.asarray(bulk_states.eps_p6[e, ipid, :], dtype=float)
                        ezz_old = float(bulk_states.eps_zz[e, ipid])
                        kappa_old = float(bulk_states.kappa[e, ipid])
                        kt0 = float(bulk_states.kappa_t[e, ipid])
                        kc0 = float(bulk_states.kappa_c[e, ipid])

                        if int(bulk_kind) == 1:
                            E_mat = float(bulk_params[0])
                            nu_mat = float(bulk_params[1])
                            # BLOQUE D: Apply penalty to E_mat for void elements
                            if is_void_elem:
                                E_mat *= void_penalty_factor
                            sig, Ct = elastic_integrate_plane_stress_numba(eps, E_mat, nu_mat, dt0, dc0)
                            eps_p6_new = eps_p6_old
                            ezz_new = ezz_old
                            kappa_new = kappa_old
                            dt_new, dc_new = dt0, dc0
                            kt_new, kc_new = kt0, kc0
                            wpl_new, wft_new, wfc_new = wpl0, wft0, wfc0
                        elif int(bulk_kind) == 2:
                            E_mat = float(bulk_params[0])
                            nu_mat = float(bulk_params[1])
                            alpha = float(bulk_params[2])
                            k0 = float(bulk_params[3])
                            Hh = float(bulk_params[4])
                            # BLOQUE D: Apply penalty to E_mat for void elements
                            if is_void_elem:
                                E_mat *= void_penalty_factor
                            sig, Ct, eps_p6_new, ezz_new, kappa_new, dW = dp_integrate_plane_stress_numba(
                                eps, eps_p6_old, ezz_old, kappa_old, E_mat, nu_mat, alpha, k0, Hh
                            )
                            dt_new, dc_new = dt0, dc0
                            kt_new, kc_new = kt0, kc0
                            wpl_new = wpl0 + dW
                            wft_new, wfc_new = wft0, wfc0
                        elif int(bulk_kind) == 3:
                            E_mat = float(bulk_params[0])
                            nu_mat = float(bulk_params[1])
                            alpha = float(bulk_params[2])
                            k0 = float(bulk_params[3])
                            Hh = float(bulk_params[4])
                            ft = float(bulk_params[5])
                            fc = float(bulk_params[6])
                            Gf_t = float(bulk_params[7])
                            Gf_c = float(bulk_params[8])
                            lch = float(bulk_params[9])
                            # BLOQUE D: Apply penalty to E_mat for void elements
                            if is_void_elem:
                                E_mat *= void_penalty_factor
                            sig, Ct, eps_p6_new, ezz_new, kappa_new, dt_new, dc_new, kt_new, kc_new, wpl_new, wft_new, wfc_new = cdp_integrate_plane_stress_numba(
                                eps, eps_p6_old, ezz_old, kappa_old, dt0, dc0, kt0, kc0, wpl0, wft0, wfc0,
                                E_mat, nu_mat, alpha, k0, Hh, ft, fc, Gf_t, Gf_c, lch
                            )
                        else:
                            raise ValueError(f"Unknown bulk_kind={bulk_kind}")

                        # Store trial state
                        eps_p3_new = np.array([eps_p6_new[0], eps_p6_new[1], eps_p6_new[3]], dtype=float)
                        if isinstance(bulk_updates, BulkStatePatch):
                            bulk_updates.add_values(
                                e, ipid,
                                eps=np.asarray(eps, dtype=float),
                                sigma=np.asarray(sig, dtype=float),
                                eps_p=eps_p3_new,
                                damage_t=float(dt_new),
                                damage_c=float(dc_new),
                                kappa=float(kappa_new),
                                w_plastic=float(wpl_new),
                                w_fract_t=float(wft_new),
                                w_fract_c=float(wfc_new),
                                eps_p6=np.asarray(eps_p6_new, dtype=float),
                                eps_zz=float(ezz_new),
                                kappa_t=float(kt_new),
                                kappa_c=float(kc_new),
                                cdp_w_t=0.0,
                                cdp_eps_in_c=0.0,
                            )
                    elif bulk_states is not None and material is not None:
                        # Python material path (retrocompat or ConcreteCDPReal with tables)
                        # BLOQUE D: Override material C for void elements
                        C_orig = None
                        if is_void_elem and hasattr(material, 'C'):
                            C_orig = material.C.copy()
                            material.C = C_eff

                        if use_bulk_arrays:
                            assert isinstance(bulk_states, BulkStateArrays)
                            mp0 = bulk_states.get_mp(e, ipid)
                        else:
                            mp0 = bulk_states.get((e, ipid), mp_default())
                        mp = mp0.copy_shallow()
                        sig, Ct = material.integrate(mp, eps)

                        # BLOQUE D: Restore original material C
                        if C_orig is not None:
                            material.C = C_orig
                        if isinstance(bulk_updates, BulkStatePatch):
                            bulk_updates.add(e, ipid, mp)
                        else:
                            bulk_updates[(e, ipid)] = mp
                    else:
                        # Fallback: linear elastic (retrocompat)
                        # BLOQUE D: Use C_eff for void elements
                        sig = C_eff @ eps
                        Ct = C_eff

                    # store aux data for nonlocal averaging
                    aux_gp_pos.append((x_gp, y_gp))
                    aux_gp_sig.append((float(sig[0]), float(sig[1]), float(sig[2])))

                    w_bulk = detJ * w * thickness_eff
                    Ke = (B.T @ Ct @ B) * w_bulk
                    fe_int = (B.T @ sig) * w_bulk
    
                    # scatter (sparse triplets)
                    for a_loc, A in enumerate(edofs):
                        fint[A] += fe_int[a_loc]
                    for a_loc, A in enumerate(edofs):
                        for b_loc, Bidx in enumerate(edofs):
                            data.append(float(Ke[a_loc, b_loc]))
                            rows.append(int(A))
                            cols.append(int(Bidx))
    
    # Cohesive integration: per crack, per cut element.
    # Phase-1: return a patch (sparse update) instead of materializing a full
    # trial dictionary every Newton iteration.
    use_coh_arrays = isinstance(coh_states, CohesiveStateArrays)
    coh_updates: Union[dict[tuple, CohesiveState], CohesiveStatePatch]
    coh_updates = CohesiveStatePatch.empty() if use_coh_arrays else dict(coh_states)

    for k, crack in enumerate(cracks):
        if not crack.active:
            continue

        for e, conn in enumerate(elems):
            xe = np.array([_node_xy(nodes, i) for i in conn], dtype=float)
            if not crack.cuts_element(xe):
                continue

            bbox = (float(np.min(xe[:, 0])), float(np.max(xe[:, 0])), float(np.min(xe[:, 1])), float(np.max(xe[:, 1])))
            seg = clip_segment_to_bbox(crack.p0(), crack.pt(), *bbox)
            if seg is None:
                continue
            pA, pB = seg
            seg_len = float(np.linalg.norm(pB - pA))
            if seg_len <= 1e-14:
                continue

            # 2-pt Gauss on [0,1]
            sgp = [-1.0/math.sqrt(3.0), +1.0/math.sqrt(3.0)]
            sgw = [1.0, 1.0]

            # build element dof list incl this crack dofs
            edofs = []
            for a in conn:
                edofs.append(int(dofs.std[a, 0]))
                edofs.append(int(dofs.std[a, 1]))
            Hmap = dofs.H[k]
            hcols = []
            for a_local, a in enumerate(conn):
                for comp in (0, 1):
                    d = int(Hmap[a, comp])
                    if d >= 0:
                        hcols.append((a_local, comp, len(edofs)))
                        edofs.append(d)

            q_e = q[np.array(edofs, dtype=int)]

            # normal to crack (opening mode)
            t = (crack.pt() - crack.p0())
            t = t / max(1e-14, float(np.linalg.norm(t)))
            n = np.array([-t[1], t[0]], dtype=float)

            for igp, shat in enumerate(sgp):
                s = 0.5*(shat + 1.0)
                w = sgw[igp] * 0.5
                x = float(pA[0] + s*(pB[0] - pA[0]))
                y = float(pA[1] + s*(pB[1] - pA[1]))

                # param coords
                xi, eta = map_global_to_parent_Q4(x, y, xe)
                N, _dN_dxi_s, _dN_deta_s = q4_shape(xi, eta)

                # displacement jump operator J such that jump = J @ q_e
                # only enriched dofs contribute to jump (via N*(H+ - H-)) ~ 2N for cut elements
                Jop = np.zeros((2, len(edofs)), dtype=float)

                # standard dofs: continuous, no jump

                # enriched dofs: jump across crack is 2*N*a_i (since H flips sign)
                for (a_local, comp, col) in hcols:
                    # Continuation/ramping: scale jump operator to gradually activate
                    # the discontinuity constraints after initiation/growth.
                    val = float(enr_scale) * 2.0 * float(N[a_local])
                    if comp == 0:
                        Jop[0, col] = val
                    else:
                        Jop[1, col] = val

                jump_u = Jop @ q_e
                delta_n = float(np.dot(jump_u, n))

                key = (k, e, igp)
                # --- Cohesive state update (Phase 2: value-kernel option) ---
                if use_numba and (coh_params is not None) and use_coh_arrays:
                    assert isinstance(coh_states, CohesiveStateArrays)
                    dm_old, dmg_old = coh_states.get_values(e, igp, k=k)
                    t_n, dtn_dd, dm_new, dmg_new = cohesive_update_values_numba(
                        delta_n,
                        dm_old,
                        dmg_old,
                        coh_params,
                        visc_damp=float(visc_damp),
                    )
                    assert isinstance(coh_updates, CohesiveStatePatch)
                    coh_updates.add_values(k, e, igp, delta_max=dm_new, damage=dmg_new)
                else:
                    if use_coh_arrays:
                        assert isinstance(coh_states, CohesiveStateArrays)
                        st_old = coh_states.get_state(e, igp, k=k)
                    else:
                        st_old = coh_states.get(key, CohesiveState())
                    t_n, dtn_dd, st_new = cohesive_update(law, delta_n, st_old, visc_damp=visc_damp)
                    if isinstance(coh_updates, CohesiveStatePatch):
                        coh_updates.add(k, e, igp, st_new)
                    else:
                        coh_updates[key] = st_new

                # Fibre bridging contribution (BLOQUE 6)
                if hasattr(model, 'fibre_bridging_cfg') and model.fibre_bridging_cfg is not None:
                    # Import here to avoid circular dependency
                    from xfem_clean.fibre_bridging import fibre_traction_tangent

                    # Estimate delta_l (patch length represented by this cohesive point)
                    # Approximate as segment length (could be refined)
                    delta_l = seg_len

                    # Compute fibre contribution (uses crack opening delta_n)
                    t_fibre, k_fibre = fibre_traction_tangent(
                        w_n=delta_n,  # Crack opening (m)
                        delta_l=delta_l,  # Patch length (m)
                        cfg=model.fibre_bridging_cfg,
                        rng=None,  # Will use cfg.random_seed
                    )

                    # Add to cohesive traction and tangent
                    t_n += t_fibre
                    dtn_dd += k_fibre

                # cohesive force vector in global dofs
                # f = J^T * (t_n * n)
                tr_vec = t_n * n
                fe = Jop.T @ tr_vec

                # cohesive tangent
                Ke = Jop.T @ (dtn_dd * np.outer(n, n)) @ Jop

                # scale by segment length
                scale = seg_len * w * thickness

                for a_loc, A in enumerate(edofs):
                    fint[A] += fe[a_loc] * scale
                for a_loc, A in enumerate(edofs):
                    for b_loc, Bidx in enumerate(edofs):
                        data.append(float(Ke[a_loc, b_loc] * scale))
                        rows.append(int(A))
                        cols.append(int(Bidx))


    # Build sparse global stiffness (duplicates are summed in CSR conversion)
    if len(data) == 0:
        K = sp.csr_matrix((ndof, ndof), dtype=float)
    else:
        K = sp.coo_matrix((np.array(data, dtype=float), (np.array(rows, dtype=int), np.array(cols, dtype=int))),
                          shape=(ndof, ndof)).tocsr()

    # Stabilize enriched DOFs (avoid singularities right after enrichment activation)
    if float(getattr(model, "k_stab", 0.0)) > 0.0:
        nnode = int(nodes.shape[0]) if hasattr(nodes, "shape") else len(nodes)
        diagK = K.diagonal()
        if 2*nnode <= diagK.size:
            k_ref = float(np.max(np.abs(diagK[:2*nnode])))
        else:
            k_ref = float(np.max(np.abs(diagK)))
        if (not np.isfinite(k_ref)) or (k_ref <= 0.0):
            k_ref = 1.0
        k_stab_eff = float(model.k_stab)
        if k_stab_eff < 1.0:
            k_stab_eff *= k_ref
        stab = np.zeros(ndof, dtype=float)
        stab[2*nnode:] = k_stab_eff
        K = K + sp.diags(stab, 0, shape=(ndof, ndof), format="csr")

    # --- Bond-slip or perfect-bond rebar (FASE D) ---
    bond_updates = None

    if enable_bond_slip and rebar_segs is not None and bond_law is not None and bond_states_comm is not None:
        # FASE D: Bond-slip integration
        if dofs.steel_dof_offset < 0:
            raise ValueError("Bond-slip enabled but steel DOFs not allocated. Check build_xfem_dofs_multi().")

        from xfem_clean.bond_slip import assemble_bond_slip

        dtau_existing = getattr(bond_law, "dtau_max", None)
        if dtau_existing is None or (not np.isfinite(dtau_existing)) or float(dtau_existing) >= 1e18:
            nnode = int(nodes.shape[0]) if hasattr(nodes, "shape") else len(nodes)
            diagK = K.diagonal()
            if 2 * nnode <= diagK.size:
                diag_std = np.abs(diagK[: 2 * nnode])
            else:
                diag_std = np.abs(diagK)
            diag_std = diag_std[diag_std > 0.0]
            k_ref = float(np.median(diag_std)) if diag_std.size > 0 else 1.0
            if (not np.isfinite(k_ref)) or k_ref <= 0.0:
                k_ref = 1.0

            L_char = 1.0
            if rebar_segs is not None and rebar_segs.shape[0] > 0:
                L0 = np.abs(rebar_segs[:, 2])
                L0 = L0[L0 > 0.0]
                if L0.size > 0:
                    L_char = float(np.median(L0))

            perimeter = perimeter_total
            if perimeter is None:
                perimeter = math.pi * float(getattr(model, "rebar_diameter", 0.0))

            dtau_max = float(getattr(model, "bond_tangent_cap_factor", 1.0)) * k_ref / max(1e-30, perimeter * L_char)
            bond_law.dtau_max = float(dtau_max)
            if getattr(model, "bond_k_cap", None) is None:
                model.bond_k_cap = float(dtau_max)

        # Generate segment_mask for bond-disabled regions (FASE D)
        segment_mask = None
        bond_disabled_x_range = getattr(model, 'bond_disabled_x_range', None)
        if bond_disabled_x_range is not None and rebar_segs is not None:
            # Mask segments in the disabled x-range
            # segment_mask[i] = True means bond is DISABLED for segment i
            n_seg = rebar_segs.shape[0]
            segment_mask = np.zeros(n_seg, dtype=bool)  # Initialize to False (bond enabled)
            x_min, x_max = bond_disabled_x_range
            for i in range(n_seg):
                n1 = int(rebar_segs[i, 0])
                n2 = int(rebar_segs[i, 1])
                x_mid = 0.5 * (nodes[n1, 0] + nodes[n2, 0])
                if x_min <= x_mid <= x_max:
                    segment_mask[i] = True  # Disable bond in this segment (True = disabled)

        f_bond, K_bond, bond_updates = assemble_bond_slip(
            u_total=q,
            steel_segments=rebar_segs,
            steel_dof_offset=dofs.steel_dof_offset,
            bond_law=bond_law,
            bond_states=bond_states_comm,
            steel_dof_map=dofs.steel,
            steel_EA=steel_EA,
            use_numba=use_numba,
            perimeter=perimeter_total,  # Pass explicit perimeter (FASE D)
            segment_mask=segment_mask,  # Pass segment mask (FASE D)
            bond_gamma=bond_gamma,  # BLOQUE A: Bond-slip continuation parameter
            bond_k_cap=model.bond_k_cap,
            bond_s_eps=model.bond_s_eps,
        )

        # Add bond-slip contribution to global system
        fint += f_bond
        K = K + K_bond
        _raise_on_nonfinite(
            "bond-slip",
            {
                "f_bond": f_bond,
                "K_bond": K_bond,
                "fint": fint,
                "K": K,
            },
            bulk_kind=bulk_kind,
            enable_bond_slip=enable_bond_slip,
            use_numba=use_numba,
            case_id=case_id,
            u_bar=u_bar,
        )

    elif rebar_segs is not None and not enable_bond_slip:
        # Legacy: perfect-bond rebar (no slip)
        f_rb, K_rb = rebar_contrib(
            nodes,
            rebar_segs,
            q,
            model.steel_A_total,
            model.steel_E,
            model.steel_fy,
            model.steel_fu,
            model.steel_Eh,
        )

        # Embed into XFEM global system if needed
        if f_rb.shape[0] != dofs.ndof:
            f_full = np.zeros(dofs.ndof, dtype=float)
            f_full[: f_rb.shape[0]] = f_rb
            f_rb = f_full

        if getattr(K_rb, "shape", None) is not None and K_rb.shape != (dofs.ndof, dofs.ndof):
            K_full = sp.lil_matrix((dofs.ndof, dofs.ndof), dtype=float)
            K_full[: K_rb.shape[0], : K_rb.shape[1]] = K_rb
            K_rb = K_full.tocsr()
        else:
            K_rb = K_rb.tocsr() if sp.issparse(K_rb) else sp.csr_matrix(K_rb)

        fint = fint + f_rb
        K = K + K_rb

    aux_gp_pos = np.array(aux_gp_pos, dtype=float)
    aux_gp_sig = np.array(aux_gp_sig, dtype=float)

    return K, fint, fext, coh_updates, bulk_updates, aux_gp_pos, aux_gp_sig, bond_updates


def _array_for_stats(arr):
    if arr is None:
        return None
    if sp.issparse(arr):
        return arr.data
    return np.asarray(arr)


def _format_finite_stats(name: str, arr) -> str:
    data = _array_for_stats(arr)
    if data is None:
        return f"{name}: none"
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        return f"{name}: nan=0, inf=0, finite_min=None, finite_max=None"
    nan_count = int(np.isnan(data).sum())
    inf_count = int(np.isinf(data).sum())
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        finite_min = None
        finite_max = None
    else:
        finite_min = float(np.min(finite))
        finite_max = float(np.max(finite))
    return (
        f"{name}: nan={nan_count}, inf={inf_count}, "
        f"finite_min={finite_min}, finite_max={finite_max}"
    )


def _raise_on_nonfinite(
    label: str,
    arrays: dict,
    *,
    bulk_kind: int,
    enable_bond_slip: bool,
    use_numba: bool,
    case_id: str,
    u_bar: float,
) -> None:
    diagnostics = []
    has_issue = False
    for name, arr in arrays.items():
        data = _array_for_stats(arr)
        if data is None:
            diagnostics.append(f"{name}: none")
            continue
        data = np.asarray(data, dtype=float)
        if data.size == 0:
            diagnostics.append(f"{name}: nan=0, inf=0, finite_min=None, finite_max=None")
            continue
        if not np.isfinite(data).all():
            has_issue = True
        diagnostics.append(_format_finite_stats(name, data))
    if has_issue:
        flags = (
            f"bulk_kind={bulk_kind}, enable_bond_slip={enable_bond_slip}, "
            f"use_numba={use_numba}, case_id={case_id}, u_bar={u_bar}"
        )
        message = f"Non-finite values detected ({label}). " + " | ".join(diagnostics) + f" | {flags}"
        raise RuntimeError(message)


def _candidate_points_zone(model: XFEMModel, zone: str, nx: int) -> list[tuple[float, float]]:
    """Generate candidate points for crack initiation."""
    L = model.L
    H = model.H
    y0 = 0.0

    # Optional override for non-3PB geometries (e.g., cantilevers):
    # allow candidates near the fixed end by specifying a window as fractions of L.
    dominant_window = getattr(model, "dominant_window", None)
    use_dominant = (
        getattr(model, "cand_mode", "") == "dominant"
        and dominant_window is not None
        and isinstance(dominant_window, (tuple, list))
        and len(dominant_window) == 2
    )

    if zone == "flexure":
        if use_dominant:
            a, b = float(dominant_window[0]), float(dominant_window[1])
            a = max(0.0, min(1.0, a))
            b = max(0.0, min(1.0, b))
            x0 = a * L
            x1 = b * L
        else:
            x0 = 0.15 * L
            x1 = 0.85 * L
    elif zone == "shear_left":
        x0 = 0.08 * L
        x1 = 0.35 * L
    elif zone == "shear_right":
        x0 = 0.65 * L
        x1 = 0.92 * L
    else:
        x0 = 0.0
        x1 = L

    xs = np.linspace(x0, x1, max(3, int(0.5*nx)))
    pts = [(float(x), float(y0)) for x in xs]
    return pts


def _too_close_to_existing(x0: float, y0: float, cracks: list[XFEMCrack], min_spacing: float) -> bool:
    for c in cracks:
        if not c.active:
            continue
        if abs(x0 - c.x0) < min_spacing and abs(y0 - c.y0) < 0.05:
            return True
    return False


def _init_crack_from_stress(x0: float, y0: float, sigbar: np.ndarray, model: XFEMModel) -> XFEMCrack:
    # direction of maximum principal stress
    s1, v1 = principal_max_dir(sigbar)
    # crack tangent is perpendicular to v1 (crack normal aligned with v1)
    t = np.array([-v1[1], v1[0]], dtype=float)
    if t[1] < 0:
        t = -t

    # enforce growth toward interior and toward midspan
    if x0 < 0.5 * model.L and t[0] < 0:
        t = -t
    if x0 > 0.5 * model.L and t[0] > 0:
        t = -t

    t = t / max(1e-14, float(np.linalg.norm(t)))
    ds = model.H / max(1, model.ny)  # about one element height

    tip_x = x0 + ds * float(t[0])
    tip_y = y0 + ds * float(t[1])

    angle = math.degrees(math.atan2(t[1], t[0]))
    stop_y = float(model.crack_tip_stop_y) if getattr(model, "crack_tip_stop_y", None) is not None else (
        0.5 * float(model.H) if getattr(model, "arrest_at_half_height", True) else float(model.H)
    )
    crack = XFEMCrack(x0=float(x0), y0=float(y0), tip_x=float(tip_x), tip_y=float(tip_y),
                      stop_y=float(stop_y), angle_deg=float(angle), active=True)
    return crack


def run_analysis_xfem_multicrack(
    model: XFEMModel,
    nx=120,
    ny=20,
    nsteps=30,
    umax=0.01,
    max_cracks: int = 8,
    crack_mode: str = "option2",
    u_diag_mm: float = 2.0,
    min_crack_spacing_factor: float = 2.0,
    law: Optional[CohesiveLaw] = None,
    nodes: Optional[np.ndarray] = None,
    elems: Optional[np.ndarray] = None,
    u_targets: Optional[np.ndarray] = None,
    bc_spec: Optional["BCSpec"] = None,
    bond_law: Optional[object] = None,
    bond_layers: Optional[list] = None,
    return_bundle: bool = False,
):
    """Run displacement-controlled analysis with multiple cracks.

    Parameters
    ----------
    model : XFEMModel
        Model with geometry, material, and solver parameters
    nodes : np.ndarray, optional
        Node coordinates [nnode, 2] (m). If None, generates structured mesh.
    elems : np.ndarray, optional
        Element connectivity [nelem, 4]. If None, generates structured mesh.
    u_targets : np.ndarray, optional
        Displacement trajectory (m). If None, uses linspace(0, umax, nsteps).
    bc_spec : BCSpec, optional
        Boundary condition specification. If None, defaults to 3PB beam BCs.
    return_bundle : bool, optional
        If True, return dict with comprehensive results for postprocessing:
        {nodes, elems, u, history, cracks, bond_states, rebar_segs, dofs, coh_states, bulk_states}
        If False (default), return tuple (backward compatible)

    crack_mode:
        - "option1": multiple flexural (mostly vertical) cracks only.
        - "option2": flexural cracks + allow diagonal/shear cracks after u>=u_diag_mm.

    Returns
    -------
    nodes, elems, q, results, cracks (if return_bundle=False)
    OR
    dict (if return_bundle=True)
    """
    # Phase 2: CDP support enabled for multi-crack!

    # Mesh (reuse the same generator used by the single-crack solver)
    if nodes is None or elems is None:
        nodes, elems = structured_quad_mesh(model.L, model.H, nx, ny)
    else:
        # Use provided mesh (from solver_interface)
        pass

    # Infer mesh spacing (dx, dy) from unique coordinate grids
    xs = np.unique(nodes[:, 0]); ys = np.unique(nodes[:, 1])
    xs.sort(); ys.sort()
    dx = float(np.min(np.diff(xs))) if len(xs) > 1 else float(model.L)
    dy = float(np.min(np.diff(ys))) if len(ys) > 1 else float(model.H)
    model.ny = ny
    model.lch = math.sqrt(dx*dy)

    # Cohesive law (shared by all cracks). If not provided, build the default
    # bilinear law from model parameters.
    if law is None:
        Kn = model.Kn_factor * model.E / max(1e-12, model.lch)
        law = CohesiveLaw(Kn=Kn, ft=model.ft, Gf=model.Gf)

    # Optional Numba kernels: pack cohesive parameters once.
    use_numba = bool(getattr(model, "use_numba", False))
    coh_params = pack_cohesive_law_params(law) if use_numba else None

    # Nonlocal averaging radius for crack initiation/propagation.
    # Keep consistent with the single-crack solver: `crack_rho` is a physical length [m].
    rho = float(model.crack_rho)
    r_cut = 3.0 * rho
    stop_y = float(model.crack_tip_stop_y) if model.crack_tip_stop_y is not None else (
        0.5 * float(model.H) if getattr(model, "arrest_at_half_height", True) else float(model.H)
    )

    C = plane_stress_C(model.E, model.nu)
    # Rebar segments
    rebar_segs = None
    if bond_layers is not None and len(bond_layers) > 0:
        try:
            seg_list = []
            for layer in bond_layers:
                seg = getattr(layer, "segments", None)
                if seg is None:
                    continue
                seg = np.asarray(seg, dtype=float)
                if seg.ndim == 2 and seg.shape[1] >= 2 and seg.shape[0] > 0:
                    seg_list.append(seg)
            if len(seg_list) > 0:
                rebar_segs = np.vstack(seg_list)
        except Exception:
            rebar_segs = None

    if rebar_segs is None:
        rebar_segs = prepare_rebar_segments(nodes, cover=model.cover)
    # Boundary conditions (FASE D: bc_spec support)
    if bc_spec is not None:
        # Use bc_spec from solver_interface (allows pullout, custom BCs, etc.)
        # NOTE: bc_spec uses DOF indices, but multicrack uses (node, comp) tuples
        # We'll need to resolve this after DOFs are built
        fixed = None  # Will be resolved later
        load_node = None  # Will be resolved from bc_spec.prescribed_dofs
    else:
        # Default: 3-point bending beam BCs
        left_node = find_nearest_node(nodes, 0.0, 0.0)
        right_node = find_nearest_node(nodes, model.L, 0.0)
        load_node = find_nearest_node(nodes, model.L/2.0, model.H)

        fixed = {
            (left_node, 0): 0.0,
            (left_node, 1): 0.0,
            (right_node, 1): 0.0,
        }

    # displacement control (FASE D: u_targets support)
    if u_targets is None:
        u_targets = np.linspace(0.0, umax, nsteps+1)[1:]
    else:
        # Use provided u_targets (e.g., from cyclic loading)
        pass

    # Bond-slip configuration (FASE D)
    # Use external bond_law if provided, otherwise create default
    enable_bond_slip = bool(getattr(model, "enable_bond_slip", False))
    bond_states = None
    steel_EA = 0.0
    perimeter_total = None  # Perimeter for bond-slip (FASE D)

    if enable_bond_slip and rebar_segs is not None and len(rebar_segs) > 0:
        from xfem_clean.bond_slip import BondSlipStateArrays

        n_seg = rebar_segs.shape[0]
        bond_states = BondSlipStateArrays.zeros(n_seg)

        # Use bond_law from parameter (passed from solver_interface)
        if bond_law is None:
            # Fallback: create default bond law (for backward compatibility)
            from xfem_clean.bond_slip import BondSlipModelCode2010
            bond_law = BondSlipModelCode2010(
                f_cm=model.fc,
                d_bar=model.rebar_diameter,
                condition=getattr(model, "bond_condition", "good"),
            )
            print("WARNING: Using default BondSlipModelCode2010. Pass bond_law explicitly.")

        steel_E = float(getattr(model, "steel_E", 0.0))
        steel_A_total = float(getattr(model, "steel_A_total", 0.0))
        if steel_E > 0.0 and steel_A_total > 0.0:
            steel_EA = steel_E * steel_A_total
        else:
            steel_EA = float(getattr(model, "steel_EA_min", 1e3))

        # Compute perimeter_total from rebar geometry (FASE D)
        # Check for override (e.g., FRP sheet with non-circular perimeter)
        if hasattr(model, 'bond_perimeter_override') and model.bond_perimeter_override is not None:
            perimeter_total = model.bond_perimeter_override
        else:
            # perimeter = (n_bars * π * d_bar) for each layer, summed
            d_bar = getattr(model, "rebar_diameter", 0.012)  # Default 12mm
            # For simplicity, assume all bars have same diameter (can be refined)
            # If model has steel_A_total, infer n_bars from A_total / (π*(d/2)^2)
            A_bar = np.pi * (d_bar / 2.0) ** 2
            if model.steel_A_total > 0 and A_bar > 0:
                n_bars_total = model.steel_A_total / A_bar
                perimeter_total = n_bars_total * np.pi * d_bar
            else:
                perimeter_total = np.pi * d_bar  # Default: 1 bar

    # Subdomain manager (FASE D)
    subdomain_mgr = getattr(model, 'subdomain_mgr', None)

    cracks: list[XFEMCrack] = []
    dofs = build_xfem_dofs_multi(
        nodes, elems, cracks, ny,
        rebar_segs=rebar_segs,
        enable_bond_slip=enable_bond_slip,
    )
    q_n = np.zeros(dofs.ndof, dtype=float)

    # Phase-1: cohesive history in flat arrays (Numba-friendly).
    coh_states: CohesiveStateArrays = CohesiveStateArrays.zeros(
        n_primary=int(max_cracks), nelem=int(elems.shape[0]), ngp=2
    )

    # Phase-2: bulk material history (CDP/DP support)
    bulk_kind, bulk_params = pack_bulk_params(model)
    # Estimate max integration points per element: uncut elements have 4 IPs,
    # cut elements can have up to 4 + 4*nsub_cut*nsub_cut with nsub_cut=4 => 68 IPs
    max_ip_per_elem = 4 + 4 * 4 * 4
    bulk_states: BulkStateArrays = BulkStateArrays.zeros(
        nelem=int(elems.shape[0]), max_ip=int(max_ip_per_elem)
    )

    # Initialize constitutive model (if not using Numba)
    material = None
    if bulk_kind == 0:
        # Use Python material (retrocompat or ConcreteCDPReal)
        bulk_mat = str(getattr(model, "bulk_material", "elastic")).lower().strip()
        if bulk_mat in ("cdp", "concrete"):
            from xfem_clean.constitutive import ConcreteCDP
            material = ConcreteCDP(
                E=float(model.E),
                nu=float(model.nu),
                ft=float(model.ft),
                fc=float(model.fc),
                Gf_t=float(model.Gf),
                lch=float(model.lch),
                phi_deg=float(getattr(model, "dp_phi_deg", 30.0)),
                cohesion=float(getattr(model, "dp_cohesion", None) or (0.25 * model.fc)),
                H=float(getattr(model, "dp_H", 0.0)),
            )

    results = []

    # candidate points
    pts_flex = _candidate_points_zone(model, "flexure", nx)
    pts_shear = _candidate_points_zone(model, "shear_left", nx) + _candidate_points_zone(model, "shear_right", nx)

    # Helper function to resolve bc_spec to fixed DOF dict (FASE D)
    def _build_fixed_from_bc_spec(dofs_obj, u_target: float, nnode: int) -> Dict[int, float]:
        """
        Convert bc_spec to fixed DOF dict compatible with multicrack.

        bc_spec uses global DOF indices, which we convert to multicrack fixed dict.
        Handles negative markers for steel DOFs (same as analysis_single).
        """
        if bc_spec is None:
            return {}

        fixed_dict = {}
        case_id = getattr(model, "case_id", "unknown")

        def _marker_context(dof_marker: int) -> Optional[str]:
            marker_meta = getattr(bc_spec, "steel_dof_markers", None)
            if marker_meta and dof_marker in marker_meta:
                return marker_meta[dof_marker].get("layer_id")
            return None

        # Fixed DOFs from bc_spec
        for dof, val in bc_spec.fixed_dofs.items():
            fixed_dict[int(dof)] = float(val)

        # Prescribed DOFs (load DOFs)
        prescribed_scale = bc_spec.prescribed_scale if bc_spec is not None else -1.0
        for dof_marker in bc_spec.prescribed_dofs:
            if dof_marker < 0:
                # Negative marker for steel DOF: -(2*nnode + node_id * 2)
                node_id = -(dof_marker + 2 * nnode) // 2
                component = 0  # ux for pullout

                layer_id_meta = _marker_context(dof_marker)

                if dofs_obj.steel is None or dofs_obj.steel_nodes is None:
                    if node_id < 0 or node_id >= nnode:
                        raise ValueError(
                            "Invalid steel DOF mapping: "
                            f"case_id={case_id}, node_id={node_id}, "
                            f"layer_id={layer_id_meta}, dof_marker={dof_marker}, "
                            "reason=node_id out of range"
                        )
                    fallback_dof = int(dofs_obj.std[node_id, component])
                    fixed_dict[fallback_dof] = prescribed_scale * float(u_target)
                    continue
                if node_id < 0 or node_id >= nnode:
                    raise ValueError(
                        "Invalid steel DOF mapping: "
                        f"case_id={case_id}, node_id={node_id}, "
                        f"layer_id={layer_id_meta}, dof_marker={dof_marker}, "
                        "reason=node_id out of range"
                    )
                if not dofs_obj.steel_nodes[node_id]:
                    raise ValueError(
                        "Invalid steel DOF mapping: "
                        f"case_id={case_id}, node_id={node_id}, "
                        f"layer_id={layer_id_meta}, dof_marker={dof_marker}, "
                        "reason=no steel DOFs allocated for node"
                    )
                steel_dof = dofs_obj.steel[node_id, component]
                if steel_dof < 0:
                    raise ValueError(
                        "Invalid steel DOF mapping: "
                        f"case_id={case_id}, node_id={node_id}, "
                        f"layer_id={layer_id_meta}, dof_marker={dof_marker}, "
                        "reason=steel DOF index invalid"
                    )
                fixed_dict[int(steel_dof)] = prescribed_scale * float(u_target)
            else:
                # Positive: concrete DOF
                fixed_dict[int(dof_marker)] = prescribed_scale * float(u_target)

        # Enriched DOFs at constrained nodes must also be fixed
        for a in range(nnode):
            for comp in (0, 1):
                d_std = int(dofs_obj.std[a, comp])
                if d_std in fixed_dict:
                    for Hk in dofs_obj.H:
                        dH = int(Hk[a, comp])
                        if dH >= 0:
                            fixed_dict[dH] = 0.0

        return fixed_dict

    def solve_step(u_bar, q_init, coh_states_committed, bulk_states_committed, *, enr_scale: float = 1.0, bond_gamma: float = 1.0):
        """Newton solve for one load/displacement level.

        Important: cohesive states are *not* mutated inside Newton; we always
        assemble from the committed states and obtain a trial state for the
        current iterate.
        """
        q = q_init.copy()
        # Defensive: make sure the initial guess matches the current DOF layout.
        # Adaptive substepping stores guesses on a stack; if the enrichment basis
        # changes (new crack / growth) between stack creation and use, the stored
        # guess can have a stale size.
        if q.shape[0] != int(dofs.ndof):
            nd = int(dofs.ndof)
            if q.shape[0] < nd:
                q = np.pad(q, (0, nd - q.shape[0]), mode="constant", constant_values=0.0)
            else:
                q = q[:nd].copy()

        # Build a Dirichlet map in global DOF indices (supports + displacement control)
        def build_fixed(u_target: float) -> Dict[int, float]:
            if bc_spec is not None:
                # Use bc_spec (FASE D)
                return _build_fixed_from_bc_spec(dofs, u_target, nodes.shape[0])
            else:
                # Default 3PB beam BCs
                f = {
                    int(dofs.std[left_node, 0]): 0.0,
                    int(dofs.std[left_node, 1]): 0.0,
                    int(dofs.std[right_node, 1]): 0.0,
                    int(dofs.std[load_node, 1]): -u_target,
                }
                # Enriched DOFs at constrained nodes must also be fixed to avoid singular modes.
                # Multi-crack implementation uses ONLY Heaviside enrichments (no tip DOFs).
                nnode = int(dofs.std.shape[0])
                for a in range(nnode):
                    for comp in (0, 1):
                        d_std = int(dofs.std[a, comp])
                        if d_std in f:
                            for Hk in dofs.H:
                                dH = int(Hk[a, comp])
                                if dH >= 0:
                                    f[dH] = 0.0
                return f

        fixed_step = build_fixed(u_bar)
        for dof, val in fixed_step.items():
            q[dof] = val

        last_aux_pos = np.zeros((0, 2), dtype=float)
        last_aux_sig = np.zeros((0, 3), dtype=float)
        last_fint = np.zeros(dofs.ndof, dtype=float)
        res0 = None  # reference residual for relative tolerance (set at first Newton iteration)
        last_res = None
        last_tol = None
        last_fscale = None

        for it in range(model.newton_maxit):
            K, fint, fext, coh_updates, bulk_updates, aux_pos, aux_sig, bond_updates = assemble_xfem_system_multi(
                nodes,
                elems,
                q,
                dofs,
                cracks,
                model,
                C,
                rebar_segs,
                coh_states_committed,
                u_bar,
                model.visc_damp,
                law,
                use_numba=use_numba,
                coh_params=coh_params,
                enr_scale=float(enr_scale),
                bulk_states=bulk_states_committed,
                bulk_kind=bulk_kind,
                bulk_params=bulk_params,
                material=material,
                # Bond-slip (FASE D)
                bond_law=bond_law,
                bond_states_comm=bond_states,
                enable_bond_slip=enable_bond_slip,
                steel_EA=steel_EA,
                perimeter_total=perimeter_total,  # FASE D
                bond_gamma=bond_gamma,  # BLOQUE A: Bond-slip continuation parameter
                # Subdomain support (FASE D)
                subdomain_mgr=subdomain_mgr,
            )

            # Apply nodal forces from bc_spec (e.g., axial load for walls)
            if bc_spec is not None and bc_spec.nodal_forces is not None:
                for dof, force_val in bc_spec.nodal_forces.items():
                    fext[dof] += force_val

            R = fint - fext
            _raise_on_nonfinite(
                "assemble",
                {
                    "fint": fint,
                    "fext": fext,
                    "K": K,
                    "R": R,
                },
                bulk_kind=bulk_kind,
                enable_bond_slip=enable_bond_slip,
                use_numba=use_numba,
                case_id=getattr(model, "case_id", "unknown"),
                u_bar=u_bar,
            )
            free, K_ff, r_f, _ = apply_dirichlet(K, R, fixed_step, q)
            rhs = -r_f

            last_aux_pos, last_aux_sig = aux_pos, aux_sig
            last_fint = fint
            # Note: we deliberately do NOT commit cohesive/bulk history during
            # Newton iterations. History is only committed on convergence.

            res = float(np.linalg.norm(rhs))            # Gutierrez (Eq. 4.59): ||R|| / ||F_ext|| <= beta.
            # With displacement control, use the current reaction at the loaded dofs as force scale.
            if bc_spec is not None and bc_spec.reaction_dofs:
                load_dofs = []
                nnode = int(nodes.shape[0])
                for dm in bc_spec.reaction_dofs:
                    dm = int(dm)
                    if dm < 0:
                        node_id = -(dm + 2 * nnode) // 2
                        comp = 0
                        if (
                            dofs.steel is not None
                            and dofs.steel_nodes is not None
                            and 0 <= node_id < nnode
                            and dofs.steel_nodes[node_id]
                        ):
                            load_dofs.append(int(dofs.steel[node_id, comp]))
                        else:
                            load_dofs.append(int(dofs.std[node_id, comp]))
                    else:
                        load_dofs.append(dm)
            else:
                load_dofs = [int(dofs.std[load_node, 1])] if load_node is not None else []

            if len(load_dofs) > 0:
                P_est = -float(np.sum(R[load_dofs]))
                fscale = max(1.0, abs(P_est))
            else:
                P_est = 0.0
                fscale = 1.0
            tol = model.newton_tol_r + model.newton_beta * fscale
            last_res = res
            last_tol = tol
            last_fscale = fscale
            if res < tol:
                # Commit cohesive and bulk states
                if isinstance(coh_states_committed, CohesiveStateArrays):
                    assert isinstance(coh_updates, CohesiveStatePatch)
                    coh_trial = coh_states_committed.copy()
                    coh_updates.apply_to(coh_trial)
                else:
                    coh_trial = coh_updates

                if isinstance(bulk_states_committed, BulkStateArrays):
                    assert isinstance(bulk_updates, BulkStatePatch)
                    bulk_trial = bulk_states_committed.copy()
                    bulk_updates.apply_to(bulk_trial)
                else:
                    bulk_trial = bulk_updates
                return True, q, coh_trial, bulk_trial, aux_pos, aux_sig, "res", it + 1, fint, last_res, last_tol, last_fscale

            lsmr_meta = None
            reg_lambda = None
            solver_tag = "spsolve"
            rhs_norm = float(np.linalg.norm(rhs))

            def _linear_relres(K_local, du_local) -> float:
                return float(np.linalg.norm(K_local @ du_local - rhs) / (rhs_norm + 1.0))

            def _is_bad_solution(du_local, relres_local, max_du_limit=1e3) -> bool:
                if du_local is None or not np.all(np.isfinite(du_local)):
                    return True
                if du_local.size:
                    if float(np.max(np.abs(du_local))) > max_du_limit:
                        return True
                return relres_local > 1e-6

            du_f = None
            relres = float("inf")

            try:
                du_f = spla.spsolve(K_ff, rhs)
                relres = _linear_relres(K_ff, du_f)
            except Exception:
                du_f = None

            if _is_bad_solution(du_f, relres):
                lsmr_out = spla.lsmr(K_ff, rhs, atol=1e-12, btol=1e-12, maxiter=2000)
                du_f = lsmr_out[0]
                lsmr_meta = lsmr_out
                solver_tag = "lsmr"
                relres = _linear_relres(K_ff, du_f)

            if _is_bad_solution(du_f, relres):
                diag = K_ff.diagonal()
                diag_abs = np.abs(diag)
                diag_max = float(diag_abs.max()) if diag_abs.size else 0.0
                lam = 1e-8 * max(1.0, diag_max)
                for _attempt in range(3):
                    reg_lambda = lam
                    K_reg = K_ff + lam * sp.eye(K_ff.shape[0], format="csr")
                    try:
                        du_candidate = spla.spsolve(K_reg, rhs)
                        solver_tag = "reg-spsolve"
                    except Exception:
                        lsmr_out = spla.lsmr(K_reg, rhs, atol=1e-12, btol=1e-12, maxiter=2000)
                        du_candidate = lsmr_out[0]
                        lsmr_meta = lsmr_out
                        solver_tag = "reg-lsmr"
                    relres_candidate = _linear_relres(K_reg, du_candidate)
                    if not _is_bad_solution(du_candidate, relres_candidate):
                        du_f = du_candidate
                        relres = relres_candidate
                        break
                    lam *= 10.0

            if du_f is None:
                du_f = np.zeros_like(rhs)

            norm_du = float(np.linalg.norm(du_f))
            max_du = float(np.max(np.abs(du_f))) if du_f.size else 0.0
            q_trial = q.copy()
            q_trial[free] += du_f
            for dof, val in fixed_step.items():
                q_trial[dof] = val
            max_q_trial = float(np.max(np.abs(q_trial))) if q_trial.size else 0.0
            diag = K_ff.diagonal()
            diag_abs = np.abs(diag)
            diag_max = float(diag_abs.max()) if diag_abs.size else 0.0
            diag_thresh = 1e-12 * max(1.0, diag_max)
            near_zero_diag = int(np.sum(diag_abs < diag_thresh))
            condA = None
            istop = None
            normr = None
            if lsmr_meta is not None:
                condA = float(lsmr_meta[6])
                istop = int(lsmr_meta[1])
                normr = float(lsmr_meta[3])
            if _is_bad_solution(du_f, relres) or (condA is not None and condA > 1e16):
                why = (
                    "illcond("
                    f"max|dq|={max_du:.2e}, relres={relres:.2e}, diag_max={diag_max:.2e}, "
                    f"lambda={reg_lambda if reg_lambda is not None else 0.0:.2e}, "
                    f"solver={solver_tag}, condA={condA if condA is not None else 'n/a'}"
                    ")"
                )
                return False, q, coh_states_committed, bulk_states_committed, aux_pos, aux_sig, why, it + 1, fint, last_res, last_tol, last_fscale
            # Stagnation check: use absolute tolerance only (no displacement scaling)
            # Previous version scaled by u_scale which made it too strict for small displacements
            if norm_du < model.newton_tol_du:
                return False, q, coh_states_committed, bulk_states_committed, aux_pos, aux_sig, "stagnated", it + 1, fint, last_res, last_tol, last_fscale

            # Line search (optional): backtracking on residual norm
            alpha = 1.0
            if model.line_search:
                r0 = float(np.linalg.norm(r_f))
                accepted = False
                for _ls in range(8):
                    q_try = q.copy()
                    q_try[free] += alpha * du_f
                    for dof, val in fixed_step.items():
                        q_try[dof] = val

                    K_t, fint_t, fext_t, _coh_upd_t, _bulk_upd_t, aux_pos_t, aux_sig_t, _ = assemble_xfem_system_multi(
                        nodes,
                        elems,
                        q_try,
                        dofs,
                        cracks,
                        model,
                        C,
                        rebar_segs,
                        coh_states_committed,
                        u_bar,
                        model.visc_damp,
                        law,
                        use_numba=use_numba,
                        coh_params=coh_params,
                        enr_scale=float(enr_scale),
                        bulk_states=bulk_states_committed,
                        bulk_kind=bulk_kind,
                        bulk_params=bulk_params,
                        material=material,
                        # Bond-slip (FASE D)
                        bond_law=bond_law,
                        bond_states_comm=bond_states,
                        enable_bond_slip=enable_bond_slip,
                        steel_EA=steel_EA,
                        perimeter_total=perimeter_total,  # FASE D
                        bond_gamma=bond_gamma,  # BLOQUE A: Bond-slip continuation parameter
                        # Subdomain support (FASE D)
                        subdomain_mgr=subdomain_mgr,
                    )
                    r_try = (fint_t - fext_t)[free]
                    _raise_on_nonfinite(
                        "line-search",
                        {
                            "fint": fint_t,
                            "fext": fext_t,
                            "K": K_t,
                            "R": fint_t - fext_t,
                        },
                        bulk_kind=bulk_kind,
                        enable_bond_slip=enable_bond_slip,
                        use_numba=use_numba,
                        case_id=getattr(model, "case_id", "unknown"),
                        u_bar=u_bar,
                    )
                    r1 = float(np.linalg.norm(r_try))
                    if r1 <= r0:
                        q = q_try
                        last_aux_pos, last_aux_sig = aux_pos_t, aux_sig_t
                        last_fint = fint_t
                        accepted = True
                        break
                    alpha *= 0.5

                if accepted:
                    continue

            # No (or failed) line search: full step
            q[free] += du_f
            for dof, val in fixed_step.items():
                q[dof] = val

        return False, q, coh_states_committed, bulk_states_committed, last_aux_pos, last_aux_sig, "maxit", model.newton_maxit, last_fint, last_res, last_tol, last_fscale

    def ramp_solve_step(u_bar, q_init, coh_committed, bulk_committed, bond_gamma: float = 1.0):
        """Gutierrez-style adaptive ramping/continuation after init/grow.

        We solve the same displacement level multiple times while gradually
        activating enrichment/cohesive contributions. This greatly improves
        robustness right after topology changes (new crack or growth) where
        a direct Newton step can stagnate.

        Returns the same tuple as `solve_step` with `enr_scale=1.0` on success.
        """

        # Defaults tuned for robustness (cheap because alpha=0 solves instantly).
        a0 = float(getattr(model, "ramp_alpha0", 0.0))
        da = float(getattr(model, "ramp_dalpha0", 0.25))
        da_min = float(getattr(model, "ramp_dalpha_min", 0.02))

        a = max(0.0, min(1.0, a0))
        q_cur = q_init
        coh_cur = coh_committed
        bulk_cur = bulk_committed

        # Always do an initial solve at alpha=a (including a=0)
        ok, q_cur, coh_cur, bulk_cur, aux_pos, aux_sig, why, iters, fint_last, last_res, last_tol, last_fscale = solve_step(
            u_bar, q_cur, coh_cur, bulk_cur, enr_scale=a, bond_gamma=bond_gamma
        )
        if not ok:
            return False, q_cur, coh_cur, bulk_cur, aux_pos, aux_sig, why, iters, fint_last, last_res, last_tol, last_fscale

        # Continuation to full enrichment
        while a < 1.0:
            a_try = min(1.0, a + da)
            ok, q_try, coh_try, bulk_try, aux_pos, aux_sig, why, iters, fint_last, last_res, last_tol, last_fscale = solve_step(
                u_bar, q_cur, coh_cur, bulk_cur, enr_scale=a_try, bond_gamma=bond_gamma
            )
            if ok:
                a = a_try
                q_cur = q_try
                coh_cur = coh_try
                bulk_cur = bulk_try
                da = min(0.5, da * 1.5)
                continue

            # failed: reduce ramp increment
            da *= 0.5
            if da < da_min:
                return False, q_try, coh_try, bulk_try, aux_pos, aux_sig, why, iters, fint_last, last_res, last_tol, last_fscale

        return True, q_cur, coh_cur, bulk_cur, aux_pos, aux_sig, "ramp", 0, fint_last, last_res, last_tol, last_fscale

    # adaptive substepping stack (same logic as single)
    total_substeps = 0
    last_res = None
    last_tol = None
    last_fscale = None
    last_reason = None
    stop_requested = False

    for istep, u1 in enumerate(u_targets, start=1):
        if stop_requested:
            break
        u0 = results[-1]["u"] if results else 0.0
        bond_comm_init = bond_states.copy() if bond_states is not None else None
        stack = [(0, u0, u1, q_n.copy(), coh_states.copy(), bulk_states.copy(), bond_comm_init)]

        while stack:
            lvl, ua, ub, q_start, coh_comm, bulk_comm, bond_comm = stack.pop()
            # Restore bond_states for this substep
            if bond_comm is not None:
                bond_states = bond_comm
            du = ub - ua
            print(f"[substep] lvl={lvl:02d} u0={ua*1e3:5.3f}mm -> u1={ub*1e3:5.3f}mm  du={du*1e3:5.3f}mm  ncr={len([c for c in cracks if c.active])}")
            total_substeps += 1
            if total_substeps > model.max_total_substeps:
                res_str = f"{last_res:.6e}" if last_res is not None else "n/a"
                tol_str = f"{last_tol:.6e}" if last_tol is not None else "n/a"
                fscale_str = f"{last_fscale:.6e}" if last_fscale is not None else "n/a"
                reason_str = last_reason if last_reason is not None else "n/a"
                raise RuntimeError(
                    "Anti-hang guardrail triggered: max_total_substeps exceeded.\n"
                    f"u0={ua:.6e}, u1={ub:.6e}, du={du:.6e}, lvl={lvl}, maxit={model.newton_maxit}, "
                    f"last||rhs||={res_str}, tol={tol_str}, fscale={fscale_str}, reason={reason_str}, "
                    f"bond_slip={'on' if enable_bond_slip else 'off'}"
                )

            # BLOQUE 4: Bond-slip gamma continuation (multicrack)
            # If bond-slip is enabled and gamma ramping is active, solve multiple times
            # with increasing gamma to improve convergence (same as single-crack solver)
            use_gamma_ramp = (
                enable_bond_slip
                and bond_states is not None
                and model.bond_gamma_strategy == "ramp_steps"
                and model.bond_gamma_ramp_steps > 1
            )

            if use_gamma_ramp:
                # Create gamma sequence: [gamma_min, ..., gamma_max=1.0]
                gamma_vals = np.linspace(
                    float(model.bond_gamma_min),
                    float(model.bond_gamma_max),
                    int(model.bond_gamma_ramp_steps),
                )
                # Ensure gamma=1.0 is included
                if gamma_vals[-1] != 1.0:
                    gamma_vals = np.append(gamma_vals, 1.0)

                if getattr(model, "debug_substeps", False):
                    print(f"    [bond-gamma] ramp sequence: {gamma_vals}")

                # Ramp through gammas
                q_cur = q_start
                coh_cur = coh_comm
                bulk_cur = bulk_comm
                ok_final = False

                for i_gamma, gamma in enumerate(gamma_vals):
                    if getattr(model, "debug_substeps", False) or getattr(model, "debug_bond_gamma", False):
                        print(f"    [bond-gamma] step={i_gamma+1}/{len(gamma_vals)}, gamma={gamma:.3f}, u={ub*1e3:.3f}mm")

                    ok, q_sol, coh_trial, bulk_trial, aux_pos, aux_sig, why, iters, fint_last, last_res, last_tol, last_fscale = solve_step(
                        ub, q_cur, coh_cur, bulk_cur, bond_gamma=gamma
                    )
                    last_reason = why

                    if not ok:
                        if getattr(model, "debug_substeps", False):
                            print(f"    [bond-gamma] failed at gamma={gamma:.3f}")
                        # If a gamma substep fails, abandon this increment (trigger substepping)
                        ok_final = False
                        break

                    # Accept this gamma step (use as initial guess for next gamma)
                    q_cur = q_sol
                    coh_cur = coh_trial
                    bulk_cur = bulk_trial
                    ok_final = True

                # Use final result
                ok = ok_final
                if ok:
                    q_sol = q_cur
                    coh_trial = coh_cur
                    bulk_trial = bulk_cur
            else:
                # No gamma ramp: single solve_step with gamma=1.0 (default)
                ok, q_sol, coh_trial, bulk_trial, aux_pos, aux_sig, why, iters, fint_last, last_res, last_tol, last_fscale = solve_step(
                    ub, q_start, coh_comm, bulk_comm, bond_gamma=1.0
                )
                last_reason = why
            if ok:
                print(f"    [newton] converged({why}) it={iters:02d} ||rhs||=OK u={ub*1e3:.3f}mm")

                # ------------------------------------------------------------
                # Gutierrez-style inner loop:
                # After any crack initiation/growth, rebuild DOFs and **re-solve**
                # at the *same* ub until no further crack update is triggered.
                # This avoids carrying a non-equilibrated state into the next step.
                # ------------------------------------------------------------
                cracks_before = copy.deepcopy(cracks)
                dofs_before = dofs

                q_loc = q_sol
                coh_loc = coh_trial
                bulk_loc = bulk_trial
                aux_pos_loc = aux_pos
                aux_sig_loc = aux_sig
                fint_loc = fint_last

                inner_failed = False

                for _inner in range(int(model.crack_max_inner)):
                    changed = False
                    initiated = False
                    min_spacing = min_crack_spacing_factor * dx

                    # ---- Crack initiation ----
                    nactive = len([c for c in cracks if c.active])
                    allow_diag = (crack_mode.lower() == "option2") and (ub * 1e3 >= u_diag_mm)

                    if nactive < max_cracks:
                        zones = ["flexure"]
                        if allow_diag:
                            zones.append("shear")

                        best = None
                        for zone in zones:
                            pts = pts_flex if zone == "flexure" else pts_shear
                            for (x0, y0) in pts:
                                if _too_close_to_existing(x0, y0, cracks, min_spacing):
                                    continue
                                sigbar = nonlocal_bar_stress(
                                    aux_pos_loc, aux_sig_loc, x0, y0 + 1e-6, rho, y_max=stop_y, r_cut=r_cut
                                )
                                s1, _ = principal_max_dir(sigbar)
                                if best is None or s1 > best[0]:
                                    best = (s1, x0, y0, sigbar)

                        if best is not None and best[0] >= model.ft * float(model.ft_initiation_factor):
                            s1, x0, y0, sigbar = best

                            # In option1 force vertical-ish flexural cracks by overriding direction
                            if crack_mode.lower() == "option1":
                                crack = XFEMCrack(
                                    x0=float(x0), y0=float(y0),
                                    tip_x=float(x0), tip_y=float(y0 + dy),
                                    stop_y=float(stop_y), angle_deg=90.0, active=True
                                )
                            else:
                                crack = _init_crack_from_stress(x0, y0, sigbar, model)

                            cracks.append(crack)
                            initiated = True
                            print(
                                f"[crack] initiate #{len(cracks)} at ({crack.x0:.4f},{crack.y0:.4f}) "
                                f"tip=({crack.tip_x:.4f},{crack.tip_y:.4f}) angle={crack.angle_deg:.1f}° "
                                f"s1={s1/1e6:.3f}MPa ft={model.ft/1e6:.3f}MPa"
                            )
                            changed = True

                    # ---- Crack propagation: grow at most one crack per inner iteration ----
                    if cracks:
                        grow_best = None
                        for idx, c in enumerate(cracks):
                            if not c.active:
                                continue
                            if c.tip_y >= stop_y:
                                continue
                            px = c.tip_x + 0.25 * dy * math.cos(math.radians(c.angle_deg))
                            py = c.tip_y + 0.25 * dy * math.sin(math.radians(c.angle_deg))
                            sigbar = nonlocal_bar_stress(
                                aux_pos_loc, aux_sig_loc, px, py, rho, y_max=stop_y, r_cut=r_cut
                            )
                            s1, _ = principal_max_dir(sigbar)
                            if s1 >= model.ft * float(model.ft_initiation_factor):
                                if grow_best is None or s1 > grow_best[0]:
                                    grow_best = (s1, idx, sigbar)

                        if grow_best is not None:
                            s1, idx, sigbar = grow_best
                            c = cracks[idx]
                            s1_tip, v1 = principal_max_dir(sigbar)
                            t = np.array([-v1[1], v1[0]], dtype=float)
                            if t[1] < 0:
                                t = -t
                            if c.x0 < 0.5 * model.L and t[0] < 0:
                                t = -t
                            if c.x0 > 0.5 * model.L and t[0] > 0:
                                t = -t
                            t = t / max(1e-14, float(np.linalg.norm(t)))

                            c.tip_x = float(c.tip_x + dy * float(t[0]))
                            c.tip_y = float(c.tip_y + dy * float(t[1]))
                            c.angle_deg = float(math.degrees(math.atan2(t[1], t[0])))
                            print(
                                f"[crack] grow #{idx+1} tip=({c.tip_x:.4f},{c.tip_y:.4f}) "
                                f"angle={c.angle_deg:.1f}° s1={s1/1e6:.3f}MPa"
                            )
                            changed = True

                    # ---- Junction detection (P0: crack coalescence) ----
                    # After crack growth, check if any cracks have coalesced
                    tol_merge = getattr(model, "junction_merge_tolerance", 0.01)  # 10mm default
                    junctions = detect_crack_coalescence(cracks, nodes, elems, tol_merge=tol_merge)

                    if junctions:
                        for junc in junctions:
                            print(
                                f"[junction] detected: crack#{junc.secondary_crack_id+1} → "
                                f"crack#{junc.main_crack_id+1} at ({junc.junction_point[0]:.4f},{junc.junction_point[1]:.4f}) "
                                f"in elem#{junc.element_id}"
                            )
                            # Arrest secondary crack at junction
                            arrest_secondary_crack_at_junction(junc, cracks)
                            print(f"[junction] arrested crack#{junc.secondary_crack_id+1}, added junction enrichment")
                            changed = True

                    if not changed:
                        break

                    # Rebuild DOFs and transfer equilibrium guess, then re-solve at same ub
                    dofs_new = build_xfem_dofs_multi(
                        nodes, elems, cracks, ny,
                        rebar_segs=rebar_segs,
                        enable_bond_slip=enable_bond_slip,
                    )
                    q_loc = transfer_q_between_dofs_multi(q_loc, dofs, dofs_new)
                    dofs = dofs_new

                    # Always re-equilibrate at the same u_bar after *initiation* or *growth*.
                    # Use adaptive continuation (enrichment/cohesive ramping) to stabilize the Newton solve.
                    ok2, q_loc, coh_loc, bulk_loc, aux_pos_loc, aux_sig_loc, why2, it2, fint_loc, last_res, last_tol, last_fscale = ramp_solve_step(
                        ub, q_loc, coh_loc, bulk_loc
                    )
                    last_reason = why2
                    if not ok2:
                        print(f"    [inner] re-solve failed({why2}) it={it2:02d} at u={ub*1e3:.3f}mm -> will subdivide")
                        cracks[:] = cracks_before
                        dofs = dofs_before
                        inner_failed = True
                        ok = False
                        why = why2
                        iters = it2
                        break
                    else:
                        print(f"    [inner] re-solve converged({why2}) it={it2:02d} at u={ub*1e3:.3f}mm")

                if inner_failed:
                    # fall through to bisection logic below
                    pass
                else:
                    # Accept final equilibrium at ub (after inner crack updates)
                    q_n = q_loc
                    coh_states = coh_loc
                    bulk_states = bulk_loc
                    # Bond-slip commit (FASE D): bond_states are already committed via bond_comm in stack
                    # (bond_updates is set during assembly but not available in this scope)                    # Extract reaction force (FASE D: bc_spec support)
                    if bc_spec is not None and bc_spec.reaction_dofs:
                        dof_loads = []
                        nnode = int(nodes.shape[0])
                        for dm in bc_spec.reaction_dofs:
                            dm = int(dm)
                            if dm < 0:
                                node_id = -(dm + 2 * nnode) // 2
                                comp = 0
                                if (
                                    dofs.steel is not None
                                    and dofs.steel_nodes is not None
                                    and 0 <= node_id < nnode
                                    and dofs.steel_nodes[node_id]
                                ):
                                    dof_loads.append(int(dofs.steel[node_id, comp]))
                                else:
                                    dof_loads.append(int(dofs.std[node_id, comp]))
                            else:
                                dof_loads.append(dm)
                    else:
                        dof_loads = [int(dofs.std[load_node, 1])] if load_node is not None else []

                    P = -float(np.sum(fint_loc[dof_loads])) if len(dof_loads) > 0 else 0.0

                    results.append({
                        "step": istep,
                        "u": float(ub),
                        "P": float(P),
                        "ncr": len([c for c in cracks if c.active]),
                    })

                    # Optional early stop: useful for "until first crack" regressions.
                    if getattr(model, "stop_at_first_crack", False) and results[-1]["ncr"] > 0:
                        print("    [stop] first crack detected -> stopping analysis early")
                        stop_requested = True
                        stack.clear()

                    # When the first half converges, the next substep on the stack (if any)
                    # starts at this accepted `ub`. Update its initial state accordingly.
                    if stack:
                        lvl_n, ua_n, ub_n, _q_s, _coh_s, _bulk_s, _bond_s = stack[-1]
                        if abs(float(ua_n) - float(ub)) < 1e-14:
                            bond_copy = bond_states.copy() if bond_states is not None else None
                            stack[-1] = (lvl_n, ua_n, ub_n, q_n.copy(), coh_states.copy(), bulk_states.copy(), bond_copy)

                    continue

            # not ok -> subdivide
            print(f"    [newton] failed({why}) it={iters:02d} u={ub*1e3:.3f}mm")
            du_min = float(
                getattr(
                    model,
                    "substep_du_min",
                    getattr(model, "newton_tol_du_abs", model.newton_tol_du),
                )
            )
            if abs(float(du)) < du_min:
                res_str = f"{last_res:.6e}" if last_res is not None else "n/a"
                tol_str = f"{last_tol:.6e}" if last_tol is not None else "n/a"
                fscale_str = f"{last_fscale:.6e}" if last_fscale is not None else "n/a"
                raise RuntimeError(
                    "Substepping stalled: |du| below minimum.\n"
                    f"u0={ua:.6e}, u1={ub:.6e}, du={du:.6e}, lvl={lvl}, maxit={model.newton_maxit}, "
                    f"last||rhs||={res_str}, tol={tol_str}, fscale={fscale_str}, "
                    f"bond_slip={'on' if enable_bond_slip else 'off'}"
                )
            if lvl >= model.max_subdiv:
                res_str = f"{last_res:.6e}" if last_res is not None else "n/a"
                tol_str = f"{last_tol:.6e}" if last_tol is not None else "n/a"
                fscale_str = f"{last_fscale:.6e}" if last_fscale is not None else "n/a"
                raise RuntimeError(
                    "Substepping exceeded max_subdiv.\n"
                    f"u0={ua:.6e}, u1={ub:.6e}, du={du:.6e}, lvl={lvl}, maxit={model.newton_maxit}, "
                    f"last||rhs||={res_str}, tol={tol_str}, fscale={fscale_str}, "
                    f"bond_slip={'on' if enable_bond_slip else 'off'}"
                )

            um = 0.5*(ua + ub)
            bond_copy = bond_comm.copy() if bond_comm is not None else None
            stack.append((lvl+1, um, ub, q_start.copy(), coh_comm.copy(), bulk_comm.copy(), bond_copy))
            stack.append((lvl+1, ua, um, q_start.copy(), coh_comm.copy(), bulk_comm.copy(), bond_copy))

    # Return format selection (BLOQUE 2: ResultBundle)
    if return_bundle:
        # Comprehensive bundle for postprocessing (compatible with postprocess_comprehensive.py)
        return {
            'nodes': nodes,
            'elems': elems,
            'u': q_n,
            'history': np.array(results),  # List of dicts, keep as object array
            'cracks': cracks,
            'bond_states': bond_states,
            'rebar_segs': rebar_segs,
            'dofs': dofs,
            'coh_states': coh_states,
            'bulk_states': bulk_states,
        }
    else:
        # Backward compatible tuple return
        return nodes, elems, q_n, results, cracks
