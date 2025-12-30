"""Displacement-controlled single-crack XFEM analysis driver."""

from __future__ import annotations

import copy
import math
from typing import Dict, Tuple, Optional, Union, List
from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla

from xfem_clean.cohesive_laws import CohesiveLaw, CohesiveState, cohesive_fracture_energy
from xfem_clean.numba.kernels_cohesive import pack_cohesive_law_params
from xfem_clean.numba.kernels_bulk import pack_bulk_params
from xfem_clean.material_point import MaterialPoint
from xfem_clean.crack_criteria import (
    candidate_points_bottom_edge_midpoints,
    nonlocal_bar_stress,
    principal_max_dir,
)
from xfem_clean.fem.bcs import apply_dirichlet
from xfem_clean.fem.mesh import structured_quad_mesh
from xfem_clean.rebar import prepare_rebar_segments, rebar_contrib

from xfem_clean.xfem.assembly_single import assemble_xfem_system
from xfem_clean.xfem.dofs_single import XFEMDofs, build_xfem_dofs, transfer_q_between_dofs
from xfem_clean.xfem.geometry import XFEMCrack
from xfem_clean.xfem.material import plane_stress_C
from xfem_clean.xfem.material_factory import make_bulk_material
from xfem_clean.xfem.model import XFEMModel
from xfem_clean.xfem.state_arrays import (
    BulkStateArrays,
    BulkStatePatch,
    CohesiveStateArrays,
    CohesiveStatePatch,
)


@dataclass
class BCSpec:
    """
    Boundary condition specification for XFEM analysis.

    Attributes
    ----------
    fixed_dofs : Dict[int, float]
        Fixed DOF values (e.g., {dof: value})
    prescribed_dofs : List[int]
        DOFs to prescribe displacement (for displacement control)
    prescribed_scale : float
        Scale factor for prescribed displacement (positive or negative)
    reaction_dofs : List[int]
        DOFs to measure reaction forces
    """
    fixed_dofs: Dict[int, float]
    prescribed_dofs: List[int]
    prescribed_scale: float = 1.0
    reaction_dofs: Optional[List[int]] = None


def run_analysis_xfem(
    model: XFEMModel,
    nx: int,
    ny: int,
    nsteps: int,
    umax: float,
    law: Optional[CohesiveLaw] = None,
    return_states: bool = False,
    bc_spec: Optional[BCSpec] = None,
    bond_law: Optional[Any] = None,
    return_bundle: bool = False,
    u_targets: Optional[np.ndarray] = None,
):
    """Run the single-crack XFEM prototype (stable linear version + cohesive).

    Parameters
    ----------
    bc_spec : BCSpec, optional
        Boundary condition specification. If None, defaults to 3-point bending:
        - Fixed: left bottom (ux=0, uy=0) and right bottom (uy=0)
        - Prescribed: top center nodes (uy=-umax)
    return_bundle : bool, optional
        If True, return dict with comprehensive results:
        {nodes, elems, u, history, crack, mp_states, bond_states, rebar_segs, dofs}
        If False (default), return tuple (backward compatible)
    u_targets : np.ndarray, optional
        Custom displacement trajectory [m]. If provided, overrides nsteps/umax.
        Enables cyclic loading: displacement control follows this exact sequence.
        If None (default), uses monotonic linspace(0, umax, nsteps).
    """

    # Use mesh from model if provided (avoids re-meshing in solver_interface)
    if hasattr(model, '_nodes') and hasattr(model, '_elems'):
        nodes = model._nodes
        elems = model._elems
    else:
        nodes, elems = structured_quad_mesh(model.L, model.H, nx, ny)

    nnode = nodes.shape[0]

    # Calculate element dimensions (needed for crack spacing, load width, etc.)
    dx = model.L / nx
    dy = model.H / ny

    # Default BCs: 3-point bending (for backward compatibility)
    if bc_spec is None:
        left = int(np.argmin(nodes[:, 0]))
        right = int(np.argmax(nodes[:, 0]))
        fixed_base = {2 * left: 0.0, 2 * left + 1: 0.0, 2 * right + 1: 0.0}

        y_top = model.H
        top_nodes = np.where(np.isclose(nodes[:, 1], y_top))[0]
        load_halfwidth = model.load_halfwidth if (model.load_halfwidth and model.load_halfwidth > 0.0) else 2.0 * dx
        load_nodes = top_nodes[np.where(np.abs(nodes[top_nodes, 0] - model.L / 2) <= load_halfwidth)[0]]
        if len(load_nodes) == 0:
            load_nodes = np.array([int(top_nodes[np.argmin(np.abs(nodes[top_nodes, 0] - model.L / 2))])], dtype=int)
        load_dofs = [2 * n + 1 for n in load_nodes]
    else:
        # Use BCs from bc_spec
        fixed_base = dict(bc_spec.fixed_dofs)
        load_dofs = list(bc_spec.prescribed_dofs)
        # NOTE: Negative DOFs mark steel nodes to be resolved after DOF manager is created

    rebar_segs = prepare_rebar_segments(nodes, cover=model.cover)

    if model.cand_windows is not None:
        windows = list(model.cand_windows)
    elif model.cand_mode == "three":
        windows = list(model.cand_windows_three)
    elif model.cand_mode == "dominant":
        windows = [model.dominant_window] if model.dominant_crack else [(0.0, 1.0)]
    else:
        windows = [(0.0, 1.0)]

    cps = candidate_points_bottom_edge_midpoints(
        nodes,
        elems,
        model.L,
        model.crack_margin,
        windows=windows,
        window=windows[0],
        spacing=dx,
    )

    x0 = model.L / 2
    if np.any(np.isclose(nodes[:, 0], x0)):
        x0 += 1e-6 * dx
    stop_y = model.crack_tip_stop_y if model.crack_tip_stop_y is not None else 0.5 * model.H
    if model.arrest_at_half_height:
        stop_y = float(min(stop_y, 0.5 * model.H))
    else:
        stop_y = float(stop_y)

    crack = XFEMCrack(x0=x0, y0=0.0, tip_x=x0, tip_y=0.0, stop_y=stop_y, active=False)

    # Bulk material model (retrocompatible default). The solver/assembly now
    # track integration-point states (MaterialPoint) so damage/plasticity
    # models (Drucker–Prager/CDP) can be integrated next.
    C = plane_stress_C(model.E, model.nu)
    material = make_bulk_material(model)

    if law is None:
        Kn = model.Kn_factor * model.E / max(1e-12, model.lch)
        law = CohesiveLaw(Kn=Kn, ft=model.ft, Gf=model.Gf)

    # Optional Numba kernels: pack cohesive parameters once.
    use_numba = bool(getattr(model, "use_numba", False))
    coh_params = pack_cohesive_law_params(law) if use_numba else None
    bulk_kind, bulk_params = pack_bulk_params(model) if use_numba else (0, None)

    results = []
    u_std = np.zeros(2 * nnode, dtype=float)

    # Phase-1: store IP history in flat arrays (Numba-friendly) while keeping
    # the public MaterialPoint/CohesiveState interfaces intact.
    #
    # Bulk integration uses 2×2 Gauss per subdomain and nsub=4 for cut elements:
    #   max_ip = 4 (canonical) + 4 * (nsub*nsub) (subcells) = 68
    MAX_IP_PER_ELEM = 4 + 4 * (4 * 4)

    coh_states: CohesiveStateArrays = CohesiveStateArrays.zeros(
        n_primary=1, nelem=int(elems.shape[0]), ngp=2
    )
    mp_states: BulkStateArrays = BulkStateArrays.zeros(
        nelem=int(elems.shape[0]), max_ip=int(MAX_IP_PER_ELEM)
    )
    dofs: Optional[XFEMDofs] = None

    # Bond-slip state initialization (Phase 2)
    bond_states = None
    if model.enable_bond_slip and rebar_segs is not None and len(rebar_segs) > 0:
        from xfem_clean.bond_slip import BondSlipStateArrays, BondSlipModelCode2010

        n_seg = rebar_segs.shape[0]
        bond_states = BondSlipStateArrays.zeros(n_seg)

        # Use bond_law from parameter if provided, otherwise create default
        if bond_law is None:
            bond_law = BondSlipModelCode2010(
                f_cm=model.fc,
                d_bar=model.rebar_diameter,
                condition=model.bond_condition,
            )

    # Subdomain manager (Phase C - thesis cases)
    subdomain_mgr = getattr(model, 'subdomain_mgr', None)

    # Bond-disabled x-range (for pullout empty elements)
    bond_disabled_x_range = getattr(model, 'bond_disabled_x_range', None)

    def _compute_global_dissipation(
        aux: Dict,
        mp: Union[Dict[Tuple[int, int], MaterialPoint], BulkStateArrays],
        coh: Union[Dict[Tuple[int, int], CohesiveState], CohesiveStateArrays],
    ) -> Dict[str, float]:
        """Compute global dissipations (J) using the same quadrature used in assembly."""
        W_pl = 0.0
        W_ft = 0.0
        W_fc = 0.0
        gp_eid = aux.get("gp_eid", [])
        gp_ipid = aux.get("gp_ipid", [])
        gp_wgt = aux.get("gp_weight", aux.get("gp_wgt", []))
        for e, ip, wgt in zip(gp_eid, gp_ipid, gp_wgt):
            ee = int(e)
            ii = int(ip)
            if isinstance(mp, BulkStateArrays):
                if ee < 0 or ee >= mp.nelem or ii < 0 or ii >= mp.max_ip or (not bool(mp.active[ee, ii])):
                    continue
                W_pl += float(wgt) * float(mp.w_plastic[ee, ii])
                W_ft += float(wgt) * float(mp.w_fract_t[ee, ii])
                W_fc += float(wgt) * float(mp.w_fract_c[ee, ii])
            else:
                st = mp.get((ee, ii), None)
                if st is None:
                    continue
                W_pl += float(wgt) * float(getattr(st, "w_plastic", 0.0))
                W_ft += float(wgt) * float(getattr(st, "w_fract_t", 0.0))
                W_fc += float(wgt) * float(getattr(st, "w_fract_c", 0.0))

        W_coh = 0.0
        coh_wgt = aux.get("coh_weight", aux.get("coh_wgt", []))
        coh_dm = aux.get("coh_delta_max", [])
        for wgt, dm in zip(coh_wgt, coh_dm):
            W_coh += float(wgt) * float(cohesive_fracture_energy(law, float(dm)))

        return dict(
            W_plastic=float(W_pl),
            W_damage_t=float(W_ft),
            W_damage_c=float(W_fc),
            W_cohesive=float(W_coh),
            W_fracture=float(W_ft + W_coh),
            W_diss_total=float(W_pl + W_ft + W_fc + W_coh),
        )

    def _std_only_dofs() -> XFEMDofs:
        """Create DOF mapping with only standard DOFs (and optionally steel DOFs for bond-slip)."""
        std = np.arange(2 * nnode, dtype=int).reshape(nnode, 2)
        H = -np.ones((nnode, 2), dtype=int)
        tipd = -np.ones((nnode, 4, 2), dtype=int)

        # Bond-slip: allocate steel DOFs even when crack is inactive
        steel = None
        steel_dof_offset = -1
        steel_nodes = None
        ndof = 2 * nnode

        if model.enable_bond_slip and rebar_segs is not None and len(rebar_segs) > 0:
            steel_dof_offset = ndof  # Steel DOFs start after standard DOFs
            steel_nodes = np.zeros(nnode, dtype=bool)
            steel = -np.ones((nnode, 2), dtype=int)

            # Identify nodes used by rebar segments
            for seg in rebar_segs:
                n1 = int(seg[0])
                n2 = int(seg[1])
                steel_nodes[n1] = True
                steel_nodes[n2] = True

            # Allocate steel DOFs for rebar nodes
            idx = ndof
            for n in np.where(steel_nodes)[0]:
                steel[n, 0] = idx
                steel[n, 1] = idx + 1
                idx += 2
            ndof = idx

        return XFEMDofs(
            std=std,
            H=H,
            tip=tipd,
            ndof=ndof,
            H_nodes=np.zeros(nnode, dtype=bool),
            tip_nodes=np.zeros(nnode, dtype=bool),
            steel=steel,
            steel_dof_offset=steel_dof_offset,
            steel_nodes=steel_nodes,
        )

    def _resolve_steel_dofs_in_bc_spec(dofs_obj: XFEMDofs):
        """
        Resolve negative DOF markers in bc_spec to actual steel DOFs.

        Modifies load_dofs and fixed_base to replace negative markers
        with actual steel DOFs from dofs_obj.
        """
        nonlocal load_dofs, fixed_base

        if bc_spec is None:
            return

        # Check if we have steel DOFs to resolve
        if dofs_obj.steel is None:
            return

        # Resolve prescribed_dofs (load_dofs)
        resolved_load_dofs = []
        for dof_marker in bc_spec.prescribed_dofs:
            if dof_marker < 0:
                # Negative marker: -(2*nnode + node_id * 2)
                # Extract node_id
                node_id = -(dof_marker + 2 * nnode) // 2
                component = 0  # ux for pullout

                if node_id >= 0 and node_id < nnode:
                    steel_dof = dofs_obj.steel[node_id, component]
                    if steel_dof >= 0:
                        resolved_load_dofs.append(steel_dof)
                    else:
                        print(f"WARNING: Steel DOF for node {node_id} not allocated")
            else:
                # Positive: already a concrete DOF
                resolved_load_dofs.append(dof_marker)

        load_dofs = resolved_load_dofs

        # Resolve reaction_dofs (if needed)
        if bc_spec.reaction_dofs:
            resolved_reaction_dofs = []
            for dof_marker in bc_spec.reaction_dofs:
                if dof_marker < 0:
                    node_id = -(dof_marker + 2 * nnode) // 2
                    component = 0

                    if node_id >= 0 and node_id < nnode:
                        steel_dof = dofs_obj.steel[node_id, component]
                        if steel_dof >= 0:
                            resolved_reaction_dofs.append(steel_dof)
                else:
                    resolved_reaction_dofs.append(dof_marker)

            bc_spec.reaction_dofs = resolved_reaction_dofs

    def _tip_patch() -> Tuple[float, float, float, float]:
        r = float(model.tip_enr_radius)
        if r <= 0.0:
            return (1e30, -1e30, 1e30, -1e30)
        return (crack.tip_x - r, crack.tip_x + r, max(0.0, crack.tip_y - r), crack.tip_y + r)

    def solve_step(
        u_target: float,
        u_guess: np.ndarray,
        crack_in: XFEMCrack,
        dofs_in: XFEMDofs,
        coh_committed,
        mp_committed,
        bond_committed=None,
    ):
        nonlocal total_newton_solves
        if model.debug_newton:
            print(f"        [solve_step] ENTRY: u_target={u_target*1e3:.3f}mm, ndof={dofs_in.ndof}")
        q = u_guess.copy()

        for it in range(int(model.newton_maxit)):
            total_newton_solves += 1  # Track each Newton iteration
            if model.debug_newton and it == 0:
                print(f"        [solve_step] Starting Newton loop...")
            fixed = dict(fixed_base)
            for a in range(nnode):
                for comp in (0, 1):
                    d_std = int(dofs_in.std[a, comp])
                    if d_std in fixed:
                        dH = int(dofs_in.H[a, comp])
                        if dH >= 0:
                            fixed[dH] = 0.0
                        for k in range(4):
                            dt = int(dofs_in.tip[a, k, comp])
                            if dt >= 0:
                                fixed[dt] = 0.0
                        # Bond-slip: DO NOT fix steel DOFs - they should be free to slip!
                        # Steel is coupled to concrete through bond-slip interface only

            # Apply prescribed displacement with scale factor from bc_spec
            prescribed_scale = bc_spec.prescribed_scale if bc_spec is not None else -1.0
            for dof in load_dofs:
                fixed[int(dof)] = prescribed_scale * float(u_target)

            for dof, val in fixed.items():
                q[int(dof)] = float(val)

            if model.debug_newton:
                print(f"        [newton] it={it:02d} calling assemble_xfem_system...")
            K, fint, coh_updates, mp_updates, aux, bond_updates, reinforcement_updates, contact_updates = assemble_xfem_system(
                nodes,
                elems,
                dofs_in,
                crack_in,
                C,
                model.b,
                q,
                law,
                coh_committed,
                tip_enr_radius=model.tip_enr_radius,
                k_stab=model.k_stab,
                visc_damp=model.visc_damp,
                material=material,
                mp_states_comm=mp_committed,
                use_numba=use_numba,
                coh_params=coh_params,
                bulk_kind=int(bulk_kind),
                bulk_params=bulk_params,
                tip_enrichment_type=model.tip_enrichment_type,
                rebar_segs=rebar_segs,
                bond_law=bond_law,
                bond_states_comm=bond_committed,  # FIX 2: Use bond_committed, not global bond_states
                enable_bond_slip=model.enable_bond_slip,
                steel_EA=model.steel_EA_min if model.enable_bond_slip else 0.0,  # Min stiffness to avoid rigid mode
                rebar_diameter=model.rebar_diameter if model.enable_bond_slip else None,
                bond_disabled_x_range=bond_disabled_x_range,  # Empty element bond masking
                subdomain_mgr=subdomain_mgr,  # FASE C: Pass subdomain manager
            )
            if model.debug_newton:
                print(f"        [newton] it={it:02d} assemble done, K.shape={K.shape}")

            # Perfect bond rebar contribution (legacy: only if bond-slip disabled)
            # When bond-slip is enabled, rebar forces are handled in assemble_bond_slip()
            if not model.enable_bond_slip:
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
                fint = fint + f_rb
                K = K + K_rb

            r = fint.copy()
            free, K_ff, r_f, _ = apply_dirichlet(K, r, fixed, q)
            rhs = -r_f
            norm_r = float(np.linalg.norm(rhs))

            P_est = -sum(r[d] for d in load_dofs)
            fscale = max(1.0, abs(float(P_est)))
            tol = float(model.newton_tol_r) + float(model.newton_beta) * fscale
            if norm_r < tol:
                P = -sum(r[d] for d in load_dofs)
                if model.debug_newton:
                    print(
                        f"    [newton] converged(res) it={it+1:02d} ||rhs||={norm_r:.3e} u={u_target*1e3:.3f}mm"
                    )
                # Apply trial history updates as a *patch* (Phase-1) or as
                # direct dict updates (legacy).
                if isinstance(coh_committed, CohesiveStateArrays):
                    assert isinstance(coh_updates, CohesiveStatePatch)
                    coh_trial = coh_committed.copy()
                    coh_updates.apply_to(coh_trial)
                else:
                    coh_trial = dict(coh_committed)
                    coh_trial.update(coh_updates)

                if isinstance(mp_committed, BulkStateArrays):
                    assert isinstance(mp_updates, BulkStatePatch)
                    mp_trial = mp_committed.copy()
                    mp_updates.apply_to(mp_trial)
                else:
                    mp_trial = dict(mp_committed)
                    mp_trial.update(mp_updates)

                # Bond-slip state update (Phase 2)
                bond_trial = None
                if bond_updates is not None:
                    bond_trial = bond_updates  # Already a fresh copy from assemble_bond_slip

                return True, q, coh_trial, mp_trial, aux, float(P), bond_trial

            # Diagonal equilibration for ill-conditioned systems (Priority #0 fix)
            if model.enable_diagonal_scaling:
                from xfem_clean.utils.scaling import diagonal_equilibration, unscale_solution
                K_ff_scaled, rhs_scaled, D_inv = diagonal_equilibration(K_ff, rhs, eps=1e-12)
                try:
                    du_f_scaled = spla.spsolve(K_ff_scaled, rhs_scaled)
                except Exception:
                    du_f_scaled = spla.lsmr(K_ff_scaled, rhs_scaled, atol=1e-12, btol=1e-12, maxiter=2000)[0]
                if not np.all(np.isfinite(du_f_scaled)):
                    du_f_scaled = spla.lsmr(K_ff_scaled, rhs_scaled, atol=1e-12, btol=1e-12, maxiter=2000)[0]
                du_f = unscale_solution(du_f_scaled, D_inv)
            else:
                try:
                    du_f = spla.spsolve(K_ff, rhs)
                except Exception:
                    du_f = spla.lsmr(K_ff, rhs, atol=1e-12, btol=1e-12, maxiter=2000)[0]
                if not np.all(np.isfinite(du_f)):
                    du_f = spla.lsmr(K_ff, rhs, atol=1e-12, btol=1e-12, maxiter=2000)[0]
            norm_du = float(np.linalg.norm(du_f))
            if norm_du < float(model.newton_tol_du):
                if model.debug_newton:
                    print(
                        f"    [newton] stagnated      it={it+1:02d} ||du||={norm_du:.3e} ||rhs||={norm_r:.3e} u={u_target*1e3:.3f}mm"
                    )
                return False, q, coh_committed, mp_committed, aux, 0.0, bond_states

            if model.line_search:
                q0 = q.copy()
                best = None
                best_q = None
                alpha_ls = 1.0
                accepted = False
                for _ls in range(8):
                    q_try = q0.copy()
                    q_try[free] += alpha_ls * du_f
                    for dof, val in fixed.items():
                        q_try[int(dof)] = float(val)

                    _, fint_t, _coh_upd_t, _mp_upd_t, _aux_t, _bond_upd_t, _reinf_upd_t, _contact_upd_t = assemble_xfem_system(
                        nodes,
                        elems,
                        dofs_in,
                        crack_in,
                        C,
                        model.b,
                        q_try,
                        law,
                        coh_committed,
                        tip_enr_radius=model.tip_enr_radius,
                        k_stab=model.k_stab,
                        visc_damp=model.visc_damp,
                        material=material,
                        mp_states_comm=mp_committed,
                        use_numba=use_numba,
                        coh_params=coh_params,
                        bulk_kind=int(bulk_kind),
                        bulk_params=bulk_params,
                        rebar_segs=rebar_segs,
                        bond_law=bond_law,
                        bond_states_comm=bond_committed,  # FIX 2: Use bond_committed, not global bond_states
                        enable_bond_slip=model.enable_bond_slip,
                        steel_EA=model.steel_EA_min if model.enable_bond_slip else 0.0,  # Min stiffness to avoid rigid mode
                        rebar_diameter=model.rebar_diameter if model.enable_bond_slip else None,
                        bond_disabled_x_range=bond_disabled_x_range,  # Empty element bond masking
                        subdomain_mgr=subdomain_mgr,  # FASE C: Pass subdomain manager
                    )
                    # Perfect bond rebar (only if bond-slip disabled)
                    if not model.enable_bond_slip:
                        f_rb_t, _Krb_t = rebar_contrib(
                            nodes,
                            rebar_segs,
                            q_try,
                            model.steel_A_total,
                            model.steel_E,
                            model.steel_fy,
                            model.steel_fu,
                            model.steel_Eh,
                        )
                        fint_t = fint_t + f_rb_t
                    r_try = fint_t
                    _, _, r_f_try, _ = apply_dirichlet(K, r_try, fixed, q_try)
                    n_try = float(np.linalg.norm(r_f_try))
                    if best is None or n_try < best:
                        best = n_try
                        best_q = q_try
                    if n_try < norm_r:
                        q = q_try
                        accepted = True
                        break
                    alpha_ls *= 0.5
                if not accepted:
                    if best_q is not None:
                        q = best_q
                    else:
                        q[free] += du_f
            else:
                q[free] += du_f

            for dof, val in fixed.items():
                q[int(dof)] = float(val)

        if model.debug_substeps:
            print(f"    [newton] failed(maxit) u={u_target*1e3:.3f}mm")
        return False, q, coh_committed, mp_committed, aux, 0.0, bond_committed

    def curvature_mid(q_full: np.ndarray) -> float:
        y0 = float(np.min(nodes[:, 1]))
        bottom = np.where(np.isclose(nodes[:, 1], y0))[0]
        xs = nodes[bottom, 0]
        ws = q_full[2 * bottom + 1]
        m = np.abs(xs - model.L / 2) <= model.L / 6
        xs2 = xs[m]
        ws2 = ws[m]
        if len(xs2) < 5:
            return 0.0
        A = np.vstack([xs2**2, xs2, np.ones_like(xs2)]).T
        coef, *_ = np.linalg.lstsq(A, ws2, rcond=None)
        return float(2.0 * coef[0])

    u_n = 0.0
    q_n = u_std
    step_counter = 0

    # Anti-hang instrumentation: track total substeps and Newton solves
    total_substeps = 0
    total_newton_solves = 0
    substep_report_interval = 200  # Print diagnostic every N substeps

    # BLOQUE 3: Support custom u_targets for cyclic loading
    if u_targets is not None:
        # Use provided trajectory (cyclic or custom)
        u_sequence = u_targets
        nsteps_actual = len(u_sequence)
    else:
        # Default: monotonic linspace
        u_sequence = np.linspace(0.0, umax, nsteps + 1)[1:]  # Exclude initial 0
        nsteps_actual = nsteps

    for istep in range(1, nsteps_actual + 1):
        u_target = float(u_sequence[istep - 1])
        du_total = u_target - u_n

        stack = [(u_n, du_total, 0)]
        while stack:
            u0, du, level = stack.pop()
            u1 = u0 + du

            # Anti-hang guardrail: abort if we exceed maximum total substeps
            total_substeps += 1
            if total_substeps > model.max_total_substeps:
                raise RuntimeError(
                    f"Anti-hang guardrail triggered: total_substeps={total_substeps} > max_total_substeps={model.max_total_substeps}.\n"
                    f"Last state: step={istep}/{nsteps_actual}, u0={u0:.6e}, u1={u1:.6e}, level={level}, "
                    f"crack={'on' if crack.active else 'off'}, tip=({crack.tip_x:.4f},{crack.tip_y:.4f})m, "
                    f"total_newton_solves={total_newton_solves}"
                )

            # Periodic diagnostic output (every N substeps)
            if total_substeps % substep_report_interval == 0:
                print(
                    f"[DIAGNOSTIC] total_substeps={total_substeps}, total_newton={total_newton_solves}, "
                    f"step={istep}/{nsteps_actual}, u1={u1*1e3:.3f}mm, level={level}"
                )

            if model.debug_substeps:
                print(
                    f"[substep] lvl={level:02d} u0={u0*1e3:.3f}mm -> u1={u1*1e3:.3f}mm  du={du*1e3:.3f}mm  "
                    f"crack={'on' if crack.active else 'off'} tip=({crack.tip_x:.3f},{crack.tip_y:.3f})m"
                )

            q_backup = q_n.copy()
            crack_backup = XFEMCrack(
                x0=crack.x0,
                y0=crack.y0,
                tip_x=crack.tip_x,
                tip_y=crack.tip_y,
                stop_y=crack.stop_y,
                angle_deg=getattr(crack, "angle_deg", 90.0),
                active=crack.active,
            )
            coh_backup = coh_states.copy() if hasattr(coh_states, "copy") else copy.deepcopy(coh_states)
            mp_backup = mp_states.copy() if hasattr(mp_states, "copy") else {k: v.copy_shallow() for k, v in mp_states.items()}
            dofs_backup = dofs
            # FIX 1: Backup bond_states to enable rollback in case of substep failure
            bond_backup = bond_states.copy() if (bond_states is not None and hasattr(bond_states, "copy")) else bond_states

            inner_updates = 0
            q_guess = None
            need_split = False
            split_reason = ""

            while True:
                if model.debug_substeps:
                    print(f"    [inner] inner_updates={inner_updates}, crack_active={crack.active}, need_split={need_split}")
                force_accept = False

                if not crack.active:
                    dofs_local = _std_only_dofs()
                    # Resolve steel DOFs from bc_spec markers (first time only)
                    if bc_spec is not None and inner_updates == 0:
                        _resolve_steel_dofs_in_bc_spec(dofs_local)
                else:
                    if dofs is None:
                        yH = float(min(crack.tip_y, crack.stop_y))
                        dofs = build_xfem_dofs(
                            nodes, elems, crack, H_region_ymax=yH, tip_patch=_tip_patch(),
                            rebar_segs=rebar_segs, enable_bond_slip=model.enable_bond_slip
                        )
                        # Resolve steel DOFs from bc_spec markers (first time crack activates)
                        if bc_spec is not None:
                            _resolve_steel_dofs_in_bc_spec(dofs)
                    dofs_local = dofs

                if q_guess is None:
                    base = q_backup
                    base_dofs = dofs_backup if crack_backup.active else _std_only_dofs()
                else:
                    base = q_guess
                    base_dofs = dofs_local

                if len(base) == dofs_local.ndof:
                    q_guess_loc = base.copy()
                else:
                    q_guess_loc = transfer_q_between_dofs(base, base_dofs, dofs_local)

                if model.debug_substeps:
                    print(f"    [inner] calling solve_step with u1={u1*1e3:.3f}mm")
                ok, q_sol, coh_trial, mp_trial, aux, P, bond_trial = solve_step(
                    u1,
                    q_guess_loc,
                    crack,
                    dofs_local,
                    coh_states,
                    mp_states,
                    bond_states,
                )
                if model.debug_substeps:
                    print(f"    [inner] solve_step returned ok={ok}")
                if not ok:
                    need_split = True
                    split_reason = "newton"
                    break

                changed = False
                if aux is not None:
                    gp_pos = aux["gp_pos"]
                    gp_sig = aux["gp_sig"]

                    did_initiate = False

                    if not crack.active:
                        y_max_init = min(stop_y, model.cand_ymax_factor * dy)
                        best_s = -1.0
                        best_xy = None
                        best_v = None

                        for (xc, yc) in cps:
                            sigbar = nonlocal_bar_stress(
                                gp_pos,
                                gp_sig,
                                float(xc),
                                float(yc),
                                rho=model.crack_rho,
                                y_max=y_max_init,
                                r_cut=3.0 * model.crack_rho,
                            )
                            s1, v1 = principal_max_dir(sigbar)
                            if s1 > best_s:
                                best_s = s1
                                best_xy = (float(xc), float(yc))
                                best_v = v1

                        if best_xy is None:
                            best_xy = (model.L / 2.0, 0.0)
                            best_v = np.array([1.0, 0.0], dtype=float)

                        if best_s >= model.ft * model.ft_initiation_factor:
                            crack.active = True
                            crack.x0 = float(best_xy[0])
                            crack.y0 = float(best_xy[1])

                            if np.any(np.isclose(nodes[:, 0], crack.x0, atol=1e-12)):
                                crack.x0 += 1e-6 * dx

                            v1 = np.array(best_v, dtype=float)
                            tv = np.array([-v1[1], v1[0]], dtype=float)
                            tv = tv / (float(np.linalg.norm(tv)) + 1e-15)
                            if tv[1] < 0.0:
                                tv *= -1.0

                            seglen = float(model.crack_seg_length) if model.crack_seg_length is not None else float(dy)
                            tip = np.array([crack.x0, crack.y0], dtype=float) + seglen * tv
                            tip[1] = min(float(tip[1]), float(crack.stop_y))
                            tip[0] = max(0.0, min(float(tip[0]), float(model.L)))

                            crack.tip_x = float(tip[0])
                            crack.tip_y = float(tip[1])

                            yH = float(min(crack.tip_y, crack.stop_y))
                            dofs_new = build_xfem_dofs(
                                nodes, elems, crack, H_region_ymax=yH, tip_patch=_tip_patch(),
                                rebar_segs=rebar_segs, enable_bond_slip=model.enable_bond_slip
                            )
                            q_sol = transfer_q_between_dofs(q_sol, dofs_local, dofs_new)
                            dofs = dofs_new
                            did_initiate = True
                            force_accept = True

                            ang = math.degrees(math.atan2(tv[1], tv[0]))
                            print(
                                f"[crack] initiate at ({crack.x0:.4f},{crack.y0:.4f}) m  "
                                f"tip=({crack.tip_x:.4f},{crack.tip_y:.4f}) m  angle={ang:.1f}°  "
                                f"sigma1_bar={best_s/1e6:.3f} MPa  ft={model.ft/1e6:.3f} MPa"
                            )

                    if crack.active and (not did_initiate) and crack.tip_y < crack.stop_y - 1e-12:
                        tipx = float(crack.tip_x)
                        tipy = float(crack.tip_y)
                        y_max_tip = min(crack.stop_y, tipy + 2.0 * dy)

                        sigbar = nonlocal_bar_stress(
                            gp_pos,
                            gp_sig,
                            tipx,
                            tipy,
                            rho=model.crack_rho,
                            y_max=y_max_tip,
                            r_cut=3.0 * model.crack_rho,
                        )
                        s1, _v1 = principal_max_dir(sigbar)
                        if s1 >= model.ft:
                            seglen = float(model.crack_seg_length) if model.crack_seg_length is not None else float(dy)
                            tv_new = crack.tvec()
                            tv_new = tv_new / (float(np.linalg.norm(tv_new)) + 1e-15)
                            if tv_new[1] < 0.0:
                                tv_new *= -1.0

                            tip = np.array([tipx, tipy], dtype=float) + seglen * tv_new
                            tip[1] = min(float(tip[1]), float(crack.stop_y))
                            tip[0] = max(0.0, min(float(tip[0]), float(model.L)))

                            if float(np.hypot(tip[0] - tipx, tip[1] - tipy)) > 1e-12:
                                crack.tip_x = float(tip[0])
                                crack.tip_y = float(tip[1])

                                yH = float(min(crack.tip_y, crack.stop_y))
                                dofs_new = build_xfem_dofs(
                                    nodes, elems, crack, H_region_ymax=yH, tip_patch=_tip_patch(),
                                    rebar_segs=rebar_segs, enable_bond_slip=model.enable_bond_slip
                                )

                                # Junction detection (Dissertation Eq. 4.64-4.66)
                                # NOTE: Full junction enrichment requires multi-crack support.
                                # For future multi-crack solver, detect coalescence here:
                                # if model.enable_junction_enrichment:
                                #     from xfem_clean.junction import detect_crack_coalescence
                                #     junctions = detect_crack_coalescence(
                                #         cracks=[crack], nodes=nodes, elems=elems,
                                #         tol_merge=model.junction_merge_tolerance
                                #     )

                                # L2 DOF projection (Dissertation Eq. 4.60-4.63)
                                if model.enable_dof_projection and dofs_new.ndof != dofs_local.ndof:
                                    from xfem_clean.dof_mapping import project_dofs_l2
                                    q_sol = project_dofs_l2(
                                        q_old=q_sol,
                                        nodes_old=nodes,
                                        elems=elems,
                                        dofs_old=dofs_local,
                                        dofs_new=dofs_new,
                                        patch_elements=None,  # Auto-detect affected elements
                                        use_standard_part=True,
                                    )
                                else:
                                    # Standard transfer (current implementation)
                                    q_sol = transfer_q_between_dofs(q_sol, dofs_local, dofs_new)

                                dofs = dofs_new
                                changed = True

                                ang = math.degrees(math.atan2(tv_new[1], tv_new[0]))
                                print(
                                    f"[crack] grow tip=({crack.tip_x:.4f},{crack.tip_y:.4f}) m  "
                                    f"angle={ang:.1f}°  sigma1_bar={s1/1e6:.3f} MPa"
                                )

                if force_accept:
                    coh_states = coh_trial
                    mp_states = mp_trial
                    if bond_trial is not None:
                        bond_states = bond_trial
                    break

                if not changed:
                    coh_states = coh_trial
                    mp_states = mp_trial
                    if bond_trial is not None:
                        bond_states = bond_trial
                    break

                inner_updates += 1
                if inner_updates >= int(model.crack_max_inner):
                    need_split = True
                    split_reason = "crack_updates"
                    break

                coh_states = coh_trial
                mp_states = mp_trial
                if bond_trial is not None:
                    bond_states = bond_trial
                q_guess = q_sol.copy()
                continue

            if need_split:
                q_n = q_backup
                crack = crack_backup
                coh_states = coh_backup
                mp_states = mp_backup
                dofs = dofs_backup
                # FIX 1: Restore bond_states to previous state when substep fails
                bond_states = bond_backup
                if level >= int(model.max_subdiv):
                    raise RuntimeError(
                        f"Substepping exceeded max_subdiv={model.max_subdiv} (reason={split_reason}) at u={u1} m"
                    )
                stack.append((u0 + 0.5 * du, 0.5 * du, level + 1))
                stack.append((u0, 0.5 * du, level + 1))
                continue

            q_n = q_sol
            u_n = u1
            step_counter += 1

            kappa = curvature_mid(q_n)
            R = float("inf") if abs(kappa) < 1e-16 else 1.0 / kappa
            M = P * model.L / 4.0
            ang = math.degrees(math.atan2(crack.tvec()[1], crack.tvec()[0])) if crack.active else 0.0
            ed = _compute_global_dissipation(aux, mp_states, coh_states)
            results.append(
                [
                    step_counter,
                    u_n,
                    P,
                    M,
                    kappa,
                    R,
                    crack.tip_x,
                    crack.tip_y,
                    ang,
                    float(crack.active),
                    ed["W_plastic"],
                    ed["W_damage_t"],
                    ed["W_damage_c"],
                    ed["W_cohesive"],
                    ed["W_diss_total"],
                ]
            )

            print(
                f"step={step_counter:04d} u={u_n*1e3:7.3f} mm  P={P/1e3:9.3f} kN  "
                f"M={M/1e3:9.3f} kN·m  kappa={kappa: .3e}  tip=({crack.tip_x: .3f},{crack.tip_y: .3f})"
            )
            print(
                f"       diss[J]  plastic={ed['W_plastic']:.3e}  fracture={ed['W_fracture']:.3e} "
                f"(bulk_t={ed['W_damage_t']:.3e} + coh={ed['W_cohesive']:.3e})  crush={ed['W_damage_c']:.3e}  "
                f"total={ed['W_diss_total']:.3e}"
            )

    # Return format selection
    if return_bundle:
        # Comprehensive bundle for postprocessing (FASE G)
        return {
            'nodes': nodes,
            'elems': elems,
            'u': q_n,
            'history': np.asarray(results, dtype=float),
            'crack': crack,
            'mp_states': mp_states,
            'bond_states': bond_states,
            'rebar_segs': rebar_segs,
            'dofs': dofs,
            'coh_states': coh_states,
        }
    elif return_states:
        return nodes, elems, q_n, np.asarray(results, dtype=float), crack, mp_states
    else:
        return nodes, elems, q_n, np.asarray(results, dtype=float), crack


