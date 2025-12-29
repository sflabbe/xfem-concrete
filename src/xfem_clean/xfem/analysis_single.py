"""Displacement-controlled single-crack XFEM analysis driver."""

from __future__ import annotations

import copy
import math
from typing import Dict, Tuple, Optional, Union

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


def run_analysis_xfem(
    model: XFEMModel,
    nx: int,
    ny: int,
    nsteps: int,
    umax: float,
    law: Optional[CohesiveLaw] = None,
    return_states: bool = False,
):
    """Run the single-crack XFEM prototype (stable linear version + cohesive)."""

    nodes, elems = structured_quad_mesh(model.L, model.H, nx, ny)
    nnode = nodes.shape[0]

    left = int(np.argmin(nodes[:, 0]))
    right = int(np.argmax(nodes[:, 0]))
    fixed_base = {2 * left: 0.0, 2 * left + 1: 0.0, 2 * right + 1: 0.0}

    y_top = model.H
    top_nodes = np.where(np.isclose(nodes[:, 1], y_top))[0]
    dx = model.L / nx
    dy = model.H / ny
    load_halfwidth = model.load_halfwidth if (model.load_halfwidth and model.load_halfwidth > 0.0) else 2.0 * dx
    load_nodes = top_nodes[np.where(np.abs(nodes[top_nodes, 0] - model.L / 2) <= load_halfwidth)[0]]
    if len(load_nodes) == 0:
        load_nodes = np.array([int(top_nodes[np.argmin(np.abs(nodes[top_nodes, 0] - model.L / 2))])], dtype=int)
    load_dofs = [2 * n + 1 for n in load_nodes]

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
        std = np.arange(2 * nnode, dtype=int).reshape(nnode, 2)
        H = -np.ones((nnode, 2), dtype=int)
        tipd = -np.ones((nnode, 4, 2), dtype=int)
        return XFEMDofs(
            std=std,
            H=H,
            tip=tipd,
            ndof=2 * nnode,
            H_nodes=np.zeros(nnode, dtype=bool),
            tip_nodes=np.zeros(nnode, dtype=bool),
        )

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
    ):
        q = u_guess.copy()

        for it in range(int(model.newton_maxit)):
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

            for dof in load_dofs:
                fixed[int(dof)] = -float(u_target)

            for dof, val in fixed.items():
                q[int(dof)] = float(val)

            K, fint, coh_updates, mp_updates, aux = assemble_xfem_system(
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
            )

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

                return True, q, coh_trial, mp_trial, aux, float(P)

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
                return False, q, coh_committed, mp_committed, aux, 0.0

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

                    _, fint_t, _coh_upd_t, _mp_upd_t, _aux_t = assemble_xfem_system(
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
                    )
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
        return False, q, coh_committed, mp_committed, aux, 0.0

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

    for istep in range(1, nsteps + 1):
        u_target = (istep / nsteps) * umax
        du_total = u_target - u_n

        stack = [(u_n, du_total, 0)]
        while stack:
            u0, du, level = stack.pop()
            u1 = u0 + du

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

            inner_updates = 0
            q_guess = None
            need_split = False
            split_reason = ""

            while True:
                force_accept = False

                if not crack.active:
                    dofs_local = _std_only_dofs()
                else:
                    if dofs is None:
                        yH = float(min(crack.tip_y, crack.stop_y))
                        dofs = build_xfem_dofs(nodes, elems, crack, H_region_ymax=yH, tip_patch=_tip_patch())
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

                ok, q_sol, coh_trial, mp_trial, aux, P = solve_step(
                    u1,
                    q_guess_loc,
                    crack,
                    dofs_local,
                    coh_states,
                    mp_states,
                )
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
                            dofs_new = build_xfem_dofs(nodes, elems, crack, H_region_ymax=yH, tip_patch=_tip_patch())
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
                                dofs_new = build_xfem_dofs(nodes, elems, crack, H_region_ymax=yH, tip_patch=_tip_patch())
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
                    break

                if not changed:
                    coh_states = coh_trial
                    mp_states = mp_trial
                    break

                inner_updates += 1
                if inner_updates >= int(model.crack_max_inner):
                    need_split = True
                    split_reason = "crack_updates"
                    break

                coh_states = coh_trial
                mp_states = mp_trial
                q_guess = q_sol.copy()
                continue

            if need_split:
                q_n = q_backup
                crack = crack_backup
                coh_states = coh_backup
                mp_states = mp_backup
                dofs = dofs_backup
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

    if return_states:
        return nodes, elems, q_n, np.asarray(results, dtype=float), crack, mp_states
    return nodes, elems, q_n, np.asarray(results, dtype=float), crack


