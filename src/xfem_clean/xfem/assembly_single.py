"""Assembly routines for the single-crack XFEM prototype."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp

from xfem_clean.cohesive_laws import CohesiveLaw, CohesiveState, cohesive_update
from xfem_clean.numba.kernels_cohesive import cohesive_update_values_numba
from xfem_clean.numba.kernels_bulk import (
    elastic_integrate_plane_stress_numba,
    dp_integrate_plane_stress_numba,
)
from xfem_clean.constitutive import ConstitutiveModel, LinearElasticPlaneStress
from xfem_clean.material_point import MaterialPoint, mp_default
from xfem_clean.xfem.state_arrays import (
    BulkStateArrays,
    BulkStatePatch,
    CohesiveStateArrays,
    CohesiveStatePatch,
)
from xfem_clean.xfem.geometry import XFEMCrack, clip_segment_to_bbox, branch_F_and_grad
from xfem_clean.xfem.dofs_single import XFEMDofs
from xfem_clean.xfem.q4_utils import element_bounds, element_x_y, element_dN_dxdy, element_local_xi_at_x, element_local_eta_at_y
from xfem_clean.xfem.enrichment_single import build_B_enriched
from xfem_clean.fem.q4 import q4_shape
from xfem_clean.crack_criteria import principal_max_2d


def assemble_xfem_system(
    nodes: np.ndarray,
    elems: np.ndarray,
    dofs: XFEMDofs,
    crack: XFEMCrack,
    C: np.ndarray,
    thickness: float,
    q: np.ndarray,
    law: CohesiveLaw,
    coh_states_comm: Union[Dict[Tuple[int, int], CohesiveState], CohesiveStateArrays],
    tip_enr_radius: float,
    k_stab: float = 1e-9,
    visc_damp: float = 0.0,
    material: Optional[ConstitutiveModel] = None,
    mp_states_comm: Optional[Union[Dict[Tuple[int, int], MaterialPoint], BulkStateArrays]] = None,
    use_numba: bool = False,
    coh_params: Optional[np.ndarray] = None,
    bulk_kind: int = 0,
    bulk_params: Optional[np.ndarray] = None,
) -> Tuple[
    sp.csr_matrix,
    np.ndarray,
    Union[Dict[Tuple[int, int], CohesiveState], CohesiveStatePatch],
    Union[Dict[Tuple[int, int], MaterialPoint], BulkStatePatch],
    Dict[str, np.ndarray],
]:
    """Assemble global tangent and internal force vector.

    Returns
    -------
    K : csr_matrix
    fint : ndarray
    coh_updates : dict
        Trial cohesive states for this iterate (do not mutate committed history).
    mp_updates : dict
        Trial material-point states for this iterate (do not mutate committed history).
    aux : dict
        Gauss-point data for postprocessing / nonlocal crack criteria.
    """

    ndof = int(dofs.ndof)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    fint = np.zeros(ndof, dtype=float)

    use_coh_arrays = isinstance(coh_states_comm, CohesiveStateArrays)
    use_mp_arrays = isinstance(mp_states_comm, BulkStateArrays)

    use_bulk_numba = bool(use_numba) and use_mp_arrays and (bulk_params is not None) and (int(bulk_kind) in (1, 2, 3))

    coh_updates: Union[Dict[Tuple[int, int], CohesiveState], CohesiveStatePatch]
    mp_updates: Union[Dict[Tuple[int, int], MaterialPoint], BulkStatePatch]
    coh_updates = CohesiveStatePatch.empty() if use_coh_arrays else {}
    mp_updates = BulkStatePatch.empty() if use_mp_arrays else {}

    gp_pos = []
    gp_sig = []
    gp_s1 = []
    gp_eid = []
    gp_ipid = []
    gp_wgt = []

    coh_wgt = []
    coh_delta_max = []

    crack_active = bool(crack.active)
    p0 = crack.p0() if crack_active else None
    pt = crack.pt() if crack_active else None
    tip_x = float(crack.tip_x) if crack_active else float(crack.x0)
    tip_y = float(crack.tip_y) if crack_active else float(crack.y0)

    g = [-1.0 / math.sqrt(3.0), +1.0 / math.sqrt(3.0)]
    w = [1.0, 1.0]

    # Default retrocompatible material: linear elastic plane stress.
    if material is None:
        material = LinearElasticPlaneStress(E=float(1.0), nu=0.0)  # placeholder, overridden below
        material.C = np.asarray(C, dtype=float)
    if mp_states_comm is None:
        mp_states_comm = {}

    for e, conn in enumerate(elems):
        xe = nodes[conn, :]
        xmin, xmax, ymin, ymax = element_bounds(xe)

        is_cut = False
        if crack_active:
            seg = clip_segment_to_bbox(p0, pt, xmin, xmax, ymin, ymax)
            if seg is not None:
                phis = [crack.phi(float(xe[a, 0]), float(xe[a, 1])) for a in range(4)]
                if (min(phis) < 0.0) and (max(phis) > 0.0):
                    is_cut = True

        if not is_cut:
            subdomains = [(-1.0, 1.0, -1.0, 1.0)]
        else:
            nsub = 4
            xis = np.linspace(-1.0, 1.0, nsub + 1)
            etas = np.linspace(-1.0, 1.0, nsub + 1)
            subdomains = []
            for i in range(nsub):
                for j in range(nsub):
                    subdomains.append((float(xis[i]), float(xis[i + 1]), float(etas[j]), float(etas[j + 1])))

        # bulk integration
        for sidx, (xi_a, xi_b, eta_a, eta_b) in enumerate(subdomains):
            for ixi in range(2):
                xi_hat = g[ixi]
                wx = w[ixi] * 0.5 * (xi_b - xi_a)
                xi = 0.5 * (xi_b - xi_a) * xi_hat + 0.5 * (xi_b + xi_a)
                for ieta in range(2):
                    eta_hat = g[ieta]
                    wy = w[ieta] * 0.5 * (eta_b - eta_a)
                    eta = 0.5 * (eta_b - eta_a) * eta_hat + 0.5 * (eta_b + eta_a)

                    x, y, N, dN_dxi, dN_deta = element_x_y(xi, eta, xe)
                    dN_dx, dN_dy, detJ = element_dN_dxdy(dN_dxi, dN_deta, xe)
                    weight = wx * wy * detJ * float(thickness)

                    Hgp = crack.H(x, y) if crack_active else -1.0
                    B, edofs = build_B_enriched(
                        x,
                        y,
                        N,
                        dN_dx,
                        dN_dy,
                        xe,
                        conn,
                        dofs,
                        crack,
                        tip_enr_radius,
                        crack_active,
                        Hgp,
                        tip_x,
                        tip_y,
                    )
                    qe = q[edofs]
                    eps = B @ qe

                    # Integration-point id:
                    # - uncut elements: 4 canonical 2×2 Gauss points -> ids 0..3
                    # - cut elements: keep canonical ids 0..3 reserved for post/crack criteria,
                    #   and store the (subdomain, gauss) states at ids 4..(4+4*nsub-1)
                    ip_local = int(ixi + 2 * ieta)
                    ipid = ip_local if (not is_cut) else int(4 + 4 * sidx + ip_local)

                    if use_mp_arrays:
                        assert isinstance(mp_states_comm, BulkStateArrays)
                        if use_bulk_numba:
                            # Numba bulk kernel path (stateless, array-based)
                            dt0 = float(mp_states_comm.damage_t[e, ipid])
                            dc0 = float(mp_states_comm.damage_c[e, ipid])
                            wpl0 = float(mp_states_comm.w_plastic[e, ipid])
                            wft0 = float(mp_states_comm.w_fract_t[e, ipid])
                            wfc0 = float(mp_states_comm.w_fract_c[e, ipid])
                            eps_p6_old = np.asarray(mp_states_comm.eps_p6[e, ipid, :], dtype=float)
                            ezz_old = float(mp_states_comm.eps_zz[e, ipid])
                            kappa_old = float(mp_states_comm.kappa[e, ipid])
                            kt0 = float(mp_states_comm.kappa_t[e, ipid])
                            kc0 = float(mp_states_comm.kappa_c[e, ipid])
                            cwt0 = float(mp_states_comm.cdp_w_t[e, ipid])
                            ceps0 = float(mp_states_comm.cdp_eps_in_c[e, ipid])

                            if int(bulk_kind) == 1:
                                E = float(bulk_params[0])
                                nu = float(bulk_params[1])
                                sig, Ct = elastic_integrate_plane_stress_numba(eps, E, nu, dt0, dc0)
                                eps_p6_new = eps_p6_old
                                ezz_new = ezz_old
                                kappa_new = kappa_old
                                dt_new = dt0
                                dc_new = dc0
                                kt_new = kt0
                                kc_new = kc0
                                wpl_new = wpl0
                                wft_new = wft0
                                wfc_new = wfc0
                                dW = 0.0
                            elif int(bulk_kind) == 2:
                                E = float(bulk_params[0])
                                nu = float(bulk_params[1])
                                alpha = float(bulk_params[2])
                                k0 = float(bulk_params[3])
                                Hh = float(bulk_params[4])
                                sig, Ct, eps_p6_new, ezz_new, kappa_new, dW = dp_integrate_plane_stress_numba(
                                    eps,
                                    eps_p6_old,
                                    ezz_old,
                                    kappa_old,
                                    E,
                                    nu,
                                    alpha,
                                    k0,
                                    Hh,
                                )
                                dt_new = dt0
                                dc_new = dc0
                                kt_new = kt0
                                kc_new = kc0
                                wpl_new = wpl0 + dW
                                wft_new = wft0
                                wfc_new = wfc0
                            elif int(bulk_kind) == 3:
                                # CDP-lite kernel
                                from xfem_clean.numba.kernels_bulk import cdp_integrate_plane_stress_numba
                                E = float(bulk_params[0])
                                nu = float(bulk_params[1])
                                alpha = float(bulk_params[2])
                                k0 = float(bulk_params[3])
                                Hh = float(bulk_params[4])
                                ft = float(bulk_params[5])
                                fc = float(bulk_params[6])
                                Gf_t = float(bulk_params[7])
                                Gf_c = float(bulk_params[8])
                                lch = float(bulk_params[9])
                                sig, Ct, eps_p6_new, ezz_new, kappa_new, dt_new, dc_new, kt_new, kc_new, wpl_new, wft_new, wfc_new = cdp_integrate_plane_stress_numba(
                                    eps,
                                    eps_p6_old,
                                    ezz_old,
                                    kappa_old,
                                    dt0,
                                    dc0,
                                    kt0,
                                    kc0,
                                    wpl0,
                                    wft0,
                                    wfc0,
                                    E,
                                    nu,
                                    alpha,
                                    k0,
                                    Hh,
                                    ft,
                                    fc,
                                    Gf_t,
                                    Gf_c,
                                    lch,
                                )
                            else:
                                raise ValueError(f"Unknown bulk_kind={bulk_kind}")

                            eps_p3_new = np.array([eps_p6_new[0], eps_p6_new[1], eps_p6_new[3]], dtype=float)
                            if isinstance(mp_updates, BulkStatePatch):
                                mp_updates.add_values(
                                    e,
                                    ipid,
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
                                    cdp_w_t=cwt0,
                                    cdp_eps_in_c=ceps0,
                                )
                            else:
                                mp = mp_default().copy_shallow()
                                mp.eps = np.asarray(eps, dtype=float)
                                mp.sigma = np.asarray(sig, dtype=float)
                                mp.eps_p = eps_p3_new
                                mp.damage_t = float(dt_new)
                                mp.damage_c = float(dc_new)
                                mp.kappa = float(kappa_new)
                                mp.w_plastic = float(wpl_new)
                                mp.w_fract_t = float(wft_new)
                                mp.w_fract_c = float(wfc_new)
                                mp.extra["eps_p6"] = np.asarray(eps_p6_new, dtype=float)
                                mp.extra["eps_zz"] = float(ezz_new)
                                mp.extra["kappa_t"] = float(kt_new)
                                mp.extra["kappa_c"] = float(kc_new)
                                mp.extra["cdp_w_t"] = cwt0
                                mp.extra["cdp_eps_in_c"] = ceps0
                                mp_updates[(e, ipid)] = mp
                        else:
                            mp0 = mp_states_comm.get_mp(e, ipid)
                            mp = mp0.copy_shallow()
                            sig, Ct = material.integrate(mp, eps)
                            if isinstance(mp_updates, BulkStatePatch):
                                mp_updates.add(e, ipid, mp)
                            else:
                                mp_updates[(e, ipid)] = mp
                    elif mp_states_comm is not None:
                        mp0 = mp_states_comm.get((e, ipid), mp_default())
                        mp = mp0.copy_shallow()
                        sig, Ct = material.integrate(mp, eps)
                        mp_updates[(e, ipid)] = mp
                    else:
                        mp = mp_default().copy_shallow()
                        sig, Ct = material.integrate(mp, eps)

                    fe = B.T @ sig * weight
                    Ke = (B.T @ Ct @ B) * weight

                    fint[edofs] += fe
                    rr = np.repeat(edofs, len(edofs))
                    cc = np.tile(edofs, len(edofs))
                    rows.extend(rr.tolist())
                    cols.extend(cc.tolist())
                    data.extend(Ke.reshape(-1).tolist())

                    gp_pos.append([x, y])
                    gp_sig.append(sig.tolist())
                    gp_s1.append(principal_max_2d(sig))
                    gp_eid.append(int(e))
                    gp_ipid.append(int(ipid))
                    gp_wgt.append(float(weight))

        # material-point tracking on canonical 2x2 Gauss points
        # (kept stable even when cut elements are subdivided for bulk integration).
        # These 4 points are used for post-processing and nonlocal crack criteria.
        if is_cut:
            for iy, eta_c in enumerate(g):
                for ix, xi_c in enumerate(g):
                    gpid = int(ix + 2 * iy)
                    x, y, N, dN_dxi, dN_deta = element_x_y(float(xi_c), float(eta_c), xe)
                    dN_dx, dN_dy, _detJ = element_dN_dxdy(dN_dxi, dN_deta, xe)

                    Hgp = crack.H(x, y) if crack_active else -1.0
                    B, edofs = build_B_enriched(
                        x,
                        y,
                        N,
                        dN_dx,
                        dN_dy,
                        xe,
                        conn,
                        dofs,
                        crack,
                        tip_enr_radius,
                        crack_active,
                        Hgp,
                        tip_x,
                        tip_y,
                    )
                    qe = q[np.asarray(edofs, dtype=int)]
                    eps = B @ qe

                    if use_mp_arrays:
                        assert isinstance(mp_states_comm, BulkStateArrays)
                        if use_bulk_numba:
                            dt0 = float(mp_states_comm.damage_t[e, gpid])
                            dc0 = float(mp_states_comm.damage_c[e, gpid])
                            wpl0 = float(mp_states_comm.w_plastic[e, gpid])
                            wft0 = float(mp_states_comm.w_fract_t[e, gpid])
                            wfc0 = float(mp_states_comm.w_fract_c[e, gpid])
                            eps_p6_old = np.asarray(mp_states_comm.eps_p6[e, gpid, :], dtype=float)
                            ezz_old = float(mp_states_comm.eps_zz[e, gpid])
                            kappa_old = float(mp_states_comm.kappa[e, gpid])
                            kt0 = float(mp_states_comm.kappa_t[e, gpid])
                            kc0 = float(mp_states_comm.kappa_c[e, gpid])
                            cwt0 = float(mp_states_comm.cdp_w_t[e, gpid])
                            ceps0 = float(mp_states_comm.cdp_eps_in_c[e, gpid])

                            if int(bulk_kind) == 1:
                                E = float(bulk_params[0])
                                nu = float(bulk_params[1])
                                sig, _Ct = elastic_integrate_plane_stress_numba(eps, E, nu, dt0, dc0)
                                eps_p6_new = eps_p6_old
                                ezz_new = ezz_old
                                kappa_new = kappa_old
                                dW = 0.0
                            else:
                                E = float(bulk_params[0])
                                nu = float(bulk_params[1])
                                alpha = float(bulk_params[2])
                                k0 = float(bulk_params[3])
                                Hh = float(bulk_params[4])
                                sig, _Ct, eps_p6_new, ezz_new, kappa_new, dW = dp_integrate_plane_stress_numba(
                                    eps,
                                    eps_p6_old,
                                    ezz_old,
                                    kappa_old,
                                    E,
                                    nu,
                                    alpha,
                                    k0,
                                    Hh,
                                )
                            eps_p3_new = np.array([eps_p6_new[0], eps_p6_new[1], eps_p6_new[3]], dtype=float)
                            if isinstance(mp_updates, BulkStatePatch):
                                mp_updates.add_values(
                                    e,
                                    gpid,
                                    eps=np.asarray(eps, dtype=float),
                                    sigma=np.asarray(sig, dtype=float),
                                    eps_p=eps_p3_new,
                                    damage_t=dt0,
                                    damage_c=dc0,
                                    kappa=float(kappa_new),
                                    w_plastic=float(wpl0 + dW),
                                    w_fract_t=wft0,
                                    w_fract_c=wfc0,
                                    eps_p6=np.asarray(eps_p6_new, dtype=float),
                                    eps_zz=float(ezz_new),
                                    kappa_t=kt0,
                                    kappa_c=kc0,
                                    cdp_w_t=cwt0,
                                    cdp_eps_in_c=ceps0,
                                )
                            else:
                                mp = mp_default().copy_shallow()
                                mp.eps = np.asarray(eps, dtype=float)
                                mp.sigma = np.asarray(sig, dtype=float)
                                mp.eps_p = eps_p3_new
                                mp.damage_t = dt0
                                mp.damage_c = dc0
                                mp.kappa = float(kappa_new)
                                mp.w_plastic = float(wpl0 + dW)
                                mp.w_fract_t = wft0
                                mp.w_fract_c = wfc0
                                mp.extra["eps_p6"] = np.asarray(eps_p6_new, dtype=float)
                                mp.extra["eps_zz"] = float(ezz_new)
                                mp.extra["kappa_t"] = kt0
                                mp.extra["kappa_c"] = kc0
                                mp.extra["cdp_w_t"] = cwt0
                                mp.extra["cdp_eps_in_c"] = ceps0
                                mp_updates[(e, gpid)] = mp
                        else:
                            mp0 = mp_states_comm.get_mp(e, gpid)
                            mp = mp0.copy_shallow()
                            material.integrate(mp, eps)
                            if isinstance(mp_updates, BulkStatePatch):
                                mp_updates.add(e, gpid, mp)
                            else:
                                mp_updates[(e, gpid)] = mp
                    elif mp_states_comm is not None:
                        mp0 = mp_states_comm.get((e, gpid), mp_default())
                        mp = mp0.copy_shallow()
                        material.integrate(mp, eps)
                        mp_updates[(e, gpid)] = mp
                    else:
                        mp = mp_default().copy_shallow()
                        material.integrate(mp, eps)

        # cohesive line integration in this element
        if crack_active:
            seg = clip_segment_to_bbox(p0, pt, xmin, xmax, ymin, ymax)
            if seg is not None:
                qA, qB = seg
                v = qB - qA
                Ls = float(np.linalg.norm(v))
                if Ls > 1e-12:
                    nvec = crack.nvec()

                    for igp, shat in enumerate(g):
                        p = 0.5 * (1.0 - shat) * qA + 0.5 * (1.0 + shat) * qB
                        x = float(p[0])
                        y = float(p[1])
                        jac = 0.5 * Ls

                        xi_s = element_local_xi_at_x(x, xe)
                        eta = element_local_eta_at_y(y, xe)
                        N, _dN_dxi, _dN_deta = q4_shape(xi_s, eta)

                        epsn = 1e-6 * max(1e-12, min(xmax - xmin, ymax - ymin))
                        Fp, _, _ = branch_F_and_grad(x + epsn * nvec[0], y + epsn * nvec[1], tip_x, tip_y)
                        Fm, _, _ = branch_F_and_grad(x - epsn * nvec[0], y - epsn * nvec[1], tip_x, tip_y)
                        dF_jump = Fp - Fm

                        edofs = []
                        gvec = []

                        for a in range(4):
                            n = int(conn[a])
                            if dofs.H[n, 0] >= 0:
                                edofs.append(int(dofs.H[n, 0]))
                                gvec.append(2.0 * float(N[a]) * float(nvec[0]))
                            if dofs.H[n, 1] >= 0:
                                edofs.append(int(dofs.H[n, 1]))
                                gvec.append(2.0 * float(N[a]) * float(nvec[1]))

                        for a in range(4):
                            n = int(conn[a])
                            if dofs.tip[n, 0, 0] >= 0:
                                for k in range(4):
                                    edofs.append(int(dofs.tip[n, k, 0]))
                                    gvec.append(float(N[a]) * float(dF_jump[k]) * float(nvec[0]))
                                    edofs.append(int(dofs.tip[n, k, 1]))
                                    gvec.append(float(N[a]) * float(dF_jump[k]) * float(nvec[1]))

                        if len(edofs) > 0:
                            edofs = np.asarray(edofs, dtype=int)
                            gvec = np.asarray(gvec, dtype=float)
                            delta = float(np.dot(gvec, q[edofs]))

                            # --- Cohesive state update (Phase 2: value-kernel option) ---
                            if use_numba and (coh_params is not None) and use_coh_arrays:
                                assert isinstance(coh_states_comm, CohesiveStateArrays)
                                dm_old, dmg_old = coh_states_comm.get_values(e, igp, k=0)
                                T, ksec, dm_new, dmg_new = cohesive_update_values_numba(
                                    delta,
                                    dm_old,
                                    dmg_old,
                                    coh_params,
                                    visc_damp=float(visc_damp),
                                )
                                assert isinstance(coh_updates, CohesiveStatePatch)
                                coh_updates.add_values(0, e, igp, delta_max=dm_new, damage=dmg_new)
                                dm_for_energy = float(dm_new)
                            else:
                                if use_coh_arrays:
                                    assert isinstance(coh_states_comm, CohesiveStateArrays)
                                    st = coh_states_comm.get_state(e, igp, k=0)
                                else:
                                    st = coh_states_comm.get((e, igp), CohesiveState())
                                T, ksec, st2 = cohesive_update(law, delta, st, visc_damp=visc_damp)
                                if isinstance(coh_updates, CohesiveStatePatch):
                                    coh_updates.add(0, e, igp, st2)
                                else:
                                    coh_updates[(e, igp)] = st2
                                dm_for_energy = float(st2.delta_max)

                            # Save cohesive quadrature info for energy accounting.
                            wline = float(w[igp]) * float(jac) * float(thickness)
                            coh_wgt.append(float(wline))
                            coh_delta_max.append(float(dm_for_energy))

                            fint[edofs] += gvec * T * wline
                            Kc = np.outer(gvec, gvec) * (ksec * wline)
                            rr = np.repeat(edofs, len(edofs))
                            cc = np.tile(edofs, len(edofs))
                            rows.extend(rr.tolist())
                            cols.extend(cc.tolist())
                            data.extend(Kc.reshape(-1).tolist())

    K = sp.coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()

    # stabilization on enriched dofs
    if float(k_stab) > 0.0:
        nnode = nodes.shape[0]
        diagK = K.diagonal()
        if 2 * nnode <= diagK.size:
            k_ref = float(np.max(np.abs(diagK[: 2 * nnode])))
        else:
            k_ref = float(np.max(np.abs(diagK)))
        if (not np.isfinite(k_ref)) or (k_ref <= 0.0):
            k_ref = 1.0
        k_stab_eff = float(k_stab)
        if k_stab_eff < 1.0:
            k_stab_eff *= k_ref
        stab = np.zeros(ndof, dtype=float)
        stab[2 * nnode :] = k_stab_eff
        K = K + sp.diags(stab, 0, shape=(ndof, ndof), format="csr")

    aux = {
        "gp_pos": np.asarray(gp_pos, dtype=float),
        "gp_sig": np.asarray(gp_sig, dtype=float),
        "gp_sigma1": np.asarray(gp_s1, dtype=float),
        "gp_eid": np.asarray(gp_eid, dtype=int),
        "gp_ipid": np.asarray(gp_ipid, dtype=int),
        "gp_weight": np.asarray(gp_wgt, dtype=float),
        "coh_weight": np.asarray(coh_wgt, dtype=float),
        "coh_delta_max": np.asarray(coh_delta_max, dtype=float),
    }
    return K, fint, coh_updates, mp_updates, aux