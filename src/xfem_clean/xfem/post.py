"""Post-processing helpers (nodal stress averaging)."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import numpy as np

from xfem_clean.xfem.dofs_single import XFEMDofs, build_xfem_dofs
from xfem_clean.xfem.enrichment_single import build_B_enriched, TipEnrichmentType
from xfem_clean.xfem.geometry import XFEMCrack
from xfem_clean.xfem.material import plane_stress_C
from xfem_clean.xfem.model import XFEMModel
from xfem_clean.xfem.q4_utils import element_x_y, element_dN_dxdy
from xfem_clean.material_point import MaterialPoint
from xfem_clean.xfem.state_arrays import BulkStateArrays


def nodal_average_state_fields(
    nodes: np.ndarray,
    elems: np.ndarray,
    q: np.ndarray,
    model: XFEMModel,
    crack: XFEMCrack,
    mp_states: Optional[Union[Dict[Tuple[int, int], MaterialPoint], BulkStateArrays]] = None,
) -> Dict[str, np.ndarray]:
    """Return nodal-averaged stress invariants and material state fields.

    This function is retrocompatible with the purely elastic solver:
    - If `mp_states` is provided, stresses and damage are read from those
      integration-point states.
    - If not, stresses are recomputed from `q` assuming linear elasticity and
      damage fields are returned as zeros.

    The averaging uses the *canonical 2×2 Gauss points per element* so the
    mapping (element_id, gp_id) stays stable even when cut elements are
    subdivided for integration elsewhere in the code.
    """

    nnode = int(nodes.shape[0])

    xs = np.unique(nodes[:, 0]); ys = np.unique(nodes[:, 1])
    xs.sort(); ys.sort()
    dx = float(np.min(np.diff(xs))) if len(xs) > 1 else float(model.L)
    dy = float(np.min(np.diff(ys))) if len(ys) > 1 else float(model.H)

    if not crack.active:
        std = np.arange(2 * nnode, dtype=int).reshape(nnode, 2)
        H = -np.ones((nnode, 2), dtype=int)
        tipd = -np.ones((nnode, 4, 2), dtype=int)
        dofs = XFEMDofs(
            std=std,
            H=H,
            tip=tipd,
            ndof=2 * nnode,
            H_nodes=np.zeros(nnode, dtype=bool),
            tip_nodes=np.zeros(nnode, dtype=bool),
        )
    else:
        stop_y = float(min(crack.stop_y, 0.5 * model.H))
        y_max_init = float(min(stop_y, model.cand_ymax_factor * dy))
        dofs = build_xfem_dofs(
            nodes,
            elems,
            crack,
            H_region_ymax=y_max_init,
            tip_patch=(
                crack.tip_x - model.tip_enr_radius,
                crack.tip_x + model.tip_enr_radius,
                max(0.0, crack.tip_y - model.tip_enr_radius),
                crack.tip_y + model.tip_enr_radius,
            ),
        )

    C = plane_stress_C(model.E, model.nu)

    sigma1 = np.zeros(nnode, dtype=float)
    mises = np.zeros(nnode, dtype=float)
    tresca = np.zeros(nnode, dtype=float)
    dmg_t = np.zeros(nnode, dtype=float)
    dmg_c = np.zeros(nnode, dtype=float)
    kappa = np.zeros(nnode, dtype=float)
    wsum = np.zeros(nnode, dtype=float)

    g = np.array([-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)], dtype=float)
    w = np.array([1.0, 1.0], dtype=float)

    tip_x = float(crack.tip_x)
    tip_y = float(crack.tip_y)

    for e, conn in enumerate(elems):
        conn = np.asarray(conn, dtype=int)
        xe = nodes[conn, :]
        for ixi in range(2):
            for ieta in range(2):
                xi = float(g[ixi]); eta = float(g[ieta])
                wg = float(w[ixi] * w[ieta])
                x, y, N, dN_dxi, dN_deta = element_x_y(xi, eta, xe)
                dN_dx, dN_dy, detJ = element_dN_dxdy(dN_dxi, dN_deta, xe)

                gpid = int(ixi + 2 * ieta)
                mp = None
                if mp_states is not None:
                    if isinstance(mp_states, BulkStateArrays):
                        ee = int(e)
                        ii = int(gpid)
                        if 0 <= ee < mp_states.nelem and 0 <= ii < mp_states.max_ip and bool(mp_states.active[ee, ii]):
                            mp = mp_states.get_mp(ee, ii)
                    else:
                        mp = mp_states.get((int(e), int(gpid)))

                if mp is None:
                    # retrocompatible fallback: recompute stresses from q
                    heaviside_side = crack.H(x, y) if crack.active else -1.0
                    tip_enr_type: TipEnrichmentType = getattr(model, "tip_enrichment_type", "non_singular_cohesive")
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
                        tip_enr_radius=model.tip_enr_radius,
                        crack_active=crack.active,
                        heaviside_side=heaviside_side,
                        tip_x=tip_x,
                        tip_y=tip_y,
                        tip_enrichment_type=tip_enr_type,
                    )
                    qe = q[np.asarray(edofs, dtype=int)]
                    eps = B @ qe
                    sig = C @ eps
                    dt = 0.0
                    dc = 0.0
                    kk = 0.0
                else:
                    sig = np.asarray(mp.sigma, dtype=float).reshape(3)
                    dt = float(getattr(mp, "damage_t", 0.0))
                    dc = float(getattr(mp, "damage_c", 0.0))
                    kk = float(getattr(mp, "kappa", 0.0))

                sxx = float(sig[0]); syy = float(sig[1]); sxy = float(sig[2])
                tr2 = 0.5 * (sxx + syy)
                rad = math.sqrt(max(0.0, (0.5 * (sxx - syy)) ** 2 + sxy * sxy))
                s1 = tr2 + rad
                s2 = tr2 - rad

                vm = math.sqrt(max(0.0, sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy))

                smax = max(s1, s2, 0.0)
                smin = min(s1, s2, 0.0)
                trc = smax - smin

                weight = wg * float(detJ)

                for a_local, a in enumerate(conn):
                    wa = float(N[a_local]) * weight
                    sigma1[a] += wa * s1
                    mises[a] += wa * vm
                    tresca[a] += wa * trc
                    dmg_t[a] += wa * dt
                    dmg_c[a] += wa * dc
                    kappa[a] += wa * kk
                    wsum[a] += wa

    wsum = np.where(wsum <= 1e-30, 1.0, wsum)
    sigma1 /= wsum
    mises /= wsum
    tresca /= wsum

    dmg_t /= wsum
    dmg_c /= wsum
    kappa /= wsum

    return {
        "sigma1": sigma1,
        "mises": mises,
        "tresca": tresca,
        "damage_t": dmg_t,
        "damage_c": dmg_c,
        "kappa": kappa,
    }


def nodal_average_stress_fields(
    nodes: np.ndarray,
    elems: np.ndarray,
    q: np.ndarray,
    model: XFEMModel,
    crack: XFEMCrack,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backwards-compatible wrapper returning only stress invariants."""

    out = nodal_average_state_fields(nodes, elems, q, model, crack, mp_states=None)
    return out["sigma1"], out["mises"], out["tresca"]
