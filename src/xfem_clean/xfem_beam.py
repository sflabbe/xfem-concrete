"""
2D RC beam prototype with:
- Q4 plane-stress elements
- Concrete Damaged Plasticity (CDP) inspired by Abaqus (Lubliner/Lee-Fenves form)
- Rankine term via sigma_max (maximum principal stress)
- Separate tension/compression damage variables (dt, dc)
- Bilinear rebar as a truss-line (nonlinear axial) embedded at y=cover
- Displacement control, MODIFIED NEWTON (secant K assembled once per load step) + line-search

This is a research prototype: simplified but numerically robust for calibration workflows.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# -----------------------------
# Optional acceleration via Numba
# -----------------------------
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



# -----------------------------
# Q4 shape functions (bilinear)
# -----------------------------
def q4_shape(xi: float, eta: float):
    # N1..N4 (counter-clockwise)
    N = 0.25 * np.array(
        [(1 - xi) * (1 - eta),
         (1 + xi) * (1 - eta),
         (1 + xi) * (1 + eta),
         (1 - xi) * (1 + eta)],
        dtype=float,
    )
    dN_dxi = 0.25 * np.array(
        [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)], dtype=float
    )
    dN_deta = 0.25 * np.array(
        [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)], dtype=float
    )
    return N, dN_dxi, dN_deta


def plane_stress_C(E: float, nu: float) -> np.ndarray:
    fac = E / (1.0 - nu * nu)
    C = fac * np.array(
        [[1.0, nu, 0.0],
         [nu, 1.0, 0.0],
         [0.0, 0.0, 0.5 * (1.0 - nu)]],
        dtype=float
    )
    return C


def voigt_to_sig3(sig: np.ndarray) -> np.ndarray:
    # plane stress -> embed in 3D with sigma_zz=0
    sxx, syy, txy = sig
    S = np.zeros((3, 3), dtype=float)
    S[0, 0] = sxx
    S[1, 1] = syy
    S[0, 1] = txy
    S[1, 0] = txy
    return S


def invariants_p_q(sig3: np.ndarray) -> Tuple[float, float, np.ndarray]:
    # Abaqus convention: compression positive -> p = -I1/3
    I1 = float(np.trace(sig3))
    p = -I1 / 3.0
    s = sig3 - (I1 / 3.0) * np.eye(3)
    J2 = 0.5 * float(np.sum(s * s))
    q = math.sqrt(max(0.0, 3.0 * J2))
    return p, q, s


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


def apply_damage_unilateral(sig_eff: np.ndarray, dt: float, dc: float) -> np.ndarray:
    # simple unilateral damage on principal stresses (2D spectral)
    sxx, syy, txy = sig_eff
    S2 = np.array([[sxx, txy], [txy, syy]], dtype=float)
    vals, vecs = np.linalg.eigh(S2)
    vals_d = vals.copy()
    for i in range(2):
        if vals_d[i] >= 0.0:
            vals_d[i] *= (1.0 - dt)
        else:
            vals_d[i] *= (1.0 - dc)
    Sd = vecs @ np.diag(vals_d) @ vecs.T
    return np.array([Sd[0, 0], Sd[1, 1], Sd[0, 1]], dtype=float)


def von_mises_plane_stress(sig: np.ndarray) -> float:
    sxx, syy, txy = map(float, sig)
    return math.sqrt(max(0.0, sxx * sxx - sxx * syy + syy * syy + 3.0 * txy * txy))


def tresca_plane_stress(sig: np.ndarray) -> float:
    sxx, syy, txy = map(float, sig)
    S2 = np.array([[sxx, txy], [txy, syy]], dtype=float)
    vals = np.linalg.eigvalsh(S2)
    smax = float(max(vals[1], 0.0))
    smin = float(min(vals[0], 0.0))
    return 0.5 * (smax - smin)


# -----------------------------
# Numba kernels (sparse triplets)
# -----------------------------
@njit(cache=True)
def _q4_ref_derivs():
    # gauss points +/- 1/sqrt(3)
    g = 1.0 / math.sqrt(3.0)
    xis = np.array([-g,  g,  g, -g], dtype=np.float64)
    etas= np.array([-g, -g,  g,  g], dtype=np.float64)

    dN_dxi = np.empty((4,4), dtype=np.float64)
    dN_deta= np.empty((4,4), dtype=np.float64)
    for k in range(4):
        xi = xis[k]; eta = etas[k]
        dN_dxi[k,0]  = -0.25*(1.0-eta)
        dN_dxi[k,1]  =  0.25*(1.0-eta)
        dN_dxi[k,2]  =  0.25*(1.0+eta)
        dN_dxi[k,3]  = -0.25*(1.0+eta)

        dN_deta[k,0] = -0.25*(1.0-xi)
        dN_deta[k,1] = -0.25*(1.0+xi)
        dN_deta[k,2] =  0.25*(1.0+xi)
        dN_deta[k,3] =  0.25*(1.0-xi)
    return dN_dxi, dN_deta


@njit(cache=True, parallel=True)
def _assemble_bulk_secant_triplets_numba(nodes, elems, C0, dmg_elem, thickness):
    nE = elems.shape[0]
    ndof = 2 * nodes.shape[0]

    rows = np.empty(nE * 64, dtype=np.int64)
    cols = np.empty(nE * 64, dtype=np.int64)
    data = np.empty(nE * 64, dtype=np.float64)

    dN_dxi, dN_deta = _q4_ref_derivs()

    for e in prange(nE):
        conn = elems[e]
        x = np.empty(4, dtype=np.float64)
        y = np.empty(4, dtype=np.float64)
        for a in range(4):
            x[a] = nodes[conn[a], 0]
            y[a] = nodes[conn[a], 1]

        dmg = dmg_elem[e]
        # Csec = (1-dmg) * C0
        Csec = (1.0 - dmg) * C0

        Ke = np.zeros((8,8), dtype=np.float64)

        for k in range(4):
            # Jacobian
            J00 = 0.0; J01 = 0.0; J10 = 0.0; J11 = 0.0
            for a in range(4):
                J00 += dN_dxi[k,a]  * x[a]
                J01 += dN_dxi[k,a]  * y[a]
                J10 += dN_deta[k,a] * x[a]
                J11 += dN_deta[k,a] * y[a]
            detJ = J00*J11 - J01*J10
            invdet = 1.0 / detJ
            iJ00 =  invdet * J11
            iJ01 = -invdet * J01
            iJ10 = -invdet * J10
            iJ11 =  invdet * J00

            dN_dx = np.empty(4, dtype=np.float64)
            dN_dy = np.empty(4, dtype=np.float64)
            for a in range(4):
                dN_dx[a] = iJ00*dN_dxi[k,a] + iJ01*dN_deta[k,a]
                dN_dy[a] = iJ10*dN_dxi[k,a] + iJ11*dN_deta[k,a]

            # B matrix (3x8)
            B = np.zeros((3,8), dtype=np.float64)
            for a in range(4):
                B[0,2*a]   = dN_dx[a]
                B[1,2*a+1] = dN_dy[a]
                B[2,2*a]   = dN_dy[a]
                B[2,2*a+1] = dN_dx[a]

            # Ke += B^T Csec B * detJ * thickness
            # compute CB = Csec @ B (3x8)
            CB = np.zeros((3,8), dtype=np.float64)
            for i in range(3):
                for j in range(8):
                    s = 0.0
                    for k2 in range(3):
                        s += Csec[i,k2]*B[k2,j]
                    CB[i,j] = s

            w = detJ * thickness
            for i in range(8):
                for j in range(8):
                    s = 0.0
                    for k2 in range(3):
                        s += B[k2,i]*CB[k2,j]
                    Ke[i,j] += s * w

        # scatter to triplets
        edofs = np.empty(8, dtype=np.int64)
        for a in range(4):
            edofs[2*a]   = 2*conn[a]
            edofs[2*a+1] = 2*conn[a] + 1

        base = e * 64
        t = 0
        for i in range(8):
            for j in range(8):
                rows[base+t] = edofs[i]
                cols[base+t] = edofs[j]
                data[base+t] = Ke[i,j]
                t += 1

    return rows, cols, data, ndof


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

# -----------------------------
# CDP material (prototype)
# -----------------------------
@dataclass
class CDPMaterial:
    # elastic
    E: float
    nu: float
    # CDP potential (Abaqus-like)
    psi_deg: float = 36.0         # dilation angle
    ecc: float = 0.1              # eccentricity
    fb0_fc0: float = 1.16         # biaxial/uniaxial ratio
    Kc: float = 2.0 / 3.0         # shape factor
    # characteristic length for crack-opening -> strain mapping
    lch: float = 0.05             # [m]
    # tension softening table (crack opening in m, stress in Pa, damage in [0,1])
    w_tab: np.ndarray = None
    sig_t_tab: np.ndarray = None
    dt_tab: np.ndarray = None
    # compression table (inelastic strain in [-], stress in Pa, damage in [0,1])
    eps_in_c_tab: np.ndarray = None
    sig_c_tab: np.ndarray = None
    dc_tab: np.ndarray = None
    # initial strengths
    f_t0: float = 2.0e6           # Pa (will be overwritten)
    f_c0: float = 20.0e6          # Pa (will be overwritten)

    def C(self) -> np.ndarray:
        return plane_stress_C(self.E, self.nu)

    def _interp(self, x: float, x_tab: np.ndarray, y_tab: np.ndarray) -> float:
        if x <= float(x_tab[0]):
            return float(y_tab[0])
        if x >= float(x_tab[-1]):
            return float(y_tab[-1])
        return float(np.interp(x, x_tab, y_tab))

    def sigma_t_eff(self, w: float) -> float:
        return self._interp(w, self.w_tab, self.sig_t_tab)

    def damage_t(self, w: float) -> float:
        return self._interp(w, self.w_tab, self.dt_tab)

    def sigma_c_eff(self, eps_in: float) -> float:
        return self._interp(eps_in, self.eps_in_c_tab, self.sig_c_tab)

    def damage_c(self, eps_in: float) -> float:
        return self._interp(eps_in, self.eps_in_c_tab, self.dc_tab)


@dataclass
class CDPState:
    eps_p: np.ndarray  # (3,) plastic strain in Voigt
    w_t: float         # tensile inelastic measure (crack opening eq.) [m]
    eps_in_c: float    # compressive inelastic strain measure [-]
    dt: float
    dc: float

    @staticmethod
    def zeros():
        return CDPState(eps_p=np.zeros(3, dtype=float), w_t=0.0, eps_in_c=0.0, dt=0.0, dc=0.0)


def cdp_yield(sig_eff: np.ndarray, state: CDPState, mat: CDPMaterial) -> float:
    sig3 = voigt_to_sig3(sig_eff)
    p, q, _ = invariants_p_q(sig3)
    sig_max = principal_max_2d(sig_eff)
    sig_min = principal_min_2d(sig_eff)

    fbfc = mat.fb0_fc0
    alpha = (fbfc - 1.0) / (2.0 * fbfc - 1.0)

    # current effective strengths from tables
    # Floors prevent ill-conditioning when tension strength -> 0 after full cracking.
    sig_c_tab = max(1e-6, mat.sigma_c_eff(state.eps_in_c))
    sig_t_tab = max(1e-6, mat.sigma_t_eff(state.w_t))
    sig_t_floor = 0.02 * mat.f_t0
    sig_c_floor = 0.05 * mat.f_c0
    sig_t = max(sig_t_tab, sig_t_floor)
    sig_c = max(sig_c_tab, sig_c_floor)

    beta = (sig_c / sig_t) * (1.0 - alpha) - (1.0 + alpha)
    gamma = 3.0 * (1.0 - mat.Kc) / (2.0 * mat.Kc - 1.0)

    term = q - 3.0 * alpha * p + beta * max(sig_max, 0.0) - gamma * max(-sig_min, 0.0)
    F = term / (1.0 - alpha) - sig_c
    return float(F)


def cdp_flow_dir(sig_eff: np.ndarray, mat: CDPMaterial) -> np.ndarray:
    # Plastic potential (Abaqus-like):
    # G = sqrt( (ecc * f_t0 * tan(psi))^2 + q^2 ) - p * tan(psi)
    sig3 = voigt_to_sig3(sig_eff)
    p, q, s = invariants_p_q(sig3)

    psi = math.radians(mat.psi_deg)
    tanpsi = math.tan(psi)
    A = mat.ecc * mat.f_t0 * tanpsi
    denom = math.sqrt(A * A + q * q)
    dg_dq = 0.0 if denom < 1e-12 else (q / denom)

    dp_dsig = -(1.0 / 3.0) * np.eye(3)
    dq_dsig = np.zeros((3, 3), dtype=float) if q < 1e-12 else (3.0 / (2.0 * q)) * s

    dG = (-tanpsi) * dp_dsig + dg_dq * dq_dsig
    # return 2D Voigt components
    return np.array([dG[0, 0], dG[1, 1], dG[0, 1]], dtype=float)


def cdp_update(strain: np.ndarray, state0: CDPState, mat: CDPMaterial) -> Tuple[np.ndarray, CDPState, float]:
    """
    Return nominal stress (Voigt), updated state, and a Rankine index = sigma1_eff / f_t0.

    NOTE:
    - For robustness we freeze the flow direction at the trial stress (common "modified return mapping").
    - We bracket the plastic multiplier using a cheap estimate to avoid expensive doubling from ~0.
    """
    C = mat.C()
    sig_tr = C @ (strain - state0.eps_p)
    rankine = principal_max_2d(sig_tr) / max(1e-9, mat.f_t0)

    Ftr = cdp_yield(sig_tr, state0, mat)
    if Ftr <= 1e-10:
        sig_nom = apply_damage_unilateral(sig_tr, state0.dt, state0.dc)
        return sig_nom, state0, float(rankine)

    m = cdp_flow_dir(sig_tr, mat)
    Cm = C @ m
    denom_est = float(np.dot(m, Cm)) + 1e-18
    dl_est = max(0.0, float(Ftr) / denom_est)

    mnorm = float(np.linalg.norm(m)) + 1e-14
    in_tension = (principal_max_2d(sig_tr) >= 0.0)

    def F_of(dl: float) -> float:
        sig = sig_tr - dl * Cm
        st = CDPState(
            eps_p=state0.eps_p + dl * m,
            w_t=state0.w_t,
            eps_in_c=state0.eps_in_c,
            dt=state0.dt,
            dc=state0.dc,
        )
        if in_tension:
            st.w_t = state0.w_t + dl * mnorm * mat.lch
        else:
            st.eps_in_c = state0.eps_in_c + dl * mnorm
        return cdp_yield(sig, st, mat)

    lo = 0.0
    hi = max(1e-14, dl_est)

    F_hi = F_of(hi)
    grow = 0
    while F_hi > 0.0 and hi < 1e2:
        hi *= 2.0
        F_hi = F_of(hi)
        grow += 1
        if grow > 40:
            break

    # bisection (few iters are enough for this prototype)
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        if F_of(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    dl = hi

    eps_p = state0.eps_p + dl * m
    w_t = state0.w_t
    eps_in_c = state0.eps_in_c
    if in_tension:
        w_t = state0.w_t + dl * mnorm * mat.lch
    else:
        eps_in_c = state0.eps_in_c + dl * mnorm

    dt = float(np.clip(mat.damage_t(w_t), 0.0, 0.9999))
    dc = float(np.clip(mat.damage_c(eps_in_c), 0.0, 0.9999))
    state = CDPState(eps_p=eps_p, w_t=w_t, eps_in_c=eps_in_c, dt=dt, dc=dc)

    sig_eff = C @ (strain - eps_p)
    sig_nom = apply_damage_unilateral(sig_eff, dt, dc)
    return sig_nom, state, float(rankine)


# -----------------------------
# Rebar: bilinear truss line
# -----------------------------
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
def structured_quad_mesh(L: float, H: float, nx: int, ny: int):
    xs = np.linspace(0.0, L, nx + 1)
    ys = np.linspace(0.0, H, ny + 1)
    nodes = np.array([[x, y] for y in ys for x in xs], dtype=float)

    def nid(i, j):  # i along x, j along y
        return j * (nx + 1) + i

    elems = []
    for j in range(ny):
        for i in range(nx):
            n1 = nid(i, j)
            n2 = nid(i + 1, j)
            n3 = nid(i + 1, j + 1)
            n4 = nid(i, j + 1)
            elems.append([n1, n2, n3, n4])
    return nodes, np.array(elems, dtype=int)


def element_B_detJ(xi: float, eta: float, xe: np.ndarray):
    _, dN_dxi, dN_deta = q4_shape(xi, eta)
    J = np.zeros((2, 2), dtype=float)
    J[0, 0] = float(np.dot(dN_dxi, xe[:, 0]))
    J[0, 1] = float(np.dot(dN_dxi, xe[:, 1]))
    J[1, 0] = float(np.dot(dN_deta, xe[:, 0]))
    J[1, 1] = float(np.dot(dN_deta, xe[:, 1]))
    detJ = float(np.linalg.det(J))
    invJ = np.linalg.inv(J)

    dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
    dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta

    B = np.zeros((3, 8), dtype=float)
    for a in range(4):
        B[0, 2 * a] = dN_dx[a]
        B[1, 2 * a + 1] = dN_dy[a]
        B[2, 2 * a] = dN_dy[a]
        B[2, 2 * a + 1] = dN_dx[a]
    return B, detJ

def precompute_B_detJ(nodes: np.ndarray, elems: np.ndarray):
    """
    Precompute B matrices and detJ for all elements and Gauss points.
    This drastically reduces Python overhead in bulk_internal_force and line-search.
    Returns:
      B_all[e,igp,3,8], detJ_all[e,igp]
    """
    gauss = [(-1/math.sqrt(3), -1/math.sqrt(3)),
             (+1/math.sqrt(3), -1/math.sqrt(3)),
             (+1/math.sqrt(3), +1/math.sqrt(3)),
             (-1/math.sqrt(3), +1/math.sqrt(3))]
    nE = elems.shape[0]
    B_all = np.zeros((nE, 4, 3, 8), dtype=float)
    detJ_all = np.zeros((nE, 4), dtype=float)
    for e, conn in enumerate(elems):
        xe = nodes[conn, :]
        for igp, (xi, eta) in enumerate(gauss):
            B, detJ = element_B_detJ(xi, eta, xe)
            B_all[e, igp, :, :] = B
            detJ_all[e, igp] = detJ
    return B_all, detJ_all


def assemble_bulk_secant_K(nodes: np.ndarray, elems: np.ndarray, mat: CDPMaterial, states_committed: List[List[CDPState]], thickness: float, precomp=None):
    """
    Modified Newton: assemble a secant stiffness once per load step, based on committed damage.
    Accelerated with numba (if available).
    """
    C0 = mat.C()
    nE = elems.shape[0]
    # per-element damage proxy from committed states
    dmg_elem = np.zeros(nE, dtype=float)
    for e in range(nE):
        dmg = 0.0
        for gp in range(4):
            st = states_committed[e][gp]
            if st.dt > dmg: dmg = st.dt
            if st.dc > dmg: dmg = st.dc
        dmg_elem[e] = dmg

    if NUMBA_AVAILABLE:
        rows, cols, data, ndof = _assemble_bulk_secant_triplets_numba(
            nodes.astype(np.float64),
            elems.astype(np.int64),
            C0.astype(np.float64),
            dmg_elem.astype(np.float64),
            float(thickness)
        )
        K = sp.coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()
        return K

    # fallback pure python
    nnode = nodes.shape[0]
    ndof = 2 * nnode
    rows, cols, data = [], [], []
    gauss = [(-1/math.sqrt(3), -1/math.sqrt(3)),
             (+1/math.sqrt(3), -1/math.sqrt(3)),
             (+1/math.sqrt(3), +1/math.sqrt(3)),
             (-1/math.sqrt(3), +1/math.sqrt(3))]

    for e, conn in enumerate(elems):
        xe = nodes[conn, :]
        dmg = dmg_elem[e]
        Csec = (1.0 - dmg) * C0
        Ke = np.zeros((8, 8), dtype=float)
        for igp, (xi, eta) in enumerate(gauss):
            if precomp is None:
                B, detJ = element_B_detJ(xi, eta, xe)
            else:
                B_all, detJ_all = precomp
                B = B_all[e, igp]
                detJ = float(detJ_all[e, igp])
            Ke += (B.T @ Csec @ B) * detJ * thickness
        edofs = []
        for a in conn:
            edofs.extend([2*a, 2*a+1])
        for i in range(8):
            for j in range(8):
                rows.append(edofs[i]); cols.append(edofs[j]); data.append(Ke[i, j])

    K = sp.coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()
    return K



def bulk_internal_force(nodes: np.ndarray, elems: np.ndarray, mat: CDPMaterial, u: np.ndarray, states_committed: List[List[CDPState]], thickness: float, precomp=None):
    """
    Compute bulk internal force and trial updated states (not committed).
    Returns: f_int (ndof_u,), states_trial, gp_stress_eff, gp_stress_nom, gp_rankine
    """
    nnode = nodes.shape[0]
    ndof = 2 * nnode
    f = np.zeros(ndof, dtype=float)
    states_trial: List[List[CDPState]] = [[CDPState.zeros() for _ in range(4)] for _ in range(len(elems))]

    gauss = [(-1/math.sqrt(3), -1/math.sqrt(3)),
             (+1/math.sqrt(3), -1/math.sqrt(3)),
             (+1/math.sqrt(3), +1/math.sqrt(3)),
             (-1/math.sqrt(3), +1/math.sqrt(3))]
    wgt = 1.0

    gp_sig_nom = np.zeros((len(elems), 4, 3), dtype=float)
    gp_rank = np.zeros((len(elems), 4), dtype=float)

    for e, conn in enumerate(elems):
        xe = nodes[conn, :]
        ue = np.zeros(8, dtype=float)
        for a in range(4):
            ue[2*a]   = u[2*conn[a]]
            ue[2*a+1] = u[2*conn[a]+1]

        fe = np.zeros(8, dtype=float)
        for igp, (xi, eta) in enumerate(gauss):
            B, detJ = element_B_detJ(xi, eta, xe)
            eps = B @ ue
            sig_nom, st, rank = cdp_update(eps, states_committed[e][igp], mat)

            fe += (B.T @ sig_nom) * detJ * wgt * thickness
            states_trial[e][igp] = st
            gp_sig_nom[e, igp, :] = sig_nom
            gp_rank[e, igp] = rank

        edofs = []
        for a in conn:
            edofs.extend([2*a, 2*a+1])
        for i in range(8):
            f[edofs[i]] += fe[i]

    return f, states_trial, gp_sig_nom, gp_rank


def apply_dirichlet(K: sp.csr_matrix, r: np.ndarray, fixed: Dict[int, float], u: np.ndarray):
    """
    Nonlinear Dirichlet handling:
    - Enforce u[dof] = value directly on the iterate u
    - The Newton solve is done only on free dofs: K_ff * du_f = -r_f
      with du_fixed = 0.
    """
    ndof = K.shape[0]
    all_ids = np.arange(ndof, dtype=int)
    fixed_ids = np.array(sorted(fixed.keys()), dtype=int)
    free = np.setdiff1d(all_ids, fixed_ids)

    # Enforce values on current iterate BEFORE extracting free equations.
    # Note: in a residual-based Newton method, if the element assembly already
    # evaluated r(u) and K(u) using the iterate with prescribed values enforced,
    # then the free residual is simply r[free] and the Newton system is
    #   K_ff * du_f = -r_f
    # with du_fixed = 0.
    for dof in fixed_ids:
        u[dof] = fixed[dof]

    K_ff = K[free, :][:, free]
    r_f = r[free]
    return free, K_ff, r_f, fixed_ids


# -----------------------------
# High-level model and solver
# -----------------------------
@dataclass
class Model:
    L: float
    H: float
    b: float
    cdp: CDPMaterial

    # steel (bilinear)
    steel_E: float = 200e9
    steel_fy: float = 500e6
    steel_fu: float = 540e6
    steel_Eh: float = 2.0e9
    steel_A_total: float = 2e-4  # m^2
    cover: float = 0.035         # m

    # pseudo-dynamics (quasi-static) via HHT-α
    rho: float = 2500.0          # kg/m^3 (concrete density for lumped mass)
    rayleigh_aM: float = 0.0     # C = aM*M + aK*K (Rayleigh damping)
    rayleigh_aK: float = 2e-3
    hht_alpha: float = -0.10     # must be in [-1/3, 0]
    dt: Optional[float] = None   # pseudo time step; if None -> total_time/nsteps
    total_time: float = 5.0      # ramp duration (quasi-static -> larger)

    # solver params
    newton_tol_r: float = 1e-3
    newton_tol_rel: float = 1e-6  # relative force residual tolerance
    newton_tol_du: float = 1e-9
    newton_maxit: int = 20



def run_analysis_static(model: Model, nx: int, ny: int, nsteps: int, umax: float, line_search: bool = True):
    nodes, elems = structured_quad_mesh(model.L, model.H, nx, ny)
    nnode = nodes.shape[0]
    ndof = 2 * nnode
    u = np.zeros(ndof, dtype=float)

    # precompute B matrices for speed
    precomp = precompute_B_detJ(nodes, elems)

    # precompute B matrices for speed
    precomp = precompute_B_detJ(nodes, elems)

    # committed CDP states: [elem][gp]
    states_comm = [[CDPState.zeros() for _ in range(4)] for _ in range(len(elems))]

    # supports: left pin, right roller
    left = int(np.argmin(nodes[:, 0]))
    right = int(np.argmax(nodes[:, 0]))
    fixed_base = {2*left: 0.0, 2*left+1: 0.0, 2*right+1: 0.0}

    # loading strip: top nodes near midspan
    y_top = model.H
    top_nodes = np.where(np.isclose(nodes[:, 1], y_top))[0]
    dx = model.L / nx
    load_halfwidth = 2.0 * dx
    load_nodes = top_nodes[np.where(np.abs(nodes[top_nodes, 0] - model.L/2) <= load_halfwidth)[0]]
    if len(load_nodes) == 0:
        load_nodes = np.array([int(top_nodes[np.argmin(np.abs(nodes[top_nodes,0]-model.L/2))])], dtype=int)
    load_dofs = [2*n + 1 for n in load_nodes]

    # rebar segments
    rebar_segs = prepare_rebar_segments(nodes, model.cover)

    results = []
    gp_sig_nom_last = None
    gp_rank_last = None
    states_last = None

    for step in range(1, nsteps + 1):
        u_imp = (step / nsteps) * umax

        fixed = dict(fixed_base)
        for dof in load_dofs:
            fixed[dof] = -u_imp  # downward

        # assemble secant K once per step (MODIFIED NEWTON)
        K_bulk = assemble_bulk_secant_K(nodes, elems, model.cdp, states_comm, thickness=model.b)
        # rebar tangent updated per iteration (depends on u), but we start with elastic
        # (still modified Newton for concrete bulk)
        K = K_bulk

        # Newton iterations with fixed K_bulk
        u_it = u.copy()
        # enforce prescribed values on iterate
        for dof, val in fixed.items():
            u_it[dof] = val
        for it in range(model.newton_maxit):
            # internal forces
            f_bulk, states_trial, gp_sig_nom, gp_rank = bulk_internal_force(
                nodes, elems, model.cdp, u_it, states_comm, thickness=model.b, precomp=precomp
            )

            f_rb, K_rb = rebar_contrib(
                nodes, rebar_segs, u_it, model.steel_A_total,
                model.steel_E, model.steel_fy, model.steel_fu, model.steel_Eh
            )

            r = f_bulk + f_rb  # residual = internal forces (no external loads; displacement control)

            # build tangent (bulk fixed + current rebar tangent)
            Kt = K_bulk + K_rb

            free, K_ff, r_f, fixed_ids = apply_dirichlet(Kt, r, fixed, u_it)
            rhs = -r_f

            # solve increment
            du_f = spla.spsolve(K_ff, rhs)

            # line-search on residual norm
            norm0 = float(np.linalg.norm(rhs))
            alpha = 1.0
            accepted = False

            if line_search:
                u_base = u_it.copy()
                for _ls in range(10):
                    u_try = u_base.copy()
                    u_try[free] += alpha * du_f
                    f_bulk_t, _, _, _ = bulk_internal_force(nodes, elems, model.cdp, u_try, states_comm, thickness=model.b, precomp=precomp)
                    f_rb_t, _ = rebar_contrib(nodes, rebar_segs, u_try, model.steel_A_total, model.steel_E, model.steel_fy, model.steel_fu, model.steel_Eh)
                    r_t = f_bulk_t + f_rb_t
                    _, _, r_f_t, _ = apply_dirichlet(Kt, r_t, fixed, u_try)
                    norm_t = float(np.linalg.norm(r_f_t))
                    if norm_t < norm0:
                        u_it = u_try
                        for dof, val in fixed.items():
                            u_it[dof] = val
                        accepted = True
                        break
                    alpha *= 0.5
            if not line_search or not accepted:
                u_it[free] += du_f
            for dof, val in fixed.items():
                u_it[dof] = val

            norm_du = float(np.linalg.norm(du_f))
            # recompute residual norm quickly for stop
            f_bulk2, states_trial2, gp_sig_nom2, gp_rank2 = bulk_internal_force(nodes, elems, model.cdp, u_it, states_comm, thickness=model.b)
            f_rb2, _ = rebar_contrib(nodes, rebar_segs, u_it, model.steel_A_total, model.steel_E, model.steel_fy, model.steel_fu, model.steel_Eh)
            r2 = f_bulk2 + f_rb2
            _, _, r_f2, _ = apply_dirichlet(Kt, r2, fixed, u_it)
            norm_r = float(np.linalg.norm(r_f2))

            if norm_r < model.newton_tol_r and norm_du < model.newton_tol_du:
                gp_sig_nom_last = gp_sig_nom2
                gp_rank_last = gp_rank2
                states_last = states_trial2
                break

        # commit step
        u = u_it
        if states_last is None:
            # if not converged, still commit last trial (prototype choice)
            states_last = states_trial
            gp_sig_nom_last = gp_sig_nom
            gp_rank_last = gp_rank
        states_comm = states_last

        # reactions -> load P
        # reaction forces are minus residual at constrained dofs
        f_bulk, _, _, _ = bulk_internal_force(nodes, elems, model.cdp, u, states_comm, thickness=model.b)
        f_rb, _ = rebar_contrib(nodes, rebar_segs, u, model.steel_A_total, model.steel_E, model.steel_fy, model.steel_fu, model.steel_Eh)
        r_full = f_bulk + f_rb
        P = -sum(r_full[d] for d in load_dofs)  # N
        M = (P * model.L / 4.0)                 # N*m (simply supported, midspan point load equivalent)

        # curvature proxy at midspan: use section curvature from epsilon_x across height (simple fit)
        kappa = curvature_midspan(nodes, elems, u, model.cdp, states_comm, x0=model.L/2, thickness=model.b)

        R = 1e20 if abs(kappa) < 1e-16 else 1.0 / kappa  # Use large value instead of inf

        results.append([step, u_imp, P, M, kappa, R])

        print(f"step={step:03d} u={u_imp*1e3:7.2f} mm  P={P/1e3:9.3f} kN  M={M/1e3:9.3f} kN·m  kappa={kappa:10.3e}  R={R:8.3f} m")

    return nodes, elems, u, results, states_comm, gp_sig_nom_last, gp_rank_last, rebar_segs

# -----------------------------
# Pseudo-dynamics for quasi-static: HHT-α integrator
# -----------------------------
def assemble_lumped_mass(nodes: np.ndarray, elems: np.ndarray, rho: float, thickness: float) -> np.ndarray:
    """
    Lumped mass (diagonal) for 2D plane stress:
      m_node = rho * thickness * area/4 (per element) summed over adjacent elements.
    Mass is assigned equally to ux and uy dofs.
    Returns: Mdiag of length ndof.
    """
    nnode = nodes.shape[0]
    mnode = np.zeros(nnode, dtype=float)

    for conn in elems:
        x = nodes[conn, 0]
        y = nodes[conn, 1]
        # quad area (split into triangles)
        area1 = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (y[1]-y[0])*(x[2]-x[0]))
        area2 = 0.5 * abs((x[2]-x[0])*(y[3]-y[0]) - (y[2]-y[0])*(x[3]-x[0]))
        area = max(1e-16, area1 + area2)
        me = rho * thickness * area
        share = 0.25 * me
        for a in conn:
            mnode[int(a)] += share

    ndof = 2 * nnode
    Mdiag = np.zeros(ndof, dtype=float)
    for i in range(nnode):
        Mdiag[2*i]   = mnode[i]
        Mdiag[2*i+1] = mnode[i]
    return Mdiag


def _hht_try_step(model: Model,
                  nodes: np.ndarray,
                  elems: np.ndarray,
                  precomp,
                  rebar_segs: np.ndarray,
                  Mdiag: np.ndarray,
                  M: sp.csr_matrix,
                  u_n: np.ndarray,
                  v_n: np.ndarray,
                  acc_n: np.ndarray,
                  f_int_n: np.ndarray,
                  states_comm: List[List[CDPState]],
                  fixed_base: Dict[int, float],
                  load_dofs: List[int],
                  u_imp_target: float,
                  dt: float,
                  line_search: bool = True):
    """
    Attempt a single implicit HHT-α step from state (u_n,v_n,acc_n,states_comm,f_int_n) to prescribed displacement u_imp_target.
    Returns: (ok, u_np1, v_np1, acc_np1, f_int_np1, states_comm_np1, gp_sig_nom_last, gp_rank_last, P_internal)
    """
    alpha = float(model.hht_alpha)
    gamma = 0.5 - alpha
    beta = 0.25 * (1.0 - alpha) * (1.0 - alpha)

    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0/(2.0*beta) - 1.0

    fixed = dict(fixed_base)
    for dof in load_dofs:
        fixed[dof] = -u_imp_target  # downward

    # predictors
    u_pred = u_n + dt * v_n + (0.5 - beta) * dt * dt * acc_n
    v_pred = v_n + (1.0 - gamma) * dt * acc_n

    u_it = u_pred.copy()
    for dof, val in fixed.items():
        u_it[dof] = val

    # previous weighted forces
    # (use secant damping from current K_bulk each iter; for g_prev we use C based on committed)
    K_bulk0 = assemble_bulk_secant_K(nodes, elems, model.cdp, states_comm, thickness=model.b, precomp=precomp)
    if model.rayleigh_aM != 0.0 or model.rayleigh_aK != 0.0:
        C0 = (float(model.rayleigh_aM) * M + float(model.rayleigh_aK) * K_bulk0).tocsr()
    else:
        C0 = sp.csr_matrix((u_n.size, u_n.size))

    g_prev = f_int_n + (C0 @ v_n) + (Mdiag * acc_n)

    gp_sig_nom_last = None
    gp_rank_last = None

    for it in range(model.newton_maxit):
        # kinematics from current u_it
        acc_it = a0 * (u_it - u_n) - a2 * v_n - a3 * acc_n
        v_it = v_n + dt * ((1.0 - gamma) * acc_n + gamma * acc_it)

        # internal forces + trial states (nonlinear)
        f_bulk, states_trial, gp_sig_nom, gp_rank = bulk_internal_force(
            nodes, elems, model.cdp, u_it, states_comm, thickness=model.b, precomp=precomp
        )
        f_rb, K_rb = rebar_contrib(
            nodes, rebar_segs, u_it, model.steel_A_total,
            model.steel_E, model.steel_fy, model.steel_fu, model.steel_Eh
        )
        f_int = f_bulk + f_rb

        # update secant bulk stiffness for this iterate (improves convergence)
        K_bulk = assemble_bulk_secant_K(nodes, elems, model.cdp, states_trial if it > 0 else states_comm, thickness=model.b, precomp=precomp)

        # Rayleigh damping
        if model.rayleigh_aM != 0.0 or model.rayleigh_aK != 0.0:
            C = (float(model.rayleigh_aM) * M + float(model.rayleigh_aK) * K_bulk).tocsr()
        else:
            C = sp.csr_matrix((u_n.size, u_n.size))

        f_d = C @ v_it
        f_m = Mdiag * acc_it

        # residual (no external forces besides constraints)
        r = (1.0 + alpha) * (f_int + f_d + f_m) - alpha * g_prev

        # tangent and effective stiffness
        Kt = K_bulk + K_rb
        K_eff = (1.0 + alpha) * (Kt + a1 * C + a0 * sp.diags(Mdiag, 0, shape=(u_n.size, u_n.size), format="csr"))

        free, K_ff, r_f, _ = apply_dirichlet(K_eff, r, fixed, u_it)
        rhs = -r_f

        norm_r = float(np.linalg.norm(rhs))
        ref = max(1.0, float(np.linalg.norm(((1.0 + alpha) * f_int)[free])))
        if norm_r < (model.newton_tol_r + model.newton_tol_rel * ref):
            gp_sig_nom_last = gp_sig_nom
            gp_rank_last = gp_rank
            states_comm = states_trial
            # finalize kinematics
            acc_np1 = a0 * (u_it - u_n) - a2 * v_n - a3 * acc_n
            v_np1 = v_n + dt * ((1.0 - gamma) * acc_n + gamma * acc_np1)
            f_int_np1 = f_int.copy()
            P_internal = -sum(f_int_np1[d] for d in load_dofs)
            return True, u_it, v_np1, acc_np1, f_int_np1, states_comm, gp_sig_nom_last, gp_rank_last, float(P_internal)

        du_f = spla.spsolve(K_ff, rhs)
        norm_du = float(np.linalg.norm(du_f))
        if norm_du < model.newton_tol_du:
            gp_sig_nom_last = gp_sig_nom
            gp_rank_last = gp_rank
            states_comm = states_trial
            acc_np1 = a0 * (u_it - u_n) - a2 * v_n - a3 * acc_n
            v_np1 = v_n + dt * ((1.0 - gamma) * acc_n + gamma * acc_np1)
            f_int_np1 = f_int.copy()
            P_internal = -sum(f_int_np1[d] for d in load_dofs)
            return True, u_it, v_np1, acc_np1, f_int_np1, states_comm, gp_sig_nom_last, gp_rank_last, float(P_internal)

        # optional line-search
        if line_search:
            u_base = u_it.copy()
            best = None
            alpha_ls = 1.0
            accepted = False
            for _ls in range(8):
                u_try = u_base.copy()
                u_try[free] += alpha_ls * du_f
                for dof, val in fixed.items():
                    u_try[dof] = val

                acc_try = a0 * (u_try - u_n) - a2 * v_n - a3 * acc_n
                v_try = v_n + dt * ((1.0 - gamma) * acc_n + gamma * acc_try)

                f_bulk_t, _, _, _ = bulk_internal_force(
                    nodes, elems, model.cdp, u_try, states_comm, thickness=model.b, precomp=precomp
                )
                f_rb_t, _ = rebar_contrib(
                    nodes, rebar_segs, u_try, model.steel_A_total,
                    model.steel_E, model.steel_fy, model.steel_fu, model.steel_Eh
                )
                f_int_t = f_bulk_t + f_rb_t
                r_try = (1.0 + alpha) * (f_int_t + (C @ v_try) + (Mdiag * acc_try)) - alpha * g_prev
                _, _, r_f_try, _ = apply_dirichlet(K_eff, r_try, fixed, u_try)
                n_try = float(np.linalg.norm(r_f_try))
                if best is None or n_try < best:
                    best = n_try
                if n_try < norm_r:
                    u_it = u_try
                    accepted = True
                    break
                alpha_ls *= 0.5
            if not accepted:
                u_it[free] += du_f
        else:
            u_it[free] += du_f

        for dof, val in fixed.items():
            u_it[dof] = val

    return False, u_n, v_n, acc_n, f_int_n, states_comm, gp_sig_nom_last, gp_rank_last, 0.0


def run_analysis(model: Model, nx: int, ny: int, nsteps: int, umax: float, line_search: bool = True,
                 max_subdiv: int = 10, min_dt: float = 1e-4):
    """
    Quasi-static solver using pseudo-time HHT-α (implicit) **with adaptive substepping**.

    - The prescribed displacement is ramped from 0 -> umax.
    - If a step fails to converge, it is subdivided (Δu, Δt halved) recursively until convergence
      or limits are reached.

    Args:
      max_subdiv: maximum bisections per nominal step
      min_dt: minimum allowed pseudo-time step

    Returns the same tuple as before, but `results` may contain more than `nsteps` entries.
    """
    nodes, elems = structured_quad_mesh(model.L, model.H, nx, ny)
    nnode = nodes.shape[0]
    ndof = 2 * nnode

    precomp = precompute_B_detJ(nodes, elems)

    states_comm = [[CDPState.zeros() for _ in range(4)] for _ in range(len(elems))]

    # supports
    left = int(np.argmin(nodes[:, 0]))
    right = int(np.argmax(nodes[:, 0]))
    fixed_base = {2*left: 0.0, 2*left+1: 0.0, 2*right+1: 0.0}

    # loading strip
    y_top = model.H
    top_nodes = np.where(np.isclose(nodes[:, 1], y_top))[0]
    dx = model.L / nx
    load_halfwidth = 2.0 * dx
    load_nodes = top_nodes[np.where(np.abs(nodes[top_nodes, 0] - model.L/2) <= load_halfwidth)[0]]
    if len(load_nodes) == 0:
        load_nodes = np.array([int(top_nodes[np.argmin(np.abs(nodes[top_nodes,0]-model.L/2))])], dtype=int)
    load_dofs = [2*n + 1 for n in load_nodes]

    rebar_segs = prepare_rebar_segments(nodes, model.cover)

    # base pseudo-time
    dt_base = float(model.dt) if model.dt is not None else float(model.total_time) / float(nsteps)
    alpha = float(model.hht_alpha)
    if alpha > 0.0 or alpha < -1.0/3.0:
        raise ValueError("HHT-α requires alpha in [-1/3, 0].")

    # mass (diagonal) + sparse
    Mdiag = assemble_lumped_mass(nodes, elems, rho=float(model.rho), thickness=float(model.b))
    M = sp.diags(Mdiag, 0, shape=(ndof, ndof), format="csr")

    # state variables
    u_n = np.zeros(ndof, dtype=float)
    v_n = np.zeros(ndof, dtype=float)
    acc_n = np.zeros(ndof, dtype=float)
    f_int_n = np.zeros(ndof, dtype=float)

    results = []
    gp_sig_nom_last = None
    gp_rank_last = None

    # current ramp value
    u_imp_n = 0.0
    t_n = 0.0
    accepted_count = 0

    for istep in range(1, nsteps + 1):
        u_imp_target = (istep / nsteps) * umax
        du_total = u_imp_target - u_imp_n
        dt_total = dt_base

        # stack of sub-intervals to process (t0,u0)->(t1,u1)
        stack = [(t_n, u_imp_n, dt_total, du_total, 0)]  # (t0, u0, dt, du, level)

        while stack:
            t0, u0, dt, du, level = stack.pop()
            if dt < min_dt:
                raise RuntimeError(f"Adaptive HHT: dt fell below min_dt ({min_dt}).")
            u1 = u0 + du
            ok, u_np1, v_np1, acc_np1, f_int_np1, states_np1, gp_sig_nom, gp_rank, P_int = _hht_try_step(
                model, nodes, elems, precomp, rebar_segs,
                Mdiag, M,
                u_n, v_n, acc_n, f_int_n, states_comm,
                fixed_base, load_dofs,
                u1, dt,
                line_search=line_search
            )

            if ok:
                # accept
                u_n = u_np1
                v_n = v_np1
                acc_n = acc_np1
                f_int_n = f_int_np1
                states_comm = states_np1
                gp_sig_nom_last = gp_sig_nom
                gp_rank_last = gp_rank
                t_n = t0 + dt
                u_imp_n = u1
                accepted_count += 1

                Mmid = (P_int * model.L / 4.0)
                kappa = curvature_midspan(nodes, elems, u_n, model.cdp, states_comm, x0=model.L/2, thickness=model.b)
                R = 1e20 if abs(kappa) < 1e-16 else 1.0 / kappa  # Use large value instead of inf

                results.append([accepted_count, t_n, u_imp_n, P_int, Mmid, kappa, R])

                print(f"step={accepted_count:04d} (nom={istep:03d}, lvl={level}) "
                      f"t={t_n:7.3f}s dt={dt:8.4g}s u={u_imp_n*1e3:7.3f}mm "
                      f"P={P_int/1e3:9.3f}kN M={Mmid/1e3:9.3f}kN·m kappa={kappa:10.3e}")
            else:
                # subdivide
                if level >= max_subdiv:
                    raise RuntimeError(f"Adaptive HHT: step could not converge after max_subdiv={max_subdiv} (u_target={u1}).")
                dt2 = 0.5 * dt
                du2 = 0.5 * du
                # process second half after first half => push second half first so first half runs next (stack LIFO)
                stack.append((t0 + dt2, u0 + du2, dt2, du2, level + 1))
                stack.append((t0, u0, dt2, du2, level + 1))

    return nodes, elems, u_n, results, states_comm, gp_sig_nom_last, gp_rank_last, rebar_segs






def curvature_midspan(nodes, elems, u, mat: CDPMaterial, states_comm, x0: float, thickness: float) -> float:
    # take elements whose centroid x is close to midspan; fit epsilon_xx(y) linear
    centroids = np.mean(nodes[elems], axis=1)
    idx = np.where(np.abs(centroids[:,0] - x0) < 0.5*(nodes[:,0].max()/len(np.unique(nodes[:,0])) + 1e-9))[0]
    if len(idx) == 0:
        idx = np.array([int(np.argmin(np.abs(centroids[:,0]-x0)))], dtype=int)

    ys = []
    exx = []
    gauss = [(-1/math.sqrt(3), -1/math.sqrt(3)),
             (+1/math.sqrt(3), -1/math.sqrt(3)),
             (+1/math.sqrt(3), +1/math.sqrt(3)),
             (-1/math.sqrt(3), +1/math.sqrt(3))]

    for e in idx:
        conn = elems[e]
        xe = nodes[conn,:]
        ue = np.zeros(8, dtype=float)
        for a in range(4):
            ue[2*a] = u[2*conn[a]]
            ue[2*a+1] = u[2*conn[a]+1]
        for igp, (xi,eta) in enumerate(gauss):
            B, _ = element_B_detJ(xi,eta,xe)
            eps = B @ ue
            ys.append(float(np.mean(xe[:,1])))
            exx.append(float(eps[0]))

    ys = np.array(ys); exx=np.array(exx)
    if len(ys) < 2:
        return 0.0
    # fit exx = a + b*y -> curvature kappa = -b
    A = np.vstack([np.ones_like(ys), ys]).T
    sol, *_ = np.linalg.lstsq(A, exx, rcond=None)
    b = float(sol[1])
    return -b


def element_center_fields(nodes, elems, u, mat: CDPMaterial, states_comm):
    # element center: average gauss fields
    nE = len(elems)
    sig_vm = np.zeros(nE, dtype=float)
    sig_tr = np.zeros(nE, dtype=float)
    dt = np.zeros(nE, dtype=float)
    dc = np.zeros(nE, dtype=float)
    rank = np.zeros(nE, dtype=float)
    for e in range(nE):
        s = 0.0; t=0.0; dtt=0.0; dcc=0.0; rr=0.0
        for gp in range(4):
            # reconstruct nominal stress by re-evaluating update at same strain (cheap)
            # (use committed state as "state0")
            pass
    # We instead approximate using committed damage only and last gp ranks (computed in run_analysis)
    # For plotting we compute dt/dc from committed states and use sigma estimates from elastic strains:
    C = mat.C()
    gauss = [(-1/math.sqrt(3), -1/math.sqrt(3)),
             (+1/math.sqrt(3), -1/math.sqrt(3)),
             (+1/math.sqrt(3), +1/math.sqrt(3)),
             (-1/math.sqrt(3), +1/math.sqrt(3))]
    for e, conn in enumerate(elems):
        xe = nodes[conn,:]
        ue = np.zeros(8, dtype=float)
        for a in range(4):
            ue[2*a] = u[2*conn[a]]
            ue[2*a+1] = u[2*conn[a]+1]
        vm=0.0; tr=0.0; dtt=0.0; dcc=0.0; rr=0.0
        for igp,(xi,eta) in enumerate(gauss):
            B,_ = element_B_detJ(xi,eta,xe)
            eps = B @ ue
            st0 = states_comm[e][igp]
            sig_eff = C @ (eps - st0.eps_p)
            sig_nom = apply_damage_unilateral(sig_eff, st0.dt, st0.dc)
            vm += von_mises_plane_stress(sig_nom)
            tr += tresca_plane_stress(sig_nom)
            dtt += st0.dt
            dcc += st0.dc
            rr += principal_max_2d(sig_eff)/max(1e-9, mat.f_t0)
        sig_vm[e] = vm/4.0
        sig_tr[e] = tr/4.0
        dt[e] = dtt/4.0
        dc[e] = dcc/4.0
        rank[e] = rr/4.0
    return sig_vm, sig_tr, dt, dc, rank
