"""Cohesive laws used by the XFEM prototypes (stable implementation)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# -----------------------------
# Cohesive bilinear law (Mode I)
# -----------------------------
@dataclass
class CohesiveLaw:
    Kn: float   # penalty stiffness per unit area [Pa/m]
    ft: float   # tensile strength [Pa]
    Gf: float   # fracture energy [J/m^2] = [N/m]
    law: str = "bilinear"  # "bilinear" (default) or "reinhardt"
    kres_factor: float = 1e-4  # residual stiffness fraction of Kn
    kcap_factor: float = 1.0   # cap factor for secant stiffness (<= Kn*kcap_factor)

    # Reinhardt (Gutiérrez Eq. (3.24) + (3.29)) parameters (defaults from thesis)
    c1: float = 3.0
    c2: float = 6.93
    # Critical crack opening w_c (softening branch length). If <=0 and law=reinhardt, computed from Gf/ft.
    wcrit: float = 0.0

    # P2: Mixed-mode parameters (Mode I + Mode II)
    mode: str = "I"  # "I" (Mode I only, default) or "mixed" (Mode I + Mode II)
    # Shear strength (for mixed mode)
    tau_max: float = 0.0  # If 0, defaults to ft
    # Shear penalty stiffness (for mixed mode)
    Kt: float = 0.0  # If 0, defaults to Kn
    # Mode II fracture energy
    Gf_II: float = 0.0  # If 0, defaults to Gf (Mode I)

    # PART C: Wells-type shear stiffness degradation (exponential with opening)
    shear_model: str = "constant"  # "constant" (default) or "wells"
    k_s0: float = 0.0  # Initial shear stiffness [Pa/m] (wells model)
    k_s1: float = 0.0  # Final shear stiffness [Pa/m] (wells model, degraded)

    def __post_init__(self) -> None:
        if self.law.lower().startswith("rein") and self.wcrit <= 0.0:
            # Choose w_c such that:  ∫_0^{w_c} t(w) dw = Gf, with t(w)=ft*f(w/w_c)
            # => w_c = Gf / (ft * ∫_0^1 f(x) dx)
            I = self._reinhardt_I(self.c1, self.c2)
            self.wcrit = float(self.Gf / (max(1e-30, self.ft) * max(1e-30, I)))

        # P2: Set defaults for mixed-mode parameters
        if self.mode.lower() == "mixed":
            if self.tau_max <= 0.0:
                self.tau_max = self.ft  # Default: shear strength = tensile strength
            if self.Kt <= 0.0:
                self.Kt = self.Kn  # Default: shear stiffness = normal stiffness
            if self.Gf_II <= 0.0:
                self.Gf_II = self.Gf  # Default: Mode II fracture energy = Mode I

            # PART C: Set defaults for Wells-type shear model
            if self.shear_model.lower() == "wells":
                if self.k_s0 <= 0.0:
                    self.k_s0 = self.Kt  # Default: initial shear stiffness = Kt
                if self.k_s1 <= 0.0:
                    self.k_s1 = 0.01 * self.k_s0  # Default: degraded to 1% of initial

    @staticmethod
    def _reinhardt_I(c1: float, c2: float, n: int = 4096) -> float:
        # Dimensionless integral I = ∫_0^1 f(x) dx for the Reinhardt curve (numerical quadrature).
        xs = np.linspace(0.0, 1.0, int(n))
        fx = (1.0 + (c1 * xs) ** 3) * np.exp(-c2 * xs) - xs * (1.0 + c1 ** 3) * math.exp(-c2)
        return float(np.trapz(fx, xs))

    @property
    def delta0(self) -> float:
        # Opening at peak tensile strength (end of initial elastic stage)
        return self.ft / max(1e-30, self.Kn)

    @property
    def deltaf(self) -> float:
        # Final opening where traction vanishes.
        if self.law.lower().startswith("rein"):
            return float(self.delta0 + max(0.0, self.wcrit))
        # bilinear: area = 1/2 ft * deltaf = Gf  => deltaf = 2 Gf / ft
        return 2.0 * self.Gf / max(1e-30, self.ft)



@dataclass
class CohesiveState:
    # History variables (committed at the end of an accepted substep).
    # NOTE: during Newton iterations we must NOT overwrite these in-place,
    # otherwise the irreversibility (max) will be spuriously accumulated and
    # the solver may stagnate.
    delta_max: float = 0.0
    damage: float = 0.0

    def copy(self) -> "CohesiveState":
        return CohesiveState(delta_max=float(self.delta_max), damage=float(self.damage))


def cohesive_update(law: CohesiveLaw, delta: float, st: CohesiveState, visc_damp: float = 0.0) -> Tuple[float, float, CohesiveState]:
    """Update cohesive traction and tangent (modified Newton).

    - Initial elastic stage (0 <= |δ| <= δ0): t = Kn * δ
    - Softening stage:
        * bilinear: linear drop to 0 at δf
        * reinhardt: Reinhardt/Gutiérrez curve in terms of crack opening w = |δ| - δ0
    - Unloading/reloading: straight line through the origin with secant stiffness at the max opening.

    IMPORTANT: Unilateral opening behavior (P0.1 fix)
    - Only positive opening (δ > 0) contributes to damage
    - Compression (δ < 0) returns zero traction (crack closes freely, no cohesive resistance)
    """
    Kn = float(law.Kn)
    ft = float(law.ft)
    d0 = float(law.delta0)
    k_res = float(law.kres_factor) * Kn
    k_cap = float(law.kcap_factor) * Kn

    delta = float(delta)

    # P0.1 FIX: Unilateral opening - only positive opening contributes to damage
    # Compression (delta < 0) should return zero traction (crack closes freely)
    if delta <= 0.0:
        # Crack is closed or in compression - no cohesive traction
        # Use residual stiffness to avoid singularity, but no damage accumulation
        st2 = CohesiveState(delta_max=float(st.delta_max), damage=float(st.damage))
        return 0.0, k_res, st2

    delta_abs = delta  # Since delta > 0 here, delta_abs = delta

    # History (irreversible max opening - only from positive opening)
    dm_old = float(st.delta_max)
    dm = max(dm_old, delta_abs)

    # --- Elastic stage (common to all laws) ---
    if dm <= d0 + 1e-18:
        # No damage yet
        d_eq = 0.0
        if visc_damp > 0.0:
            d_new = (d_eq + float(visc_damp) * float(st.damage)) / (1.0 + float(visc_damp))
            d_new = max(float(st.damage), d_new)
        else:
            d_new = max(float(st.damage), d_eq)

        k_alg = max(Kn, k_res)
        T = k_alg * delta
        st2 = CohesiveState(delta_max=dm, damage=d_new)
        return T, k_alg, st2

    # --- Softening envelope ---
    def bilinear_env(d: float) -> float:
        df = float(law.deltaf)
        if d <= d0:
            return Kn * d
        if d >= df:
            return 0.0
        return ft * (1.0 - (d - d0) / max(1e-30, (df - d0)))

    def reinhardt_env_w(w: float) -> float:
        w = float(w)
        if w <= 0.0:
            return ft
        wc = float(max(1e-30, law.wcrit))
        if w >= wc:
            return 0.0
        x = w / wc
        c1 = float(law.c1)
        c2 = float(law.c2)
        return ft * ((1.0 + (c1 * x) ** 3) * math.exp(-c2 * x) - x * (1.0 + c1 ** 3) * math.exp(-c2))

    if law.law.lower().startswith("rein"):
        # Work in crack-opening measure w = |δ| - δ0
        wmax = max(0.0, dm - d0)
        T_env = float(max(0.0, reinhardt_env_w(wmax)))
        # damage proxy
        d_eq = 1.0 - (T_env / max(1e-30, ft))
        d_eq = min(1.0, max(0.0, d_eq))
        if visc_damp > 0.0:
            d_new = (d_eq + float(visc_damp) * float(st.damage)) / (1.0 + float(visc_damp))
            d_new = max(float(st.damage), d_new)
        else:
            d_new = max(float(st.damage), d_eq)

        # Secant stiffness at max opening (used for unloading/reloading)
        k_sec = T_env / max(1e-15, dm)
        k_sec = min(k_sec, k_cap)
        k_alg = max(k_sec, k_res)

        # Current traction magnitude
        if delta_abs < dm_old - 1e-14:
            # unloading/reloading: linear through origin
            T_mag = k_sec * delta_abs
        else:
            # loading: elastic up to d0, then envelope
            if delta_abs <= d0:
                T_mag = Kn * delta_abs
            else:
                T_mag = float(max(0.0, reinhardt_env_w(delta_abs - d0)))

        if delta == 0.0:
            T = 0.0
        else:
            T = (1.0 if delta > 0.0 else -1.0) * T_mag

        st2 = CohesiveState(delta_max=dm, damage=d_new)
        return T, k_alg, st2

    # --- bilinear law ---
    T_env = float(max(0.0, bilinear_env(dm)))
    d_eq = 1.0 - (T_env / max(1e-30, ft))
    d_eq = min(1.0, max(0.0, d_eq))
    if visc_damp > 0.0:
        d_new = (d_eq + float(visc_damp) * float(st.damage)) / (1.0 + float(visc_damp))
        d_new = max(float(st.damage), d_new)
    else:
        d_new = max(float(st.damage), d_eq)

    k_sec = T_env / max(1e-15, dm)
    k_sec = min(k_sec, k_cap)
    k_alg = max(k_sec, k_res)
    T = k_alg * delta

    st2 = CohesiveState(delta_max=dm, damage=d_new)
    return T, k_alg, st2


# -----------------------------
# Mixed-mode cohesive law (Mode I + Mode II)
# -----------------------------

def cohesive_update_mixed(
    law: CohesiveLaw,
    delta_n: float,
    delta_t: float,
    st: CohesiveState,
    visc_damp: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, CohesiveState]:
    """Update mixed-mode cohesive traction and tangent (P3 implementation).

    Mixed-mode formulation per model spec:
    - Effective separation: δ_eff = sqrt((δ_n_pos)² + β(δ_t)²)
    - Damage: d(g_max) where g_max = max(previous, δ_eff)
    - Tractions: t_n = (1-d) K_n δ_n_pos, t_t = (1-d) K_t δ_t
    - Tangent: 2x2 matrix with cross-coupling

    Parameters
    ----------
    law : CohesiveLaw
        Must have mode="mixed" and mixed-mode parameters set
    delta_n : float
        Normal opening (positive = opening) [m]
    delta_t : float
        Tangential slip (signed) [m]
    st : CohesiveState
        Committed state (delta_max stores g_max)
    visc_damp : float
        Viscous damping parameter

    Returns
    -------
    t : np.ndarray
        Traction vector [t_n, t_t] [Pa]
    K : np.ndarray
        2x2 tangent matrix [[dtn_ddn, dtn_ddt], [dtt_ddn, dtt_ddt]] [Pa/m]
    st2 : CohesiveState
        Updated trial state

    Notes
    -----
    Unilateral opening: only δ_n_pos = max(δ_n, 0) contributes to normal traction.
    Compression (δ_n < 0) gives zero normal traction but tangential may be active.
    """
    # Check mode
    if law.mode.lower() != "mixed":
        raise ValueError(f"cohesive_update_mixed requires mode='mixed', got mode='{law.mode}'")

    # Parameters
    Kn = float(law.Kn)
    Kt = float(law.Kt) if law.Kt > 0 else Kn
    ft = float(law.ft)
    tau_max = float(law.tau_max) if law.tau_max > 0 else ft
    Gf_I = float(law.Gf)
    Gf_II = float(law.Gf_II) if law.Gf_II > 0 else Gf_I
    k_res = float(law.kres_factor) * Kn

    # Unilateral opening
    delta_n_pos = max(0.0, float(delta_n))
    delta_t_val = float(delta_t)

    # Effective separation parameter beta (ratio of stiffnesses)
    beta = Kt / max(1e-30, Kn)

    # Critical separations
    delta0_n = ft / max(1e-30, Kn)
    deltaf_n = 2.0 * Gf_I / max(1e-30, ft)
    delta0_t = tau_max / max(1e-30, Kt)
    deltaf_t = 2.0 * Gf_II / max(1e-30, tau_max)

    # Effective critical separations (simplified: use normal mode)
    delta0_eff = delta0_n
    deltaf_eff = deltaf_n

    # Compute effective separation
    delta_eff = math.sqrt(delta_n_pos**2 + beta * delta_t_val**2)

    # History: g_max = max(previous, current)
    g_old = float(st.delta_max)  # Note: reusing delta_max field for g_max
    g_max = max(g_old, delta_eff)

    # --- Elastic regime (no damage) ---
    if g_max <= delta0_eff + 1e-18:
        # No damage
        d = 0.0
        t_n = Kn * delta_n_pos
        t_t = Kt * delta_t_val

        # Tangent (elastic, with unilateral for normal)
        if delta_n > 0:
            dtn_ddn = Kn
        else:
            dtn_ddn = 0.0  # Compression: no normal traction
        dtn_ddt = 0.0
        dtt_ddn = 0.0
        dtt_ddt = Kt

        t = np.array([t_n, t_t], dtype=float)
        K_mat = np.array([[dtn_ddn, dtn_ddt], [dtt_ddn, dtt_ddt]], dtype=float)
        st2 = CohesiveState(delta_max=g_max, damage=d)
        return t, K_mat, st2

    # --- Softening regime ---
    # Bilinear damage evolution
    if g_max >= deltaf_eff:
        d = 1.0  # Fully damaged
    else:
        d = (g_max - delta0_eff) / max(1e-30, (deltaf_eff - delta0_eff))
        d = min(1.0, max(0.0, d))

    # Envelope tractions at g_max (using damage parameter)
    T_env_n = ft * (1.0 - d)
    T_env_t = tau_max * (1.0 - d)

    # Secant stiffnesses (for unloading/reloading behavior like Mode I)
    k_sec_n = T_env_n / max(1e-15, g_max)
    k_sec_t = T_env_t / max(1e-15, g_max)

    # Cap at initial stiffness
    k_sec_n = min(k_sec_n, Kn)
    k_sec_t = min(k_sec_t, Kt)

    # Apply residual stiffness floor
    k_alg_n = max(k_sec_n, k_res)
    k_alg_t_base = max(k_sec_t, k_res * Kt / max(1e-30, Kn))  # Base secant stiffness

    # =========================================================================
    # PART C: Wells-type shear stiffness degradation (exponential with opening)
    # =========================================================================
    if law.shear_model.lower() == "wells":
        # Wells model: k_s(w) = k_s0 * exp(h_s * w)
        # where h_s = ln(k_s1 / k_s0) is the decay parameter
        k_s0 = float(law.k_s0) if law.k_s0 > 0 else Kt
        k_s1 = float(law.k_s1) if law.k_s1 > 0 else 0.01 * k_s0

        # Decay parameter (negative for degradation)
        h_s = math.log(k_s1 / max(1e-30, k_s0))

        # Shear stiffness as function of opening
        # Use delta_n_pos (positive opening only) for degradation
        k_s_w = k_s0 * math.exp(h_s * delta_n_pos)

        # Shear traction (linear in slip, modulated by opening-dependent stiffness)
        k_alg_t = max(k_s_w, k_res)  # Apply residual floor
        t_n = k_alg_n * delta_n_pos
        t_t = k_alg_t * delta_t_val

        # Store h_s for tangent calculation
        h_s_stored = h_s
    else:
        # Standard constant shear stiffness
        k_alg_t = k_alg_t_base
        t_n = k_alg_n * delta_n_pos
        t_t = k_alg_t * delta_t_val
        h_s_stored = 0.0  # No opening-dependent degradation

    # --- Tangent matrix (consistent with secant stiffness formulation) ---
    # PART C: Enhanced for Wells-type shear model
    # For Wells: t_t = k_s(w) * s where k_s(w) = k_s0 * exp(h_s * w)
    # => dt_t/dw = (dk_s/dw) * s = h_s * k_s(w) * s  (cross-coupling!)
    # => dt_t/ds = k_s(w)

    if law.shear_model.lower() == "wells":
        # Wells model: direct derivatives
        # Normal traction: t_n = k_alg_n * delta_n_pos (unchanged from standard)
        # Shear traction: t_t = k_s(w) * delta_t where k_s(w) = k_s0 * exp(h_s * w)

        if delta_n > 0:
            # dt_n/dw: same as standard (damage-based)
            # dd/dg = 1 / (deltaf - delta0) for delta0 < g < deltaf
            if delta0_eff < g_max < deltaf_eff:
                dd_dg = 1.0 / max(1e-30, (deltaf_eff - delta0_eff))
            else:
                dd_dg = 0.0

            # ∂g/∂δ_n
            if delta_eff > 1e-18:
                dg_ddn = delta_n_pos / delta_eff
            else:
                dg_ddn = 0.0

            # ∂k_alg_n/∂g
            if g_max > 1e-18:
                dk_alg_n_dg = -ft / g_max * dd_dg - k_alg_n / g_max
            else:
                dk_alg_n_dg = 0.0

            dtn_ddn = dk_alg_n_dg * dg_ddn * delta_n_pos + k_alg_n
            dtn_ddt = 0.0  # No coupling from slip to normal (wells model)
        else:
            # Compression: no normal traction or tangent
            dtn_ddn = 0.0
            dtn_ddt = 0.0

        # dt_t/dw = h_s * k_s(w) * delta_t  (Wells cross-coupling)
        # dt_t/ds = k_s(w)
        if delta_n > 0:
            dtt_ddn = h_s_stored * k_alg_t * delta_t_val  # Cross-coupling term (PART C key feature!)
        else:
            dtt_ddn = 0.0  # No cross-coupling in compression

        dtt_ddt = k_alg_t  # Linear in slip

    else:
        # Standard formulation (damage-based coupling through effective separation)
        # dd/dg = 1 / (deltaf - delta0) for delta0 < g < deltaf
        if delta0_eff < g_max < deltaf_eff:
            dd_dg = 1.0 / max(1e-30, (deltaf_eff - delta0_eff))
        else:
            dd_dg = 0.0

        # ∂g/∂δ_n and ∂g/∂δ_t
        if delta_eff > 1e-18:
            if delta_n > 0:
                dg_ddn = delta_n_pos / delta_eff
            else:
                dg_ddn = 0.0  # Compression: g doesn't increase with negative δ_n
            dg_ddt = beta * delta_t_val / delta_eff
        else:
            dg_ddn = 0.0
            dg_ddt = 0.0

        # ∂k_alg_n/∂g and ∂k_alg_t/∂g
        # k_alg_n = ft * (1-d) / g => dk_alg_n/dg = -ft/g * dd/dg - k_alg_n/g
        # k_alg_t = tau_max * (1-d) / g => dk_alg_t/dg = -tau_max/g * dd/dg - k_alg_t/g
        if g_max > 1e-18:
            dk_alg_n_dg = -ft / g_max * dd_dg - k_alg_n / g_max
            dk_alg_t_dg = -tau_max / g_max * dd_dg - k_alg_t / g_max
        else:
            dk_alg_n_dg = 0.0
            dk_alg_t_dg = 0.0

        # Tangent components (with unilateral)
        if delta_n > 0:
            # dtn/ddn = dk_alg_n/dg * dg/ddn * delta_n_pos + k_alg_n
            dtn_ddn = dk_alg_n_dg * dg_ddn * delta_n_pos + k_alg_n
            # dtn/ddt = dk_alg_n/dg * dg/ddt * delta_n_pos
            dtn_ddt = dk_alg_n_dg * dg_ddt * delta_n_pos
        else:
            # Compression: no normal traction or tangent
            dtn_ddn = 0.0
            dtn_ddt = 0.0

        # dtt/ddn = dk_alg_t/dg * dg/ddn * delta_t
        dtt_ddn = dk_alg_t_dg * dg_ddn * delta_t_val
        # dtt/ddt = dk_alg_t/dg * dg/ddt * delta_t + k_alg_t
        dtt_ddt = dk_alg_t_dg * dg_ddt * delta_t_val + k_alg_t

    t = np.array([t_n, t_t], dtype=float)
    K_mat = np.array([[dtn_ddn, dtn_ddt], [dtt_ddn, dtt_ddt]], dtype=float)

    st2 = CohesiveState(delta_max=g_max, damage=d)
    return t, K_mat, st2


# Keep original Mode I function for backward compatibility
def cohesive_update_mode_I(law: CohesiveLaw, delta: float, st: CohesiveState, visc_damp: float = 0.0) -> Tuple[float, float, CohesiveState]:
    """Alias for cohesive_update (Mode I only) for clarity."""
    return cohesive_update(law, delta, st, visc_damp)


def cohesive_fracture_energy(law: CohesiveLaw, delta_max: float, n_quad: int = 256) -> float:
    """Return dissipated fracture energy per unit area [J/m^2] at a cohesive point.

    The cohesive state is tracked by the irreversible maximum opening ``delta_max``.
    This function integrates the *envelope* traction–separation curve from 0 to
    ``delta_max`` (monotonic, path-independent under the usual unloading rule).

    Notes
    -----
    For the Reinhardt law the softening branch integral is evaluated numerically.
    """
    dm = abs(float(delta_max))
    if dm <= 0.0:
        return 0.0

    Kn = float(law.Kn)
    ft = float(law.ft)
    d0 = float(law.delta0)
    df = float(law.deltaf)

    # Elastic contribution up to min(dm, d0)
    de = min(dm, d0)
    G_el = 0.5 * Kn * de * de

    if dm <= d0 + 1e-18:
        return float(G_el)

    # Softening contribution
    if law.law.lower().startswith("rein"):
        # Work in w = delta - delta0
        wc = float(max(1e-30, law.wcrit))
        wmax = max(0.0, dm - d0)
        wmax = min(wmax, wc)

        if wmax <= 0.0:
            return float(G_el)

        # Reinhardt envelope traction in terms of w
        c1 = float(law.c1)
        c2 = float(law.c2)

        xs = np.linspace(0.0, wmax / wc, int(max(16, n_quad)))
        fx = (1.0 + (c1 * xs) ** 3) * np.exp(-c2 * xs) - xs * (1.0 + c1**3) * math.exp(-c2)
        t = ft * fx
        G_soft = float(np.trapz(t, xs)) * wc
        return float(G_el + G_soft)

    # Bilinear: closed form
    if dm >= df - 1e-18:
        return float(law.Gf)

    # Integral from d0 to dm of ft*(1 - (d-d0)/(df-d0)) dd
    a = dm - d0
    L = max(1e-30, (df - d0))
    G_soft = ft * a - 0.5 * ft * (a * a) / L
    return float(G_el + G_soft)

# -----------------------------
