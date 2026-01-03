"""Numba-friendly cohesive law kernels (Phase 2).

The existing cohesive implementation (``xfem_clean.cohesive_laws.cohesive_update``)
operates on Python objects (`CohesiveLaw`, `CohesiveState`). This file provides a
value-based equivalent suitable for Numba's ``nopython`` mode.

Design
------
* Inputs/outputs are scalars and small NumPy arrays.
* No Python objects, dicts, or dataclasses.
* Law parameters are packed once into a float array.

Supported laws:
* Bilinear (default)
* Reinhardt (Gutierrez Eq. 3.24)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from xfem_clean.cohesive_laws import CohesiveLaw
from xfem_clean.numba.utils import njit


def pack_cohesive_law_params(law: CohesiveLaw) -> np.ndarray:
    """Pack a :class:`~xfem_clean.cohesive_laws.CohesiveLaw` into a float array.

    The returned array is small and can be passed into Numba kernels.

    Layout (float64)
    ----------------
    p[0]  law_id   (0=bilinear, 1=reinhardt)
    p[1]  Kn
    p[2]  ft
    p[3]  delta0
    p[4]  deltaf
    p[5]  k_res
    p[6]  k_cap
    p[7]  c1
    p[8]  c2
    p[9]  wcrit
    """
    law_name = (law.law or "bilinear").lower()
    law_id = 1.0 if law_name.startswith("rein") else 0.0
    Kn = float(law.Kn)
    ft = float(law.ft)
    d0 = float(law.delta0)
    df = float(law.deltaf)
    k_res = float(law.kres_factor) * Kn
    k_cap = float(law.kcap_factor) * Kn
    c1 = float(getattr(law, "c1", 3.0))
    c2 = float(getattr(law, "c2", 6.93))
    wcrit = float(getattr(law, "wcrit", 0.0))
    return np.array((law_id, Kn, ft, d0, df, k_res, k_cap, c1, c2, wcrit), dtype=np.float64)


@njit(cache=True)
def _reinhardt_env_w(w: float, ft: float, c1: float, c2: float, wc: float) -> float:
    if w <= 0.0:
        return ft
    if wc <= 0.0:
        return 0.0
    if w >= wc:
        return 0.0
    x = w / wc
    # Gutierrez Eq (3.24) - normalized curve
    return ft * ((1.0 + (c1 * x) ** 3) * math.exp(-c2 * x) - x * (1.0 + c1 ** 3) * math.exp(-c2))


@njit(cache=True)
def cohesive_update_values_numba(
    delta: float,
    delta_max_old: float,
    damage_old: float,
    params: np.ndarray,
    visc_damp: float = 0.0,
) -> tuple[float, float, float, float]:
    """Value-based cohesive update.

    Returns
    -------
    T : float
        Traction (signed, Mode I scalar)
    k_alg : float
        Algorithmic tangent/secant stiffness used in modified Newton
    delta_max_new : float
    damage_new : float

    IMPORTANT: Unilateral opening behavior (P0.1 fix)
    - Only positive opening (δ > 0) contributes to damage
    - Compression (δ < 0) returns zero traction
    """

    law_id = int(params[0] + 0.5)
    Kn = float(params[1])
    ft = float(params[2])
    d0 = float(params[3])
    df = float(params[4])
    k_res = float(params[5])
    k_cap = float(params[6])
    c1 = float(params[7])
    c2 = float(params[8])
    wc = float(params[9])

    # P0.1 FIX: Unilateral opening - only positive opening contributes to damage
    if delta <= 0.0:
        # Crack is closed or in compression - no cohesive traction
        return 0.0, k_res, float(delta_max_old), float(damage_old)

    delta_abs = delta  # Since delta > 0 here
    dm_old = float(delta_max_old)
    dm = dm_old
    if delta_abs > dm:
        dm = delta_abs

    # Elastic stage
    if dm <= d0 + 1e-18:
        d_eq = 0.0
        if visc_damp > 0.0:
            d_new = (d_eq + visc_damp * float(damage_old)) / (1.0 + visc_damp)
            if d_new < float(damage_old):
                d_new = float(damage_old)
        else:
            d_new = float(damage_old) if float(damage_old) > d_eq else d_eq

        k_alg = Kn
        if k_alg < k_res:
            k_alg = k_res
        T = k_alg * delta
        return T, k_alg, dm, d_new

    # --- Softening envelope at max opening dm ---
    if law_id == 1:
        # Reinhardt envelope based on w = |δ| - δ0
        wmax = dm - d0
        if wmax < 0.0:
            wmax = 0.0
        T_env = _reinhardt_env_w(wmax, ft, c1, c2, wc)
        if T_env < 0.0:
            T_env = 0.0
    else:
        # Bilinear envelope
        if dm <= d0:
            T_env = Kn * dm
        elif dm >= df:
            T_env = 0.0
        else:
            denom = df - d0
            if denom <= 0.0:
                denom = 1e-30
            T_env = ft * (1.0 - (dm - d0) / denom)
            if T_env < 0.0:
                T_env = 0.0

    # Damage proxy
    denom_ft = ft if ft > 1e-30 else 1e-30
    d_eq = 1.0 - (T_env / denom_ft)
    if d_eq < 0.0:
        d_eq = 0.0
    if d_eq > 1.0:
        d_eq = 1.0

    if visc_damp > 0.0:
        d_new = (d_eq + visc_damp * float(damage_old)) / (1.0 + visc_damp)
        if d_new < float(damage_old):
            d_new = float(damage_old)
    else:
        d_new = d_eq if d_eq > float(damage_old) else float(damage_old)

    # Secant stiffness at max opening
    denom_dm = dm if dm > 1e-15 else 1e-15
    k_sec = T_env / denom_dm
    if k_sec > k_cap:
        k_sec = k_cap
    k_alg = k_sec
    if k_alg < k_res:
        k_alg = k_res

    # Traction at current delta
    if law_id == 1:
        # Unloading/reloading rule like cohesive_update
        if delta_abs < dm_old - 1e-14:
            T_mag = k_sec * delta_abs
        else:
            if delta_abs <= d0:
                T_mag = Kn * delta_abs
            else:
                T_mag = _reinhardt_env_w(delta_abs - d0, ft, c1, c2, wc)
                if T_mag < 0.0:
                    T_mag = 0.0
        if delta == 0.0:
            T = 0.0
        else:
            T = T_mag if delta > 0.0 else -T_mag
        return T, k_alg, dm, d_new

    # Bilinear: modified Newton uses k_alg and linear relation
    T = k_alg * delta
    return T, k_alg, dm, d_new


# ==============================================================================
# MIXED-MODE COHESIVE (THESIS PARITY)
# ==============================================================================

def pack_cohesive_mixed_law_params(law: CohesiveLaw) -> np.ndarray:
    """Pack a mixed-mode CohesiveLaw into a float array for Numba.

    Layout (float64)
    ----------------
    p[0]  Kn           - Normal stiffness [Pa/m]
    p[1]  Kt           - Tangential stiffness [Pa/m]
    p[2]  delta0_n     - Elastic limit (normal) [m]
    p[3]  deltaf_n     - Final opening (normal) [m]
    p[4]  beta         - Stiffness ratio Kt/Kn for effective separation
    p[5]  kp           - Compression penalty stiffness [Pa/m]
    p[6]  k_res        - Residual stiffness (normal) [Pa/m]
    p[7]  shear_model  - 0=constant, 1=wells
    p[8]  k_s0         - Wells: initial shear stiffness [Pa/m]
    p[9]  k_s1         - Wells: final shear stiffness [Pa/m]
    p[10] w1           - Wells: characteristic opening [m]
    p[11] use_cyclic   - 0=monotonic, 1=cyclic closure
    p[12] delta0_eff   - Effective elastic limit [m]
    p[13] deltaf_eff   - Effective final opening [m]
    """
    Kn = float(law.Kn)
    Kt = float(law.Kt) if hasattr(law, 'Kt') and law.Kt is not None else Kn

    delta0_n = float(law.delta0)
    deltaf_n = float(law.deltaf)

    # Compute beta for effective separation
    beta = Kt / max(1e-30, Kn) if Kt > 0 else 1.0

    # Compression penalty stiffness (typically large, e.g., 1000 * Kn)
    kp = float(getattr(law, 'k_penalty', 1000.0 * Kn))

    # Residual stiffness
    k_res = float(law.kres_factor) * Kn if hasattr(law, 'kres_factor') else 0.0

    # Shear model parameters
    shear_model = getattr(law, 'shear_model', 'constant')
    shear_model_id = 1.0 if shear_model == 'wells' else 0.0

    k_s0 = float(getattr(law, 'k_s0', Kt))
    k_s1 = float(getattr(law, 'k_s1', 0.01 * k_s0))
    w1 = float(getattr(law, 'w1', 1.0e-3))  # Default: 1mm

    use_cyclic = 1.0 if getattr(law, 'use_cyclic_closure', False) else 0.0

    # Effective limits (precomputed for efficiency)
    delta0_eff = math.sqrt(delta0_n**2 + beta * 0.0)  # Pure mode I at elastic limit
    deltaf_eff = math.sqrt(deltaf_n**2 + beta * 0.0)  # Pure mode I at final opening

    return np.array((
        Kn, Kt, delta0_n, deltaf_n, beta, kp, k_res,
        shear_model_id, k_s0, k_s1, w1, use_cyclic,
        delta0_eff, deltaf_eff
    ), dtype=np.float64)


@njit(cache=True)
def cohesive_update_mixed_values_numba(
    delta_n: float,
    delta_t: float,
    delta_max_old: float,  # Stores gmax (effective separation) or wmax (cyclic)
    damage_old: float,
    params: np.ndarray,
    visc_damp: float = 0.0,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Mixed-mode cohesive update (THESIS PARITY - Numba kernel).

    Implements:
    - Unilateral opening (δn_pos = max(δn, 0))
    - Compression penalty (if δn < 0)
    - Effective separation: δeff = sqrt(δn_pos² + β*δt²)
    - Damage evolution from gmax
    - Wells shear model: ks(w) = ks0 * exp(hs*w_eff)
    - Cyclic closure with w_max history

    Returns
    -------
    t_n : float
        Normal traction [Pa]
    t_t : float
        Tangential traction [Pa]
    dtn_ddn : float
        ∂t_n/∂δ_n [Pa/m]
    dtn_ddt : float
        ∂t_n/∂δ_t [Pa/m]
    dtt_ddn : float
        ∂t_t/∂δ_n [Pa/m] (Wells cross-coupling!)
    dtt_ddt : float
        ∂t_t/∂δ_t [Pa/m]
    delta_max_new : float
        Updated gmax or wmax
    damage_new : float
        Updated damage parameter
    """

    # Unpack parameters
    Kn = float(params[0])
    Kt = float(params[1])
    delta0_n = float(params[2])
    deltaf_n = float(params[3])
    beta = float(params[4])
    kp = float(params[5])
    k_res = float(params[6])
    shear_model_id = int(params[7] + 0.5)
    k_s0 = float(params[8])
    k_s1 = float(params[9])
    w1 = float(params[10])
    use_cyclic = int(params[11] + 0.5)
    delta0_eff = float(params[12])
    deltaf_eff = float(params[13])

    # Step 1: Unilateral opening
    delta_n_pos = max(0.0, delta_n)
    delta_t_val = delta_t

    # Step 2: Compression penalty (if δn < 0 and cyclic closure enabled)
    if delta_n < 0.0 and use_cyclic == 1:
        # Compression: penalty stiffness in normal direction
        t_n = kp * delta_n
        dtn_ddn = kp
        dtn_ddt = 0.0

        # Shear remains active (friction-like behavior)
        t_t = 0.0  # Or could use residual shear
        dtt_ddn = 0.0
        dtt_ddt = Kt

        # No damage accumulation in compression
        return t_n, t_t, dtn_ddn, dtn_ddt, dtt_ddn, dtt_ddt, delta_max_old, damage_old

    # Step 3: Effective separation
    delta_eff = math.sqrt(delta_n_pos**2 + beta * delta_t_val**2)

    # Step 4: Damage evolution from gmax
    g_old = float(delta_max_old)
    g_max = max(g_old, delta_eff)

    # Elastic regime
    if g_max <= delta0_eff + 1e-18:
        d = 0.0
        t_n = Kn * delta_n_pos
        t_t = Kt * delta_t_val

        # Elastic tangent
        dtn_ddn = Kn if delta_n > 0.0 else 0.0
        dtn_ddt = 0.0
        dtt_ddn = 0.0
        dtt_ddt = Kt

        # No damage update
        d_new = damage_old if damage_old > d else d
        return t_n, t_t, dtn_ddn, dtn_ddt, dtt_ddn, dtt_ddt, g_max, d_new

    # Softening regime: compute damage
    if g_max >= deltaf_eff:
        d = 1.0
    else:
        d = (g_max - delta0_eff) / max(1e-30, (deltaf_eff - delta0_eff))

    # Viscous damage regularization
    if visc_damp > 0.0:
        d_new = (d + visc_damp * damage_old) / (1.0 + visc_damp)
        if d_new < damage_old:
            d_new = damage_old
    else:
        d_new = d if d > damage_old else damage_old

    # Reduced stiffness
    k_alg_n = (1.0 - d_new) * Kn
    if k_alg_n < k_res:
        k_alg_n = k_res

    # Normal traction
    t_n = k_alg_n * delta_n_pos

    # Step 5-6: Shear model (Wells or constant)
    if shear_model_id == 1:
        # Wells model: ks(W) = ks0 * exp(hs * W)
        h_s = math.log(k_s1 / max(1e-30, k_s0)) / max(1e-30, w1)

        # Determine effective opening for shear degradation
        if use_cyclic == 1:
            # Cyclic: use maximum opening reached (stored in delta_max)
            W = g_old  # w_max stored in delta_max field
        else:
            # Monotonic: use current opening
            W = delta_n_pos

        # Shear stiffness as function of opening
        k_s_w = k_s0 * math.exp(h_s * W)
        k_alg_t = max(k_s_w, k_res)

        # Shear traction
        t_t = k_alg_t * delta_t_val

        # Tangent matrix with Wells cross-coupling
        if delta_n > 0.0:
            if use_cyclic == 1 and delta_n_pos < g_old - 1e-14:
                # Unloading: W = wmax is constant, no cross-coupling
                dtt_ddn = 0.0
            else:
                # Loading OR monotonic: W = current w, normal cross-coupling
                dtt_ddn = h_s * k_alg_t * delta_t_val
        else:
            dtt_ddn = 0.0

        dtt_ddt = k_alg_t

    else:
        # Constant shear model (damage-based)
        k_alg_t = (1.0 - d_new) * Kt
        if k_alg_t < k_res:
            k_alg_t = k_res

        t_t = k_alg_t * delta_t_val

        # Damage-based tangent (no Wells cross-coupling)
        # Compute damage derivative if in softening
        if delta0_eff < g_max < deltaf_eff:
            dd_dg = 1.0 / max(1e-30, (deltaf_eff - delta0_eff))
        else:
            dd_dg = 0.0

        # Effective separation derivatives
        if delta_eff > 1e-18:
            dg_ddn = delta_n_pos / delta_eff if delta_n > 0.0 else 0.0
            dg_ddt = beta * delta_t_val / delta_eff
        else:
            dg_ddn = 0.0
            dg_ddt = 0.0

        # Stiffness derivatives
        dk_alg_t_dg = -dd_dg * Kt

        # Tangent components
        dtt_ddn = dk_alg_t_dg * dg_ddn * delta_t_val
        dtt_ddt = dk_alg_t_dg * dg_ddt * delta_t_val + k_alg_t

    # Normal direction tangent (same for both models)
    if delta0_eff < g_max < deltaf_eff:
        dd_dg = 1.0 / max(1e-30, (deltaf_eff - delta0_eff))
    else:
        dd_dg = 0.0

    if delta_eff > 1e-18:
        dg_ddn = delta_n_pos / delta_eff if delta_n > 0.0 else 0.0
        dg_ddt = beta * delta_t_val / delta_eff
    else:
        dg_ddn = 0.0
        dg_ddt = 0.0

    dk_alg_n_dg = -dd_dg * Kn

    dtn_ddn = dk_alg_n_dg * dg_ddn * delta_n_pos + k_alg_n if delta_n > 0.0 else 0.0
    dtn_ddt = dk_alg_n_dg * dg_ddt * delta_n_pos

    return t_n, t_t, dtn_ddn, dtn_ddt, dtt_ddn, dtt_ddt, g_max, d_new
