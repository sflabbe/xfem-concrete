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
    """Pack a :class:`~xfem_clean.cohesive_laws.CohesiveLaw` into a unified float array.

    The returned array supports both Mode I and mixed-mode cohesive laws.

    Unified Layout (float64, 21 elements)
    --------------------------------------
    p[0]  law_id          (0=bilinear, 1=reinhardt)
    p[1]  mode_id         (0=Mode I, 1=mixed)
    p[2]  Kn              [Pa/m]
    p[3]  ft              [Pa]
    p[4]  delta0          [m]
    p[5]  deltaf          [m]
    p[6]  k_res           [Pa/m] (computed as kres_factor * Kn)
    p[7]  k_cap           [Pa/m] (computed as kcap_factor * Kn)
    p[8]  c1              (Reinhardt parameter)
    p[9]  c2              (Reinhardt parameter)
    p[10] wcrit           [m] (Reinhardt critical opening)
    p[11] Kt              [Pa/m] (tangential stiffness for mixed-mode)
    p[12] tau_max         [Pa] (shear strength for mixed-mode)
    p[13] Gf_II           [J/m²] (Mode II fracture energy)
    p[14] kp              [Pa/m] (compression penalty stiffness)
    p[15] shear_model_id  (0=constant, 1=wells)
    p[16] k_s0            [Pa/m] (Wells: initial shear stiffness)
    p[17] k_s1            [Pa/m] (Wells: final shear stiffness)
    p[18] w1              [m] (Wells: characteristic opening)
    p[19] hs              [1/m] (Wells: decay parameter = ln(k_s1/k_s0)/w1)
    p[20] use_cyclic_closure (0=no, 1=yes)

    Notes
    -----
    - Mode I kernels ignore mixed-mode parameters (p[11:21])
    - Mixed-mode kernels use all parameters
    - Backward compatible: existing Mode I code works unchanged
    """
    # Basic parameters (Mode I)
    law_name = (law.law or "bilinear").lower()
    law_id = 1.0 if law_name.startswith("rein") else 0.0
    mode_id = 1.0 if (hasattr(law, 'mode') and law.mode.lower() == "mixed") else 0.0

    Kn = float(law.Kn)
    ft = float(law.ft)
    d0 = float(law.delta0)
    df = float(law.deltaf)
    k_res = float(law.kres_factor) * Kn
    k_cap = float(law.kcap_factor) * Kn
    c1 = float(getattr(law, "c1", 3.0))
    c2 = float(getattr(law, "c2", 6.93))
    wcrit = float(getattr(law, "wcrit", 0.0))

    # Mixed-mode parameters (defaults for Mode I)
    if mode_id > 0.5:  # Mixed mode
        Kt = float(law.Kt) if hasattr(law, 'Kt') and law.Kt > 0 else Kn
        tau_max = float(law.tau_max) if hasattr(law, 'tau_max') and law.tau_max > 0 else ft
        Gf_II = float(law.Gf_II) if hasattr(law, 'Gf_II') and law.Gf_II > 0 else float(law.Gf)
        kp = float(law.kp) if hasattr(law, 'kp') and law.kp > 0 else Kn

        # Wells shear model parameters
        shear_model = getattr(law, 'shear_model', 'constant')
        shear_model_id = 1.0 if shear_model.lower() == 'wells' else 0.0

        if shear_model_id > 0.5:  # Wells model
            k_s0 = float(law.k_s0) if hasattr(law, 'k_s0') and law.k_s0 > 0 else Kt
            k_s1 = float(law.k_s1) if hasattr(law, 'k_s1') and law.k_s1 > 0 else 0.01 * k_s0
            w1 = float(law.w1) if hasattr(law, 'w1') and law.w1 > 0 else 1.0e-3
            # Compute hs = ln(k_s1/k_s0)/w1 with guard against division
            hs = math.log(max(1e-30, k_s1) / max(1e-30, k_s0)) / max(1e-30, w1)
        else:
            k_s0 = Kt
            k_s1 = 0.01 * Kt
            w1 = 1.0e-3
            hs = 0.0

        use_cyclic_closure = 1.0 if getattr(law, 'use_cyclic_closure', False) else 0.0
    else:
        # Mode I: set mixed-mode params to zero
        Kt = 0.0
        tau_max = 0.0
        Gf_II = 0.0
        kp = Kn
        shear_model_id = 0.0
        k_s0 = 0.0
        k_s1 = 0.0
        w1 = 1.0e-3
        hs = 0.0
        use_cyclic_closure = 0.0

    return np.array((
        law_id, mode_id, Kn, ft, d0, df, k_res, k_cap, c1, c2, wcrit,
        Kt, tau_max, Gf_II, kp, shear_model_id, k_s0, k_s1, w1, hs, use_cyclic_closure
    ), dtype=np.float64)


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
    """Value-based cohesive update (Mode I).

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

    Notes
    -----
    Uses unified param layout (21 elements) but ignores mixed-mode params (p[11:21]).
    """

    law_id = int(params[0] + 0.5)
    # mode_id = params[1]  # Not used in Mode I kernel
    Kn = float(params[2])
    ft = float(params[3])
    d0 = float(params[4])
    df = float(params[5])
    k_res = float(params[6])
    k_cap = float(params[7])
    c1 = float(params[8])
    c2 = float(params[9])
    wc = float(params[10])

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
# Note: pack_cohesive_law_params() now handles both Mode I and mixed-mode
# in a unified layout. The old pack_cohesive_mixed_law_params() is removed.


@njit(cache=True)
def cohesive_update_mixed_values_numba(
    delta_n: float,
    delta_t: float,
    delta_max_old: float,  # Stores gmax (effective separation)
    damage_old: float,
    params: np.ndarray,
    visc_damp: float = 0.0,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Mixed-mode cohesive update (THESIS PARITY - Numba kernel).

    Implements:
    - Unilateral opening (δn_pos = max(δn, 0))
    - Compression penalty (if δn < 0 and cyclic closure enabled)
    - Effective separation: δeff = sqrt(δn_pos² + β*δt²)
    - Damage evolution from gmax
    - Wells shear model: ks(w) = ks0 * exp(hs*w_eff)
    - Cyclic closure with w_max history

    Parameters
    ----------
    delta_n : float
        Normal opening (positive = opening) [m]
    delta_t : float
        Tangential slip (signed) [m]
    delta_max_old : float
        Committed maximum effective separation (g_max) [m]
    damage_old : float
        Committed damage parameter
    params : np.ndarray
        Unified cohesive law parameters (21 elements, see pack_cohesive_law_params)
    visc_damp : float
        Viscous damping parameter (default 0.0)

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
        Updated gmax
    damage_new : float
        Updated damage parameter

    Notes
    -----
    Uses unified param layout (21 elements):
    p[0]=law_id, p[1]=mode_id, p[2]=Kn, p[3]=ft, p[4]=delta0, p[5]=deltaf,
    p[6]=k_res, p[7]=k_cap, p[8]=c1, p[9]=c2, p[10]=wcrit,
    p[11]=Kt, p[12]=tau_max, p[13]=Gf_II, p[14]=kp, p[15]=shear_model_id,
    p[16]=k_s0, p[17]=k_s1, p[18]=w1, p[19]=hs, p[20]=use_cyclic_closure
    """

    # Unpack parameters from unified layout
    # law_id = params[0]  # Not used in mixed-mode (bilinear assumed for now)
    # mode_id = params[1]  # Not used (caller ensures mode=mixed)
    Kn = float(params[2])
    ft = float(params[3])
    delta0_n = float(params[4])
    deltaf_n = float(params[5])
    k_res = float(params[6])
    # k_cap = params[7]  # Not used in this kernel
    # c1, c2, wcrit = params[8:11]  # Reinhardt params, not used for mixed-mode
    Kt = float(params[11])
    tau_max = float(params[12])
    Gf_II = float(params[13])
    kp = float(params[14])
    shear_model_id = int(params[15] + 0.5)
    k_s0 = float(params[16])
    k_s1 = float(params[17])
    w1 = float(params[18])
    hs = float(params[19])
    use_cyclic = int(params[20] + 0.5)

    # Compute derived parameters
    beta = Kt / max(1e-30, Kn)  # Stiffness ratio for effective separation
    delta0_eff = delta0_n  # Simplified: use normal mode critical separation
    deltaf_eff = deltaf_n

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


@njit(cache=True)
def cohesive_eval_mixed_traction_numba(
    delta_n: float,
    delta_t: float,
    delta_max: float,
    damage: float,
    params: np.ndarray,
) -> tuple[float, float]:
    """Evaluate mixed-mode traction WITHOUT updating state (for dissipation tracking).

    This is used to compute old tractions at previous time step for energy-consistent
    dissipation computation. It does NOT update the history variables.

    Parameters
    ----------
    delta_n, delta_t : float
        Normal and tangential openings [m]
    delta_max, damage : float
        Committed state (not updated)
    params : np.ndarray
        Unified cohesive law parameters (21 elements)

    Returns
    -------
    t_n, t_t : float
        Normal and tangential tractions [Pa]
    """
    # Call the full kernel but discard state updates and tangents
    t_n, t_t, _, _, _, _, _, _ = cohesive_update_mixed_values_numba(
        delta_n, delta_t, delta_max, damage, params, visc_damp=0.0
    )
    return t_n, t_t
