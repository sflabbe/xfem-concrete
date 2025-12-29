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

    delta_abs = abs(delta)
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
