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

    def __post_init__(self) -> None:
        if self.law.lower().startswith("rein") and self.wcrit <= 0.0:
            # Choose w_c such that:  ∫_0^{w_c} t(w) dw = Gf, with t(w)=ft*f(w/w_c)
            # => w_c = Gf / (ft * ∫_0^1 f(x) dx)
            I = self._reinhardt_I(self.c1, self.c2)
            self.wcrit = float(self.Gf / (max(1e-30, self.ft) * max(1e-30, I)))

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
    """
    Kn = float(law.Kn)
    ft = float(law.ft)
    d0 = float(law.delta0)
    k_res = float(law.kres_factor) * Kn
    k_cap = float(law.kcap_factor) * Kn

    delta = float(delta)
    delta_abs = abs(delta)

    # History (irreversible max opening)
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
