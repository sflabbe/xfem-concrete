"""Constitutive models for bulk material response (plane stress).

The XFEM assembly calls a model through a small interface that takes a
:class:`~xfem_clean.material_point.MaterialPoint` state and an in-plane strain
vector (Voigt ordering ``[exx, eyy, gxy]`` where ``gxy = du/dy + dv/dx``), and
returns the in-plane stress vector ``[sxx, syy, txy]`` plus an algorithmic
tangent ``C_t`` (3x3).

This module provides:
  - Linear elastic plane-stress (retrocompatible, with damage placeholders)
  - Drucker–Prager (associative) return mapping + consistent tangent
  - A "CDP-lite" concrete model: DP plasticity on *effective stress* with
    split scalar damage in tension/compression (damage frozen per Newton
    iteration, so the elastoplastic tangent remains consistent).

Notes on plane stress
---------------------
For plasticity/damage, plane stress requires solving an internal out-of-plane
strain ``ezz`` such that ``szz = 0``. We do this with a small local Newton
iteration inside ``integrate`` using the algorithmic tangent.

The implementation is intentionally compact and NumPy-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple, Optional, Dict, Any

import numpy as np
import math

from xfem_clean.material_point import MaterialPoint
from xfem_clean.linear_elastic import plane_stress_C


# ----------------------------
# Utilities (Voigt <-> tensor)
# ----------------------------


def _strain6_to_tensor(eps6: np.ndarray) -> np.ndarray:
    """Engineering-strain Voigt6 -> symmetric strain tensor."""
    e = np.asarray(eps6, dtype=float).reshape(6)
    E = np.zeros((3, 3), dtype=float)
    E[0, 0] = e[0]
    E[1, 1] = e[1]
    E[2, 2] = e[2]
    E[0, 1] = E[1, 0] = 0.5 * e[3]
    E[1, 2] = E[2, 1] = 0.5 * e[4]
    E[0, 2] = E[2, 0] = 0.5 * e[5]
    return E


def _tensor_to_strain6(E: np.ndarray) -> np.ndarray:
    """Symmetric strain tensor -> engineering-strain Voigt6."""
    e = np.zeros(6, dtype=float)
    e[0] = float(E[0, 0])
    e[1] = float(E[1, 1])
    e[2] = float(E[2, 2])
    e[3] = float(2.0 * E[0, 1])
    e[4] = float(2.0 * E[1, 2])
    e[5] = float(2.0 * E[0, 2])
    return e


def _stress6_to_tensor(sig6: np.ndarray) -> np.ndarray:
    """Stress Voigt6 -> symmetric stress tensor (no factor for shear)."""
    s = np.asarray(sig6, dtype=float).reshape(6)
    S = np.zeros((3, 3), dtype=float)
    S[0, 0] = s[0]
    S[1, 1] = s[1]
    S[2, 2] = s[2]
    S[0, 1] = S[1, 0] = s[3]
    S[1, 2] = S[2, 1] = s[4]
    S[0, 2] = S[2, 0] = s[5]
    return S


def _tensor_to_stress6(S: np.ndarray) -> np.ndarray:
    """Symmetric stress tensor -> stress Voigt6."""
    s = np.zeros(6, dtype=float)
    s[0] = float(S[0, 0])
    s[1] = float(S[1, 1])
    s[2] = float(S[2, 2])
    s[3] = float(S[0, 1])
    s[4] = float(S[1, 2])
    s[5] = float(S[0, 2])
    return s


def _iso_lame(E: float, nu: float) -> Tuple[float, float, float]:
    """Return (lambda, mu, K) for 3D isotropic elasticity."""
    E = float(E)
    nu = float(nu)
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    K = lam + 2.0 * mu / 3.0
    return lam, mu, K


def _Ce6_iso(E: float, nu: float) -> np.ndarray:
    """3D isotropic stiffness in engineering-strain Voigt6."""
    lam, mu, _K = _iso_lame(E, nu)
    C = np.zeros((6, 6), dtype=float)
    # normal-normal
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
    C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lam
    # shear (engineering)
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C


def _principal_max_2d(sig2: np.ndarray) -> float:
    sxx, syy, txy = [float(x) for x in np.asarray(sig2, dtype=float).reshape(3)]
    s_avg = 0.5 * (sxx + syy)
    R = float(np.sqrt((0.5 * (sxx - syy)) ** 2 + txy**2))
    return s_avg + R


def _principal_min_2d(sig2: np.ndarray) -> float:
    sxx, syy, txy = [float(x) for x in np.asarray(sig2, dtype=float).reshape(3)]
    s_avg = 0.5 * (sxx + syy)
    R = float(np.sqrt((0.5 * (sxx - syy)) ** 2 + txy**2))
    return s_avg - R


def _condense_plane_stress(C6: np.ndarray) -> np.ndarray:
    """Schur-condense a 6x6 tangent to plane stress (xx,yy,xy) eliminating zz."""
    # Use the 4x4 block [xx, yy, zz, xy] and eliminate zz.
    idx = [0, 1, 2, 3]
    C4 = np.asarray(C6, dtype=float)[np.ix_(idx, idx)]
    Cii = C4[np.ix_([0, 1, 3], [0, 1, 3])]
    Ci2 = C4[np.ix_([0, 1, 3], [2])]
    C2i = C4[np.ix_([2], [0, 1, 3])]
    C22 = float(C4[2, 2])
    if abs(C22) < 1e-24:
        return Cii
    return Cii - (Ci2 @ C2i) / C22


# ----------------------------
# Protocol
# ----------------------------


class ConstitutiveModel(Protocol):
    def integrate(self, mp: MaterialPoint, eps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (sigma, C_tangent) and update `mp` in-place."""


# ----------------------------
# Linear elastic (plane stress)
# ----------------------------


@dataclass
class LinearElasticPlaneStress:
    """Retrocompatible linear elastic plane-stress with damage placeholders."""

    E: float
    nu: float

    def __post_init__(self) -> None:
        self.C = plane_stress_C(float(self.E), float(self.nu))

    def integrate(self, mp: MaterialPoint, eps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        eps = np.asarray(eps, dtype=float).reshape(3)
        d = float(max(0.0, min(0.999999, mp.damage)))
        Ceff = (1.0 - d) * self.C
        sig = Ceff @ eps
        mp.eps[...] = eps
        mp.sigma[...] = sig
        return sig, Ceff


# -------------------------------------
# Drucker–Prager (associative) plasticity
# -------------------------------------


def _dp_alpha_k_from_mc(phi_deg: float, cohesion: float) -> Tuple[float, float]:
    """Return (alpha, k) for DP surface matching Mohr–Coulomb (inscribed).

    Uses pressure p = -tr(sigma)/3 (positive in compression) and
    yield f = q + alpha*p - k.
    """
    phi = np.deg2rad(float(phi_deg))
    s = float(np.sin(phi))
    c = float(np.cos(phi))
    denom = np.sqrt(3.0) * (3.0 - s)
    alpha = 2.0 * s / denom
    k = 6.0 * float(cohesion) * c / denom
    return float(alpha), float(k)


def _dp_return_mapping_3d(
    E: float,
    nu: float,
    sig_tr6: np.ndarray,
    eps_p6_old: np.ndarray,
    kappa_old: float,
    alpha: float,
    k0: float,
    H: float,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """3D DP update on effective stress. Returns (sig6, eps_p6, kappa, Cep6)."""
    lam, mu, K = _iso_lame(E, nu)
    sig_tr6 = np.asarray(sig_tr6, dtype=float).reshape(6)
    S_tr = _stress6_to_tensor(sig_tr6)

    # pressure positive in compression
    p_tr = float(-np.trace(S_tr) / 3.0)
    I = np.eye(3)
    s_tr = S_tr + p_tr * I
    J2 = 0.5 * float(np.sum(s_tr * s_tr))
    q_tr = float(np.sqrt(max(0.0, 3.0 * J2)))

    k_old = float(k0 + H * float(kappa_old))
    f_tr = float(q_tr + alpha * p_tr - k_old)
    if f_tr <= 0.0:
        # Elastic
        Ce6 = _Ce6_iso(E, nu)
        return sig_tr6, np.asarray(eps_p6_old, dtype=float).reshape(6), float(kappa_old), Ce6

    denom = float(3.0 * mu + (alpha**2) * K + H)
    if denom <= 1e-24:
        denom = 1e-24
    dgamma = f_tr / denom

    # Update invariants
    q_new = max(0.0, q_tr - 3.0 * mu * dgamma)
    p_new = p_tr - alpha * K * dgamma

    if q_tr > 1e-18:
        fac = q_new / q_tr
        s_new = fac * s_tr
    else:
        s_new = np.zeros((3, 3), dtype=float)

    S_new = s_new - p_new * I  # because S = s - pI with p=pressure
    sig6 = _tensor_to_stress6(S_new)

    # Flow direction n = df/dsigma (associative), with p = -tr(sigma)/3
    if q_new > 1e-18:
        n_dev = 1.5 * s_new / q_new
    else:
        n_dev = np.zeros((3, 3), dtype=float)
    n = n_dev - (alpha / 3.0) * I

    # Plastic strain update (tensor)
    Ep_old = _strain6_to_tensor(np.asarray(eps_p6_old, dtype=float).reshape(6))
    Ep_new = Ep_old + float(dgamma) * n
    eps_p6 = _tensor_to_strain6(Ep_new)
    kappa = float(kappa_old + dgamma)

    # Consistent elastoplastic tangent (damage handled outside)
    # a = Ce : n = lambda*tr(n)I + 2mu*n
    trn = float(np.trace(n))
    a = lam * trn * I + 2.0 * mu * n
    n_dot_a = float(np.sum(n * a))
    denom_t = float(H + n_dot_a)
    if denom_t <= 1e-24:
        denom_t = 1e-24

    def apply_Cep(Eps_tensor: np.ndarray) -> np.ndarray:
        # Ce:E
        trE = float(np.trace(Eps_tensor))
        CeE = lam * trE * I + 2.0 * mu * Eps_tensor
        a_dot_E = float(np.sum(a * Eps_tensor))
        return CeE - (a * a_dot_E) / denom_t

    # Build Cep6 by column-by-column application to unit engineering strains
    Cep6 = np.zeros((6, 6), dtype=float)
    for j in range(6):
        ej = np.zeros(6, dtype=float)
        ej[j] = 1.0
        Eps = _strain6_to_tensor(ej)
        Sj = apply_Cep(Eps)
        Cep6[:, j] = _tensor_to_stress6(Sj)

    return sig6, eps_p6, kappa, Cep6


@dataclass
class DruckerPrager:
    """Associative Drucker–Prager plasticity with isotropic hardening.

    Parameters
    ----------
    E, nu:
        3D elastic constants.
    phi_deg, cohesion:
        Mohr–Coulomb-like parameters used to compute (alpha, k0).
        Cohesion is in stress units (Pa).
    H:
        Isotropic hardening modulus in stress units (Pa) on the plastic
        multiplier ``mp.kappa``.
    """

    E: float
    nu: float
    phi_deg: float = 30.0
    cohesion: float = 1.0e6
    H: float = 0.0
    plane_stress_tol: float = 1e-9
    plane_stress_maxit: int = 25

    def __post_init__(self) -> None:
        self.alpha, self.k0 = _dp_alpha_k_from_mc(self.phi_deg, self.cohesion)
        self.Ce6 = _Ce6_iso(float(self.E), float(self.nu))

    def integrate(self, mp: MaterialPoint, eps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        eps = np.asarray(eps, dtype=float).reshape(3)

        # Energy bookkeeping (densities, J/m^3)
        sigma_old = np.array(mp.sigma, copy=True)
        eps_p_old2 = np.array(mp.eps_p, copy=True)
        w_pl_old = float(getattr(mp, "w_plastic", 0.0))

        # Pull 6D plastic strain and last plane-stress ezz guess from mp.extra
        eps_p6_old = mp.extra.get("eps_p6", None)
        if eps_p6_old is None:
            eps_p6_old = np.zeros(6, dtype=float)
        else:
            eps_p6_old = np.asarray(eps_p6_old, dtype=float).reshape(6)

        ezz = mp.extra.get("eps_zz", None)
        if ezz is None:
            # elastic plane-stress guess
            ezz = -float(self.nu) / max(1e-12, (1.0 - float(self.nu))) * float(eps[0] + eps[1])
        ezz = float(ezz)

        # local plane-stress Newton on ezz
        sig6 = None
        Cep6 = None
        eps_p6 = None
        kappa = float(mp.kappa)
        for _it in range(int(self.plane_stress_maxit)):
            eps6 = np.array([eps[0], eps[1], ezz, eps[2], 0.0, 0.0], dtype=float)
            # trial stress from elastic predictor with current plastic strain
            sig_tr6 = self.Ce6 @ (eps6 - eps_p6_old)
            sig6, eps_p6, kappa, Cep6 = _dp_return_mapping_3d(
                self.E,
                self.nu,
                sig_tr6,
                eps_p6_old,
                float(mp.kappa),
                float(self.alpha),
                float(self.k0),
                float(self.H),
            )
            r = float(sig6[2])  # szz
            if abs(r) <= float(self.plane_stress_tol) * max(
                1.0, float(abs(sig6[0]) + abs(sig6[1]) + abs(sig6[3]))
            ):
                break
            dsd_ezz = float(Cep6[2, 2])
            if abs(dsd_ezz) < 1e-18:
                break
            ezz -= r / dsd_ezz

        assert sig6 is not None and Cep6 is not None and eps_p6 is not None

        # Condense to plane-stress tangent
        Ct = _condense_plane_stress(Cep6)
        sig2 = np.array([sig6[0], sig6[1], sig6[3]], dtype=float)

        # update mp
        mp.eps[...] = eps
        mp.sigma[...] = sig2
        mp.kappa = float(kappa)
        mp.extra["eps_p6"] = np.array(eps_p6, copy=True)
        mp.extra["eps_zz"] = float(ezz)
        mp.eps_p[...] = np.array([eps_p6[0], eps_p6[1], eps_p6[3]], dtype=float)

        # Plastic dissipation increment (midpoint rule). Clamp to >=0 for numerical noise.
        d_eps_p = mp.eps_p - eps_p_old2
        sig_avg = 0.5 * (sigma_old + sig2)
        dW = float(sig_avg @ d_eps_p)
        if dW < 0.0:
            dW = 0.0
        mp.w_plastic = float(w_pl_old + dW)
        return sig2, Ct


# -------------------------------------
# Concrete damage plasticity (lite)
# -------------------------------------


@dataclass
class ConcreteCDP:
    """Concrete "CDP-lite": DP plasticity on effective stress + split scalar damage.

    This is a pragmatic stepping stone toward a full Lee–Fenves CDP.

    - Plasticity: associative Drucker–Prager (effective stress) with isotropic hardening.
    - Damage: two scalar variables (tension/compression) updated from principal
      strains (smeared softening). Damage is *frozen* within a Newton iteration,
      so the returned tangent is consistent for the elastoplastic part.

    Required parameters should be provided in Pa and meters.
    """

    E: float
    nu: float
    ft: float
    fc: float
    Gf_t: float
    lch: float
    phi_deg: float = 30.0
    cohesion: Optional[float] = None
    H: float = 0.0
    Gf_c: Optional[float] = None
    plane_stress_tol: float = 1e-9
    plane_stress_maxit: int = 25

    def __post_init__(self) -> None:
        self.Ce6 = _Ce6_iso(float(self.E), float(self.nu))
        self.alpha, self.k0 = _dp_alpha_k_from_mc(self.phi_deg, float(self.cohesion or (0.25 * self.fc)))
        # If cohesion is not provided, use a rough default tied to fc.
        if self.cohesion is None:
            # keep k0 consistent with fc magnitude (uniaxial compression approx)
            # For uniaxial compression sigma=-fc => q=fc, p=fc/3 => k ≈ fc(1+alpha/3)
            self.k0 = float(self.fc * (1.0 + self.alpha / 3.0))
        self.eps0_t = float(self.ft / max(1e-12, self.E))
        self.epsf_t = float(self.eps0_t + 2.0 * self.Gf_t / max(1e-12, self.ft * max(1e-12, self.lch)))
        if self.Gf_c is None:
            self.Gf_c = float(10.0 * self.Gf_t)
        self.eps0_c = float(self.fc / max(1e-12, self.E))
        self.epsf_c = float(self.eps0_c + 2.0 * float(self.Gf_c) / max(1e-12, self.fc * max(1e-12, self.lch)))

    def _update_damage_from_strain(self, mp: MaterialPoint, eps2: np.ndarray) -> None:
        # Use principal strains in 2D as a simple damage driver.
        exx, eyy, gxy = [float(x) for x in np.asarray(eps2, dtype=float).reshape(3)]
        exy = 0.5 * gxy
        # principal strains (2D)
        e_avg = 0.5 * (exx + eyy)
        R = float(np.sqrt((0.5 * (exx - eyy)) ** 2 + exy**2))
        e1 = e_avg + R
        e2 = e_avg - R
        e_t = max(0.0, float(e1))
        e_c = max(0.0, float(-e2))

        # tension damage
        k_t = float(mp.extra.get("kappa_t", 0.0))
        k_t = max(k_t, e_t)
        mp.extra["kappa_t"] = k_t
        if k_t > self.eps0_t:
            sig_t = float(self.ft * max(0.0, 1.0 - (k_t - self.eps0_t) / max(1e-12, (self.epsf_t - self.eps0_t))))
            d_t = 1.0 - sig_t / max(1e-12, (self.E * k_t))
            mp.damage_t = float(max(mp.damage_t, min(0.999999, max(0.0, d_t))))

            # Fracture energy density in tension (crack-band, linear softening):
            #   W_f,t = ∫_{eps0}^{k_t} sigma(ε) dε, capped by Gf_t/l_ch
            a = min(k_t, self.epsf_t) - self.eps0_t
            L = max(1e-12, (self.epsf_t - self.eps0_t))
            Wt = float(self.ft * (a - 0.5 * (a * a) / L))
            Wt = min(Wt, float(self.Gf_t) / max(1e-12, float(self.lch)))
            mp.w_fract_t = float(max(mp.w_fract_t, Wt))

        # compression damage
        k_c = float(mp.extra.get("kappa_c", 0.0))
        k_c = max(k_c, e_c)
        mp.extra["kappa_c"] = k_c
        if k_c > self.eps0_c:
            sig_c = float(self.fc * max(0.0, 1.0 - (k_c - self.eps0_c) / max(1e-12, (self.epsf_c - self.eps0_c))))
            d_c = 1.0 - sig_c / max(1e-12, (self.E * k_c))
            mp.damage_c = float(max(mp.damage_c, min(0.999999, max(0.0, d_c))))

            # Crushing energy density (compression softening):
            a = min(k_c, self.epsf_c) - self.eps0_c
            L = max(1e-12, (self.epsf_c - self.eps0_c))
            Wc = float(self.fc * (a - 0.5 * (a * a) / L))
            # Under our epsf_c construction this equals Gf_c/lch at full softening.
            Wc = min(Wc, float(self.Gf_c) / max(1e-12, float(self.lch)))
            mp.w_fract_c = float(max(mp.w_fract_c, Wc))

    def integrate(self, mp: MaterialPoint, eps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        eps = np.asarray(eps, dtype=float).reshape(3)

        # Energy bookkeeping (densities, J/m^3)
        sigma_old = np.array(mp.sigma, copy=True)
        eps_p_old2 = np.array(mp.eps_p, copy=True)
        w_pl_old = float(getattr(mp, "w_plastic", 0.0))

        # Freeze damage within this integrate call.
        d = float(max(0.0, min(0.999999, mp.damage)))

        # Pull 6D plastic strain and last plane-stress ezz guess from mp.extra
        eps_p6_old = mp.extra.get("eps_p6", None)
        if eps_p6_old is None:
            eps_p6_old = np.zeros(6, dtype=float)
        else:
            eps_p6_old = np.asarray(eps_p6_old, dtype=float).reshape(6)

        ezz = mp.extra.get("eps_zz", None)
        if ezz is None:
            ezz = -float(self.nu) / max(1e-12, (1.0 - float(self.nu))) * float(eps[0] + eps[1])
        ezz = float(ezz)

        sig6_eff = None
        Cep6 = None
        eps_p6 = None
        kappa = float(mp.kappa)
        for _it in range(int(self.plane_stress_maxit)):
            eps6 = np.array([eps[0], eps[1], ezz, eps[2], 0.0, 0.0], dtype=float)
            sig_tr6_eff = self.Ce6 @ (eps6 - eps_p6_old)
            sig6_eff, eps_p6, kappa, Cep6 = _dp_return_mapping_3d(
                self.E,
                self.nu,
                sig_tr6_eff,
                eps_p6_old,
                float(mp.kappa),
                float(self.alpha),
                float(self.k0),
                float(self.H),
            )
            # plane stress is enforced on the *effective stress* (sigma_eff_zz=0)
            # so the constraint is insensitive to a later damage scaling.
            r = float(sig6_eff[2])
            if abs(r) <= float(self.plane_stress_tol) * max(
                1.0, float(abs(sig6_eff[0]) + abs(sig6_eff[1]) + abs(sig6_eff[3]))
            ):
                break
            dsd_ezz = float(Cep6[2, 2])
            if abs(dsd_ezz) < 1e-18:
                break
            ezz -= r / dsd_ezz

        assert sig6_eff is not None and Cep6 is not None and eps_p6 is not None

        # Now update damage based on current strain (commit handled by caller).
        # This updates mp.damage_{t,c} monotonically.
        self._update_damage_from_strain(mp, eps)
        d = float(max(0.0, min(0.999999, mp.damage)))

        sig6 = (1.0 - d) * sig6_eff
        Cep6_eff = Cep6
        Cep6_d = (1.0 - d) * Cep6_eff

        Ct = _condense_plane_stress(Cep6_d)
        sig2 = np.array([sig6[0], sig6[1], sig6[3]], dtype=float)

        mp.eps[...] = eps
        mp.sigma[...] = sig2
        mp.kappa = float(kappa)
        mp.extra["eps_p6"] = np.array(eps_p6, copy=True)
        mp.extra["eps_zz"] = float(ezz)
        mp.eps_p[...] = np.array([eps_p6[0], eps_p6[1], eps_p6[3]], dtype=float)

        # Plastic dissipation density increment: dW_p = sigma : d eps_p
        deps_p2 = mp.eps_p - eps_p_old2
        sig_avg2 = 0.5 * (sigma_old + sig2)
        dW = float(np.dot(sig_avg2, deps_p2))
        if dW > 0.0:
            mp.w_plastic = float(w_pl_old + dW)
        else:
            mp.w_plastic = float(w_pl_old)
        return sig2, Ct


# ----------------------------
# Backwards-compatible aliases
# ----------------------------


# ----------------------------
# Concrete Damaged Plasticity (Lee–Fenves / Abaqus-like) "real" variant
# ----------------------------

def _voigt3_to_sig3(sig2: np.ndarray) -> np.ndarray:
    """Plane-stress Voigt3 -> 3D stress tensor with szz=0."""
    sxx, syy, txy = [float(x) for x in np.asarray(sig2, dtype=float).reshape(3)]
    return np.array([[sxx, txy, 0.0],
                     [txy, syy, 0.0],
                     [0.0, 0.0, 0.0]], dtype=float)


def _invariants_p_q(sig3: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Return mean pressure p (positive in compression), q, and deviatoric tensor s."""
    sig3 = np.asarray(sig3, dtype=float).reshape(3, 3)
    tr = float(np.trace(sig3))
    p = -tr / 3.0
    I = np.eye(3, dtype=float)
    s = sig3 + p * I
    j2 = 0.5 * float(np.sum(s * s))
    q = math.sqrt(max(0.0, 3.0 * j2))
    return p, q, s


def _dq_dsig_voigt(sig2: np.ndarray) -> np.ndarray:
    """Derivative of q wrt plane-stress Voigt3."""
    sig3 = _voigt3_to_sig3(sig2)
    p, q, s = _invariants_p_q(sig3)
    if q < 1e-14:
        return np.zeros(3, dtype=float)
    dq_dsig = (3.0 / (2.0 * q)) * s  # tensor
    return np.array([dq_dsig[0, 0], dq_dsig[1, 1], dq_dsig[0, 1]], dtype=float)


def _principal_grads_2d(sig2: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Return sigma1>=sigma2 and their gradients wrt [sxx, syy, txy]."""
    sxx, syy, txy = [float(x) for x in np.asarray(sig2, dtype=float).reshape(3)]
    s_avg = 0.5 * (sxx + syy)
    dx = 0.5 * (sxx - syy)
    R = math.sqrt(dx * dx + txy * txy)
    if R < 1e-14:
        # nearly isotropic: arbitrary direction, gradients reduce to average split
        sig1 = s_avg
        sig2v = s_avg
        g1 = np.array([0.5, 0.5, 0.0], dtype=float)
        g2 = np.array([0.5, 0.5, 0.0], dtype=float)
        return sig1, sig2v, g1, g2

    sig1 = s_avg + R
    sig2v = s_avg - R

    dR_dsxx = dx / (2.0 * R)
    dR_dsyy = -dx / (2.0 * R)
    dR_dtxy = txy / R

    g1 = np.array([0.5 + dR_dsxx, 0.5 + dR_dsyy, dR_dtxy], dtype=float)
    g2 = np.array([0.5 - dR_dsxx, 0.5 - dR_dsyy, -dR_dtxy], dtype=float)
    return float(sig1), float(sig2v), g1, g2



def _damage_operator_matrix(sig_eff: np.ndarray, dt: float, dc: float) -> np.ndarray:
    """Linear operator M such that sigma_nom = M * sigma_eff (damage frozen, directions frozen).

    We freeze the current principal direction angle θ and apply damage in the rotated
    (principal) stress basis (σ11, σ22, τ12). Normal components are scaled by dt/dc
    depending on sign; shear is scaled by the geometric mean sqrt(s11*s22).
    """
    sig_eff = np.asarray(sig_eff, dtype=float).reshape(3)
    sig1, sig2, _, _ = _principal_grads_2d(sig_eff)

    sxx, syy, txy = [float(x) for x in sig_eff]
    if abs(txy) < 1e-14 and abs(sxx - syy) < 1e-14:
        theta = 0.0
    else:
        theta = 0.5 * math.atan2(2.0 * txy, (sxx - syy))

    c = math.cos(theta)
    s = math.sin(theta)
    c2 = c * c
    s2 = s * s
    cs = c * s

    # Stress transformation (global -> rotated)
    T = np.array([
        [c2, s2, 2.0 * cs],
        [s2, c2, -2.0 * cs],
        [-cs, cs, c2 - s2],
    ], dtype=float)

    # Inverse transform (rotated -> global)
    Tinv = np.array([
        [c2, s2, -2.0 * cs],
        [s2, c2, 2.0 * cs],
        [cs, -cs, c2 - s2],
    ], dtype=float)

    s11 = (1.0 - dt) if sig1 >= 0.0 else (1.0 - dc)
    s22 = (1.0 - dt) if sig2 >= 0.0 else (1.0 - dc)
    s12 = math.sqrt(max(0.0, float(s11) * float(s22)))

    D = np.diag([float(s11), float(s22), float(s12)])
    M = Tinv @ D @ T
    return M


def _interp_with_slope(x: float, x_tab: np.ndarray, y_tab: np.ndarray) -> Tuple[float, float]:
    """Piecewise-linear interpolation returning (y, dy/dx)."""
    x_tab = np.asarray(x_tab, dtype=float).reshape(-1)
    y_tab = np.asarray(y_tab, dtype=float).reshape(-1)
    if len(x_tab) != len(y_tab) or len(x_tab) < 2:
        return float(y_tab[0] if len(y_tab) else 0.0), 0.0

    if x <= float(x_tab[0]):
        dy = (float(y_tab[1]) - float(y_tab[0])) / (float(x_tab[1]) - float(x_tab[0]) + 1e-18)
        return float(y_tab[0]), float(dy)
    if x >= float(x_tab[-1]):
        dy = (float(y_tab[-1]) - float(y_tab[-2])) / (float(x_tab[-1]) - float(x_tab[-2]) + 1e-18)
        return float(y_tab[-1]), float(dy)

    i = int(np.searchsorted(x_tab, x)) - 1
    i = max(0, min(i, len(x_tab) - 2))
    x0 = float(x_tab[i]); x1 = float(x_tab[i + 1])
    y0 = float(y_tab[i]); y1 = float(y_tab[i + 1])
    t = (x - x0) / (x1 - x0 + 1e-18)
    y = y0 * (1.0 - t) + y1 * t
    dy = (y1 - y0) / (x1 - x0 + 1e-18)
    return float(y), float(dy)


@dataclass
class ConcreteCDPReal:
    """Concrete Damaged Plasticity (Abaqus-like Lee–Fenves form) with return mapping + algorithmic tangent.

    This is a "real" CDP surface (Lubliner/Lee–Fenves type) in plane stress with:
      - yield surface controlled by (fb0/fc0, Kc)
      - non-associative hyperbolic plastic potential (psi, ecc)
      - split scalar damage variables (dt/dc) driven by uniaxial tables
      - robust modified return mapping (flow direction frozen at trial stress)
      - algorithmic tangent consistent with the modified return mapping and table hardening

    Notes
    -----
    - Damage is frozen within the integration call (no damage derivative in Ct).
    - Plane stress is enforced by using a plane-stress elastic matrix directly.
    - Uniaxial tables are taken from cdp_generator calibration (recommended).
    """

    E: float
    nu: float
    psi_deg: float
    ecc: float
    fb0_fc0: float
    Kc: float
    lch: float

    # tables (SI)
    w_tab_m: np.ndarray
    sig_t_tab_pa: np.ndarray
    dt_tab: np.ndarray

    eps_in_c_tab: np.ndarray
    sig_c_tab_pa: np.ndarray
    dc_tab: np.ndarray

    f_t0: float
    f_c0: float

    # numerical floors for stability
    sig_t_floor_frac: float = 0.02
    sig_c_floor_frac: float = 0.05

    def __post_init__(self) -> None:
        self.Ce = plane_stress_C(float(self.E), float(self.nu))
        self.w_tab_m = np.asarray(self.w_tab_m, dtype=float).reshape(-1)
        self.sig_t_tab_pa = np.asarray(self.sig_t_tab_pa, dtype=float).reshape(-1)
        self.dt_tab = np.asarray(self.dt_tab, dtype=float).reshape(-1)
        self.eps_in_c_tab = np.asarray(self.eps_in_c_tab, dtype=float).reshape(-1)
        self.sig_c_tab_pa = np.asarray(self.sig_c_tab_pa, dtype=float).reshape(-1)
        self.dc_tab = np.asarray(self.dc_tab, dtype=float).reshape(-1)

        # sanitize monotonic damage
        self.dt_tab = np.clip(self.dt_tab, 0.0, 0.9999)
        self.dc_tab = np.clip(self.dc_tab, 0.0, 0.9999)

        # Interpret generator tables as *nominal* stresses together with damage.
        # For return mapping / yielding we need the corresponding *effective* strengths.
        self.sig_t_eff_tab_pa = self.sig_t_tab_pa / np.maximum(1e-12, (1.0 - self.dt_tab))
        self.sig_c_eff_tab_pa = self.sig_c_tab_pa / np.maximum(1e-12, (1.0 - self.dc_tab))

        # Precompute cumulative integrals of nominal curves for energy accounting.
        def _cumtrapz(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float).reshape(-1)
            y = np.asarray(y, dtype=float).reshape(-1)
            if x.size < 2:
                return np.zeros_like(x)
            out = np.zeros_like(x)
            dx = np.diff(x)
            out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
            return out

        self._phi_t_tab = _cumtrapz(self.w_tab_m, self.sig_t_tab_pa)  # [J/m^2]
        self._phi_c_tab = _cumtrapz(self.eps_in_c_tab, self.sig_c_tab_pa)  # [J/m^3]

    # ---- table accessors (effective stresses) ----
    def sigma_t_eff(self, w_t: float) -> Tuple[float, float]:
        return _interp_with_slope(float(w_t), self.w_tab_m, self.sig_t_eff_tab_pa)

    def sigma_t_nom(self, w_t: float) -> Tuple[float, float]:
        return _interp_with_slope(float(w_t), self.w_tab_m, self.sig_t_tab_pa)

    def damage_t(self, w_t: float) -> float:
        y, _ = _interp_with_slope(float(w_t), self.w_tab_m, self.dt_tab)
        return float(np.clip(y, 0.0, 0.9999))

    def sigma_c_eff(self, eps_in_c: float) -> Tuple[float, float]:
        return _interp_with_slope(float(eps_in_c), self.eps_in_c_tab, self.sig_c_eff_tab_pa)

    def sigma_c_nom(self, eps_in_c: float) -> Tuple[float, float]:
        return _interp_with_slope(float(eps_in_c), self.eps_in_c_tab, self.sig_c_tab_pa)

    def _phi_t(self, w_t: float) -> float:
        y, _ = _interp_with_slope(float(w_t), self.w_tab_m, self._phi_t_tab)
        return float(max(0.0, y))

    def _phi_c(self, eps_in_c: float) -> float:
        y, _ = _interp_with_slope(float(eps_in_c), self.eps_in_c_tab, self._phi_c_tab)
        return float(max(0.0, y))

    def damage_c(self, eps_in_c: float) -> float:
        y, _ = _interp_with_slope(float(eps_in_c), self.eps_in_c_tab, self.dc_tab)
        return float(np.clip(y, 0.0, 0.9999))

    # ---- CDP mechanics ----
    def _alpha(self) -> float:
        fbfc = float(self.fb0_fc0)
        return float((fbfc - 1.0) / (2.0 * fbfc - 1.0))

    def _gamma(self) -> float:
        Kc = float(self.Kc)
        return float(3.0 * (1.0 - Kc) / (2.0 * Kc - 1.0))

    def _yield(self, sig_eff: np.ndarray, w_t: float, eps_in_c: float) -> Tuple[float, Dict[str, float]]:
        """Yield function F and a small cache with strength values (for tangent)."""
        sig_eff = np.asarray(sig_eff, dtype=float).reshape(3)
        sig3 = _voigt3_to_sig3(sig_eff)
        p, q, _ = _invariants_p_q(sig3)

        sig_max = _principal_max_2d(sig_eff)
        sig_min = _principal_min_2d(sig_eff)

        alpha = self._alpha()

        sig_c_tab, dsigc = self.sigma_c_eff(eps_in_c)
        sig_t_tab, dsigt = self.sigma_t_eff(w_t)

        # Floors prevent ill-conditioning when tension strength tends to 0 after full cracking.
        sig_t = max(float(sig_t_tab), float(self.sig_t_floor_frac) * float(self.f_t0))
        sig_c = max(float(sig_c_tab), float(self.sig_c_floor_frac) * float(self.f_c0))

        beta = (sig_c / sig_t) * (1.0 - alpha) - (1.0 + alpha)
        gamma = self._gamma()

        term = q - 3.0 * alpha * p + beta * max(sig_max, 0.0) - gamma * max(-sig_min, 0.0)
        F = term / (1.0 - alpha) - sig_c

        cache = dict(
            alpha=float(alpha),
            beta=float(beta),
            gamma=float(gamma),
            sig_t=float(sig_t),
            sig_c=float(sig_c),
            sig_max=float(sig_max),
            sig_min=float(sig_min),
            dsigc=float(dsigc),
            dsigt=float(dsigt),
            p=float(p),
            q=float(q),
        )
        return float(F), cache

    def _flow_dir(self, sig_eff: np.ndarray) -> np.ndarray:
        """Non-associative plastic potential flow direction m = dG/dsigma (Voigt3)."""
        sig_eff = np.asarray(sig_eff, dtype=float).reshape(3)
        sig3 = _voigt3_to_sig3(sig_eff)
        p, q, s = _invariants_p_q(sig3)

        psi = math.radians(float(self.psi_deg))
        tanpsi = math.tan(psi)

        A = float(self.ecc) * float(self.f_t0) * tanpsi
        denom = math.sqrt(A * A + q * q)
        dg_dq = 0.0 if denom < 1e-14 else (q / denom)

        dp_voigt = np.array([-1.0 / 3.0, -1.0 / 3.0, 0.0], dtype=float)
        dq_voigt = _dq_dsig_voigt(sig_eff)

        m = dg_dq * dq_voigt - tanpsi * dp_voigt
        return np.asarray(m, dtype=float).reshape(3)

    def _yield_grad(self, sig_eff: np.ndarray, cache: Dict[str, float]) -> np.ndarray:
        """Gradient n = dF/dsigma (Voigt3), strengths treated as constants."""
        sig_eff = np.asarray(sig_eff, dtype=float).reshape(3)

        alpha = float(cache["alpha"])
        beta = float(cache["beta"])
        gamma = float(cache["gamma"])
        sig_max = float(cache["sig_max"])
        sig_min = float(cache["sig_min"])

        dq = _dq_dsig_voigt(sig_eff)
        dp = np.array([-1.0 / 3.0, -1.0 / 3.0, 0.0], dtype=float)

        s1, s2, g1, g2 = _principal_grads_2d(sig_eff)
        # Map max/min to the corresponding gradient
        if s1 >= s2:
            grad_max = g1
            grad_min = g2
        else:
            grad_max = g2
            grad_min = g1

        n = dq - 3.0 * alpha * dp
        if sig_max > 0.0:
            n = n + beta * grad_max
        if sig_min < 0.0:
            # term uses -gamma*max(-sig_min,0) -> +gamma*sig_min when sig_min<0
            n = n + gamma * grad_min

        n = n / (1.0 - alpha)
        return np.asarray(n, dtype=float).reshape(3)

    def integrate(self, mp: MaterialPoint, eps2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate (plane stress) CDP: return nominal stress and algorithmic Ct."""
        eps2 = np.asarray(eps2, dtype=float).reshape(3)

        # --- Energy bookkeeping (densities, J/m^3) ---
        w_pl_old = float(getattr(mp, "w_plastic", 0.0))
        w_ft_old = float(getattr(mp, "w_fract_t", 0.0))
        w_fc_old = float(getattr(mp, "w_fract_c", 0.0))

        # Retrieve state
        eps_p0 = np.asarray(mp.eps_p, dtype=float).reshape(3)
        w_t0 = float(mp.extra.get("cdp_w_t", 0.0))
        eps_in_c0 = float(mp.extra.get("cdp_eps_in_c", 0.0))
        dt0 = float(mp.damage_t)
        dc0 = float(mp.damage_c)

        sig_tr = self.Ce @ (eps2 - eps_p0)
        # previous committed effective stress (for plastic work increment)
        sig_eff_old = self.Ce @ (np.asarray(mp.eps, dtype=float).reshape(3) - eps_p0)

        Ftr, cache_tr = self._yield(sig_tr, w_t0, eps_in_c0)
        if Ftr <= 1e-10:
            # Elastic step (damage frozen)
            M = _damage_operator_matrix(sig_tr, dt0, dc0)
            sig_nom = M @ sig_tr
            Ct = M @ self.Ce
            mp.eps[...] = eps2
            mp.sigma[...] = sig_nom
            return sig_nom, Ct

        # Plastic step: modified return mapping (freeze m at trial stress)
        mvec = self._flow_dir(sig_tr)
        Cm = self.Ce @ mvec
        mnorm = float(np.linalg.norm(mvec)) + 1e-14

        in_tension = (_principal_max_2d(sig_tr) >= 0.0)

        def state_from_dl(dl: float) -> Tuple[np.ndarray, float, float, float, float]:
            eps_p = eps_p0 + dl * mvec
            w_t = w_t0 + (dl * mnorm * float(self.lch) if in_tension else 0.0)
            eps_in_c = eps_in_c0 + (dl * mnorm if (not in_tension) else 0.0)

            dt = dt0
            dc = dc0
            if in_tension:
                dt = max(dt0, self.damage_t(w_t))
            else:
                dc = max(dc0, self.damage_c(eps_in_c))
            return eps_p, w_t, eps_in_c, float(dt), float(dc)

        def F_of(dl: float) -> float:
            # sigma = sig_tr - dl * Ce*m
            sig = sig_tr - dl * Cm
            _, w_t, eps_in_c, _, _ = state_from_dl(dl)
            F, _ = self._yield(sig, w_t, eps_in_c)
            return float(F)

        denom_est = float(np.dot(mvec, Cm)) + 1e-18
        dl_est = max(0.0, float(Ftr) / denom_est)

        lo = 0.0
        hi = max(1e-16, dl_est)
        # Expand upper bracket until F(hi)<=0
        for _ in range(40):
            if F_of(hi) <= 0.0:
                break
            hi *= 2.0
        else:
            # fallback: accept last hi (will likely be over-softened); keeps code progressing
            pass

        # Bisection
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if F_of(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        dl = hi

        eps_p, w_t, eps_in_c, dt, dc = state_from_dl(dl)
        sig_eff = sig_tr - dl * Cm
        Ff, cache = self._yield(sig_eff, w_t, eps_in_c)

        # Algorithmic tangent (consistent with modified return mapping + table hardening)
        nvec = self._yield_grad(sig_eff, cache)
        den_base = float(nvec @ (self.Ce @ mvec))

        sig_max_pos = max(float(cache["sig_max"]), 0.0)
        sig_t = float(cache["sig_t"])
        sig_c = float(cache["sig_c"])

        dsig_t_dlambda = 0.0
        dsig_c_dlambda = 0.0
        if in_tension:
            # sigma_t depends on w_t (in m)
            _, dsigt_dw = self.sigma_t_eff(w_t)
            dsig_t_dlambda = float(dsigt_dw) * (mnorm * float(self.lch))
        else:
            # sigma_c depends on eps_in_c
            _, dsigc_de = self.sigma_c_eff(eps_in_c)
            dsig_c_dlambda = float(dsigc_de) * (mnorm)

        dF_dsig_c = (sig_max_pos / sig_t) - 1.0
        dF_dsig_t = -sig_c * sig_max_pos / (sig_t * sig_t)

        hard = dF_dsig_c * dsig_c_dlambda + dF_dsig_t * dsig_t_dlambda
        den = float(den_base - hard)
        if abs(den) < 1e-14:
            den = 1e-14 if den >= 0.0 else -1e-14

        B = nvec @ self.Ce  # row (3,)
        Ct_eff = self.Ce - np.outer(Cm, B) / den

        # Apply damage operator (frozen)
        M = _damage_operator_matrix(sig_eff, dt, dc)
        sig_nom = M @ sig_eff
        Ct = M @ Ct_eff

        # Update dissipation densities (monotonic / history-driven)
        deps_p = np.asarray(eps_p - eps_p0, dtype=float).reshape(3)
        if float(np.linalg.norm(deps_p)) > 0.0:
            sig_avg = 0.5 * (np.asarray(sig_eff_old, dtype=float).reshape(3) + np.asarray(sig_eff, dtype=float).reshape(3))
            dWpl = float(sig_avg @ deps_p)
            if dWpl > 0.0:
                mp.w_plastic = float(w_pl_old + dWpl)
            else:
                mp.w_plastic = float(w_pl_old)
        else:
            mp.w_plastic = float(w_pl_old)
        # Commit into mp (trial object; driver decides commit)
        mp.eps[...] = eps2
        mp.eps_p[...] = eps_p
        mp.damage_t = float(dt)
        mp.damage_c = float(dc)
        mp.extra["cdp_w_t"] = float(w_t)
        mp.extra["cdp_eps_in_c"] = float(eps_in_c)
        mp.kappa = float(max(w_t / max(1e-12, float(self.lch)), eps_in_c))
        mp.sigma[...] = sig_nom

        # --- dissipation bookkeeping ---
        deps_p = eps_p - eps_p0
        if float(np.linalg.norm(deps_p)) > 0.0:
            sig_eff_new = np.asarray(sig_eff, dtype=float).reshape(3)
            sig_eff_avg = 0.5 * (np.asarray(sig_eff_old, dtype=float).reshape(3) + sig_eff_new)
            dWp = float(np.dot(sig_eff_avg, deps_p))
            if dWp > 0.0:
                mp.w_plastic = float(w_pl_old + dWp)
            else:
                mp.w_plastic = float(w_pl_old)
        else:
            mp.w_plastic = float(w_pl_old)

        # Damage-related dissipations from the nominal uniaxial curves.
        # (These are envelope energies, hence monotonic in w_t / eps_in_c.)
        mp.w_fract_t = float(max(w_ft_old, self._phi_t(w_t) / max(1e-12, float(self.lch))))
        mp.w_fract_c = float(max(w_fc_old, self._phi_c(eps_in_c)))

        return sig_nom, Ct


class DruckerPragerPlaceholder:
    """Backwards-compatible placeholder (elastic).

    Older scripts imported this name with only (E, nu). Keep it behaving as
    linear elastic to avoid surprising failures.
    """

    E: float
    nu: float

    def __post_init__(self) -> None:
        self._lin = LinearElasticPlaneStress(self.E, self.nu)

    def integrate(self, mp: MaterialPoint, eps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._lin.integrate(mp, eps)


@dataclass
class CDPPlaceholder:
    """Backwards-compatible placeholder (elastic + damage).

    The real implementation is :class:`ConcreteCDP`.
    """

    E: float
    nu: float

    def __post_init__(self) -> None:
        self._lin = LinearElasticPlaneStress(self.E, self.nu)

    def integrate(self, mp: MaterialPoint, eps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._lin.integrate(mp, eps)
