"""Numba-friendly bulk constitutive kernels (Phase 2b).

These kernels are **stateless** (no Python objects) and operate on primitive
NumPy arrays / floats so they can compile in Numba ``nopython`` mode.

Scope (Phase 2b)
---------------
* Linear elastic plane-stress.
* Drucker–Prager (associated) with isotropic hardening, plane-stress enforced
  by a local Newton iteration on ``eps_zz``.

Notes
-----
* The solver stores history in :class:`~xfem_clean.xfem.state_arrays.BulkStateArrays`.
  The kernels below work directly on the raw arrays (eps_p6, kappa, eps_zz).
* For ``bulk_material='cdp'`` we keep using the Python constitutive model for now.
  The CDP return mapping is significantly more complex; it will be moved to Numba
  in a later phase.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from xfem_clean.numba.utils import njit


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


@njit(cache=True)
def _voigt6_dot(a: np.ndarray, b: np.ndarray) -> float:
    """Tensor double contraction in Voigt-6 with tensor shear stresses.

    Assumes vectors are ordered as ``[xx, yy, zz, xy, yz, xz]`` with **tensor**
    shear stresses (not engineering) and **engineering** shear strains in the
    elasticity matrix.
    """

    return (
        a[0] * b[0]
        + a[1] * b[1]
        + a[2] * b[2]
        + 2.0 * (a[3] * b[3] + a[4] * b[4] + a[5] * b[5])
    )


@njit(cache=True)
def _iso_C6(E: float, nu: float) -> np.ndarray:
    """3D isotropic elasticity (Voigt-6), engineering shear strains."""
    C = np.zeros((6, 6), dtype=np.float64)
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    for i in range(3):
        for j in range(3):
            C[i, j] = lam
        C[i, i] += 2.0 * mu
    C[3, 3] = mu
    C[4, 4] = mu
    C[5, 5] = mu
    return C


@njit(cache=True)
def _dp_alpha_k(phi_rad: float, cohesion: float) -> Tuple[float, float]:
    """Standard Drucker–Prager parameters (circumscribed Mohr–Coulomb).

    Returns ``alpha`` and ``k`` for:

        f = q + alpha * p - k
    """

    s = math.sin(phi_rad)
    c = math.cos(phi_rad)
    denom = math.sqrt(3.0) * (3.0 - s)
    alpha = 2.0 * s / denom
    k = 6.0 * cohesion * c / denom
    return alpha, k


@njit(cache=True)
def _dp_return_mapping_3d(
    eps6: np.ndarray,
    eps_p6_old: np.ndarray,
    kappa_old: float,
    E: float,
    nu: float,
    alpha: float,
    k0: float,
    H: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Associated DP return mapping in 3D.

    Returns
    -------
    sigma6 : (6,) ndarray
    Cep6   : (6,6) ndarray (algorithmic tangent)
    eps_p6_new : (6,) ndarray (engineering strains)
    kappa_new : float
    dg : float (plastic multiplier increment)
    """

    Ce = _iso_C6(E, nu)
    eps_e = eps6 - eps_p6_old
    sig_trial = Ce @ eps_e

    # invariants
    p = (sig_trial[0] + sig_trial[1] + sig_trial[2]) / 3.0
    sdev = np.empty(6, dtype=np.float64)
    sdev[0] = sig_trial[0] - p
    sdev[1] = sig_trial[1] - p
    sdev[2] = sig_trial[2] - p
    sdev[3] = sig_trial[3]
    sdev[4] = sig_trial[4]
    sdev[5] = sig_trial[5]
    q = math.sqrt(max(0.0, 1.5 * _voigt6_dot(sdev, sdev)))

    f = q + alpha * p - (k0 + H * kappa_old)
    if f <= 0.0 or q < 1e-14:
        return sig_trial, Ce, eps_p6_old, kappa_old, 0.0

    # df/dsigma
    inv_q = 1.0 / q
    n = np.empty(6, dtype=np.float64)
    n[0] = 1.5 * inv_q * sdev[0] + alpha / 3.0
    n[1] = 1.5 * inv_q * sdev[1] + alpha / 3.0
    n[2] = 1.5 * inv_q * sdev[2] + alpha / 3.0
    n[3] = 1.5 * inv_q * sdev[3]
    n[4] = 1.5 * inv_q * sdev[4]
    n[5] = 1.5 * inv_q * sdev[5]

    # plastic flow in engineering strain space: shear needs factor 2
    M = np.empty(6, dtype=np.float64)
    M[0] = n[0]
    M[1] = n[1]
    M[2] = n[2]
    M[3] = 2.0 * n[3]
    M[4] = 2.0 * n[4]
    M[5] = 2.0 * n[5]

    A = Ce @ M
    denom = _voigt6_dot(n, A) + H
    if abs(denom) < 1e-18:
        return sig_trial, Ce, eps_p6_old, kappa_old, 0.0
    dg = f / denom

    sigma = sig_trial - dg * A
    eps_p6_new = eps_p6_old + dg * M
    kappa_new = kappa_old + dg

    # Consistent tangent: Cep = Ce - (Ce*M) ⊗ (Ce^T*(w*n)) / denom
    wn = np.empty(6, dtype=np.float64)
    wn[0] = n[0]
    wn[1] = n[1]
    wn[2] = n[2]
    wn[3] = 2.0 * n[3]
    wn[4] = 2.0 * n[4]
    wn[5] = 2.0 * n[5]
    B = Ce.T @ wn
    Cep = Ce - np.outer(A, B) / denom

    return sigma, Cep, eps_p6_new, kappa_new, dg


# -----------------------------------------------------------------------------
# Public kernels
# -----------------------------------------------------------------------------


@njit(cache=True)
def elastic_integrate_plane_stress_numba(
    eps3: np.ndarray,
    E: float,
    nu: float,
    damage_t: float,
    damage_c: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Linear elastic plane-stress (Voigt3: xx, yy, xy)."""
    ex, ey, gxy = eps3[0], eps3[1], eps3[2]
    fac = E / (1.0 - nu * nu)
    C = np.zeros((3, 3), dtype=np.float64)
    C[0, 0] = fac
    C[0, 1] = fac * nu
    C[1, 0] = fac * nu
    C[1, 1] = fac
    C[2, 2] = fac * (1.0 - nu) / 2.0

    sig = C @ eps3

    d = damage_t
    if damage_c > d:
        d = damage_c
    if d > 0.0:
        s = 1.0 - d
        sig = sig * s
        C = C * s
    return sig, C


@njit(cache=True)
def dp_integrate_plane_stress_numba(
    eps3: np.ndarray,
    eps_p6_old: np.ndarray,
    eps_zz_old: float,
    kappa_old: float,
    E: float,
    nu: float,
    alpha: float,
    k0: float,
    H: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Drucker–Prager (associated) plane-stress integration.

    Returns
    -------
    sigma3 : (3,) ndarray
    Ct3    : (3,3) ndarray
    eps_p6_new : (6,) ndarray
    eps_zz_new : float
    kappa_new  : float
    dW_plastic : float (increment, J/m^3)
    """

    # Local Newton on eps_zz enforcing sigma_zz = 0
    ezz = eps_zz_old
    eps_p6 = eps_p6_old.copy()
    kappa = kappa_old
    sigma6 = np.zeros(6, dtype=np.float64)
    Cep6 = _iso_C6(E, nu)
    dg_last = 0.0

    for _it in range(20):
        eps6 = np.zeros(6, dtype=np.float64)
        eps6[0] = eps3[0]
        eps6[1] = eps3[1]
        eps6[2] = ezz
        eps6[3] = eps3[2]

        sigma6, Cep6, eps_p6_new, kappa_new, dg = _dp_return_mapping_3d(
            eps6, eps_p6_old, kappa_old, E, nu, alpha, k0, H
        )

        r = sigma6[2]
        if abs(r) < 1e-9 * max(1.0, abs(k0)):
            eps_p6 = eps_p6_new
            kappa = kappa_new
            dg_last = dg
            break

        dsd_ezz = Cep6[2, 2]
        if abs(dsd_ezz) < 1e-18:
            eps_p6 = eps_p6_new
            kappa = kappa_new
            dg_last = dg
            break

        # Newton update
        ezz = ezz - r / dsd_ezz
        eps_p6 = eps_p6_new
        kappa = kappa_new
        dg_last = dg

    # Plane-stress condensation
    sigma3 = np.empty(3, dtype=np.float64)
    sigma3[0] = sigma6[0]
    sigma3[1] = sigma6[1]
    sigma3[2] = sigma6[3]

    Caa = np.empty((3, 3), dtype=np.float64)
    Caa[0, 0] = Cep6[0, 0]
    Caa[0, 1] = Cep6[0, 1]
    Caa[0, 2] = Cep6[0, 3]
    Caa[1, 0] = Cep6[1, 0]
    Caa[1, 1] = Cep6[1, 1]
    Caa[1, 2] = Cep6[1, 3]
    Caa[2, 0] = Cep6[3, 0]
    Caa[2, 1] = Cep6[3, 1]
    Caa[2, 2] = Cep6[3, 3]

    Cab = np.empty(3, dtype=np.float64)
    Cab[0] = Cep6[0, 2]
    Cab[1] = Cep6[1, 2]
    Cab[2] = Cep6[3, 2]
    Cbb = Cep6[2, 2]
    if abs(Cbb) < 1e-18:
        Ct3 = Caa
    else:
        Ct3 = Caa - np.outer(Cab, np.array([Cep6[2, 0], Cep6[2, 1], Cep6[2, 3]])) / Cbb

    # Plastic dissipation increment (approx): sigma : d_eps_p
    deps_p6 = eps_p6 - eps_p6_old
    dW = _voigt6_dot(sigma6, deps_p6)
    if dW < 0.0:
        dW = 0.0

    return sigma3, Ct3, eps_p6, ezz, kappa, dW


# -----------------------------------------------------------------------------
# Packing helpers used by the Python driver
# -----------------------------------------------------------------------------


def pack_bulk_params(model) -> Tuple[int, Optional[np.ndarray]]:
    """Return (bulk_kind, params) for Numba bulk kernels.

    bulk_kind:
      0 -> disabled / use Python material.integrate
      1 -> elastic plane-stress
      2 -> Drucker–Prager plane-stress
    """
    bulk = str(getattr(model, "bulk_material", "elastic")).lower().strip()
    if bulk in ("elastic", "lin", "linear"):
        return 1, np.array([float(model.E), float(model.nu)], dtype=float)
    if bulk in ("dp", "drucker-prager", "druckerprager"):
        phi = float(getattr(model, "dp_phi_deg", 30.0))
        cohesion = float(getattr(model, "dp_cohesion", 0.0))
        H = float(getattr(model, "dp_H", 0.0))
        alpha, k0 = _dp_alpha_k(math.radians(phi), cohesion)
        return 2, np.array([float(model.E), float(model.nu), float(alpha), float(k0), float(H)], dtype=float)
    # CDP not ported in Phase 2b yet.
    return 0, None
