"""State storage helpers for Numba-friendly XFEM kernels.

Phase-1 (data-oriented) refactor
--------------------------------
The original solver stores history variables in Python dictionaries keyed by
``(elem_id, ip_id)`` or ``(crack_id, elem_id, gp_id)``. This is convenient but
blocks Numba's ``nopython`` mode and also creates a lot of per-iteration Python
overhead.

This module provides:

* :class:`BulkStateArrays` + :class:`BulkStatePatch` for material-point states
  (elastic / Drucker–Prager / CDP).
* :class:`CohesiveStateArrays` + :class:`CohesiveStatePatch` for cohesive-zone
  history variables.

The *patch* objects follow the same idea used elsewhere in the codebase:
assembly returns a *trial* update without mutating committed history. At
convergence, the patch can be applied to a copy of the committed arrays.

Notes
-----
This is an intentionally conservative stepping stone. It still uses
:class:`~xfem_clean.material_point.MaterialPoint` as the public state container
so existing constitutive models remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from xfem_clean.material_point import MaterialPoint, mp_default
from xfem_clean.cohesive_laws import CohesiveState


# ----------------------------------------------------------------------------
# Bulk (material-point) arrays
# ----------------------------------------------------------------------------


@dataclass
class BulkStatePatch:
    """Sparse set of bulk state updates.

    Stores per-IP updated values (trial) without mutating committed arrays.
    """

    idx_e: List[int]
    idx_ip: List[int]
    eps: List[np.ndarray]
    sigma: List[np.ndarray]
    eps_p: List[np.ndarray]
    damage_t: List[float]
    damage_c: List[float]
    kappa: List[float]
    w_plastic: List[float]
    w_fract_t: List[float]
    w_fract_c: List[float]

    # Extra fields used by DP/CDP plane-stress condensation
    eps_p6: List[np.ndarray]
    eps_zz: List[float]
    kappa_t: List[float]
    kappa_c: List[float]
    cdp_w_t: List[float]
    cdp_eps_in_c: List[float]

    @classmethod
    def empty(cls) -> "BulkStatePatch":
        return cls(
            idx_e=[],
            idx_ip=[],
            eps=[],
            sigma=[],
            eps_p=[],
            damage_t=[],
            damage_c=[],
            kappa=[],
            w_plastic=[],
            w_fract_t=[],
            w_fract_c=[],
            eps_p6=[],
            eps_zz=[],
            kappa_t=[],
            kappa_c=[],
            cdp_w_t=[],
            cdp_eps_in_c=[],
        )

    def add(self, e: int, ip: int, mp: MaterialPoint) -> None:
        self.idx_e.append(int(e))
        self.idx_ip.append(int(ip))
        self.eps.append(np.asarray(mp.eps, dtype=float).copy())
        self.sigma.append(np.asarray(mp.sigma, dtype=float).copy())
        self.eps_p.append(np.asarray(mp.eps_p, dtype=float).copy())
        self.damage_t.append(float(getattr(mp, "damage_t", 0.0)))
        self.damage_c.append(float(getattr(mp, "damage_c", 0.0)))
        self.kappa.append(float(getattr(mp, "kappa", 0.0)))
        self.w_plastic.append(float(getattr(mp, "w_plastic", getattr(mp, "w_plastic", 0.0))))
        self.w_fract_t.append(float(getattr(mp, "w_fract_t", 0.0)))
        self.w_fract_c.append(float(getattr(mp, "w_fract_c", 0.0)))

        eps_p6 = mp.extra.get("eps_p6", None)
        if eps_p6 is None:
            eps_p6 = np.zeros(6, dtype=float)
        self.eps_p6.append(np.asarray(eps_p6, dtype=float).reshape(6).copy())
        self.eps_zz.append(float(mp.extra.get("eps_zz", 0.0)))
        self.kappa_t.append(float(mp.extra.get("kappa_t", 0.0)))
        self.kappa_c.append(float(mp.extra.get("kappa_c", 0.0)))
        self.cdp_w_t.append(float(mp.extra.get("cdp_w_t", 0.0)))
        self.cdp_eps_in_c.append(float(mp.extra.get("cdp_eps_in_c", 0.0)))

    def add_values(
        self,
        e: int,
        ip: int,
        *,
        eps: np.ndarray,
        sigma: np.ndarray,
        eps_p: np.ndarray,
        damage_t: float,
        damage_c: float,
        kappa: float,
        w_plastic: float,
        w_fract_t: float,
        w_fract_c: float,
        eps_p6: np.ndarray,
        eps_zz: float,
        kappa_t: float,
        kappa_c: float,
        cdp_w_t: float,
        cdp_eps_in_c: float,
    ) -> None:
        """Add an update directly from primitive values (Numba-friendly path)."""

        self.idx_e.append(int(e))
        self.idx_ip.append(int(ip))
        self.eps.append(np.asarray(eps, dtype=float).reshape(3).copy())
        self.sigma.append(np.asarray(sigma, dtype=float).reshape(3).copy())
        self.eps_p.append(np.asarray(eps_p, dtype=float).reshape(3).copy())
        self.damage_t.append(float(damage_t))
        self.damage_c.append(float(damage_c))
        self.kappa.append(float(kappa))
        self.w_plastic.append(float(w_plastic))
        self.w_fract_t.append(float(w_fract_t))
        self.w_fract_c.append(float(w_fract_c))

        self.eps_p6.append(np.asarray(eps_p6, dtype=float).reshape(6).copy())
        self.eps_zz.append(float(eps_zz))
        self.kappa_t.append(float(kappa_t))
        self.kappa_c.append(float(kappa_c))
        self.cdp_w_t.append(float(cdp_w_t))
        self.cdp_eps_in_c.append(float(cdp_eps_in_c))

    def apply_to(self, target: "BulkStateArrays") -> None:
        for e, ip, eps, sig, eps_p, dt, dc, kk, wpl, wft, wfc, eps_p6, ezz, kt, kc, cwt, ceps in zip(
            self.idx_e,
            self.idx_ip,
            self.eps,
            self.sigma,
            self.eps_p,
            self.damage_t,
            self.damage_c,
            self.kappa,
            self.w_plastic,
            self.w_fract_t,
            self.w_fract_c,
            self.eps_p6,
            self.eps_zz,
            self.kappa_t,
            self.kappa_c,
            self.cdp_w_t,
            self.cdp_eps_in_c,
        ):
            target.set_values(
                int(e),
                int(ip),
                eps=eps,
                sigma=sig,
                eps_p=eps_p,
                damage_t=float(dt),
                damage_c=float(dc),
                kappa=float(kk),
                w_plastic=float(wpl),
                w_fract_t=float(wft),
                w_fract_c=float(wfc),
                eps_p6=eps_p6,
                eps_zz=float(ezz),
                kappa_t=float(kt),
                kappa_c=float(kc),
                cdp_w_t=float(cwt),
                cdp_eps_in_c=float(ceps),
            )


class BulkStateArrays:
    """Struct-of-arrays container for per-integration-point states."""

    def __init__(self, nelem: int, max_ip: int):
        self.nelem = int(nelem)
        self.max_ip = int(max_ip)

        self.active = np.zeros((self.nelem, self.max_ip), dtype=bool)
        self.eps = np.zeros((self.nelem, self.max_ip, 3), dtype=float)
        self.sigma = np.zeros((self.nelem, self.max_ip, 3), dtype=float)
        self.eps_p = np.zeros((self.nelem, self.max_ip, 3), dtype=float)
        self.damage_t = np.zeros((self.nelem, self.max_ip), dtype=float)
        self.damage_c = np.zeros((self.nelem, self.max_ip), dtype=float)
        self.kappa = np.zeros((self.nelem, self.max_ip), dtype=float)
        self.w_plastic = np.zeros((self.nelem, self.max_ip), dtype=float)
        self.w_fract_t = np.zeros((self.nelem, self.max_ip), dtype=float)
        self.w_fract_c = np.zeros((self.nelem, self.max_ip), dtype=float)

        # DP/CDP plane-stress auxiliary variables (kept as fixed arrays, not dicts)
        self.eps_p6 = np.zeros((self.nelem, self.max_ip, 6), dtype=float)
        self.eps_zz = np.zeros((self.nelem, self.max_ip), dtype=float)
        self.kappa_t = np.zeros((self.nelem, self.max_ip), dtype=float)
        self.kappa_c = np.zeros((self.nelem, self.max_ip), dtype=float)
        self.cdp_w_t = np.zeros((self.nelem, self.max_ip), dtype=float)
        self.cdp_eps_in_c = np.zeros((self.nelem, self.max_ip), dtype=float)

    @classmethod
    def zeros(cls, nelem: int, max_ip: int) -> "BulkStateArrays":
        return cls(nelem=int(nelem), max_ip=int(max_ip))

    def copy(self) -> "BulkStateArrays":
        other = BulkStateArrays(self.nelem, self.max_ip)
        for name in (
            "active",
            "eps",
            "sigma",
            "eps_p",
            "damage_t",
            "damage_c",
            "kappa",
            "w_plastic",
            "w_fract_t",
            "w_fract_c",
            "eps_p6",
            "eps_zz",
            "kappa_t",
            "kappa_c",
            "cdp_w_t",
            "cdp_eps_in_c",
        ):
            setattr(other, name, np.array(getattr(self, name), copy=True))
        return other

    def get_mp(self, e: int, ip: int) -> MaterialPoint:
        e = int(e)
        ip = int(ip)
        if (e < 0) or (e >= self.nelem) or (ip < 0) or (ip >= self.max_ip) or (not bool(self.active[e, ip])):
            return mp_default()

        mp = MaterialPoint(
            eps=np.array(self.eps[e, ip, :], copy=True),
            sigma=np.array(self.sigma[e, ip, :], copy=True),
            eps_p=np.array(self.eps_p[e, ip, :], copy=True),
            damage_t=float(self.damage_t[e, ip]),
            damage_c=float(self.damage_c[e, ip]),
            kappa=float(self.kappa[e, ip]),
            w_plastic=float(self.w_plastic[e, ip]),
            w_fract_t=float(self.w_fract_t[e, ip]),
            w_fract_c=float(self.w_fract_c[e, ip]),
        )
        mp.extra["eps_p6"] = np.array(self.eps_p6[e, ip, :], copy=True)
        mp.extra["eps_zz"] = float(self.eps_zz[e, ip])
        mp.extra["kappa_t"] = float(self.kappa_t[e, ip])
        mp.extra["kappa_c"] = float(self.kappa_c[e, ip])
        mp.extra["cdp_w_t"] = float(self.cdp_w_t[e, ip])
        mp.extra["cdp_eps_in_c"] = float(self.cdp_eps_in_c[e, ip])
        return mp

    def set_from_mp(self, e: int, ip: int, mp: MaterialPoint) -> None:
        self.set_values(
            e,
            ip,
            eps=np.asarray(mp.eps, dtype=float).reshape(3),
            sigma=np.asarray(mp.sigma, dtype=float).reshape(3),
            eps_p=np.asarray(mp.eps_p, dtype=float).reshape(3),
            damage_t=float(getattr(mp, "damage_t", 0.0)),
            damage_c=float(getattr(mp, "damage_c", 0.0)),
            kappa=float(getattr(mp, "kappa", 0.0)),
            w_plastic=float(getattr(mp, "w_plastic", 0.0)),
            w_fract_t=float(getattr(mp, "w_fract_t", 0.0)),
            w_fract_c=float(getattr(mp, "w_fract_c", 0.0)),
            eps_p6=np.asarray(mp.extra.get("eps_p6", np.zeros(6)), dtype=float).reshape(6),
            eps_zz=float(mp.extra.get("eps_zz", 0.0)),
            kappa_t=float(mp.extra.get("kappa_t", 0.0)),
            kappa_c=float(mp.extra.get("kappa_c", 0.0)),
            cdp_w_t=float(mp.extra.get("cdp_w_t", 0.0)),
            cdp_eps_in_c=float(mp.extra.get("cdp_eps_in_c", 0.0)),
        )

    def set_values(
        self,
        e: int,
        ip: int,
        *,
        eps: np.ndarray,
        sigma: np.ndarray,
        eps_p: np.ndarray,
        damage_t: float,
        damage_c: float,
        kappa: float,
        w_plastic: float,
        w_fract_t: float,
        w_fract_c: float,
        eps_p6: np.ndarray,
        eps_zz: float,
        kappa_t: float,
        kappa_c: float,
        cdp_w_t: float,
        cdp_eps_in_c: float,
    ) -> None:
        e = int(e)
        ip = int(ip)
        if (e < 0) or (e >= self.nelem) or (ip < 0) or (ip >= self.max_ip):
            return
        self.active[e, ip] = True
        self.eps[e, ip, :] = np.asarray(eps, dtype=float).reshape(3)
        self.sigma[e, ip, :] = np.asarray(sigma, dtype=float).reshape(3)
        self.eps_p[e, ip, :] = np.asarray(eps_p, dtype=float).reshape(3)
        self.damage_t[e, ip] = float(damage_t)
        self.damage_c[e, ip] = float(damage_c)
        self.kappa[e, ip] = float(kappa)
        self.w_plastic[e, ip] = float(w_plastic)
        self.w_fract_t[e, ip] = float(w_fract_t)
        self.w_fract_c[e, ip] = float(w_fract_c)

        self.eps_p6[e, ip, :] = np.asarray(eps_p6, dtype=float).reshape(6)
        self.eps_zz[e, ip] = float(eps_zz)
        self.kappa_t[e, ip] = float(kappa_t)
        self.kappa_c[e, ip] = float(kappa_c)
        self.cdp_w_t[e, ip] = float(cdp_w_t)
        self.cdp_eps_in_c[e, ip] = float(cdp_eps_in_c)


# ----------------------------------------------------------------------------
# Cohesive arrays
# ----------------------------------------------------------------------------


@dataclass
class CohesiveStatePatch:
    """Sparse set of cohesive updates."""

    idx_k: List[int]
    idx_e: List[int]
    idx_gp: List[int]
    delta_max: List[float]
    damage: List[float]

    @classmethod
    def empty(cls) -> "CohesiveStatePatch":
        return cls(idx_k=[], idx_e=[], idx_gp=[], delta_max=[], damage=[])

    def add(self, k: int, e: int, gp: int, st: CohesiveState) -> None:
        self.idx_k.append(int(k))
        self.idx_e.append(int(e))
        self.idx_gp.append(int(gp))
        self.delta_max.append(float(st.delta_max))
        self.damage.append(float(st.damage))

    def add_values(self, k: int, e: int, gp: int, *, delta_max: float, damage: float) -> None:
        self.idx_k.append(int(k))
        self.idx_e.append(int(e))
        self.idx_gp.append(int(gp))
        self.delta_max.append(float(delta_max))
        self.damage.append(float(damage))

    def apply_to(self, target: "CohesiveStateArrays") -> None:
        for k, e, gp, dm, dmg in zip(self.idx_k, self.idx_e, self.idx_gp, self.delta_max, self.damage):
            target.delta_max[int(k), int(e), int(gp)] = float(dm)
            target.damage[int(k), int(e), int(gp)] = float(dmg)


class CohesiveStateArrays:
    """Struct-of-arrays container for cohesive states.

    Shape is ``(n_primary, nelem, ngp)``.

    * Single-crack solver uses ``n_primary = 1`` and indexes crack ``k=0``.
    * Multi-crack solver uses ``n_primary = max_cracks`` and indexes ``k``.
    """

    def __init__(self, n_primary: int, nelem: int, ngp: int):
        self.n_primary = int(n_primary)
        self.nelem = int(nelem)
        self.ngp = int(ngp)
        self.delta_max = np.zeros((self.n_primary, self.nelem, self.ngp), dtype=float)
        self.damage = np.zeros((self.n_primary, self.nelem, self.ngp), dtype=float)

    @classmethod
    def zeros(cls, n_primary: int, nelem: int, ngp: int) -> "CohesiveStateArrays":
        return cls(n_primary=int(n_primary), nelem=int(nelem), ngp=int(ngp))

    def copy(self) -> "CohesiveStateArrays":
        other = CohesiveStateArrays(self.n_primary, self.nelem, self.ngp)
        other.delta_max = np.array(self.delta_max, copy=True)
        other.damage = np.array(self.damage, copy=True)
        return other

    def get_state(self, e: int, gp: int, k: int = 0) -> CohesiveState:
        return CohesiveState(
            delta_max=float(self.delta_max[int(k), int(e), int(gp)]),
            damage=float(self.damage[int(k), int(e), int(gp)]),
        )

    def get_values(self, e: int, gp: int, k: int = 0) -> Tuple[float, float]:
        return (
            float(self.delta_max[int(k), int(e), int(gp)]),
            float(self.damage[int(k), int(e), int(gp)]),
        )

