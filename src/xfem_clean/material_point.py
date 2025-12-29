"""Material-point (integration-point) state containers.

This module introduces an explicit state object per integration point (IP).
The goal is to evolve the current elastic-lineal XFEM prototype toward
material nonlinearity (e.g., Drucker–Prager, Concrete Damaged Plasticity) with
minimal changes to the FEM/XFEM assembly and Newton loop.

The state is intentionally small and NumPy-based so it can be stored in a dict
keyed by (element_id, gp_id).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class MaterialPoint:
    """History variables at one integration point.

    The codebase uses plane-stress Voigt ordering: [xx, yy, xy].

    Fields are chosen to be a superset for future constitutive models:
    - eps_p: plastic strain (Voigt)
    - damage_t / damage_c: tension/compression damage variables in [0,1]
    - kappa: generic equivalent history variable (e.g., plastic multiplier,
      equivalent strain, etc.)

    Energy bookkeeping (all are *densities* [J/m^3]):
    - w_plastic: accumulated plastic dissipation density (bulk)
    - w_fract_t: accumulated tension cracking (damage) dissipation density
    - w_fract_c: accumulated compression crushing (damage) dissipation density
    """

    eps: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    sigma: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    eps_p: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    damage_t: float = 0.0
    damage_c: float = 0.0
    kappa: float = 0.0

    # dissipations (densities) for energy split/validation
    w_plastic: float = 0.0
    w_fract_t: float = 0.0
    w_fract_c: float = 0.0
    # free-form container for model-specific scalars/vectors
    extra: Dict[str, Any] = field(default_factory=dict)

    def copy_shallow(self) -> "MaterialPoint":
        """Copy with NumPy arrays duplicated (safe for trial/commit workflows)."""
        return MaterialPoint(
            eps=np.array(self.eps, copy=True),
            sigma=np.array(self.sigma, copy=True),
            eps_p=np.array(self.eps_p, copy=True),
            damage_t=float(self.damage_t),
            damage_c=float(self.damage_c),
            kappa=float(self.kappa),
            w_plastic=float(self.w_plastic),
            w_fract_t=float(self.w_fract_t),
            w_fract_c=float(self.w_fract_c),
            extra=dict(self.extra),
        )

    @property
    def damage(self) -> float:
        """Convenience combined damage (max of tension/compression)."""
        return float(max(self.damage_t, self.damage_c))


def mp_default() -> MaterialPoint:
    """Factory for dict.get default (keeps call sites tidy)."""
    return MaterialPoint()
