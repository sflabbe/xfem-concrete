"""Newton convergence helpers (stable rules)."""

from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class NewtonConvergence:
    newton_tol_r: float = 1e-6
    newton_beta: float = 1e-9
    newton_tol_du: float = 1e-14

    def residual_tolerance(self, reaction_estimate: float) -> float:
        fscale = max(1.0, abs(float(reaction_estimate)))
        return float(self.newton_tol_r + self.newton_beta * fscale)

    def residual_converged(self, norm_rhs: float, reaction_estimate: float) -> bool:
        return float(norm_rhs) < self.residual_tolerance(reaction_estimate)

    def stagnated(self, norm_du: float) -> bool:
        return float(norm_du) < float(self.newton_tol_du)
