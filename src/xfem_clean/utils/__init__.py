"""Utility helpers for runners and post-processing."""

from .scaling import diagonal_equilibration, unscale_solution, check_conditioning_improvement
from .numpy_compat import trapezoid

__all__ = [
    "diagonal_equilibration",
    "unscale_solution",
    "check_conditioning_improvement",
    "trapezoid",
]
