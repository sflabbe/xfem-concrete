"""Hook for applying compression damage to bulk stress states.

This module provides utilities to integrate compression damage
with existing constitutive models.

Reference: Dissertation 10.5445/IR/1000124842, Chapter 3, Section 3.2
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

from xfem_clean.compression_damage import (
    ConcreteCompressionModel,
    stress_update_compression_damage,
)


def apply_compression_damage_degradation(
    eps: np.ndarray,
    sigma_undamaged: np.ndarray,
    D_undamaged: np.ndarray,
    compression_model: ConcreteCompressionModel,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Apply compression damage degradation to stress state.

    This function can be used as a post-processor after the primary
    constitutive model (elastic, DP, CDP) to add compression damage.

    Parameters
    ----------
    eps : np.ndarray
        Strain tensor [εxx, εyy, γxy]
    sigma_undamaged : np.ndarray
        Stress from primary constitutive model [σxx, σyy, σxy]
    D_undamaged : np.ndarray
        Tangent stiffness from primary model [3, 3]
    compression_model : ConcreteCompressionModel
        Compression damage model parameters

    Returns
    -------
    sigma_damaged : np.ndarray
        Degraded stress tensor
    D_damaged : np.ndarray
        Degraded tangent stiffness
    d_c : float
        Compression damage parameter [0, 1]
    """
    # Compute compression damage using secant approach
    sigma_c, D_c, d_c = stress_update_compression_damage(
        eps=eps,
        model=compression_model,
        D_0=D_undamaged,
    )

    # The compression damage model returns the compression-degraded state
    # For combined models (e.g., CDP + compression), more sophisticated
    # coupling is needed. For now, use simple degradation factor.

    # Simple approach: use compression damage to degrade the undamaged state
    sigma_damaged = (1.0 - d_c) * sigma_undamaged
    D_damaged = (1.0 - d_c) * D_undamaged

    return sigma_damaged, D_damaged, d_c


def compression_damage_stress_update(
    eps: np.ndarray,
    compression_model: ConcreteCompressionModel,
    E: float,
    nu: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Pure compression damage stress update (elastic + compression damage).

    Use this for pure elastic + compression damage (no tension damage).

    Parameters
    ----------
    eps : np.ndarray
        Strain tensor [εxx, εyy, γxy]
    compression_model : ConcreteCompressionModel
        Compression model
    E : float
        Young's modulus [Pa]
    nu : float
        Poisson's ratio

    Returns
    -------
    sigma : np.ndarray
        Stress tensor with compression damage
    D_sec : np.ndarray
        Secant stiffness
    d_c : float
        Compression damage
    """
    # Build undamaged elastic stiffness
    c = E / (1.0 - nu * nu)
    D_0 = np.array([
        [c, c * nu, 0.0],
        [c * nu, c, 0.0],
        [0.0, 0.0, c * (1.0 - nu) / 2.0]
    ], dtype=float)

    # Apply compression damage
    sigma, D_sec, d_c = stress_update_compression_damage(
        eps=eps,
        model=compression_model,
        D_0=D_0,
    )

    return sigma, D_sec, d_c
