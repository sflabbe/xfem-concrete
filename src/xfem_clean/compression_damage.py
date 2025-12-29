"""Concrete compression damage model (isotropic).

This module implements the concrete compression damage model exactly as
described in the dissertation Eq. (3.44-3.46).

Key features:
- Parabolic stress-strain relation up to peak (f_c, ε_c1)
- Constant stress plateau beyond peak (no softening)
- Damage computed from secant stiffness

Reference: Dissertation 10.5445/IR/1000124842, Chapter 3, Section 3.2
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# =============================================================================
# Compression Damage Model (Eq. 3.44-3.46)
# =============================================================================

@dataclass
class ConcreteCompressionModel:
    """Concrete compression damage model (parabolic, no softening).

    Per Eq. (3.46):
        σ_c(ε) = f_c * [2*(ε/ε_c1) - (ε/ε_c1)²]  for 0 ≤ ε ≤ ε_c1
        σ_c(ε) = f_c                               for ε ≥ ε_c1

    Damage from secant stiffness (Eq. 3.44-3.45):
        E_sec(ε) = σ_c(ε) / ε
        d_c = 1 - E_sec / E_0

    Parameters
    ----------
    f_c : float
        Compressive strength (positive) [Pa]
    eps_c1 : float
        Strain at peak stress (positive) [-]
    E_0 : float
        Initial elastic modulus [Pa]

    Attributes
    ----------
    f_c : float
        Compressive strength
    eps_c1 : float
        Peak strain
    E_0 : float
        Initial modulus
    """
    f_c: float
    eps_c1: float
    E_0: float

    def __post_init__(self):
        """Validate parameters."""
        if self.f_c <= 0.0:
            raise ValueError("Compressive strength f_c must be positive")
        if self.eps_c1 <= 0.0:
            raise ValueError("Peak strain eps_c1 must be positive")
        if self.E_0 <= 0.0:
            raise ValueError("Elastic modulus E_0 must be positive")

    def sigma_epsilon_curve(self, eps: float) -> Tuple[float, float]:
        """Compute stress and tangent modulus from strain.

        Per Eq. (3.46):
            σ_c(ε) = f_c * [2*(ε/ε_c1) - (ε/ε_c1)²]  for 0 ≤ ε ≤ ε_c1
            σ_c(ε) = f_c                               for ε ≥ ε_c1

        Tangent:
            dσ/dε = f_c * [2/ε_c1 - 2*ε/ε_c1²]  for 0 ≤ ε < ε_c1
            dσ/dε = 0                            for ε ≥ ε_c1

        Parameters
        ----------
        eps : float
            Compressive strain (positive for compression)

        Returns
        -------
        sigma : float
            Compressive stress (positive)
        E_t : float
            Tangent modulus [Pa]
        """
        eps = abs(eps)  # Ensure positive

        if eps <= 0.0:
            # No compression
            return 0.0, self.E_0

        if eps <= self.eps_c1:
            # Rising parabolic branch
            ratio = eps / self.eps_c1
            sigma = self.f_c * (2.0 * ratio - ratio**2)

            # Tangent modulus
            E_t = self.f_c / self.eps_c1 * (2.0 - 2.0 * ratio)

        else:
            # Plateau (no softening)
            sigma = self.f_c
            E_t = 0.0  # Perfectly plastic plateau

        return float(sigma), float(E_t)

    def compute_damage(self, eps: float) -> float:
        """Compute damage parameter from secant stiffness.

        Per Eq. (3.44-3.45):
            E_sec = σ_c(ε) / ε
            d_c = 1 - E_sec / E_0

        Parameters
        ----------
        eps : float
            Compressive strain (positive)

        Returns
        -------
        d_c : float
            Damage parameter [0, 1] where 0 = undamaged, 1 = fully damaged
        """
        eps = abs(eps)

        if eps < 1e-14:
            # No strain, no damage
            return 0.0

        # Compute stress
        sigma, _ = self.sigma_epsilon_curve(eps)

        # Secant modulus
        E_sec = sigma / eps

        # Damage
        d_c = 1.0 - E_sec / self.E_0

        # Clamp to [0, 1]
        d_c = max(0.0, min(1.0, d_c))

        return float(d_c)


# =============================================================================
# Equivalent Compressive Strain
# =============================================================================

def compute_equivalent_compressive_strain(
    eps: np.ndarray,
) -> float:
    """Compute equivalent compressive strain from strain tensor.

    For plane stress (2D), principal strains are:
        ε_1, ε_2

    Equivalent compressive strain:
        ε_eq,c = max(0, -min(ε_1, ε_2))

    Parameters
    ----------
    eps : np.ndarray
        Strain tensor in Voigt notation [εxx, εyy, γxy]

    Returns
    -------
    eps_eq_c : float
        Equivalent compressive strain (positive)
    """
    # Compute principal strains
    eps_xx = eps[0]
    eps_yy = eps[1]
    gamma_xy = eps[2]

    # 2D principal strain formula
    eps_avg = 0.5 * (eps_xx + eps_yy)
    R = math.sqrt(((eps_xx - eps_yy) / 2.0)**2 + (gamma_xy / 2.0)**2)

    eps_1 = eps_avg + R  # Maximum principal
    eps_2 = eps_avg - R  # Minimum principal

    # Equivalent compressive strain (negative principal → compression)
    eps_eq_c = max(0.0, -min(eps_1, eps_2))

    return float(eps_eq_c)


# =============================================================================
# Stress Update with Compression Damage
# =============================================================================

def stress_update_compression_damage(
    eps: np.ndarray,
    model: ConcreteCompressionModel,
    D_0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute stress tensor with compression damage.

    Uses secant stiffness approach:
        σ = (1 - d_c) * D_0 : ε

    Parameters
    ----------
    eps : np.ndarray
        Strain tensor [εxx, εyy, γxy] (Voigt notation)
    model : ConcreteCompressionModel
        Compression model parameters
    D_0 : np.ndarray
        Undamaged elastic stiffness matrix [3, 3]

    Returns
    -------
    sigma : np.ndarray
        Stress tensor [σxx, σyy, σxy] (Voigt notation)
    D_sec : np.ndarray
        Secant stiffness matrix [3, 3]
    d_c : float
        Damage parameter
    """
    # Compute equivalent compressive strain
    eps_eq_c = compute_equivalent_compressive_strain(eps)

    # Compute damage
    d_c = model.compute_damage(eps_eq_c)

    # Degraded stiffness
    D_sec = (1.0 - d_c) * D_0

    # Stress
    sigma = np.dot(D_sec, eps)

    return sigma, D_sec, float(d_c)


# =============================================================================
# Uniaxial Compression Test
# =============================================================================

def uniaxial_compression_test(
    model: ConcreteCompressionModel,
    eps_max: float = 0.01,
    n_steps: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate uniaxial compression test.

    Useful for validation and visualization of the model.

    Parameters
    ----------
    model : ConcreteCompressionModel
    eps_max : float
        Maximum compressive strain
    n_steps : int
        Number of load steps

    Returns
    -------
    eps_history : np.ndarray
        Strain history
    sigma_history : np.ndarray
        Stress history
    damage_history : np.ndarray
        Damage history
    """
    eps_history = np.linspace(0.0, eps_max, n_steps)
    sigma_history = np.zeros(n_steps, dtype=float)
    damage_history = np.zeros(n_steps, dtype=float)

    for i, eps in enumerate(eps_history):
        sigma, _ = model.sigma_epsilon_curve(eps)
        d_c = model.compute_damage(eps)

        sigma_history[i] = sigma
        damage_history[i] = d_c

    return eps_history, sigma_history, damage_history


# =============================================================================
# Integration with Existing Constitutive Models
# =============================================================================

def combine_tension_compression_damage(
    eps: np.ndarray,
    d_t: float,
    d_c: float,
    D_0: np.ndarray,
    split_mode: str = "spectral",
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine tension and compression damage.

    Different strategies for combining d_t and d_c:
    1. "max": d = max(d_t, d_c)
    2. "product": d = 1 - (1 - d_t)(1 - d_c)
    3. "spectral": Split strain into tension/compression parts

    Parameters
    ----------
    eps : np.ndarray
        Strain tensor
    d_t : float
        Tension damage [0, 1]
    d_c : float
        Compression damage [0, 1]
    D_0 : np.ndarray
        Undamaged stiffness
    split_mode : str
        Combination mode

    Returns
    -------
    sigma : np.ndarray
        Stress tensor
    D_eff : np.ndarray
        Effective stiffness
    """
    if split_mode == "max":
        # Simple maximum
        d = max(d_t, d_c)
        D_eff = (1.0 - d) * D_0
        sigma = np.dot(D_eff, eps)

    elif split_mode == "product":
        # Energy-based combination
        d = 1.0 - (1.0 - d_t) * (1.0 - d_c)
        D_eff = (1.0 - d) * D_0
        sigma = np.dot(D_eff, eps)

    elif split_mode == "spectral":
        # Spectral decomposition (split tension/compression)
        sigma_t, sigma_c = spectral_split_stress(eps, D_0)

        sigma = (1.0 - d_t) * sigma_t + (1.0 - d_c) * sigma_c

        # Effective tangent (approximate)
        d_avg = 0.5 * (d_t + d_c)
        D_eff = (1.0 - d_avg) * D_0

    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    return sigma, D_eff


def spectral_split_stress(
    eps: np.ndarray,
    D_0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split stress into tension and compression parts via spectral decomposition.

    Based on principal strains:
        ε⁺ = <ε_i>_+ (tension)
        ε⁻ = <ε_i>_- (compression)

    Parameters
    ----------
    eps : np.ndarray
        Strain tensor [εxx, εyy, γxy]
    D_0 : np.ndarray
        Elastic stiffness

    Returns
    -------
    sigma_tension : np.ndarray
        Tension part of stress
    sigma_compression : np.ndarray
        Compression part of stress
    """
    # Simplified: use sign of principal strains
    eps_xx = eps[0]
    eps_yy = eps[1]
    gamma_xy = eps[2]

    # Principal strains
    eps_avg = 0.5 * (eps_xx + eps_yy)
    R = math.sqrt(((eps_xx - eps_yy) / 2.0)**2 + (gamma_xy / 2.0)**2)

    eps_1 = eps_avg + R
    eps_2 = eps_avg - R

    # Macaulay brackets
    eps_1_plus = max(0.0, eps_1)
    eps_1_minus = min(0.0, eps_1)

    eps_2_plus = max(0.0, eps_2)
    eps_2_minus = min(0.0, eps_2)

    # Reconstruct tensors (simplified for isotropic case)
    # Full version would use eigenvector projection

    # Approximation: split based on trace and deviatoric parts
    eps_vol = eps_xx + eps_yy
    eps_vol_plus = max(0.0, eps_vol)
    eps_vol_minus = min(0.0, eps_vol)

    # Deviatoric (split proportionally)
    factor_plus = (eps_1_plus + eps_2_plus) / (abs(eps_1) + abs(eps_2) + 1e-14)
    factor_minus = abs(eps_1_minus + eps_2_minus) / (abs(eps_1) + abs(eps_2) + 1e-14)

    eps_plus = eps * factor_plus
    eps_minus = eps * factor_minus

    # Stresses
    sigma_tension = np.dot(D_0, eps_plus)
    sigma_compression = np.dot(D_0, eps_minus)

    return sigma_tension, sigma_compression


# =============================================================================
# Utilities
# =============================================================================

def get_default_compression_model(f_c_mpa: float = 30.0) -> ConcreteCompressionModel:
    """Get default compression model from compressive strength.

    Uses Model Code 2010 relations:
        ε_c1 = 0.7 * f_cm^0.31 / 1000
        E_0 = 21500 * (f_cm / 10)^(1/3)

    Parameters
    ----------
    f_c_mpa : float
        Compressive strength [MPa]

    Returns
    -------
    model : ConcreteCompressionModel
    """
    # Model Code 2010 formulas
    eps_c1 = 0.7 * (f_c_mpa ** 0.31) / 1000.0  # Peak strain

    # Elastic modulus
    E_0_mpa = 21500.0 * ((f_c_mpa / 10.0) ** (1.0/3.0))

    # Convert to Pa
    f_c = f_c_mpa * 1e6
    E_0 = E_0_mpa * 1e6

    model = ConcreteCompressionModel(
        f_c=f_c,
        eps_c1=eps_c1,
        E_0=E_0,
    )

    return model
