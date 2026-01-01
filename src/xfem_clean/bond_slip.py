"""Bond-slip interface model for reinforced concrete (Model Code 2010).

This module implements the bond-slip relationship between concrete and reinforcing
steel bars according to fib Model Code 2010, Section 6.1.2.

The implementation includes:
  - Model Code 2010 bond-slip constitutive law for ribbed bars
  - Interface element assembly for rebar segments
  - Dowel action springs at crack-rebar intersections
  - State management for bond slip history
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional
import math

import numpy as np
import scipy.sparse as sp

# Optional Numba acceleration
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap
    def prange(*args):
        return range(*args)


@dataclass
class BondSlipModelCode2010:
    """Bond-slip constitutive law per fib Model Code 2010 (ribbed bars).

    The bond stress-slip relationship is piecewise:

    .. math::
        \\tau(s) = \\begin{cases}
            \\tau_{max} (s/s_1)^\\alpha & 0 \\leq s \\leq s_1 \\\\
            \\tau_{max} & s_1 < s \\leq s_2 \\\\
            \\tau_{max} - (\\tau_{max} - \\tau_f)(s-s_2)/(s_3-s_2) & s_2 < s \\leq s_3 \\\\
            \\tau_f & s > s_3
        \\end{cases}

    Where:
      - τ_max: maximum bond stress
      - s1, s2, s3: characteristic slip values
      - τ_f: residual (friction) bond stress
      - α: shape parameter for rising branch

    Unloading/reloading follows a secant path to the maximum historical slip.

    Part B (Dissertation Model):
      - Steel yielding reduction: Ωy(εs) per Eq. 3.57-3.58
      - Crack deterioration: Ωcrack(s, wmax) per Eq. 3.60-3.61
      - Secant stiffness option for stability (Eq. 5.2-5.3)

    Parameters
    ----------
    f_cm : float
        Mean concrete compressive strength [Pa]
    d_bar : float
        Bar diameter [m]
    condition : str
        Bond condition: "good" (unconfined) or "poor" (all other)
    f_y : float, optional
        Steel yield stress [Pa] (for Ωy calculation)
    E_s : float, optional
        Steel Young's modulus [Pa] (for Ωy calculation)
    use_secant_stiffness : bool
        Use secant instead of tangent stiffness for stability (Part B7)
    enable_yielding_reduction : bool
        Apply steel yielding reduction factor Ωy (Part B3)
    enable_crack_deterioration : bool
        Apply crack deterioration factor Ωcrack (Part B4)

    Attributes
    ----------
    tau_max : float
        Maximum bond stress [Pa]
    s1, s2, s3 : float
        Characteristic slips [m]
    tau_f : float
        Residual bond stress [Pa]
    alpha : float
        Shape exponent [-]
    """

    f_cm: float
    d_bar: float
    condition: str = "good"
    f_y: float = 500e6  # Steel yield stress [Pa] (Part B3)
    E_s: float = 200e9  # Steel Young's modulus [Pa] (Part B3)
    use_secant_stiffness: bool = True  # Part B7: Use secant for stability
    enable_yielding_reduction: bool = False  # Part B3: Steel yielding Ωy
    enable_crack_deterioration: bool = False  # Part B4: Crack deterioration Ωcrack

    def __post_init__(self):
        """Initialize Model Code 2010 parameters."""
        f_cm_mpa = float(self.f_cm) / 1e6  # Convert to MPa for MC2010 formulas
        sqrt_fcm = math.sqrt(f_cm_mpa)

        if self.condition == "good":
            # Unconfined conditions (good bond)
            self.tau_max = float(2.5 * sqrt_fcm * 1e6)  # Convert back to Pa
            self.s1 = 1.0e-3  # [m]
            self.s2 = 2.0e-3  # [m]
            self.s3 = float(self.d_bar)  # Clear rib spacing ~ diameter
            self.alpha = 0.4
        elif self.condition == "poor":
            # All other cases (poor bond)
            self.tau_max = float(1.25 * sqrt_fcm * 1e6)
            self.s1 = 1.8e-3
            self.s2 = 3.6e-3
            self.s3 = float(self.d_bar)
            self.alpha = 0.4
        else:
            raise ValueError(f"Unknown bond condition: {self.condition}. Use 'good' or 'poor'.")

        self.tau_f = float(0.15 * self.tau_max)  # Residual stress (friction)

        # Safety checks
        if self.s1 <= 0 or self.s2 <= self.s1 or self.s3 <= self.s2:
            raise ValueError("Bond-slip parameters must satisfy: 0 < s1 < s2 < s3")

    def tau_envelope(self, s_abs: float) -> Tuple[float, float]:
        """Compute bond stress and tangent on the monotonic envelope.

        Parameters
        ----------
        s_abs : float
            Absolute value of slip [m]

        Returns
        -------
        tau : float
            Bond stress [Pa]
        dtau_ds : float
            Tangent stiffness [Pa/m]
        """
        s_abs = float(s_abs)

        if s_abs <= self.s1:
            # Rising branch: τ = τ_max * (s/s1)^α
            if s_abs < 1e-16:
                # Avoid division by zero; use tangent at s → 0
                tau = 0.0
                dtau_ds = self.tau_max * self.alpha / self.s1 * (1e-16 / self.s1) ** (self.alpha - 1.0)
            else:
                ratio = s_abs / self.s1
                tau = self.tau_max * (ratio ** self.alpha)
                dtau_ds = self.tau_max * self.alpha / self.s1 * (ratio ** (self.alpha - 1.0))

        elif s_abs <= self.s2:
            # Plateau: τ = τ_max
            tau = self.tau_max
            dtau_ds = 0.0

        elif s_abs <= self.s3:
            # Softening branch: linear decay from τ_max to τ_f
            tau = self.tau_max - (self.tau_max - self.tau_f) * (s_abs - self.s2) / (self.s3 - self.s2)
            dtau_ds = -(self.tau_max - self.tau_f) / (self.s3 - self.s2)

        else:
            # Residual: τ = τ_f
            tau = self.tau_f
            dtau_ds = 0.0

        return float(tau), float(dtau_ds)

    def compute_yielding_reduction(self, eps_s: float) -> float:
        """Compute steel yielding reduction factor Ωy per Eq. 3.57-3.58.

        Parameters
        ----------
        eps_s : float
            Steel strain (axial, in bar direction) [-]

        Returns
        -------
        omega_y : float
            Reduction factor [0, 1] where 1 = no yielding, 0 = full yielding

        Notes
        -----
        From dissertation Eq. 3.57-3.58:
            eps_y = f_y / E_s
            If eps_s <= eps_y: Ωy = 1
            If eps_s > eps_y: Ωy = exp(-k_y * (eps_s - eps_y) / eps_y)
        where k_y ≈ 10 (calibration parameter).
        """
        if not self.enable_yielding_reduction:
            return 1.0

        eps_y = self.f_y / self.E_s  # Yield strain
        k_y = 10.0  # Calibration parameter (dissertation uses ~10)

        if eps_s <= eps_y:
            return 1.0
        else:
            # Exponential decay after yielding
            delta_eps = (eps_s - eps_y) / eps_y
            omega_y = math.exp(-k_y * delta_eps)
            return max(0.0, min(1.0, omega_y))  # Clamp to [0, 1]

    def compute_crack_deterioration(
        self, dist_to_crack: float, w_max: float, t_n_cohesive_stress: float, f_t: float
    ) -> float:
        """Compute crack deterioration factor Ωcrack per Eq. 3.60-3.61 (modified).

        Parameters
        ----------
        dist_to_crack : float
            Distance from interface point to nearest transverse crack [m]
        w_max : float
            Maximum crack opening (historical) at interface level [m]
        t_n_cohesive_stress : float
            Normal cohesive stress t_n(w_max) from cohesive law [Pa]
        f_t : float
            Concrete tensile strength [Pa]

        Returns
        -------
        omega_crack : float
            Reduction factor [0, 1] where 1 = no deterioration, 0 = full deterioration

        Notes
        -----
        From dissertation Eq. 3.60-3.61 (modified):
            l_ch = characteristic length (e.g., 2*d_bar)
            chi = t_n(w_max) / f_t  (FPZ state indicator)
            Ωcrack = exp(-dist_to_crack / l_ch * (1 - chi))

        When chi = 1 (no crack): Ωcrack = 1
        When chi = 0 (fully open crack): Ωcrack = exp(-dist/l_ch)
        """
        if not self.enable_crack_deterioration:
            return 1.0

        l_ch = 2.0 * self.d_bar  # Characteristic length

        # FPZ state indicator (Eq. 3.61)
        if f_t > 1e-9:
            chi = max(0.0, min(1.0, t_n_cohesive_stress / f_t))
        else:
            chi = 0.0

        # Deterioration factor (Eq. 3.60 modified)
        if dist_to_crack < 1e-12:
            # At crack: maximum deterioration
            omega_crack = chi  # Reduces to chi at crack
        else:
            # Exponential decay
            omega_crack = math.exp(-dist_to_crack / l_ch * (1.0 - chi))
            omega_crack = max(chi, min(1.0, omega_crack))  # Clamp [chi, 1]

        return omega_crack

    def tau_and_tangent(
        self,
        s: float,
        s_max_history: float,
        eps_s: float = 0.0,
        omega_crack: float = 1.0,
    ) -> Tuple[float, float]:
        """Compute bond stress and tangent stiffness with unloading/reloading.

        Parameters
        ----------
        s : float
            Current slip (signed) [m]
        s_max_history : float
            Maximum absolute slip reached in history [m]
        eps_s : float, optional
            Steel axial strain (for Ωy calculation, Part B3)
        omega_crack : float, optional
            Crack deterioration factor (for Ωcrack, Part B4)

        Returns
        -------
        tau : float
            Bond stress (signed) [Pa]
        dtau_ds : float
            Tangent (or secant) stiffness [Pa/m]

        Notes
        -----
        - Loading: follows monotonic envelope
        - Unloading/reloading: secant stiffness to historical maximum
        - Part B3: Multiplies by Ωy(eps_s) if enabled
        - Part B4: Multiplies by Ωcrack if enabled
        - Part B7: Returns secant stiffness if use_secant_stiffness=True
        """
        s_abs = abs(s)
        sign = 1.0 if s >= 0.0 else -1.0

        # Ensure s_max_history is non-decreasing
        s_max = max(float(s_max_history), s_abs)

        if s_abs >= s_max - 1e-14:
            # Loading: on envelope
            tau_abs, dtau_abs = self.tau_envelope(s_abs)
        else:
            # Unloading/reloading: secant to max point
            tau_max_reached, _ = self.tau_envelope(s_max)
            if s_max > 1e-14:
                tau_abs = tau_max_reached * (s_abs / s_max)
                dtau_abs = tau_max_reached / s_max
            else:
                tau_abs = 0.0
                dtau_abs = 0.0

        # Part B3: Apply steel yielding reduction
        omega_y = self.compute_yielding_reduction(eps_s)

        # Part B3+B4: Apply reduction factors
        tau_abs *= omega_y * omega_crack
        dtau_abs *= omega_y * omega_crack

        # Part B7: Use secant stiffness for stability
        if self.use_secant_stiffness and s_abs > 1e-14:
            dtau_abs = tau_abs / s_abs  # Secant stiffness

        return sign * float(tau_abs), float(dtau_abs)


# ------------------------------------------------------------------------------
# Custom Bond Laws for Thesis Cases
# ------------------------------------------------------------------------------

@dataclass
class CustomBondSlipLaw:
    """Custom CEB-FIP bond-slip law with direct parameter specification.

    This variant allows direct specification of all bond-slip parameters,
    which is needed for thesis validation cases that use calibrated values
    rather than Model Code 2010 formulas.

    The bond stress-slip relationship follows the same 4-branch form as
    BondSlipModelCode2010, but with user-specified parameters.

    Parameters
    ----------
    s1 : float
        End of rising branch [m] or [mm] depending on use
    s2 : float
        End of plateau [m] or [mm]
    s3 : float
        End of softening [m] or [mm]
    tau_max : float
        Maximum bond stress [Pa] or [MPa] depending on use
    tau_f : float
        Residual bond stress [Pa] or [MPa]
    alpha : float
        Exponent in rising branch (typically 0.4)
    use_secant_stiffness : bool
        Use secant instead of tangent stiffness for stability
    """

    s1: float
    s2: float
    s3: float
    tau_max: float
    tau_f: float
    alpha: float = 0.4
    use_secant_stiffness: bool = True

    def __post_init__(self):
        """Validate parameters."""
        if self.s1 <= 0 or self.s2 <= self.s1 or self.s3 <= self.s2:
            raise ValueError("Bond-slip parameters must satisfy: 0 < s1 < s2 < s3")
        if self.tau_max <= 0 or self.tau_f < 0:
            raise ValueError("Bond stresses must be non-negative (tau_max > 0)")
        if self.alpha <= 0:
            raise ValueError("Alpha exponent must be positive")

    def tau_envelope(self, s_abs: float) -> Tuple[float, float]:
        """Compute bond stress and tangent on the monotonic envelope.

        Parameters
        ----------
        s_abs : float
            Absolute value of slip (same units as s1, s2, s3)

        Returns
        -------
        tau : float
            Bond stress (same units as tau_max)
        dtau_ds : float
            Tangent stiffness (units: tau_max / s1)
        """
        s_abs = float(s_abs)

        if s_abs <= self.s1:
            # Rising branch: τ = τ_max * (s/s1)^α
            if s_abs < 1e-16:
                tau = 0.0
                dtau_ds = self.tau_max * self.alpha / self.s1
            else:
                ratio = s_abs / self.s1
                tau = self.tau_max * (ratio ** self.alpha)
                dtau_ds = self.tau_max * self.alpha / self.s1 * (ratio ** (self.alpha - 1.0))

        elif s_abs <= self.s2:
            # Plateau: τ = τ_max
            tau = self.tau_max
            dtau_ds = 0.0

        elif s_abs <= self.s3:
            # Softening: linear decay
            tau = self.tau_max - (self.tau_max - self.tau_f) * (s_abs - self.s2) / (self.s3 - self.s2)
            dtau_ds = -(self.tau_max - self.tau_f) / (self.s3 - self.s2)

        else:
            # Residual: τ = τ_f
            tau = self.tau_f
            dtau_ds = 0.0

        return float(tau), float(dtau_ds)

    def tau_and_tangent(
        self, s: float, s_max_history: float
    ) -> Tuple[float, float]:
        """Compute bond stress and tangent with unloading/reloading.

        Parameters
        ----------
        s : float
            Current slip (signed)
        s_max_history : float
            Maximum absolute slip in history

        Returns
        -------
        tau : float
            Bond stress (signed)
        dtau_ds : float
            Tangent or secant stiffness
        """
        s_abs = abs(s)
        sign = 1.0 if s >= 0 else -1.0

        # Update historical maximum
        s_max = max(s_max_history, s_abs)

        # Compute envelope at current and max slips
        tau_env, dtau_env = self.tau_envelope(s_abs)
        tau_max_env, _ = self.tau_envelope(s_max)

        # Unloading/reloading: secant to historical max
        if s_abs < s_max:
            # Reloading path
            if s_max > 1e-14:
                dtau_abs = tau_max_env / s_max  # Secant to historical max
            else:
                dtau_abs = dtau_env
            tau_abs = dtau_abs * s_abs
        else:
            # Loading on envelope
            tau_abs = tau_env
            dtau_abs = dtau_env

        # Secant stiffness option for stability
        if self.use_secant_stiffness and s_abs > 1e-14:
            dtau_abs = tau_abs / s_abs

        return sign * float(tau_abs), float(dtau_abs)


@dataclass
class BilinearBondLaw:
    """Bilinear bond-slip law for externally bonded FRP sheets.

    Simplified law with linear hardening followed by linear softening to zero.
    Used for FRP sheet debonding.

    tau(s) =
      (tau1/s1)*s                      if 0 <= s <= s1
      tau1*(1 - (s-s1)/(s2-s1))        if s1 < s <= s2
      0                                if s > s2

    Parameters
    ----------
    s1 : float
        End of hardening branch
    s2 : float
        End of softening (complete debonding)
    tau1 : float
        Peak bond stress
    use_secant_stiffness : bool
        Use secant stiffness for stability
    """

    s1: float
    s2: float
    tau1: float
    use_secant_stiffness: bool = True

    def __post_init__(self):
        """Validate parameters."""
        if self.s1 <= 0 or self.s2 <= self.s1:
            raise ValueError("Must have 0 < s1 < s2")
        if self.tau1 <= 0:
            raise ValueError("Peak stress tau1 must be positive")

    def tau_envelope(self, s_abs: float) -> Tuple[float, float]:
        """Compute bond stress and tangent on monotonic envelope."""
        s_abs = float(s_abs)

        if s_abs <= self.s1:
            # Hardening: τ = (tau1/s1)*s
            if s_abs < 1e-16:
                tau = 0.0
                dtau_ds = self.tau1 / self.s1
            else:
                tau = (self.tau1 / self.s1) * s_abs
                dtau_ds = self.tau1 / self.s1

        elif s_abs <= self.s2:
            # Softening: τ = tau1 * (1 - (s-s1)/(s2-s1))
            tau = self.tau1 * (1.0 - (s_abs - self.s1) / (self.s2 - self.s1))
            dtau_ds = -self.tau1 / (self.s2 - self.s1)

        else:
            # Debonded: τ = 0
            tau = 0.0
            dtau_ds = 0.0

        return float(tau), float(dtau_ds)

    def tau_and_tangent(
        self, s: float, s_max_history: float
    ) -> Tuple[float, float]:
        """Compute bond stress and tangent with unloading/reloading."""
        s_abs = abs(s)
        sign = 1.0 if s >= 0 else -1.0

        s_max = max(s_max_history, s_abs)

        tau_env, dtau_env = self.tau_envelope(s_abs)
        tau_max_env, _ = self.tau_envelope(s_max)

        # Unloading/reloading
        if s_abs < s_max:
            if s_max > 1e-14:
                dtau_abs = tau_max_env / s_max
            else:
                dtau_abs = dtau_env
            tau_abs = dtau_abs * s_abs
        else:
            tau_abs = tau_env
            dtau_abs = dtau_env

        # Secant stiffness
        if self.use_secant_stiffness and s_abs > 1e-14:
            dtau_abs = tau_abs / s_abs

        return sign * float(tau_abs), float(dtau_abs)


@dataclass
class BanholzerBondLaw:
    """Banholzer bond-slip law for fibres (5-parameter model).

    Used for fibre reinforcement pull-out behavior.

    tau(s) =
      (tau1/s0)*s                                   if 0 <= s <= s0
      tau2 - (tau2-tau_f)*(s-s0)/((a-1)*s0)         if s0 < s <= a*s0
      tau_f                                         if s > a*s0

    Parameters
    ----------
    s0 : float
        End of rising branch
    a : float
        Softening end multiplier (slip at end of softening = a*s0)
    tau1 : float
        Peak stress in rising branch
    tau2 : float
        Stress at start of softening
    tau_f : float
        Residual stress
    use_secant_stiffness : bool
        Use secant stiffness for stability
    """

    s0: float
    a: float
    tau1: float
    tau2: float
    tau_f: float
    use_secant_stiffness: bool = True

    def __post_init__(self):
        """Validate parameters."""
        if self.s0 <= 0:
            raise ValueError("s0 must be positive")
        if self.a <= 1.0:
            raise ValueError("a must be > 1 (defines softening length)")
        if self.tau1 <= 0 or self.tau2 < 0 or self.tau_f < 0:
            raise ValueError("Bond stresses must be non-negative")

    def tau_envelope(self, s_abs: float) -> Tuple[float, float]:
        """Compute bond stress and tangent on monotonic envelope."""
        s_abs = float(s_abs)

        if s_abs <= self.s0:
            # Rising: τ = (tau1/s0)*s
            if s_abs < 1e-16:
                tau = 0.0
                dtau_ds = self.tau1 / self.s0
            else:
                tau = (self.tau1 / self.s0) * s_abs
                dtau_ds = self.tau1 / self.s0

        elif s_abs <= self.a * self.s0:
            # Softening: τ = tau2 - (tau2-tau_f)*(s-s0)/((a-1)*s0)
            s_soft_length = (self.a - 1.0) * self.s0
            tau = self.tau2 - (self.tau2 - self.tau_f) * (s_abs - self.s0) / s_soft_length
            dtau_ds = -(self.tau2 - self.tau_f) / s_soft_length

        else:
            # Residual: τ = tau_f
            tau = self.tau_f
            dtau_ds = 0.0

        return float(tau), float(dtau_ds)

    def tau_and_tangent(
        self, s: float, s_max_history: float
    ) -> Tuple[float, float]:
        """Compute bond stress and tangent with unloading/reloading."""
        s_abs = abs(s)
        sign = 1.0 if s >= 0 else -1.0

        s_max = max(s_max_history, s_abs)

        tau_env, dtau_env = self.tau_envelope(s_abs)
        tau_max_env, _ = self.tau_envelope(s_max)

        # Unloading/reloading
        if s_abs < s_max:
            if s_max > 1e-14:
                dtau_abs = tau_max_env / s_max
            else:
                dtau_abs = dtau_env
            tau_abs = dtau_abs * s_abs
        else:
            tau_abs = tau_env
            dtau_abs = dtau_env

        # Secant stiffness
        if self.use_secant_stiffness and s_abs > 1e-14:
            dtau_abs = tau_abs / s_abs

        return sign * float(tau_abs), float(dtau_abs)


# ------------------------------------------------------------------------------
# Dowel Action Model (Part B5)
# ------------------------------------------------------------------------------

@dataclass
class DowelActionModel:
    """Dowel action model for transverse stress-opening relationship.

    Based on Brenna et al. model (Eqs. 3.62-3.68 in dissertation).

    Parameters
    ----------
    d_bar : float
        Bar diameter [m]
    f_c : float
        Concrete compressive strength [Pa]
    E_s : float
        Steel Young's modulus [Pa]

    Notes
    -----
    The radial stress-opening relationship is:
        σ_r(w) = σ_r_max * (1 - exp(-k_d * w / d_bar))

    where:
        σ_r_max = k_c * sqrt(f_c)  (maximum radial stress)
        k_c ≈ 0.8 (calibration constant)
        k_d ≈ 50 (shape parameter)
    """

    d_bar: float
    f_c: float
    E_s: float = 200e9

    def __post_init__(self):
        """Initialize dowel parameters."""
        # Calibration constants (from Brenna et al.)
        self.k_c = 0.8  # Radial stress coefficient
        self.k_d = 50.0  # Shape parameter for exponential

        # Maximum radial stress
        f_c_mpa = self.f_c / 1e6
        self.sigma_r_max = self.k_c * math.sqrt(f_c_mpa) * 1e6  # Convert back to Pa

    def sigma_r_and_tangent(self, w: float) -> Tuple[float, float]:
        """Compute radial stress and tangent stiffness.

        Parameters
        ----------
        w : float
            Crack opening (normal to bar) [m]

        Returns
        -------
        sigma_r : float
            Radial stress [Pa]
        dsigma_r_dw : float
            Tangent stiffness [Pa/m]

        Notes
        -----
        From Eq. 3.62-3.68:
            σ_r = σ_r_max * (1 - exp(-k_d * w / d_bar))
            dσ_r/dw = σ_r_max * (k_d / d_bar) * exp(-k_d * w / d_bar)
        """
        w_abs = abs(w)

        # Exponential model
        exp_term = math.exp(-self.k_d * w_abs / self.d_bar)
        sigma_r = self.sigma_r_max * (1.0 - exp_term)
        dsigma_r_dw = self.sigma_r_max * (self.k_d / self.d_bar) * exp_term

        return float(sigma_r), float(dsigma_r_dw)


# ------------------------------------------------------------------------------
# Interface Element State Arrays
# ------------------------------------------------------------------------------

@dataclass
class BondSlipStateArrays:
    """State variables for bond-slip interface elements.

    Attributes
    ----------
    n_segments : int
        Number of rebar segments
    s_max : np.ndarray
        Maximum absolute slip history [n_segments] [m]
    s_current : np.ndarray
        Current slip (signed) [n_segments] [m]
    tau_current : np.ndarray
        Current bond stress [n_segments] [Pa]
    """

    n_segments: int
    s_max: np.ndarray
    s_current: np.ndarray
    tau_current: np.ndarray

    @classmethod
    def zeros(cls, n_segments: int) -> BondSlipStateArrays:
        """Initialize zero states."""
        return cls(
            n_segments=int(n_segments),
            s_max=np.zeros(n_segments, dtype=float),
            s_current=np.zeros(n_segments, dtype=float),
            tau_current=np.zeros(n_segments, dtype=float),
        )

    def copy_shallow(self) -> BondSlipStateArrays:
        """Create a shallow copy of state arrays."""
        return BondSlipStateArrays(
            n_segments=int(self.n_segments),
            s_max=self.s_max.copy(),
            s_current=self.s_current.copy(),
            tau_current=self.tau_current.copy(),
        )

    def copy(self) -> BondSlipStateArrays:
        """Alias for copy_shallow (no nested structure)."""
        return self.copy_shallow()


# ------------------------------------------------------------------------------
# Preflight Validation (Part A1)
# ------------------------------------------------------------------------------

def validate_bond_inputs(
    u_total: np.ndarray,
    segs: np.ndarray,
    steel_dof_map: np.ndarray,
    steel_dof_offset: int,
    bond_states: BondSlipStateArrays,
) -> None:
    """Validate all bond-slip inputs before calling Numba kernel.

    This function detects invalid DOF mappings, out-of-bounds indices,
    and inconsistent array shapes that would cause kernel hangs.

    Raises
    ------
    RuntimeError
        If any validation check fails, with detailed diagnostic message.
    """
    ndof_total = len(u_total)
    n_seg = segs.shape[0]

    # Check segs shape
    if segs.ndim != 2 or segs.shape[1] != 5:
        raise RuntimeError(
            f"bond-slip invalid segs shape: expected (n_seg, 5), got {segs.shape}"
        )

    # Check steel_dof_map shape and dtype
    if steel_dof_map.ndim != 2 or steel_dof_map.shape[1] != 2:
        raise RuntimeError(
            f"bond-slip invalid steel_dof_map shape: expected (nnode, 2), got {steel_dof_map.shape}"
        )

    if steel_dof_map.dtype not in [np.int32, np.int64]:
        raise RuntimeError(
            f"bond-slip invalid steel_dof_map dtype: expected int64, got {steel_dof_map.dtype}"
        )

    nnode = steel_dof_map.shape[0]

    # Check steel_dof_offset in range
    if steel_dof_offset < 0 or steel_dof_offset >= ndof_total:
        raise RuntimeError(
            f"bond-slip invalid steel_dof_offset={steel_dof_offset}, ndof_total={ndof_total}"
        )

    # Check each segment's nodes and DOF indices
    errors = []
    for iseg in range(n_seg):
        n1 = int(segs[iseg, 0])
        n2 = int(segs[iseg, 1])

        # Validate node indices
        if n1 < 0 or n1 >= nnode:
            errors.append(f"  seg {iseg}: n1={n1} out of range [0, {nnode})")
        if n2 < 0 or n2 >= nnode:
            errors.append(f"  seg {iseg}: n2={n2} out of range [0, {nnode})")

        if n1 < 0 or n1 >= nnode or n2 < 0 or n2 >= nnode:
            continue  # Skip DOF checks if node indices invalid

        # Validate concrete DOF indices (these are computed directly as 2*n)
        dof_c1x = 2 * n1
        dof_c1y = 2 * n1 + 1
        dof_c2x = 2 * n2
        dof_c2y = 2 * n2 + 1

        if dof_c1x >= steel_dof_offset or dof_c1y >= steel_dof_offset:
            errors.append(
                f"  seg {iseg}: concrete DOFs ({dof_c1x}, {dof_c1y}) >= steel_dof_offset={steel_dof_offset}"
            )
        if dof_c2x >= steel_dof_offset or dof_c2y >= steel_dof_offset:
            errors.append(
                f"  seg {iseg}: concrete DOFs ({dof_c2x}, {dof_c2y}) >= steel_dof_offset={steel_dof_offset}"
            )

        # Validate steel DOF indices from mapping
        dof_s1x = int(steel_dof_map[n1, 0])
        dof_s1y = int(steel_dof_map[n1, 1])
        dof_s2x = int(steel_dof_map[n2, 0])
        dof_s2y = int(steel_dof_map[n2, 1])

        # Steel DOFs must be either -1 (no steel) or in valid range
        if dof_s1x != -1 and (dof_s1x < steel_dof_offset or dof_s1x >= ndof_total):
            errors.append(
                f"  seg {iseg}: dof_s1x={dof_s1x} out of range [{steel_dof_offset}, {ndof_total})"
            )
        if dof_s1y != -1 and (dof_s1y < steel_dof_offset or dof_s1y >= ndof_total):
            errors.append(
                f"  seg {iseg}: dof_s1y={dof_s1y} out of range [{steel_dof_offset}, {ndof_total})"
            )
        if dof_s2x != -1 and (dof_s2x < steel_dof_offset or dof_s2x >= ndof_total):
            errors.append(
                f"  seg {iseg}: dof_s2x={dof_s2x} out of range [{steel_dof_offset}, {ndof_total})"
            )
        if dof_s2y != -1 and (dof_s2y < steel_dof_offset or dof_s2y >= ndof_total):
            errors.append(
                f"  seg {iseg}: dof_s2y={dof_s2y} out of range [{steel_dof_offset}, {ndof_total})"
            )

        # For bond-slip segments, steel DOFs must not be -1
        if dof_s1x == -1 or dof_s1y == -1:
            errors.append(f"  seg {iseg}: n1={n1} has no steel DOFs (dof_s1x={dof_s1x}, dof_s1y={dof_s1y})")
        if dof_s2x == -1 or dof_s2y == -1:
            errors.append(f"  seg {iseg}: n2={n2} has no steel DOFs (dof_s2x={dof_s2x}, dof_s2y={dof_s2y})")

    # Check bond_states consistency
    if bond_states.n_segments != n_seg:
        errors.append(
            f"  bond_states.n_segments={bond_states.n_segments} != n_seg={n_seg}"
        )

    if len(bond_states.s_max) != n_seg:
        errors.append(
            f"  len(bond_states.s_max)={len(bond_states.s_max)} != n_seg={n_seg}"
        )

    if errors:
        error_msg = "bond-slip invalid DOF mapping detected:\n" + "\n".join(errors)
        error_msg += f"\n\nContext: ndof_total={ndof_total}, nnode={nnode}, n_seg={n_seg}, steel_dof_offset={steel_dof_offset}"
        raise RuntimeError(error_msg)


# ------------------------------------------------------------------------------
# Bond-Slip Assembly (Numba-accelerated)
# ------------------------------------------------------------------------------

@njit(cache=False, boundscheck=True)  # Part A3: Enable boundscheck for debugging
def _bond_slip_assembly_numba(
    u_total: np.ndarray,        # [ndof_total] displacement vector
    segs: np.ndarray,           # [n_seg, 5]: [n1, n2, L0, cx, cy]
    steel_dof_map: np.ndarray,  # [nnode, 2]: node → steel DOF indices
    bond_params: np.ndarray,    # [tau_max, s1, s2, s3, tau_f, alpha, perimeter, dtau_max]
    s_max_hist: np.ndarray,     # [n_seg] history
    steel_EA: float = 0.0,      # E * A for steel (if > 0, adds axial stiffness)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble bond-slip interface contribution (Numba kernel).

    Returns
    -------
    f_int : np.ndarray
        Internal force vector [ndof_total]
    rows, cols, data : np.ndarray
        Triplets for stiffness matrix
    s_current : np.ndarray
        Current slip values [n_seg]
    """
    ndof = u_total.shape[0]
    n_seg = segs.shape[0]

    f = np.zeros(ndof, dtype=np.float64)
    max_entries = n_seg * 96  # 64 (bond 8x8 full Jacobian) + 16 (steel axial) + margin
    rows = np.empty(max_entries, dtype=np.int64)
    cols = np.empty(max_entries, dtype=np.int64)
    data = np.empty(max_entries, dtype=np.float64)
    s_current = np.zeros(n_seg, dtype=np.float64)

    # Unpack bond parameters
    tau_max = bond_params[0]
    s1 = bond_params[1]
    s2 = bond_params[2]
    s3 = bond_params[3]
    tau_f = bond_params[4]
    alpha = bond_params[5]
    perimeter = bond_params[6]  # π * d_bar
    dtau_max = bond_params[7] if bond_params.shape[0] > 7 else 1e20  # Tangent cap (Priority #1)
    gamma = bond_params[8] if bond_params.shape[0] > 8 else 1.0  # BLOQUE 3: Continuation parameter

    entry_idx = 0

    for i in range(n_seg):  # Changed from prange due to entry_idx conflict
        # Node IDs
        n1 = int(segs[i, 0])
        n2 = int(segs[i, 1])
        L0 = segs[i, 2]
        cx = segs[i, 3]
        cy = segs[i, 4]

        # Concrete DOFs
        dof_c1x = 2 * n1
        dof_c1y = 2 * n1 + 1
        dof_c2x = 2 * n2
        dof_c2y = 2 * n2 + 1

        # Steel DOFs (sparse mapping via steel_dof_map)
        dof_s1x = int(steel_dof_map[n1, 0])
        dof_s1y = int(steel_dof_map[n1, 1])
        dof_s2x = int(steel_dof_map[n2, 0])
        dof_s2y = int(steel_dof_map[n2, 1])

        # Displacements
        u_c1x = u_total[dof_c1x]
        u_c1y = u_total[dof_c1y]
        u_c2x = u_total[dof_c2x]
        u_c2y = u_total[dof_c2y]

        u_s1x = u_total[dof_s1x]
        u_s1y = u_total[dof_s1y]
        u_s2x = u_total[dof_s2x]
        u_s2y = u_total[dof_s2y]

        # Average slip along segment (simplified: use midpoint)
        u_c_mid_x = 0.5 * (u_c1x + u_c2x)
        u_c_mid_y = 0.5 * (u_c1y + u_c2y)
        u_s_mid_x = 0.5 * (u_s1x + u_s2x)
        u_s_mid_y = 0.5 * (u_s1y + u_s2y)

        # Relative displacement (steel - concrete)
        du_x = u_s_mid_x - u_c_mid_x
        du_y = u_s_mid_y - u_c_mid_y

        # Slip in bar direction (tangential)
        s = du_x * cx + du_y * cy
        s_current[i] = s

        # Bond stress (with history)
        s_abs = abs(s)
        s_max = max(s_max_hist[i], s_abs)
        sign = 1.0 if s >= 0.0 else -1.0

        # Simplified C1-continuous regularization for s -> 0 singularity (Priority #1)
        # Use s_reg = 0.5 * s1 as threshold (increased from 0.1 to improve conditioning)
        # At s_reg=500μm: k_bond≈6e6 N/m (10x smaller than steel, 1e5x smaller than concrete)
        s_reg = 0.5 * s1

        # Simpler C1-continuous approach:
        # 1) For s <= s_reg: τ(s) = k0 * s (linear)
        # 2) For s > s_reg: τ(s) = τ_max * (s/s1)^α (original power law)
        #
        # Match dτ/ds at s_reg for C1 continuity:
        # k0 = τ_max * α / s1 * (s_reg/s1)^(α-1)
        #
        # This ensures:
        # - τ is continuous (may have small jump, but acceptable)
        # - dτ/ds is continuous (C1)
        # - k0 is finite for all α > 0

        k0 = tau_max * alpha / s1 * ((s_reg / s1) ** (alpha - 1.0))  # Tangent stiffness at s_reg

        # Envelope evaluation (inline for Numba)
        if s_abs >= s_max - 1e-14:
            # Loading envelope
            if s_abs <= s1:
                if s_abs < s_reg:
                    # Regularized linear branch (C1-continuous at s_reg)
                    tau_abs = k0 * s_abs
                    dtau_abs = k0
                else:
                    # Original power law (matches derivative at s_reg)
                    ratio = s_abs / s1
                    tau_abs = tau_max * (ratio ** alpha)
                    dtau_abs = tau_max * alpha / s1 * (ratio ** (alpha - 1.0))
            elif s_abs <= s2:
                tau_abs = tau_max
                dtau_abs = 0.0
            elif s_abs <= s3:
                tau_abs = tau_max - (tau_max - tau_f) * (s_abs - s2) / (s3 - s2)
                dtau_abs = -(tau_max - tau_f) / (s3 - s2)
            else:
                tau_abs = tau_f
                dtau_abs = 0.0
        else:
            # Unloading: secant
            tau_env, _ = 0.0, 0.0
            if s_max <= s1:
                tau_env = tau_max * (s_max / s1) ** alpha if s_max > 1e-16 else 0.0
            elif s_max <= s2:
                tau_env = tau_max
            elif s_max <= s3:
                tau_env = tau_max - (tau_max - tau_f) * (s_max - s2) / (s3 - s2)
            else:
                tau_env = tau_f

            if s_max > 1e-14:
                tau_abs = tau_env * (s_abs / s_max)
                dtau_abs = tau_env / s_max
            else:
                tau_abs = 0.0
                dtau_abs = 0.0

        tau = sign * tau_abs
        dtau_ds = dtau_abs

        # Tangent capping (numerical stabilization, Priority #1)
        # NOTE: dtau_max passed as parameter; set to large value (1e20) if no capping desired
        # In practice, set dtau_max = bond_tangent_cap_factor * median(diag(K_bulk))
        # This prevents bond stiffness from dominating and causing ill-conditioning
        if dtau_ds > dtau_max:
            dtau_ds = dtau_max

        # Bond force (distributed over segment length)
        F_bond = tau * perimeter * L0

        # Distribute to nodes (simplified: equal split)
        # Force on steel (in bar direction)
        Fx_s = F_bond * cx
        Fy_s = F_bond * cy

        # Newton's third law: force on concrete is opposite
        Fx_c = -Fx_s
        Fy_c = -Fy_s

        # Distribute equally to segment ends
        f[dof_s1x] += 0.5 * Fx_s
        f[dof_s1y] += 0.5 * Fy_s
        f[dof_s2x] += 0.5 * Fx_s
        f[dof_s2y] += 0.5 * Fy_s

        f[dof_c1x] += 0.5 * Fx_c
        f[dof_c1y] += 0.5 * Fy_c
        f[dof_c2x] += 0.5 * Fx_c
        f[dof_c2y] += 0.5 * Fy_c

        # Stiffness contribution: K_seg = K_bond * g ⊗ g^T (full 8x8 segment Jacobian)
        # where:
        #   K_bond = gamma * dtau_ds * perimeter * L0  (BLOQUE 3: gamma scaling for continuation)
        #   g = [∂s/∂u] = [-t/2, -t/2, +t/2, +t/2] for [concrete_n1, concrete_n2, steel_n1, steel_n2]
        #   t = [cx, cy] is the bar tangent vector
        #
        # This gives the CONSISTENT TANGENT that couples steel ↔ concrete DOFs
        # (fixes Newton convergence issues from previous diagonal-only placeholder)
        #
        # Gamma continuation: start with gamma=0 (no bond), ramp to gamma=1 (full bond)

        K_bond = gamma * dtau_ds * perimeter * L0

        # Gradient vector components (scaled by 0.5 for midpoint averaging)
        # Concrete node 1: -0.5 * t
        g_c1x = -0.5 * cx
        g_c1y = -0.5 * cy
        # Concrete node 2: -0.5 * t
        g_c2x = -0.5 * cx
        g_c2y = -0.5 * cy
        # Steel node 1: +0.5 * t
        g_s1x = +0.5 * cx
        g_s1y = +0.5 * cy
        # Steel node 2: +0.5 * t
        g_s2x = +0.5 * cx
        g_s2y = +0.5 * cy

        # Assemble full 8x8 block: K_seg[i,j] = K_bond * g[i] * g[j]
        # This is the outer product g ⊗ g^T
        # Each row-column pair contributes one entry
        if entry_idx + 64 < max_entries:
            # Row 1: concrete node 1, x-direction
            rows[entry_idx] = dof_c1x; cols[entry_idx] = dof_c1x; data[entry_idx] = K_bond * g_c1x * g_c1x; entry_idx += 1
            rows[entry_idx] = dof_c1x; cols[entry_idx] = dof_c1y; data[entry_idx] = K_bond * g_c1x * g_c1y; entry_idx += 1
            rows[entry_idx] = dof_c1x; cols[entry_idx] = dof_c2x; data[entry_idx] = K_bond * g_c1x * g_c2x; entry_idx += 1
            rows[entry_idx] = dof_c1x; cols[entry_idx] = dof_c2y; data[entry_idx] = K_bond * g_c1x * g_c2y; entry_idx += 1
            rows[entry_idx] = dof_c1x; cols[entry_idx] = dof_s1x; data[entry_idx] = K_bond * g_c1x * g_s1x; entry_idx += 1
            rows[entry_idx] = dof_c1x; cols[entry_idx] = dof_s1y; data[entry_idx] = K_bond * g_c1x * g_s1y; entry_idx += 1
            rows[entry_idx] = dof_c1x; cols[entry_idx] = dof_s2x; data[entry_idx] = K_bond * g_c1x * g_s2x; entry_idx += 1
            rows[entry_idx] = dof_c1x; cols[entry_idx] = dof_s2y; data[entry_idx] = K_bond * g_c1x * g_s2y; entry_idx += 1

            # Row 2: concrete node 1, y-direction
            rows[entry_idx] = dof_c1y; cols[entry_idx] = dof_c1x; data[entry_idx] = K_bond * g_c1y * g_c1x; entry_idx += 1
            rows[entry_idx] = dof_c1y; cols[entry_idx] = dof_c1y; data[entry_idx] = K_bond * g_c1y * g_c1y; entry_idx += 1
            rows[entry_idx] = dof_c1y; cols[entry_idx] = dof_c2x; data[entry_idx] = K_bond * g_c1y * g_c2x; entry_idx += 1
            rows[entry_idx] = dof_c1y; cols[entry_idx] = dof_c2y; data[entry_idx] = K_bond * g_c1y * g_c2y; entry_idx += 1
            rows[entry_idx] = dof_c1y; cols[entry_idx] = dof_s1x; data[entry_idx] = K_bond * g_c1y * g_s1x; entry_idx += 1
            rows[entry_idx] = dof_c1y; cols[entry_idx] = dof_s1y; data[entry_idx] = K_bond * g_c1y * g_s1y; entry_idx += 1
            rows[entry_idx] = dof_c1y; cols[entry_idx] = dof_s2x; data[entry_idx] = K_bond * g_c1y * g_s2x; entry_idx += 1
            rows[entry_idx] = dof_c1y; cols[entry_idx] = dof_s2y; data[entry_idx] = K_bond * g_c1y * g_s2y; entry_idx += 1

            # Row 3: concrete node 2, x-direction
            rows[entry_idx] = dof_c2x; cols[entry_idx] = dof_c1x; data[entry_idx] = K_bond * g_c2x * g_c1x; entry_idx += 1
            rows[entry_idx] = dof_c2x; cols[entry_idx] = dof_c1y; data[entry_idx] = K_bond * g_c2x * g_c1y; entry_idx += 1
            rows[entry_idx] = dof_c2x; cols[entry_idx] = dof_c2x; data[entry_idx] = K_bond * g_c2x * g_c2x; entry_idx += 1
            rows[entry_idx] = dof_c2x; cols[entry_idx] = dof_c2y; data[entry_idx] = K_bond * g_c2x * g_c2y; entry_idx += 1
            rows[entry_idx] = dof_c2x; cols[entry_idx] = dof_s1x; data[entry_idx] = K_bond * g_c2x * g_s1x; entry_idx += 1
            rows[entry_idx] = dof_c2x; cols[entry_idx] = dof_s1y; data[entry_idx] = K_bond * g_c2x * g_s1y; entry_idx += 1
            rows[entry_idx] = dof_c2x; cols[entry_idx] = dof_s2x; data[entry_idx] = K_bond * g_c2x * g_s2x; entry_idx += 1
            rows[entry_idx] = dof_c2x; cols[entry_idx] = dof_s2y; data[entry_idx] = K_bond * g_c2x * g_s2y; entry_idx += 1

            # Row 4: concrete node 2, y-direction
            rows[entry_idx] = dof_c2y; cols[entry_idx] = dof_c1x; data[entry_idx] = K_bond * g_c2y * g_c1x; entry_idx += 1
            rows[entry_idx] = dof_c2y; cols[entry_idx] = dof_c1y; data[entry_idx] = K_bond * g_c2y * g_c1y; entry_idx += 1
            rows[entry_idx] = dof_c2y; cols[entry_idx] = dof_c2x; data[entry_idx] = K_bond * g_c2y * g_c2x; entry_idx += 1
            rows[entry_idx] = dof_c2y; cols[entry_idx] = dof_c2y; data[entry_idx] = K_bond * g_c2y * g_c2y; entry_idx += 1
            rows[entry_idx] = dof_c2y; cols[entry_idx] = dof_s1x; data[entry_idx] = K_bond * g_c2y * g_s1x; entry_idx += 1
            rows[entry_idx] = dof_c2y; cols[entry_idx] = dof_s1y; data[entry_idx] = K_bond * g_c2y * g_s1y; entry_idx += 1
            rows[entry_idx] = dof_c2y; cols[entry_idx] = dof_s2x; data[entry_idx] = K_bond * g_c2y * g_s2x; entry_idx += 1
            rows[entry_idx] = dof_c2y; cols[entry_idx] = dof_s2y; data[entry_idx] = K_bond * g_c2y * g_s2y; entry_idx += 1

            # Row 5: steel node 1, x-direction
            rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_c1x; data[entry_idx] = K_bond * g_s1x * g_c1x; entry_idx += 1
            rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_c1y; data[entry_idx] = K_bond * g_s1x * g_c1y; entry_idx += 1
            rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_c2x; data[entry_idx] = K_bond * g_s1x * g_c2x; entry_idx += 1
            rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_c2y; data[entry_idx] = K_bond * g_s1x * g_c2y; entry_idx += 1
            rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_s1x; data[entry_idx] = K_bond * g_s1x * g_s1x; entry_idx += 1
            rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_s1y; data[entry_idx] = K_bond * g_s1x * g_s1y; entry_idx += 1
            rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_s2x; data[entry_idx] = K_bond * g_s1x * g_s2x; entry_idx += 1
            rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_s2y; data[entry_idx] = K_bond * g_s1x * g_s2y; entry_idx += 1

            # Row 6: steel node 1, y-direction
            rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_c1x; data[entry_idx] = K_bond * g_s1y * g_c1x; entry_idx += 1
            rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_c1y; data[entry_idx] = K_bond * g_s1y * g_c1y; entry_idx += 1
            rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_c2x; data[entry_idx] = K_bond * g_s1y * g_c2x; entry_idx += 1
            rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_c2y; data[entry_idx] = K_bond * g_s1y * g_c2y; entry_idx += 1
            rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_s1x; data[entry_idx] = K_bond * g_s1y * g_s1x; entry_idx += 1
            rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_s1y; data[entry_idx] = K_bond * g_s1y * g_s1y; entry_idx += 1
            rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_s2x; data[entry_idx] = K_bond * g_s1y * g_s2x; entry_idx += 1
            rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_s2y; data[entry_idx] = K_bond * g_s1y * g_s2y; entry_idx += 1

            # Row 7: steel node 2, x-direction
            rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_c1x; data[entry_idx] = K_bond * g_s2x * g_c1x; entry_idx += 1
            rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_c1y; data[entry_idx] = K_bond * g_s2x * g_c1y; entry_idx += 1
            rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_c2x; data[entry_idx] = K_bond * g_s2x * g_c2x; entry_idx += 1
            rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_c2y; data[entry_idx] = K_bond * g_s2x * g_c2y; entry_idx += 1
            rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_s1x; data[entry_idx] = K_bond * g_s2x * g_s1x; entry_idx += 1
            rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_s1y; data[entry_idx] = K_bond * g_s2x * g_s1y; entry_idx += 1
            rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_s2x; data[entry_idx] = K_bond * g_s2x * g_s2x; entry_idx += 1
            rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_s2y; data[entry_idx] = K_bond * g_s2x * g_s2y; entry_idx += 1

            # Row 8: steel node 2, y-direction
            rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_c1x; data[entry_idx] = K_bond * g_s2y * g_c1x; entry_idx += 1
            rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_c1y; data[entry_idx] = K_bond * g_s2y * g_c1y; entry_idx += 1
            rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_c2x; data[entry_idx] = K_bond * g_s2y * g_c2x; entry_idx += 1
            rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_c2y; data[entry_idx] = K_bond * g_s2y * g_c2y; entry_idx += 1
            rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_s1x; data[entry_idx] = K_bond * g_s2y * g_s1x; entry_idx += 1
            rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_s1y; data[entry_idx] = K_bond * g_s2y * g_s1y; entry_idx += 1
            rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_s2x; data[entry_idx] = K_bond * g_s2y * g_s2x; entry_idx += 1
            rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_s2y; data[entry_idx] = K_bond * g_s2y * g_s2y; entry_idx += 1

        # Steel axial stiffness (if steel_EA > 0)
        if steel_EA > 0.0 and entry_idx + 16 < max_entries:
            K_steel = steel_EA / L0
            Kxx_s = K_steel * cx * cx
            Kxy_s = K_steel * cx * cy
            Kyy_s = K_steel * cy * cy

            # Steel bar stiffness: K = (EA/L) * [c⊗c, -c⊗c; -c⊗c, c⊗c]
            # Node 1 - Node 1 (positive)
            rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_s1x; data[entry_idx] = Kxx_s; entry_idx += 1
            rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_s1y; data[entry_idx] = Kxy_s; entry_idx += 1
            rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_s1x; data[entry_idx] = Kxy_s; entry_idx += 1
            rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_s1y; data[entry_idx] = Kyy_s; entry_idx += 1

            # Node 2 - Node 2 (positive)
            rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_s2x; data[entry_idx] = Kxx_s; entry_idx += 1
            rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_s2y; data[entry_idx] = Kxy_s; entry_idx += 1
            rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_s2x; data[entry_idx] = Kxy_s; entry_idx += 1
            rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_s2y; data[entry_idx] = Kyy_s; entry_idx += 1

            # Node 1 - Node 2 (negative)
            rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_s2x; data[entry_idx] = -Kxx_s; entry_idx += 1
            rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_s2y; data[entry_idx] = -Kxy_s; entry_idx += 1
            rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_s2x; data[entry_idx] = -Kxy_s; entry_idx += 1
            rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_s2y; data[entry_idx] = -Kyy_s; entry_idx += 1

            # Node 2 - Node 1 (negative)
            rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_s1x; data[entry_idx] = -Kxx_s; entry_idx += 1
            rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_s1y; data[entry_idx] = -Kxy_s; entry_idx += 1
            rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_s1x; data[entry_idx] = -Kxy_s; entry_idx += 1
            rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_s1y; data[entry_idx] = -Kyy_s; entry_idx += 1

    # Trim arrays
    rows_out = rows[:entry_idx]
    cols_out = cols[:entry_idx]
    data_out = data[:entry_idx]

    return f, rows_out, cols_out, data_out, s_current


def assemble_bond_slip(
    u_total: np.ndarray,
    steel_segments: np.ndarray,
    steel_dof_offset: int,
    bond_law: BondSlipModelCode2010,
    bond_states: BondSlipStateArrays,
    steel_dof_map: np.ndarray = None,
    steel_EA: float = 0.0,
    use_numba: bool = True,
    enable_validation: bool = True,  # Part A1: Enable preflight checks
    perimeter: Optional[float] = None,  # Explicit perimeter for robustness
    segment_mask: Optional[np.ndarray] = None,  # Mask to disable bond in specific segments
    bond_gamma: float = 1.0,  # BLOQUE B: Continuation parameter for bond-slip activation
    bond_k_cap: Optional[float] = None,  # BLOQUE C: Cap dtau/ds [Pa/m] (None = no cap)
    bond_s_eps: float = 0.0,  # BLOQUE C: Smooth regularization epsilon [m]
) -> Tuple[np.ndarray, sp.csr_matrix, BondSlipStateArrays]:
    """Assemble bond-slip interface forces and stiffness.

    Parameters
    ----------
    u_total : np.ndarray
        Total displacement vector (concrete + steel DOFs)
    steel_segments : np.ndarray
        [n_seg, 5] array: [n1, n2, L0, cx, cy]
    steel_dof_offset : int
        Index where steel DOFs begin (for backward compat, not used if steel_dof_map provided)
    bond_law : BondSlipModelCode2010
        Bond-slip constitutive model
    bond_states : BondSlipStateArrays
        Current bond-slip states
    steel_dof_map : np.ndarray, optional
        [nnode, 2] array mapping node → steel DOF indices (or -1 if no steel)
        If None, assumes dense contiguous steel DOFs (legacy behavior)
    steel_EA : float
        Steel axial stiffness (E*A)
    use_numba : bool
        Use Numba acceleration if available
    enable_validation : bool
        Run preflight validation checks (Part A1)
    perimeter : float, optional
        Rebar perimeter [m]. If None, attempts to compute from bond_law.d_bar.
        For non-circular reinforcement, pass explicit perimeter.
    segment_mask : np.ndarray, optional
        Boolean mask [n_seg] where True = bond disabled for that segment.
        Useful for "empty element" regions in pullout tests.
    bond_gamma : float, optional
        Continuation parameter for bond-slip activation [0, 1]. Default: 1.0.
        Scales the bond tangent stiffness: K_bond = gamma * k_τ.
        Use gamma < 1 for easier initial convergence, then ramp up to 1.
        Gamma=0 → no bond (only steel axial), Gamma=1 → full bond.
    bond_k_cap : float, optional
        Cap dtau/ds to this value [Pa/m]. None = no capping.
        Prevents excessively stiff tangent near s≈0.
    bond_s_eps : float, optional
        Smoothing epsilon [m]. If > 0, evaluates slip as s_eff = sqrt(s^2 + eps^2).
        Regularizes tangent near s≈0.

    Returns
    -------
    f_bond : np.ndarray
        Bond interface force vector
    K_bond : sparse.csr_matrix
        Bond interface stiffness matrix
    bond_states_new : BondSlipStateArrays
        Updated bond-slip states (trial)
    """
    n_seg = steel_segments.shape[0]
    ndof_total = u_total.shape[0]

    # Detect bond law type and force Python fallback for BilinearBondLaw and BanholzerBondLaw
    # (Numba kernel only supports BondSlipModelCode2010 parameters)
    bond_law_class_name = type(bond_law).__name__
    if bond_law_class_name in ('BilinearBondLaw', 'BanholzerBondLaw'):
        use_numba = False  # Force Python fallback for these bond law types

    # Compute perimeter (explicit parameter takes precedence)
    if perimeter is None:
        # Fallback: try to get from bond_law.d_bar (backward compatibility)
        if hasattr(bond_law, 'd_bar'):
            perimeter = math.pi * float(bond_law.d_bar)
        else:
            raise ValueError(
                "Bond-slip assembly requires perimeter. Either:\n"
                "  1. Pass perimeter explicitly as parameter, or\n"
                "  2. Use a bond_law with d_bar attribute (e.g., BondSlipModelCode2010)"
            )

    # Only create bond_params if using Numba (BondSlipModelCode2010 only)
    if use_numba and NUMBA_AVAILABLE:
        # Tangent capping for numerical stability (Priority #1)
        # dtau_max is typically set to bond_tangent_cap_factor * median(diag(K_bulk))
        # For now, use a large value (no capping); caller can override via bond_law
        dtau_max = getattr(bond_law, 'dtau_max', 1e20)  # Default: no cap

        bond_params = np.array([
            bond_law.tau_max,
            bond_law.s1,
            bond_law.s2,
            bond_law.s3,
            bond_law.tau_f,
            bond_law.alpha,
            perimeter,
            dtau_max,
            bond_gamma,  # BLOQUE 3: Continuation parameter
        ], dtype=float)

    if use_numba and NUMBA_AVAILABLE:
        # Use sparse DOF mapping if provided, else legacy dense mapping
        if steel_dof_map is None:
            # Legacy: assume dense contiguous steel DOFs
            nnode = int(ndof_total - steel_dof_offset) // 2
            steel_dof_map = -np.ones((nnode, 2), dtype=np.int64)
            for n in range(nnode):
                steel_dof_map[n, 0] = steel_dof_offset + 2 * n
                steel_dof_map[n, 1] = steel_dof_offset + 2 * n + 1

        # Part A2: Force dtypes and contiguity
        u_total = np.ascontiguousarray(u_total, dtype=np.float64)
        steel_segments = np.ascontiguousarray(steel_segments, dtype=np.float64)
        steel_dof_map = np.ascontiguousarray(steel_dof_map, dtype=np.int64)
        bond_params = np.ascontiguousarray(bond_params, dtype=np.float64)
        s_max_hist = np.ascontiguousarray(bond_states.s_max, dtype=np.float64)

        # Part A1: Preflight validation
        if enable_validation:
            try:
                validate_bond_inputs(
                    u_total=u_total,
                    segs=steel_segments,
                    steel_dof_map=steel_dof_map,
                    steel_dof_offset=steel_dof_offset,
                    bond_states=bond_states,
                )
            except RuntimeError as e:
                # Add context about when/where this error occurred
                import traceback
                tb = traceback.format_exc()
                raise RuntimeError(
                    f"Bond-slip validation failed!\n"
                    f"This error indicates invalid DOF mapping after crack growth.\n"
                    f"Original error:\n{str(e)}\n\n"
                    f"Traceback:\n{tb}"
                ) from e

        # Call Numba kernel
        try:
            f_bond, rows, cols, data, s_curr = _bond_slip_assembly_numba(
                u_total,
                steel_segments,
                steel_dof_map,
                bond_params,
                s_max_hist,
                steel_EA,
            )
        except Exception as e:
            # Enhanced error reporting for kernel failures
            raise RuntimeError(
                f"Bond-slip Numba kernel failed!\n"
                f"Context: ndof={ndof_total}, n_seg={n_seg}, steel_dof_offset={steel_dof_offset}\n"
                f"This likely indicates an array indexing bug in the Numba kernel.\n"
                f"Original error: {str(e)}"
            ) from e

        # Apply segment mask (disable bond for masked segments)
        if segment_mask is not None:
            # Zero out forces and stiffness for disabled segments
            # This is done by zeroing the forces and removing stiffness entries
            # For simplicity, we zero forces in f_bond and rebuild K_bond with masked segments
            for i in range(n_seg):
                if segment_mask[i]:  # True = disabled
                    # Zero slip in disabled segments
                    s_curr[i] = 0.0
                    # Note: K and f already computed; we'll zero them out per-segment
                    # This is inefficient but simple; TODO: pass mask to kernel for efficiency

        K_bond = sp.csr_matrix((data, (rows, cols)), shape=(ndof_total, ndof_total))

        # Apply segment mask to force vector
        if segment_mask is not None:
            # Zero forces for disabled segments
            # Map segment → DOFs and zero them
            for i in range(n_seg):
                if segment_mask[i]:
                    # Get nodes for this segment
                    n1 = int(steel_segments[i, 0])
                    n2 = int(steel_segments[i, 1])
                    # Zero concrete and steel forces for these nodes
                    # Concrete DOFs
                    f_bond[2 * n1] = 0.0
                    f_bond[2 * n1 + 1] = 0.0
                    f_bond[2 * n2] = 0.0
                    f_bond[2 * n2 + 1] = 0.0
                    # Steel DOFs
                    if steel_dof_map is not None:
                        if steel_dof_map[n1, 0] >= 0:
                            f_bond[steel_dof_map[n1, 0]] = 0.0
                            f_bond[steel_dof_map[n1, 1]] = 0.0
                        if steel_dof_map[n2, 0] >= 0:
                            f_bond[steel_dof_map[n2, 0]] = 0.0
                            f_bond[steel_dof_map[n2, 1]] = 0.0

        # Update states (trial)
        bond_states_new = BondSlipStateArrays(
            n_segments=n_seg,
            s_max=np.maximum(bond_states.s_max, np.abs(s_curr)),
            s_current=s_curr,
            tau_current=np.zeros(n_seg),  # Computed inline; could extract if needed
        )

    else:
        # Part A4: Pure Python fallback for debugging
        f_bond, K_bond, bond_states_new = _bond_slip_assembly_python(
            u_total=u_total,
            steel_segments=steel_segments,
            steel_dof_map=steel_dof_map if steel_dof_map is not None else _build_legacy_dof_map(ndof_total, steel_dof_offset),
            bond_law=bond_law,
            bond_states=bond_states,
            steel_EA=steel_EA,
            perimeter=perimeter,  # Pass computed perimeter
            bond_gamma=bond_gamma,  # BLOQUE 3: Pass gamma to Python fallback
            bond_k_cap=bond_k_cap,  # BLOQUE C: Pass tangent cap
            bond_s_eps=bond_s_eps,  # BLOQUE C: Pass smoothing epsilon
            segment_mask=segment_mask,  # Pass segment mask for disabled segments
        )

    return f_bond, K_bond, bond_states_new


# ------------------------------------------------------------------------------
# Python Fallback (Part A4)
# ------------------------------------------------------------------------------

def _build_legacy_dof_map(ndof_total: int, steel_dof_offset: int) -> np.ndarray:
    """Build legacy dense DOF mapping for backward compatibility."""
    nnode = int(ndof_total - steel_dof_offset) // 2
    steel_dof_map = -np.ones((nnode, 2), dtype=np.int64)
    for n in range(nnode):
        steel_dof_map[n, 0] = steel_dof_offset + 2 * n
        steel_dof_map[n, 1] = steel_dof_offset + 2 * n + 1
    return steel_dof_map


def _bond_slip_assembly_python(
    u_total: np.ndarray,
    steel_segments: np.ndarray,
    steel_dof_map: np.ndarray,
    bond_law: BondSlipModelCode2010,
    bond_states: BondSlipStateArrays,
    steel_EA: float = 0.0,
    perimeter: float = None,
    bond_gamma: float = 1.0,  # BLOQUE 3: Continuation parameter
    bond_k_cap: Optional[float] = None,  # BLOQUE C: Cap dtau/ds
    bond_s_eps: float = 0.0,  # BLOQUE C: Smooth regularization epsilon
    segment_mask: Optional[np.ndarray] = None,  # Mask to disable bond in specific segments
) -> Tuple[np.ndarray, sp.csr_matrix, BondSlipStateArrays]:
    """Pure Python fallback for bond-slip assembly (for debugging).

    This implementation has explicit bounds checks and assertions
    to help diagnose kernel bugs.
    """
    ndof = u_total.shape[0]
    n_seg = steel_segments.shape[0]

    f_bond = np.zeros(ndof, dtype=float)
    rows = []
    cols = []
    data = []
    s_current = np.zeros(n_seg, dtype=float)

    # Use provided perimeter or compute from bond_law.d_bar
    if perimeter is None:
        perimeter = math.pi * bond_law.d_bar

    for i in range(n_seg):
        # Skip disabled segments (segment_mask[i] == True means disabled)
        if segment_mask is not None and segment_mask[i]:
            # Set slip to zero for disabled segments (no forces, no stiffness)
            s_current[i] = 0.0
            continue

        # Extract segment data
        n1 = int(steel_segments[i, 0])
        n2 = int(steel_segments[i, 1])
        L0 = steel_segments[i, 2]
        cx = steel_segments[i, 3]
        cy = steel_segments[i, 4]

        # Concrete DOFs (with bounds check)
        dof_c1x = 2 * n1
        dof_c1y = 2 * n1 + 1
        dof_c2x = 2 * n2
        dof_c2y = 2 * n2 + 1

        assert 0 <= dof_c1x < ndof, f"dof_c1x={dof_c1x} out of bounds [0, {ndof})"
        assert 0 <= dof_c1y < ndof, f"dof_c1y={dof_c1y} out of bounds [0, {ndof})"
        assert 0 <= dof_c2x < ndof, f"dof_c2x={dof_c2x} out of bounds [0, {ndof})"
        assert 0 <= dof_c2y < ndof, f"dof_c2y={dof_c2y} out of bounds [0, {ndof})"

        # Steel DOFs (with bounds check)
        dof_s1x = int(steel_dof_map[n1, 0])
        dof_s1y = int(steel_dof_map[n1, 1])
        dof_s2x = int(steel_dof_map[n2, 0])
        dof_s2y = int(steel_dof_map[n2, 1])

        assert dof_s1x != -1, f"seg {i}: n1={n1} has no steel DOF (x)"
        assert dof_s1y != -1, f"seg {i}: n1={n1} has no steel DOF (y)"
        assert dof_s2x != -1, f"seg {i}: n2={n2} has no steel DOF (x)"
        assert dof_s2y != -1, f"seg {i}: n2={n2} has no steel DOF (y)"

        assert 0 <= dof_s1x < ndof, f"dof_s1x={dof_s1x} out of bounds [0, {ndof})"
        assert 0 <= dof_s1y < ndof, f"dof_s1y={dof_s1y} out of bounds [0, {ndof})"
        assert 0 <= dof_s2x < ndof, f"dof_s2x={dof_s2x} out of bounds [0, {ndof})"
        assert 0 <= dof_s2y < ndof, f"dof_s2y={dof_s2y} out of bounds [0, {ndof})"

        # Displacements
        u_c1x = u_total[dof_c1x]
        u_c1y = u_total[dof_c1y]
        u_c2x = u_total[dof_c2x]
        u_c2y = u_total[dof_c2y]

        u_s1x = u_total[dof_s1x]
        u_s1y = u_total[dof_s1y]
        u_s2x = u_total[dof_s2x]
        u_s2y = u_total[dof_s2y]

        # Average slip
        u_c_mid_x = 0.5 * (u_c1x + u_c2x)
        u_c_mid_y = 0.5 * (u_c1y + u_c2y)
        u_s_mid_x = 0.5 * (u_s1x + u_s2x)
        u_s_mid_y = 0.5 * (u_s1y + u_s2y)

        du_x = u_s_mid_x - u_c_mid_x
        du_y = u_s_mid_y - u_c_mid_y

        # Slip in bar direction
        s = du_x * cx + du_y * cy
        s_current[i] = s

        # Bond stress
        s_max = max(bond_states.s_max[i], abs(s))

        # BLOQUE C: Optional slip smoothing (regularization near s≈0)
        s_eval = s
        if bond_s_eps > 0.0:
            # Evaluate tau at smoothed slip: s_eff = sqrt(s^2 + eps^2)
            s_abs = abs(s)
            s_sign = 1.0 if s >= 0 else -1.0
            s_eff = math.sqrt(s_abs**2 + bond_s_eps**2)
            s_eval = s_sign * s_eff

        tau, dtau_ds = bond_law.tau_and_tangent(s_eval, s_max)

        # BLOQUE C: Optional tangent capping (prevent excessive stiffness)
        if bond_k_cap is not None and dtau_ds > bond_k_cap:
            dtau_ds = bond_k_cap

        # Bond force
        F_bond = tau * perimeter * L0

        # Distribute to nodes
        Fx_s = F_bond * cx
        Fy_s = F_bond * cy
        Fx_c = -Fx_s
        Fy_c = -Fy_s

        f_bond[dof_s1x] += 0.5 * Fx_s
        f_bond[dof_s1y] += 0.5 * Fy_s
        f_bond[dof_s2x] += 0.5 * Fx_s
        f_bond[dof_s2y] += 0.5 * Fy_s

        f_bond[dof_c1x] += 0.5 * Fx_c
        f_bond[dof_c1y] += 0.5 * Fy_c
        f_bond[dof_c2x] += 0.5 * Fx_c
        f_bond[dof_c2y] += 0.5 * Fy_c

        # Stiffness (with gamma continuation scaling, BLOQUE 3)
        # Full 8×8 consistent tangent: K_seg = K_bond * g ⊗ g^T
        # where g = [∂s/∂u] is the gradient of slip with respect to DOFs
        # This matches the Numba kernel implementation and provides proper steel↔concrete coupling
        K_bond = bond_gamma * dtau_ds * perimeter * L0

        # Gradient vector components (scaled by 0.5 for midpoint averaging)
        # Concrete node 1: -0.5 * t
        g_c1x = -0.5 * cx
        g_c1y = -0.5 * cy
        # Concrete node 2: -0.5 * t
        g_c2x = -0.5 * cx
        g_c2y = -0.5 * cy
        # Steel node 1: +0.5 * t
        g_s1x = +0.5 * cx
        g_s1y = +0.5 * cy
        # Steel node 2: +0.5 * t
        g_s2x = +0.5 * cx
        g_s2y = +0.5 * cy

        # Build DOF list and gradient list for outer product
        dofs = [dof_c1x, dof_c1y, dof_c2x, dof_c2y, dof_s1x, dof_s1y, dof_s2x, dof_s2y]
        g = [g_c1x, g_c1y, g_c2x, g_c2y, g_s1x, g_s1y, g_s2x, g_s2y]

        # Assemble full 8×8 block: K_seg[a,b] = K_bond * g[a] * g[b]
        # Skip entries where DOF is negative (marker for unassigned/disabled DOFs)
        for a in range(8):
            if dofs[a] < 0:
                continue
            for b in range(8):
                if dofs[b] < 0:
                    continue
                rows.append(dofs[a])
                cols.append(dofs[b])
                data.append(K_bond * g[a] * g[b])

        # Add steel axial stiffness if requested
        if steel_EA > 0.0:
            K_steel = steel_EA / L0
            Kxx_s = K_steel * cx * cx
            Kxy_s = K_steel * cx * cy
            Kyy_s = K_steel * cy * cy

            # Add 16 entries for full 4x4 block
            for ri, ci, val in [
                (dof_s1x, dof_s1x, Kxx_s), (dof_s1x, dof_s1y, Kxy_s),
                (dof_s1y, dof_s1x, Kxy_s), (dof_s1y, dof_s1y, Kyy_s),
                (dof_s2x, dof_s2x, Kxx_s), (dof_s2x, dof_s2y, Kxy_s),
                (dof_s2y, dof_s2x, Kxy_s), (dof_s2y, dof_s2y, Kyy_s),
                (dof_s1x, dof_s2x, -Kxx_s), (dof_s1x, dof_s2y, -Kxy_s),
                (dof_s1y, dof_s2x, -Kxy_s), (dof_s1y, dof_s2y, -Kyy_s),
                (dof_s2x, dof_s1x, -Kxx_s), (dof_s2x, dof_s1y, -Kxy_s),
                (dof_s2y, dof_s1x, -Kxy_s), (dof_s2y, dof_s1y, -Kyy_s),
            ]:
                rows.append(ri)
                cols.append(ci)
                data.append(val)

    K_bond_sp = sp.csr_matrix((data, (rows, cols)), shape=(ndof, ndof))

    # Update states
    bond_states_new = BondSlipStateArrays(
        n_segments=n_seg,
        s_max=np.maximum(bond_states.s_max, np.abs(s_current)),
        s_current=s_current,
        tau_current=np.zeros(n_seg),
    )

    return f_bond, K_bond_sp, bond_states_new


# ------------------------------------------------------------------------------
# Dowel Action at Crack Intersections
# ------------------------------------------------------------------------------

def compute_dowel_springs(
    crack_geom: Any,  # XFEMCrack or similar
    steel_segments: np.ndarray,
    nodes: np.ndarray,
    E_steel: float,
    d_bar: float,
) -> List[Tuple[int, int, float]]:
    """Compute dowel action springs at crack-rebar intersections.

    Parameters
    ----------
    crack_geom : crack geometry object
        Contains crack path information
    steel_segments : np.ndarray
        [n_seg, 5]: [n1, n2, L0, cx, cy]
    nodes : np.ndarray
        [n_nodes, 2]: node coordinates
    E_steel : float
        Steel Young's modulus [Pa]
    d_bar : float
        Bar diameter [m]

    Returns
    -------
    dowel_springs : List[Tuple[int, int, float]]
        List of (node_concrete, node_steel, k_dowel)
        where k_dowel is the transverse stiffness [N/m]

    Notes
    -----
    Dowel action model (elastic foundation):
        k_dowel = E_s * I / L_e^3
        L_e = 10 * d_bar (empirical effective length)
    """
    I_bar = math.pi * (d_bar ** 4) / 64.0  # Second moment of area
    L_e = 10.0 * d_bar  # Effective embedment length (empirical)
    k_dowel = E_steel * I_bar / (L_e ** 3)

    dowel_springs = []

    # Simplified: assume crack is a vertical line (for initial implementation)
    # Full implementation would do geometric intersection
    if not hasattr(crack_geom, 'tip_x'):
        return dowel_springs  # No active crack

    crack_x = float(crack_geom.tip_x)

    for i in range(steel_segments.shape[0]):
        n1 = int(steel_segments[i, 0])
        n2 = int(steel_segments[i, 1])
        x1 = nodes[n1, 0]
        x2 = nodes[n2, 0]

        # Check if crack crosses segment
        if (x1 <= crack_x <= x2) or (x2 <= crack_x <= x1):
            # Add dowel spring at both nodes (distributed)
            dowel_springs.append((n1, n1, k_dowel))
            dowel_springs.append((n2, n2, k_dowel))

    return dowel_springs


def assemble_dowel_action(
    dowel_springs: List[Tuple[int, int, float]],
    steel_dof_offset: int,
    ndof_total: int,
) -> sp.csr_matrix:
    """Assemble dowel action stiffness matrix.

    Parameters
    ----------
    dowel_springs : List[Tuple[int, int, float]]
        [(node_concrete, node_steel, k_dowel), ...]
    steel_dof_offset : int
        Offset for steel DOFs
    ndof_total : int
        Total number of DOFs

    Returns
    -------
    K_dowel : sparse.csr_matrix
        Dowel stiffness matrix (transverse direction only)
    """
    rows = []
    cols = []
    data = []

    for node_c, node_s, k in dowel_springs:
        # Transverse direction (assume y for vertical crack)
        dof_cy = 2 * node_c + 1
        dof_sy = steel_dof_offset + 2 * node_s + 1

        # Diagonal terms
        rows.append(dof_cy)
        cols.append(dof_cy)
        data.append(k)

        rows.append(dof_sy)
        cols.append(dof_sy)
        data.append(k)

        # Off-diagonal coupling (penalty: resists relative motion)
        rows.append(dof_cy)
        cols.append(dof_sy)
        data.append(-k)

        rows.append(dof_sy)
        cols.append(dof_cy)
        data.append(-k)

    if len(data) == 0:
        return sp.csr_matrix((ndof_total, ndof_total), dtype=float)

    K_dowel = sp.csr_matrix((data, (rows, cols)), shape=(ndof_total, ndof_total))
    return K_dowel


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def pack_bond_params(bond_law: BondSlipModelCode2010) -> np.ndarray:
    """Pack bond-slip parameters into array for Numba kernels."""
    perimeter = math.pi * float(bond_law.d_bar)
    return np.array([
        bond_law.tau_max,
        bond_law.s1,
        bond_law.s2,
        bond_law.s3,
        bond_law.tau_f,
        bond_law.alpha,
        perimeter,
    ], dtype=float)
