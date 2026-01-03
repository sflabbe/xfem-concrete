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

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional, Union
import math

import numpy as np
import scipy.sparse as sp

# Optional Numba acceleration
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
    # Import optimized kernel from dedicated module (cache=True for faster startup)
    from xfem_clean.numba.kernels_bond_slip import bond_slip_assembly_kernel
except Exception:
    NUMBA_AVAILABLE = False
    bond_slip_assembly_kernel = None  # Will use Python fallback
    def njit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap
    def prange(*args):
        return range(*args)


@dataclass
class BondLayer:
    """Bond layer specification for reinforcement (steel rebar or FRP).

    This structure encapsulates all parameters needed for bond-slip modeling
    of a single reinforcement layer, replacing the legacy approach of inventing
    geometry from cover distance.

    Attributes
    ----------
    segments : np.ndarray
        Segment connectivity and geometry [nseg, 5]: [n1, n2, L0, cx, cy]
        where n1, n2 are node IDs, L0 is reference length, cx, cy is unit tangent
    EA : float
        Axial stiffness (N): E * A_total for the layer
        For steel: E_s * (n_bars * π * d²/4)
        For FRP: E_frp * t_frp * b_eff
    perimeter : float
        Bond perimeter (m) for traction integration
        For steel bars: n_bars * π * d
        For FRP sheets: b_eff (effective width)
    bond_law : Union[BondSlipModelCode2010, CustomBondSlipLaw, BilinearBondLaw, BanholzerBondLaw]
        Constitutive law for bond stress-slip relationship
    segment_mask : Optional[np.ndarray]
        Boolean mask [nseg] where True = bond disabled for that segment
        (e.g., for empty elements in pullout tests). Default: None (all active)
    enable_dowel : bool
        Enable dowel action (transverse stress) for this layer. Default: False
    dowel_model : Optional[DowelActionModel]
        Dowel action constitutive model. Required if enable_dowel=True.
    layer_id : str
        Human-readable identifier for this layer (e.g., "rebar_layer_1", "frp_bottom")

    Notes
    -----
    This structure enables:
    - Explicit control over reinforcement geometry (no cover-based invention)
    - Per-layer bond law specification (allows FRP + steel in same model)
    - Segment-level masking for bond-disabled regions
    - Dowel action on/off per layer
    """

    segments: np.ndarray  # [nseg, 5]: [n1, n2, L0, cx, cy]
    EA: float  # Axial stiffness (N)
    perimeter: float  # Bond perimeter (m)
    bond_law: Any  # Bond constitutive law (use Any to avoid circular import)
    segment_mask: Optional[np.ndarray] = None  # [nseg] bool: True = disabled
    enable_dowel: bool = False
    dowel_model: Optional[Any] = None  # DowelActionModel or None
    layer_id: str = "bond_layer"

    def __post_init__(self):
        """Validate bond layer parameters."""
        if self.segments.ndim != 2 or self.segments.shape[1] != 5:
            raise ValueError(
                f"segments must be shape (nseg, 5), got {self.segments.shape}"
            )
        if self.EA <= 0:
            raise ValueError(f"EA must be positive, got {self.EA}")
        if self.perimeter <= 0:
            raise ValueError(f"perimeter must be positive, got {self.perimeter}")

        # Validate segment_mask shape if provided
        if self.segment_mask is not None:
            if self.segment_mask.shape[0] != self.segments.shape[0]:
                raise ValueError(
                    f"segment_mask length {self.segment_mask.shape[0]} != "
                    f"n_segments {self.segments.shape[0]}"
                )

        # Validate dowel model if enabled
        if self.enable_dowel and self.dowel_model is None:
            raise ValueError(
                "dowel_model must be provided when enable_dowel=True"
            )


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
    f_u : float, optional
        Steel ultimate stress [Pa] (for Ωy εu calculation). Default: 1.5*f_y
    H : float, optional
        Steel hardening modulus [Pa] (for Ωy εu calculation). Default: 0.01*E_s
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
    f_u: float = 0.0  # Steel ultimate stress [Pa] (Part B3, THESIS PARITY). Default: 1.5*f_y if 0
    H: float = 0.0  # Steel hardening modulus [Pa] (Part B3, THESIS PARITY). Default: 0.01*E_s if 0
    use_secant_stiffness: bool = True  # Part B7: Use secant for stability (DEPRECATED, use tangent_mode)
    tangent_mode: str = "secant_thesis"  # "consistent" | "secant_thesis" (Task D)
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

        # THESIS PARITY: Set steel ultimate properties defaults
        if self.f_u <= 0.0:
            self.f_u = 1.5 * self.f_y  # Typical: fu = 1.5*fy for steel
        if self.H <= 0.0:
            self.H = 0.01 * self.E_s  # Typical: H = 1% of E_s

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

    def compute_yielding_reduction(self, eps_s: float, eps_u: float = None) -> float:
        """Compute steel yielding reduction factor Ωy per Eq. 3.57-3.58.

        Parameters
        ----------
        eps_s : float
            Steel strain (axial, in bar direction) [-]
        eps_u : float, optional
            Ultimate strain. If None, computed from fu and H per thesis spec.

        Returns
        -------
        omega_y : float
            Reduction factor [0, 1] where 1 = no yielding, 0 = full yielding

        Notes
        -----
        From dissertation Eq. 3.57-3.58 (exact):
            eps_y = f_y / E_s
            eps_u = eps_y + (fu - fy) / H  (if H > 0)
                 OR eps_u = fu / E_s        (fallback)
            If eps_s <= eps_y: Ωy = 1
            If eps_y < eps_s:
                ξ = clamp((eps_s - eps_y) / (eps_u - eps_y), 0, +∞)
                Ωy = 1 - 0.85 * (1 - exp(-5 * ξ))
        """
        if not self.enable_yielding_reduction:
            return 1.0

        eps_y = self.f_y / self.E_s  # Yield strain

        # THESIS PARITY: Compute eps_u from fu and H per spec
        if eps_u is None:
            if self.H > 0.0 and self.f_u > self.f_y:
                # Bilinear hardening: εu = εy + (fu - fy) / H
                eps_u = eps_y + (self.f_u - self.f_y) / self.H
            else:
                # Fallback: εu = fu / E_s
                eps_u = self.f_u / self.E_s

        if eps_s <= eps_y:
            return 1.0
        else:
            # Eq. 3.58 (exact form from thesis)
            # ξ = clamp((eps_s - eps_y) / (eps_u - eps_y), 0, +∞)  (no upper clamp per spec)
            xi = max(0.0, (eps_s - eps_y) / max(1e-30, (eps_u - eps_y)))
            omega_y = 1.0 - 0.85 * (1.0 - math.exp(-5.0 * xi))
            # Note: As ξ→∞, Ωy→0.15 (do not clamp unless you want a minimum of 0.15 explicitly)
            return omega_y  # No upper clamp; formula naturally stays in [0.15, 1.0]

    def compute_crack_deterioration(
        self, dist_to_crack: float, w_max: float, t_n_cohesive_stress: float, f_t: float
    ) -> float:
        """Compute crack deterioration factor Ω꜀ per Eq. 3.60 (exact).

        Parameters
        ----------
        dist_to_crack : float
            Distance x from interface point to nearest transverse crack [m]
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
        From dissertation Eq. 3.60 (exact):
            For distance x ≤ 2*l from transverse crack:
                Ω꜀ = 0.5 * (x/l) + (t_n(w_max)/f_t) * (1 - 0.5 * (x/l))
            where:
                l = characteristic length (typically 2 * d_bar or l_ch)
                t_n(w_max)/f_t = FPZ damage indicator (1 = uncracked, 0 = macrocrack)

        For x > 2*l: Ω꜀ = 1 (no deterioration far from crack)
        """
        if not self.enable_crack_deterioration:
            return 1.0

        # Characteristic length for bond deterioration (bar diameter φ)
        phi = self.d_bar  # Bar diameter [m]

        # FPZ state indicator: ratio = t_n(w_max) / f_t
        if f_t > 1e-9:
            ratio_tn_ft = max(0.0, min(1.0, t_n_cohesive_stress / f_t))
        else:
            ratio_tn_ft = 0.0  # Assume fully cracked if no tensile strength

        # Thesis Eq. 3.60 (exact):
        # For x <= 2φ: Ωλ = 0.5 * x / φ,  Ωc = Ωλ + r*(1 - Ωλ)
        # For x > 2φ:  Ωc = 1.0
        if abs(dist_to_crack) <= 2.0 * phi:
            # Within deterioration zone (x ≤ 2φ)
            omega_lambda = 0.5 * abs(dist_to_crack) / phi
            omega_crack = omega_lambda + ratio_tn_ft * (1.0 - omega_lambda)
            return max(0.0, min(1.0, omega_crack))
        else:
            # Beyond deterioration zone (x > 2φ): no deterioration
            return 1.0

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

        # Task D: Tangent mode selection for convergence robustness (Section 5.1)
        # "secant_thesis" = secant moduli per thesis Section 5.1 (replaces tangent with τ/s for stability)
        # "consistent" = consistent tangent (dτ/ds)
        if self.tangent_mode == "secant_thesis" and s_abs > 1e-14:
            dtau_abs = tau_abs / s_abs  # Secant stiffness (thesis approach)
        elif self.tangent_mode == "consistent":
            # Keep tangent as-is (dtau_abs already computed)
            pass
        else:
            # Backward compatibility: use_secant_stiffness flag (deprecated)
            if self.use_secant_stiffness and s_abs > 1e-14:
                dtau_abs = tau_abs / s_abs

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
    use_secant_stiffness: bool = True  # DEPRECATED, use tangent_mode
    tangent_mode: str = "secant_thesis"  # "consistent" | "secant_thesis" (Task D)

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
        self,
        s: float,
        s_max_history: float,
        eps_s: float = 0.0,  # PART B: Steel strain (for compatibility)
        omega_crack: float = 1.0,  # PART B: Crack deterioration (for compatibility)
    ) -> Tuple[float, float]:
        """Compute bond stress and tangent with unloading/reloading.

        Parameters
        ----------
        s : float
            Current slip (signed)
        s_max_history : float
            Maximum absolute slip in history
        eps_s : float, optional
            Steel axial strain (ignored for CustomBondSlipLaw, Part B compatibility)
        omega_crack : float, optional
            Crack deterioration factor (ignored for CustomBondSlipLaw, Part B compatibility)

        Returns
        -------
        tau : float
            Bond stress (signed)
        dtau_ds : float
            Tangent or secant stiffness

        Notes
        -----
        PART B: eps_s and omega_crack are accepted for API compatibility but not used
        by this class. Use BondSlipModelCode2010 for yielding/crack reduction features.
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

        # Tangent mode selection (Task D)
        if self.tangent_mode == "secant_thesis" and s_abs > 1e-14:
            dtau_abs = tau_abs / s_abs
        elif self.tangent_mode == "consistent":
            pass  # Keep tangent as-is
        else:
            # Backward compatibility
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
    use_secant_stiffness: bool = True  # DEPRECATED, use tangent_mode
    tangent_mode: str = "secant_thesis"  # "consistent" | "secant_thesis" (Task D)

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
        self,
        s: float,
        s_max_history: float,
        eps_s: float = 0.0,  # PART B: Steel strain (for compatibility)
        omega_crack: float = 1.0,  # PART B: Crack deterioration (for compatibility)
    ) -> Tuple[float, float]:
        """Compute bond stress and tangent with unloading/reloading.

        Notes
        -----
        PART B: eps_s and omega_crack are accepted for API compatibility but not used.
        """
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

        # Tangent mode selection (Task D)
        if self.tangent_mode == "secant_thesis" and s_abs > 1e-14:
            dtau_abs = tau_abs / s_abs
        elif self.tangent_mode == "consistent":
            pass  # Keep tangent as-is
        else:
            # Backward compatibility
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
    use_secant_stiffness: bool = True  # DEPRECATED, use tangent_mode
    tangent_mode: str = "secant_thesis"  # "consistent" | "secant_thesis" (Task D)

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
        self,
        s: float,
        s_max_history: float,
        eps_s: float = 0.0,  # PART B: Steel strain (for compatibility)
        omega_crack: float = 1.0,  # PART B: Crack deterioration (for compatibility)
    ) -> Tuple[float, float]:
        """Compute bond stress and tangent with unloading/reloading.

        Notes
        -----
        PART B: eps_s and omega_crack are accepted for API compatibility but not used.
        """
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

        # Tangent mode selection (Task D)
        if self.tangent_mode == "secant_thesis" and s_abs > 1e-14:
            dtau_abs = tau_abs / s_abs
        elif self.tangent_mode == "consistent":
            pass  # Keep tangent as-is
        else:
            # Backward compatibility
            if self.use_secant_stiffness and s_abs > 1e-14:
                dtau_abs = tau_abs / s_abs

        return sign * float(tau_abs), float(dtau_abs)


# ------------------------------------------------------------------------------
# Dowel Action Model (Part B5)
# ------------------------------------------------------------------------------

@dataclass
class DowelActionModel:
    """Dowel action model for transverse stress-opening relationship.

    Based on thesis equations 3.62-3.68 (Brenna et al. model).

    This model captures the transverse (normal) stress between steel and concrete
    at crack intersections due to dowel action.

    Parameters
    ----------
    d_bar : float
        Bar diameter [m]
    f_c : float
        Concrete compressive strength [Pa]

    Notes
    -----
    The model uses MPa and mm internally for numerical stability, following
    the thesis formulation. All inputs/outputs are converted to SI units (Pa, m).

    Equations (THESIS PARITY - updated constants):
    (1) sigma(w) = ω̃(w) * k0 * w
    (2) k0 = 599.96 * fc^0.75 / φ          (fc in MPa, φ in mm, k0 in MPa/mm)
    (3) q(w) = 40*w*φ - b
    (4) g(w) = a + sqrt( d^2 * q(w)^2 + c^2 )
    (5) ω̃(w) = [ 1.5 * g(w) ]^(-4/3)
    (6) Tangent: dσ/dw = k0*( ω̃ + w*dω̃/dw )

    Constants (THESIS PARITY):
        a = 0.16
        b = 0.19
        c = 0.67
        d = 0.26

    where fc is in MPa, φ is in mm, w is in mm.
    """

    d_bar: float  # [m]
    f_c: float    # [Pa]

    def sigma_and_tangent(self, w: float) -> Tuple[float, float]:
        """Compute dowel stress and tangent stiffness.

        Parameters
        ----------
        w : float
            Crack opening (normal to bar) [m]

        Returns
        -------
        sigma : float
            Dowel stress [Pa]
        dsigma_dw : float
            Tangent stiffness [Pa/m]

        Notes
        -----
        Uses thesis equations 3.62-3.68 with internal MPa/mm units for stability.
        """
        # Convert to thesis units (MPa, mm)
        fc_mpa = self.f_c / 1e6  # Pa → MPa
        phi_mm = self.d_bar * 1e3  # m → mm
        w_mm = abs(w) * 1e3  # m → mm (use absolute value for opening)

        # THESIS PARITY: Updated constants (exact values from spec)
        a = 0.16
        b = 0.19
        c = 0.67
        d = 0.26

        # Eq: k0 (initial stiffness) [MPa/mm]
        k0 = 599.96 * (fc_mpa ** 0.75) / phi_mm

        # Compute ω̃ (reduction factor) via intermediate functions
        # q(w) = 40*w*φ - b
        q = 40.0 * w_mm * phi_mm - b

        # g(w) = a + sqrt( d^2 * q(w)^2 + c^2 )
        sqrt_arg = d**2 * q**2 + c**2
        g = a + math.sqrt(max(0.0, sqrt_arg))  # Safeguard for numerical stability

        # ω̃(w) = [ 1.5 * g(w) ]^(-4/3)
        Y = 1.5 * g
        if Y > 1e-14:
            omega = Y ** (-4.0/3.0)
        else:
            omega = 0.0  # Avoid division by zero

        # sigma(w) = ω̃ * k0 * w [MPa]
        sigma_mpa = omega * k0 * w_mm

        # Analytical tangent: dσ/dw = k0 * (ω̃ + w * dω̃/dw)
        # Chain rule:
        # dq/dw = 40*φ
        # dg/dw = d^2 * q / sqrt(...) * dq/dw  (if sqrt_arg > 0)
        # dY/dw = 1.5 * dg/dw
        # dω̃/dw = (-4/3) * Y^(-7/3) * dY/dw

        dq_dw = 40.0 * phi_mm

        if sqrt_arg > 1e-14:
            dg_dw = (d**2 * q / math.sqrt(sqrt_arg)) * dq_dw
        else:
            dg_dw = 0.0

        dY_dw = 1.5 * dg_dw

        if Y > 1e-14:
            domega_dw = (-4.0/3.0) * (Y ** (-7.0/3.0)) * dY_dw
        else:
            domega_dw = 0.0

        # dσ/dw = k0 * (ω̃ + w * dω̃/dw) [MPa/mm]
        dsigma_dw_mpa_mm = k0 * (omega + w_mm * domega_dw)

        # Convert back to SI units (Pa, m)
        sigma_pa = sigma_mpa * 1e6  # MPa → Pa
        dsigma_dw_pa_m = dsigma_dw_mpa_mm * 1e9  # MPa/mm → Pa/m

        # Handle sign (w can be negative for compression, but model is for opening only)
        sign = 1.0 if w >= 0.0 else -1.0

        return sign * float(sigma_pa), float(dsigma_dw_pa_m)


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
# NOTE: The Numba kernel has been moved to xfem_clean/numba/kernels_bond_slip.py
#       for better organization and to enable cache=True for faster startup.
#       See bond_slip_assembly_kernel() in that module.

# Legacy kernel removed - now imported from dedicated module


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
    # Dowel action parameters (P4)
    enable_dowel: bool = False,  # Enable dowel action (transverse)
    dowel_model: Optional[DowelActionModel] = None,  # Dowel action constitutive model
    # THESIS PARITY: Crack deterioration context (Ωc, GOAL #1)
    crack_context: Optional[np.ndarray] = None,  # [n_seg, 2]: [crack_dist, tn_ratio] for Ωc
    # TASK 5: Physical dissipation tracking
    u_total_prev: Optional[np.ndarray] = None,  # Displacement at previous time step (for dissipation)
    compute_dissipation: bool = False,  # Enable physical dissipation computation
    # API compatibility: conditional aux return
    return_aux: bool = False,  # If True, return aux dict as 4th value
) -> Tuple[np.ndarray, sp.csr_matrix, BondSlipStateArrays, Dict[str, float]]:
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
    enable_dowel : bool, optional
        Enable dowel action (transverse stress perpendicular to bar). Default: False.
    dowel_model : DowelActionModel, optional
        Dowel action constitutive model. Required if enable_dowel=True.
        If None and enable_dowel=True, creates default model from bond_law params.
    crack_context : np.ndarray, optional
        Crack deterioration context [n_seg, 2] for computing Ωc (THESIS PARITY, GOAL #1).
        crack_context[i, 0] = distance to nearest crack [m]
        crack_context[i, 1] = tn/ft ratio at crack location [-]
        If None, Ωc = 1.0 everywhere (no deterioration).
    u_total_prev : np.ndarray, optional
        Displacement vector at previous time step (for dissipation tracking).
        If None and compute_dissipation=True, dissipation will be zero.
    compute_dissipation : bool
        If True, compute physical dissipation increment using trapezoidal rule.
        Requires u_total_prev to be provided.
    return_aux : bool, optional
        If True, return auxiliary dict as 4th value. If False (default), return
        only 3 values (f_bond, K_bond, bond_states_new).
        Note: Automatically set to True when compute_dissipation=True.

    Returns
    -------
    f_bond : np.ndarray
        Bond interface force vector
    K_bond : sparse.csr_matrix
        Bond interface stiffness matrix
    bond_states_new : BondSlipStateArrays
        Updated bond-slip states (trial)
    aux : dict, optional
        Auxiliary data (only returned if return_aux=True), including:
        - D_bond_inc: Bond dissipation increment [J] (if compute_dissipation=True)
        - D_dowel_inc: Dowel dissipation increment [J] (if enable_dowel and compute_dissipation)
    """
    n_seg = steel_segments.shape[0]
    ndof_total = u_total.shape[0]

    # NOTE: Numba kernel expects BondSlipModelCode2010-like parameters (tau_max, s1, s2, etc.)
    # If bond_law doesn't have these attributes, will automatically fall back to Python.
    # This allows using Numba with compatible bond laws without hardcoding type checks.

    # Dowel action setup (P4)
    if enable_dowel and dowel_model is None:
        # Create default dowel model from bond_law parameters
        if hasattr(bond_law, 'd_bar') and hasattr(bond_law, 'f_cm'):
            dowel_model = DowelActionModel(d_bar=bond_law.d_bar, f_c=bond_law.f_cm)
        else:
            raise ValueError(
                "enable_dowel=True requires dowel_model or bond_law with d_bar and f_cm attributes"
            )

    # Numba kernel now supports dowel action (no need to force Python fallback)

    # Auto-force return_aux=True when compute_dissipation=True (API convenience)
    if compute_dissipation and not return_aux:
        return_aux = True

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

    # Try to create bond_params if using Numba
    # If bond_law doesn't have required attributes, will fall back to Python
    bond_params = None
    if use_numba and NUMBA_AVAILABLE:
        try:
            # Tangent capping for numerical stability (Priority #1)
            # dtau_max is typically set to bond_tangent_cap_factor * median(diag(K_bulk))
            # For now, use a large value (no capping); caller can override via bond_law
            dtau_max = getattr(bond_law, 'dtau_max', 1e20)  # Default: no cap

            # PART B: Get yielding reduction parameters from bond law
            f_y = getattr(bond_law, 'f_y', 500e6)  # Default: 500 MPa
            E_s = getattr(bond_law, 'E_s', 200e9)  # Default: 200 GPa
            f_u = getattr(bond_law, 'f_u', 1.5 * f_y)  # THESIS PARITY: Ultimate stress
            H = getattr(bond_law, 'H', 0.01 * E_s)  # THESIS PARITY: Hardening modulus
            enable_omega_y = 1.0 if getattr(bond_law, 'enable_yielding_reduction', False) else 0.0

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
                f_y,  # PART B: Steel yield stress
                E_s,  # PART B: Steel Young's modulus
                enable_omega_y,  # PART B: Enable yielding reduction flag
                f_u,  # THESIS PARITY: Steel ultimate stress
                H,  # THESIS PARITY: Steel hardening modulus
            ], dtype=float)
        except AttributeError:
            # Bond law doesn't have required attributes (e.g., BilinearBondLaw, BanholzerBondLaw)
            # Fall back to Python assembly which works with any bond law via tau_and_tangent()
            use_numba = False

    if use_numba and NUMBA_AVAILABLE and bond_params is not None:
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

        # Call Numba kernel (from dedicated module with cache=True)
        try:
            # Prepare segment_mask for Numba (needs to be None or contiguous bool array)
            segment_mask_numba = None
            if segment_mask is not None:
                segment_mask_numba = np.ascontiguousarray(segment_mask, dtype=np.bool_)

            # Prepare crack_context for Numba (if provided)
            crack_context_numba = None
            if crack_context is not None:
                crack_context_numba = np.ascontiguousarray(crack_context, dtype=np.float64)

            # Prepare dowel_params for Numba (if dowel enabled)
            dowel_params_numba = None
            if enable_dowel and dowel_model is not None:
                # Extract phi and fc from dowel model
                phi = dowel_model.d_bar  # Bar diameter [m]
                fc = dowel_model.f_c     # Concrete compressive strength [Pa]
                dowel_params_numba = np.array([phi, fc], dtype=np.float64)

            # Prepare u_total_prev for Numba (if dissipation tracking enabled)
            u_total_prev_numba = None
            if compute_dissipation and u_total_prev is not None:
                u_total_prev_numba = np.ascontiguousarray(u_total_prev, dtype=np.float64)

            f_bond, rows, cols, data, s_curr, D_bond_inc, D_dowel_inc = bond_slip_assembly_kernel(
                u_total,
                steel_segments,
                steel_dof_map,
                bond_params,
                s_max_hist,
                steel_EA,
                segment_mask_numba,
                crack_context_numba,
                dowel_params_numba,
                u_total_prev_numba,
                compute_dissipation,
            )
        except Exception as e:
            # Enhanced error reporting for kernel failures
            raise RuntimeError(
                f"Bond-slip Numba kernel failed!\n"
                f"Context: ndof={ndof_total}, n_seg={n_seg}, steel_dof_offset={steel_dof_offset}\n"
                f"This likely indicates an array indexing bug in the Numba kernel.\n"
                f"Original error: {str(e)}"
            ) from e

        # Build sparse K_bond matrix from COO triplets
        # Masked segments are already skipped in the kernel, so no post-processing needed
        K_bond = sp.csr_matrix((data, (rows, cols)), shape=(ndof_total, ndof_total))

        # Update states (trial)
        bond_states_new = BondSlipStateArrays(
            n_segments=n_seg,
            s_max=np.maximum(bond_states.s_max, np.abs(s_curr)),
            s_current=s_curr,
            tau_current=np.zeros(n_seg),  # Computed inline; could extract if needed
        )

        # TASK 5: Dissipation tracking (Numba path - now implemented)
        aux = {"D_bond_inc": D_bond_inc, "D_dowel_inc": D_dowel_inc}

    else:
        # Part A4: Pure Python fallback for debugging
        f_bond, K_bond, bond_states_new, aux = _bond_slip_assembly_python(
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
            # Dowel action (P4)
            enable_dowel=enable_dowel,
            dowel_model=dowel_model,
            # THESIS PARITY: Crack deterioration (Ωc)
            crack_context=crack_context,
            # TASK 5: Physical dissipation tracking
            u_total_prev=u_total_prev,
            compute_dissipation=compute_dissipation,
        )

    # API compatibility: conditionally return aux
    if return_aux:
        return f_bond, K_bond, bond_states_new, aux
    else:
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
    # Dowel action (P4)
    enable_dowel: bool = False,
    dowel_model: Optional[DowelActionModel] = None,
    # THESIS PARITY: Crack deterioration (Ωc, GOAL #1)
    crack_context: Optional[np.ndarray] = None,  # [n_seg, 2]: [crack_dist, tn_ratio]
    # TASK 5: Physical dissipation tracking
    u_total_prev: Optional[np.ndarray] = None,
    compute_dissipation: bool = False,
) -> Tuple[np.ndarray, sp.csr_matrix, BondSlipStateArrays, Dict[str, float]]:
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

    # TASK 5: Dissipation accumulators
    D_bond_inc = 0.0  # Bond dissipation increment [J]
    D_dowel_inc = 0.0  # Dowel dissipation increment [J]

    # Use provided perimeter or compute from bond_law.d_bar
    if perimeter is None:
        perimeter = math.pi * bond_law.d_bar

    for i in range(n_seg):
        # =================================================================
        # PART A FIX: Extract geometry and DOFs BEFORE mask check
        # =================================================================
        # Extract segment data (needed for both bond and steel axial)
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

        # =================================================================
        # PART A FIX: Steel axial contribution ALWAYS assembled (before mask check)
        # =================================================================
        # This is the bar element behavior independent of bond interface.
        # Even if bond is disabled (masked), the steel bar must carry axial loads.
        if steel_EA > 0.0:
            K_steel = steel_EA / L0
            Kxx_s = K_steel * cx * cx
            Kxy_s = K_steel * cx * cy
            Kyy_s = K_steel * cy * cy

            # Compute steel axial displacement (du = u2 - u1)
            du_steel_x = u_s2x - u_s1x
            du_steel_y = u_s2y - u_s1y

            # Axial elongation in bar direction: axial = du · c
            axial = du_steel_x * cx + du_steel_y * cy

            # Axial force: N = (EA/L) * axial
            N_steel = K_steel * axial

            # Internal force contribution: f = N * c at each node
            # Node 1: f1 = -N * c (compression if pulled)
            # Node 2: f2 = +N * c (tension if pulled)
            f_bond[dof_s1x] += -N_steel * cx
            f_bond[dof_s1y] += -N_steel * cy
            f_bond[dof_s2x] += +N_steel * cx
            f_bond[dof_s2y] += +N_steel * cy

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

        # =================================================================
        # PART A FIX: Mask check - Skip ONLY bond shear (and dowel)
        # =================================================================
        # CRITICAL FIX: Masked segments (bond disabled) skip bond/dowel contributions
        # but STILL include steel axial element (already assembled above).
        if segment_mask is not None and segment_mask[i]:
            # Set slip to zero for disabled segments (no bond forces/stiffness)
            s_current[i] = 0.0
            continue

        # =================================================================
        # Bond shear interface (only if NOT masked)
        # =================================================================

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

        # =====================================================================
        # PART B: Compute steel strain (eps_s) for yielding reduction
        # =====================================================================
        eps_s = 0.0
        if steel_EA > 0.0 and L0 > 1e-14:
            # Steel axial displacement (already computed above for steel element)
            # axial = (u_s2 - u_s1) · c
            # eps_s = axial / L0
            du_steel_x = u_s2x - u_s1x
            du_steel_y = u_s2y - u_s1y
            axial = du_steel_x * cx + du_steel_y * cy
            eps_s = axial / L0

        # =====================================================================
        # THESIS PARITY: Compute crack deterioration factor Ωc (GOAL #1)
        # =====================================================================
        omega_crack = 1.0  # Default: no deterioration
        if crack_context is not None and bond_law.enable_crack_deterioration:
            # Extract precomputed crack context for this segment
            crack_dist = crack_context[i, 0]  # Distance to nearest crack [m]
            tn_ratio = crack_context[i, 1]    # tn/ft ratio at crack [-]

            # Compute Ωc using bond_law method
            # Note: We don't have w_max or tn directly here; use tn_ratio = tn/ft
            # Reconstruct tn from ratio (requires ft from bond_law)
            if hasattr(bond_law, 'f_cm'):
                # Estimate ft from f_cm (crude approximation: ft ≈ 0.3*sqrt(f_cm) in MPa)
                ft_approx = 0.3 * math.sqrt(bond_law.f_cm / 1e6) * 1e6  # [Pa]
                tn_cohesive = tn_ratio * ft_approx
            else:
                # Fallback: assume tn_ratio is already meaningful
                tn_cohesive = 3e6  # Default: 3 MPa
                tn_ratio_clamped = max(0.0, min(1.0, tn_ratio))

            # Call compute_crack_deterioration (note: ft is estimated above)
            omega_crack = bond_law.compute_crack_deterioration(
                dist_to_crack=crack_dist,
                w_max=0.0,  # Not used in current formula (only dist and tn matter)
                t_n_cohesive_stress=tn_cohesive,
                f_t=ft_approx if hasattr(bond_law, 'f_cm') else 3e6
            )

        # BLOQUE C: Optional slip smoothing (regularization near s≈0)
        s_eval = s
        if bond_s_eps > 0.0:
            # Evaluate tau at smoothed slip: s_eff = sqrt(s^2 + eps^2)
            s_abs = abs(s)
            s_sign = 1.0 if s >= 0 else -1.0
            s_eff = math.sqrt(s_abs**2 + bond_s_eps**2)
            s_eval = s_sign * s_eff

        # NUMBA PARITY: Apply same C1-continuous regularization as Numba kernel
        # This prevents singular tangent at s=0 and improves conditioning
        # (matches kernels_bond_slip.py lines 288-304)
        apply_regularization = True  # Match Numba kernel behavior
        if apply_regularization and hasattr(bond_law, 's1') and hasattr(bond_law, 'alpha'):
            s_abs_eval = abs(s_eval)
            s_sign = 1.0 if s_eval >= 0 else -1.0
            s_reg = 0.5 * bond_law.s1

            # Check if we're in the regularized region (loading on envelope with small slip)
            if s_abs_eval < s_reg and s_abs_eval >= s_max - 1e-14:
                # Loading envelope in regularized region: use linear branch
                # k0 = tangent at s_reg to ensure C1 continuity
                k0 = bond_law.tau_max * bond_law.alpha / bond_law.s1 * ((s_reg / bond_law.s1) ** (bond_law.alpha - 1.0))
                tau = s_sign * k0 * s_abs_eval
                dtau_ds = k0

                # Apply Part B reduction factors (Ωy * Ωc) to match Numba kernel
                omega_y = bond_law.compute_yielding_reduction(eps_s) if hasattr(bond_law, 'compute_yielding_reduction') else 1.0
                tau *= omega_y * omega_crack
                dtau_ds *= omega_y * omega_crack
            else:
                # Use standard bond law evaluation (power law or secant)
                tau, dtau_ds = bond_law.tau_and_tangent(s_eval, s_max, eps_s=eps_s, omega_crack=omega_crack)
        else:
            # Fallback: use standard bond law evaluation
            tau, dtau_ds = bond_law.tau_and_tangent(s_eval, s_max, eps_s=eps_s, omega_crack=omega_crack)

        # BLOQUE C: Optional tangent capping (prevent excessive stiffness)
        if bond_k_cap is not None and dtau_ds > bond_k_cap:
            dtau_ds = bond_k_cap

        # TASK 5: Bond dissipation tracking (trapezoidal rule)
        if compute_dissipation and u_total_prev is not None:
            # Compute slip at previous time step
            u_c1x_old = u_total_prev[dof_c1x]
            u_c1y_old = u_total_prev[dof_c1y]
            u_c2x_old = u_total_prev[dof_c2x]
            u_c2y_old = u_total_prev[dof_c2y]
            u_s1x_old = u_total_prev[dof_s1x]
            u_s1y_old = u_total_prev[dof_s1y]
            u_s2x_old = u_total_prev[dof_s2x]
            u_s2y_old = u_total_prev[dof_s2y]

            u_c_mid_x_old = 0.5 * (u_c1x_old + u_c2x_old)
            u_c_mid_y_old = 0.5 * (u_c1y_old + u_c2y_old)
            u_s_mid_x_old = 0.5 * (u_s1x_old + u_s2x_old)
            u_s_mid_y_old = 0.5 * (u_s1y_old + u_s2y_old)

            du_x_old = u_s_mid_x_old - u_c_mid_x_old
            du_y_old = u_s_mid_y_old - u_c_mid_y_old
            s_old = du_x_old * cx + du_y_old * cy

            # Compute tau at old slip using committed state (do not update state history)
            # Use committed s_max from bond_states (not the updated s_max)
            s_max_committed = float(bond_states.s_max[i])

            # Compute eps_s at old state
            eps_s_old = 0.0
            if steel_EA > 0.0 and L0 > 1e-14:
                du_steel_x_old = u_s2x_old - u_s1x_old
                du_steel_y_old = u_s2y_old - u_s1y_old
                axial_old = du_steel_x_old * cx + du_steel_y_old * cy
                eps_s_old = axial_old / L0

            # Evaluate tau_old using same bond law (with committed s_max, no viscosity)
            tau_old, _ = bond_law.tau_and_tangent(s_old, s_max_committed, eps_s=eps_s_old, omega_crack=omega_crack)

            # Trapezoidal dissipation:
            # ΔD = 0.5 * (tau_old + tau_new) * (s_new - s_old) * perimeter * L0
            d_slip = s - s_old
            diss_local = 0.5 * (tau_old + tau) * d_slip * perimeter * L0
            D_bond_inc += diss_local

        # Bond force
        F_bond = tau * perimeter * L0

        # Distribute bond force to nodes (tangential direction)
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

        # Dowel action (P4): transverse stress perpendicular to bar
        if enable_dowel and dowel_model is not None:
            # Normal direction (perpendicular to bar): n = (-cy, cx)
            nx = -cy
            ny = cx

            # Opening (normal relative displacement): w = du·n
            w = du_x * nx + du_y * ny

            # Use max(w, 0) to avoid strange behavior in compression
            w_pos = max(w, 0.0)

            # Compute dowel stress and tangent
            sigma_dowel, dsigma_dw = dowel_model.sigma_and_tangent(w_pos)

            # TASK 5: Dowel dissipation tracking (trapezoidal rule)
            if compute_dissipation and u_total_prev is not None:
                # Compute opening at previous time step
                du_x_old = u_s_mid_x_old - u_c_mid_x_old  # Already computed above
                du_y_old = u_s_mid_y_old - u_c_mid_y_old
                w_old = du_x_old * nx + du_y_old * ny
                w_old_pos = max(w_old, 0.0)

                # Compute sigma_old
                sigma_dowel_old, _ = dowel_model.sigma_and_tangent(w_old_pos)

                # Trapezoidal dissipation:
                # ΔD = 0.5 * (sigma_old + sigma_new) * (w_new - w_old) * perimeter * L0
                d_opening = w_pos - w_old_pos
                diss_dowel_local = 0.5 * (sigma_dowel_old + sigma_dowel) * d_opening * perimeter * L0
                D_dowel_inc += diss_dowel_local

            # Convert traction to force: F = sigma * perimeter * L0
            F_dowel = sigma_dowel * perimeter * L0

            # Distribute to nodes (normal direction)
            Fx_dowel_s = F_dowel * nx
            Fy_dowel_s = F_dowel * ny
            Fx_dowel_c = -Fx_dowel_s
            Fy_dowel_c = -Fy_dowel_s

            f_bond[dof_s1x] += 0.5 * Fx_dowel_s
            f_bond[dof_s1y] += 0.5 * Fy_dowel_s
            f_bond[dof_s2x] += 0.5 * Fx_dowel_s
            f_bond[dof_s2y] += 0.5 * Fy_dowel_s

            f_bond[dof_c1x] += 0.5 * Fx_dowel_c
            f_bond[dof_c1y] += 0.5 * Fy_dowel_c
            f_bond[dof_c2x] += 0.5 * Fx_dowel_c
            f_bond[dof_c2y] += 0.5 * Fy_dowel_c

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

        # Dowel stiffness (P4): K_dowel = (dsigma_dw * perimeter * L0) * (g_w ⊗ g_w)
        if enable_dowel and dowel_model is not None:
            # Gradient of opening w wrt DOFs: g_w = [∂w/∂u]
            # w = du·n where n = (-cy, cx) and du = u_s - u_c
            # Concrete node 1: -0.5 * n
            g_w_c1x = -0.5 * nx
            g_w_c1y = -0.5 * ny
            # Concrete node 2: -0.5 * n
            g_w_c2x = -0.5 * nx
            g_w_c2y = -0.5 * ny
            # Steel node 1: +0.5 * n
            g_w_s1x = +0.5 * nx
            g_w_s1y = +0.5 * ny
            # Steel node 2: +0.5 * n
            g_w_s2x = +0.5 * nx
            g_w_s2y = +0.5 * ny

            # Dowel stiffness scalar
            K_dowel = dsigma_dw * perimeter * L0

            # Gradient list for dowel
            g_w = [g_w_c1x, g_w_c1y, g_w_c2x, g_w_c2y, g_w_s1x, g_w_s1y, g_w_s2x, g_w_s2y]

            # Assemble dowel stiffness block (8×8 outer product)
            for a in range(8):
                if dofs[a] < 0:
                    continue
                for b in range(8):
                    if dofs[b] < 0:
                        continue
                    rows.append(dofs[a])
                    cols.append(dofs[b])
                    data.append(K_dowel * g_w[a] * g_w[b])

        # NOTE: Steel axial stiffness and internal force now assembled BEFORE mask check (lines 1420-1462)
        # This ensures masked segments (bond disabled) still include steel bar behavior.

    K_bond_sp = sp.csr_matrix((data, (rows, cols)), shape=(ndof, ndof))

    # Update states
    bond_states_new = BondSlipStateArrays(
        n_segments=n_seg,
        s_max=np.maximum(bond_states.s_max, np.abs(s_current)),
        s_current=s_current,
        tau_current=np.zeros(n_seg),
    )

    # Auxiliary data (TASK 5: dissipation)
    aux = {
        "D_bond_inc": float(D_bond_inc),
        "D_dowel_inc": float(D_dowel_inc),
    }

    # Always return 4 values (conditional return handled by caller)
    return f_bond, K_bond_sp, bond_states_new, aux


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
# THESIS PARITY: Crack Context Precomputation (Ωc, GOAL #1)
# ------------------------------------------------------------------------------

def precompute_crack_context_for_bond(
    steel_segments: np.ndarray,
    nodes: Optional[np.ndarray] = None,
    cracks: Optional[List[Any]] = None,  # List of XFEMCrack objects
    cohesive_states: Optional[Any] = None,  # CohesiveStateArrays or None
    cohesive_law: Optional[Any] = None,  # CohesiveLaw for tn(wmax) evaluation
    ft: float = 3e6,  # Concrete tensile strength [Pa]
    phi_tolerance: float = None,  # Tolerance for "transverse crack" detection (defaults to 0.5*d_bar)
) -> np.ndarray:
    """Precompute crack deterioration context for bond segments (THESIS PARITY, GOAL #1).

    This function computes the crack_context array needed by assemble_bond_slip
    to evaluate the crack deterioration factor Ωc for each bond segment.

    Algorithm (per thesis Eq. 3.60-3.61):
    -------
    1. For each bond segment i:
       - Compute midpoint p_i and bar axis direction c_i
       - Project p_i onto bar axis line

    2. Find nearest "transverse crack":
       - Check each crack for intersection with bar
       - Crack is "transverse" if:
         (a) Intersects bar line within tolerance (φ/2)
         (b) Crack tangent not parallel to bar axis (angle > 30°)
       - Compute signed distance x_i along bar axis

    3. Extract cohesive state at crack:
       - Get wmax from cohesive history at crack location
       - Compute tn = cohesive_law(wmax, ...)
       - Compute r_i = clamp(tn / ft, 0, 1)

    4. Return [x_i, r_i] for each segment

    Parameters
    ----------
    steel_segments : np.ndarray
        [n_seg, 5]: [n1, n2, L0, cx, cy] bond segment geometry
    nodes : np.ndarray
        [n_nodes, 2]: node coordinates [m]
    cracks : list of XFEMCrack, optional
        Active crack objects with geometry (p0, pt, tvec, nvec methods).
        If None, returns crack_context with Ωc=1 everywhere (no cracks).
    cohesive_states : CohesiveStateArrays, optional
        Cohesive state arrays with delta_max[k, e, gp] for each crack/element/GP.
        If None, assumes r=0 (no cohesive traction).
    cohesive_law : CohesiveLaw, optional
        Cohesive law for evaluating tn(wmax). Required if cohesive_states provided.
    ft : float, optional
        Concrete tensile strength [Pa] for computing r = tn/ft ratio.
    phi_tolerance : float, optional
        Tolerance for detecting crack intersection with bar [m].
        Defaults to 0.5 * estimated_bar_diameter if None.

    Returns
    -------
    crack_context : np.ndarray
        [n_seg, 2] array where:
        crack_context[i, 0] = signed distance to nearest crack along bar axis [m]
                              (positive = crack ahead, negative = crack behind)
        crack_context[i, 1] = r = tn(wmax)/ft ratio at nearest crack [-], clamped to [0, 1]

    Notes
    -----
    - If no crack intersects the bar within tolerance: x = +inf, r = 1.0 → Ωc = 1.0 (no deterioration)
    - For multiple crack intersections: uses nearest crack along bar axis
    - For cracks that don't intersect: distance computed to nearest point on crack line
    """
    n_seg = steel_segments.shape[0]
    crack_context = np.zeros((n_seg, 2), dtype=float)

    # Default: no cracks (large distance → Ωc = 1.0)
    crack_context[:, 0] = 1e10  # Very large distance
    crack_context[:, 1] = 1.0   # r = 1.0 → no deterioration

    # If nodes is None, return default context (no deterioration)
    if nodes is None:
        return crack_context

    if cracks is None or len(cracks) == 0:
        return crack_context  # No cracks: Ωc = 1 everywhere

    # Estimate tolerance if not provided (use 0.5 * typical bar diameter)
    if phi_tolerance is None:
        # Estimate bar diameter from segment length (crude heuristic)
        typical_segment_length = np.median(steel_segments[:, 2]) if n_seg > 0 else 0.1
        phi_tolerance = 0.5 * max(0.01, min(0.02, typical_segment_length * 0.1))  # 10-20mm

    # Process each bond segment
    for i in range(n_seg):
        n1 = int(steel_segments[i, 0])
        n2 = int(steel_segments[i, 1])
        L0 = steel_segments[i, 2]
        cx = steel_segments[i, 3]  # Bar axis unit vector
        cy = steel_segments[i, 4]

        # Bond segment midpoint and bar direction
        if n1 >= nodes.shape[0] or n2 >= nodes.shape[0]:
            # Node index out of bounds - skip this segment
            continue

        p1 = nodes[n1]
        p2 = nodes[n2]
        p_mid = 0.5 * (p1 + p2)
        c_bar = np.array([cx, cy], dtype=float)  # Bar axis direction

        # Find nearest transverse crack
        min_dist_signed = 1e10
        nearest_r = 1.0
        found_crack = False

        for k, crack in enumerate(cracks):
            if not hasattr(crack, 'active') or not crack.active:
                continue

            # Crack geometry
            p0_crack = crack.p0()  # Crack start point
            pt_crack = crack.pt()  # Crack tip
            t_crack = crack.tvec()  # Crack tangent vector
            n_crack = crack.nvec()  # Crack normal vector

            # Check if crack is "transverse" to bar (not parallel)
            # Angle between crack tangent and bar axis
            cos_angle = abs(np.dot(t_crack, c_bar))
            if cos_angle > 0.866:  # cos(30°) ≈ 0.866 → skip nearly parallel cracks
                continue

            # Find intersection point between infinite crack line and bar line
            # Bar line: P = p_mid + s * c_bar
            # Crack line: Q = p0_crack + t * t_crack
            # Solve: p_mid + s*c_bar = p0_crack + t*t_crack
            #   => [c_bar, -t_crack] @ [s, t]^T = p0_crack - p_mid

            A = np.column_stack([c_bar, -t_crack])
            b = p0_crack - p_mid

            # Check if lines are parallel (det(A) ≈ 0)
            det_A = A[0,0]*A[1,1] - A[0,1]*A[1,0]
            if abs(det_A) < 1e-12:
                # Lines are parallel - compute perpendicular distance
                # Distance from p_mid to infinite crack line
                dist_perp = abs(np.dot(n_crack, p_mid - p0_crack))
                if dist_perp < phi_tolerance:
                    # Bar runs along crack - use midpoint projection
                    s_intersect = np.dot(p_mid - p0_crack, t_crack)
                    dist_signed = 0.0  # Crack at bar location
                else:
                    continue  # Too far from crack
            else:
                # Solve for intersection parameters
                s_t = np.linalg.solve(A, b)
                s_intersect = s_t[0]  # Distance along bar axis to intersection
                t_intersect = s_t[1]  # Distance along crack axis to intersection

                # Check if intersection is within active crack segment
                s_crack_tip = crack.s_tip()  # Length of active crack
                if t_intersect < -1e-6 or t_intersect > s_crack_tip + 1e-6:
                    # Intersection point is outside active crack segment
                    # Compute distance to nearest endpoint instead
                    dist_to_p0 = np.linalg.norm(p_mid - p0_crack)
                    dist_to_pt = np.linalg.norm(p_mid - pt_crack)
                    if dist_to_p0 < phi_tolerance or dist_to_pt < phi_tolerance:
                        # Near crack endpoint
                        s_to_p0 = np.dot(p0_crack - p_mid, c_bar)
                        s_to_pt = np.dot(pt_crack - p_mid, c_bar)
                        s_intersect = s_to_p0 if dist_to_p0 < dist_to_pt else s_to_pt
                        dist_signed = s_intersect
                    else:
                        continue  # Outside crack and not close enough
                else:
                    # Valid intersection within active crack
                    dist_signed = s_intersect

            # Check if this is the nearest crack
            if abs(dist_signed) < abs(min_dist_signed):
                min_dist_signed = dist_signed
                found_crack = True

                # Extract cohesive state at this crack location
                if cohesive_states is not None and cohesive_law is not None:
                    # Get wmax from cohesive state
                    # Cohesive states are indexed by [k, e, gp]
                    # For simplicity, use maximum wmax across all elements/GPs for this crack
                    try:
                        if hasattr(cohesive_states, 'delta_max') and cohesive_states.delta_max.shape[0] > k:
                            wmax_vals = cohesive_states.delta_max[k, :, :]
                            wmax = float(np.max(wmax_vals)) if wmax_vals.size > 0 else 0.0

                            # Evaluate cohesive law at wmax
                            # Use cohesive_update to get current traction
                            from xfem_clean.cohesive_laws import CohesiveState, cohesive_update
                            dummy_state = CohesiveState(delta_max=wmax, damage=0.0)
                            tn, _, _ = cohesive_update(cohesive_law, wmax, dummy_state, visc_damp=0.0)

                            # Compute r = tn/ft ratio
                            nearest_r = max(0.0, min(1.0, tn / max(1e-9, ft)))
                        else:
                            nearest_r = 0.0  # No state data
                    except Exception:
                        nearest_r = 0.0  # Error in state extraction, conservative value
                else:
                    nearest_r = 0.0  # No cohesive state provided

        # Store results
        if found_crack:
            crack_context[i, 0] = min_dist_signed
            crack_context[i, 1] = nearest_r
        # else: keep default values (large distance, r=1.0)

    return crack_context


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
