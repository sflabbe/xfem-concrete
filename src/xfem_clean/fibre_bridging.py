"""
Fibre bridging contribution for fibre-reinforced concrete.

Implements additional traction in cohesive crack due to fibre pull-out,
using Banholzer bond-slip law for individual fibres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


@dataclass
class FibreBridgingConfig:
    """Configuration for fibre bridging model.

    Parameters
    ----------
    E_fibre : float
        Fibre Young's modulus (Pa)
    d_fibre : float
        Fibre diameter (m)
    L_fibre : float
        Fibre length (m)
    density_m2 : float
        Fibre density (fibres/m²) in crack zone
    thickness : float
        Specimen thickness (m)
    orientation_mean_deg : float
        Mean fibre orientation relative to crack normal (degrees)
    orientation_std_deg : float
        Standard deviation of orientation (degrees). If 0, all fibres at mean angle.
    bond_law_params : dict
        Banholzer bond law parameters:
        - s0 (m): end of rising branch
        - a (-): softening multiplier
        - tau1 (Pa): peak in rising
        - tau2 (Pa): peak in softening
        - tau_f (Pa): residual
    explicit_fraction : float
        Fraction of fibres to model explicitly (for computational efficiency).
        Actual forces are scaled by 1/explicit_fraction.
        Default: 0.1 (model 10%, scale forces by 10)
    random_seed : int
        Random seed for fibre orientation generation
    """

    E_fibre: float
    d_fibre: float
    L_fibre: float
    density_m2: float
    thickness: float
    orientation_mean_deg: float = 0.0
    orientation_std_deg: float = 30.0  # Default spread
    bond_law_params: dict = None
    explicit_fraction: float = 0.1
    random_seed: int = 42

    def __post_init__(self):
        """Validate parameters."""
        if self.E_fibre <= 0:
            raise ValueError("Fibre modulus must be positive")
        if self.d_fibre <= 0:
            raise ValueError("Fibre diameter must be positive")
        if self.L_fibre <= 0:
            raise ValueError("Fibre length must be positive")
        if self.density_m2 <= 0:
            raise ValueError("Fibre density must be positive")
        if self.thickness <= 0:
            raise ValueError("Thickness must be positive")
        if not (0 < self.explicit_fraction <= 1.0):
            raise ValueError("explicit_fraction must be in (0, 1]")

        # Set default bond law if not provided
        if self.bond_law_params is None:
            # Default Banholzer parameters (example)
            self.bond_law_params = {
                's0': 0.01e-3,  # 0.01 mm in meters
                'a': 20.0,
                'tau1': 5.0e6,  # 5 MPa in Pa
                'tau2': 3.0e6,  # 3 MPa
                'tau_f': 2.25e6,  # 2.25 MPa
            }


def banholzer_tau_and_tangent(s_abs: float, params: dict) -> Tuple[float, float]:
    """Compute bond stress and tangent from Banholzer law.

    Banholzer law (5 parameters):
        tau(s) = (tau1/s0)*s                                   if 0 <= s <= s0
        tau(s) = tau2 - (tau2-tau_f)*(s-s0)/((a-1)*s0)         if s0 < s <= a*s0
        tau(s) = tau_f                                         if s > a*s0

    Parameters
    ----------
    s_abs : float
        Absolute slip (m)
    params : dict
        Banholzer parameters: s0, a, tau1, tau2, tau_f

    Returns
    -------
    tau : float
        Bond stress (Pa)
    dtau_ds : float
        Tangent stiffness (Pa/m)
    """
    s0 = params['s0']
    a = params['a']
    tau1 = params['tau1']
    tau2 = params['tau2']
    tau_f = params['tau_f']

    if s_abs <= s0:
        # Rising: τ = (tau1/s0)*s
        if s_abs < 1e-16:
            tau = 0.0
            dtau_ds = tau1 / s0
        else:
            tau = (tau1 / s0) * s_abs
            dtau_ds = tau1 / s0

    elif s_abs <= a * s0:
        # Softening: τ = tau2 - (tau2-tau_f)*(s-s0)/((a-1)*s0)
        s_soft_length = (a - 1.0) * s0
        tau = tau2 - (tau2 - tau_f) * (s_abs - s0) / s_soft_length
        dtau_ds = -(tau2 - tau_f) / s_soft_length

    else:
        # Residual: τ = tau_f
        tau = tau_f
        dtau_ds = 0.0

    return float(tau), float(dtau_ds)


def fibre_traction_tangent(
    w_n: float,
    delta_l: float,
    cfg: FibreBridgingConfig,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Compute additional traction and tangent from fibre bridging.

    Parameters
    ----------
    w_n : float
        Normal crack opening (m). Only positive values contribute.
    delta_l : float
        Patch length represented by this cohesive point (m)
    cfg : FibreBridgingConfig
        Fibre bridging configuration
    rng : np.random.Generator, optional
        Random number generator for orientation sampling. If None, creates one with cfg.random_seed.

    Returns
    -------
    t_add : float
        Additional normal traction from fibres (Pa)
    k_add : float
        Additional tangent stiffness d(t_add)/d(w_n) (Pa/m)

    Notes
    -----
    Model:
    - Crack opening w_n projects onto fibre axis: δ = w_n * |cos(θ)|
    - Slip per side: s_eff = δ / 2
    - Axial force per fibre: N_f = p * L_e * τ(s_eff)
      where p = π*d (perimeter), L_e = L/2 (embedment per side)
    - Normal component: t_fibre = N_f * |cos(θ)| / A_patch
    - Patch area: A_patch = thickness * delta_l
    - Number of fibres in patch: n_patch = density_m2 * A_patch * explicit_fraction
    - Forces are scaled by (1 / explicit_fraction) to represent all fibres
    """

    # Only positive openings contribute
    if w_n <= 0.0:
        return 0.0, 0.0

    # Initialize RNG if not provided
    if rng is None:
        rng = np.random.default_rng(cfg.random_seed)

    # Patch area
    A_patch = cfg.thickness * delta_l

    # Number of explicit fibres in this patch
    n_patch_raw = cfg.density_m2 * A_patch * cfg.explicit_fraction
    n_patch = max(1, int(round(n_patch_raw)))

    # Scale factor to represent all fibres
    scale_factor = 1.0 / cfg.explicit_fraction

    # Fibre geometry
    p_fibre = math.pi * cfg.d_fibre  # Perimeter
    L_e = cfg.L_fibre / 2.0  # Embedment per side

    # Generate fibre orientations (angle relative to crack normal)
    # θ = 0 means fibre aligned with crack normal (perpendicular to crack)
    # θ = 90° means fibre parallel to crack (no contribution)
    if cfg.orientation_std_deg > 0:
        # Normal distribution around mean
        theta_deg = rng.normal(
            loc=cfg.orientation_mean_deg,
            scale=cfg.orientation_std_deg,
            size=n_patch,
        )
    else:
        # All fibres at mean angle
        theta_deg = np.full(n_patch, cfg.orientation_mean_deg)

    # Clamp to [0, 90] to avoid negative contributions
    theta_deg = np.clip(theta_deg, 0.0, 90.0)
    theta_rad = np.deg2rad(theta_deg)
    cos_theta = np.cos(theta_rad)
    abs_cos_theta = np.abs(cos_theta)

    # Crack opening projected onto fibre axis
    delta = w_n * abs_cos_theta  # [n_patch]

    # Slip per side
    s_eff = delta / 2.0  # [n_patch]

    # Compute bond stress for each fibre
    tau = np.zeros(n_patch, dtype=float)
    dtau_ds = np.zeros(n_patch, dtype=float)

    for i in range(n_patch):
        tau[i], dtau_ds[i] = banholzer_tau_and_tangent(s_eff[i], cfg.bond_law_params)

    # Axial force per fibre
    N_f = p_fibre * L_e * tau  # [n_patch]

    # Normal component of force
    F_n = N_f * abs_cos_theta  # [n_patch]

    # Total normal force (summed over explicit fibres, scaled)
    F_total = scale_factor * np.sum(F_n)

    # Traction
    t_add = F_total / A_patch

    # Tangent: d(t_add)/d(w_n)
    # Chain rule: dN_f/dw = p*L_e*(dτ/ds)*(ds/dw)
    # ds_eff/dw = 0.5 * |cos(θ)|
    # dF_n/dw = dN_f/dw * |cos(θ)| = p*L_e*(dτ/ds)*0.5*|cos(θ)|^2

    dF_n_dw = p_fibre * L_e * dtau_ds * 0.5 * (abs_cos_theta ** 2)  # [n_patch]
    dF_total_dw = scale_factor * np.sum(dF_n_dw)
    k_add = dF_total_dw / A_patch

    return float(t_add), float(k_add)


# =============================================================================
# Conversion from CaseConfig to FibreBridgingConfig
# =============================================================================

def fibre_config_from_case(fibre_reinf, thickness_m: float) -> FibreBridgingConfig:
    """Convert FibreReinforcement from case_config to FibreBridgingConfig.

    Parameters
    ----------
    fibre_reinf : FibreReinforcement
        Fibre reinforcement configuration from case
    thickness_m : float
        Specimen thickness (m)

    Returns
    -------
    cfg : FibreBridgingConfig
        Configuration for fibre bridging model
    """
    # Extract fibre properties
    fibre = fibre_reinf.fibre
    bond_law = fibre_reinf.bond_law

    # Convert units: mm → m, MPa → Pa, fibres/cm² → fibres/m²
    E_fibre = fibre.E * 1e6  # MPa → Pa
    d_fibre = fibre.diameter * 1e-3  # mm → m
    L_fibre = fibre.length * 1e-3  # mm → m
    density_cm2 = fibre.density  # fibres/cm²
    density_m2 = density_cm2 * 1e4  # fibres/cm² → fibres/m²

    # Bond law parameters (Banholzer)
    bond_params = {
        's0': bond_law.s0 * 1e-3,  # mm → m
        'a': bond_law.a,
        'tau1': bond_law.tau1 * 1e6,  # MPa → Pa
        'tau2': bond_law.tau2 * 1e6,
        'tau_f': bond_law.tau_f * 1e6,
    }

    # Orientation parameters
    orientation_mean_deg = fibre.orientation_deg
    # Assume some spread if not specified (fibres aren't perfectly aligned)
    orientation_std_deg = 30.0  # Default: ±30° spread

    # Explicit fraction (for speed, model 10% of fibres and scale forces by 10)
    explicit_fraction = 0.1

    # Random seed
    random_seed = fibre_reinf.random_seed

    cfg = FibreBridgingConfig(
        E_fibre=E_fibre,
        d_fibre=d_fibre,
        L_fibre=L_fibre,
        density_m2=density_m2,
        thickness=thickness_m,
        orientation_mean_deg=orientation_mean_deg,
        orientation_std_deg=orientation_std_deg,
        bond_law_params=bond_params,
        explicit_fraction=explicit_fraction,
        random_seed=random_seed,
    )

    return cfg
