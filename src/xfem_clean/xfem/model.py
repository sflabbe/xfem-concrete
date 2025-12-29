"""Model/parameter container for the XFEM beam examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class XFEMModel:
    L: float
    H: float
    b: float
    E: float
    nu: float
    ft: float
    Gf: float

    steel_A_total: float
    steel_E: float

    # Bulk constitutive model selector
    bulk_material: str = "elastic"  # elastic | dp | cdp

    # Drucker–Prager parameters (used when bulk_material == 'dp')
    dp_phi_deg: float = 30.0
    dp_cohesion: float = 2.0e6  # Pa
    dp_H: float = 0.0  # isotropic hardening modulus (Pa)

    # Concrete CDP-lite parameters (used when bulk_material == 'cdp')
    fc: float = 32.0e6  # Pa (compression strength magnitude)
    cdp_phi_deg: float = 30.0
    cdp_H: float = 0.0

    # CDP Generator calibration (used when bulk_material == 'cdp')
    cdp_use_generator: bool = True
    cdp_class: Optional[str] = None  # e.g. "C20/25" (if set, overrides fc for generator f_cm)
    cdp_ec1: float = 0.0022          # strain at peak compressive strength
    cdp_eclim: float = 0.0035        # ultimate strain
    cdp_strain_rate: float = 0.0     # 1/s (quasi-static default)
    cdp_use_dilation_angle: bool = True  # if True, set cdp_phi_deg = dilation angle from generator
    cdp_override_E: bool = True
    cdp_override_ft: bool = True
    cdp_override_Gf: bool = True

    # Stored calibration summary (filled when generator is used)
    cdp_calibrated: bool = False
    cdp_dilation_angle: Optional[float] = None
    cdp_Kc: Optional[float] = None
    cdp_fbfc: Optional[float] = None
    cdp_E_mpa: Optional[float] = None
    cdp_fctm_mpa: Optional[float] = None
    cdp_Gf_nmm: Optional[float] = None
    cdp_l0_m: Optional[float] = None
    cdp_lch_mm_used: Optional[float] = None

    # CDP potential parameters (Abaqus-like)
    cdp_psi_deg: Optional[float] = None   # dilation angle psi [deg] (if calibrated, equals generator dilation angle)
    cdp_ecc: float = 0.1                  # eccentricity for hyperbolic potential (Abaqus default ~0.1)

    # CDP uniaxial hardening/damage tables (filled by cdp_generator calibration; SI units)
    # Tension: crack opening w [m], effective stress [Pa], damage dt [-]
    cdp_w_tab_m: Optional[Tuple[float, ...]] = None
    cdp_sig_t_tab_pa: Optional[Tuple[float, ...]] = None
    cdp_dt_tab: Optional[Tuple[float, ...]] = None

    # Compression: inelastic strain eps_in [-], effective stress [Pa], damage dc [-]
    cdp_eps_in_c_tab: Optional[Tuple[float, ...]] = None
    cdp_sig_c_tab_pa: Optional[Tuple[float, ...]] = None
    cdp_dc_tab: Optional[Tuple[float, ...]] = None
    # Reinforcement material
    steel_fy: float = 0.0
    steel_fu: float = 0.0
    steel_Eh: float = 0.0
    # legacy aliases (old runner names)
    steel_sigy: float = 0.0
    steel_H: float = 0.0

    cover: float = 0.05

    # Small hysteresis factor for crack initiation to avoid chatter.
    ft_initiation_factor: float = 1.001

    # solver params
    newton_maxit: int = 25
    newton_tol_r: float = 1e-6
    newton_tol_rel: float = 1e-8
    newton_beta: float = 1e-3
    newton_tol_du: float = 1e-8
    line_search: bool = True
    max_subdiv: int = 12

    # Optional Numba acceleration (Phase 2/3)
    use_numba: bool = False

    # crack controls
    crack_margin: float = 0.3
    crack_rho: float = 0.25
    crack_tip_stop_y: Optional[float] = None
    arrest_at_half_height: bool = True

    # inner cracking loop / adaptivity
    crack_max_inner: int = 8
    crack_seg_length: Optional[float] = None
    dominant_crack: bool = True
    dominant_window: Tuple[float, float] = (0.40, 0.60)
    cand_mode: str = "dominant"
    cand_windows_three: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (0.30, 0.40), (0.45, 0.55), (0.60, 0.70)
    )
    cand_windows: Optional[Tuple[Tuple[float, float], ...]] = None

    debug_substeps: bool = False
    debug_newton: bool = False
    cand_ymax_factor: float = 2.0
    crack_dt_tip: float = 0.6
    tip_enr_radius: float = 0.20
    k_stab: float = 1e-6

    # cohesive penalty scaling
    Kn_factor: float = 0.1
    visc_damp: float = 0.0
    lch: float = 0.05

    # Load application patch (half-width in meters)
    load_halfwidth: float = 0.0

    def __post_init__(self):
        """Backwards-compatible parameter normalization."""

        # Normalize bulk material selector
        bm = (self.bulk_material or "elastic").strip().lower()
        aliases = {
            "linear": "elastic",
            "lineal": "elastic",
            "le": "elastic",
            "elastic-linear": "elastic",
            "druckerprager": "dp",
            "drucker-prager": "dp",
            "drucker_prager": "dp",
            "dp": "dp",
            "cdp": "cdp",
            "concrete-cdp": "cdp",
            "concrete_cdp": "cdp",
            "cdp-lite": "cdp-lite",
            "cdplite": "cdp-lite",
            "cdp_lite": "cdp-lite",
            "cdp-lite2": "cdp-lite",
        }
        bm = aliases.get(bm, bm)
        if bm not in ("elastic", "dp", "cdp", "cdp-lite"):
            raise ValueError(f"Unknown bulk_material='{self.bulk_material}'. Use 'elastic', 'dp', 'cdp' or 'cdp-lite'.")
        self.bulk_material = bm


        if (self.steel_fy == 0.0) and (self.steel_sigy != 0.0):
            self.steel_fy = self.steel_sigy
        if (self.steel_Eh == 0.0) and (self.steel_H != 0.0):
            self.steel_Eh = self.steel_H
        if self.steel_fu == 0.0 and self.steel_fy != 0.0:
            self.steel_fu = 1.145 * self.steel_fy
