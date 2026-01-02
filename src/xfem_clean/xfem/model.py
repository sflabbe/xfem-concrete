"""Model/parameter container for the XFEM beam examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Literal, List, TYPE_CHECKING

if TYPE_CHECKING:
    from xfem_clean.reinforcement import ReinforcementLayer
    from xfem_clean.contact_rebar import RebarContactPoint
    from xfem_clean.numerical_aspects import StabilizationParams
    from xfem_clean.compression_damage import ConcreteCompressionModel

# Type alias for tip enrichment mode
TipEnrichmentType = Literal["lefm_branch", "non_singular_cohesive"]


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

    # Bond-slip interface modeling (Phase 2)
    enable_bond_slip: bool = False
    rebar_diameter: float = 0.016  # meters (default: 16mm)
    bond_condition: str = "good"   # "good" | "poor" (Model Code 2010)
    steel_EA_min: float = 1e3      # Minimum steel axial stiffness [N] to avoid rigid body mode
    bond_tangent_cap_factor: float = 1e2  # Cap bond tangent at factor × median(diag(K_bulk))

    # Bond layers (Task B: explicit layer-based modeling, thesis parity)
    bond_layers: List = field(default_factory=list)  # List[BondLayer]

    # Bond-slip continuation parameters (BLOQUE B: convergence improvement)
    bond_gamma_strategy: str = "ramp_steps"  # "ramp_steps" | "adaptive_on_fail" | "disabled"
    bond_gamma_ramp_steps: int = 5  # Number of gamma values in ramp (0→1)
    bond_gamma_min: float = 0.0     # Start gamma (0 = no bond, only steel axial EA)
    bond_gamma_max: float = 1.0     # End gamma (1 = full bond-slip)

    # Bond-slip tangent regularization (BLOQUE C: stabilization near s≈0)
    bond_k_cap: Optional[float] = None  # Cap dtau/ds [Pa/m] (None = no cap)
    bond_s_eps: float = 0.0             # Smooth regularization epsilon [m] (0 = disabled)

    # Dowel action at crack-rebar intersections (Phase 5)
    enable_dowel: bool = False
    dowel_penalty_factor: float = 1.0

    # Small hysteresis factor for crack initiation to avoid chatter.
    ft_initiation_factor: float = 1.001

    # solver params
    newton_maxit: int = 25
    newton_tol_r: float = 1e-6
    newton_tol_rel: float = 1e-8
    newton_beta: float = 1e-3
    newton_tol_du: float = 1e-8
    line_search: bool = True
    enable_diagonal_scaling: bool = True  # Diagonal equilibration for ill-conditioned systems
    max_subdiv: int = 12
    max_total_substeps: int = 50000  # Anti-hang: abort if total substeps exceeds this limit

    # Optional Numba acceleration (Phase 2/3)
    # None = auto-detect (use if available), True/False = explicit override
    use_numba: Optional[bool] = None

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

    # Tip enrichment type: non-singular for cohesive cracks (Gutierrez 2020) or classical LEFM
    tip_enrichment_type: TipEnrichmentType = "non_singular_cohesive"

    # cohesive penalty scaling
    Kn_factor: float = 0.1
    visc_damp: float = 0.0
    lch: float = 0.05

    # Load application patch (half-width in meters)
    load_halfwidth: float = 0.0

    # ========================================================================
    # Dissertation Parity Features (Chapter 4)
    # ========================================================================

    # Reinforcement layers (Heaviside enrichment, Eq. 4.92-4.103)
    reinforcement_layers: List = field(default_factory=list)  # List[ReinforcementLayer]

    # Rebar contact points (penalty method, Eq. 4.120-4.129)
    rebar_contact_points: List = field(default_factory=list)  # List[RebarContactPoint]

    # Numerical stabilization parameters (Chapter 4.3)
    stabilization: Optional[object] = None  # StabilizationParams (initialized in __post_init__)

    # Compression damage model (Eq. 3.44-3.46)
    compression_damage_model: Optional[object] = None  # ConcreteCompressionModel

    # Quadrature knobs
    n_gauss_line: int = 7  # Line integration for reinforcement (thesis default)
    n_gauss_quad: int = 2  # Standard 2x2 for quadrilaterals
    n_gauss_tri: int = 3  # Triangle integration order
    n_gauss_near_tip: int = 4  # Higher-order integration near crack tip

    # Enable flags for dissertation features
    enable_reinforcement_heaviside: bool = False  # Mesh-independent reinforcement (Eq. 4.92)
    enable_rebar_contact: bool = False  # Rebar-rebar penalty contact (Eq. 4.120)
    enable_junction_enrichment: bool = False  # Junction enrichment at coalescence (Eq. 4.64)
    enable_dof_projection: bool = False  # L2 DOF projection on topology change (Eq. 4.60)
    enable_dolbow_removal: bool = False  # Ill-conditioning node removal (Eq. 4.3)
    enable_kinked_tips: bool = False  # Kinked crack tip coordinates (Chapter 4.3)
    enable_compression_damage: bool = False  # Compression damage model (Eq. 3.44)

    # Junction detection tolerance
    junction_merge_tolerance: float = 0.01  # meters

    def __post_init__(self):
        """Backwards-compatible parameter normalization."""

        # Auto-detect Numba availability if not explicitly set
        if self.use_numba is None:
            try:
                from xfem_clean.numba.utils import NUMBA_AVAILABLE
                self.use_numba = NUMBA_AVAILABLE
            except Exception:
                self.use_numba = False

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
        if bm not in ("elastic", "dp", "cdp", "cdp-lite", "compression-damage"):
            raise ValueError(f"Unknown bulk_material='{self.bulk_material}'. Use 'elastic', 'dp', 'cdp', 'cdp-lite', or 'compression-damage'.")
        self.bulk_material = bm


        if (self.steel_fy == 0.0) and (self.steel_sigy != 0.0):
            self.steel_fy = self.steel_sigy
        if (self.steel_Eh == 0.0) and (self.steel_H != 0.0):
            self.steel_Eh = self.steel_H
        if self.steel_fu == 0.0 and self.steel_fy != 0.0:
            self.steel_fu = 1.145 * self.steel_fy

        # Initialize stabilization parameters if None
        if self.stabilization is None and self.enable_dolbow_removal:
            from xfem_clean.numerical_aspects import StabilizationParams
            self.stabilization = StabilizationParams(
                use_dolbow_removal=self.enable_dolbow_removal,
                use_kinked_tips=self.enable_kinked_tips,
            )
