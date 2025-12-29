"""
Case configuration dataclasses for Gutiérrez thesis examples.

Provides structured configuration for geometry, materials, reinforcement,
bond laws, loading programs, and output specifications.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import json
import yaml


class LoadingType(Enum):
    """Loading protocol type"""
    MONOTONIC = "monotonic"
    CYCLIC = "cyclic"
    DRIFT_PROTOCOL = "drift_protocol"


class ReinforcementType(Enum):
    """Type of reinforcement"""
    EMBEDDED_REBAR = "embedded_rebar"
    EXTERNALLY_BONDED = "externally_bonded"  # FRP sheet
    FIBRE = "fibre"


class BondLawType(Enum):
    """Bond-slip law type"""
    CEB_FIP = "ceb_fip"  # 4-branch for embedded rebar
    BILINEAR = "bilinear"  # For FRP sheet
    BANHOLZER = "banholzer"  # For fibres
    PERFECT = "perfect"  # No slip


class ElementType(Enum):
    """Element type"""
    Q4 = "Q4"
    T3 = "T3"
    VOID = "void"  # Inactive element


# ============================================================================
# GEOMETRY
# ============================================================================

@dataclass
class GeometryConfig:
    """Geometry configuration"""
    # Domain size (mm)
    length: float  # L
    height: float  # H
    thickness: float  # b (out-of-plane)

    # Mesh
    n_elem_x: int
    n_elem_y: int
    element_type: str = "Q4"  # "Q4" or "T3"

    # Notch/crack starter (optional)
    notch_depth: Optional[float] = None
    notch_x: Optional[float] = None  # Position along x

    # Symmetry
    use_symmetry: bool = False  # Model only half

    def to_dict(self) -> Dict[str, Any]:
        return {
            "length": self.length,
            "height": self.height,
            "thickness": self.thickness,
            "n_elem_x": self.n_elem_x,
            "n_elem_y": self.n_elem_y,
            "element_type": self.element_type,
            "notch_depth": self.notch_depth,
            "notch_x": self.notch_x,
            "use_symmetry": self.use_symmetry,
        }


# ============================================================================
# MATERIALS
# ============================================================================

@dataclass
class ConcreteConfig:
    """Concrete material properties (MPa, mm, N)"""
    # Elastic
    E: float  # Young's modulus (MPa)
    nu: float  # Poisson's ratio

    # Strength
    f_c: float  # Compressive strength (MPa)
    f_t: float  # Tensile strength (MPa)

    # Fracture
    G_f: float  # Fracture energy (N/mm)

    # Optional: crushing strain
    eps_cu: Optional[float] = None  # Crushing strain (negative)

    # Material model
    model_type: str = "cdp_full"  # "elastic", "dp", "cdp_lite", "cdp_full"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "E": self.E,
            "nu": self.nu,
            "f_c": self.f_c,
            "f_t": self.f_t,
            "G_f": self.G_f,
            "eps_cu": self.eps_cu,
            "model_type": self.model_type,
        }


@dataclass
class SteelConfig:
    """Steel material properties (MPa, mm, N)"""
    # Elastic
    E: float  # Young's modulus (MPa)
    nu: float  # Poisson's ratio

    # Plasticity
    f_y: float  # Yield strength (MPa)
    f_u: Optional[float] = None  # Ultimate strength (MPa)
    hardening_modulus: Optional[float] = None  # E_h (MPa)

    # For failure modeling (optional)
    f_ult: Optional[float] = None  # Fracture stress (MPa)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "E": self.E,
            "nu": self.nu,
            "f_y": self.f_y,
            "f_u": self.f_u,
            "hardening_modulus": self.hardening_modulus,
            "f_ult": self.f_ult,
        }


@dataclass
class FibreConfig:
    """Fibre material properties"""
    # Elastic
    E: float  # Young's modulus (MPa)
    nu: float  # Poisson's ratio

    # Geometry
    diameter: float  # mm
    length: float  # mm

    # Distribution
    density: float  # fibres per cm^2 in crack zone
    orientation_deg: float = 0.0  # Mean orientation (0=horizontal)
    volume_fraction_multiplier: float = 1.0  # For sensitivity studies

    def to_dict(self) -> Dict[str, Any]:
        return {
            "E": self.E,
            "nu": self.nu,
            "diameter": self.diameter,
            "length": self.length,
            "density": self.density,
            "orientation_deg": self.orientation_deg,
            "volume_fraction_multiplier": self.volume_fraction_multiplier,
        }


# ============================================================================
# BOND-SLIP LAWS
# ============================================================================

@dataclass
class CEBFIPBondLaw:
    """
    CEB-FIP bond-slip law for embedded rebar (4 branches).

    tau(s) =
      tau_max * (s/s1)^alpha          if 0 <= s <= s1
      tau_max                         if s1 < s <= s2
      tau_max - (tau_max-tau_f)*(s-s2)/(s3-s2)  if s2 < s <= s3
      tau_f                           if s > s3
    """
    s1: float  # End of rising branch (mm)
    s2: float  # End of plateau (mm)
    s3: float  # End of softening (mm)
    tau_max: float  # Maximum bond stress (MPa)
    tau_f: float  # Residual bond stress (MPa)
    alpha: float = 0.4  # Exponent in rising branch

    law_type: str = "ceb_fip"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "law_type": self.law_type,
            "s1": self.s1,
            "s2": self.s2,
            "s3": self.s3,
            "tau_max": self.tau_max,
            "tau_f": self.tau_f,
            "alpha": self.alpha,
        }


@dataclass
class BilinearBondLaw:
    """
    Bilinear bond-slip law for externally bonded FRP (softening to 0).

    tau(s) =
      (tau1/s1)*s                      if 0 <= s <= s1
      tau1*(1 - (s-s1)/(s2-s1))        if s1 < s <= s2
      0                                if s > s2
    """
    s1: float  # End of hardening (mm)
    s2: float  # End of softening (mm)
    tau1: float  # Peak bond stress (MPa)

    law_type: str = "bilinear"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "law_type": self.law_type,
            "s1": self.s1,
            "s2": self.s2,
            "tau1": self.tau1,
        }


@dataclass
class BanholzerBondLaw:
    """
    Banholzer bond-slip law for fibres (5 parameters).

    tau(s) =
      (tau1/s0)*s                                   if 0 <= s <= s0
      tau2 - (tau2-tau_f)*(s-s0)/((a-1)*s0)         if s0 < s <= a*s0
      tau_f                                         if s > a*s0
    """
    s0: float  # End of rising branch (mm)
    a: float  # Softening end multiplier (a*s0)
    tau1: float  # Peak in rising (MPa)
    tau2: float  # Peak in softening (MPa)
    tau_f: float  # Residual (MPa)

    law_type: str = "banholzer"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "law_type": self.law_type,
            "s0": self.s0,
            "a": self.a,
            "tau1": self.tau1,
            "tau2": self.tau2,
            "tau_f": self.tau_f,
        }


# ============================================================================
# REINFORCEMENT
# ============================================================================

@dataclass
class RebarLayer:
    """Single rebar layer (embedded)"""
    # Geometry
    diameter: float  # mm
    y_position: float  # Distance from bottom (mm)
    n_bars: int  # Number of bars

    # Material
    steel: SteelConfig

    # Bond law
    bond_law: CEBFIPBondLaw

    # Orientation
    orientation_deg: float = 0.0  # 0=horizontal, 90=vertical

    # Masking (for empty elements)
    bond_disabled_x_range: Optional[Tuple[float, float]] = None  # (x_min, x_max)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diameter": self.diameter,
            "y_position": self.y_position,
            "n_bars": self.n_bars,
            "steel": self.steel.to_dict(),
            "bond_law": self.bond_law.to_dict(),
            "orientation_deg": self.orientation_deg,
            "bond_disabled_x_range": self.bond_disabled_x_range,
        }


@dataclass
class FRPSheet:
    """Externally bonded FRP sheet"""
    # Geometry
    thickness: float  # mm
    width: float  # mm
    bonded_length: float  # mm
    y_position: float  # Position on face (mm)

    # Material
    E: float  # MPa
    nu: float

    # Bond law
    bond_law: BilinearBondLaw

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thickness": self.thickness,
            "width": self.width,
            "bonded_length": self.bonded_length,
            "y_position": self.y_position,
            "E": self.E,
            "nu": self.nu,
            "bond_law": self.bond_law.to_dict(),
        }


@dataclass
class FibreReinforcement:
    """Fibre reinforcement"""
    fibre: FibreConfig
    bond_law: BanholzerBondLaw

    # Generation
    random_seed: int = 42
    active_near_crack_only: bool = True
    activation_distance: float = 100.0  # mm

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fibre": self.fibre.to_dict(),
            "bond_law": self.bond_law.to_dict(),
            "random_seed": self.random_seed,
            "active_near_crack_only": self.active_near_crack_only,
            "activation_distance": self.activation_distance,
        }


# ============================================================================
# LOADING
# ============================================================================

@dataclass
class MonotonicLoading:
    """Monotonic displacement control"""
    max_displacement: float  # mm
    n_steps: int

    # Application region
    load_x_center: float  # x-coordinate of load center (mm)
    load_halfwidth: float  # Half-width of loading zone (mm)

    loading_type: str = "monotonic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loading_type": self.loading_type,
            "max_displacement": self.max_displacement,
            "n_steps": self.n_steps,
            "load_x_center": self.load_x_center,
            "load_halfwidth": self.load_halfwidth,
        }


@dataclass
class CyclicLoading:
    """Cyclic loading protocol"""
    # Drift levels (%) or absolute displacements (mm)
    targets: List[float]  # List of target displacements

    # Application
    load_x_center: float
    load_halfwidth: float

    # Optional parameters (must come after required params)
    n_cycles_per_target: int = 1
    axial_load: Optional[float] = None  # Constant axial load (N)
    loading_type: str = "cyclic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loading_type": self.loading_type,
            "targets": self.targets,
            "n_cycles_per_target": self.n_cycles_per_target,
            "load_x_center": self.load_x_center,
            "load_halfwidth": self.load_halfwidth,
            "axial_load": self.axial_load,
        }


# ============================================================================
# SUBDOMAINS / SPECIAL ELEMENTS
# ============================================================================

@dataclass
class SubdomainConfig:
    """Element-level material or property override"""
    # Element selection (by region or index list)
    x_range: Optional[Tuple[float, float]] = None
    y_range: Optional[Tuple[float, float]] = None
    element_indices: Optional[List[int]] = None

    # Overrides
    material_type: Optional[str] = None  # "void", "rigid", "concrete"
    E_override: Optional[float] = None  # For rigid body
    thickness_override: Optional[float] = None  # For void (h=0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x_range": self.x_range,
            "y_range": self.y_range,
            "element_indices": self.element_indices,
            "material_type": self.material_type,
            "E_override": self.E_override,
            "thickness_override": self.thickness_override,
        }


# ============================================================================
# OUTPUTS
# ============================================================================

@dataclass
class OutputConfig:
    """Output specifications"""
    # Directory
    output_dir: str = "outputs"
    case_name: str = "case"

    # CSV outputs
    save_load_displacement: bool = True
    save_crack_data: bool = True
    save_energy: bool = True

    # PNG outputs
    save_crack_pattern: bool = True
    save_damage_field: bool = True
    save_deformed_shape: bool = True

    # JSON metrics
    save_metrics: bool = True

    # VTK
    save_vtk: bool = True
    vtk_frequency: int = 10  # Save every N steps

    # Post-processing
    compute_CTOD: bool = False  # Crack tip opening displacement
    compute_crack_widths: bool = True
    compute_slip_profiles: bool = True  # slip(x) along reinforcement
    compute_bond_profiles: bool = True  # tau(x) along reinforcement
    compute_steel_forces: bool = True  # Axial force in bars
    compute_base_moment: bool = False  # For walls

    # CTOD measurement points (if applicable)
    ctod_node_pairs: Optional[List[Tuple[int, int]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "case_name": self.case_name,
            "save_load_displacement": self.save_load_displacement,
            "save_crack_data": self.save_crack_data,
            "save_energy": self.save_energy,
            "save_crack_pattern": self.save_crack_pattern,
            "save_damage_field": self.save_damage_field,
            "save_deformed_shape": self.save_deformed_shape,
            "save_metrics": self.save_metrics,
            "save_vtk": self.save_vtk,
            "vtk_frequency": self.vtk_frequency,
            "compute_CTOD": self.compute_CTOD,
            "compute_crack_widths": self.compute_crack_widths,
            "compute_slip_profiles": self.compute_slip_profiles,
            "compute_bond_profiles": self.compute_bond_profiles,
            "compute_steel_forces": self.compute_steel_forces,
            "compute_base_moment": self.compute_base_moment,
            "ctod_node_pairs": self.ctod_node_pairs,
        }


# ============================================================================
# MASTER CASE CONFIGURATION
# ============================================================================

@dataclass
class CaseConfig:
    """Master configuration for a thesis case"""
    # Metadata
    case_id: str  # e.g., "01_pullout_lettow"
    description: str

    # Configuration
    geometry: GeometryConfig
    concrete: ConcreteConfig
    loading: Any  # MonotonicLoading or CyclicLoading
    outputs: OutputConfig

    # Reinforcement (optional)
    rebar_layers: List[RebarLayer] = field(default_factory=list)
    frp_sheets: List[FRPSheet] = field(default_factory=list)
    fibres: Optional[FibreReinforcement] = None

    # Subdomains (optional)
    subdomains: List[SubdomainConfig] = field(default_factory=list)

    # Solver parameters
    max_steps: int = 1000
    tolerance: float = 1e-6
    use_line_search: bool = True
    use_substepping: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON/YAML export"""
        return {
            "case_id": self.case_id,
            "description": self.description,
            "geometry": self.geometry.to_dict(),
            "concrete": self.concrete.to_dict(),
            "loading": self.loading.to_dict(),
            "outputs": self.outputs.to_dict(),
            "rebar_layers": [r.to_dict() for r in self.rebar_layers],
            "frp_sheets": [f.to_dict() for f in self.frp_sheets],
            "fibres": self.fibres.to_dict() if self.fibres else None,
            "subdomains": [s.to_dict() for s in self.subdomains],
            "max_steps": self.max_steps,
            "tolerance": self.tolerance,
            "use_line_search": self.use_line_search,
            "use_substepping": self.use_substepping,
        }

    def save_json(self, filepath: str):
        """Save to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_yaml(self, filepath: str):
        """Save to YAML file"""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load_json(cls, filepath: str) -> 'CaseConfig':
        """Load from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def load_yaml(cls, filepath: str) -> 'CaseConfig':
        """Load from YAML file"""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CaseConfig':
        """Construct from dictionary (inverse of to_dict)"""
        # TODO: Implement full deserialization
        # For now, this is a placeholder
        raise NotImplementedError("Deserialization not yet implemented")
