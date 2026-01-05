"""
Case configuration dataclasses for Gutiérrez thesis examples.

Provides structured configuration for geometry, materials, reinforcement,
bond laws, loading programs, and output specifications.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import json


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GeometryConfig':
        return cls(**data)


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConcreteConfig':
        return cls(**data)


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SteelConfig':
        return cls(**data)


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FibreConfig':
        return cls(**data)


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CEBFIPBondLaw':
        return cls(
            s1=data['s1'],
            s2=data['s2'],
            s3=data['s3'],
            tau_max=data['tau_max'],
            tau_f=data['tau_f'],
            alpha=data.get('alpha', 0.4),
        )


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BilinearBondLaw':
        return cls(
            s1=data['s1'],
            s2=data['s2'],
            tau1=data['tau1'],
        )


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BanholzerBondLaw':
        return cls(
            s0=data['s0'],
            a=data['a'],
            tau1=data['tau1'],
            tau2=data['tau2'],
            tau_f=data['tau_f'],
        )


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RebarLayer':
        return cls(
            diameter=data['diameter'],
            y_position=data['y_position'],
            n_bars=data['n_bars'],
            steel=SteelConfig.from_dict(data['steel']),
            bond_law=CEBFIPBondLaw.from_dict(data['bond_law']),
            orientation_deg=data.get('orientation_deg', 0.0),
            bond_disabled_x_range=data.get('bond_disabled_x_range'),
        )


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FRPSheet':
        return cls(
            thickness=data['thickness'],
            width=data['width'],
            bonded_length=data['bonded_length'],
            y_position=data['y_position'],
            E=data['E'],
            nu=data['nu'],
            bond_law=BilinearBondLaw.from_dict(data['bond_law']),
        )


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FibreReinforcement':
        return cls(
            fibre=FibreConfig.from_dict(data['fibre']),
            bond_law=BanholzerBondLaw.from_dict(data['bond_law']),
            random_seed=data.get('random_seed', 42),
            active_near_crack_only=data.get('active_near_crack_only', True),
            activation_distance=data.get('activation_distance', 100.0),
        )


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MonotonicLoading':
        return cls(
            max_displacement=data['max_displacement'],
            n_steps=data['n_steps'],
            load_x_center=data['load_x_center'],
            load_halfwidth=data['load_halfwidth'],
        )


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
    targets_are_trajectory: bool = False  # If True, `targets` is an explicit u(mm) trajectory
    loading_type: str = "cyclic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loading_type": self.loading_type,
            "targets": self.targets,
            "n_cycles_per_target": self.n_cycles_per_target,
            "load_x_center": self.load_x_center,
            "load_halfwidth": self.load_halfwidth,
            "axial_load": self.axial_load,
            "targets_are_trajectory": self.targets_are_trajectory,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CyclicLoading':
        return cls(
            targets=data['targets'],
            load_x_center=data['load_x_center'],
            load_halfwidth=data['load_halfwidth'],
            n_cycles_per_target=data.get('n_cycles_per_target', 1),
            axial_load=data.get('axial_load'),
            targets_are_trajectory=data.get('targets_are_trajectory', False),
        )


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SubdomainConfig':
        # Convert list to tuple for ranges if needed
        x_range = tuple(data['x_range']) if data.get('x_range') else None
        y_range = tuple(data['y_range']) if data.get('y_range') else None
        return cls(
            x_range=x_range,
            y_range=y_range,
            element_indices=data.get('element_indices'),
            material_type=data.get('material_type'),
            E_override=data.get('E_override'),
            thickness_override=data.get('thickness_override'),
        )


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OutputConfig':
        # Convert ctod_node_pairs list of lists to list of tuples
        ctod_pairs = data.get('ctod_node_pairs')
        if ctod_pairs:
            ctod_pairs = [tuple(p) for p in ctod_pairs]
        return cls(
            output_dir=data.get('output_dir', 'outputs'),
            case_name=data.get('case_name', 'case'),
            save_load_displacement=data.get('save_load_displacement', True),
            save_crack_data=data.get('save_crack_data', True),
            save_energy=data.get('save_energy', True),
            save_crack_pattern=data.get('save_crack_pattern', True),
            save_damage_field=data.get('save_damage_field', True),
            save_deformed_shape=data.get('save_deformed_shape', True),
            save_metrics=data.get('save_metrics', True),
            save_vtk=data.get('save_vtk', True),
            vtk_frequency=data.get('vtk_frequency', 10),
            compute_CTOD=data.get('compute_CTOD', False),
            compute_crack_widths=data.get('compute_crack_widths', True),
            compute_slip_profiles=data.get('compute_slip_profiles', True),
            compute_bond_profiles=data.get('compute_bond_profiles', True),
            compute_steel_forces=data.get('compute_steel_forces', True),
            compute_base_moment=data.get('compute_base_moment', False),
            ctod_node_pairs=ctod_pairs,
        )


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
        import yaml
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
        import yaml
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CaseConfig':
        """Construct from dictionary (inverse of to_dict)"""
        # Parse loading based on loading_type
        loading_data = data['loading']
        loading_type = loading_data.get('loading_type', 'monotonic')
        if loading_type == 'monotonic':
            loading = MonotonicLoading.from_dict(loading_data)
        elif loading_type == 'cyclic':
            loading = CyclicLoading.from_dict(loading_data)
        else:
            raise ValueError(f"Unknown loading_type: {loading_type}")

        # Parse reinforcement
        rebar_layers = [RebarLayer.from_dict(r) for r in data.get('rebar_layers', [])]
        frp_sheets = [FRPSheet.from_dict(f) for f in data.get('frp_sheets', [])]
        fibres = FibreReinforcement.from_dict(data['fibres']) if data.get('fibres') else None

        # Parse subdomains
        subdomains = [SubdomainConfig.from_dict(s) for s in data.get('subdomains', [])]

        return cls(
            case_id=data['case_id'],
            description=data['description'],
            geometry=GeometryConfig.from_dict(data['geometry']),
            concrete=ConcreteConfig.from_dict(data['concrete']),
            loading=loading,
            outputs=OutputConfig.from_dict(data['outputs']),
            rebar_layers=rebar_layers,
            frp_sheets=frp_sheets,
            fibres=fibres,
            subdomains=subdomains,
            max_steps=data.get('max_steps', 1000),
            tolerance=data.get('tolerance', 1e-6),
            use_line_search=data.get('use_line_search', True),
            use_substepping=data.get('use_substepping', True),
        )
