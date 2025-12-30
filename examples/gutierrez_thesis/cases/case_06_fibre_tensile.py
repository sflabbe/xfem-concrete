"""
Case 06: Fibre-Reinforced Tensile Test

Uniaxial tensile test on fibre-reinforced concrete with Banholzer bond law.

Reference: Gutiérrez thesis Chapter 5, Fibre tensile test
"""

from examples.gutierrez_thesis.case_config import (
    CaseConfig,
    GeometryConfig,
    ConcreteConfig,
    FibreConfig,
    BanholzerBondLaw,
    FibreReinforcement,
    MonotonicLoading,
    OutputConfig,
)


def create_case_06() -> CaseConfig:
    """
    Fibre-reinforced tensile specimen.

    Specimen: 100x100x10 mm (thin specimen with notch)
    Fibres: 3.02 fibres/cm² density, 12 mm length
    """

    # Geometry
    geometry = GeometryConfig(
        length=100.0,  # mm
        height=100.0,  # mm
        thickness=10.0,  # mm (thin specimen)
        n_elem_x=20,
        n_elem_y=20,
        element_type="Q4",
        notch_depth=10.0,  # mm (notch at center to localize cracking)
        notch_x=50.0,  # mm (center)
    )

    # Concrete (matrix)
    concrete = ConcreteConfig(
        E=30000.0,  # MPa
        nu=0.2,
        f_c=40.0,  # MPa
        f_t=3.0,  # MPa
        G_f=0.080,  # N/mm (lower for fibre-reinforced)
        model_type="cdp_lite",
    )

    # Fibre properties
    fibre = FibreConfig(
        E=200000.0,  # MPa (steel fibre)
        nu=0.3,
        diameter=0.5,  # mm (typical steel fibre)
        length=12.0,  # mm
        density=3.02,  # fibres/cm² in crack zone
        orientation_deg=0.0,  # Mean orientation (random around this)
        volume_fraction_multiplier=1.0,
    )

    # Banholzer bond law for fibres
    banholzer = BanholzerBondLaw(
        s0=0.01,  # mm (end of rising branch)
        a=20.0,  # Softening multiplier
        tau1=5.0,  # MPa (peak in rising)
        tau2=3.0,  # MPa (stress at start of softening)
        tau_f=2.25,  # MPa (residual)
    )

    # Fibre reinforcement
    fibres = FibreReinforcement(
        fibre=fibre,
        bond_law=banholzer,
        random_seed=42,
        active_near_crack_only=True,
        activation_distance=50.0,  # mm (activate fibres within 50mm of crack)
    )

    # Loading (tension control)
    loading = MonotonicLoading(
        max_displacement=2.0,  # mm
        n_steps=200,
        load_x_center=100.0,  # Right edge
        load_halfwidth=10.0,
    )

    # Outputs
    outputs = OutputConfig(
        output_dir="examples/gutierrez_thesis/outputs/case_06",
        case_name="fibre_tensile",
        save_load_displacement=True,
        save_crack_data=True,
        save_crack_pattern=True,
        save_vtk=True,
        vtk_frequency=10,
        compute_crack_widths=True,
    )

    # Assemble case
    case = CaseConfig(
        case_id="06_fibre_tensile",
        description="Fibre-reinforced tensile test with Banholzer bond law",
        geometry=geometry,
        concrete=concrete,
        loading=loading,
        outputs=outputs,
        fibres=fibres,
        tolerance=1e-6,
        use_line_search=True,
        use_substepping=True,
    )

    return case
