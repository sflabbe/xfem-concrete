"""
Case 03: Tensile Member STN12

Reinforced concrete tensile member with distributed cracking.

Reference: Gutiérrez thesis Chapter 5, STN12 specimen
"""

from examples.gutierrez_thesis.case_config import (
    CaseConfig,
    GeometryConfig,
    ConcreteConfig,
    SteelConfig,
    CEBFIPBondLaw,
    RebarLayer,
    MonotonicLoading,
    OutputConfig,
)


def create_case_03() -> CaseConfig:
    """
    Tensile member STN12 (Alvarez 1998).

    Member: 1000x100x100 mm
    Rebar: 2Ø12 mm
    """

    # Geometry
    geometry = GeometryConfig(
        length=1000.0,  # mm
        height=100.0,  # mm
        thickness=100.0,  # mm
        n_elem_x=50,
        n_elem_y=5,
        element_type="Q4",
    )

    # Concrete
    concrete = ConcreteConfig(
        E=27000.0,  # MPa
        nu=0.2,
        f_c=30.0,  # MPa
        f_t=2.9,  # MPa
        G_f=0.080,  # N/mm
        model_type="cdp_lite",  # Tension damage
    )

    # Steel (2 bars Ø12)
    steel_12 = SteelConfig(
        E=200000.0,  # MPa
        nu=0.3,
        f_y=550.0,  # MPa
        f_u=650.0,  # MPa
        hardening_modulus=2000.0,  # MPa
    )

    # Bond law (Model Code 2010 parameters for good bond)
    bond_law = CEBFIPBondLaw(
        s1=1.0,  # mm
        s2=2.0,  # mm
        s3=12.0,  # mm (d_bar)
        tau_max=13.0,  # MPa (calibrated)
        tau_f=2.0,  # MPa
        alpha=0.4,
    )

    # Rebar layer (centered, 2 bars)
    rebar = RebarLayer(
        diameter=12.0,  # mm
        y_position=50.0,  # Center
        n_bars=2,
        steel=steel_12,
        bond_law=bond_law,
        orientation_deg=0.0,  # Horizontal
    )

    # Loading (tension control at both ends)
    loading = MonotonicLoading(
        max_displacement=5.0,  # mm total elongation
        n_steps=200,
        load_x_center=1000.0,  # Right edge
        load_halfwidth=10.0,
    )

    # Outputs
    outputs = OutputConfig(
        output_dir="examples/gutierrez_thesis/outputs/case_03",
        case_name="tensile_stn12",
        save_load_displacement=True,
        save_crack_data=True,
        save_crack_pattern=True,
        save_vtk=True,
        vtk_frequency=10,
        compute_slip_profiles=True,
        compute_bond_profiles=True,
        compute_steel_forces=True,
    )

    # Assemble case
    case = CaseConfig(
        case_id="03_tensile_stn12",
        description="Tensile member STN12 with distributed cracking",
        geometry=geometry,
        concrete=concrete,
        loading=loading,
        outputs=outputs,
        rebar_layers=[rebar],
        tolerance=1e-6,
        use_line_search=True,
        use_substepping=True,
    )

    return case
