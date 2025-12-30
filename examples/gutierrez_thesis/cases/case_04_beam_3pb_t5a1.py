"""
Case 04: Three-Point Bending Beam T5A1

Reinforced concrete beam under 3PB loading.

Reference: Gutiérrez thesis Chapter 5, T5A1 beam
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


def create_case_04() -> CaseConfig:
    """
    3PB beam T5A1.

    Beam: 1500x250x120 mm (span 1400 mm)
    Rebar: 2Ø16 mm bottom, cover 30 mm
    """

    # Geometry
    geometry = GeometryConfig(
        length=1500.0,  # mm (total length)
        height=250.0,  # mm
        thickness=120.0,  # mm
        n_elem_x=60,
        n_elem_y=10,
        element_type="Q4",
        notch_depth=25.0,  # mm (initial notch at midspan)
        notch_x=750.0,  # mm (midspan)
    )

    # Concrete
    concrete = ConcreteConfig(
        E=31000.0,  # MPa
        nu=0.2,
        f_c=35.0,  # MPa
        f_t=3.2,  # MPa
        G_f=0.100,  # N/mm
        model_type="cdp_full",  # Full CDP with compression damage
    )

    # Steel (2 bars Ø16)
    steel_16 = SteelConfig(
        E=200000.0,  # MPa
        nu=0.3,
        f_y=500.0,  # MPa
        f_u=600.0,  # MPa
        hardening_modulus=2000.0,  # MPa
    )

    # Bond law
    bond_law = CEBFIPBondLaw(
        s1=1.0,  # mm
        s2=2.0,  # mm
        s3=16.0,  # mm (d_bar)
        tau_max=15.0,  # MPa
        tau_f=2.5,  # MPa
        alpha=0.4,
    )

    # Rebar layer (bottom)
    rebar = RebarLayer(
        diameter=16.0,  # mm
        y_position=30.0,  # Cover 30 mm from bottom
        n_bars=2,
        steel=steel_16,
        bond_law=bond_law,
        orientation_deg=0.0,  # Horizontal
    )

    # Loading (displacement control at midspan top)
    loading = MonotonicLoading(
        max_displacement=10.0,  # mm (midspan deflection)
        n_steps=200,
        load_x_center=750.0,  # Midspan
        load_halfwidth=25.0,  # Loading plate width
    )

    # Outputs
    outputs = OutputConfig(
        output_dir="examples/gutierrez_thesis/outputs/case_04",
        case_name="beam_3pb_t5a1",
        save_load_displacement=True,
        save_crack_data=True,
        save_crack_pattern=True,
        save_damage_field=True,
        save_vtk=True,
        vtk_frequency=10,
        compute_crack_widths=True,
        compute_slip_profiles=True,
        compute_bond_profiles=True,
        compute_steel_forces=True,
    )

    # Assemble case
    case = CaseConfig(
        case_id="04_beam_3pb_t5a1",
        description="3PB beam T5A1 with bond-slip",
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
