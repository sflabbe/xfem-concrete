"""
Case 04b: Three-Point Bending Beam T6A1 (BOSCO)

Reinforced concrete beam under 3PB loading with 8Ø12 bottom reinforcement.

Reference: Gutiérrez thesis Chapter 5, Fig. 5.17, Tables 5.10-5.12
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


def create_case_04b() -> CaseConfig:
    """
    3PB beam T6A1 (BOSCO) - 8Ø12 bottom reinforcement.

    Beam: 4000x400x200 mm
    Rebar: 8Ø12 mm bottom (y=35mm), 2Ø10 mm top (y=355mm)
    Loading: Midspan displacement control
    """

    # Geometry
    geometry = GeometryConfig(
        length=4000.0,  # mm
        height=400.0,  # mm
        thickness=200.0,  # mm (out-of-plane)
        n_elem_x=80,
        n_elem_y=16,
        element_type="Q4",
        notch_depth=None,  # No initial notch
        notch_x=None,
    )

    # Concrete (typical values for BOSCO tests, ~30-35 MPa)
    concrete = ConcreteConfig(
        E=30000.0,  # MPa
        nu=0.2,
        f_c=32.0,  # MPa (compressive strength)
        f_t=3.0,  # MPa (tensile strength)
        G_f=0.100,  # N/mm (fracture energy)
        eps_cu=-0.0035,  # Crushing strain
        model_type="cdp_full",
    )

    # Steel Ø12 (bottom reinforcement)
    steel_12 = SteelConfig(
        E=200000.0,  # MPa
        nu=0.3,
        f_y=500.0,  # MPa (yield strength)
        f_u=600.0,  # MPa (ultimate strength)
        hardening_modulus=2000.0,  # MPa
    )

    # Steel Ø10 (top reinforcement)
    steel_10 = SteelConfig(
        E=200000.0,  # MPa
        nu=0.3,
        f_y=500.0,  # MPa
        f_u=600.0,  # MPa
        hardening_modulus=2000.0,  # MPa
    )

    # Bond law for Ø12
    bond_law_12 = CEBFIPBondLaw(
        s1=1.0,  # mm
        s2=2.0,  # mm
        s3=12.0,  # mm (≈ d_bar)
        tau_max=12.0,  # MPa
        tau_f=2.0,  # MPa (residual)
        alpha=0.4,
    )

    # Bond law for Ø10
    bond_law_10 = CEBFIPBondLaw(
        s1=0.9,  # mm
        s2=1.8,  # mm
        s3=10.0,  # mm (≈ d_bar)
        tau_max=13.0,  # MPa
        tau_f=2.2,  # MPa
        alpha=0.4,
    )

    # Bottom rebar layer (8Ø12) - HIGHER REINFORCEMENT
    rebar_bottom = RebarLayer(
        diameter=12.0,  # mm
        y_position=35.0,  # mm (cover 35mm from bottom)
        n_bars=8,  # ← 8 bars (vs 4 in T5A1)
        steel=steel_12,
        bond_law=bond_law_12,
        orientation_deg=0.0,  # Horizontal
    )

    # Top rebar layer (2Ø10)
    rebar_top = RebarLayer(
        diameter=10.0,  # mm
        y_position=355.0,  # mm (400 - 45mm cover)
        n_bars=2,
        steel=steel_10,
        bond_law=bond_law_10,
        orientation_deg=0.0,  # Horizontal
    )

    # Loading (displacement control at midspan top)
    loading = MonotonicLoading(
        max_displacement=15.0,  # mm (midspan deflection)
        n_steps=200,
        load_x_center=2000.0,  # Midspan (L/2)
        load_halfwidth=50.0,  # Loading plate half-width
    )

    # Outputs
    outputs = OutputConfig(
        output_dir="examples/gutierrez_thesis/outputs/case_04b",
        case_name="beam_3pb_t6a1_bosco",
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
        case_id="04b_beam_3pb_t6a1_bosco",
        description="3PB beam T6A1 (BOSCO) - 8Ø12 bottom + 2Ø10 top",
        geometry=geometry,
        concrete=concrete,
        loading=loading,
        outputs=outputs,
        rebar_layers=[rebar_bottom, rebar_top],
        tolerance=1e-6,
        use_line_search=True,
        use_substepping=True,
    )

    return case
