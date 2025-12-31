"""
Case 08: RC Beam Strengthened with CFRP Sheet (VVBS3)

3PB beam with externally bonded CFRP sheet on bottom face.

Reference: Gutiérrez thesis Chapter 5, Fig. 5.30, Tables 5.17-5.19
"""

from examples.gutierrez_thesis.case_config import (
    CaseConfig,
    GeometryConfig,
    ConcreteConfig,
    SteelConfig,
    CEBFIPBondLaw,
    RebarLayer,
    FRPSheet,
    BilinearBondLaw,
    MonotonicLoading,
    OutputConfig,
)


def create_case_08() -> CaseConfig:
    """
    3PB beam VVBS3 strengthened with CFRP sheet.

    Geometry:
        - Total length: 4300 mm
        - Span: 3700 mm
        - Height: 450 mm
        - Width: 200 mm
    Reinforcement:
        - Bottom: 3Ø20
        - Top: 2Ø16
        - Stirrups: Ø10 (optional)
    CFRP:
        - Width: 100 mm
        - Thickness: 1.4 mm
        - Bonded length: 3500 mm (100mm offset from each support)
        - E = 170.8 GPa
    """

    # Geometry
    geometry = GeometryConfig(
        length=4300.0,  # mm
        height=450.0,  # mm
        thickness=200.0,  # mm (width)
        n_elem_x=86,
        n_elem_y=18,
        element_type="Q4",
        notch_depth=None,
        notch_x=None,
    )

    # Concrete (Table 5.17)
    concrete = ConcreteConfig(
        E=24000.0,  # MPa
        nu=0.2,
        f_c=28.2,  # MPa
        f_t=2.9,  # MPa
        G_f=0.052,  # N/mm
        eps_cu=-0.023,  # εc3 = -2.3%
        model_type="cdp_full",
    )

    # Steel properties (Table 5.18)
    # Ø20
    steel_20 = SteelConfig(
        E=204000.0,  # MPa
        nu=0.3,
        f_y=513.3,  # MPa
        f_u=616.2,  # MPa
        hardening_modulus=810.0,  # MPa
    )

    # Ø16
    steel_16 = SteelConfig(
        E=203300.0,  # MPa
        nu=0.3,
        f_y=510.0,  # MPa
        f_u=646.5,  # MPa
        hardening_modulus=1085.0,  # MPa
    )

    # Ø10 (stirrups - optional for now)
    steel_10 = SteelConfig(
        E=196600.0,  # MPa
        nu=0.3,
        f_y=445.6,  # MPa
        f_u=572.6,  # MPa
        hardening_modulus=1669.0,  # MPa
    )

    # Bond laws (Table 5.19)
    # Ø20
    bond_law_20 = CEBFIPBondLaw(
        s1=0.36,  # mm
        s2=0.36,  # mm
        s3=8.0,  # mm
        tau_max=8.72,  # MPa
        tau_f=3.49,  # MPa
        alpha=0.4,
    )

    # Ø16
    bond_law_16 = CEBFIPBondLaw(
        s1=0.72,  # mm
        s2=0.72,  # mm
        s3=6.4,  # mm
        tau_max=11.54,  # MPa
        tau_f=4.61,  # MPa
        alpha=0.4,
    )

    # Ø10 (stirrups)
    bond_law_10 = CEBFIPBondLaw(
        s1=1.0,  # mm
        s2=2.0,  # mm
        s3=4.0,  # mm
        tau_max=13.27,  # MPa
        tau_f=5.31,  # MPa
        alpha=0.4,
    )

    # Bottom reinforcement (3Ø20)
    # Assuming 40mm cover
    rebar_bottom = RebarLayer(
        diameter=20.0,  # mm
        y_position=40.0,  # mm
        n_bars=3,
        steel=steel_20,
        bond_law=bond_law_20,
        orientation_deg=0.0,
    )

    # Top reinforcement (2Ø16)
    # Assuming 35mm cover from top
    rebar_top = RebarLayer(
        diameter=16.0,  # mm
        y_position=415.0,  # mm (450 - 35)
        n_bars=2,
        steel=steel_16,
        bond_law=bond_law_16,
        orientation_deg=0.0,
    )

    # CFRP Sheet (Table 5.19 - bilinear bond law)
    # Bonded on bottom face from x=300mm to x=4000mm (3700mm span, 100mm offset each side)
    # Support positions: x = (4300-3700)/2 = 300mm and x = 4000mm
    frp_bond = BilinearBondLaw(
        s1=0.02,  # mm
        s2=0.25,  # mm
        tau1=6.47,  # MPa (tau_max)
    )

    frp_sheet = FRPSheet(
        thickness=1.4,  # mm
        width=100.0,  # mm
        bonded_length=3500.0,  # mm (3700 - 2*100)
        y_position=0.0,  # Bottom face
        E=170800.0,  # MPa (170.8 GPa)
        nu=0.3,
        bond_law=frp_bond,
    )

    # Loading (displacement control at midspan top)
    loading = MonotonicLoading(
        max_displacement=20.0,  # mm
        n_steps=200,
        load_x_center=2150.0,  # Midspan (L/2)
        load_halfwidth=50.0,  # Loading plate
    )

    # Outputs
    outputs = OutputConfig(
        output_dir="examples/gutierrez_thesis/outputs/case_08",
        case_name="beam_3pb_vvbs3_cfrp",
        save_load_displacement=True,
        save_crack_data=True,
        save_crack_pattern=True,
        save_damage_field=True,
        save_vtk=True,
        vtk_frequency=10,
        compute_crack_widths=True,
        compute_slip_profiles=True,  # For both rebar and FRP
        compute_bond_profiles=True,  # tau(x) and slip(x)
        compute_steel_forces=True,
    )

    # Assemble case
    case = CaseConfig(
        case_id="08_beam_3pb_vvbs3_cfrp",
        description="3PB beam VVBS3 with CFRP sheet strengthening",
        geometry=geometry,
        concrete=concrete,
        loading=loading,
        outputs=outputs,
        rebar_layers=[rebar_bottom, rebar_top],
        frp_sheets=[frp_sheet],
        tolerance=1e-6,
        use_line_search=True,
        use_substepping=True,
    )

    return case
