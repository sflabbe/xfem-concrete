"""
Case 07: Four-Point Bending Beam (JASON et al.)

RC beam under 4-point bending loading creating a constant moment zone.

Reference: Gutiérrez thesis Chapter 5, Fig. 5.25, Tables 5.13-5.16
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


def create_case_07() -> CaseConfig:
    """
    4PB beam (Jason et al.) with crack width validation.

    Geometry:
        - Total length: 3000 mm
        - Span: 2850 mm (supports at 75mm from each end)
        - Height: 250 mm
        - Width: 150 mm
    Loads:
        - Two loads P/2 separated by 1500 mm
        - Creates pure bending zone in middle
    Reinforcement:
        - Bottom: 2Ø12
        - Top: 2Ø8
    """

    # Geometry
    geometry = GeometryConfig(
        length=3000.0,  # mm
        height=250.0,  # mm
        thickness=150.0,  # mm (width)
        n_elem_x=60,
        n_elem_y=10,
        element_type="Q4",
        notch_depth=None,
        notch_x=None,
    )

    # Concrete (Table 5.13)
    concrete = ConcreteConfig(
        E=29800.0,  # MPa
        nu=0.2,
        f_c=39.0,  # MPa
        f_t=2.5,  # MPa
        G_f=0.1,  # N/mm
        eps_cu=-0.024,  # εc3 = -2.4%
        model_type="cdp_full",
    )

    # Steel (Table 5.14)
    # Ø12
    steel_12 = SteelConfig(
        E=190000.0,  # MPa
        nu=0.3,
        f_y=550.0,  # MPa
        f_u=590.0,  # MPa
        hardening_modulus=828.0,  # MPa (H from table)
    )

    # Ø8
    steel_8 = SteelConfig(
        E=190000.0,  # MPa
        nu=0.3,
        f_y=550.0,  # MPa (assuming same as Ø12)
        f_u=590.0,  # MPa
        hardening_modulus=828.0,  # MPa
    )

    # Bond laws (Table 5.15)
    # Ø12
    bond_law_12 = CEBFIPBondLaw(
        s1=0.47,  # mm
        s2=0.47,  # mm (no plateau)
        s3=0.57,  # mm
        tau_max=11.58,  # MPa
        tau_f=0.12,  # MPa (very low residual)
        alpha=0.4,
    )

    # Ø8
    bond_law_8 = CEBFIPBondLaw(
        s1=0.87,  # mm
        s2=0.87,  # mm (no plateau)
        s3=1.05,  # mm
        tau_max=14.8,  # MPa
        tau_f=0.15,  # MPa
        alpha=0.4,
    )

    # Bottom reinforcement (2Ø12)
    # Assuming 30mm cover from bottom
    rebar_bottom = RebarLayer(
        diameter=12.0,  # mm
        y_position=30.0,  # mm (cover ~30mm)
        n_bars=2,
        steel=steel_12,
        bond_law=bond_law_12,
        orientation_deg=0.0,
    )

    # Top reinforcement (2Ø8)
    # Assuming 25mm cover from top
    rebar_top = RebarLayer(
        diameter=8.0,  # mm
        y_position=225.0,  # mm (250 - 25mm cover)
        n_bars=2,
        steel=steel_8,
        bond_law=bond_law_8,
        orientation_deg=0.0,
    )

    # Loading
    # For 4PB: loads at x=750mm and x=2250mm (1500mm apart, centered)
    # NOTE: Current MonotonicLoading only supports single load region
    # Using midpoint approximation for now (x=1500mm)
    # TODO: Extend solver to support dual load points for true 4PB
    loading = MonotonicLoading(
        max_displacement=10.0,  # mm
        n_steps=200,
        load_x_center=1500.0,  # mm (midspan - approximation)
        load_halfwidth=750.0,  # mm (covers both load points as wide region)
    )

    # Outputs
    outputs = OutputConfig(
        output_dir="examples/gutierrez_thesis/outputs/case_07",
        case_name="beam_4pb_jason_4pbt",
        save_load_displacement=True,
        save_crack_data=True,
        save_crack_pattern=True,
        save_damage_field=True,
        save_vtk=True,
        vtk_frequency=10,
        compute_crack_widths=True,  # For validation vs Table 5.16
        compute_slip_profiles=True,
        compute_bond_profiles=True,
        compute_steel_forces=True,
    )

    # Assemble case
    case = CaseConfig(
        case_id="07_beam_4pb_jason_4pbt",
        description="4PB beam (Jason et al.) with crack width tracking",
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
