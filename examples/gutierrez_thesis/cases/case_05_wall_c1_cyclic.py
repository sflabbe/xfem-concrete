"""
Case 05: RC Wall C1 - Cyclic Loading

Reinforced concrete shear wall under cyclic horizontal loading with constant axial load.

Reference: Gutiérrez thesis Chapter 5, Wall C1
"""

from examples.gutierrez_thesis.case_config import (
    CaseConfig,
    GeometryConfig,
    ConcreteConfig,
    SteelConfig,
    CEBFIPBondLaw,
    RebarLayer,
    CyclicLoading,
    OutputConfig,
    SubdomainConfig,
)


def create_case_05() -> CaseConfig:
    """
    RC wall C1 under cyclic loading.

    Wall: 1400x2800x120 mm (L x H x thickness)
    Reinforcement: horizontal + vertical mesh
    Axial load: 290 kN (constant compression)
    Drift protocol: [0.2, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5]% drift
    """

    # Geometry
    geometry = GeometryConfig(
        length=1400.0,  # mm (width)
        height=2800.0,  # mm (height)
        thickness=120.0,  # mm
        n_elem_x=28,
        n_elem_y=56,
        element_type="Q4",
    )

    # Concrete
    concrete = ConcreteConfig(
        E=30000.0,  # MPa
        nu=0.2,
        f_c=40.0,  # MPa
        f_t=3.5,  # MPa
        G_f=0.120,  # N/mm
        model_type="cdp_full",  # Full CDP (cyclic behavior)
    )

    # Steel (mesh reinforcement)
    steel_10 = SteelConfig(
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
        s3=10.0,  # mm (d_bar)
        tau_max=12.0,  # MPa
        tau_f=2.0,  # MPa
        alpha=0.4,
    )

    # Horizontal rebar layers (multiple layers for mesh)
    # Simplified: 3 representative layers
    rebar_h1 = RebarLayer(
        diameter=10.0,  # mm
        y_position=200.0,  # mm from bottom
        n_bars=7,  # Spaced along width
        steel=steel_10,
        bond_law=bond_law,
        orientation_deg=0.0,  # Horizontal
    )

    rebar_h2 = RebarLayer(
        diameter=10.0,  # mm
        y_position=1400.0,  # mm (middle)
        n_bars=7,
        steel=steel_10,
        bond_law=bond_law,
        orientation_deg=0.0,  # Horizontal
    )

    # Vertical rebar (edge columns - critical for shear walls)
    # Note: Vertical rebar should be orientation_deg=90
    rebar_v1 = RebarLayer(
        diameter=12.0,  # mm (thicker for boundary elements)
        y_position=60.0,  # Left edge
        n_bars=4,  # Vertical bars
        steel=steel_10,
        bond_law=bond_law,
        orientation_deg=90.0,  # Vertical
    )

    # Loading: cyclic drift protocol
    # Drift (%) converted to horizontal displacement at top (H=2800mm)
    # u_top = drift(%) * H / 100
    drift_levels_pct = [0.2, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5]
    H = 2800.0  # mm
    drift_targets_mm = [d / 100.0 * H for d in drift_levels_pct]

    loading = CyclicLoading(
        targets=drift_targets_mm,  # mm (horizontal displacement at top)
        n_cycles_per_target=2,  # 2 cycles per drift level
        load_x_center=700.0,  # Center of top edge (rigid loading beam)
        load_halfwidth=700.0,  # Full width (rigid beam)
        axial_load=290000.0,  # N (290 kN constant compression)
    )

    # Subdomain: rigid loading beam at top (200 mm height)
    # This models the steel beam that distributes horizontal load
    rigid_beam = SubdomainConfig(
        x_range=None,  # Full width
        y_range=(2600.0, 2800.0),  # Top 200 mm
        material_type="rigid",
        E_override=200000.0 * 1e6,  # Pa (steel stiffness)
    )

    # Outputs
    outputs = OutputConfig(
        output_dir="examples/gutierrez_thesis/outputs/case_05",
        case_name="wall_c1_cyclic",
        save_load_displacement=True,
        save_crack_data=True,
        save_crack_pattern=True,
        save_damage_field=True,
        save_vtk=True,
        vtk_frequency=5,
        compute_crack_widths=True,
        compute_slip_profiles=True,
        compute_base_moment=True,  # Moment-drift curve
    )

    # Assemble case
    case = CaseConfig(
        case_id="05_wall_c1_cyclic",
        description="RC wall C1 under cyclic loading with axial load",
        geometry=geometry,
        concrete=concrete,
        loading=loading,
        outputs=outputs,
        rebar_layers=[rebar_h1, rebar_h2, rebar_v1],
        subdomains=[rigid_beam],
        tolerance=1e-6,
        use_line_search=True,
        use_substepping=True,
    )

    return case
