"""
Case 01: Pull-Out Test (Lettow)

Direct pull-out test with bond-slip validation.
Uses load element + empty element configuration.

Reference: Lettow (2006), Gutiérrez thesis Chapter 5
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
    SubdomainConfig,
)


def create_case_01() -> CaseConfig:
    """
    Pull-out test configuration.

    Block: 200x200x200 mm
    Rebar: Ø12 mm, embedment 36 mm
    Load element + empty element (E=0, bond~0)
    """

    # Geometry
    geometry = GeometryConfig(
        length=200.0,  # mm
        height=200.0,  # mm
        thickness=200.0,  # mm (out-of-plane, 3D approximation)
        n_elem_x=10,
        n_elem_y=10,
        element_type="Q4",
        notch_depth=None,
        notch_x=None,
        use_symmetry=False,
    )

    # Concrete
    concrete = ConcreteConfig(
        E=26287.0,  # MPa
        nu=0.2,
        f_c=30.0,  # MPa (typical for this test)
        f_t=3.0,  # MPa (typical)
        G_f=0.1,  # N/mm (typical)
        eps_cu=None,
        model_type="elastic",  # No damage for pull-out test
    )

    # Steel (rebar Ø12)
    steel_12 = SteelConfig(
        E=200000.0,  # MPa
        nu=0.3,
        f_y=500.0,  # MPa (typical)
        f_u=600.0,  # MPa
        hardening_modulus=2000.0,  # MPa
    )

    # Bond law (Lettow parameters)
    bond_law_lettow = CEBFIPBondLaw(
        s1=0.77,  # mm
        s2=1.37,  # mm
        s3=7.5,  # mm
        tau_max=11.5,  # MPa
        tau_f=4.0,  # MPa
        alpha=0.4,
    )

    # Rebar layer (horizontal, centered)
    rebar = RebarLayer(
        diameter=12.0,  # mm
        y_position=100.0,  # Center of block
        n_bars=1,
        steel=steel_12,
        bond_law=bond_law_lettow,
        orientation_deg=0.0,  # Horizontal
        bond_disabled_x_range=(0.0, 164.0),  # Empty element: 0-164mm, bond disabled
    )

    # Loading (displacement control on load element)
    loading = MonotonicLoading(
        max_displacement=5.0,  # mm slip
        n_steps=100,
        load_x_center=0.0,  # Left edge (load element)
        load_halfwidth=10.0,  # Small zone
    )

    # Subdomains
    # Empty element: E=0, h=0 (or very small) from x=0 to x=164mm
    # Load element: x=0 to x=20mm (apply displacement)
    # Bonded element: x=164 to x=200mm (36mm embedment)
    subdomains = [
        SubdomainConfig(
            x_range=(0.0, 164.0),
            y_range=None,
            material_type="void",  # No stiffness
            thickness_override=0.0,  # h=0
        ),
    ]

    # Outputs
    outputs = OutputConfig(
        output_dir="examples/gutierrez_thesis/outputs/case_01",
        case_name="pullout_lettow",
        save_load_displacement=True,
        save_crack_data=False,  # No cracks in pull-out
        save_energy=True,
        save_crack_pattern=False,
        save_damage_field=False,
        save_deformed_shape=True,
        save_metrics=True,
        save_vtk=True,
        vtk_frequency=5,
        compute_CTOD=False,
        compute_crack_widths=False,
        compute_slip_profiles=True,  # slip(x) along bar
        compute_bond_profiles=True,  # tau(x) along bar
        compute_steel_forces=True,  # N(x) along bar
        compute_base_moment=False,
    )

    # Assemble case
    case = CaseConfig(
        case_id="01_pullout_lettow",
        description="Pull-out test with bond-slip (Lettow 2006)",
        geometry=geometry,
        concrete=concrete,
        loading=loading,
        outputs=outputs,
        rebar_layers=[rebar],
        frp_sheets=[],
        fibres=None,
        subdomains=subdomains,
        max_steps=1000,
        tolerance=1e-6,
        use_line_search=True,
        use_substepping=True,
    )

    return case


if __name__ == "__main__":
    # Create and save case configuration
    case = create_case_01()
    case.save_json("case_01_pullout_lettow.json")
    case.save_yaml("case_01_pullout_lettow.yaml")
    print(f"Case {case.case_id} configuration saved.")
