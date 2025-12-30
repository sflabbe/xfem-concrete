"""
Case 02: SSPOT FRP Sheet Test

Single-Shear Push-Off Test for FRP sheet debonding validation.

Reference: Gutiérrez thesis Chapter 5, SSPOT test
"""

from examples.gutierrez_thesis.case_config import (
    CaseConfig,
    GeometryConfig,
    ConcreteConfig,
    FRPSheet,
    BilinearBondLaw,
    MonotonicLoading,
    OutputConfig,
)


def create_case_02() -> CaseConfig:
    """
    SSPOT FRP sheet debonding test.

    Specimen: 150x150x50 mm
    FRP sheet: bonded on bottom face
    """

    # Geometry
    geometry = GeometryConfig(
        length=150.0,  # mm
        height=150.0,  # mm
        thickness=50.0,  # mm
        n_elem_x=15,
        n_elem_y=15,
        element_type="Q4",
    )

    # Concrete
    concrete = ConcreteConfig(
        E=30000.0,  # MPa
        nu=0.2,
        f_c=35.0,  # MPa
        f_t=3.0,  # MPa
        G_f=0.1,  # N/mm
        model_type="elastic",  # No damage for debonding test
    )

    # FRP sheet bond law (bilinear, softening to 0)
    frp_bond = BilinearBondLaw(
        s1=0.5,  # mm (peak slip)
        s2=1.0,  # mm (complete debonding)
        tau1=1.0,  # MPa (peak bond stress)
    )

    # FRP sheet configuration
    frp = FRPSheet(
        thickness=0.165,  # mm (CFRP typical)
        width=50.0,  # mm
        bonded_length=100.0,  # mm
        y_position=0.0,  # Bottom face
        E=230000.0,  # MPa (CFRP)
        nu=0.3,
        bond_law=frp_bond,
    )

    # Loading (shear slip at free end)
    loading = MonotonicLoading(
        max_displacement=2.0,  # mm
        n_steps=100,
        load_x_center=150.0,  # Right edge
        load_halfwidth=10.0,
    )

    # Outputs
    outputs = OutputConfig(
        output_dir="examples/gutierrez_thesis/outputs/case_02",
        case_name="sspot_frp",
        save_load_displacement=True,
        save_crack_data=False,
        save_vtk=True,
        vtk_frequency=5,
        compute_slip_profiles=True,  # FRP slip(x)
        compute_bond_profiles=True,  # FRP tau(x)
    )

    # Assemble case
    case = CaseConfig(
        case_id="02_sspot_frp",
        description="SSPOT FRP sheet debonding test",
        geometry=geometry,
        concrete=concrete,
        loading=loading,
        outputs=outputs,
        rebar_layers=[],
        frp_sheets=[frp],
        fibres=None,
        subdomains=[],
        tolerance=1e-6,
        use_line_search=True,
        use_substepping=True,
    )

    return case
