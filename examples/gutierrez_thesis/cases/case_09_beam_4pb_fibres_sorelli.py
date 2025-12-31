"""
Case 09: Fibre Reinforced 4PB Beam (SORELLI et al.)

Small-scale 4-point bending beam with fibre bridging and notch.
CTOD (Crack Tip Opening Displacement) measurement enabled.

Reference: Gutiérrez thesis Chapter 5, Section 5.4.4, Fig. 5.38-5.39
"""

from examples.gutierrez_thesis.case_config import (
    CaseConfig,
    GeometryConfig,
    ConcreteConfig,
    FibreConfig,
    FibreReinforcement,
    BanholzerBondLaw,
    MonotonicLoading,
    OutputConfig,
)


def create_case_09() -> CaseConfig:
    """
    Fibre reinforced 4PB beam (Sorelli et al.) with CTOD tracking.

    Specimen:
        - Dimensions: 320x100x40 mm (L x H x b)
        - Span: 300 mm (supports at x=10mm and x=310mm)
        - Notch: depth 85mm from bottom (ligament = 15mm)
        - Notch position: x=160mm (midspan)
    Loading:
        - 4-point bending
        - Load points at x=110mm and x=210mm (100mm apart)
    Fibres:
        - Density: 3.43 fibres/cm² = 34300 fibres/m²
        - 116 fibres in crack section
        - Model 50% explicitly (explicit_fraction=0.5)
    """

    # Geometry
    geometry = GeometryConfig(
        length=320.0,  # mm
        height=100.0,  # mm
        thickness=40.0,  # mm (out-of-plane width)
        n_elem_x=64,
        n_elem_y=20,
        element_type="Q4",
        notch_depth=85.0,  # mm from bottom (ligament = 15mm)
        notch_x=160.0,  # mm (midspan)
    )

    # Concrete (high-performance fibre-reinforced concrete)
    # Using typical HPFRC values (adjust based on thesis if available)
    concrete = ConcreteConfig(
        E=35000.0,  # MPa (typical for HPFRC)
        nu=0.2,
        f_c=60.0,  # MPa (high strength)
        f_t=5.0,  # MPa (enhanced by fibres)
        G_f=0.15,  # N/mm
        eps_cu=-0.0035,
        model_type="cdp_full",
    )

    # Fibre properties (steel fibres)
    # Assuming typical hooked-end steel fibres
    fibre = FibreConfig(
        E=200000.0,  # MPa (steel)
        nu=0.3,
        diameter=0.5,  # mm (typical steel fibre)
        length=30.0,  # mm (typical for such specimens)
        density=34300.0,  # fibres/m² (3.43 fibres/cm²)
        orientation_deg=0.0,  # Horizontal orientation preferred
        volume_fraction_multiplier=1.0,
    )

    # Fibre bond law (Banholzer)
    # Using typical values for steel fibres in concrete
    fibre_bond = BanholzerBondLaw(
        s0=0.5,  # mm
        a=3.0,  # Softening multiplier
        tau1=8.0,  # MPa (peak in rising)
        tau2=6.0,  # MPa (peak in softening)
        tau_f=1.0,  # MPa (residual)
    )

    # Fibre reinforcement
    fibres = FibreReinforcement(
        fibre=fibre,
        bond_law=fibre_bond,
        random_seed=42,
        active_near_crack_only=True,
        activation_distance=50.0,  # mm (activate fibres near crack)
    )

    # Loading
    # 4PB: loads at x=110mm and x=210mm
    # Using wide loading region to approximate both load points
    loading = MonotonicLoading(
        max_displacement=2.0,  # mm (small specimen, small displacement)
        n_steps=200,
        load_x_center=160.0,  # mm (midspan)
        load_halfwidth=50.0,  # mm (covers both load points)
    )

    # Outputs
    # CTOD measurement: difference in vertical displacement across notch tip
    # Node pairs will be identified near notch tip at x=160mm, y≈85mm
    outputs = OutputConfig(
        output_dir="examples/gutierrez_thesis/outputs/case_09",
        case_name="beam_4pb_fibres_sorelli",
        save_load_displacement=True,
        save_crack_data=True,
        save_crack_pattern=True,
        save_damage_field=True,
        save_vtk=True,
        vtk_frequency=5,
        compute_CTOD=True,  # Enable CTOD computation
        compute_crack_widths=True,
        compute_slip_profiles=False,  # No rebar
        compute_bond_profiles=False,
        compute_steel_forces=False,
        # CTOD node pairs will be set dynamically near notch tip
        ctod_node_pairs=None,  # Auto-detect nodes near notch
    )

    # Assemble case
    case = CaseConfig(
        case_id="09_beam_4pb_fibres_sorelli",
        description="Fibre reinforced 4PB beam (Sorelli) with CTOD tracking",
        geometry=geometry,
        concrete=concrete,
        loading=loading,
        outputs=outputs,
        rebar_layers=[],  # No conventional reinforcement
        fibres=fibres,
        tolerance=1e-6,
        use_line_search=True,
        use_substepping=True,
    )

    return case
