"""
Smoke test to verify basic integration of multicrack + bond-slip + postprocess.

Tests that the plumbing works end-to-end without requiring full simulations.
"""

import sys
from pathlib import Path
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import numpy as np
except ImportError as e:
    pytest.skip(f"Import error: {e}", allow_module_level=True)

def test_imports():
    """Test that all critical imports work."""
    print("Testing imports...")

    # Core solver imports
    from xfem_clean.xfem.analysis_single import run_analysis_xfem, BCSpec
    from xfem_clean.xfem.multicrack import run_analysis_xfem_multicrack
    from xfem_clean.cohesive_laws import CohesiveLaw
    from xfem_clean.bond_slip import CustomBondSlipLaw

    # Case config imports
    from examples.gutierrez_thesis.case_config import (
        CaseConfig,
        ConcreteConfig,
        GeometryConfig,
        MonotonicLoading,
        OutputConfig,
    )

    # Solver interface imports
    from examples.gutierrez_thesis.solver_interface import (
        run_case_solver,
        map_bond_law,
        build_bcs_from_case,
        generate_cyclic_u_targets,
    )

    # Postprocess imports
    from examples.gutierrez_thesis.postprocess_comprehensive import (
        postprocess_case,
        compute_slip_profile,
        compute_bond_stress_profile,
    )

    print("✓ All imports successful")


def test_cyclic_trajectory_generator():
    """Test cyclic u_targets generator."""
    print("\nTesting cyclic trajectory generator...")

    from examples.gutierrez_thesis.solver_interface import generate_cyclic_u_targets

    # Test simple case: 2 targets, 1 cycle each
    targets_mm = [1.0, 2.0]
    n_cycles = 1

    u_traj = generate_cyclic_u_targets(targets_mm, n_cycles)

    print(f"  Generated trajectory: {u_traj}")
    print(f"  Length: {len(u_traj)}")

    # Should have: 0, +1, 0, -1, 0, +2, 0, -2, 0 (with duplicates removed)
    assert len(u_traj) > 0, "Trajectory should not be empty"
    assert u_traj[0] == 0.0, "Should start at zero"

    print("✓ Cyclic trajectory generator works")


def test_bc_spec_creation():
    """Test BC specification creation."""
    print("\nTesting BC specification creation...")

    from examples.gutierrez_thesis.case_config import (
        CaseConfig,
        ConcreteConfig,
        GeometryConfig,
        MonotonicLoading,
        OutputConfig,
        RebarLayer,
        SteelConfig,
        CEBFIPBondLaw,
    )
    from examples.gutierrez_thesis.solver_interface import (
        case_config_to_xfem_model,
        build_bcs_from_case,
    )
    from xfem_clean.fem.mesh import structured_quad_mesh

    # Create a minimal pullout case
    case = CaseConfig(
        case_id="test_pullout",
        description="Smoke test pullout",
        geometry=GeometryConfig(
            length=300.0,  # mm
            height=100.0,
            thickness=100.0,
            n_elem_x=6,
            n_elem_y=2,
        ),
        concrete=ConcreteConfig(
            E=30000.0,  # MPa
            nu=0.2,
            f_c=30.0,
            f_t=3.0,
            G_f=0.1,  # N/mm
            model_type="elastic",
        ),
        loading=MonotonicLoading(
            max_displacement=1.0,  # mm
            n_steps=3,
            load_x_center=0.0,
            load_halfwidth=50.0,
        ),
        outputs=OutputConfig(
            output_dir="test_outputs",
            case_name="smoke_test",
        ),
        rebar_layers=[
            RebarLayer(
                diameter=12.0,  # mm
                y_position=50.0,
                n_bars=1,
                steel=SteelConfig(E=200000.0, nu=0.3, f_y=500.0),
                bond_law=CEBFIPBondLaw(
                    s1=0.6, s2=0.6, s3=1.0,
                    tau_max=10.0, tau_f=5.0,
                ),
                bond_disabled_x_range=(0.0, 100.0),  # Empty elements
            )
        ],
    )

    # Create model
    model = case_config_to_xfem_model(case)

    # Create mesh
    nodes, elems = structured_quad_mesh(model.L, model.H, 6, 2)

    # Build BCs
    from xfem_clean.rebar import prepare_rebar_segments
    rebar_segs = prepare_rebar_segments(nodes, cover=model.cover)
    bc_spec = build_bcs_from_case(case, nodes, model, rebar_segs=rebar_segs)

    print(f"  Fixed DOFs: {len(bc_spec.fixed_dofs)}")
    print(f"  Prescribed DOFs: {len(bc_spec.prescribed_dofs)}")
    print(f"  Prescribed scale: {bc_spec.prescribed_scale}")

    assert len(bc_spec.prescribed_dofs) > 0, "Should have prescribed DOFs"
    assert bc_spec.prescribed_scale == 1.0, "Pullout should have positive scale"

    print("✓ BC specification creation works")


def test_bond_law_mapping():
    """Test bond law mapping."""
    print("\nTesting bond law mapping...")

    from examples.gutierrez_thesis.case_config import CEBFIPBondLaw
    from examples.gutierrez_thesis.solver_interface import map_bond_law

    # Create config bond law
    config_law = CEBFIPBondLaw(
        s1=0.6,  # mm
        s2=0.8,  # Must be > s1
        s3=1.0,
        tau_max=10.0,  # MPa
        tau_f=5.0,
        alpha=0.4,
    )

    # Map to solver bond law
    solver_law = map_bond_law(config_law)

    print(f"  Mapped bond law type: {type(solver_law).__name__}")

    # Test tau computation
    s_test = 0.001  # 1 mm in meters
    tau, k_t = solver_law.tau_and_tangent(s_test, s_test)

    print(f"  tau at s=1mm: {tau/1e6:.2f} MPa")

    assert tau > 0, "Bond stress should be positive for non-zero slip"

    print("✓ Bond law mapping works")


if __name__ == "__main__":
    print("="*70)
    print("SMOKE TESTS - Multicrack + Bond-slip + Postprocess Integration")
    print("="*70)

    try:
        test_imports()
        test_cyclic_trajectory_generator()
        test_bc_spec_creation()
        test_bond_law_mapping()

        print("\n" + "="*70)
        print("✓ ALL SMOKE TESTS PASSED")
        print("="*70)

    except Exception as e:
        print(f"\n✗ SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
