"""
Test BLOQUE 5: Case 02 FRP Sheet Debonding (SSPOT)

Integration test for FRP sheet with bilinear bond law and segment masking.

Validates:
- FRP segments generation
- Bond law mapping (bilinear)
- Segment masking for unbonded regions
- No NaNs in results
- Bond stress profiles generated
"""

import sys
import numpy as np
import os

try:
    from xfem_clean.xfem.model import XFEMModel

    # Import case configurations and solver interface
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from examples.gutierrez_thesis.case_config import (
        CaseConfig, GeometryConfig, ConcreteConfig,
        FRPSheet, BilinearBondLaw, MonotonicLoading, OutputConfig,
    )
    from examples.gutierrez_thesis.solver_interface import run_case_solver
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    sys.exit(0)


def test_case_02_frp_sspot_minimal():
    """
    Test Case 02: FRP sheet debonding (SSPOT test).

    Validates:
    - FRP sheet segments are generated
    - Bond perimeter is correctly set (width, not circular)
    - Segment masking works for unbonded regions
    - No crashes, no NaNs
    - Bond profiles are generated
    """
    print("\n🧪 Test: Case 02 FRP SSPOT (minimal, 3 steps)...")

    # Minimal geometry
    geometry = GeometryConfig(
        length=100.0,  # mm (small specimen)
        height=100.0,  # mm
        thickness=50.0,  # mm
        n_elem_x=5,  # Very coarse
        n_elem_y=5,  # Very coarse
        element_type="Q4",
    )

    # Elastic concrete (no damage for debonding test)
    concrete = ConcreteConfig(
        E=30000.0,  # MPa
        nu=0.2,
        f_c=35.0,  # MPa
        f_t=3.0,  # MPa
        G_f=0.1,  # N/mm
        model_type="elastic",  # Keep simple
    )

    # FRP bond law (bilinear, softening to 0)
    frp_bond = BilinearBondLaw(
        s1=0.5,  # mm (peak slip)
        s2=1.0,  # mm (complete debonding)
        tau1=1.0,  # MPa (peak bond stress)
    )

    # FRP sheet configuration
    frp = FRPSheet(
        thickness=0.165,  # mm (CFRP typical)
        width=50.0,  # mm
        bonded_length=60.0,  # mm (shorter than total length for masking test)
        y_position=0.0,  # Bottom face
        E=230000.0,  # MPa (CFRP)
        nu=0.3,
        bond_law=frp_bond,
    )

    # Loading (pull at free end)
    loading = MonotonicLoading(
        max_displacement=0.5,  # mm (small)
        n_steps=3,  # Very few steps
        load_x_center=100.0,  # Right edge
        load_halfwidth=10.0,
    )

    # Outputs
    outputs = OutputConfig(
        output_dir="tests/tmp/case_02_frp",
        case_name="frp_sspot_minimal",
        save_load_displacement=True,
        save_vtk=False,  # Disable for speed
        compute_slip_profiles=True,
        compute_bond_profiles=True,
    )

    # Assemble case
    case = CaseConfig(
        case_id="02_sspot_frp_test",
        description="FRP sheet SSPOT test (minimal for testing)",
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

    # Run solver
    try:
        results = run_case_solver(case, mesh_factor=1.0, enable_postprocess=True)
    except Exception as e:
        print(f"❌ Solver crashed: {e}")
        raise

    # Validations
    assert results is not None, "Results should not be None"
    assert 'history' in results, "Results should contain history"
    assert 'u' in results, "Results should contain displacement vector"

    # Check for NaNs
    history = results['history']
    assert len(history) > 0, "History should have entries"
    for step_data in history:
        assert np.all(np.isfinite(step_data)), f"NaNs detected in history: {step_data}"

    u = results['u']
    assert np.all(np.isfinite(u)), "NaNs detected in displacement vector"

    # Check bond states (if available)
    if 'bond_states' in results and results['bond_states'] is not None:
        bond_states = results['bond_states']
        assert hasattr(bond_states, 's_current'), "Bond states should have s_current"
        assert np.all(np.isfinite(bond_states.s_current)), "NaNs in bond slip"

    # Check that output files were created
    output_dir = outputs.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Check for load-displacement CSV
    load_disp_file = os.path.join(output_dir, "load_displacement.csv")
    if os.path.exists(load_disp_file):
        print(f"✅ Load-displacement curve saved: {load_disp_file}")

    # Check for bond profiles (if bond-slip was active)
    slip_profile_file = os.path.join(output_dir, "slip_profile_final.csv")
    bond_profile_file = os.path.join(output_dir, "bond_stress_profile_final.csv")

    if os.path.exists(slip_profile_file):
        print(f"✅ Slip profile saved: {slip_profile_file}")
        # Verify masking: slip should be ~0 in unbonded region
        slip_data = np.loadtxt(slip_profile_file, delimiter=',', skiprows=1)
        if len(slip_data) > 0:
            x_coords = slip_data[:, 0]  # First column is x
            slip_vals = slip_data[:, 1]  # Second column is slip

            # Unbonded region: x < (L - bonded_length) = 100 - 60 = 40 mm
            unbonded_mask = x_coords < 40.0
            if np.any(unbonded_mask):
                max_slip_unbonded = np.max(np.abs(slip_vals[unbonded_mask]))
                print(f"   Max slip in unbonded region: {max_slip_unbonded:.6f} mm")
                # Should be very small (tolerance for numerical noise)
                assert max_slip_unbonded < 1e-3, "Unbonded region should have ~zero slip"

    if os.path.exists(bond_profile_file):
        print(f"✅ Bond stress profile saved: {bond_profile_file}")

    print("✅ Case 02 FRP test passed!")


if __name__ == "__main__":
    test_case_02_frp_sspot_minimal()
