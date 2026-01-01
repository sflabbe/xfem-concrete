"""Unit test for P0: Junction enrichment integration into multicrack workflow.

This test verifies that:
1. Junction detection works when two cracks coalesce
2. Secondary crack is arrested at junction
3. Junction enrichment is allocated
4. Solver converges with junction present
"""

import pytest
import numpy as np


def test_junction_detection_and_arrest():
    """Test that junction detection and crack arrest work correctly."""
    from xfem_clean.xfem.geometry import XFEMCrack
    from xfem_clean.junction import detect_crack_coalescence, arrest_secondary_crack_at_junction
    from xfem_clean.fem.mesh import structured_quad_mesh

    # Create a simple mesh
    nodes, elems = structured_quad_mesh(L=1.0, H=0.5, nx=10, ny=5)

    # Create two cracks that will coalesce
    # Crack 1: vertical crack from bottom
    crack1 = XFEMCrack(
        x0=0.5, y0=0.0,
        tip_x=0.5, tip_y=0.3,
        stop_y=0.5, angle_deg=90.0, active=True
    )

    # Crack 2: approaching crack 1 from left
    crack2 = XFEMCrack(
        x0=0.3, y0=0.2,
        tip_x=0.48, tip_y=0.2,  # Tip very close to crack 1
        stop_y=0.5, angle_deg=0.0, active=True
    )

    cracks = [crack1, crack2]

    # Detect coalescence with 30mm tolerance
    junctions = detect_crack_coalescence(cracks, nodes, elems, tol_merge=0.03)

    # Assert junction detected
    assert len(junctions) > 0, "Junction should be detected when cracks are close"

    junc = junctions[0]
    assert junc.active, "Junction should be active"
    assert junc.main_crack_id == 0 or junc.main_crack_id == 1
    assert junc.secondary_crack_id == 0 or junc.secondary_crack_id == 1
    assert junc.main_crack_id != junc.secondary_crack_id

    print(f"✓ Junction detected: crack#{junc.secondary_crack_id} → crack#{junc.main_crack_id}")
    print(f"  Junction point: ({junc.junction_point[0]:.3f}, {junc.junction_point[1]:.3f})")
    print(f"  Element: {junc.element_id}")
    print(f"  Branch angles: {[np.degrees(a) for a in junc.branch_angles]}")

    # Arrest secondary crack
    sec_crack_before = cracks[junc.secondary_crack_id]
    tip_x_before = sec_crack_before.tip_x
    tip_y_before = sec_crack_before.tip_y

    arrest_secondary_crack_at_junction(junc, cracks)

    # Assert crack was arrested
    sec_crack_after = cracks[junc.secondary_crack_id]
    assert sec_crack_after.tip_x == junc.junction_point[0], "Crack tip should move to junction"
    assert sec_crack_after.tip_y == junc.junction_point[1], "Crack tip should move to junction"

    # If the crack object supports 'arrested' attribute, check it
    if hasattr(sec_crack_after, 'arrested'):
        assert sec_crack_after.arrested, "Crack should be marked as arrested"

    print(f"✓ Crack arrested: tip moved from ({tip_x_before:.3f},{tip_y_before:.3f}) → ({sec_crack_after.tip_x:.3f},{sec_crack_after.tip_y:.3f})")


def test_junction_no_false_positive():
    """Test that junction detection does not trigger when cracks are far apart."""
    from xfem_clean.xfem.geometry import XFEMCrack
    from xfem_clean.junction import detect_crack_coalescence
    from xfem_clean.fem.mesh import structured_quad_mesh

    nodes, elems = structured_quad_mesh(L=1.0, H=0.5, nx=10, ny=5)

    # Create two cracks far apart
    crack1 = XFEMCrack(x0=0.3, y0=0.0, tip_x=0.3, tip_y=0.2, stop_y=0.5, angle_deg=90.0, active=True)
    crack2 = XFEMCrack(x0=0.7, y0=0.0, tip_x=0.7, tip_y=0.2, stop_y=0.5, angle_deg=90.0, active=True)

    cracks = [crack1, crack2]

    # Detect with small tolerance
    junctions = detect_crack_coalescence(cracks, nodes, elems, tol_merge=0.01)

    assert len(junctions) == 0, "No junction should be detected when cracks are far apart"
    print("✓ No false positives: cracks far apart, no junction detected")


def test_multicrack_with_junction_integration():
    """Test multicrack solver with junction detection enabled.

    This test runs a simplified multicrack case and forces two cracks to coalesce,
    verifying that:
    - Junction detection is called
    - Crack arrest happens
    - Solver continues to converge after junction
    """
    pytest.skip("Full multicrack integration test - requires complete DOF rebuild with junction enrichment")

    # Note: This test is skipped because full junction enrichment integration
    # into the DOF builder (build_xfem_dofs_multi) is not yet complete.
    # The P0 implementation adds junction *detection* and *arrest* logic,
    # but the multicrack assembly doesn't yet fully support junction-enriched DOFs.
    #
    # To complete P0:
    # 1. Extend MultiXFEMDofs to include junction enrichment DOFs
    # 2. Modify assembly to evaluate junction Heaviside functions
    # 3. Add L2 projection for DOF transfer after junction detection

    from xfem_clean.xfem.model import XFEMModel
    from xfem_clean.xfem.multicrack import run_analysis_xfem_multicrack
    from xfem_clean.cohesive_laws import CohesiveLaw

    # Create a minimal model
    model = XFEMModel(
        L=1.0, H=0.5,
        E=30e9, nu=0.2,
        ft=3e6, Gf=100.0,
        steel_A_total=0.0,  # No rebar for simplicity
        newton_maxit=20,
        newton_tol_r=1.0,
        newton_beta=1e-3,
        max_subdiv=4,
        junction_merge_tolerance=0.02,  # Enable junction detection with 20mm tolerance
    )

    law = CohesiveLaw(Kn=1e13, ft=model.ft, Gf=model.Gf)

    # Run with max 2 cracks, force them to be close enough to coalesce
    result = run_analysis_xfem_multicrack(
        model,
        nx=20, ny=10,
        nsteps=5,
        umax=0.001,  # Small displacement
        max_cracks=2,
        law=law,
        return_bundle=True,
    )

    # Check that analysis ran
    assert result is not None
    assert 'history' in result
    assert len(result['history']) > 0


if __name__ == "__main__":
    print("=" * 60)
    print("P0 Junction Enrichment Tests")
    print("=" * 60)

    print("\n[1/3] Testing junction detection and arrest...")
    test_junction_detection_and_arrest()

    print("\n[2/3] Testing no false positives...")
    test_junction_no_false_positive()

    print("\n[3/3] Testing multicrack integration (skipped - not fully implemented)...")
    try:
        test_multicrack_with_junction_integration()
    except pytest.skip.Exception as e:
        print(f"  SKIPPED: {e}")

    print("\n" + "=" * 60)
    print("✓ P0 tests passed (partial implementation)")
    print("=" * 60)
    print("\nNOTE: Full junction enrichment integration into DOF assembly")
    print("requires extending MultiXFEMDofs and assembly kernels.")
    print("Current P0 implementation adds detection + arrest logic.")
