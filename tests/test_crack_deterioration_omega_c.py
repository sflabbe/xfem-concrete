"""Unit tests for crack deterioration factor Ωc (Thesis Parity, TASK 1).

This test verifies the geometric crack-bar intersection algorithm and
the application of the crack deterioration factor Ωc per thesis Eq. 3.60-3.61.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest


def test_omega_c_no_cracks():
    """Test that Ωc = 1.0 when there are no cracks."""
    from xfem_clean.bond_slip import precompute_crack_context_for_bond

    # Simple horizontal bar
    nodes = np.array([[0.0, 0.1], [0.1, 0.1], [0.2, 0.1]], dtype=float)
    steel_segments = np.array([
        [0, 1, 0.1, 1.0, 0.0],  # Horizontal segment from node 0 to 1
        [1, 2, 0.1, 1.0, 0.0],  # Horizontal segment from node 1 to 2
    ], dtype=float)

    # No cracks
    crack_context = precompute_crack_context_for_bond(
        steel_segments=steel_segments,
        nodes=nodes,
        cracks=None,
    )

    # Should return large distance and r=1.0 for all segments
    assert crack_context.shape == (2, 2)
    assert np.all(crack_context[:, 0] > 1e9)  # Large distance
    assert np.allclose(crack_context[:, 1], 1.0)  # r = 1.0 → Ωc = 1.0

    print("✓ No cracks: Ωc = 1.0 for all segments")


def test_omega_c_transverse_crack_intersection():
    """Test crack intersection detection for a transverse crack."""
    from xfem_clean.bond_slip import precompute_crack_context_for_bond
    from xfem_clean.xfem.geometry import XFEMCrack

    # Horizontal bar at y=0.1, from x=0 to x=0.3
    nodes = np.array([
        [0.0, 0.1],
        [0.1, 0.1],
        [0.2, 0.1],
        [0.3, 0.1],
    ], dtype=float)
    steel_segments = np.array([
        [0, 1, 0.1, 1.0, 0.0],  # Segment 0-1: midpoint at x=0.05
        [1, 2, 0.1, 1.0, 0.0],  # Segment 1-2: midpoint at x=0.15
        [2, 3, 0.1, 1.0, 0.0],  # Segment 2-3: midpoint at x=0.25
    ], dtype=float)

    # Vertical crack at x=0.15 (should intersect segment 1-2)
    crack = XFEMCrack(
        x0=0.15, y0=0.0,
        tip_x=0.15, tip_y=0.2,
        stop_y=0.2,
        angle_deg=90.0,
        active=True
    )

    crack_context = precompute_crack_context_for_bond(
        steel_segments=steel_segments,
        nodes=nodes,
        cracks=[crack],
        cohesive_states=None,
        ft=3e6,
    )

    # Segment 0-1 (midpoint at x=0.05): crack is at x=0.15 → distance = +0.10 (ahead)
    # Segment 1-2 (midpoint at x=0.15): crack at x=0.15 → distance ≈ 0.0 (intersects)
    # Segment 2-3 (midpoint at x=0.25): crack at x=0.15 → distance = -0.10 (behind)

    assert crack_context.shape == (3, 2)

    # Check distances (signed along bar axis)
    assert abs(crack_context[0, 0] - 0.10) < 0.01, f"Segment 0: expected distance ≈0.10, got {crack_context[0, 0]}"
    assert abs(crack_context[1, 0]) < 0.01, f"Segment 1: expected distance ≈0.0, got {crack_context[1, 0]}"
    assert abs(crack_context[2, 0] - (-0.10)) < 0.01, f"Segment 2: expected distance ≈-0.10, got {crack_context[2, 0]}"

    # Without cohesive states, r should be 0.0 (conservative)
    assert np.allclose(crack_context[:, 1], 0.0), f"Expected r=0 without cohesive states, got {crack_context[:, 1]}"

    print("✓ Transverse crack intersection detected correctly")
    print(f"  Distances: {crack_context[:, 0]}")


def test_omega_c_formula():
    """Test the Ωc formula computation (thesis Eq. 3.60)."""
    from xfem_clean.bond_slip import BondSlipModelCode2010

    # Create bond law with crack deterioration enabled
    bond_law = BondSlipModelCode2010(
        f_cm=30e6,  # 30 MPa
        d_bar=0.012,  # 12 mm diameter
        condition="good",
        enable_crack_deterioration=True,
    )

    φ = bond_law.d_bar  # 12 mm = 0.012 m
    ft = 3e6  # 3 MPa (typical)

    # Test cases: distance x and traction ratio r
    test_cases = [
        # (x [m], r [-], expected Ωc)
        (0.0,     0.0, 0.0),        # At crack, no traction → Ωc = 0
        (0.0,     1.0, 1.0),        # At crack, full traction → Ωc = 1
        (φ,       0.0, 0.5),        # x = φ, no traction → Ωλ = 0.5
        (φ,       1.0, 1.0),        # x = φ, full traction → Ωc = 1
        (2*φ,     0.0, 1.0),        # x = 2φ, boundary → Ωc = 1
        (2*φ,     0.5, 1.0),        # x = 2φ, r=0.5 → Ωc = 1 (at boundary)
        (3*φ,     0.0, 1.0),        # x > 2φ → Ωc = 1 (no deterioration)
        (0.006,   0.5, 0.625),      # x = φ/2, r=0.5: Ωλ=0.25, Ωc = 0.25+0.5*0.75 = 0.625
    ]

    print("\n  Testing Ωc formula (Eq. 3.60):")
    print("  x/φ      r       Ωc_expected   Ωc_actual")
    print("  " + "-"*45)

    for x, r, expected_omega_c in test_cases:
        # Compute tn from r
        tn = r * ft

        # Call the method
        omega_c = bond_law.compute_crack_deterioration(
            dist_to_crack=x,
            w_max=0.0,  # Not used in current formula
            t_n_cohesive_stress=tn,
            f_t=ft
        )

        # Check result
        assert np.isclose(omega_c, expected_omega_c, atol=1e-6), (
            f"Ωc mismatch at x={x:.4f}m, r={r:.2f}: "
            f"expected={expected_omega_c:.4f}, got={omega_c:.4f}"
        )

        print(f"  {x/φ:5.2f}    {r:4.2f}    {expected_omega_c:6.4f}      {omega_c:6.4f}   ✓")

    print("  " + "-"*45)
    print("  ✓ Ωc formula verified")


def test_omega_c_effect_on_bond_stress():
    """Test that Ωc reduces bond stress as expected."""
    from xfem_clean.bond_slip import BondSlipModelCode2010

    # Create bond law with crack deterioration enabled
    bond_law = BondSlipModelCode2010(
        f_cm=30e6,  # 30 MPa
        d_bar=0.012,  # 12 mm diameter
        condition="good",
        enable_crack_deterioration=True,
    )

    # Test parameters
    s = 0.5e-3  # 0.5 mm slip
    s_max = 0.5e-3
    eps_s = 0.0  # No steel yielding

    # Case 1: No crack deterioration (Ωc = 1.0)
    tau_no_crack, dtau_no_crack = bond_law.tau_and_tangent(
        s, s_max, eps_s=eps_s, omega_crack=1.0
    )

    # Case 2: At crack with no traction (Ωc = 0.0, worst case)
    tau_at_crack, dtau_at_crack = bond_law.tau_and_tangent(
        s, s_max, eps_s=eps_s, omega_crack=0.0
    )

    # Case 3: Partial deterioration (Ωc = 0.5)
    tau_partial, dtau_partial = bond_law.tau_and_tangent(
        s, s_max, eps_s=eps_s, omega_crack=0.5
    )

    # Verify scaling
    assert np.isclose(tau_at_crack, 0.0, atol=1e-9), "Ωc=0 should give zero bond stress"
    assert np.isclose(tau_partial, 0.5 * tau_no_crack, rtol=1e-6), (
        f"Ωc=0.5 should give half bond stress: expected={0.5*tau_no_crack/1e6:.2f} MPa, "
        f"got={tau_partial/1e6:.2f} MPa"
    )

    print(f"\n  Bond stress reduction by Ωc:")
    print(f"    No crack (Ωc=1.0):    τ = {tau_no_crack/1e6:.2f} MPa")
    print(f"    Partial (Ωc=0.5):     τ = {tau_partial/1e6:.2f} MPa ({tau_partial/tau_no_crack*100:.0f}%)")
    print(f"    At crack (Ωc=0.0):    τ = {tau_at_crack/1e6:.2f} MPa")
    print(f"  ✓ Ωc correctly scales bond stress")


def test_omega_c_combined_with_omega_y():
    """Test that Ωc and Ωy combine multiplicatively."""
    from xfem_clean.bond_slip import BondSlipModelCode2010

    # Create bond law with BOTH yielding and crack deterioration enabled
    bond_law = BondSlipModelCode2010(
        f_cm=30e6,
        d_bar=0.012,
        condition="good",
        f_y=500e6,
        E_s=200e9,
        enable_yielding_reduction=True,
        enable_crack_deterioration=True,
    )

    s = 0.5e-3
    s_max = 0.5e-3

    # Reference: no yielding, no crack
    eps_y = bond_law.f_y / bond_law.E_s
    tau_ref, _ = bond_law.tau_and_tangent(s, s_max, eps_s=0.0, omega_crack=1.0)

    # Case 1: Yielding only (Ωy < 1, Ωc = 1)
    eps_s_yielded = 2.0 * eps_y
    tau_yield_only, _ = bond_law.tau_and_tangent(s, s_max, eps_s=eps_s_yielded, omega_crack=1.0)
    omega_y = tau_yield_only / tau_ref

    # Case 2: Crack deterioration only (Ωy = 1, Ωc < 1)
    omega_c = 0.5
    tau_crack_only, _ = bond_law.tau_and_tangent(s, s_max, eps_s=0.0, omega_crack=omega_c)

    # Case 3: Both (Ωy < 1, Ωc < 1)
    tau_combined, _ = bond_law.tau_and_tangent(s, s_max, eps_s=eps_s_yielded, omega_crack=omega_c)

    # Should have: tau_combined = tau_ref * Ωy * Ωc
    expected_combined = tau_ref * omega_y * omega_c
    assert np.isclose(tau_combined, expected_combined, rtol=1e-6), (
        f"Combined factors should multiply: expected={expected_combined/1e6:.2f} MPa, "
        f"got={tau_combined/1e6:.2f} MPa"
    )

    print(f"\n  Combined Ωy × Ωc:")
    print(f"    Reference (Ωy=1, Ωc=1):    τ = {tau_ref/1e6:.2f} MPa")
    print(f"    Yielding (Ωy={omega_y:.3f}, Ωc=1): τ = {tau_yield_only/1e6:.2f} MPa")
    print(f"    Crack (Ωy=1, Ωc=0.5):      τ = {tau_crack_only/1e6:.2f} MPa")
    print(f"    Combined (Ωy={omega_y:.3f}, Ωc=0.5): τ = {tau_combined/1e6:.2f} MPa")
    print(f"  ✓ Factors combine multiplicatively: τ = τ_ref × {omega_y:.3f} × 0.5 = {tau_combined/1e6:.2f} MPa")


if __name__ == "__main__":
    print("=" * 60)
    print("TASK 1: Crack Deterioration Ωc Tests")
    print("=" * 60)

    print("\n[1/5] Testing Ωc with no cracks...")
    test_omega_c_no_cracks()

    print("\n[2/5] Testing crack-bar intersection geometry...")
    test_omega_c_transverse_crack_intersection()

    print("\n[3/5] Testing Ωc formula (Eq. 3.60)...")
    test_omega_c_formula()

    print("\n[4/5] Testing Ωc effect on bond stress...")
    test_omega_c_effect_on_bond_stress()

    print("\n[5/5] Testing Ωy × Ωc combination...")
    test_omega_c_combined_with_omega_y()

    print("\n" + "=" * 60)
    print("✓ All TASK 1 tests passed!")
    print("=" * 60)
