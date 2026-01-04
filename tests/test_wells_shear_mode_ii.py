"""Unit tests for PART C: Wells-type shear traction (Mode II).

This test verifies that:
1. Shear stiffness degrades exponentially with opening: k_s(w) = k_s0 * exp(h_s * w)
   where h_s = ln(k_s1/k_s0) / w1  [SI units: 1/m]
   and w1 is the reference opening (default: 1 mm for thesis parity, Gutierrez Eq. 3.26-3.27)
2. Shear traction scales linearly with slip: t_t = k_s(w) * s
3. Cross-coupling tangent: dt_t/dw = h_s * k_s(w) * s
4. Pure Mode I (s=0) reproduces standard behavior
5. Mixed-mode 2x2 tangent has correct structure

Thesis reference: Gutierrez dissertation (KIT IR 10.5445/IR/1000124842)
  - Eq. (3.26): k_s(ω_n) = d0 * exp[ln(d1/d0) * ω_n/ω_ref]
  - Eq. (3.27): h_s corresponds to ln(d1/d0)/ω_ref (normalized by reference opening)
  - ω_ref = w1 = 1 mm (default in this codebase for thesis parity)
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import math


def test_wells_shear_stiffness_degradation():
    """Test that k_s degrades exponentially with opening."""
    from xfem_clean.cohesive_laws import CohesiveLaw, CohesiveState, cohesive_update_mixed

    # Create Wells-type cohesive law
    law = CohesiveLaw(
        Kn=1e13,  # 10^13 Pa/m
        ft=3e6,   # 3 MPa
        Gf=100.0,  # 100 J/m^2
        mode="mixed",
        shear_model="wells",  # Enable Wells model
        k_s0=1e13,  # Initial shear stiffness
        k_s1=1e11,  # Final shear stiffness (1% of initial)
    )

    # Fixed slip, vary opening
    s = 0.1e-3  # 0.1 mm slip (constant)
    openings = np.array([0.0, 0.5e-3, 1.0e-3, 2.0e-3, 5.0e-3])  # mm

    # Decay parameter (Thesis Eq. 3.26-3.27: h_s = ln(k_s1/k_s0) / w1)
    h_s = math.log(law.k_s1 / law.k_s0) / law.w1

    # Residual stiffness floor (implementation detail for numerical stability)
    k_res = law.kres_factor * law.Kn

    print("\n  Wells Shear Stiffness Degradation:")
    print("  w (mm)    k_s(w) / k_s0    t_t (MPa)    Expected k_s/k_s0")
    print("  " + "-" * 60)

    state = CohesiveState(delta_max=0.0, damage=0.0)

    for w in openings:
        # Compute mixed-mode response
        t, K_mat, st_new = cohesive_update_mixed(law, delta_n=w, delta_t=s, st=state)

        t_n = t[0]
        t_t = t[1]

        # Expected shear stiffness (with residual floor applied)
        k_s_expected = max(law.k_s0 * math.exp(h_s * w), k_res)
        k_s_actual = t_t / s if s > 1e-14 else 0.0

        # Verify exponential degradation
        assert np.isclose(k_s_actual, k_s_expected, rtol=1e-3), (
            f"k_s mismatch at w={w*1e3:.2f} mm: expected={k_s_expected:.2e}, actual={k_s_actual:.2e}"
        )

        print(f"  {w*1e3:5.2f}     {k_s_actual/law.k_s0:.6f}         "
              f"{t_t/1e6:.3f}        {k_s_expected/law.k_s0:.6f}")

    print("  " + "-" * 60)
    print("  ✓ Exponential degradation k_s(w) = k_s0 * exp(h_s * w) verified")


def test_wells_linear_in_slip():
    """Test that shear traction is linear in slip at fixed opening."""
    from xfem_clean.cohesive_laws import CohesiveLaw, CohesiveState, cohesive_update_mixed

    law = CohesiveLaw(
        Kn=1e13,
        ft=3e6,
        Gf=100.0,
        mode="mixed",
        shear_model="wells",
        k_s0=1e13,
        k_s1=1e11,
    )

    # Fixed opening, vary slip
    w = 1.0e-3  # 1 mm opening (fixed)
    slips = np.array([0.0, 0.05e-3, 0.1e-3, 0.2e-3, 0.5e-3])  # mm

    # Decay parameter (Thesis Eq. 3.26-3.27: h_s = ln(k_s1/k_s0) / w1)
    h_s = math.log(law.k_s1 / law.k_s0) / law.w1

    # Residual stiffness floor (implementation detail for numerical stability)
    k_res = law.kres_factor * law.Kn
    k_s_w = max(law.k_s0 * math.exp(h_s * w), k_res)  # Shear stiffness at w (with floor)

    print("\n  Wells Shear Traction (Linear in Slip):")
    print("  s (mm)    t_t (MPa)    t_t/s (Pa/m)    k_s(w) (Pa/m)")
    print("  " + "-" * 60)

    state = CohesiveState(delta_max=0.0, damage=0.0)

    for s in slips:
        # Compute mixed-mode response
        t, K_mat, st_new = cohesive_update_mixed(law, delta_n=w, delta_t=s, st=state)

        t_t = t[1]
        dt_t_ds = K_mat[1, 1]

        # Expected shear traction
        t_t_expected = k_s_w * s

        # Verify linearity
        if s > 1e-14:
            assert np.isclose(t_t, t_t_expected, rtol=1e-3), (
                f"t_t mismatch at s={s*1e3:.3f} mm: expected={t_t_expected/1e6:.3f} MPa, actual={t_t/1e6:.3f} MPa"
            )

            # Verify tangent
            assert np.isclose(dt_t_ds, k_s_w, rtol=1e-3), (
                f"dt_t/ds mismatch: expected={k_s_w:.2e}, actual={dt_t_ds:.2e}"
            )

            print(f"  {s*1e3:5.3f}     {t_t/1e6:6.3f}       {t_t/s:.2e}       {k_s_w:.2e}")
        else:
            print(f"  {s*1e3:5.3f}     {t_t/1e6:6.3f}       ---              {k_s_w:.2e}")

    print("  " + "-" * 60)
    print("  ✓ Linear relationship t_t = k_s(w) * s verified")


def test_wells_cross_coupling_tangent():
    """Test that cross-coupling tangent dt_t/dw = h_s * k_s(w) * s."""
    from xfem_clean.cohesive_laws import CohesiveLaw, CohesiveState, cohesive_update_mixed

    law = CohesiveLaw(
        Kn=1e13,
        ft=3e6,
        Gf=100.0,
        mode="mixed",
        shear_model="wells",
        k_s0=1e13,
        k_s1=1e11,
    )

    # Test at various (w, s) pairs
    test_cases = [
        (0.5e-3, 0.1e-3),  # Small opening, small slip
        (1.0e-3, 0.2e-3),  # Moderate opening, moderate slip
        (2.0e-3, 0.5e-3),  # Large opening, large slip
    ]

    # Decay parameter (Thesis Eq. 3.26-3.27: h_s = ln(k_s1/k_s0) / w1)
    h_s = math.log(law.k_s1 / law.k_s0) / law.w1

    # Residual stiffness floor (implementation detail for numerical stability)
    k_res = law.kres_factor * law.Kn

    print("\n  Wells Cross-Coupling Tangent (dt_t/dw):")
    print("  w (mm)  s (mm)   dt_t/dw (Pa/m^2)   Expected")
    print("  " + "-" * 60)

    state = CohesiveState(delta_max=0.0, damage=0.0)

    for w, s in test_cases:
        # Compute mixed-mode response
        t, K_mat, st_new = cohesive_update_mixed(law, delta_n=w, delta_t=s, st=state)

        dt_t_dw = K_mat[1, 0]  # Cross-coupling term

        # Expected cross-coupling (accounting for residual floor)
        k_s_w_unfloored = law.k_s0 * math.exp(h_s * w)
        k_s_w = max(k_s_w_unfloored, k_res)
        # dt_t/dw = h_s * k_s(w) * s, but only if k_s(w) > k_res (not floored)
        # If floored, dt_t/dw = 0 (constant at floor)
        if k_s_w_unfloored > k_res:
            dt_t_dw_expected = h_s * k_s_w * s
        else:
            dt_t_dw_expected = 0.0  # Floored, no derivative

        # Verify cross-coupling
        assert np.isclose(dt_t_dw, dt_t_dw_expected, rtol=1e-3), (
            f"dt_t/dw mismatch at (w={w*1e3:.2f}, s={s*1e3:.2f}): "
            f"expected={dt_t_dw_expected:.2e}, actual={dt_t_dw:.2e}"
        )

        print(f"  {w*1e3:5.2f}   {s*1e3:5.3f}    {dt_t_dw:.2e}      {dt_t_dw_expected:.2e}")

    print("  " + "-" * 60)
    print("  ✓ Cross-coupling dt_t/dw = h_s * k_s(w) * s verified")


def test_mode_i_regression_wells():
    """Test that pure Mode I (s=0) reproduces standard cohesive behavior."""
    from xfem_clean.cohesive_laws import CohesiveLaw, CohesiveState, cohesive_update, cohesive_update_mixed

    # Create two identical laws: one Wells, one standard
    law_wells = CohesiveLaw(
        Kn=1e13,
        ft=3e6,
        Gf=100.0,
        mode="mixed",
        shear_model="wells",
        k_s0=1e13,
        k_s1=1e11,
    )

    law_standard = CohesiveLaw(
        Kn=1e13,
        ft=3e6,
        Gf=100.0,
        mode="I",  # Mode I only
    )

    # Test at various openings
    openings = np.array([0.0, 0.5e-3, 1.0e-3, 2.0e-3, 5.0e-3])

    print("\n  Mode I Regression (s=0):")
    print("  w (mm)    t_n Wells (MPa)   t_n Standard (MPa)   Match?")
    print("  " + "-" * 60)

    state = CohesiveState(delta_max=0.0, damage=0.0)

    for w in openings:
        # Wells with s=0 (pure Mode I)
        t_wells, K_wells, st_wells = cohesive_update_mixed(law_wells, delta_n=w, delta_t=0.0, st=state)
        t_n_wells = t_wells[0]
        t_t_wells = t_wells[1]

        # Standard Mode I
        t_n_std, K_std, st_std = cohesive_update(law_standard, delta=w, st=state)

        # Verify Mode I behavior matches
        assert np.isclose(t_n_wells, t_n_std, rtol=1e-2), (
            f"Mode I mismatch at w={w*1e3:.2f} mm: Wells={t_n_wells/1e6:.3f} MPa, Std={t_n_std/1e6:.3f} MPa"
        )

        # Shear traction should be zero
        assert abs(t_t_wells) < 1e-6, (
            f"Shear traction should be zero at s=0, got {t_t_wells/1e6:.3f} MPa"
        )

        match = "✓" if np.isclose(t_n_wells, t_n_std, rtol=1e-2) else "✗"
        print(f"  {w*1e3:5.2f}     {t_n_wells/1e6:12.6f}       {t_n_std/1e6:12.6f}       {match}")

    print("  " + "-" * 60)
    print("  ✓ Pure Mode I (s=0) regression test passed")


def test_tangent_matrix_structure():
    """Test that 2x2 tangent matrix has correct structure."""
    from xfem_clean.cohesive_laws import CohesiveLaw, CohesiveState, cohesive_update_mixed

    law = CohesiveLaw(
        Kn=1e13,
        ft=3e6,
        Gf=100.0,
        mode="mixed",
        shear_model="wells",
        k_s0=1e13,
        k_s1=1e11,
    )

    w = 1.0e-3  # 1 mm opening
    s = 0.2e-3  # 0.2 mm slip

    state = CohesiveState(delta_max=0.0, damage=0.0)

    # Compute mixed-mode response
    t, K_mat, st_new = cohesive_update_mixed(law, delta_n=w, delta_t=s, st=state)

    print("\n  Wells 2x2 Tangent Matrix Structure:")
    print("  K_mat =")
    print(f"    [[{K_mat[0,0]:.2e},  {K_mat[0,1]:.2e}],   # [[dt_n/dw, dt_n/ds],")
    print(f"     [{K_mat[1,0]:.2e},  {K_mat[1,1]:.2e}]]   #  [dt_t/dw, dt_t/ds]]")

    # Verify structure
    # dt_n/ds should be zero (no coupling from slip to normal in Wells model)
    assert abs(K_mat[0, 1]) < 1e-6, (
        f"dt_n/ds should be zero in Wells model, got {K_mat[0,1]:.2e}"
    )

    # dt_t/dw should be nonzero (cross-coupling from opening to shear)
    assert abs(K_mat[1, 0]) > 1e6, (
        f"dt_t/dw should be nonzero (cross-coupling), got {K_mat[1,0]:.2e}"
    )

    # dt_n/dw should be positive (normal stiffness)
    assert K_mat[0, 0] > 0, (
        f"dt_n/dw should be positive, got {K_mat[0,0]:.2e}"
    )

    # dt_t/ds should be positive (shear stiffness)
    assert K_mat[1, 1] > 0, (
        f"dt_t/ds should be positive, got {K_mat[1,1]:.2e}"
    )

    print("\n  ✓ Tangent matrix structure verified:")
    print("    - dt_n/ds = 0 (no coupling from slip to normal)")
    print("    - dt_t/dw ≠ 0 (cross-coupling from opening to shear)")
    print("    - dt_n/dw > 0 (positive normal stiffness)")
    print("    - dt_t/ds > 0 (positive shear stiffness)")


if __name__ == "__main__":
    print("=" * 70)
    print("PART C: Wells-Type Shear Traction (Mode II) Tests")
    print("=" * 70)

    print("\n[1/5] Testing exponential shear stiffness degradation...")
    test_wells_shear_stiffness_degradation()

    print("\n[2/5] Testing linear scaling with slip...")
    test_wells_linear_in_slip()

    print("\n[3/5] Testing cross-coupling tangent...")
    test_wells_cross_coupling_tangent()

    print("\n[4/5] Testing Mode I regression (s=0)...")
    test_mode_i_regression_wells()

    print("\n[5/5] Testing 2x2 tangent matrix structure...")
    test_tangent_matrix_structure()

    print("\n" + "=" * 70)
    print("✓ All PART C tests passed!")
    print("=" * 70)
    print("\nImplementation Summary:")
    print("  ✅ Wells-type shear law: k_s(w) = k_s0 * exp(h_s * w)")
    print("  ✅ Shear traction: t_t = k_s(w) * s")
    print("  ✅ Cross-coupling tangent: dt_t/dw = h_s * k_s(w) * s")
    print("  ✅ Mixed-mode 2x2 tangent matrix with correct structure")
    print("  ✅ Backward compatibility with Mode I (s=0)")
