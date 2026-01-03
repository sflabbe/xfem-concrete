"""Unit test for dissipation formula (TASK 5).

This test verifies the trapezoidal dissipation formula directly
without relying on full assembly.
"""

import sys
import os
repo_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(repo_root, 'src'))

import numpy as np


def test_trapezoidal_dissipation_formula():
    """Test that trapezoidal formula gives correct work/dissipation."""
    from xfem_clean.cohesive_laws import CohesiveLaw, CohesiveState, cohesive_update

    # Simple bilinear cohesive law
    law = CohesiveLaw(
        Kn=1e12,  # 1 TPa/m
        ft=3e6,   # 3 MPa
        Gf=100.0, # 100 N/m
        mode="I",
        law="bilinear",
    )

    # Compute critical opening: δc = 2*Gf / ft
    delta_c = 2.0 * law.Gf / law.ft
    print(f"\n  Critical opening δc = {delta_c*1e6:.2f} μm")

    # Test case 1: Elastic loading from 0 to δ_peak
    # δ_peak = ft/Kn
    delta_peak = law.ft / law.Kn
    print(f"  Peak opening δ_peak = {delta_peak*1e6:.2f} μm")

    st0 = CohesiveState()

    # At delta = 0
    T_0, _, st1 = cohesive_update(law, 0.0, st0)
    # At delta = delta_peak
    T_peak, _, st2 = cohesive_update(law, delta_peak, st0)

    print(f"  T(0) = {T_0/1e6:.3f} MPa")
    print(f"  T(δ_peak) = {T_peak/1e6:.3f} MPa")

    # Work done (trapezoidal):
    # W = 0.5 * (T_0 + T_peak) * (delta_peak - 0)
    W_trap = 0.5 * (T_0 + T_peak) * delta_peak

    # Exact work (elastic):
    # W = integral from 0 to δ_peak of Kn*δ dδ = 0.5*Kn*δ_peak^2
    W_exact = 0.5 * law.Kn * delta_peak**2

    print(f"  Work (trapezoidal): {W_trap:.6e} J/m")
    print(f"  Work (exact):       {W_exact:.6e} J/m")
    print(f"  Relative error:     {abs(W_trap - W_exact)/W_exact * 100:.3f}%")

    assert abs(W_trap - W_exact) / W_exact < 1e-10, "Trapezoidal should be exact for linear"
    print(f"  ✓ Elastic work matches exact")

    # Test case 2: Softening
    # Go from delta_peak to 2*delta_peak (into softening)
    delta_1 = delta_peak
    delta_2 = 2.0 * delta_peak

    T_1, _, st3 = cohesive_update(law, delta_1, st0)
    T_2, _, st4 = cohesive_update(law, delta_2, st3)

    print(f"\n  Softening regime:")
    print(f"  δ_1 = {delta_1*1e6:.2f} μm, T_1 = {T_1/1e6:.3f} MPa")
    print(f"  δ_2 = {delta_2*1e6:.2f} μm, T_2 = {T_2/1e6:.3f} MPa")

    # Dissipation (trapezoidal):
    D_trap = 0.5 * (T_1 + T_2) * (delta_2 - delta_1)

    print(f"  Dissipation (trapezoidal): {D_trap:.6e} J/m")

    # In softening, this is truly dissipated (not recoverable)
    assert D_trap > 0, "Softening should produce positive dissipation"
    print(f"  ✓ Softening produces positive dissipation")

    # Test case 3: Complete fracture (integrate to δ_c)
    # Total dissipation should equal Gf
    n_steps = 100
    delta_values = np.linspace(0, delta_c, n_steps + 1)

    st = CohesiveState()
    D_total = 0.0
    T_prev = 0.0

    for i in range(1, len(delta_values)):
        delta = delta_values[i]
        T, _, st = cohesive_update(law, delta, st)

        # Trapezoidal dissipation
        d_delta = delta - delta_values[i-1]
        dD = 0.5 * (T_prev + T) * d_delta
        D_total += dD

        T_prev = T

    print(f"\n  Complete fracture (0 → δc):")
    print(f"  Total dissipation: {D_total:.6e} J/m")
    print(f"  Gf:                {law.Gf:.6e} J/m")
    print(f"  Relative error:    {abs(D_total - law.Gf)/law.Gf * 100:.3f}%")

    # Should match Gf within numerical integration error
    assert abs(D_total - law.Gf) / law.Gf < 0.01, "Total dissipation should equal Gf"
    print(f"  ✓ Total dissipation matches Gf")


if __name__ == "__main__":
    print("=" * 70)
    print("TASK 5: Dissipation Formula Verification")
    print("=" * 70)

    test_trapezoidal_dissipation_formula()

    print("\n" + "=" * 70)
    print("✓ Dissipation formula verification passed!")
    print("=" * 70)
