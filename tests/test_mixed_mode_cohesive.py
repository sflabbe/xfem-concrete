"""Unit tests for mixed-mode cohesive (TASK 3).

This test verifies the mixed-mode cohesive law implementation with
Mode I (normal opening) + Mode II (shear slip) coupling.
"""

import sys
import os
repo_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(repo_root, 'src'))

import numpy as np
import pytest


def test_mixed_mode_pure_mode_I():
    """Test that pure Mode I (no shear) matches Mode I-only results."""
    from xfem_clean.cohesive_laws import CohesiveLaw, CohesiveState, cohesive_update, cohesive_update_mixed

    # Create law with mixed-mode enabled
    law_mixed = CohesiveLaw(
        Kn=1e12,  # 1e12 Pa/m
        ft=3e6,   # 3 MPa
        Gf=100.0, # 100 N/m
        mode="mixed",
        law="bilinear",
    )

    # Create Mode I-only law for comparison
    law_modeI = CohesiveLaw(
        Kn=1e12,
        ft=3e6,
        Gf=100.0,
        mode="I",
        law="bilinear",
    )

    # Test at various openings
    delta_n_values = [1e-6, 5e-6, 1e-5, 3e-5, 6e-5, 1e-4]  # m
    delta_t = 0.0  # No shear

    print("\n  Pure Mode I comparison:")
    print("  δn [μm]   tn_mixed [MPa]  tn_modeI [MPa]  Match")
    print("  " + "-"*55)

    st = CohesiveState()
    for delta_n in delta_n_values:
        # Mixed-mode with δt=0
        t_mixed, K_mixed, st2_mixed = cohesive_update_mixed(law_mixed, delta_n, delta_t, st)
        tn_mixed = t_mixed[0]
        ts_mixed = t_mixed[1]

        # Mode I only
        tn_modeI, _, st2_modeI = cohesive_update(law_modeI, delta_n, st)

        # Should match
        assert np.isclose(tn_mixed, tn_modeI, rtol=1e-6), (
            f"Pure Mode I should match: δn={delta_n*1e6:.2f}μm, "
            f"tn_mixed={tn_mixed/1e6:.3f} MPa, tn_modeI={tn_modeI/1e6:.3f} MPa"
        )
        assert np.isclose(ts_mixed, 0.0, atol=1e-3), f"Shear traction should be zero with no slip"

        match_symbol = "✓" if np.isclose(tn_mixed, tn_modeI, rtol=1e-6) else "✗"
        print(f"  {delta_n*1e6:6.2f}    {tn_mixed/1e6:8.3f}        {tn_modeI/1e6:8.3f}      {match_symbol}")

        # Update state
        st = st2_mixed

    print("  " + "-"*55)
    print("  ✓ Pure Mode I matches Mode I-only law")


def test_mixed_mode_pure_shear():
    """Test pure shear (δn=0, δs≠0) behavior."""
    from xfem_clean.cohesive_laws import CohesiveLaw, CohesiveState, cohesive_update_mixed

    law = CohesiveLaw(
        Kn=1e12,
        ft=3e6,
        Gf=100.0,
        mode="mixed",
        tau_max=3e6,  # Same as ft
        Kt=1e12,      # Same as Kn
        Gf_II=100.0,  # Same as Gf
        law="bilinear",
    )

    delta_n = 0.0
    delta_t_values = [1e-6, 5e-6, 1e-5, 2e-5, 5e-5]

    print("\n  Pure shear (δn=0, δs≠0):")
    print("  δs [μm]   ts [MPa]   K_ss [GPa/m]")
    print("  " + "-"*40)

    st = CohesiveState()
    for delta_t in delta_t_values:
        t, K, st2 = cohesive_update_mixed(law, delta_n, delta_t, st)
        tn = t[0]
        ts = t[1]

        # Normal traction should be zero
        assert np.isclose(tn, 0.0, atol=1e-3), f"Normal traction should be zero with no opening"

        # Shear traction should be non-zero
        assert abs(ts) > 0, f"Shear traction should be non-zero"

        # Tangent
        K_ss = K[1, 1]

        print(f"  {delta_t*1e6:6.2f}    {ts/1e6:6.3f}     {K_ss/1e9:8.2f}")

        st = st2

    print("  " + "-"*40)
    print("  ✓ Pure shear produces shear traction only")


def test_mixed_mode_cross_coupling():
    """Test cross-coupling: opening affects shear stiffness."""
    from xfem_clean.cohesive_laws import CohesiveLaw, CohesiveState, cohesive_update_mixed

    # Wells-type shear model with cross-coupling
    law = CohesiveLaw(
        Kn=1e12,
        ft=3e6,
        Gf=100.0,
        mode="mixed",
        shear_model="wells",
        k_s0=1e12,    # Initial shear stiffness
        k_s1=1e10,    # Degraded shear stiffness (1% of k_s0)
        w1=1e-4,      # Characteristic opening = 0.1 mm
        law="bilinear",
    )

    # Test: as opening increases, shear stiffness should degrade
    delta_t = 1e-5  # Fixed shear slip = 10 μm
    delta_n_values = [0.0, 2e-5, 5e-5, 1e-4, 2e-4]  # Opening from 0 to 0.2 mm

    print("\n  Wells-type cross-coupling (opening → shear degradation):")
    print("  δn [μm]   δs [μm]   ts [MPa]   K_ns [MPa/m]   K_ss [GPa/m]")
    print("  " + "-"*65)

    st = CohesiveState()
    ts_prev = None
    for delta_n in delta_n_values:
        t, K, st2 = cohesive_update_mixed(law, delta_n, delta_t, st)
        tn = t[0]
        ts = t[1]

        # Cross-coupling terms
        K_ns = K[0, 1]  # ∂tn/∂δs (should be small or zero)
        K_sn = K[1, 0]  # ∂ts/∂δn (Wells cross-coupling!)
        K_ss = K[1, 1]  # ∂ts/∂δs

        print(f"  {delta_n*1e6:6.1f}    {delta_t*1e6:5.1f}    {ts/1e6:6.3f}     {K_sn/1e6:9.2f}     {K_ss/1e9:7.2f}")

        # Shear traction should decrease with opening (degradation)
        if ts_prev is not None:
            assert abs(ts) <= abs(ts_prev) + 1e-3, (
                f"Shear traction should degrade with opening: "
                f"ts_prev={ts_prev/1e6:.3f} MPa, ts={ts/1e6:.3f} MPa"
            )

        ts_prev = ts
        st = st2

    print("  " + "-"*65)
    print("  ✓ Opening degrades shear stiffness (Wells model)")


def test_mixed_mode_cyclic_closure():
    """Test cyclic closure: compression penalty + w_max for shear."""
    from xfem_clean.cohesive_laws import CohesiveLaw, CohesiveState, cohesive_update_mixed

    law = CohesiveLaw(
        Kn=1e12,
        ft=3e6,
        Gf=100.0,
        mode="mixed",
        use_cyclic_closure=True,
        kp=5e12,  # Compression penalty (5× Kn)
        shear_model="wells",
        k_s0=1e12,
        law="bilinear",
    )

    # Cyclic loading: open → close → open again
    delta_n_cycle = [0.0, 2e-5, 4e-5, 2e-5, 0.0, -1e-5, 0.0, 3e-5]  # μm
    delta_t = 1e-5  # Constant shear

    print("\n  Cyclic closure test:")
    print("  Step  δn [μm]   tn [MPa]   ts [MPa]   w_max [μm]")
    print("  " + "-"*55)

    st = CohesiveState()
    for step, delta_n in enumerate(delta_n_cycle):
        t, K, st2 = cohesive_update_mixed(law, delta_n, delta_t, st)
        tn = t[0]
        ts = t[1]
        w_max = st2.delta_max

        # Check compression penalty
        if delta_n < 0:
            assert tn < 0, f"Compression should give negative traction: tn={tn/1e6:.3f} MPa"
        elif delta_n > 0:
            assert tn >= 0, f"Tension should give positive traction"

        print(f"  {step:4d}  {delta_n*1e6:7.2f}   {tn/1e6:7.3f}    {ts/1e6:6.3f}     {w_max*1e6:7.2f}")

        st = st2

    print("  " + "-"*55)
    print("  ✓ Cyclic closure with compression penalty works")


def test_mixed_mode_tangent_consistency():
    """Test tangent matrix consistency via finite differences."""
    from xfem_clean.cohesive_laws import CohesiveLaw, CohesiveState, cohesive_update_mixed

    law = CohesiveLaw(
        Kn=1e12,
        ft=3e6,
        Gf=100.0,
        mode="mixed",
        shear_model="wells",
        k_s0=1e12,
        law="bilinear",
    )

    # Test point in softening regime
    delta_n = 3e-5  # 30 μm
    delta_t = 1e-5  # 10 μm
    st = CohesiveState(delta_max=2e-5, damage=0.3)  # Already damaged

    # Analytical tangent
    t0, K_analytical, _ = cohesive_update_mixed(law, delta_n, delta_t, st)

    # Finite difference tangent
    eps = 1e-9  # Small perturbation
    K_fd = np.zeros((2, 2))

    # ∂t/∂δn
    t_plus, _, _ = cohesive_update_mixed(law, delta_n + eps, delta_t, st)
    K_fd[:, 0] = (t_plus - t0) / eps

    # ∂t/∂δs
    t_plus, _, _ = cohesive_update_mixed(law, delta_n, delta_t + eps, st)
    K_fd[:, 1] = (t_plus - t0) / eps

    print("\n  Tangent consistency check (FD vs analytical):")
    print("  Component   Analytical [GPa/m]   FD [GPa/m]      Error [%]")
    print("  " + "-"*65)

    components = [
        ("K_nn", (0, 0)),
        ("K_ns", (0, 1)),
        ("K_sn", (1, 0)),
        ("K_ss", (1, 1)),
    ]

    for name, (i, j) in components:
        K_ana = K_analytical[i, j] / 1e9  # Convert to GPa/m
        K_fd_val = K_fd[i, j] / 1e9

        if abs(K_ana) > 1e-6:
            error = 100 * abs(K_ana - K_fd_val) / abs(K_ana)
        else:
            error = abs(K_ana - K_fd_val)

        match = "✓" if error < 1.0 else "✗"
        print(f"  {name:6s}     {K_ana:15.6f}      {K_fd_val:12.6f}      {error:6.2f}  {match}")

        # Should match within 1% (FD error)
        if abs(K_ana) > 1e-3:  # Only check non-negligible components
            assert error < 1.0, f"{name} tangent mismatch: {error:.2f}% error"

    print("  " + "-"*65)
    print("  ✓ Tangent matrix matches finite differences")


def test_mixed_mode_damage_evolution():
    """Test that damage evolves correctly with mixed-mode loading."""
    from xfem_clean.cohesive_laws import CohesiveLaw, CohesiveState, cohesive_update_mixed

    law = CohesiveLaw(
        Kn=1e12,
        ft=3e6,
        Gf=100.0,
        mode="mixed",
        law="bilinear",
    )

    # Proportional loading: δn and δs increase together
    n_steps = 10
    delta_n_max = 8e-5  # 80 μm
    delta_t_max = 4e-5  # 40 μm

    print("\n  Damage evolution under mixed-mode loading:")
    print("  δn [μm]  δs [μm]  δ_eff [μm]  d [-]    tn [MPa]  ts [MPa]")
    print("  " + "-"*70)

    st = CohesiveState()
    damage_prev = 0.0
    for i in range(n_steps + 1):
        frac = i / n_steps
        delta_n = frac * delta_n_max
        delta_t = frac * delta_t_max

        t, K, st2 = cohesive_update_mixed(law, delta_n, delta_t, st)
        tn = t[0]
        ts = t[1]
        d = st2.damage
        delta_eff = st2.delta_max

        print(f"  {delta_n*1e6:6.2f}   {delta_t*1e6:6.2f}     {delta_eff*1e6:7.2f}     {d:5.3f}    {tn/1e6:6.3f}    {ts/1e6:6.3f}")

        # Damage should be monotonically increasing (or constant)
        assert d >= damage_prev - 1e-9, f"Damage should not decrease: {damage_prev:.4f} → {d:.4f}"

        damage_prev = d
        st = st2

    print("  " + "-"*70)
    print("  ✓ Damage evolves monotonically")


if __name__ == "__main__":
    print("=" * 70)
    print("TASK 3: Mixed-Mode Cohesive Tests")
    print("=" * 70)

    print("\n[1/6] Testing pure Mode I (no shear)...")
    test_mixed_mode_pure_mode_I()

    print("\n[2/6] Testing pure shear (no opening)...")
    test_mixed_mode_pure_shear()

    print("\n[3/6] Testing Wells-type cross-coupling...")
    test_mixed_mode_cross_coupling()

    print("\n[4/6] Testing cyclic closure...")
    test_mixed_mode_cyclic_closure()

    print("\n[5/6] Testing tangent consistency...")
    test_mixed_mode_tangent_consistency()

    print("\n[6/6] Testing damage evolution...")
    test_mixed_mode_damage_evolution()

    print("\n" + "=" * 70)
    print("✓ All TASK 3 mixed-mode cohesive tests passed!")
    print("=" * 70)
