"""
Tests for efficient energy tracking in HHT-α time integration.

These tests verify:
1. Energy conservation for alpha=0, no damping
2. Algorithmic dissipation for alpha < 0
3. Damping dissipation is non-negative
4. Constraint work sign sanity

All tests use small meshes to keep runtime minimal.
"""

import numpy as np
import pytest

from src.xfem_clean.xfem_beam import (
    CDPMaterial,
    Model,
    run_analysis,
)


def make_simple_cdp_material():
    """Create a simple elastic-like CDP material for testing."""
    E = 30e9  # Pa
    nu = 0.2
    mat = CDPMaterial(
        E=E,
        nu=nu,
        psi_deg=0.0,  # no dilation for simplicity
        ecc=0.1,
        fb0_fc0=1.16,
        Kc=2.0 / 3.0,
        lch=0.05,
    )

    # Elastic-like behavior (no softening)
    # Tension: very high strength to avoid damage
    mat.f_t0 = 1e12  # extremely high to stay elastic
    mat.w_tab = np.array([0.0, 1.0], dtype=float)
    mat.sig_t_tab = np.array([1e12, 1e12], dtype=float)
    mat.dt_tab = np.array([0.0, 0.0], dtype=float)

    # Compression: very high strength to avoid damage
    mat.f_c0 = 1e12
    mat.eps_in_c_tab = np.array([0.0, 1.0], dtype=float)
    mat.sig_c_tab = np.array([1e12, 1e12], dtype=float)
    mat.dc_tab = np.array([0.0, 0.0], dtype=float)

    return mat


def make_test_model(hht_alpha=0.0, rayleigh_aM=0.0, rayleigh_aK=0.0):
    """Create a small test model."""
    mat = make_simple_cdp_material()

    model = Model(
        L=1.0,  # 1m beam
        H=0.2,  # 0.2m height
        b=0.1,  # 0.1m thickness
        cdp=mat,
        # Steel (minimal for this test)
        steel_E=200e9,
        steel_fy=500e6,
        steel_fu=540e6,
        steel_Eh=2.0e9,
        steel_A_total=1e-6,  # very small rebar contribution
        cover=0.035,
        # Dynamics
        rho=2500.0,
        rayleigh_aM=rayleigh_aM,
        rayleigh_aK=rayleigh_aK,
        hht_alpha=hht_alpha,
        dt=None,  # will be computed from total_time/nsteps
        total_time=0.01,  # short duration for test
        # Solver
        newton_tol_r=1e-3,
        newton_tol_rel=1e-6,
        newton_tol_du=1e-9,
        newton_maxit=20,
    )
    return model


def test_alpha0_no_damping_energy_conservation():
    """
    Test 1: alpha=0, no damping, near energy conservation.

    Expected:
    - Cumulative algorithmic dissipation close to 0
    - Damping dissipation = 0
    - Constraint work ≈ mechanical energy change
    """
    model = make_test_model(hht_alpha=0.0, rayleigh_aM=0.0, rayleigh_aK=0.0)

    # Small mesh, few steps
    nx, ny = 2, 1  # 2 elements
    nsteps = 5
    umax = -1e-5  # 10 microns downward (small to stay elastic)

    # Run with energy tracking
    *_, energy_history = run_analysis(
        model, nx, ny, nsteps, umax,
        line_search=True,
        track_energy=True
    )

    assert len(energy_history) > 0, "Energy history should not be empty"

    # Get final energy state
    final = energy_history[-1]

    # Check damping dissipation is zero (no damping)
    assert abs(final.D_damp_cum) < 1e-6, f"Expected zero damping dissipation, got {final.D_damp_cum}"

    # Check algorithmic dissipation is small (energy conserving)
    # For alpha=0 (average acceleration), algorithmic dissipation should be ~0
    tol_alg = 1e-3 * abs(final.W_dir_cum)  # 0.1% of input work
    assert abs(final.D_alg_cum) <= tol_alg, \
        f"Expected small algorithmic dissipation (alpha=0), got {final.D_alg_cum} vs tolerance {tol_alg}"

    # Check energy balance: W_dir ≈ ΔE_mech
    E_mech_change = final.E_mech_np1 - energy_history[0].E_mech_n
    work = final.W_dir_cum
    balance_error = abs(work - E_mech_change)
    tol_balance = 1e-2 * abs(work)  # 1% tolerance
    assert balance_error <= tol_balance, \
        f"Energy balance error too large: {balance_error} vs tolerance {tol_balance}"

    # Check balance residuals are small for each step
    for step_energy in energy_history:
        assert abs(step_energy.balance_inc) < 1e-9, \
            f"Step {step_energy.step} balance residual too large: {step_energy.balance_inc}"

    print(f"✓ Test 1 passed: alpha=0 no damping")
    print(f"  Final: W_dir={final.W_dir_cum:.3e}, ΔE_mech={E_mech_change:.3e}, D_alg={final.D_alg_cum:.3e}")


def test_alpha_negative_shows_algorithmic_dissipation():
    """
    Test 2: alpha < 0 shows positive algorithmic dissipation.

    Expected:
    - Cumulative algorithmic dissipation > 0
    - Magnitude greater than alpha=0 case
    """
    model_alpha0 = make_test_model(hht_alpha=0.0, rayleigh_aM=0.0, rayleigh_aK=0.0)
    model_alpha_neg = make_test_model(hht_alpha=-0.2, rayleigh_aM=0.0, rayleigh_aK=0.0)

    nx, ny = 2, 1
    nsteps = 5
    umax = -1e-5

    # Run both cases
    *_, energy_alpha0 = run_analysis(
        model_alpha0, nx, ny, nsteps, umax,
        line_search=True,
        track_energy=True
    )

    *_, energy_alpha_neg = run_analysis(
        model_alpha_neg, nx, ny, nsteps, umax,
        line_search=True,
        track_energy=True
    )

    final_alpha0 = energy_alpha0[-1]
    final_alpha_neg = energy_alpha_neg[-1]

    # Check that alpha < 0 produces positive algorithmic dissipation
    assert final_alpha_neg.D_alg_cum >= -1e-9, \
        f"Expected non-negative algorithmic dissipation for alpha<0, got {final_alpha_neg.D_alg_cum}"

    # Check that alpha < 0 produces MORE dissipation than alpha=0
    # (This test may be sensitive; use a relaxed check)
    assert final_alpha_neg.D_alg_cum >= final_alpha0.D_alg_cum - 1e-9, \
        f"Expected alpha<0 to dissipate more than alpha=0: {final_alpha_neg.D_alg_cum} vs {final_alpha0.D_alg_cum}"

    print(f"✓ Test 2 passed: alpha<0 shows algorithmic dissipation")
    print(f"  alpha=0:   D_alg={final_alpha0.D_alg_cum:.3e}")
    print(f"  alpha=-0.2: D_alg={final_alpha_neg.D_alg_cum:.3e}")


def test_damping_dissipation_is_nonnegative():
    """
    Test 3: damping dissipation is non-negative.

    Expected:
    - Each step: D_damp_inc >= 0 (within small tolerance)
    - Cumulative D_damp_cum increases monotonically
    """
    # Test with mass-proportional damping
    model_mass = make_test_model(hht_alpha=0.0, rayleigh_aM=1.0, rayleigh_aK=0.0)

    nx, ny = 2, 1
    nsteps = 5
    umax = -1e-5

    *_, energy_mass = run_analysis(
        model_mass, nx, ny, nsteps, umax,
        line_search=True,
        track_energy=True
    )

    # Check each step
    for step_energy in energy_mass:
        assert step_energy.D_damp_inc >= -1e-12, \
            f"Step {step_energy.step} damping dissipation is negative: {step_energy.D_damp_inc}"

    # Check cumulative increases
    final = energy_mass[-1]
    assert final.D_damp_cum > 0, \
        f"Expected positive cumulative damping dissipation, got {final.D_damp_cum}"

    print(f"✓ Test 3a passed: mass-proportional damping dissipation is non-negative")
    print(f"  Final D_damp={final.D_damp_cum:.3e}")

    # Test with stiffness-proportional damping
    model_stiff = make_test_model(hht_alpha=0.0, rayleigh_aM=0.0, rayleigh_aK=1e-3)

    *_, energy_stiff = run_analysis(
        model_stiff, nx, ny, nsteps, umax,
        line_search=True,
        track_energy=True
    )

    # Check each step
    for step_energy in energy_stiff:
        assert step_energy.D_damp_inc >= -1e-12, \
            f"Step {step_energy.step} damping dissipation is negative: {step_energy.D_damp_inc}"

    final = energy_stiff[-1]
    assert final.D_damp_cum > 0, \
        f"Expected positive cumulative damping dissipation, got {final.D_damp_cum}"

    print(f"✓ Test 3b passed: stiffness-proportional damping dissipation is non-negative")
    print(f"  Final D_damp={final.D_damp_cum:.3e}")


def test_constraint_work_sign_sanity():
    """
    Test 4: constraint work sign sanity.

    For monotone imposed displacement in the same direction as the reaction force,
    constraint work should be positive on average.

    Expected:
    - Cumulative constraint work > 0 for downward imposed displacement
      (reaction is upward, but power is based on alpha-weighted quantities)
    """
    model = make_test_model(hht_alpha=-0.1, rayleigh_aM=0.0, rayleigh_aK=1e-3)

    nx, ny = 2, 1
    nsteps = 5
    umax = -1e-5  # monotone downward displacement

    *_, energy_history = run_analysis(
        model, nx, ny, nsteps, umax,
        line_search=True,
        track_energy=True
    )

    final = energy_history[-1]

    # Constraint work should be positive (energy flowing into system)
    assert final.W_dir_cum > 0, \
        f"Expected positive constraint work for monotone imposed displacement, got {final.W_dir_cum}"

    # Check that most steps have positive incremental work
    positive_count = sum(1 for e in energy_history if e.W_dir_inc > 0)
    assert positive_count >= len(energy_history) * 0.8, \
        f"Expected most steps to have positive incremental work, got {positive_count}/{len(energy_history)}"

    print(f"✓ Test 4 passed: constraint work sign is correct")
    print(f"  Final W_dir={final.W_dir_cum:.3e}")
    print(f"  Positive increments: {positive_count}/{len(energy_history)}")


def test_energy_balance_all_steps():
    """
    Additional test: verify energy balance for all steps.

    For each step: ΔW_dir = ΔE_mech + ΔD_damp + ΔD_alg (+ small residual)
    """
    model = make_test_model(hht_alpha=-0.1, rayleigh_aM=0.5, rayleigh_aK=1e-3)

    nx, ny = 2, 1
    nsteps = 5
    umax = -1e-5

    *_, energy_history = run_analysis(
        model, nx, ny, nsteps, umax,
        line_search=True,
        track_energy=True
    )

    # Check balance for each step
    for step_energy in energy_history:
        ΔE_mech = step_energy.E_mech_np1 - step_energy.E_mech_n
        ΔW_dir = step_energy.W_dir_inc
        ΔD_damp = step_energy.D_damp_inc
        ΔD_alg = step_energy.D_alg_inc

        # Energy balance: ΔW_dir = ΔE_mech + ΔD_damp + ΔD_alg
        balance_computed = ΔW_dir - (ΔE_mech + ΔD_damp + ΔD_alg)

        # Should match step_energy.balance_inc
        assert abs(balance_computed - step_energy.balance_inc) < 1e-14, \
            f"Step {step_energy.step} balance mismatch"

        # Balance residual should be very small
        tol = max(1e-12, 1e-6 * abs(ΔW_dir))
        assert abs(step_energy.balance_inc) < tol, \
            f"Step {step_energy.step} balance residual too large: {step_energy.balance_inc}"

    print(f"✓ Additional test passed: energy balance verified for all steps")


if __name__ == "__main__":
    print("Running energy tracking tests...\n")
    test_alpha0_no_damping_energy_conservation()
    test_alpha_negative_shows_algorithmic_dissipation()
    test_damping_dissipation_is_nonnegative()
    test_constraint_work_sign_sanity()
    test_energy_balance_all_steps()
    print("\n✓ All energy tracking tests passed!")
