"""Unit test for PART B: Bond-slip yielding reduction factor Ωy(eps_s).

This test verifies that:
1. Ωy = 1.0 when steel is elastic (eps_s < eps_y)
2. Ωy < 1.0 when steel has yielded (eps_s > eps_y)
3. Ωy reduces bond stress and tangent consistently
4. The reduction follows the thesis formula: Ωy = 1 - 0.85*(1 - exp(-5*xi))

Thesis reference: Orlando/Gutiérrez Eq. 3.57-3.58
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np


def test_yielding_reduction_elastic():
    """Test that Ωy = 1.0 when steel is elastic (eps_s < eps_y)."""
    from xfem_clean.bond_slip import BondSlipModelCode2010

    # Create bond law with yielding reduction ENABLED
    bond_law = BondSlipModelCode2010(
        f_cm=30e6,  # 30 MPa
        d_bar=0.012,  # 12 mm
        condition="good",
        f_y=500e6,  # 500 MPa
        E_s=200e9,  # 200 GPa
        enable_yielding_reduction=True,  # ENABLE
    )

    # Elastic steel strain (below yield)
    eps_y = bond_law.f_y / bond_law.E_s
    eps_s_elastic = 0.5 * eps_y  # Half of yield strain

    # Compute bond stress at some slip
    s = 0.5e-3  # 0.5 mm slip
    s_max_hist = 0.5e-3

    # With eps_s = 0 (elastic)
    tau_elastic, dtau_elastic = bond_law.tau_and_tangent(
        s, s_max_hist, eps_s=0.0, omega_crack=1.0
    )

    # With eps_s < eps_y (still elastic)
    tau_subyelding, dtau_subyelding = bond_law.tau_and_tangent(
        s, s_max_hist, eps_s=eps_s_elastic, omega_crack=1.0
    )

    # Should be the same (no reduction for elastic steel)
    assert np.isclose(tau_elastic, tau_subyelding), (
        f"Elastic steel should have Ωy=1: tau_elastic={tau_elastic:.2e}, "
        f"tau_subyelding={tau_subyelding:.2e}"
    )
    assert np.isclose(dtau_elastic, dtau_subyelding), (
        f"Elastic steel should have same tangent: dtau_elastic={dtau_elastic:.2e}, "
        f"dtau_subyelding={dtau_subyelding:.2e}"
    )

    print(f"✓ Elastic steel: Ωy = 1.0, tau = {tau_elastic/1e6:.2f} MPa")


def test_yielding_reduction_yielded():
    """Test that Ωy < 1.0 when steel has yielded (eps_s > eps_y)."""
    from xfem_clean.bond_slip import BondSlipModelCode2010

    # Create bond law with yielding reduction ENABLED
    bond_law = BondSlipModelCode2010(
        f_cm=30e6,  # 30 MPa
        d_bar=0.012,  # 12 mm
        condition="good",
        f_y=500e6,  # 500 MPa
        E_s=200e9,  # 200 GPa
        enable_yielding_reduction=True,  # ENABLE
    )

    # Steel strain beyond yield
    eps_y = bond_law.f_y / bond_law.E_s
    eps_s_yielded = 2.0 * eps_y  # Double the yield strain

    # Compute bond stress at some slip
    s = 0.5e-3  # 0.5 mm slip
    s_max_hist = 0.5e-3

    # Without yielding (eps_s = 0, elastic)
    tau_elastic, dtau_elastic = bond_law.tau_and_tangent(
        s, s_max_hist, eps_s=0.0, omega_crack=1.0
    )

    # With yielding (eps_s > eps_y)
    tau_yielded, dtau_yielded = bond_law.tau_and_tangent(
        s, s_max_hist, eps_s=eps_s_yielded, omega_crack=1.0
    )

    # Compute expected Ωy manually (Eq. 3.58)
    # THESIS PARITY: Compute εu using the same formula as the code
    # εu = εy + (fu - fy) / H (bilinear hardening)
    f_u = bond_law.f_u  # Defaults to 1.5 * f_y
    H = bond_law.H  # Defaults to 0.01 * E_s
    if H > 0.0 and f_u > bond_law.f_y:
        eps_u = eps_y + (f_u - bond_law.f_y) / H
    else:
        eps_u = f_u / bond_law.E_s
    xi = (eps_s_yielded - eps_y) / (eps_u - eps_y)
    omega_y_expected = 1.0 - 0.85 * (1.0 - np.exp(-5.0 * xi))

    # Yielded steel should have reduced bond stress
    assert tau_yielded < tau_elastic, (
        f"Yielded steel should reduce bond: tau_elastic={tau_elastic/1e6:.2f} MPa, "
        f"tau_yielded={tau_yielded/1e6:.2f} MPa"
    )
    assert dtau_yielded < dtau_elastic, (
        f"Yielded steel should reduce tangent: dtau_elastic={dtau_elastic:.2e}, "
        f"dtau_yielded={dtau_yielded:.2e}"
    )

    # Check that reduction matches expected formula
    omega_y_actual = tau_yielded / tau_elastic
    assert np.isclose(omega_y_actual, omega_y_expected, atol=1e-6), (
        f"Ωy mismatch: expected={omega_y_expected:.4f}, actual={omega_y_actual:.4f}"
    )

    print(f"✓ Yielded steel (eps_s={eps_s_yielded:.4e}):")
    print(f"  Ωy = {omega_y_expected:.4f}")
    print(f"  tau_elastic = {tau_elastic/1e6:.2f} MPa")
    print(f"  tau_yielded = {tau_yielded/1e6:.2f} MPa (reduced by {(1-omega_y_actual)*100:.1f}%)")


def test_yielding_reduction_disabled():
    """Test that Ωy = 1.0 when yielding reduction is DISABLED."""
    from xfem_clean.bond_slip import BondSlipModelCode2010

    # Create bond law with yielding reduction DISABLED
    bond_law = BondSlipModelCode2010(
        f_cm=30e6,  # 30 MPa
        d_bar=0.012,  # 12 mm
        condition="good",
        f_y=500e6,  # 500 MPa
        E_s=200e9,  # 200 GPa
        enable_yielding_reduction=False,  # DISABLED
    )

    # Steel strain beyond yield
    eps_y = bond_law.f_y / bond_law.E_s
    eps_s_yielded = 2.0 * eps_y  # Double the yield strain

    # Compute bond stress at some slip
    s = 0.5e-3  # 0.5 mm slip
    s_max_hist = 0.5e-3

    # Without yielding (eps_s = 0)
    tau_0, dtau_0 = bond_law.tau_and_tangent(
        s, s_max_hist, eps_s=0.0, omega_crack=1.0
    )

    # With yielding (eps_s > eps_y) but DISABLED
    tau_yielded, dtau_yielded = bond_law.tau_and_tangent(
        s, s_max_hist, eps_s=eps_s_yielded, omega_crack=1.0
    )

    # Should be the same (reduction disabled)
    assert np.isclose(tau_0, tau_yielded), (
        f"Disabled yielding reduction should not change tau: tau_0={tau_0/1e6:.2f} MPa, "
        f"tau_yielded={tau_yielded/1e6:.2f} MPa"
    )
    assert np.isclose(dtau_0, dtau_yielded), (
        f"Disabled yielding reduction should not change tangent"
    )

    print(f"✓ Yielding reduction DISABLED: Ωy = 1.0 (no reduction)")


def test_yielding_reduction_formula():
    """Test that Ωy follows the thesis formula across full strain range."""
    from xfem_clean.bond_slip import BondSlipModelCode2010

    bond_law = BondSlipModelCode2010(
        f_cm=30e6,
        d_bar=0.012,
        condition="good",
        f_y=500e6,
        E_s=200e9,
        enable_yielding_reduction=True,
    )

    eps_y = bond_law.f_y / bond_law.E_s

    # THESIS PARITY: Compute εu using the same formula as the code
    # εu = εy + (fu - fy) / H (bilinear hardening)
    f_u = bond_law.f_u  # Defaults to 1.5 * f_y
    H = bond_law.H  # Defaults to 0.01 * E_s
    if H > 0.0 and f_u > bond_law.f_y:
        eps_u = eps_y + (f_u - bond_law.f_y) / H
    else:
        eps_u = f_u / bond_law.E_s

    # Test at various strain levels
    eps_s_values = np.array([
        0.0,  # No strain
        0.5 * eps_y,  # Elastic
        1.0 * eps_y,  # At yield
        1.5 * eps_y,  # Slight yielding
        3.0 * eps_y,  # Moderate yielding
        5.0 * eps_y,  # Significant yielding
        8.0 * eps_y,  # Near ultimate
    ])

    s = 0.5e-3
    s_max_hist = 0.5e-3

    print("\n  Strain Range Test:")
    print("  eps_s / eps_y    Ωy        tau (MPa)")
    print("  " + "-" * 40)

    for eps_s in eps_s_values:
        tau, _ = bond_law.tau_and_tangent(s, s_max_hist, eps_s=eps_s, omega_crack=1.0)

        # Compute expected Ωy
        if eps_s <= eps_y:
            omega_y_expected = 1.0
        else:
            xi = (eps_s - eps_y) / (eps_u - eps_y)
            omega_y_expected = 1.0 - 0.85 * (1.0 - np.exp(-5.0 * xi))
            omega_y_expected = max(0.15, min(1.0, omega_y_expected))

        # Get reference tau (eps_s = 0)
        tau_ref, _ = bond_law.tau_and_tangent(s, s_max_hist, eps_s=0.0, omega_crack=1.0)
        omega_y_actual = tau / tau_ref

        # Check consistency
        assert np.isclose(omega_y_actual, omega_y_expected, atol=1e-4), (
            f"Ωy mismatch at eps_s={eps_s:.4e}: expected={omega_y_expected:.4f}, actual={omega_y_actual:.4f}"
        )

        print(f"  {eps_s/eps_y:6.2f}          {omega_y_expected:.4f}    {tau/1e6:6.2f}")

    print("  " + "-" * 40)
    print("  ✓ Ωy formula verified across full strain range")


if __name__ == "__main__":
    print("=" * 60)
    print("PART B: Bond-Slip Yielding Reduction Tests")
    print("=" * 60)

    print("\n[1/4] Testing elastic steel (Ωy = 1.0)...")
    test_yielding_reduction_elastic()

    print("\n[2/4] Testing yielded steel (Ωy < 1.0)...")
    test_yielding_reduction_yielded()

    print("\n[3/4] Testing disabled yielding reduction...")
    test_yielding_reduction_disabled()

    print("\n[4/4] Testing Ωy formula across strain range...")
    test_yielding_reduction_formula()

    print("\n" + "=" * 60)
    print("✓ All PART B tests passed!")
    print("=" * 60)
