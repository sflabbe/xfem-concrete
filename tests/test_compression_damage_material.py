"""Unit test for P1: Compression damage model integration into material factory.

This test verifies that:
1. Compression damage model can be instantiated via material factory
2. Parabolic stress-strain curve matches thesis Eq. (3.46)
3. Damage evolution is monotonic (no softening)
4. Equivalent compressive strain calculation is correct (minimum principal strain)
"""

import pytest
import numpy as np


def test_compression_damage_factory():
    """Test that compression damage model is selectable in material factory."""
    from xfem_clean.xfem.model import XFEMModel
    from xfem_clean.xfem.material_factory import make_bulk_material
    from xfem_clean.compression_damage import ConcreteCompressionModel

    # Create model with compression damage
    model = XFEMModel(
        L=1.0, H=0.5, b=1.0,
        E=30e9, nu=0.2,
        ft=3e6, fc=30e6,
        Gf=100.0,
        steel_A_total=0.0,
        steel_E=200e9,
        bulk_material="compression-damage",  # P1: Select compression damage
    )

    # Instantiate material
    mat = make_bulk_material(model)

    # Assert it's the right type
    assert isinstance(mat, ConcreteCompressionModel), "Should return ConcreteCompressionModel"
    assert mat.f_c == 30e6, "Compressive strength should match"
    assert mat.E_0 == 30e9, "Initial modulus should match"
    assert mat.eps_c1 > 0.0, "Peak strain should be positive"

    print("✓ Compression damage model instantiated via factory")
    print(f"  f_c = {mat.f_c/1e6:.1f} MPa")
    print(f"  E_0 = {mat.E_0/1e9:.1f} GPa")
    print(f"  eps_c1 = {mat.eps_c1:.4f}")


def test_parabolic_stress_strain_curve():
    """Test parabolic stress-strain relation per thesis Eq. (3.46)."""
    from xfem_clean.compression_damage import ConcreteCompressionModel

    f_c = 30e6  # 30 MPa
    eps_c1 = 0.002  # Peak strain
    E_0 = 30e9  # 30 GPa

    model = ConcreteCompressionModel(f_c=f_c, eps_c1=eps_c1, E_0=E_0)

    # Test parabolic branch (0 ≤ ε ≤ εc1)
    eps_test = np.linspace(0.0, eps_c1, 10)
    for eps in eps_test:
        sigma, E_t = model.sigma_epsilon_curve(eps)

        # Eq. (3.46): σ = fc * [2*(ε/εc1) - (ε/εc1)²]
        ratio = eps / eps_c1
        sigma_expected = f_c * (2.0 * ratio - ratio**2)

        assert abs(sigma - sigma_expected) < 1e-3, f"Stress should match parabola at ε={eps}"

    print("✓ Parabolic stress-strain curve verified (0 ≤ ε ≤ εc1)")

    # Test plateau (ε > εc1)
    eps_plateau = np.linspace(eps_c1, 2 * eps_c1, 5)
    for eps in eps_plateau:
        sigma, E_t = model.sigma_epsilon_curve(eps)

        assert abs(sigma - f_c) < 1e-3, f"Stress should be fc={f_c} on plateau at ε={eps}"
        assert abs(E_t) < 1e-6, f"Tangent modulus should be zero on plateau at ε={eps}"

    print("✓ Constant plateau verified (ε > εc1, no softening)")


def test_damage_evolution_monotonic():
    """Test that damage increases monotonically (no healing)."""
    from xfem_clean.compression_damage import ConcreteCompressionModel

    model = ConcreteCompressionModel(f_c=30e6, eps_c1=0.002, E_0=30e9)

    eps_vals = np.linspace(0.0, 0.01, 50)
    d_prev = 0.0

    for eps in eps_vals:
        d_c = model.compute_damage(eps)

        assert d_c >= d_prev, f"Damage should be monotonic at ε={eps}"
        assert 0.0 <= d_c <= 1.0, f"Damage should be in [0,1] at ε={eps}"

        d_prev = d_c

    print("✓ Damage evolution is monotonic (no healing)")
    print(f"  Final damage at ε={eps_vals[-1]:.4f}: d_c={d_c:.3f}")


def test_equivalent_compressive_strain():
    """Test equivalent compressive strain = -min(ε1, ε2) for plane stress."""
    from xfem_clean.compression_damage import compute_equivalent_compressive_strain

    # Test 1: Uniaxial compression in x-direction
    # εxx = -0.001, εyy = +ν*εxx (expansion), γxy = 0
    eps_ux = np.array([-0.001, 0.0002, 0.0])  # Approximate ν=0.2
    eps_eq_c = compute_equivalent_compressive_strain(eps_ux)

    # Minimum principal ≈ εxx = -0.001 → eps_eq_c ≈ 0.001
    assert abs(eps_eq_c - 0.001) < 1e-4, "Uniaxial compression: eps_eq_c should equal |εxx|"

    print("✓ Equivalent compressive strain: uniaxial compression")

    # Test 2: Biaxial compression
    eps_bx = np.array([-0.002, -0.001, 0.0])
    eps_eq_c = compute_equivalent_compressive_strain(eps_bx)

    # Minimum principal = -0.002 → eps_eq_c = 0.002
    assert abs(eps_eq_c - 0.002) < 1e-4, "Biaxial compression: eps_eq_c = max compressive principal"

    print("✓ Equivalent compressive strain: biaxial compression")

    # Test 3: Tension (no compressive strain)
    eps_tens = np.array([0.001, 0.0005, 0.0])
    eps_eq_c = compute_equivalent_compressive_strain(eps_tens)

    assert eps_eq_c == 0.0, "Tension: eps_eq_c should be zero"

    print("✓ Equivalent compressive strain: tension → zero")


def test_stress_update_with_compression_damage():
    """Test full stress update with compression damage."""
    from xfem_clean.compression_damage import (
        ConcreteCompressionModel,
        stress_update_compression_damage,
    )

    model = ConcreteCompressionModel(f_c=30e6, eps_c1=0.002, E_0=30e9)

    # Plane stress elastic stiffness (isotropic)
    nu = 0.2
    E = 30e9
    D_0 = (E / (1.0 - nu**2)) * np.array([
        [1.0,  nu,   0.0],
        [nu,   1.0,  0.0],
        [0.0,  0.0,  (1.0 - nu) / 2.0]
    ])

    # Uniaxial compression strain
    eps = np.array([-0.001, 0.0, 0.0])

    sigma, D_sec, d_c = stress_update_compression_damage(eps, model, D_0)

    # Check stress
    assert sigma[0] < 0.0, "Compressive stress should be negative"
    assert abs(sigma[1]) < abs(sigma[0]) * 0.3, "Transverse stress should be small"

    # Check damage
    assert d_c > 0.0, "Damage should be positive at ε=0.001"
    assert d_c < 0.5, "Damage should be moderate at small strain"

    # Check degraded stiffness (allow small numerical tolerance)
    # D_sec = (1 - d_c) * D_0, so D_sec should be smaller
    ratio = np.linalg.norm(D_sec) / np.linalg.norm(D_0)
    assert ratio < 1.0, f"Secant stiffness should be degraded, but ratio={ratio}"
    assert ratio > 0.5, f"Secant stiffness ratio should be reasonable, but ratio={ratio}"

    print("✓ Stress update with compression damage")
    print(f"  σxx = {sigma[0]/1e6:.2f} MPa")
    print(f"  d_c = {d_c:.3f}")
    print(f"  D_sec/D_0 = {ratio:.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("P1 Compression Damage Material Factory Tests")
    print("=" * 60)

    print("\n[1/6] Testing material factory integration...")
    test_compression_damage_factory()

    print("\n[2/6] Testing parabolic stress-strain curve...")
    test_parabolic_stress_strain_curve()

    print("\n[3/6] Testing damage evolution...")
    test_damage_evolution_monotonic()

    print("\n[4/6] Testing equivalent compressive strain...")
    test_equivalent_compressive_strain()

    print("\n[5/6] Testing stress update with damage...")
    test_stress_update_with_compression_damage()

    print("\n" + "=" * 60)
    print("✓ P1 tests passed")
    print("=" * 60)
    print("\nCompression damage model is now selectable via:")
    print("  model.bulk_material = 'compression-damage'")
