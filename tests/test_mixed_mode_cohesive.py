"""Unit test for P2: Mixed-mode cohesive law (PARTIAL IMPLEMENTATION).

This test verifies that:
1. Mixed-mode parameters can be set in CohesiveLaw
2. Default values are assigned correctly
3. Backward compatibility maintained (mode="I" is default)

NOTE: Full mixed-mode implementation (shear traction update in assembly)
is not yet complete. This test validates the data structure changes only.
"""

import pytest
import numpy as np


def test_mode_i_backward_compatibility():
    """Test that Mode I (default) behavior is unchanged."""
    from xfem_clean.cohesive_laws import CohesiveLaw

    # Create law without specifying mode (should default to "I")
    law = CohesiveLaw(Kn=1e13, ft=3e6, Gf=100.0)

    assert law.mode == "I", "Default mode should be 'I' (Mode I only)"
    assert law.tau_max == 0.0, "tau_max should be 0 for Mode I"
    assert law.Kt == 0.0, "Kt should be 0 for Mode I"
    assert law.Gf_II == 0.0, "Gf_II should be 0 for Mode I"

    print("✓ Mode I backward compatibility: default mode='I'")


def test_mixed_mode_parameter_defaults():
    """Test that mixed-mode parameters get sensible defaults."""
    from xfem_clean.cohesive_laws import CohesiveLaw

    # Create mixed-mode law with defaults
    law = CohesiveLaw(
        Kn=1e13,
        ft=3e6,
        Gf=100.0,
        mode="mixed"  # Enable mixed mode
    )

    assert law.mode == "mixed", "Mode should be 'mixed'"
    assert law.tau_max == law.ft, "tau_max should default to ft"
    assert law.Kt == law.Kn, "Kt should default to Kn"
    assert law.Gf_II == law.Gf, "Gf_II should default to Gf (Mode I)"

    print("✓ Mixed-mode defaults:")
    print(f"  tau_max = {law.tau_max/1e6:.1f} MPa (= ft)")
    print(f"  Kt = {law.Kt:.2e} Pa/m (= Kn)")
    print(f"  Gf_II = {law.Gf:.1f} J/m² (= Gf)")


def test_mixed_mode_custom_parameters():
    """Test that custom mixed-mode parameters are respected."""
    from xfem_clean.cohesive_laws import CohesiveLaw

    # Create mixed-mode law with custom parameters
    law = CohesiveLaw(
        Kn=1e13,
        ft=3e6,
        Gf=100.0,
        mode="mixed",
        tau_max=2e6,  # Custom shear strength
        Kt=5e12,  # Custom shear stiffness
        Gf_II=50.0  # Custom Mode II fracture energy
    )

    assert law.mode == "mixed"
    assert law.tau_max == 2e6, "Custom tau_max should be respected"
    assert law.Kt == 5e12, "Custom Kt should be respected"
    assert law.Gf_II == 50.0, "Custom Gf_II should be respected"

    print("✓ Mixed-mode custom parameters:")
    print(f"  tau_max = {law.tau_max/1e6:.1f} MPa")
    print(f"  Kt = {law.Kt:.2e} Pa/m")
    print(f"  Gf_II = {law.Gf_II:.1f} J/m²")


def test_mixed_mode_assembly_stub():
    """Test that mixed-mode law can be instantiated (assembly not yet implemented)."""
    from xfem_clean.cohesive_laws import CohesiveLaw, CohesiveState, cohesive_update

    law = CohesiveLaw(Kn=1e13, ft=3e6, Gf=100.0, mode="mixed")
    state = CohesiveState(delta_max=0.0, damage=0.0)

    # NOTE: cohesive_update currently only handles normal (Mode I) traction
    # When full mixed-mode is implemented, this function will accept
    # both normal and tangential crack opening displacements

    # For now, verify Mode I still works
    delta_n = 1e-6  # Normal opening
    T, K, state_new = cohesive_update(law, delta_n, state)

    assert np.isfinite(T), "Traction should be finite"
    assert np.isfinite(K), "Stiffness should be finite"
    assert state_new.delta_max >= state.delta_max, "History should be monotonic"

    print("✓ Mixed-mode cohesive_update (Mode I component only)")
    print(f"  T_n = {T/1e6:.2f} MPa")
    print(f"  NOTE: Full mixed-mode assembly not yet implemented")


if __name__ == "__main__":
    print("=" * 60)
    print("P2 Mixed-Mode Cohesive Law Tests (PARTIAL)")
    print("=" * 60)

    print("\n[1/4] Testing Mode I backward compatibility...")
    test_mode_i_backward_compatibility()

    print("\n[2/4] Testing mixed-mode parameter defaults...")
    test_mixed_mode_parameter_defaults()

    print("\n[3/4] Testing custom mixed-mode parameters...")
    test_mixed_mode_custom_parameters()

    print("\n[4/4] Testing mixed-mode assembly stub...")
    test_mixed_mode_assembly_stub()

    print("\n" + "=" * 60)
    print("✓ P2 tests passed (data structure only)")
    print("=" * 60)
    print("\nIMPLEMENTATION STATUS:")
    print("  ✅ Mixed-mode parameters in CohesiveLaw dataclass")
    print("  ✅ Default value assignment")
    print("  ✅ Backward compatibility with mode='I'")
    print("  ❌ Mixed-mode traction update (normal + shear)")
    print("  ❌ Mixed-mode assembly in XFEM kernels")
    print("  ❌ Shear crack opening displacement tracking")
    print("\nTO COMPLETE P2:")
    print("  1. Extend CohesiveState to track shear displacement")
    print("  2. Implement cohesive_update_mixed() with T_n and T_t")
    print("  3. Modify assembly to compute both δ_n and δ_t")
    print("  4. Add consistent tangent stiffness for mixed mode")
