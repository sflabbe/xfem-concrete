"""Tests for Python vs Numba parity in mixed-mode cohesive law.

This test suite verifies that the Numba implementation of the mixed-mode
cohesive law produces identical results to the Python implementation.

Test Coverage:
1. Elastic regime (no damage)
2. Softening regime (damage evolution)
3. Compression penalty (cyclic closure)
4. Wells shear model (opening-dependent degradation)
5. Cross-coupling in tangent matrix (Wells model)
6. Unloading/reloading behavior
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np


def test_mixed_mode_elastic():
    """Test Python vs Numba parity for elastic regime (no damage)."""
    from xfem_clean.cohesive_laws import CohesiveLaw, CohesiveState, cohesive_update_mixed
    from xfem_clean.numba.kernels_cohesive import (
        pack_cohesive_law_params,
        cohesive_update_mixed_values_numba,
    )

    # Create mixed-mode law
    law = CohesiveLaw(
        Kn=1e13,
        ft=3e6,
        Gf=100.0,
        mode="mixed",
        Kt=5e12,
        tau_max=2e6,
    )

    # Pack params
    params = pack_cohesive_law_params(law)

    # Initial state
    state = CohesiveState(delta_max=0.0, damage=0.0)

    # Small elastic openings
    delta_n = 1e-7  # Below elastic limit
    delta_t = 2e-7

    # Python path
    t_vec_py, K_mat_py, state_py = cohesive_update_mixed(law, delta_n, delta_t, state)
    t_n_py = t_vec_py[0]
    t_t_py = t_vec_py[1]

    # Numba path
    t_n_nb, t_t_nb, dtn_ddn_nb, dtn_ddt_nb, dtt_ddn_nb, dtt_ddt_nb, dm_nb, dmg_nb = \
        cohesive_update_mixed_values_numba(delta_n, delta_t, 0.0, 0.0, params)

    # Compare tractions
    assert np.isclose(t_n_py, t_n_nb, rtol=1e-10), \
        f"Normal traction mismatch: Python={t_n_py:.6e}, Numba={t_n_nb:.6e}"
    assert np.isclose(t_t_py, t_t_nb, rtol=1e-10), \
        f"Tangential traction mismatch: Python={t_t_py:.6e}, Numba={t_t_nb:.6e}"

    # Compare tangent matrix
    assert np.isclose(K_mat_py[0, 0], dtn_ddn_nb, rtol=1e-10), \
        f"Tangent K[0,0] mismatch: Python={K_mat_py[0,0]:.6e}, Numba={dtn_ddn_nb:.6e}"
    assert np.isclose(K_mat_py[0, 1], dtn_ddt_nb, rtol=1e-10), \
        f"Tangent K[0,1] mismatch: Python={K_mat_py[0,1]:.6e}, Numba={dtn_ddt_nb:.6e}"
    assert np.isclose(K_mat_py[1, 0], dtt_ddn_nb, rtol=1e-10), \
        f"Tangent K[1,0] mismatch: Python={K_mat_py[1,0]:.6e}, Numba={dtt_ddn_nb:.6e}"
    assert np.isclose(K_mat_py[1, 1], dtt_ddt_nb, rtol=1e-10), \
        f"Tangent K[1,1] mismatch: Python={K_mat_py[1,1]:.6e}, Numba={dtt_ddt_nb:.6e}"

    # Compare state
    assert np.isclose(state_py.delta_max, dm_nb, rtol=1e-10), \
        f"delta_max mismatch: Python={state_py.delta_max:.6e}, Numba={dm_nb:.6e}"
    assert np.isclose(state_py.damage, dmg_nb, rtol=1e-10), \
        f"damage mismatch: Python={state_py.damage:.6e}, Numba={dmg_nb:.6e}"

    print(f"✓ Mixed-mode elastic: t_n={t_n_py:.2e} Pa, t_t={t_t_py:.2e} Pa (Python ≈ Numba)")


def test_mixed_mode_softening():
    """Test Python vs Numba parity for softening regime."""
    from xfem_clean.cohesive_laws import CohesiveLaw, CohesiveState, cohesive_update_mixed
    from xfem_clean.numba.kernels_cohesive import (
        pack_cohesive_law_params,
        cohesive_update_mixed_values_numba,
    )

    # Create mixed-mode law
    law = CohesiveLaw(
        Kn=1e13,
        ft=3e6,
        Gf=100.0,
        mode="mixed",
        Kt=5e12,
        tau_max=2e6,
    )

    # Pack params
    params = pack_cohesive_law_params(law)

    # State with some damage history
    delta_max_old = 5e-7  # Above elastic limit
    damage_old = 0.3
    state = CohesiveState(delta_max=delta_max_old, damage=damage_old)

    # Loading in softening regime
    delta_n = 6e-7
    delta_t = 3e-7

    # Python path
    t_vec_py, K_mat_py, state_py = cohesive_update_mixed(law, delta_n, delta_t, state)
    t_n_py = t_vec_py[0]
    t_t_py = t_vec_py[1]

    # Numba path
    t_n_nb, t_t_nb, dtn_ddn_nb, dtn_ddt_nb, dtt_ddn_nb, dtt_ddt_nb, dm_nb, dmg_nb = \
        cohesive_update_mixed_values_numba(delta_n, delta_t, delta_max_old, damage_old, params)

    # Compare tractions
    assert np.isclose(t_n_py, t_n_nb, rtol=1e-8), \
        f"Normal traction mismatch: Python={t_n_py:.6e}, Numba={t_n_nb:.6e}"
    assert np.isclose(t_t_py, t_t_nb, rtol=1e-8), \
        f"Tangential traction mismatch: Python={t_t_py:.6e}, Numba={t_t_nb:.6e}"

    # Compare tangent matrix
    assert np.isclose(K_mat_py[0, 0], dtn_ddn_nb, rtol=1e-8), \
        f"Tangent K[0,0] mismatch: Python={K_mat_py[0,0]:.6e}, Numba={dtn_ddn_nb:.6e}"
    assert np.isclose(K_mat_py[1, 1], dtt_ddt_nb, rtol=1e-8), \
        f"Tangent K[1,1] mismatch: Python={K_mat_py[1,1]:.6e}, Numba={dtt_ddt_nb:.6e}"

    # Compare state
    assert np.isclose(state_py.delta_max, dm_nb, rtol=1e-10), \
        f"delta_max mismatch: Python={state_py.delta_max:.6e}, Numba={dm_nb:.6e}"
    assert np.isclose(state_py.damage, dmg_nb, rtol=1e-8), \
        f"damage mismatch: Python={state_py.damage:.6e}, Numba={dmg_nb:.6e}"

    print(f"✓ Mixed-mode softening: t_n={t_n_py:.2e} Pa, damage={dmg_nb:.3f} (Python ≈ Numba)")


def test_mixed_mode_compression_penalty():
    """Test Python vs Numba parity for compression penalty."""
    from xfem_clean.cohesive_laws import CohesiveLaw, CohesiveState, cohesive_update_mixed
    from xfem_clean.numba.kernels_cohesive import (
        pack_cohesive_law_params,
        cohesive_update_mixed_values_numba,
    )

    # Create mixed-mode law with cyclic closure
    kp = 5e12
    law = CohesiveLaw(
        Kn=1e13,
        ft=3e6,
        Gf=100.0,
        mode="mixed",
        kp=kp,
        use_cyclic_closure=True,
    )

    # Pack params
    params = pack_cohesive_law_params(law)

    # Compression
    state = CohesiveState(delta_max=0.0, damage=0.0)
    delta_n = -1e-6  # Negative = compression
    delta_t = 0.0

    # Python path
    t_vec_py, K_mat_py, state_py = cohesive_update_mixed(law, delta_n, delta_t, state)
    t_n_py = t_vec_py[0]
    t_t_py = t_vec_py[1]

    # Numba path
    t_n_nb, t_t_nb, dtn_ddn_nb, dtn_ddt_nb, dtt_ddn_nb, dtt_ddt_nb, dm_nb, dmg_nb = \
        cohesive_update_mixed_values_numba(delta_n, delta_t, 0.0, 0.0, params)

    # Compare tractions
    assert np.isclose(t_n_py, t_n_nb, atol=1e-3), \
        f"Compression traction mismatch: Python={t_n_py:.2e}, Numba={t_n_nb:.2e}"

    # Should be t_n = kp * delta_n
    t_n_expected = kp * delta_n
    assert np.isclose(t_n_nb, t_n_expected, atol=1e-3), \
        f"Compression penalty: expected={t_n_expected:.2e}, got={t_n_nb:.2e}"

    # Tangent should be kp
    assert np.isclose(K_mat_py[0, 0], kp, atol=1e-3), \
        f"Compression tangent mismatch: Python={K_mat_py[0,0]:.2e}, expected={kp:.2e}"
    assert np.isclose(dtn_ddn_nb, kp, atol=1e-3), \
        f"Compression tangent mismatch: Numba={dtn_ddn_nb:.2e}, expected={kp:.2e}"

    print(f"✓ Compression penalty: t_n={t_n_nb:.2e} Pa (kp={kp:.2e}) (Python ≈ Numba)")


def test_mixed_mode_wells_shear():
    """Test Python vs Numba parity for Wells shear model."""
    from xfem_clean.cohesive_laws import CohesiveLaw, CohesiveState, cohesive_update_mixed
    from xfem_clean.numba.kernels_cohesive import (
        pack_cohesive_law_params,
        cohesive_update_mixed_values_numba,
    )

    # Create mixed-mode law with Wells shear
    law = CohesiveLaw(
        Kn=1e13,
        ft=3e6,
        Gf=100.0,
        mode="mixed",
        shear_model="wells",
        k_s0=1e13,
        k_s1=1e11,
        w1=1.0e-3,
        use_cyclic_closure=True,
    )

    # Pack params
    params = pack_cohesive_law_params(law)

    # First load to w_max
    delta_n_max = 0.5e-3
    delta_t = 1e-6
    state = CohesiveState(delta_max=0.0, damage=0.0)

    # Python path (loading)
    t_vec_py1, K_mat_py1, state_py1 = cohesive_update_mixed(law, delta_n_max, delta_t, state)

    # Numba path (loading)
    t_n_nb1, t_t_nb1, dtn_ddn_nb1, dtn_ddt_nb1, dtt_ddn_nb1, dtt_ddt_nb1, dm_nb1, dmg_nb1 = \
        cohesive_update_mixed_values_numba(delta_n_max, delta_t, 0.0, 0.0, params)

    # Compare loading phase
    assert np.isclose(t_vec_py1[0], t_n_nb1, rtol=0.01), \
        f"Wells loading t_n mismatch: Python={t_vec_py1[0]:.2e}, Numba={t_n_nb1:.2e}"
    assert np.isclose(t_vec_py1[1], t_t_nb1, rtol=0.01), \
        f"Wells loading t_t mismatch: Python={t_vec_py1[1]:.2e}, Numba={t_t_nb1:.2e}"

    # Unload (cyclic: shear stiffness should use w_max)
    delta_n_unload = 0.1e-3

    # Python path (unloading)
    t_vec_py2, K_mat_py2, state_py2 = cohesive_update_mixed(law, delta_n_unload, delta_t, state_py1)

    # Numba path (unloading)
    t_n_nb2, t_t_nb2, dtn_ddn_nb2, dtn_ddt_nb2, dtt_ddn_nb2, dtt_ddt_nb2, dm_nb2, dmg_nb2 = \
        cohesive_update_mixed_values_numba(delta_n_unload, delta_t, dm_nb1, dmg_nb1, params)

    # Compare unloading phase
    assert np.isclose(t_vec_py2[1], t_t_nb2, rtol=0.01), \
        f"Wells unloading t_t mismatch: Python={t_vec_py2[1]:.2e}, Numba={t_t_nb2:.2e}"

    # Verify cross-coupling (dtt_ddn should be zero during unloading in cyclic mode)
    assert np.isclose(K_mat_py2[1, 0], dtt_ddn_nb2, rtol=0.01), \
        f"Wells cross-coupling mismatch: Python={K_mat_py2[1,0]:.2e}, Numba={dtt_ddn_nb2:.2e}"

    print(f"✓ Wells shear model: t_t(load)={t_t_nb1:.2e} Pa, t_t(unload)={t_t_nb2:.2e} Pa (Python ≈ Numba)")


def test_param_packing_mode_I():
    """Test that unified param packing works for Mode I (backward compatibility)."""
    from xfem_clean.cohesive_laws import CohesiveLaw
    from xfem_clean.numba.kernels_cohesive import pack_cohesive_law_params

    # Create Mode I law
    law = CohesiveLaw(
        Kn=1e13,
        ft=3e6,
        Gf=100.0,
        law="bilinear",
        mode="I",
    )

    # Pack params
    params = pack_cohesive_law_params(law)

    # Check layout
    assert len(params) == 21, f"Unified params should have 21 elements, got {len(params)}"
    assert params[0] == 0.0, "law_id should be 0 (bilinear)"
    assert params[1] == 0.0, "mode_id should be 0 (Mode I)"
    assert params[2] == 1e13, "Kn should be packed at index 2"
    assert params[3] == 3e6, "ft should be packed at index 3"
    # Mixed-mode params should be zero/default for Mode I
    assert params[11] == 0.0, "Kt should be 0 for Mode I"
    assert params[12] == 0.0, "tau_max should be 0 for Mode I"

    print(f"✓ Param packing (Mode I): {len(params)} elements, law_id={params[0]}, mode_id={params[1]}")


def test_param_packing_mixed_mode():
    """Test that unified param packing works for mixed-mode."""
    from xfem_clean.cohesive_laws import CohesiveLaw
    from xfem_clean.numba.kernels_cohesive import pack_cohesive_law_params
    import math

    # Create mixed-mode law
    law = CohesiveLaw(
        Kn=1e13,
        ft=3e6,
        Gf=100.0,
        mode="mixed",
        Kt=5e12,
        tau_max=2e6,
        shear_model="wells",
        k_s0=1e13,
        k_s1=1e11,
        w1=1.0e-3,
        use_cyclic_closure=True,
    )

    # Pack params
    params = pack_cohesive_law_params(law)

    # Check layout
    assert len(params) == 21, f"Unified params should have 21 elements, got {len(params)}"
    assert params[1] == 1.0, "mode_id should be 1 (mixed)"
    assert params[11] == 5e12, f"Kt should be {5e12}, got {params[11]}"
    assert params[12] == 2e6, f"tau_max should be {2e6}, got {params[12]}"
    assert params[15] == 1.0, "shear_model_id should be 1 (wells)"
    assert params[16] == 1e13, f"k_s0 should be {1e13}, got {params[16]}"
    assert params[17] == 1e11, f"k_s1 should be {1e11}, got {params[17]}"
    assert params[18] == 1.0e-3, f"w1 should be {1.0e-3}, got {params[18]}"
    # hs = ln(k_s1/k_s0)/w1
    hs_expected = math.log(1e11 / 1e13) / 1.0e-3
    assert np.isclose(params[19], hs_expected, rtol=1e-10), \
        f"hs should be {hs_expected}, got {params[19]}"
    assert params[20] == 1.0, "use_cyclic_closure should be 1"

    print(f"✓ Param packing (mixed): mode_id={params[1]}, Kt={params[11]:.2e}, hs={params[19]:.2f}")


if __name__ == "__main__":
    print("=" * 70)
    print("MIXED-MODE COHESIVE NUMBA PARITY TESTS")
    print("=" * 70)

    print("\n[1/6] Testing param packing (Mode I)...")
    try:
        test_param_packing_mode_I()
    except Exception as e:
        print(f"✗ FAILED: {e}")

    print("\n[2/6] Testing param packing (mixed-mode)...")
    try:
        test_param_packing_mixed_mode()
    except Exception as e:
        print(f"✗ FAILED: {e}")

    print("\n[3/6] Testing mixed-mode elastic regime...")
    try:
        test_mixed_mode_elastic()
    except Exception as e:
        print(f"✗ FAILED: {e}")

    print("\n[4/6] Testing mixed-mode softening regime...")
    try:
        test_mixed_mode_softening()
    except Exception as e:
        print(f"✗ FAILED: {e}")

    print("\n[5/6] Testing compression penalty...")
    try:
        test_mixed_mode_compression_penalty()
    except Exception as e:
        print(f"✗ FAILED: {e}")

    print("\n[6/6] Testing Wells shear model...")
    try:
        test_mixed_mode_wells_shear()
    except Exception as e:
        print(f"✗ FAILED: {e}")

    print("\n" + "=" * 70)
    print("PARITY TESTS COMPLETE")
    print("=" * 70)
