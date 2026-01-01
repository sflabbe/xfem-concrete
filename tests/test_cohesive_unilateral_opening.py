"""Test cohesive law unilateral opening behavior (P0.1 fix).

Verifies that:
1. Compression (δ < 0) returns zero traction
2. Compression does not accumulate damage
3. Cyclic loading (open → close → reopen) only accumulates damage from opening
"""

import pytest
import numpy as np

from xfem_clean.cohesive_laws import CohesiveLaw, CohesiveState, cohesive_update
from xfem_clean.numba.kernels_cohesive import cohesive_update_values_numba, pack_cohesive_law_params


def test_cohesive_compression_zero_traction():
    """Compression (δ < 0) should return zero traction."""
    law = CohesiveLaw(Kn=1e12, ft=3e6, Gf=100.0)
    st = CohesiveState()

    # Apply compression
    delta_compression = -1e-6  # Negative = compression
    T, k, st2 = cohesive_update(law, delta_compression, st)

    # Should return zero traction
    assert T == 0.0, f"Expected zero traction for compression, got T={T}"
    # Should not accumulate damage
    assert st2.delta_max == 0.0, f"Compression should not increase delta_max, got {st2.delta_max}"
    assert st2.damage == 0.0, f"Compression should not increase damage, got {st2.damage}"


def test_cohesive_compression_no_damage_accumulation():
    """Compression after opening should not increase damage."""
    law = CohesiveLaw(Kn=1e12, ft=3e6, Gf=100.0)
    st = CohesiveState()

    # 1. Apply opening to create some damage
    delta_open = 5e-6  # Positive = opening
    T1, k1, st1 = cohesive_update(law, delta_open, st)
    assert st1.delta_max == pytest.approx(delta_open)
    initial_damage = st1.damage

    # 2. Apply compression
    delta_compress = -3e-6  # Negative = compression
    T2, k2, st2 = cohesive_update(law, delta_compress, st1)

    # Traction should be zero
    assert T2 == 0.0

    # Damage should NOT increase
    assert st2.delta_max == st1.delta_max, "Compression should not increase delta_max"
    assert st2.damage == st1.damage, "Compression should not increase damage"


def test_cohesive_cyclic_open_close_reopen():
    """Cyclic loading: damage only from opening, not closing."""
    law = CohesiveLaw(Kn=1e12, ft=3e6, Gf=100.0, law="bilinear")
    st = CohesiveState()

    # 1. First opening
    delta1 = 4e-6
    T1, k1, st1 = cohesive_update(law, delta1, st)
    assert st1.delta_max == pytest.approx(delta1)
    damage1 = st1.damage
    assert damage1 > 0, "Should have some damage from opening"

    # 2. Close (compression)
    delta2 = -2e-6
    T2, k2, st2 = cohesive_update(law, delta2, st1)
    assert T2 == 0.0, "Compression should give zero traction"
    assert st2.delta_max == st1.delta_max, "Closing should not change delta_max"
    assert st2.damage == damage1, "Closing should not change damage"

    # 3. Reopen to smaller opening (unloading path)
    delta3 = 2e-6
    T3, k3, st3 = cohesive_update(law, delta3, st2)
    assert st3.delta_max == st1.delta_max, "Partial reopening should not exceed previous max"
    assert st3.damage == damage1, "Unloading should not change damage"

    # 4. Reopen beyond previous max (new loading)
    delta4 = 6e-6
    T4, k4, st4 = cohesive_update(law, delta4, st3)
    assert st4.delta_max == pytest.approx(delta4), "New max opening should update"
    assert st4.damage > damage1, "Further opening should increase damage"


def test_cohesive_unilateral_numba_parity():
    """Numba kernel should match Python for unilateral opening."""
    law = CohesiveLaw(Kn=1e12, ft=3e6, Gf=100.0)
    params = pack_cohesive_law_params(law)
    st = CohesiveState()

    # Test compression
    delta_compress = -5e-6
    T_py, k_py, st_py = cohesive_update(law, delta_compress, st)
    T_nb, k_nb, dm_nb, d_nb = cohesive_update_values_numba(delta_compress, st.delta_max, st.damage, params)

    assert T_py == T_nb == 0.0, "Both should return zero traction for compression"
    assert dm_nb == 0.0, "Numba should not accumulate delta_max from compression"
    assert d_nb == 0.0, "Numba should not accumulate damage from compression"

    # Test opening
    delta_open = 3e-6
    T_py2, k_py2, st_py2 = cohesive_update(law, delta_open, st)
    T_nb2, k_nb2, dm_nb2, d_nb2 = cohesive_update_values_numba(delta_open, st.delta_max, st.damage, params)

    assert T_py2 == pytest.approx(T_nb2), "Python and Numba should match for opening"
    assert st_py2.delta_max == pytest.approx(dm_nb2), "delta_max should match"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
