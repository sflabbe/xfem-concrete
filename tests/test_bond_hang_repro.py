"""Minimal reproducer for bond-slip hanging issue after crack growth.

This test reproduces the exact conditions that cause the hang:
1. Bond-slip interface elements active
2. Crack grows and changes DOF count (e.g., 686 → 690)
3. Newton iteration calls assemble_bond_slip with updated DOFs

The test should:
- FAIL with diagnostic error (not hang) before fix
- PASS after fix
"""

import numpy as np
import pytest
from xfem_clean.bond_slip import (
    BondSlipModelCode2010,
    BondSlipStateArrays,
    assemble_bond_slip,
    validate_bond_inputs,
)


def test_bond_slip_dof_change():
    """Test that bond-slip handles DOF changes gracefully."""
    # Initial setup: 10 nodes, 4 steel segments
    nnode = 10
    n_seg = 4
    ndof_initial = 2 * nnode + 2 * nnode  # Concrete + steel

    # Create steel DOF map (dense, all nodes have steel)
    steel_dof_offset = 2 * nnode
    steel_dof_map = np.zeros((nnode, 2), dtype=np.int64)
    for i in range(nnode):
        steel_dof_map[i, 0] = steel_dof_offset + 2 * i
        steel_dof_map[i, 1] = steel_dof_offset + 2 * i + 1

    # Create segments
    segs = np.array([
        [0, 1, 0.1, 1.0, 0.0],  # horizontal segments
        [2, 3, 0.1, 1.0, 0.0],
        [4, 5, 0.1, 1.0, 0.0],
        [6, 7, 0.1, 1.0, 0.0],
    ], dtype=float)

    # Create bond law
    bond_law = BondSlipModelCode2010(f_cm=30e6, d_bar=0.016, condition="good")

    # Create initial state
    bond_states = BondSlipStateArrays.zeros(n_seg)

    # Initial displacement (all zeros)
    u_total = np.zeros(ndof_initial, dtype=float)

    # Initial assembly should work
    f1, K1, states1 = assemble_bond_slip(
        u_total=u_total,
        steel_segments=segs,
        steel_dof_offset=steel_dof_offset,
        bond_law=bond_law,
        bond_states=bond_states,
        steel_dof_map=steel_dof_map,
        use_numba=True,
        enable_validation=True,
    )

    assert f1.shape == (ndof_initial,)
    assert K1.shape == (ndof_initial, ndof_initial)

    # Now simulate crack growth: add 4 enriched DOFs
    ndof_new = ndof_initial + 4
    steel_dof_offset_new = steel_dof_offset + 4

    # Update steel DOF map to reflect new offset
    steel_dof_map_new = np.zeros((nnode, 2), dtype=np.int64)
    for i in range(nnode):
        steel_dof_map_new[i, 0] = steel_dof_offset_new + 2 * i
        steel_dof_map_new[i, 1] = steel_dof_offset_new + 2 * i + 1

    # New displacement vector (pad with zeros for new DOFs)
    u_total_new = np.zeros(ndof_new, dtype=float)

    # This should either:
    # 1. Work correctly (after fix)
    # 2. Raise a validation error with clear message (during fix)
    # 3. NOT hang (this is the bug we're fixing)

    # Call with new DOFs
    f2, K2, states2 = assemble_bond_slip(
        u_total=u_total_new,
        steel_segments=segs,
        steel_dof_offset=steel_dof_offset_new,
        bond_law=bond_law,
        bond_states=states1,
        steel_dof_map=steel_dof_map_new,
        use_numba=True,
        enable_validation=True,
    )

    assert f2.shape == (ndof_new,)
    assert K2.shape == (ndof_new, ndof_new)

    # Multiple calls to simulate Newton iterations
    for i in range(5):
        f_iter, K_iter, states_iter = assemble_bond_slip(
            u_total=u_total_new,
            steel_segments=segs,
            steel_dof_offset=steel_dof_offset_new,
            bond_law=bond_law,
            bond_states=states2,
            steel_dof_map=steel_dof_map_new,
            use_numba=True,
            enable_validation=True,
        )
        assert f_iter.shape == (ndof_new,)


def test_bond_slip_invalid_dof_map():
    """Test that validation catches invalid DOF mappings."""
    nnode = 10
    n_seg = 2
    ndof_total = 50
    steel_dof_offset = 20

    # Create INVALID steel DOF map (some indices out of bounds)
    steel_dof_map = np.zeros((nnode, 2), dtype=np.int64)
    steel_dof_map[0, 0] = 60  # Out of bounds!
    steel_dof_map[0, 1] = 61  # Out of bounds!

    segs = np.array([
        [0, 1, 0.1, 1.0, 0.0],
        [2, 3, 0.1, 1.0, 0.0],
    ], dtype=float)

    bond_states = BondSlipStateArrays.zeros(n_seg)
    u_total = np.zeros(ndof_total, dtype=float)

    # Validation should catch this
    with pytest.raises(RuntimeError, match="bond-slip invalid DOF mapping"):
        validate_bond_inputs(
            u_total=u_total,
            segs=segs,
            steel_dof_map=steel_dof_map,
            steel_dof_offset=steel_dof_offset,
            bond_states=bond_states,
        )


def test_bond_slip_missing_steel_dofs():
    """Test that validation catches missing steel DOFs."""
    nnode = 10
    n_seg = 2
    ndof_total = 50
    steel_dof_offset = 20

    # Create steel DOF map with -1 (no steel)
    steel_dof_map = np.full((nnode, 2), -1, dtype=np.int64)

    segs = np.array([
        [0, 1, 0.1, 1.0, 0.0],  # Node 0 has no steel!
        [2, 3, 0.1, 1.0, 0.0],
    ], dtype=float)

    bond_states = BondSlipStateArrays.zeros(n_seg)
    u_total = np.zeros(ndof_total, dtype=float)

    # Validation should catch this
    with pytest.raises(RuntimeError, match="has no steel DOFs"):
        validate_bond_inputs(
            u_total=u_total,
            segs=segs,
            steel_dof_map=steel_dof_map,
            steel_dof_offset=steel_dof_offset,
            bond_states=bond_states,
        )


def test_bond_slip_python_fallback():
    """Test that Python fallback works correctly."""
    nnode = 6
    n_seg = 2
    steel_dof_offset = 2 * nnode
    ndof_total = steel_dof_offset + 2 * nnode

    # Create steel DOF map
    steel_dof_map = np.zeros((nnode, 2), dtype=np.int64)
    for i in range(nnode):
        steel_dof_map[i, 0] = steel_dof_offset + 2 * i
        steel_dof_map[i, 1] = steel_dof_offset + 2 * i + 1

    segs = np.array([
        [0, 1, 0.1, 1.0, 0.0],
        [2, 3, 0.1, 1.0, 0.0],
    ], dtype=float)

    bond_law = BondSlipModelCode2010(f_cm=30e6, d_bar=0.016, condition="good")
    bond_states = BondSlipStateArrays.zeros(n_seg)

    # Apply some displacement
    u_total = np.random.randn(ndof_total) * 1e-4

    # Compare Python and Numba results
    f_numba, K_numba, states_numba = assemble_bond_slip(
        u_total=u_total,
        steel_segments=segs,
        steel_dof_offset=steel_dof_offset,
        bond_law=bond_law,
        bond_states=bond_states,
        steel_dof_map=steel_dof_map,
        use_numba=True,
        enable_validation=True,
    )

    f_python, K_python, states_python = assemble_bond_slip(
        u_total=u_total,
        steel_segments=segs,
        steel_dof_offset=steel_dof_offset,
        bond_law=bond_law,
        bond_states=bond_states,
        steel_dof_map=steel_dof_map,
        use_numba=False,
        enable_validation=False,  # Python has its own assertions
    )

    # Results should be very close
    np.testing.assert_allclose(f_numba, f_python, rtol=1e-10)
    np.testing.assert_allclose(K_numba.toarray(), K_python.toarray(), rtol=1e-10)
    np.testing.assert_allclose(states_numba.s_current, states_python.s_current, rtol=1e-10)


if __name__ == "__main__":
    print("Running bond-slip hang reproducer tests...")

    print("\n1. Testing DOF change handling...")
    test_bond_slip_dof_change()
    print("   ✓ PASSED")

    print("\n2. Testing invalid DOF map detection...")
    test_bond_slip_invalid_dof_map()
    print("   ✓ PASSED")

    print("\n3. Testing missing steel DOFs detection...")
    test_bond_slip_missing_steel_dofs()
    print("   ✓ PASSED")

    print("\n4. Testing Python fallback...")
    test_bond_slip_python_fallback()
    print("   ✓ PASSED")

    print("\n✅ All tests passed!")
