"""Test bond-slip Jacobian coupling (BLOQUE E).

Verifies that the 8x8 bond-slip tangent has off-diagonal steel↔concrete coupling.
This test ensures that the consistent Newton tangent is correctly implemented.
Tests both Numba kernel and Python fallback implementations.
"""
import numpy as np
import pytest
from xfem_clean.bond_slip import (
    assemble_bond_slip,
    BondSlipModelCode2010,
    BondSlipStateArrays,
)


@pytest.mark.parametrize("use_numba", [True, False], ids=["numba", "python"])
def test_bond_jacobian_has_steel_concrete_coupling(use_numba):
    """Verify bond Jacobian has non-zero off-diagonal steel↔concrete terms."""

    # Setup: 1 segment, 2 nodes (each has both concrete and steel DOFs)
    # Segment references nodes 0,1
    # Concrete DOFs are computed as 2*n (so node 0→DOFs 0,1; node 1→DOFs 2,3)
    # Steel DOFs are mapped explicitly (node 0→DOFs 4,5; node 1→DOFs 6,7)
    # Segment: [n1=0, n2=1, L0=0.1, cx=1.0, cy=0.0]  (horizontal bar)
    steel_segments = np.array([
        [0, 1, 0.1, 1.0, 0.0]  # [n1, n2, L0, cx, cy]
    ], dtype=float)

    # DOF mapping: nodes 0,1 → steel DOFs 4-7
    nnode = 2
    steel_dof_map = np.full((nnode, 2), -1, dtype=int)
    steel_dof_map[0, :] = [4, 5]  # node 0 -> steel DOFs 4,5
    steel_dof_map[1, :] = [6, 7]  # node 1 -> steel DOFs 6,7

    # Total DOFs: 8 (4 concrete + 4 steel)
    ndof_total = 8
    u_total = np.zeros(ndof_total, dtype=float)
    # Apply small displacement to steel node 0 in x-direction to trigger bond
    u_total[4] = 1e-4  # steel node 0, x-direction (DOF 4)

    # Bond law
    bond_law = BondSlipModelCode2010(
        d_bar=0.016,  # 16mm rebar
        f_cm=30e6,    # 30 MPa concrete
        condition="good",
    )

    # Bond states
    bond_states = BondSlipStateArrays.zeros(n_segments=1)

    # Steel properties
    steel_EA = 200e9 * (np.pi * 0.008**2)  # E*A for 16mm rebar

    # Assemble bond-slip system
    f_bond, K_bond, _ = assemble_bond_slip(
        u_total=u_total,
        steel_segments=steel_segments,
        steel_dof_offset=4,  # Steel DOFs start at index 4
        bond_law=bond_law,
        bond_states=bond_states,
        steel_dof_map=steel_dof_map,
        steel_EA=steel_EA,
        use_numba=use_numba,  # Test both Numba and Python fallback
        perimeter=np.pi * 0.016,  # Explicit perimeter
        bond_gamma=1.0,
    )

    # Convert to dense for inspection
    K_dense = K_bond.toarray()

    # Check: K should have steel↔concrete coupling (off-diagonal blocks)
    # Steel DOFs: 4,5,6,7
    # Concrete DOFs: 0,1,2,3 (but segment uses nodes 0,1 which map to DOFs 0,1,2,3)
    # The bond coupling is between steel DOFs and concrete DOFs via the gradient g

    # Extract steel block (rows/cols 4-7)
    K_steel = K_dense[4:8, 4:8]

    # Extract concrete block (rows/cols 0-3)
    K_concrete = K_dense[0:4, 0:4]

    # Extract steel-concrete coupling block (steel rows, concrete cols)
    K_steel_concrete = K_dense[4:8, 0:4]

    # Check: off-diagonal coupling should exist
    has_coupling = np.any(np.abs(K_steel_concrete) > 1e-12)

    assert has_coupling, (
        "Bond Jacobian should have steel↔concrete coupling (off-diagonal terms), "
        "but all coupling terms are zero. Check bond_slip.py 8x8 tangent implementation."
    )

    # Check: K should be symmetric (or nearly so)
    symmetry_error = np.linalg.norm(K_dense - K_dense.T)
    assert symmetry_error < 1e-10, (
        f"Bond Jacobian should be symmetric, but ||K - K^T|| = {symmetry_error:.2e}. "
        "Check bond_slip.py tangent assembly."
    )

    mode = "Numba" if use_numba else "Python"
    print(f"✓ [{mode}] Bond Jacobian has steel↔concrete coupling (max |K_sc| = {np.max(np.abs(K_steel_concrete)):.2e})")
    print(f"✓ [{mode}] Bond Jacobian is symmetric (||K - K^T|| = {symmetry_error:.2e})")


if __name__ == "__main__":
    print("Testing bond-slip Jacobian coupling in both modes...")
    print("\n=== Testing with Numba ===")
    test_bond_jacobian_has_steel_concrete_coupling(use_numba=True)
    print("\n=== Testing with Python fallback ===")
    test_bond_jacobian_has_steel_concrete_coupling(use_numba=False)
    print("\n✅ All tests passed!")
