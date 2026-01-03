"""Test bond dissipation tracking (TASK 5).

This test verifies that the bond assembly correctly tracks bond dissipation
when compute_dissipation=True and u_total_prev is provided.
"""

import sys
import os
repo_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(repo_root, 'src'))

import numpy as np


def test_bond_dissipation_simple_slip():
    """Test that bond dissipation formula is correct for simple slip cycle."""
    from xfem_clean.bond_slip import BondSlipModelCode2010, BondSlipStateArrays, assemble_bond_slip

    # Simple bond law
    law = BondSlipModelCode2010(
        f_cm=30e6,  # 30 MPa concrete
        d_bar=0.012,  # 12mm rebar
        condition="good",
    )

    # Single segment geometry: nodes at x=0 and x=1 (horizontal bar along x)
    steel_segments = np.array([
        [0, 1, 1.0, 1.0, 0.0]  # [n1, n2, L0, cx, cy]
    ], dtype=float)

    # DOF mapping:
    # Concrete: nodes 0,1 → DOFs 0-3
    # Steel: nodes 0,1 → DOFs 4-7
    steel_dof_map = np.array([
        [4, 5],  # node 0 → steel DOFs 4,5
        [6, 7],  # node 1 → steel DOFs 6,7
    ], dtype=np.int64)

    # Committed state (no slip history)
    bond_states_n = BondSlipStateArrays(
        n_segments=1,
        s_max=np.zeros(1, dtype=float),
        s_current=np.zeros(1, dtype=float),
        tau_current=np.zeros(1, dtype=float),
    )

    # State n: no displacement
    u_n = np.zeros(8, dtype=float)

    # State n+1: apply slip by displacing steel relative to concrete
    # To get slip s = 0.5mm in +x direction:
    # du = u_steel - u_concrete = s * cx
    # Since cx=1.0, we want du_x = 0.5mm at segment midpoint
    u_np1 = np.zeros(8, dtype=float)
    slip_target = 0.5e-3  # 0.5 mm
    # Apply displacement to steel nodes (DOFs 4-7)
    u_np1[4] = slip_target  # steel node 0, x-direction
    u_np1[6] = slip_target  # steel node 1, x-direction

    # Assemble at time n (no slip)
    f_n, K_n, states_n, aux_n = assemble_bond_slip(
        u_total=u_n,
        steel_segments=steel_segments,
        steel_dof_offset=4,
        bond_law=law,
        bond_states=bond_states_n,
        steel_dof_map=steel_dof_map,
        steel_EA=0.0,  # No steel axial for simplicity
        use_numba=False,  # Use Python path
        perimeter=np.pi * law.d_bar,
    )

    # Assemble at time n+1 with dissipation tracking
    f_np1, K_np1, states_np1, aux_np1 = assemble_bond_slip(
        u_total=u_np1,
        steel_segments=steel_segments,
        steel_dof_offset=4,
        bond_law=law,
        bond_states=states_n,  # Use committed state from time n
        steel_dof_map=steel_dof_map,
        steel_EA=0.0,
        use_numba=False,
        perimeter=np.pi * law.d_bar,
        u_total_prev=u_n,
        compute_dissipation=True,
    )

    D_bond = aux_np1["D_bond_inc"]

    print(f"\n  Bond dissipation test:")
    print(f"    Slip: 0 → {slip_target*1e3:.3f} mm")
    print(f"    Bond law: tau_max = {law.tau_max/1e6:.2f} MPa")
    print(f"    Segment length: 1.0 m")
    print(f"    Perimeter: {np.pi*law.d_bar*1e3:.2f} mm")
    print(f"    D_bond_inc: {D_bond:.6e} J")

    # Manual verification:
    # At s=0: tau=0
    # At s=0.5mm (assuming s < s1=1.0mm): tau = tau_max * (s/s1)^alpha
    s_np1 = slip_target
    if s_np1 <= law.s1:
        tau_np1 = law.tau_max * (s_np1 / law.s1) ** law.alpha
    else:
        tau_np1 = law.tau_max

    # Trapezoidal: ΔD = 0.5 * (0 + tau_np1) * (s_np1 - 0) * perimeter * L0
    D_expected = 0.5 * tau_np1 * s_np1 * (np.pi * law.d_bar) * 1.0

    print(f"    tau(s_n+1) = {tau_np1/1e6:.6f} MPa")
    print(f"    D_expected: {D_expected:.6e} J")
    print(f"    Relative error: {abs(D_bond - D_expected)/max(abs(D_expected), 1e-30) * 100:.3f}%")

    # Check that dissipation is positive and matches expected value
    assert D_bond > 0, f"Bond dissipation should be positive, got {D_bond:.3e} J"
    rel_error = abs(D_bond - D_expected) / max(abs(D_expected), 1e-30)
    assert rel_error < 0.01, f"Bond dissipation error {rel_error*100:.2f}% exceeds 1%"

    print(f"  ✓ Bond dissipation tracking works correctly")


def test_bond_dissipation_cyclic():
    """Test bond dissipation for cyclic loading (should show hysteresis)."""
    from xfem_clean.bond_slip import BondSlipModelCode2010, BondSlipStateArrays, assemble_bond_slip

    # Bond law
    law = BondSlipModelCode2010(
        f_cm=30e6,
        d_bar=0.012,
        condition="good",
    )

    # Single segment
    steel_segments = np.array([
        [0, 1, 1.0, 1.0, 0.0]
    ], dtype=float)

    steel_dof_map = np.array([
        [4, 5],
        [6, 7],
    ], dtype=np.int64)

    # Initial state
    bond_states = BondSlipStateArrays(
        n_segments=1,
        s_max=np.zeros(1, dtype=float),
        s_current=np.zeros(1, dtype=float),
        tau_current=np.zeros(1, dtype=float),
    )

    # Slip history: 0 → 1mm → 0 → 1mm
    slips = [0.0, 1.0e-3, 0.0, 1.0e-3]

    D_total = 0.0
    u_prev = np.zeros(8, dtype=float)

    for i, slip in enumerate(slips[1:], 1):
        u = np.zeros(8, dtype=float)
        u[4] = slip
        u[6] = slip

        _, _, bond_states, aux = assemble_bond_slip(
            u_total=u,
            steel_segments=steel_segments,
            steel_dof_offset=4,
            bond_law=law,
            bond_states=bond_states,
            steel_dof_map=steel_dof_map,
            steel_EA=0.0,
            use_numba=False,
            perimeter=np.pi * law.d_bar,
            u_total_prev=u_prev,
            compute_dissipation=True,
        )

        D_step = aux["D_bond_inc"]
        D_total += D_step

        print(f"  Step {i}: s={slips[i-1]*1e3:.1f}mm → {slip*1e3:.1f}mm, ΔD={D_step:.3e} J (cumulative: {D_total:.3e} J)")

        u_prev = u.copy()

    print(f"\n  Cyclic loading total dissipation: {D_total:.6e} J")

    # For cyclic loading with unloading, total dissipation should be positive
    # (energy is lost due to hysteresis)
    assert D_total > 0, f"Cyclic dissipation should be positive, got {D_total:.3e} J"

    print(f"  ✓ Cyclic bond dissipation test passed")


if __name__ == "__main__":
    print("=" * 70)
    print("TASK 5: Bond Dissipation Tracking Tests")
    print("=" * 70)

    print("\n[1/2] Testing simple slip dissipation...")
    test_bond_dissipation_simple_slip()

    print("\n[2/2] Testing cyclic loading dissipation...")
    test_bond_dissipation_cyclic()

    print("\n" + "=" * 70)
    print("✓ All bond dissipation tracking tests passed!")
    print("=" * 70)
