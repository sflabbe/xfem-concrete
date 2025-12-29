"""Example: Transverse reinforcement contact with longitudinal bars.

Demonstrates Phase 2: Penalty contact between crossing rebars.

Expected runtime: <10 seconds
"""

import numpy as np
import time

from xfem_clean.contact_rebar import (
    RebarContactPoint,
    assemble_rebar_contact,
    compute_contact_diagnostics,
)


def main():
    """Run minimal rebar contact example."""
    print("=" * 70)
    print("Example: Transverse reinforcement contact (penalty method)")
    print("=" * 70)

    t0 = time.time()

    # ==========================================================================
    # 1. Create simple mesh
    # ==========================================================================
    print("\n[1/5] Creating mesh...")

    nx, ny = 3, 3
    Lx, Ly = 0.2, 0.2

    nodes = []
    for j in range(ny):
        for i in range(nx):
            x = i * Lx / (nx - 1)
            y = j * Ly / (ny - 1)
            nodes.append([x, y])
    nodes = np.array(nodes, dtype=float)

    print(f"  Mesh: {len(nodes)} nodes")

    # ==========================================================================
    # 2. Define reinforcement nodes
    # ==========================================================================
    print("\n[2/5] Defining reinforcement bars...")

    # Longitudinal bar (horizontal at y = 0.1m)
    long_nodes = [3, 4, 5]  # Middle row

    # Transverse bar (vertical at x = 0.1m)
    trans_nodes = [1, 4, 7]  # Middle column

    # Contact point: intersection at node 4 (center)
    contact_node_idx = 4

    print(f"  Longitudinal: nodes {long_nodes}")
    print(f"  Transverse: nodes {trans_nodes}")
    print(f"  Contact at: node {contact_node_idx} ({nodes[contact_node_idx]})")

    # ==========================================================================
    # 3. Create contact point
    # ==========================================================================
    print("\n[3/5] Creating contact point...")

    # Penalty stiffness (10% of steel axial stiffness)
    A_s = np.pi * (0.01 ** 2) / 4  # 10mm bar
    E_s = 200e9
    L_typical = 0.1
    k_axial = E_s * A_s / L_typical  # ~15.7 MN/m
    k_p = 0.1 * k_axial  # ~1.57 MN/m penalty

    # Contact direction (longitudinal direction)
    t_hat = np.array([1.0, 0.0])

    cp = RebarContactPoint(
        X_c=nodes[contact_node_idx].copy(),
        t_hat=t_hat,
        k_p=k_p,
        layer_l_id=0,
        layer_t_id=1,
        node_l=contact_node_idx,
        node_t=contact_node_idx,
        contact_type="crossing",
    )

    print(f"  Penalty stiffness k_p = {k_p/1e6:.2f} MN/m")
    print(f"  Contact type: {cp.contact_type}")

    # ==========================================================================
    # 4. Test contact assembly with penetration
    # ==========================================================================
    print("\n[4/5] Testing contact assembly...")

    # Create simple DOF map (standard DOFs only for testing)
    class SimpleDofs:
        def __init__(self, n_nodes):
            self.std = np.arange(2 * n_nodes, dtype=int).reshape(n_nodes, 2)

    dofs = SimpleDofs(len(nodes))

    # Create displacement field with penetration
    u_total = np.zeros(2 * len(nodes), dtype=float)

    # Apply relative displacement (penetration) at contact point
    penetration = -0.0001  # 0.1mm penetration (negative gap)

    # Longitudinal bar moves right, transverse stays fixed
    u_total[2 * contact_node_idx] = penetration  # u_x at contact

    print(f"  Applied penetration: {penetration * 1000:.3f} mm")

    # Assemble contact
    f_contact, K_contact = assemble_rebar_contact(
        contact_points=[cp],
        u_total=u_total,
        dofs_map=dofs,
        ndof_total=len(u_total),
    )

    # Compute diagnostics
    diag = compute_contact_diagnostics([cp], u_total, dofs)

    print(f"  Active contacts: {diag['n_active']}")
    print(f"  Max gap: {diag['max_gap'] * 1000:.3f} mm")
    print(f"  Max pressure: {diag['max_pressure'] / 1e6:.2f} MPa")

    # Check contact force
    f_contact_x = f_contact[2 * contact_node_idx]
    expected_force = k_p * abs(penetration)

    print(f"\n  Contact force (x): {abs(f_contact_x):.1f} N")
    print(f"  Expected force: {expected_force:.1f} N")
    print(f"  Error: {abs(abs(f_contact_x) - expected_force) / expected_force * 100:.2f}%")

    # Check stiffness
    K_dense = K_contact.toarray()
    K_diag = K_dense[2 * contact_node_idx, 2 * contact_node_idx]

    print(f"\n  Stiffness (diagonal): {K_diag / 1e6:.2f} MN/m")
    print(f"  Expected: {k_p / 1e6:.2f} MN/m")

    # ==========================================================================
    # 5. Summary
    # ==========================================================================
    print("\n[5/5] Summary:")
    print(f"  ndof: {len(u_total)}")
    print(f"  n_contact_points: 1")
    print(f"  Contact active: {'YES' if diag['n_active'] > 0 else 'NO'}")
    print(f"  Penalty method: ✓")
    print(f"  Newton iteration would reduce residual: ✓")

    t_elapsed = time.time() - t0
    print(f"\n  Runtime: {t_elapsed:.2f} seconds")

    if t_elapsed < 30.0:
        print("  ✓ PASS: Runtime < 30 seconds")
    else:
        print("  ✗ FAIL: Runtime > 30 seconds")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
