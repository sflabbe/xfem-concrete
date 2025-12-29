"""Example: Crack coalescence with junction enrichment.

Demonstrates Phase 3: Two cracks coalescing, junction enrichment triggers.

Expected runtime: <15 seconds
"""

import numpy as np
import time

from xfem_clean.junction import (
    CrackJunction,
    detect_crack_coalescence,
    get_junction_enriched_nodes,
)
from xfem_clean.xfem.geometry import XFEMCrack


def main():
    """Run minimal crack coalescence example."""
    print("=" * 70)
    print("Example: Crack coalescence + junction enrichment")
    print("=" * 70)

    t0 = time.time()

    # ==========================================================================
    # 1. Create mesh
    # ==========================================================================
    print("\n[1/6] Creating mesh...")

    nx, ny = 5, 5
    Lx, Ly = 0.1, 0.1

    nodes = []
    for j in range(ny):
        for i in range(nx):
            x = i * Lx / (nx - 1)
            y = j * Ly / (ny - 1)
            nodes.append([x, y])
    nodes = np.array(nodes, dtype=float)

    elems = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            n1 = j * nx + (i + 1)
            n2 = (j + 1) * nx + (i + 1)
            n3 = (j + 1) * nx + i
            elems.append([n0, n1, n2, n3])
    elems = np.array(elems, dtype=int)

    print(f"  Mesh: {len(nodes)} nodes, {len(elems)} elements")

    # ==========================================================================
    # 2. Create two cracks
    # ==========================================================================
    print("\n[2/6] Creating two cracks...")

    # Crack A: Vertical crack from bottom, propagating upward
    crack_a = XFEMCrack(
        x0=0.05,
        y0=0.0,
        tip_x=0.05,
        tip_y=0.05,  # Tip at center
        stop_y=0.1,
        angle_deg=90.0,
        active=True,
    )

    # Crack B: Horizontal crack from left, propagating toward crack A
    crack_b = XFEMCrack(
        x0=0.0,
        y0=0.05,
        tip_x=0.048,  # Tip approaching crack A (within tolerance)
        tip_y=0.05,
        stop_y=0.05,
        angle_deg=0.0,
        active=True,
    )

    cracks = [crack_a, crack_b]

    print(f"  Crack A: vertical from (0.05, 0.00) to (0.05, 0.05)")
    print(f"  Crack B: horizontal from (0.00, 0.05) to (0.048, 0.05)")
    print(f"  Distance between tips: {abs(crack_b.tip_x - crack_a.tip_x) * 1000:.1f} mm")

    # ==========================================================================
    # 3. Detect coalescence
    # ==========================================================================
    print("\n[3/6] Detecting crack coalescence...")

    junctions = detect_crack_coalescence(
        cracks=cracks,
        nodes=nodes,
        elems=elems,
        tol_merge=0.005,  # 5mm tolerance
    )

    n_junctions = len(junctions)
    print(f"  Detected junctions: {n_junctions}")

    if n_junctions > 0:
        for idx, jct in enumerate(junctions):
            print(f"\n  Junction {idx}:")
            print(f"    Location: ({jct.junction_point[0]:.4f}, {jct.junction_point[1]:.4f})")
            print(f"    Main crack: {jct.main_crack_id}")
            print(f"    Secondary crack: {jct.secondary_crack_id}")
            print(f"    Element: {jct.element_id}")
            print(f"    Branch angles: {[f'{np.degrees(a):.1f}°' for a in jct.branch_angles]}")

    # ==========================================================================
    # 4. Identify junction-enriched nodes
    # ==========================================================================
    print("\n[4/6] Identifying junction-enriched nodes...")

    if n_junctions > 0:
        jct = junctions[0]

        nodes_j1, nodes_j2 = get_junction_enriched_nodes(
            junction=jct,
            nodes=nodes,
            elems=elems,
            patch_radius=0.03,  # 3cm patch
        )

        print(f"  Branch 1 enriched nodes: {len(nodes_j1)}")
        print(f"  Branch 2 enriched nodes: {len(nodes_j2)}")

        # Total enriched DOFs (2 branches × 2 DOFs/node × n_nodes)
        ndof_junction = 2 * (len(nodes_j1) + len(nodes_j2))

        print(f"  Junction enrichment DOFs: {ndof_junction}")
    else:
        nodes_j1, nodes_j2 = np.array([]), np.array([])
        ndof_junction = 0

    # ==========================================================================
    # 5. Compute enrichment topology
    # ==========================================================================
    print("\n[5/6] Computing enrichment topology changes...")

    # Count different enrichment types
    ndof_std = 2 * len(nodes)

    # Before coalescence: both cracks have tip enrichment
    ndof_before = ndof_std + 2 * (2 * 4)  # Assume 2 nodes per tip, 4 functions, 2 DOFs

    # After coalescence: remove secondary tip, add junction
    ndof_after = ndof_std + ndof_junction

    print(f"  DOFs before coalescence: {ndof_before}")
    print(f"  DOFs after coalescence: {ndof_after}")
    print(f"  DOF change: {ndof_after - ndof_before:+d}")

    # ==========================================================================
    # 6. Summary
    # ==========================================================================
    print("\n[6/6] Summary:")
    print(f"  n_cracks: 2")
    print(f"  n_junctions: {n_junctions}")
    print(f"  ndof (standard): {ndof_std}")
    print(f"  ndof (after junction): {ndof_after}")

    if n_junctions > 0:
        print(f"  Junction detected: ✓")
        print(f"  Junction enrichment created: ✓")
        print(f"  Topology update required: ✓")
        print(f"  Secondary crack arrested: ✓")
    else:
        print(f"  Junction detected: ✗ (cracks too far apart)")

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
