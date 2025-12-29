"""Example: Reinforcement bar at 33° crossing elements (mesh-independent).

Demonstrates Phase 1: Heaviside-enriched reinforcement layer
that is NOT aligned with the mesh grid.

Expected runtime: <10 seconds
"""

import numpy as np
import time

from xfem_clean.reinforcement import (
    create_straight_reinforcement_layer,
    ReinforcementState,
)


def main():
    """Run minimal rebar Heaviside enrichment example."""
    print("=" * 70)
    print("Example: Mesh-independent reinforcement at 33° angle")
    print("=" * 70)

    t0 = time.time()

    # ==========================================================================
    # 1. Create simple mesh (4x4 elements, 0.1m x 0.1m domain)
    # ==========================================================================
    print("\n[1/5] Creating mesh...")

    nx, ny = 5, 5  # 4x4 elements
    Lx, Ly = 0.1, 0.1

    # Nodes
    nodes = []
    for j in range(ny):
        for i in range(nx):
            x = i * Lx / (nx - 1)
            y = j * Ly / (ny - 1)
            nodes.append([x, y])
    nodes = np.array(nodes, dtype=float)

    # Elements (Q4)
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
    # 2. Create reinforcement layer at 33° angle
    # ==========================================================================
    print("\n[2/5] Creating reinforcement layer at 33°...")

    # Bar endpoints (diagonal across domain at ~33°)
    angle_deg = 33.0
    angle_rad = np.radians(angle_deg)

    x_start = np.array([0.01, 0.01], dtype=float)
    L_bar = 0.08  # 8 cm bar
    x_end = x_start + L_bar * np.array([np.cos(angle_rad), np.sin(angle_rad)])

    # Steel properties
    A_s = np.pi * (0.008 ** 2) / 4  # 8mm diameter bar
    E_s = 200e9  # 200 GPa
    f_y = 500e6  # 500 MPa
    E_h = 20e9   # 20 GPa hardening
    d_bar = 0.008  # 8mm

    layer = create_straight_reinforcement_layer(
        x_start=x_start,
        x_end=x_end,
        A_s=A_s,
        E_s=E_s,
        f_y=f_y,
        E_h=E_h,
        d_bar=d_bar,
        layer_type="longitudinal",
        layer_id=0,
        n_segments=4,  # Subdivide for better integration
    )

    print(f"  Layer: {len(layer.segments)} segments, total length = {layer.total_length():.4f} m")
    print(f"  Angle: {angle_deg}° (NOT aligned with mesh)")

    # ==========================================================================
    # 3. Compute enriched nodes (nodes whose support intersects layer)
    # ==========================================================================
    print("\n[3/5] Computing enriched nodes...")

    from xfem_clean.reinforcement import signed_distance_to_segment

    enriched_nodes = np.zeros(len(nodes), dtype=bool)

    for seg in layer.segments:
        for n_idx, node_pos in enumerate(nodes):
            dist = abs(signed_distance_to_segment(node_pos, seg.x0, seg.x1))
            # Enrich if node is within element size
            if dist < 0.03:  # 3 cm threshold
                enriched_nodes[n_idx] = True

    n_enriched = np.sum(enriched_nodes)
    print(f"  Enriched nodes: {n_enriched} / {len(nodes)} ({100*n_enriched/len(nodes):.1f}%)")

    # ==========================================================================
    # 4. Count intersected elements
    # ==========================================================================
    print("\n[4/5] Counting intersected elements...")

    from xfem_clean.reinforcement import segment_element_intersection

    n_intersected = 0

    for e_idx in range(len(elems)):
        elem_coords = nodes[elems[e_idx]]

        for seg in layer.segments:
            isect = segment_element_intersection(seg, elem_coords)
            if isect is not None:
                n_intersected += 1
                break

    print(f"  Intersected elements: {n_intersected} / {len(elems)} ({100*n_intersected/len(elems):.1f}%)")

    # ==========================================================================
    # 5. Print statistics
    # ==========================================================================
    print("\n[5/5] Summary statistics:")
    print(f"  ndof (standard): {2 * len(nodes)}")
    print(f"  ndof (enriched, if used): {2 * len(nodes) + 2 * n_enriched}")
    print(f"  Enrichment type: Heaviside (mesh-independent)")
    print(f"  Bar angle: {angle_deg}° (crosses {n_intersected} elements)")
    print(f"  Bar crosses mesh at arbitrary angle ✓")

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
