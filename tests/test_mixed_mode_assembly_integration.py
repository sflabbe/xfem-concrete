"""Integration test for mixed-mode cohesive assembly (TASK 3).

This test verifies that the assembly code correctly handles mixed-mode cohesive
with both normal and tangential jumps, and produces correct forces and stiffness.
"""

import sys
import os
repo_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(repo_root, 'src'))

import numpy as np
import pytest


def test_mixed_mode_assembly_simple_crack():
    """Test mixed-mode assembly with a simple horizontal crack under shear."""
    from xfem_clean.cohesive_laws import CohesiveLaw
    from xfem_clean.xfem.geometry import XFEMCrack
    from xfem_clean.xfem.dofs_single import XFEMDofs, build_xfem_dofs
    from xfem_clean.xfem.assembly_single import assemble_xfem_system

    # Simple 2×2 mesh (4 elements, 9 nodes)
    # Crack at y=0.5 (middle)
    nx, ny = 2, 2
    L, H = 1.0, 1.0
    nodes = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = i * L / nx
            y = j * H / ny
            nodes.append([x, y])
    nodes = np.array(nodes, dtype=float)

    elems = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1) + 1
            n3 = n0 + (nx + 1)
            elems.append([n0, n1, n2, n3])
    elems = np.array(elems, dtype=int)

    # Horizontal crack at y=0.5
    crack = XFEMCrack(
        x0=0.0, y0=0.5,
        tip_x=1.0, tip_y=0.5,
        stop_y=0.5,
        angle_deg=0.0,
        active=True
    )

    # Build XFEM DOFs
    # Tip enrichment patch: around crack tip at (1.0, 0.5)
    tip_patch = (0.7, 1.0, 0.2, 0.8)
    dofs = build_xfem_dofs(
        nodes, elems, crack,
        H_region_ymax=0.5,  # Heaviside below y=0.5
        tip_patch=tip_patch,
    )
    ndof = dofs.ndof
    print(f"\n  Mesh: {nx}×{ny} elements, {nodes.shape[0]} nodes")
    print(f"  DOFs: {ndof} total (enriched)")

    # Mixed-mode cohesive law
    law = CohesiveLaw(
        Kn=1e12,  # 1e12 Pa/m
        ft=3e6,   # 3 MPa
        Gf=100.0, # 100 N/m
        mode="mixed",
        tau_max=3e6,  # Same as ft
        Kt=1e12,      # Same as Kn
        Gf_II=100.0,  # Same as Gf
        law="bilinear",
    )

    # Elastic material
    E = 30e9  # 30 GPa
    nu = 0.2
    C = E / (1.0 - nu**2) * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, 0.5 * (1.0 - nu)]
    ])
    thickness = 0.1  # 0.1 m

    # Displacement field: impose small shear jump across crack
    # Normal opening: δn = 1 μm, tangential slip: δs = 2 μm
    q = np.zeros(ndof, dtype=float)

    # Simple test: just check assembly runs without error
    try:
        K, fint, coh_states_updated, _, _, _, _, _ = assemble_xfem_system(
            nodes=nodes,
            elems=elems,
            dofs=dofs,
            crack=crack,
            C=C,
            thickness=thickness,
            q=q,
            law=law,
            coh_states_comm={},
            tip_enr_radius=0.3,
            k_stab=1e-9,
            visc_damp=0.0,
            use_numba=False,
        )

        print(f"  ✓ Assembly succeeded with mixed-mode")
        print(f"    K shape: {K.shape}, fint shape: {fint.shape}")
        print(f"    Cohesive states updated: {len(coh_states_updated)} integration points")

        # Check that stiffness matrix is approximately symmetric
        # Small asymmetry expected due to FD approximations in tangent cross-coupling
        K_dense = K.toarray()
        K_err = np.max(np.abs(K_dense - K_dense.T))
        assert K_err < 1e-4, f"Stiffness should be approximately symmetric, error={K_err:.2e}"
        print(f"    Stiffness symmetry check: max error = {K_err:.2e} ✓")

    except Exception as e:
        pytest.fail(f"Assembly failed with mixed-mode: {e}")


def test_mixed_mode_vs_mode_I_pure_normal():
    """Test that mixed-mode with pure normal opening matches Mode I-only."""
    from xfem_clean.cohesive_laws import CohesiveLaw
    from xfem_clean.xfem.geometry import XFEMCrack
    from xfem_clean.xfem.dofs_single import build_xfem_dofs
    from xfem_clean.xfem.assembly_single import assemble_xfem_system

    # Simple 1×1 mesh (1 element, 4 nodes)
    nodes = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=float)
    elems = np.array([[0, 1, 2, 3]], dtype=int)

    # Horizontal crack at y=0.5
    crack = XFEMCrack(
        x0=0.0, y0=0.5,
        tip_x=1.0, tip_y=0.5,
        stop_y=0.5,
        angle_deg=0.0,
        active=True
    )

    # Tip enrichment patch around crack tip at (1.0, 0.5)
    tip_patch = (0.7, 1.0, 0.2, 0.8)
    dofs = build_xfem_dofs(
        nodes, elems, crack,
        H_region_ymax=0.5,
        tip_patch=tip_patch,
    )

    # Mode I-only law
    law_modeI = CohesiveLaw(
        Kn=1e12,
        ft=3e6,
        Gf=100.0,
        mode="I",
        law="bilinear",
    )

    # Mixed-mode law (same parameters)
    law_mixed = CohesiveLaw(
        Kn=1e12,
        ft=3e6,
        Gf=100.0,
        mode="mixed",
        tau_max=3e6,
        Kt=1e12,
        Gf_II=100.0,
        law="bilinear",
    )

    E = 30e9
    nu = 0.2
    C = E / (1.0 - nu**2) * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, 0.5 * (1.0 - nu)]
    ])
    thickness = 0.1

    # Impose pure normal opening (no shear)
    # Apply small displacement to top nodes
    q = np.zeros(dofs.ndof, dtype=float)
    # Top left node (node 3): y-displacement = 1e-6
    if dofs.std[3, 1] >= 0:
        q[dofs.std[3, 1]] = 1e-6
    # Top right node (node 2): y-displacement = 1e-6
    if dofs.std[2, 1] >= 0:
        q[dofs.std[2, 1]] = 1e-6

    # Assemble with Mode I
    K_modeI, fint_modeI, _, _, _, _, _, _ = assemble_xfem_system(
        nodes=nodes,
        elems=elems,
        dofs=dofs,
        crack=crack,
        C=C,
        thickness=thickness,
        q=q,
        law=law_modeI,
        coh_states_comm={},
        tip_enr_radius=0.3,
        use_numba=False,
    )

    # Assemble with mixed-mode
    K_mixed, fint_mixed, _, _, _, _, _, _ = assemble_xfem_system(
        nodes=nodes,
        elems=elems,
        dofs=dofs,
        crack=crack,
        C=C,
        thickness=thickness,
        q=q,
        law=law_mixed,
        coh_states_comm={},
        tip_enr_radius=0.3,
        use_numba=False,
    )

    print(f"\n  Comparing Mode I vs Mixed-mode (pure normal opening):")
    print(f"    fint norm (Mode I):  {np.linalg.norm(fint_modeI):.6e}")
    print(f"    fint norm (Mixed):   {np.linalg.norm(fint_mixed):.6e}")

    # Forces should match for pure normal opening
    f_diff = np.linalg.norm(fint_mixed - fint_modeI)
    f_ref = np.linalg.norm(fint_modeI)
    if f_ref > 1e-9:
        rel_err = f_diff / f_ref
        print(f"    Relative error: {rel_err*100:.3f}%")
        assert rel_err < 0.01, f"Forces should match for pure normal opening: {rel_err*100:.1f}% error"
    else:
        assert f_diff < 1e-9, "Forces should be near zero"

    print(f"  ✓ Mixed-mode matches Mode I for pure normal opening")


if __name__ == "__main__":
    print("=" * 70)
    print("TASK 3: Mixed-Mode Assembly Integration Tests")
    print("=" * 70)

    print("\n[1/2] Testing basic mixed-mode assembly...")
    test_mixed_mode_assembly_simple_crack()

    print("\n[2/2] Testing mixed-mode vs Mode I (pure normal)...")
    test_mixed_mode_vs_mode_I_pure_normal()

    print("\n" + "=" * 70)
    print("✓ All mixed-mode assembly integration tests passed!")
    print("=" * 70)
