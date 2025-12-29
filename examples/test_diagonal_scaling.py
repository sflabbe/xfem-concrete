"""Test diagonal scaling/equilibration for ill-conditioned systems."""

import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xfem_clean.utils.scaling import (
    diagonal_equilibration,
    unscale_solution,
    check_conditioning_improvement,
)


def test_simple_system():
    """Test diagonal scaling on a simple ill-conditioned system."""

    print("=" * 70)
    print("TEST: Diagonal Scaling on Simple Ill-Conditioned System")
    print("=" * 70)

    # Create a simple ill-conditioned system
    # K has diagonal entries spanning many orders of magnitude
    n = 100
    diag_vals = np.logspace(0, 9, n)  # 1 to 1e9
    K = sp.diags(diag_vals, format='csr')

    # Add some off-diagonal coupling
    K = K + sp.diags(0.1 * diag_vals[:-1], offsets=1, shape=(n, n), format='csr')
    K = K + sp.diags(0.1 * diag_vals[1:], offsets=-1, shape=(n, n), format='csr')
    K = K.tocsr()

    # RHS
    rhs = np.random.randn(n)

    print(f"\nOriginal system:")
    print(f"  Matrix size: {K.shape}")
    print(f"  Diagonal range: [{np.min(diag_vals):.3e}, {np.max(diag_vals):.3e}]")
    print(f"  Diagonal ratio: {np.max(diag_vals) / np.min(diag_vals):.3e}")

    # Solve without scaling
    print(f"\n1) Solving WITHOUT diagonal scaling...")
    try:
        x_orig = spla.spsolve(K, rhs)
        residual_orig = np.linalg.norm(K @ x_orig - rhs)
        print(f"   ✓ Success: ||K x - rhs|| = {residual_orig:.3e}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        x_orig = None
        residual_orig = np.inf

    # Solve with scaling
    print(f"\n2) Solving WITH diagonal scaling...")
    K_scaled, rhs_scaled, D_inv = diagonal_equilibration(K, rhs)

    # Check conditioning improvement
    info = check_conditioning_improvement(K, K_scaled)
    print(f"   Scaled diagonal range: [{info['diag_range_scaled'][0]:.3e}, {info['diag_range_scaled'][1]:.3e}]")
    print(f"   Scaled diagonal ratio: {info['diag_ratio_scaled']:.3e}")
    print(f"   Improvement factor: {info['diag_ratio_original'] / info['diag_ratio_scaled']:.3e}×")

    try:
        x_scaled_tilde = spla.spsolve(K_scaled, rhs_scaled)
        x_scaled = unscale_solution(x_scaled_tilde, D_inv)
        residual_scaled = np.linalg.norm(K @ x_scaled - rhs)
        print(f"   ✓ Success: ||K x - rhs|| = {residual_scaled:.3e}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        x_scaled = None
        residual_scaled = np.inf

    # Compare solutions
    if x_orig is not None and x_scaled is not None:
        diff = np.linalg.norm(x_orig - x_scaled) / np.linalg.norm(x_orig)
        print(f"\n3) Solution comparison:")
        print(f"   ||x_scaled - x_orig|| / ||x_orig|| = {diff:.3e}")
        if diff < 1e-6:
            print(f"   ✓ Solutions match to machine precision")
        else:
            print(f"   ⚠️  Solutions differ (may indicate numerical issues)")

    print("=" * 70)


def test_realistic_xfem_system():
    """Test diagonal scaling on a realistic XFEM-like system."""

    print("\n" * 2)
    print("=" * 70)
    print("TEST: Diagonal Scaling on Realistic XFEM System")
    print("=" * 70)

    from xfem_clean.fem.mesh import structured_quad_mesh
    from xfem_clean.rebar import prepare_rebar_segments
    from xfem_clean.xfem.dofs_single import build_xfem_dofs
    from xfem_clean.xfem.geometry import XFEMCrack
    from xfem_clean.xfem.assembly_single import assemble_xfem_system
    from xfem_clean.xfem.material import plane_stress_C
    from xfem_clean.cohesive_laws import CohesiveLaw
    from xfem_clean.bond_slip import BondSlipModelCode2010, BondSlipStateArrays

    # Setup (same as test_full_assembly_debug.py)
    L, H = 0.20, 0.10
    nx, ny = 10, 5
    nodes, elems = structured_quad_mesh(L, H, nx, ny)
    nnode = nodes.shape[0]

    rebar_segs = prepare_rebar_segments(nodes, cover=0.05)
    crack = XFEMCrack(x0=L/2, y0=0.0, tip_x=L/2, tip_y=0.0, stop_y=H, active=False)
    dofs = build_xfem_dofs(
        nodes, elems, crack,
        H_region_ymax=0.0,
        tip_patch=(0,0,0,0),
        rebar_segs=rebar_segs,
        enable_bond_slip=True
    )

    E = 30e9
    nu = 0.2
    C = plane_stress_C(E, nu)
    thickness = 0.10

    steel_E = 200e9
    d_bar = 0.012
    A_bar = np.pi * (d_bar / 2) ** 2
    steel_EA = steel_E * A_bar

    ft = 3e6
    Gf = 100
    Kn = 0.1 * E / 0.05
    law = CohesiveLaw(Kn=Kn, ft=ft, Gf=Gf)

    bond_law = BondSlipModelCode2010(f_cm=30e6, d_bar=d_bar, condition="good")
    bond_states = BondSlipStateArrays.zeros(len(rebar_segs))

    q = np.zeros(dofs.ndof)

    print(f"\nAssembling system...")
    K, fint, _, _, _, _ = assemble_xfem_system(
        nodes=nodes,
        elems=elems,
        dofs=dofs,
        crack=crack,
        C=C,
        thickness=thickness,
        q=q,
        law=law,
        coh_states_comm={},
        tip_enr_radius=0.01,
        rebar_segs=rebar_segs,
        bond_law=bond_law,
        bond_states_comm=bond_states,
        enable_bond_slip=True,
        steel_EA=steel_EA,
        use_numba=True,
    )

    print(f"  K shape: {K.shape}")
    print(f"  K nnz: {K.nnz}")

    # Check original diagonal
    diag_orig = K.diagonal()
    print(f"\nOriginal system:")
    print(f"  Diagonal range: [{np.min(np.abs(diag_orig)):.3e}, {np.max(np.abs(diag_orig)):.3e}]")
    print(f"  Diagonal ratio: {np.max(np.abs(diag_orig)) / np.max([np.min(np.abs(diag_orig)), 1e-16]):.3e}")

    # Apply diagonal scaling
    rhs = -fint  # Residual
    K_scaled, rhs_scaled, D_inv = diagonal_equilibration(K, rhs)

    # Check scaled diagonal
    info = check_conditioning_improvement(K, K_scaled)
    print(f"\nScaled system:")
    print(f"  Diagonal range: [{info['diag_range_scaled'][0]:.3e}, {info['diag_range_scaled'][1]:.3e}]")
    print(f"  Diagonal ratio: {info['diag_ratio_scaled']:.3e}")
    print(f"\nConditioning improvement:")
    print(f"  Original ratio: {info['diag_ratio_original']:.3e}")
    print(f"  Scaled ratio:   {info['diag_ratio_scaled']:.3e}")
    print(f"  Improvement:    {info['diag_ratio_original'] / info['diag_ratio_scaled']:.3e}×")

    if info['diag_ratio_scaled'] < 10:
        print(f"\n  ✓ Scaled system is well-conditioned (ratio < 10)")
    elif info['diag_ratio_scaled'] < 1000:
        print(f"\n  ✓ Scaled system has good conditioning (ratio < 1000)")
    else:
        print(f"\n  ⚠️  Scaled system still poorly conditioned (ratio = {info['diag_ratio_scaled']:.3e})")

    print("=" * 70)


if __name__ == "__main__":
    test_simple_system()
    test_realistic_xfem_system()
    print("\n✓ All tests completed\n")
