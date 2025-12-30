"""Minimal bond-slip test with diagonal scaling."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xfem_clean.fem.mesh import structured_quad_mesh
from xfem_clean.rebar import prepare_rebar_segments
from xfem_clean.xfem.dofs_single import build_xfem_dofs
from xfem_clean.xfem.geometry import XFEMCrack
from xfem_clean.xfem.assembly_single import assemble_xfem_system
from xfem_clean.xfem.material import plane_stress_C
from xfem_clean.cohesive_laws import CohesiveLaw
from xfem_clean.bond_slip import BondSlipModelCode2010, BondSlipStateArrays
from xfem_clean.fem.bcs import apply_dirichlet


def main():
    """Run minimal bond-slip test with diagonal scaling."""
    print("="*70)
    print("MINIMAL BOND-SLIP TEST WITH DIAGONAL SCALING")
    print("="*70)

    # Simple mesh
    L, H = 0.10, 0.05
    nx, ny = 5, 2
    nodes, elems = structured_quad_mesh(L, H, nx, ny)
    nnode = nodes.shape[0]

    # Rebar
    rebar_segs = prepare_rebar_segments(nodes, cover=0.025)
    print(f"\nMesh: {nnode} nodes, {len(elems)} elements, {len(rebar_segs)} rebar segments")

    # DOFs
    crack = XFEMCrack(x0=L/2, y0=0.0, tip_x=L/2, tip_y=0.0, stop_y=H, active=False)
    dofs = build_xfem_dofs(
        nodes, elems, crack,
        H_region_ymax=0.0,
        tip_patch=(0,0,0,0),
        rebar_segs=rebar_segs,
        enable_bond_slip=True
    )
    print(f"DOFs: total={dofs.ndof}, steel_offset={dofs.steel_dof_offset}")

    # Material
    E = 30e9
    nu = 0.2
    C = plane_stress_C(E, nu)
    thickness = 0.10

    steel_E = 200e9
    d_bar = 0.012
    A_bar = np.pi * (d_bar / 2) ** 2
    steel_EA_min = 1e3  # Minimum stiffness

    ft = 3e6
    Gf = 100
    Kn = 0.1 * E / 0.05
    law = CohesiveLaw(Kn=Kn, ft=ft, Gf=Gf)

    bond_law = BondSlipModelCode2010(f_cm=30e6, d_bar=d_bar, condition="good")
    bond_states = BondSlipStateArrays.zeros(len(rebar_segs))

    print(f"\nBond law parameters:")
    print(f"  tau_max = {bond_law.tau_max/1e6:.2f} MPa")
    print(f"  s1 = {bond_law.s1*1e3:.3f} mm")
    print(f"  Steel EA_min = {steel_EA_min:.1e} N")

    # Initial assembly at zero displacement
    q = np.zeros(dofs.ndof)

    print(f"\n1) Assembly at u=0...")
    K0, fint0, _, _, _, _, _, _ = assemble_xfem_system(
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
        steel_EA=steel_EA_min,
        use_numba=False,  # Lighter for manual execution
    )

    diag0 = K0.diagonal()
    print(f"   K0: shape={K0.shape}, nnz={K0.nnz}")
    print(f"   diag range: [{np.min(np.abs(diag0)):.2e}, {np.max(np.abs(diag0)):.2e}]")
    print(f"   fint norm: {np.linalg.norm(fint0):.2e}")

    # Apply small displacement (1 micron pullout)
    u_apply = 1e-6  # 1 μm
    load_dof = dofs.steel_dof_offset  # First steel DOF (x-direction at left end)
    fixed = {load_dof: u_apply}

    # Apply BCs
    r0 = fint0.copy()
    free, K0_ff, r0_f, _ = apply_dirichlet(K0, r0, fixed, q)

    print(f"\n2) After BC application:")
    print(f"   K0_ff: shape={K0_ff.shape}")
    print(f"   residual norm: {np.linalg.norm(r0_f):.2e}")

    # Test diagonal scaling
    from xfem_clean.utils.scaling import diagonal_equilibration, unscale_solution, check_conditioning_improvement

    print(f"\n3) Testing diagonal scaling...")
    K_scaled, r_scaled, D_inv = diagonal_equilibration(K0_ff, -r0_f)

    info = check_conditioning_improvement(K0_ff, K_scaled)
    print(f"   Original diag ratio: {info['diag_ratio_original']:.2e}")
    print(f"   Scaled diag ratio:   {info['diag_ratio_scaled']:.2e}")
    print(f"   Improvement factor:  {info['diag_ratio_original'] / info['diag_ratio_scaled']:.2e}×")

    # Try to solve
    import scipy.sparse.linalg as spla

    print(f"\n4) Solving system...")
    try:
        du_scaled = spla.spsolve(K_scaled, r_scaled)
        du = unscale_solution(du_scaled, D_inv)
        print(f"   ✓ Solution successful")
        print(f"   ||du|| = {np.linalg.norm(du):.3e}")

        # Check residual
        res = K0_ff @ du - (-r0_f)
        print(f"   ||K du - r|| = {np.linalg.norm(res):.3e}")

    except Exception as e:
        print(f"   ✗ Solution failed: {e}")

    print("="*70)
    print("✓ Minimal test completed")
    print("="*70)


if __name__ == "__main__":
    main()
