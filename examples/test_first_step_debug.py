"""Debug the first load step in detail."""

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
import scipy.sparse.linalg as spla

print("="*70)
print("FIRST LOAD STEP DEBUG")
print("="*70)

# Simple geometry
L, H = 0.10, 0.05
nx, ny = 5, 2
nodes, elems = structured_quad_mesh(L, H, nx, ny)

rebar_segs = prepare_rebar_segments(nodes, cover=0.025)
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

steel_EA_min = 1e3

ft = 3e6
Gf = 100
Kn = 0.1 * E / 0.05
law = CohesiveLaw(Kn=Kn, ft=ft, Gf=Gf)

bond_law = BondSlipModelCode2010(f_cm=30e6, d_bar=0.012, condition="good")
bond_states = BondSlipStateArrays.zeros(len(rebar_segs))

print(f"DOFs: {dofs.ndof}, Steel offset: {dofs.steel_dof_offset}")
print(f"Bond s_reg: {0.1 * bond_law.s1 * 1e6:.1f} μm")

# Load step: 0.1 μm pullout
u_step = 0.1e-6  # [m]
load_dof = dofs.steel_dof_offset

q = np.zeros(dofs.ndof)
fixed = {load_dof: u_step}

print(f"\n--- NEWTON ITERATION 1 ---")
print(f"Target displacement: {u_step*1e6:.2f} μm at DOF {load_dof}")

# Assembly at u=0
K, fint, _, _, _, bond_new = assemble_xfem_system(
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
    use_numba=True,
)

print(f"\nAssembly complete:")
print(f"  K: shape={K.shape}, nnz={K.nnz}")
print(f"  fint norm: {np.linalg.norm(fint):.2e}")
print(f"  Bond states: max slip = {bond_new.s_max.max()*1e6:.3f} μm")

# Apply BCs
r = fint.copy()
free, K_ff, r_f, _ = apply_dirichlet(K, r, fixed, q)
rhs = -r_f

print(f"\nAfter BC:")
print(f"  Free DOFs: {len(free)}")
print(f"  ||rhs||: {np.linalg.norm(rhs):.2e}")

# Diagonal scaling
from xfem_clean.utils.scaling import diagonal_equilibration, unscale_solution

K_ff_scaled, rhs_scaled, D_inv = diagonal_equilibration(K_ff, rhs)

diag_orig = K_ff.diagonal()
diag_scaled = K_ff_scaled.diagonal()

print(f"\nDiagonal scaling:")
print(f"  Original: min={np.min(np.abs(diag_orig)):.2e}, max={np.max(np.abs(diag_orig)):.2e}")
print(f"  Scaled:   min={np.min(np.abs(diag_scaled)):.2e}, max={np.max(np.abs(diag_scaled)):.2e}")

# Solve
print(f"\nSolving...")
try:
    du_scaled = spla.spsolve(K_ff_scaled, rhs_scaled)
    du = unscale_solution(du_scaled, D_inv)
    print(f"  ✓ Solution successful")
    print(f"  ||du||: {np.linalg.norm(du):.2e}")

    # Update
    q_new = q.copy()
    q_new[free] += du
    q_new[load_dof] = u_step

    print(f"\n  Updated displacement:")
    print(f"    max|u|: {np.max(np.abs(q_new)):.2e} m")
    print(f"    u[load_dof]: {q_new[load_dof]*1e6:.3f} μm")

    # Check if we can assemble again
    print(f"\n--- NEWTON ITERATION 2 (to check residual) ---")
    K2, fint2, _, _, _, bond_new2 = assemble_xfem_system(
        nodes=nodes,
        elems=elems,
        dofs=dofs,
        crack=crack,
        C=C,
        thickness=thickness,
        q=q_new,
        law=law,
        coh_states_comm={},
        tip_enr_radius=0.01,
        rebar_segs=rebar_segs,
        bond_law=bond_law,
        bond_states_comm=bond_states,
        enable_bond_slip=True,
        steel_EA=steel_EA_min,
        use_numba=True,
    )

    print(f"  fint2 norm: {np.linalg.norm(fint2):.2e}")
    print(f"  Bond states: max slip = {bond_new2.s_max.max()*1e6:.3f} μm")

    r2 = fint2.copy()
    _, _, r2_f, _ = apply_dirichlet(K2, r2, fixed, q_new)
    print(f"  ||residual||: {np.linalg.norm(r2_f):.2e}")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print("="*70)
