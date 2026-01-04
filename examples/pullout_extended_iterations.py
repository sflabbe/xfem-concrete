"""Pullout test with extended Newton iterations to test convergence."""

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
from xfem_clean.utils.scaling import diagonal_equilibration, unscale_solution
import scipy.sparse.linalg as spla

print("="*70)
print("PULLOUT TEST - EXTENDED ITERATIONS")
print("="*70)

# Geometry
L, H = 0.10, 0.05
nx, ny = 5, 2

# Material
E = 30e9
nu = 0.2
fc = 30e6
ft = 3e6
Gf = 100
thickness = 0.10

# Steel
d_bar = 0.012
steel_EA_min = 1e6

# Setup
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

C = plane_stress_C(E, nu)
law = CohesiveLaw(Kn=0.1*E/0.05, ft=ft, Gf=Gf)
bond_law = BondSlipModelCode2010(f_cm=fc, d_bar=d_bar, condition="good")
bond_states = BondSlipStateArrays.zeros(len(rebar_segs))

s_reg = 0.5 * bond_law.s1
print(f"Mesh: {len(nodes)} nodes, {len(elems)} elements")
print(f"DOFs: {dofs.ndof}")
print(f"s_reg: {s_reg * 1e6:.1f} μm")

# Correct pullout BCs
fixed_base = {}

# Fix all concrete
for n in range(len(nodes)):
    fixed_base[2*n] = 0.0
    fixed_base[2*n+1] = 0.0

# Anchor left steel
steel_left_node = 6
fixed_base[dofs.steel[steel_left_node, 0]] = 0.0
fixed_base[dofs.steel[steel_left_node, 1]] = 0.0

# Load right steel
steel_right_node = 11
load_dof = dofs.steel[steel_right_node, 0]
fixed_base[dofs.steel[steel_right_node, 1]] = 0.0

# Single test step at 50 μm
u_test = 50e-6

q = np.zeros(dofs.ndof)
fixed = fixed_base.copy()
fixed[load_dof] = u_test

print(f"\nTesting single step: u = {u_test*1e6:.1f} μm")
print(f"Max iterations: 100")
print(f"Convergence tolerance: ||r|| < 1e-3 N\n")

# Extended Newton loop
for it in range(100):
    K, fint, _, _, _, bond_new, _, _ = assemble_xfem_system(
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

    r = fint.copy()
    free, K_ff, r_f, _ = apply_dirichlet(K, r, fixed, q)
    rhs = -r_f
    norm_r = np.linalg.norm(rhs)

    # Check convergence before computing increment
    s_max_current = bond_new.s_max.max() if it > 0 else 0.0

    if it > 0 and norm_r < 1e-3:
        print(f"  Iter {it:3d}: ||r|| = {norm_r:.3e} N, s_max = {s_max_current*1e6:.2f} μm")
        print(f"\n✓ Converged at iteration {it}")
        break

    # Solve for increment
    K_scaled, r_scaled, D_inv = diagonal_equilibration(K_ff, rhs)
    du_scaled = spla.spsolve(K_scaled, r_scaled)
    du = unscale_solution(du_scaled, D_inv)
    norm_du = np.linalg.norm(du)

    # Print iteration info
    print(f"  Iter {it:3d}: ||r|| = {norm_r:.3e} N, ||du|| = {norm_du:.3e} m, s_max = {s_max_current*1e6:.2f} μm")

    q[free] += du
    for d, v in fixed.items():
        q[d] = v

    bond_states = bond_new

else:
    print(f"\n✗ Did not converge in 100 iterations")
    print(f"Final ||r|| = {norm_r:.3e} N")
    print(f"Final ||du|| = {norm_du:.3e} m")

reaction = -fint[load_dof]
print(f"\nFinal state:")
print(f"  Displacement: {u_test*1e6:.1f} μm")
print(f"  Reaction: {reaction/1e3:.3f} kN")
print(f"  Max slip: {bond_states.s_max.max()*1e6:.2f} μm")

print("="*70)
