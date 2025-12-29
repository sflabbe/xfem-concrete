"""Minimal working pullout test - ultra-conservative parameters."""

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
print("MINIMAL WORKING PULLOUT TEST")
print("="*70)

# Very simple geometry
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
steel_EA_min = 1e6  # INCREASED from 1e3 to improve conditioning

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

print(f"Mesh: {len(nodes)} nodes, {len(elems)} elements")
print(f"DOFs: {dofs.ndof}")
print(f"s_reg: {0.1 * bond_law.s1 * 1e6:.1f} μm")

# BCs: Fix left, pull right
left_nodes = np.where(nodes[:, 0] < 1e-6)[0]
right_nodes = np.where(nodes[:, 0] > L - 1e-6)[0]

fixed_base = {}
for n in left_nodes:
    fixed_base[2*n] = 0.0
    fixed_base[2*n+1] = 0.0
    if dofs.steel[n, 0] >= 0:
        fixed_base[dofs.steel[n, 0]] = 0.0
        fixed_base[dofs.steel[n, 1]] = 0.0

load_dofs = [2*n for n in right_nodes]

# VERY SMALL displacement steps
nsteps = 5
u_max = 50e-6  # 50 μm (half of s_reg!)

q = np.zeros(dofs.ndof)
displacements = []
reactions = []

print(f"\nRunning {nsteps} steps (u_max={u_max*1e6:.0f} μm)...\n")

for step in range(1, nsteps+1):
    u = (step / nsteps) * u_max

    fixed = fixed_base.copy()
    for dof in load_dofs:
        fixed[dof] = u

    # Simple Newton loop
    for it in range(10):
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

        r = fint.copy()
        free, K_ff, r_f, _ = apply_dirichlet(K, r, fixed, q)
        rhs = -r_f
        norm_r = np.linalg.norm(rhs)

        if norm_r < 1e-1:  # Very relaxed tolerance
            break

        K_scaled, r_scaled, D_inv = diagonal_equilibration(K_ff, rhs)
        du_scaled = spla.spsolve(K_scaled, r_scaled)
        du = unscale_solution(du_scaled, D_inv)

        q[free] += du
        for d, v in fixed.items():
            q[d] = v

    bond_states = bond_new
    reaction = -sum(r[d] for d in load_dofs)

    displacements.append(u)
    reactions.append(reaction)

    print(f"Step {step}: u={u*1e6:5.1f} μm, P={reaction/1e3:7.3f} kN, "
          f"s_max={bond_states.s_max.max()*1e6:5.2f} μm, "
          f"iter={it+1}, ||r||={norm_r:.2e}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"✓ Completed {len(displacements)} steps")
print(f"  Max displacement: {max(displacements)*1e6:.1f} μm")
print(f"  Max load: {max(reactions)/1e3:.3f} kN")
print(f"  Max slip: {bond_states.s_max.max()*1e6:.2f} μm")
print("="*70)
