"""Pullout test with CORRECT boundary conditions."""

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
print("PULLOUT TEST - CORRECT BOUNDARY CONDITIONS")
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

print(f"Mesh: {len(nodes)} nodes, {len(elems)} elements")
print(f"DOFs: {dofs.ndof} (concrete: 0-{dofs.steel_dof_offset-1}, steel: {dofs.steel_dof_offset}-{dofs.ndof-1})")
print(f"s_reg: {0.1 * bond_law.s1 * 1e6:.1f} μm")

# CORRECT PULLOUT BCs:
# 1. Fix ALL concrete to prevent rigid body motion (specimen is fixed)
# 2. Fix left end of steel bar (anchor)
# 3. Pull right end of steel bar

fixed_base = {}

# Fix all concrete DOFs (specimen is clamped)
for n in range(len(nodes)):
    fixed_base[2*n] = 0.0      # concrete x
    fixed_base[2*n+1] = 0.0    # concrete y

# Fix left end of steel bar (anchor)
# Node 6 is leftmost steel node (x=0.0, y=0.025)
steel_left_node = 6
if dofs.steel[steel_left_node, 0] >= 0:
    fixed_base[dofs.steel[steel_left_node, 0]] = 0.0  # steel x
    fixed_base[dofs.steel[steel_left_node, 1]] = 0.0  # steel y
    print(f"\nAnchor: Steel node {steel_left_node} at ({nodes[steel_left_node,0]:.3f},{nodes[steel_left_node,1]:.3f})")
    print(f"  Steel DOFs {dofs.steel[steel_left_node, 0]}, {dofs.steel[steel_left_node, 1]} fixed at 0")

# Pull right end of steel bar in x-direction
# Node 11 is rightmost steel node (x=0.10, y=0.025)
steel_right_node = 11
load_dof = dofs.steel[steel_right_node, 0]  # steel x DOF
print(f"\nLoad: Steel node {steel_right_node} at ({nodes[steel_right_node,0]:.3f},{nodes[steel_right_node,1]:.3f})")
print(f"  Steel DOF {load_dof} will be pulled")
print(f"  Steel DOF {dofs.steel[steel_right_node, 1]} fixed at 0 (no lateral motion)")
fixed_base[dofs.steel[steel_right_node, 1]] = 0.0  # no lateral motion

# Very small displacement steps
nsteps = 10
u_max = 100e-6  # 100 μm (= s_reg)

q = np.zeros(dofs.ndof)
displacements = []
reactions = []
slips = []

print(f"\nRunning {nsteps} steps (u_max={u_max*1e6:.0f} μm)...\n")

for step in range(1, nsteps+1):
    u = (step / nsteps) * u_max

    fixed = fixed_base.copy()
    fixed[load_dof] = u

    # Newton loop
    for it in range(20):
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

        if norm_r < 1e-3:  # More reasonable tolerance
            break

        K_scaled, r_scaled, D_inv = diagonal_equilibration(K_ff, rhs)
        du_scaled = spla.spsolve(K_scaled, r_scaled)
        du = unscale_solution(du_scaled, D_inv)

        q[free] += du
        for d, v in fixed.items():
            q[d] = v

    bond_states = bond_new
    reaction = -r[load_dof]

    displacements.append(u)
    reactions.append(reaction)
    slips.append(bond_states.s_max.max())

    print(f"Step {step:2d}: u={u*1e6:6.1f} μm, P={reaction/1e3:8.3f} kN, "
          f"s_max={bond_states.s_max.max()*1e6:6.2f} μm, "
          f"iter={it+1:2d}, ||r||={norm_r:.2e}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"✓ Completed {len(displacements)} steps")
print(f"  Max displacement: {max(displacements)*1e6:.1f} μm")
print(f"  Max load: {max(reactions)/1e3:.3f} kN")
print(f"  Max slip: {max(slips)*1e6:.2f} μm")
print("="*70)
