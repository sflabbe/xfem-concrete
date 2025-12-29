"""Check if bond-slip returns zero forces at zero displacement."""

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

# Simple setup
L, H = 0.20, 0.10
nx, ny = 10, 5
nodes, elems = structured_quad_mesh(L, H, nx, ny)
nnode = nodes.shape[0]
rebar_segs = prepare_rebar_segments(nodes, cover=0.05)

# Build DOFs
crack = XFEMCrack(x0=L/2, y0=0.0, tip_x=L/2, tip_y=0.0, stop_y=H, active=False)
dofs = build_xfem_dofs(
    nodes, elems, crack,
    H_region_ymax=0.0,
    tip_patch=(0,0,0,0),
    rebar_segs=rebar_segs,
    enable_bond_slip=True
)

# Material
E = 30e9
nu = 0.2
C = plane_stress_C(E, nu)
steel_EA = 200e9 * np.pi * (0.012/2)**2

# Laws
law = CohesiveLaw(Kn=0.1*E/0.05, ft=3e6, Gf=100)
bond_law = BondSlipModelCode2010(f_cm=30e6, d_bar=0.012, condition="good")
bond_states = BondSlipStateArrays.zeros(len(rebar_segs))

# Apply boundary conditions (left edge fixed, right edge with tiny displacement)
fixed = {}
load_dofs = []
for a in range(nnode):
    x, y = nodes[a, 0], nodes[a, 1]
    if abs(x) < 1e-9:  # Left edge
        for comp in [0, 1]:
            d_std = int(dofs.std[a, comp])
            fixed[d_std] = 0.0
            # Fix steel DOFs too
            if dofs.steel is not None:
                d_steel = int(dofs.steel[a, comp])
                if d_steel >= 0:
                    fixed[d_steel] = 0.0
    if abs(x - L) < 1e-9 and abs(y - H) < 1e-9:  # Top-right corner
        load_dofs.append(int(dofs.std[a, 0]))  # x-displacement

# Test at u=0
q = np.zeros(dofs.ndof)
for dof in load_dofs:
    fixed[int(dof)] = 0.0  # No displacement yet

for dof, val in fixed.items():
    q[int(dof)] = float(val)

print(f"DOFs: {dofs.ndof}")
print(f"Fixed DOFs: {len(fixed)}")
print(f"Load DOFs: {load_dofs}")

# Assemble
K, fint, _, _, _, _ = assemble_xfem_system(
    nodes=nodes,
    elems=elems,
    dofs=dofs,
    crack=crack,
    C=C,
    thickness=0.10,
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

print(f"\nAt u=0 (all DOFs zero):")
print(f"  fint norm: {np.linalg.norm(fint):.3e}")
print(f"  fint max: {np.max(np.abs(fint)):.3e}")

# Apply BCs to get residual
fext = np.zeros(dofs.ndof)
residual = fext - fint
free = np.ones(dofs.ndof, dtype=bool)
for dof in fixed.keys():
    free[int(dof)] = False

r_free = residual[free]
print(f"  residual (free DOFs) norm: {np.linalg.norm(r_free):.3e}")
print(f"  residual (free DOFs) max: {np.max(np.abs(r_free)):.3e}")

# Check which DOFs have non-zero residual
if np.max(np.abs(r_free)) > 1e-10:
    free_dofs = np.where(free)[0]
    large_residual = np.where(np.abs(residual[free_dofs]) > 1e-10)[0]
    print(f"\n  Non-zero residual DOFs: {len(large_residual)}")
    for idx in large_residual[:10]:
        dof = free_dofs[idx]
        val = residual[dof]
        if dof < 2*nnode:
            node = dof // 2
            comp = dof % 2
            print(f"    DOF {dof} (concrete node {node}, comp {comp}): residual = {val:.3e}")
        elif dof >= dofs.steel_dof_offset:
            steel_idx = dof - dofs.steel_dof_offset
            node = steel_idx // 2
            comp = steel_idx % 2
            print(f"    DOF {dof} (steel node {node}, comp {comp}): residual = {val:.3e}")
