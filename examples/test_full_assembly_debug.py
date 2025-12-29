"""Debug full system assembly including concrete bulk."""

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

# Simple setup
L, H = 0.20, 0.10
nx, ny = 10, 5
nodes, elems = structured_quad_mesh(L, H, nx, ny)
nnode = nodes.shape[0]

# Rebar segments
rebar_segs = prepare_rebar_segments(nodes, cover=0.05)
print(f"Mesh: {nnode} nodes, {len(elems)} elements")
print(f"Rebar: {len(rebar_segs)} segments")

# Build DOFs with bond-slip
crack = XFEMCrack(x0=L/2, y0=0.0, tip_x=L/2, tip_y=0.0, stop_y=H, active=False)
dofs = build_xfem_dofs(
    nodes, elems, crack,
    H_region_ymax=0.0,
    tip_patch=(0,0,0,0),
    rebar_segs=rebar_segs,
    enable_bond_slip=True
)

print(f"\nDOF structure:")
print(f"  Total DOFs: {dofs.ndof}")
print(f"  Steel offset: {dofs.steel_dof_offset}")

# Material properties
E = 30e9  # [Pa]
nu = 0.2
C = plane_stress_C(E, nu)
thickness = 0.10  # [m]

# Steel properties
steel_E = 200e9  # [Pa]
d_bar = 0.012  # [m]
A_bar = np.pi * (d_bar / 2) ** 2
steel_EA = steel_E * A_bar

# Cohesive law (won't be used)
ft = 3e6  # [Pa]
Gf = 100  # [N/m]
Kn = 0.1 * E / 0.05
law = CohesiveLaw(Kn=Kn, ft=ft, Gf=Gf)

# Bond-slip
bond_law = BondSlipModelCode2010(f_cm=30e6, d_bar=d_bar, condition="good")
bond_states = BondSlipStateArrays.zeros(len(rebar_segs))

# Zero displacement
q = np.zeros(dofs.ndof)

print(f"\nAssembling full system at q=0...")
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

print(f"✓ Assembly succeeded")
print(f"  K shape: {K.shape}")
print(f"  K nnz: {K.nnz}")
print(f"  fint norm: {np.linalg.norm(fint):.3e}")

# Check diagonal
diag = K.diagonal()
zero_diag = np.where(np.abs(diag) < 1e-6)[0]
nonzero_diag = np.where(np.abs(diag) >= 1e-6)[0]

print(f"\nDiagonal analysis:")
print(f"  Near-zero diagonal entries: {len(zero_diag)}")
print(f"  Nonzero diagonal entries: {len(nonzero_diag)}")
print(f"  Min |diagonal|: {np.min(np.abs(diag)):.3e}")
print(f"  Max |diagonal|: {np.max(np.abs(diag)):.3e}")

if len(zero_diag) > 0:
    print(f"\nFirst 10 near-zero diagonal DOFs: {zero_diag[:10]}")

# Check conditioning
try:
    K_dense = K.toarray()
    cond = np.linalg.cond(K_dense)
    print(f"  Condition number: {cond:.3e}")
except:
    print(f"  Condition number: FAILED to compute")
