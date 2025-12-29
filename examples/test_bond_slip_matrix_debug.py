"""Debug which DOFs have zero diagonal in bond-slip matrix."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xfem_clean.fem.mesh import structured_quad_mesh
from xfem_clean.rebar import prepare_rebar_segments
from xfem_clean.xfem.dofs_single import build_xfem_dofs
from xfem_clean.xfem.geometry import XFEMCrack
from xfem_clean.bond_slip import assemble_bond_slip, BondSlipModelCode2010, BondSlipStateArrays

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
print(f"  Concrete: {2*nnode}")
print(f"  Steel offset: {dofs.steel_dof_offset}")
print(f"  Steel nodes: {np.sum(dofs.steel_nodes)}")

# Steel properties
steel_E = 200e9  # [Pa]
d_bar = 0.012  # [m]
A_bar = np.pi * (d_bar / 2) ** 2
steel_EA = steel_E * A_bar

# Initialize bond-slip
n_seg = rebar_segs.shape[0]
bond_states = BondSlipStateArrays.zeros(n_seg)
bond_law = BondSlipModelCode2010(f_cm=30e6, d_bar=0.012, condition="good")

# Zero displacement
q = np.zeros(dofs.ndof)

print(f"\nAssembling bond-slip with steel_EA={steel_EA:.3e}...")
f_bond, K_bond, bond_states_new = assemble_bond_slip(
    u_total=q,
    steel_segments=rebar_segs,
    steel_dof_offset=dofs.steel_dof_offset,
    bond_law=bond_law,
    bond_states=bond_states,
    steel_dof_map=dofs.steel,
    steel_EA=steel_EA,
    use_numba=True,
)

print(f"✓ Assembly succeeded")
print(f"  K_bond nnz: {K_bond.nnz}")

# Check diagonal
diag = K_bond.diagonal()
zero_diag = np.where(diag == 0.0)[0]
nonzero_diag = np.where(diag != 0.0)[0]

print(f"\nDiagonal analysis:")
print(f"  Zero diagonal entries: {len(zero_diag)}")
print(f"  Nonzero diagonal entries: {len(nonzero_diag)}")

# Categorize zero diagonal DOFs
concrete_dofs = 2 * nnode
if len(zero_diag) > 0:
    print(f"\nZero diagonal DOFs:")
    for dof in zero_diag[:20]:  # Show first 20
        if dof < concrete_dofs:
            node = dof // 2
            comp = dof % 2
            print(f"  DOF {dof}: Concrete node {node}, comp {comp}")
        elif dof >= dofs.steel_dof_offset:
            steel_idx = dof - dofs.steel_dof_offset
            steel_node = steel_idx // 2
            comp = steel_idx % 2
            print(f"  DOF {dof}: Steel node {steel_node}, comp {comp}")
        else:
            print(f"  DOF {dof}: Enrichment")

# Check which steel nodes have stiffness
if dofs.steel is not None:
    print(f"\nSteel DOF mapping:")
    steel_nodes_with_stiffness = set()
    for dof in nonzero_diag:
        if dof >= dofs.steel_dof_offset:
            steel_idx = dof - dofs.steel_dof_offset
            steel_node = steel_idx // 2
            steel_nodes_with_stiffness.add(steel_node)

    total_steel_nodes = np.sum(dofs.steel_nodes)
    print(f"  Total steel nodes: {total_steel_nodes}")
    print(f"  Steel nodes with stiffness: {len(steel_nodes_with_stiffness)}")

    if len(steel_nodes_with_stiffness) < total_steel_nodes:
        print(f"  Missing stiffness for {total_steel_nodes - len(steel_nodes_with_stiffness)} steel nodes")
