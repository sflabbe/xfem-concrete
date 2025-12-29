"""Debug bond-slip assembly - check matrix properties."""

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

# Initialize bond-slip
n_seg = rebar_segs.shape[0]
bond_states = BondSlipStateArrays.zeros(n_seg)
bond_law = BondSlipModelCode2010(f_cm=30e6, d_bar=0.012, condition="good")

# Zero displacement
q = np.zeros(dofs.ndof)

print(f"\nAssembling bond-slip with q=0...")
try:
    f_bond, K_bond, bond_states_new = assemble_bond_slip(
        u_total=q,
        steel_segments=rebar_segs,
        steel_dof_offset=dofs.steel_dof_offset,
        bond_law=bond_law,
        bond_states=bond_states,
        steel_dof_map=dofs.steel,
        use_numba=True,
    )

    print(f"✓ Assembly succeeded")
    print(f"  f_bond norm: {np.linalg.norm(f_bond):.3e}")
    print(f"  K_bond shape: {K_bond.shape}")
    print(f"  K_bond nnz: {K_bond.nnz}")
    print(f"  K_bond diagonal sum: {K_bond.diagonal().sum():.3e}")
    print(f"  K_bond min diagonal: {K_bond.diagonal().min():.3e}")
    print(f"  K_bond max diagonal: {K_bond.diagonal().max():.3e}")

    # Check conditioning
    K_dense = K_bond.toarray()
    try:
        cond = np.linalg.cond(K_dense)
        print(f"  K_bond condition number: {cond:.3e}")
    except:
        print(f"  K_bond condition number: SINGULAR")

except Exception as e:
    print(f"✗ Assembly failed: {e}")
    import traceback
    traceback.print_exc()
