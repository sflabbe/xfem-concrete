"""Regression test for steel DOF allocation across multiple bond layers."""

import numpy as np

from xfem_clean.fem.mesh import structured_quad_mesh
from xfem_clean.rebar import prepare_rebar_segments
from xfem_clean.xfem.dofs_single import build_xfem_dofs
from xfem_clean.xfem.geometry import XFEMCrack


def test_multilayer_rebar_allocates_steel_dofs_for_all_layers():
    L, H = 4.0, 0.4
    nx, ny = 40, 8
    nodes, elems = structured_quad_mesh(L, H, nx, ny)

    segs_bot = prepare_rebar_segments(nodes, cover=0.05)
    segs_top = prepare_rebar_segments(nodes, cover=0.35)
    segs_all = np.ascontiguousarray(np.vstack([segs_bot, segs_top]), dtype=float)

    crack = XFEMCrack(
        x0=0.0,
        y0=0.0,
        tip_x=0.0,
        tip_y=0.0,
        stop_y=0.0,
        active=False,
    )

    dofs = build_xfem_dofs(
        nodes=nodes,
        elems=elems,
        crack=crack,
        H_region_ymax=-1.0,
        enable_bond_slip=True,
        rebar_segs=segs_all,
    )

    steel_nodes = np.unique(segs_all[:, :2].astype(int))
    assert steel_nodes.size > 0
    assert np.all(dofs.steel[steel_nodes, 0] >= 0)
    assert np.all(dofs.steel[steel_nodes, 1] >= 0)

    nnode = nodes.shape[0]
    nsteel = steel_nodes.size
    assert dofs.ndof == 2 * nnode + 2 * nsteel
