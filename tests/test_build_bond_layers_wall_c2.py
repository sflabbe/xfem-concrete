"""Regression test for wall C2 bond layer builder."""

import os
import sys

import numpy as np

# Add src and examples to path
repo_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(repo_root, 'src'))
sys.path.insert(0, repo_root)


def test_build_bond_layers_wall_c2():
    """Ensure wall C2 bond layers build with positive segment counts."""
    from examples.gutierrez_thesis.cases.case_10_wall_c2_cyclic import create_case_10
    from examples.gutierrez_thesis.solver_interface import build_bond_layers_from_case
    from xfem_clean.fem.mesh import structured_quad_mesh

    case = create_case_10()

    # Build mesh in SI units
    L = case.geometry.length * 1e-3
    H = case.geometry.height * 1e-3
    nodes, _ = structured_quad_mesh(L, H, case.geometry.n_elem_x, case.geometry.n_elem_y)

    bond_layers = build_bond_layers_from_case(case, nodes)

    assert len(bond_layers) == len(case.rebar_layers)
    assert len(bond_layers) == 3

    for layer in bond_layers:
        assert layer.segments.shape[0] > 0
        assert np.all(layer.segments[:, 2] > 0.0)

    layer_ids = [layer.layer_id for layer in bond_layers]
    assert "rebar_layer_2_orient90deg" in layer_ids
