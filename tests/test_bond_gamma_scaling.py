"""Tests for bond_gamma scaling of bond-slip forces and stiffness."""

import numpy as np
import pytest

from xfem_clean.bond_slip import (
    assemble_bond_slip,
    BondSlipModelCode2010,
    BondSlipStateArrays,
)


def _setup_basic_bond_case():
    n_seg = 1
    steel_segments = np.array([[0, 1, 1.0, 1.0, 0.0]], dtype=float)
    steel_dof_offset = 4
    steel_dof_map = np.array([[4, 5], [6, 7]], dtype=np.int64)
    ndof_total = 8

    u_total = np.zeros(ndof_total, dtype=float)
    u_total[4] = 1.0e-3
    u_total[6] = 1.0e-3

    bond_law = BondSlipModelCode2010(f_cm=30e6, d_bar=0.016, condition="good")
    bond_states = BondSlipStateArrays.zeros(n_seg)

    return steel_segments, steel_dof_offset, steel_dof_map, u_total, bond_law, bond_states


@pytest.mark.parametrize("use_numba", [False, True])
def test_bond_gamma_zero_scales_force_and_stiffness(use_numba):
    if use_numba:
        try:
            from xfem_clean.numba.kernels_bond_slip import bond_slip_assembly_kernel
        except Exception:
            pytest.skip("Numba not available")

    (
        steel_segments,
        steel_dof_offset,
        steel_dof_map,
        u_total,
        bond_law,
        bond_states,
    ) = _setup_basic_bond_case()

    f_bond, K_bond, _ = assemble_bond_slip(
        u_total=u_total,
        steel_segments=steel_segments,
        steel_dof_offset=steel_dof_offset,
        bond_law=bond_law,
        bond_states=bond_states,
        steel_dof_map=steel_dof_map,
        steel_EA=0.0,
        use_numba=use_numba,
        bond_gamma=0.0,
    )

    np.testing.assert_allclose(
        f_bond,
        0.0,
        atol=1e-12,
        err_msg="bond_gamma=0 should zero bond force",
    )
    np.testing.assert_allclose(
        K_bond.toarray(),
        0.0,
        atol=1e-12,
        err_msg="bond_gamma=0 should zero bond stiffness",
    )


@pytest.mark.parametrize("use_numba", [False, True])
def test_bond_gamma_scales_linearly(use_numba):
    if use_numba:
        try:
            from xfem_clean.numba.kernels_bond_slip import bond_slip_assembly_kernel
        except Exception:
            pytest.skip("Numba not available")

    (
        steel_segments,
        steel_dof_offset,
        steel_dof_map,
        u_total,
        bond_law,
        bond_states,
    ) = _setup_basic_bond_case()

    f_ref, K_ref, _ = assemble_bond_slip(
        u_total=u_total,
        steel_segments=steel_segments,
        steel_dof_offset=steel_dof_offset,
        bond_law=bond_law,
        bond_states=bond_states,
        steel_dof_map=steel_dof_map,
        steel_EA=0.0,
        use_numba=use_numba,
        bond_gamma=1.0,
    )

    bond_states_scaled = BondSlipStateArrays.zeros(steel_segments.shape[0])
    gamma = 0.3
    f_scaled, K_scaled, _ = assemble_bond_slip(
        u_total=u_total,
        steel_segments=steel_segments,
        steel_dof_offset=steel_dof_offset,
        bond_law=bond_law,
        bond_states=bond_states_scaled,
        steel_dof_map=steel_dof_map,
        steel_EA=0.0,
        use_numba=use_numba,
        bond_gamma=gamma,
    )

    np.testing.assert_allclose(
        f_scaled,
        gamma * f_ref,
        rtol=1e-10,
        atol=1e-12,
        err_msg="bond force should scale linearly with gamma",
    )
    np.testing.assert_allclose(
        K_scaled.toarray(),
        gamma * K_ref.toarray(),
        rtol=1e-10,
        atol=1e-12,
        err_msg="bond stiffness should scale linearly with gamma",
    )
