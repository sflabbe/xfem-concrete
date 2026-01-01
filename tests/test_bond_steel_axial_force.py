"""Regression test for steel axial internal force bug (Task A).

This test verifies that the steel axial internal force is correctly assembled
in both the Numba and Python implementations of bond-slip assembly.

Bug description:
Prior to this fix, the steel axial stiffness K_steel was assembled but the
corresponding internal force f_steel = K_steel @ u_local was missing. This caused
incorrect force balance and prevented reactions from appearing on fixed DOFs.

Test setup:
- Minimal 2-node segment with steel DOFs active
- No concrete coupling (segment_mask=True for bond disabled)
- Nonzero u_steel displacement imposed
- Check that internal force is nonzero and equals K_steel @ u_local
- Check that reaction appears on Dirichlet-fixed DOF
"""

import numpy as np
import pytest

from xfem_clean.bond_slip import (
    assemble_bond_slip,
    BondSlipModelCode2010,
    BondSlipStateArrays,
)


class TestSteelAxialInternalForce:
    """Regression tests for steel axial internal force assembly."""

    def test_steel_axial_force_nonzero_python(self):
        """Test that steel axial internal force is nonzero (Python fallback)."""
        # Setup: 2-node segment, horizontal, L = 1.0 m
        n_seg = 1
        steel_segments = np.array([[0, 1, 1.0, 1.0, 0.0]], dtype=float)  # [n1, n2, L, cx, cy]

        # DOF mapping: 2 nodes, concrete DOFs 0-3, steel DOFs 4-7
        steel_dof_offset = 4
        steel_dof_map = np.array([[4, 5], [6, 7]], dtype=np.int64)
        ndof_total = 8

        # Impose displacement: fix node 0, displace node 1 in x-direction
        u_total = np.zeros(ndof_total, dtype=float)
        u_total[6] = 0.001  # Steel node 1, x-direction: +1 mm displacement

        # Bond law (arbitrary, bond will be disabled)
        bond_law = BondSlipModelCode2010(f_cm=30e6, d_bar=0.016, condition="good")
        bond_states = BondSlipStateArrays.zeros(n_seg)

        # Disable bond for this segment (segment_mask=True)
        segment_mask = np.array([True], dtype=bool)

        # Steel EA (nonzero to trigger steel axial stiffness)
        E_steel = 200e9  # Pa
        A_steel = np.pi * (0.016 / 2) ** 2  # 16mm diameter
        steel_EA = E_steel * A_steel

        # Assemble with Python (use_numba=False)
        f_bond, K_bond, _ = assemble_bond_slip(
            u_total=u_total,
            steel_segments=steel_segments,
            steel_dof_offset=steel_dof_offset,
            bond_law=bond_law,
            bond_states=bond_states,
            steel_dof_map=steel_dof_map,
            steel_EA=steel_EA,
            use_numba=False,
            segment_mask=segment_mask,  # Bond disabled
        )

        # Expected: axial force N = (EA/L) * du = (EA/L) * (u1 - u0) = (EA/L) * 0.001
        L = 1.0
        du_x = u_total[6] - u_total[4]  # u1x - u0x = 0.001 m
        cx = 1.0  # horizontal
        axial_elongation = du_x * cx  # 0.001 m
        N_expected = (steel_EA / L) * axial_elongation

        # Internal force at node 0 (should be -N*cx in x-direction)
        f_node0_x = f_bond[4]  # dof_s0x
        # Internal force at node 1 (should be +N*cx in x-direction)
        f_node1_x = f_bond[6]  # dof_s1x

        # Check force magnitude
        assert abs(f_node0_x) > 1e-6, (
            "Steel axial internal force at node 0 should be nonzero "
            f"(got {f_node0_x:.4e})"
        )
        assert abs(f_node1_x) > 1e-6, (
            "Steel axial internal force at node 1 should be nonzero "
            f"(got {f_node1_x:.4e})"
        )

        # Check force direction and magnitude
        np.testing.assert_allclose(
            f_node0_x,
            -N_expected,
            rtol=1e-10,
            err_msg=f"Node 0 force mismatch: expected {-N_expected:.4e}, got {f_node0_x:.4e}",
        )
        np.testing.assert_allclose(
            f_node1_x,
            +N_expected,
            rtol=1e-10,
            err_msg=f"Node 1 force mismatch: expected {+N_expected:.4e}, got {f_node1_x:.4e}",
        )

        # Check force balance (f0 + f1 should be zero)
        force_sum = f_node0_x + f_node1_x
        assert abs(force_sum) < 1e-10, (
            f"Force balance violated: f0 + f1 = {force_sum:.4e} (should be ~0)"
        )

    def test_steel_axial_force_equals_K_times_u_python(self):
        """Test that internal force equals K @ u_local (Python fallback)."""
        # Setup same as above
        n_seg = 1
        steel_segments = np.array([[0, 1, 1.0, 1.0, 0.0]], dtype=float)
        steel_dof_offset = 4
        steel_dof_map = np.array([[4, 5], [6, 7]], dtype=np.int64)
        ndof_total = 8

        u_total = np.zeros(ndof_total, dtype=float)
        u_total[6] = 0.001  # Impose 1mm displacement

        bond_law = BondSlipModelCode2010(f_cm=30e6, d_bar=0.016, condition="good")
        bond_states = BondSlipStateArrays.zeros(n_seg)
        segment_mask = np.array([True], dtype=bool)

        E_steel = 200e9
        A_steel = np.pi * (0.016 / 2) ** 2
        steel_EA = E_steel * A_steel

        f_bond, K_bond, _ = assemble_bond_slip(
            u_total=u_total,
            steel_segments=steel_segments,
            steel_dof_offset=steel_dof_offset,
            bond_law=bond_law,
            bond_states=bond_states,
            steel_dof_map=steel_dof_map,
            steel_EA=steel_EA,
            use_numba=False,
            segment_mask=segment_mask,
        )

        # Compute f_expected = K @ u
        f_expected = K_bond @ u_total

        # Check that internal force matches K @ u
        np.testing.assert_allclose(
            f_bond,
            f_expected,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Internal force should equal K @ u",
        )

    def test_steel_axial_force_produces_reaction_on_fixed_dof_python(self):
        """Test that fixing one end produces a reaction (Python fallback)."""
        n_seg = 1
        steel_segments = np.array([[0, 1, 1.0, 1.0, 0.0]], dtype=float)
        steel_dof_offset = 4
        steel_dof_map = np.array([[4, 5], [6, 7]], dtype=np.int64)
        ndof_total = 8

        # Fix node 0 (dof 4, 5), displace node 1
        u_total = np.zeros(ndof_total, dtype=float)
        u_total[6] = 0.001  # Node 1, x-direction

        bond_law = BondSlipModelCode2010(f_cm=30e6, d_bar=0.016, condition="good")
        bond_states = BondSlipStateArrays.zeros(n_seg)
        segment_mask = np.array([True], dtype=bool)

        E_steel = 200e9
        A_steel = np.pi * (0.016 / 2) ** 2
        steel_EA = E_steel * A_steel

        f_bond, K_bond, _ = assemble_bond_slip(
            u_total=u_total,
            steel_segments=steel_segments,
            steel_dof_offset=steel_dof_offset,
            bond_law=bond_law,
            bond_states=bond_states,
            steel_dof_map=steel_dof_map,
            steel_EA=steel_EA,
            use_numba=False,
            segment_mask=segment_mask,
        )

        # Reaction at fixed DOF (node 0, x-direction)
        reaction_x = f_bond[4]  # dof_s0x

        # Expected reaction: -EA/L * du = -EA/L * 0.001
        L = 1.0
        reaction_expected = -(steel_EA / L) * 0.001

        # Check reaction is nonzero
        assert abs(reaction_x) > 1e-6, (
            f"Reaction at fixed DOF should be nonzero (got {reaction_x:.4e})"
        )

        # Check reaction magnitude
        np.testing.assert_allclose(
            reaction_x,
            reaction_expected,
            rtol=1e-10,
            err_msg=f"Reaction mismatch: expected {reaction_expected:.4e}, got {reaction_x:.4e}",
        )

    @pytest.mark.parametrize("use_numba", [False, True])
    def test_steel_axial_force_numba_vs_python(self, use_numba):
        """Test that Numba and Python give identical results for steel axial force."""
        if use_numba:
            # Try to import Numba; skip if not available
            try:
                from xfem_clean.numba.kernels_bond_slip import bond_slip_assembly_kernel
            except Exception:
                pytest.skip("Numba not available")

        # Setup
        n_seg = 1
        steel_segments = np.array([[0, 1, 1.0, 1.0, 0.0]], dtype=float)
        steel_dof_offset = 4
        steel_dof_map = np.array([[4, 5], [6, 7]], dtype=np.int64)
        ndof_total = 8

        u_total = np.zeros(ndof_total, dtype=float)
        u_total[6] = 0.001  # Node 1, x-direction

        bond_law = BondSlipModelCode2010(f_cm=30e6, d_bar=0.016, condition="good")
        bond_states = BondSlipStateArrays.zeros(n_seg)
        segment_mask = np.array([True], dtype=bool)

        E_steel = 200e9
        A_steel = np.pi * (0.016 / 2) ** 2
        steel_EA = E_steel * A_steel

        # Assemble with specified backend
        f_bond, K_bond, _ = assemble_bond_slip(
            u_total=u_total,
            steel_segments=steel_segments,
            steel_dof_offset=steel_dof_offset,
            bond_law=bond_law,
            bond_states=bond_states,
            steel_dof_map=steel_dof_map,
            steel_EA=steel_EA,
            use_numba=use_numba,
            segment_mask=segment_mask,
        )

        # Check force is nonzero
        assert np.linalg.norm(f_bond) > 1e-6, (
            f"Internal force should be nonzero (backend: {'numba' if use_numba else 'python'})"
        )

        # Reference: Python assembly
        if use_numba:
            f_ref, K_ref, _ = assemble_bond_slip(
                u_total=u_total,
                steel_segments=steel_segments,
                steel_dof_offset=steel_dof_offset,
                bond_law=bond_law,
                bond_states=bond_states,
                steel_dof_map=steel_dof_map,
                steel_EA=steel_EA,
                use_numba=False,
                segment_mask=segment_mask,
            )

            # Check Numba matches Python
            np.testing.assert_allclose(
                f_bond,
                f_ref,
                rtol=1e-10,
                atol=1e-12,
                err_msg="Numba and Python internal forces should match",
            )

            np.testing.assert_allclose(
                K_bond.toarray(),
                K_ref.toarray(),
                rtol=1e-10,
                atol=1e-12,
                err_msg="Numba and Python stiffness matrices should match",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
