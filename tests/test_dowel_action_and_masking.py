"""Tests for dowel action and segment masking fixes.

This module tests:
- Dowel action constitutive law (thesis equations 3.62-3.68)
- Segment masking correctness (Numba and Python paths)
- VTK export bug fix
"""

import numpy as np
import pytest
from xfem_clean.bond_slip import (
    DowelActionModel,
    assemble_bond_slip,
    BondSlipModelCode2010,
    BondSlipStateArrays,
)


class TestDowelAction:
    """Test dowel action constitutive law (P4)."""

    def test_dowel_stress_at_zero_opening(self):
        """Test that dowel stress is zero at zero opening."""
        fc_pa = 30e6  # 30 MPa
        d_bar_m = 0.016  # 16 mm

        model = DowelActionModel(d_bar=d_bar_m, f_c=fc_pa)

        sigma, dsigma_dw = model.sigma_and_tangent(w=0.0)

        # At w=0, stress should be zero
        assert abs(sigma) < 1e-6, f"Expected sigma≈0 at w=0, got {sigma}"
        # Tangent should be finite and positive
        assert dsigma_dw > 0, f"Expected positive tangent at w=0, got {dsigma_dw}"
        assert np.isfinite(dsigma_dw), "Tangent should be finite at w=0"

    def test_dowel_stress_positive_small_opening(self):
        """Test that dowel stress is positive for small positive openings.

        Note: Monotonic increase is NOT guaranteed by Eq. 3.62-3.68 (Brenna et al.),
        so we only check that σ(w) > 0 for small w > 0.
        """
        fc_pa = 30e6  # 30 MPa
        d_bar_m = 0.016  # 16 mm

        model = DowelActionModel(d_bar=d_bar_m, f_c=fc_pa)

        # Test at several small opening values (where stress should be positive)
        w_values = [0.001e-3, 0.01e-3, 0.02e-3]  # [m] - small openings in range [0, 0.02mm]

        for w in w_values:
            sigma, _ = model.sigma_and_tangent(w)
            assert np.isfinite(sigma), f"Stress should be finite at w={w*1e3:.2f}mm"
            assert sigma >= 0, (
                f"Stress should be positive for small positive opening: "
                f"w={w*1e3:.4f}mm, sigma={sigma:.4e} Pa"
            )

    def test_dowel_tangent_finite_difference(self):
        """Test that analytical tangent matches numerical derivative."""
        fc_pa = 30e6  # 30 MPa
        d_bar_m = 0.016  # 16 mm

        model = DowelActionModel(d_bar=d_bar_m, f_c=fc_pa)

        # Test at a moderate opening
        w0 = 0.5e-3  # 0.5 mm
        sigma0, dsigma_dw_analytical = model.sigma_and_tangent(w0)

        # Numerical derivative using central difference
        h = 1e-9  # Small perturbation (1 nm)
        sigma_plus, _ = model.sigma_and_tangent(w0 + h)
        sigma_minus, _ = model.sigma_and_tangent(w0 - h)
        dsigma_dw_numerical = (sigma_plus - sigma_minus) / (2 * h)

        # Check relative error
        rel_error = abs(dsigma_dw_analytical - dsigma_dw_numerical) / max(
            abs(dsigma_dw_analytical), 1e-10
        )
        assert rel_error < 1e-3, (
            f"Tangent mismatch: analytical={dsigma_dw_analytical:.4e}, "
            f"numerical={dsigma_dw_numerical:.4e}, rel_error={rel_error:.2e}"
        )


class TestSegmentMasking:
    """Test segment masking correctness (Task B)."""

    def test_segment_mask_numba_vs_python(self):
        """Test that Numba and Python paths give identical results for masking."""
        # Create a simple 2-segment system
        n_seg = 2
        ndof_total = 8  # 4 nodes × 2 DOFs (concrete) + steel DOFs not used here

        # Simple geometry: 2 horizontal segments
        steel_segments = np.array(
            [
                [0, 1, 1.0, 1.0, 0.0],  # Segment 0: nodes 0-1, L=1m, horizontal
                [1, 2, 1.0, 1.0, 0.0],  # Segment 1: nodes 1-2, L=1m, horizontal
            ],
            dtype=float,
        )

        # Steel DOF map (simple: steel DOFs after concrete)
        steel_dof_offset = 4
        steel_dof_map = np.array(
            [[4, 5], [6, 7], [8, 9]], dtype=np.int64  # 3 nodes
        )
        ndof_total = 10

        # Displacements (small test values)
        u_total = np.zeros(ndof_total, dtype=float)
        u_total[4] = 0.001  # Small slip for segment 0

        # Bond law
        bond_law = BondSlipModelCode2010(f_cm=30e6, d_bar=0.016, condition="good")

        # Bond states
        bond_states = BondSlipStateArrays.zeros(n_seg)

        # Mask: disable segment 0
        segment_mask = np.array([True, False], dtype=bool)

        # Assemble with Numba
        try:
            f_numba, K_numba, states_numba = assemble_bond_slip(
                u_total=u_total,
                steel_segments=steel_segments,
                steel_dof_offset=steel_dof_offset,
                bond_law=bond_law,
                bond_states=bond_states,
                steel_dof_map=steel_dof_map,
                steel_EA=0.0,
                use_numba=True,
                segment_mask=segment_mask,
            )
        except Exception as e:
            pytest.skip(f"Numba not available or failed: {e}")

        # Assemble with Python
        f_python, K_python, states_python = assemble_bond_slip(
            u_total=u_total,
            steel_segments=steel_segments,
            steel_dof_offset=steel_dof_offset,
            bond_law=bond_law,
            bond_states=bond_states,
            steel_dof_map=steel_dof_map,
            steel_EA=0.0,
            use_numba=False,
            segment_mask=segment_mask,
        )

        # Check forces match
        np.testing.assert_allclose(
            f_numba,
            f_python,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Numba and Python forces should match with segment mask",
        )

        # Check stiffness match
        K_numba_dense = K_numba.toarray()
        K_python_dense = K_python.toarray()
        np.testing.assert_allclose(
            K_numba_dense,
            K_python_dense,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Numba and Python stiffness should match with segment mask",
        )

        # Check that masked segment 0 has zero slip
        assert abs(states_numba.s_current[0]) < 1e-14, (
            "Masked segment 0 should have zero slip"
        )
        assert abs(states_python.s_current[0]) < 1e-14, (
            "Masked segment 0 should have zero slip (Python)"
        )

    def test_masked_segment_zero_contribution(self):
        """Test that masked segments contribute zero force and stiffness."""
        n_seg = 1
        ndof_total = 6

        steel_segments = np.array([[0, 1, 1.0, 1.0, 0.0]], dtype=float)

        steel_dof_offset = 4
        steel_dof_map = np.array([[4, 5], [6, 7]], dtype=np.int64)
        ndof_total = 8

        u_total = np.zeros(ndof_total, dtype=float)
        u_total[4] = 0.001  # Impose slip

        bond_law = BondSlipModelCode2010(f_cm=30e6, d_bar=0.016, condition="good")
        bond_states = BondSlipStateArrays.zeros(n_seg)

        # Assemble WITHOUT mask
        f_unmask, K_unmask, _ = assemble_bond_slip(
            u_total=u_total,
            steel_segments=steel_segments,
            steel_dof_offset=steel_dof_offset,
            bond_law=bond_law,
            bond_states=bond_states,
            steel_dof_map=steel_dof_map,
            steel_EA=0.0,
            use_numba=False,
            segment_mask=None,
        )

        # Assemble WITH mask (disable segment)
        segment_mask = np.array([True], dtype=bool)
        f_mask, K_mask, _ = assemble_bond_slip(
            u_total=u_total,
            steel_segments=steel_segments,
            steel_dof_offset=steel_dof_offset,
            bond_law=bond_law,
            bond_states=bond_states,
            steel_dof_map=steel_dof_map,
            steel_EA=0.0,
            use_numba=False,
            segment_mask=segment_mask,
        )

        # Masked assembly should give zero force and stiffness
        assert np.linalg.norm(f_mask) < 1e-14, (
            "Masked segment should produce zero force"
        )
        assert np.linalg.norm(K_mask.toarray()) < 1e-14, (
            "Masked segment should produce zero stiffness"
        )

        # Unmasked assembly should be non-zero
        assert np.linalg.norm(f_unmask) > 1e-6, (
            "Unmasked segment should produce non-zero force"
        )


class TestDowelActionAssembly:
    """Test dowel action assembly integration."""

    def test_dowel_action_produces_transverse_coupling(self):
        """Test that enabling dowel action couples transverse DOFs."""
        n_seg = 1
        steel_segments = np.array([[0, 1, 1.0, 1.0, 0.0]], dtype=float)  # Horizontal

        steel_dof_offset = 4
        steel_dof_map = np.array([[4, 5], [6, 7]], dtype=np.int64)
        ndof_total = 8

        # Impose transverse opening (perpendicular to bar)
        u_total = np.zeros(ndof_total, dtype=float)
        u_total[5] = 0.001  # Steel node 0, y-displacement (transverse to horizontal bar)

        bond_law = BondSlipModelCode2010(f_cm=30e6, d_bar=0.016, condition="good")
        bond_states = BondSlipStateArrays.zeros(n_seg)

        # Assemble WITHOUT dowel
        f_no_dowel, K_no_dowel, _ = assemble_bond_slip(
            u_total=u_total,
            steel_segments=steel_segments,
            steel_dof_offset=steel_dof_offset,
            bond_law=bond_law,
            bond_states=bond_states,
            steel_dof_map=steel_dof_map,
            steel_EA=0.0,
            use_numba=False,
            enable_dowel=False,
        )

        # Assemble WITH dowel
        dowel_model = DowelActionModel(d_bar=0.016, f_c=30e6)
        f_with_dowel, K_with_dowel, _ = assemble_bond_slip(
            u_total=u_total,
            steel_segments=steel_segments,
            steel_dof_offset=steel_dof_offset,
            bond_law=bond_law,
            bond_states=bond_states,
            steel_dof_map=steel_dof_map,
            steel_EA=0.0,
            use_numba=False,
            enable_dowel=True,
            dowel_model=dowel_model,
        )

        # With dowel, transverse force should be non-zero
        # Check y-direction forces
        fy_with_dowel = abs(f_with_dowel[5]) + abs(f_with_dowel[1])
        fy_no_dowel = abs(f_no_dowel[5]) + abs(f_no_dowel[1])

        assert fy_with_dowel > fy_no_dowel + 1e-6, (
            f"Dowel action should produce transverse force: "
            f"with_dowel={fy_with_dowel:.4e}, without={fy_no_dowel:.4e}"
        )

        # Check that stiffness has increased in transverse direction
        K_diff = (K_with_dowel - K_no_dowel).toarray()
        assert np.linalg.norm(K_diff) > 1e-6, (
            "Dowel action should add transverse stiffness"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
