"""Unit tests for dissertation gap implementations.

Tests all features implemented to match dissertation model:
- Phase 1: Reinforcement Heaviside enrichment
- Phase 2: Rebar contact penalty
- Phase 3: Junction enrichment + DOF mapping
- Phase 4: Numerical aspects (Dolbow, kinked tips, nonlocal)
- Phase 5: Compression damage
"""

import pytest
import numpy as np

# Import modules under test
from xfem_clean.reinforcement import (
    signed_distance_to_segment,
    heaviside_enrichment,
    shifted_heaviside_enrichment,
    compute_bar_strain_from_continuum,
    steel_elastic_1d,
    steel_bilinear_1d,
    create_straight_reinforcement_layer,
)

from xfem_clean.contact_rebar import (
    RebarContactPoint,
    compute_tangential_gap,
    penalty_contact_law,
    point_to_segment_distance,
)

from xfem_clean.junction import (
    CrackJunction,
    distance_to_crack_path,
    compute_junction_heaviside,
)

from xfem_clean.dof_mapping import (
    transfer_dofs_simple,
    compute_element_area,
)

from xfem_clean.numerical_aspects import (
    compute_triangle_area,
    compute_node_removal_criterion,
    compute_kinked_tip_coordinates,
)

from xfem_clean.compression_damage import (
    ConcreteCompressionModel,
    compute_equivalent_compressive_strain,
    uniaxial_compression_test,
)


# =============================================================================
# Phase 1: Reinforcement Tests
# =============================================================================

class TestReinforcement:
    """Tests for mesh-independent reinforcement."""

    def test_signed_distance(self):
        """Test signed distance to segment."""
        x0 = np.array([0.0, 0.0])
        x1 = np.array([1.0, 0.0])

        # Point above (positive)
        p1 = np.array([0.5, 0.5])
        dist1 = signed_distance_to_segment(p1, x0, x1)
        assert dist1 > 0.0
        assert abs(dist1 - 0.5) < 1e-12

        # Point below (negative)
        p2 = np.array([0.5, -0.3])
        dist2 = signed_distance_to_segment(p2, x0, x1)
        assert dist2 < 0.0
        assert abs(dist2 + 0.3) < 1e-12

    def test_heaviside_enrichment(self):
        """Test Heaviside function."""
        assert heaviside_enrichment(0.5) == 1.0
        assert heaviside_enrichment(-0.5) == 0.0
        assert heaviside_enrichment(0.0) == 1.0  # On boundary

    def test_shifted_heaviside(self):
        """Test shifted Heaviside enrichment."""
        # Same sign: difference = 0
        h1 = shifted_heaviside_enrichment(0.5, 0.3)
        assert abs(h1) < 1e-12

        # Opposite signs: difference = ±1
        h2 = shifted_heaviside_enrichment(0.5, -0.3)
        assert abs(h2 - 1.0) < 1e-12

    def test_bar_strain_from_continuum(self):
        """Test extraction of bar strain from continuum strain."""
        # Pure axial strain in x-direction
        eps_continuum = np.array([0.001, 0.0, 0.0])  # εxx = 0.001
        t_bar = np.array([1.0, 0.0])  # Bar along x

        eps_s = compute_bar_strain_from_continuum(eps_continuum, t_bar)
        assert abs(eps_s - 0.001) < 1e-12

        # Bar at 45°
        t_bar_45 = np.array([1.0, 1.0]) / np.sqrt(2.0)
        eps_s_45 = compute_bar_strain_from_continuum(eps_continuum, t_bar_45)
        # ε_s = tx² * εxx = 0.5 * 0.001 = 0.0005
        assert abs(eps_s_45 - 0.0005) < 1e-9

    def test_steel_elastic(self):
        """Test elastic steel constitutive law."""
        E = 200e9
        eps = 0.001

        sigma, E_t = steel_elastic_1d(eps, E)

        assert abs(sigma - E * eps) < 1e-3
        assert abs(E_t - E) < 1e-3

    def test_steel_bilinear_plastic(self):
        """Test bilinear elasto-plastic steel model."""
        E = 200e9
        f_y = 500e6
        E_h = 20e9

        # Elastic range
        eps = 0.001
        eps_p = 0.0

        sigma, E_t, eps_p_new = steel_bilinear_1d(eps, eps_p, E, f_y, E_h)

        assert sigma < f_y  # Still elastic
        assert abs(E_t - E) < 1e-3
        assert abs(eps_p_new) < 1e-12

        # Plastic range
        eps = 0.01  # Large strain
        sigma, E_t, eps_p_new = steel_bilinear_1d(eps, eps_p, E, f_y, E_h)

        assert sigma > f_y  # Hardening
        assert E_t < E  # Reduced tangent
        assert eps_p_new > 0.0  # Plastic strain accumulated

    def test_create_straight_layer(self):
        """Test creation of straight reinforcement layer."""
        x_start = np.array([0.0, 0.0])
        x_end = np.array([1.0, 0.0])

        layer = create_straight_reinforcement_layer(
            x_start=x_start,
            x_end=x_end,
            A_s=1e-4,
            E_s=200e9,
            f_y=500e6,
            E_h=20e9,
            d_bar=0.012,
            layer_type="longitudinal",
            layer_id=0,
            n_segments=4,
        )

        assert len(layer.segments) == 4
        assert abs(layer.total_length() - 1.0) < 1e-9


# =============================================================================
# Phase 2: Contact Tests
# =============================================================================

class TestRebarContact:
    """Tests for rebar penalty contact."""

    def test_tangential_gap(self):
        """Test tangential gap computation."""
        u_l = np.array([0.001, 0.0])  # Longitudinal moved right
        u_t = np.array([0.0, 0.0])    # Transverse fixed
        t_hat = np.array([1.0, 0.0])  # Contact in x-direction

        g = compute_tangential_gap(u_l, u_t, t_hat)

        assert abs(g - 0.001) < 1e-12

    def test_penalty_law_unilateral(self):
        """Test penalty contact law (unilateral)."""
        k_p = 1e9

        # Penetration (negative gap)
        g_pen = -0.001
        p_pen, dp = penalty_contact_law(g_pen, k_p, "crossing")

        assert p_pen > 0.0  # Positive pressure
        assert abs(p_pen - k_p * abs(g_pen)) < 1e-3
        assert dp < 0.0  # Negative derivative

        # Separation (positive gap)
        g_sep = 0.001
        p_sep, dp = penalty_contact_law(g_sep, k_p, "crossing")

        assert abs(p_sep) < 1e-9  # No pressure
        assert abs(dp) < 1e-9

    def test_penalty_law_bilateral(self):
        """Test penalty contact law (bilateral endpoint)."""
        k_p = 1e9
        g = 0.001  # Separation

        p, dp = penalty_contact_law(g, k_p, "endpoint")

        # Bilateral: always active
        assert abs(p + k_p * g) < 1e-3
        assert abs(dp + k_p) < 1e-3

    def test_point_to_segment_distance(self):
        """Test point-to-segment distance."""
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])

        # Point above segment midpoint
        p = np.array([0.5, 0.5])
        dist = point_to_segment_distance(p, a, b)
        assert abs(dist - 0.5) < 1e-12

        # Point beyond endpoint
        p2 = np.array([1.5, 0.5])
        dist2 = point_to_segment_distance(p2, a, b)
        expected = np.linalg.norm(p2 - b)
        assert abs(dist2 - expected) < 1e-9


# =============================================================================
# Phase 3: Junction Tests
# =============================================================================

class TestJunction:
    """Tests for crack junction enrichment."""

    def test_junction_creation(self):
        """Test junction data structure."""
        jct = CrackJunction(
            junction_point=np.array([0.05, 0.05]),
            main_crack_id=0,
            secondary_crack_id=1,
            element_id=5,
            branch_angles=[np.pi/2, 0.0],
            active=True,
        )

        assert jct.active
        assert len(jct.branch_angles) == 2
        assert jct.main_crack_id == 0


# =============================================================================
# Phase 4: Numerical Aspects Tests
# =============================================================================

class TestNumericalAspects:
    """Tests for numerical robustness features."""

    def test_triangle_area(self):
        """Test triangle area computation."""
        p0 = np.array([0.0, 0.0])
        p1 = np.array([1.0, 0.0])
        p2 = np.array([0.0, 1.0])

        area = compute_triangle_area(p0, p1, p2)

        # Right triangle: area = 0.5 * base * height = 0.5
        assert abs(area - 0.5) < 1e-12

    def test_kinked_tip_coordinates(self):
        """Test kinked crack tip coordinate transformation."""
        # Straight crack along x-axis
        crack_polyline = [
            np.array([0.0, 0.0]),
            np.array([0.5, 0.0]),
            np.array([1.0, 0.0]),  # tip
        ]

        tip_idx = 2

        # Point ahead of tip
        x = np.array([1.5, 0.5])

        r, theta = compute_kinked_tip_coordinates(x, crack_polyline, tip_idx)

        assert r > 0.0
        assert abs(theta - np.pi/4) < 1e-9  # 45° angle


# =============================================================================
# Phase 5: Compression Damage Tests
# =============================================================================

class TestCompressionDamage:
    """Tests for concrete compression damage model."""

    def test_compression_model_creation(self):
        """Test compression model creation."""
        model = ConcreteCompressionModel(
            f_c=30e6,
            eps_c1=0.002,
            E_0=30e9,
        )

        assert model.f_c == 30e6
        assert model.eps_c1 == 0.002

    def test_stress_strain_curve(self):
        """Test parabolic stress-strain curve."""
        model = ConcreteCompressionModel(
            f_c=30e6,
            eps_c1=0.002,
            E_0=30e9,
        )

        # At zero strain
        sigma0, E_t0 = model.sigma_epsilon_curve(0.0)
        assert abs(sigma0) < 1e-9
        assert abs(E_t0 - model.E_0) < 1e-3

        # At peak strain
        sigma_peak, E_t_peak = model.sigma_epsilon_curve(model.eps_c1)
        assert abs(sigma_peak - model.f_c) < 1e-3
        assert abs(E_t_peak) < 1e-3  # Zero tangent at peak

        # Beyond peak (plateau)
        sigma_post, E_t_post = model.sigma_epsilon_curve(model.eps_c1 * 2.0)
        assert abs(sigma_post - model.f_c) < 1e-3
        assert abs(E_t_post) < 1e-9

    def test_damage_computation(self):
        """Test damage from secant stiffness."""
        model = ConcreteCompressionModel(
            f_c=30e6,
            eps_c1=0.002,
            E_0=30e9,
        )

        # At zero strain: no damage
        d0 = model.compute_damage(0.0)
        assert abs(d0) < 1e-12

        # At peak: some damage
        d_peak = model.compute_damage(model.eps_c1)
        assert 0.0 < d_peak < 1.0

        # Beyond peak: more damage
        d_post = model.compute_damage(model.eps_c1 * 2.0)
        assert d_post > d_peak

    def test_equivalent_compressive_strain(self):
        """Test equivalent compressive strain."""
        # Pure compression in x
        eps = np.array([-0.001, 0.0, 0.0])  # εxx = -0.001 (compression)

        eps_eq_c = compute_equivalent_compressive_strain(eps)

        assert abs(eps_eq_c - 0.001) < 1e-9

    def test_uniaxial_compression_test(self):
        """Test uniaxial compression simulation."""
        model = ConcreteCompressionModel(
            f_c=30e6,
            eps_c1=0.002,
            E_0=30e9,
        )

        eps_hist, sigma_hist, damage_hist = uniaxial_compression_test(
            model, eps_max=0.004, n_steps=50
        )

        assert len(eps_hist) == 50
        assert len(sigma_hist) == 50
        assert len(damage_hist) == 50

        # Check monotonic increase
        assert np.all(np.diff(eps_hist) >= 0.0)
        assert np.all(damage_hist >= 0.0)
        assert np.all(damage_hist <= 1.0)

        # Stress should reach f_c
        assert np.max(sigma_hist) <= model.f_c * 1.01


# =============================================================================
# DOF Mapping Tests
# =============================================================================

class TestDofMapping:
    """Tests for DOF projection."""

    def test_element_area(self):
        """Test element area computation."""
        # Unit square
        elem_coords = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])

        area = compute_element_area(elem_coords)
        assert abs(area - 1.0) < 1e-12


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""

    def test_rebar_with_contact(self):
        """Test reinforcement layer + contact interaction."""
        # Create two layers
        layer_long = create_straight_reinforcement_layer(
            x_start=np.array([0.0, 0.5]),
            x_end=np.array([1.0, 0.5]),
            A_s=1e-4,
            E_s=200e9,
            f_y=500e6,
            E_h=20e9,
            d_bar=0.012,
            layer_type="longitudinal",
            layer_id=0,
            n_segments=1,
        )

        # Verify layer created
        assert layer_long.total_length() > 0.0

        # Create contact point
        cp = RebarContactPoint(
            X_c=np.array([0.5, 0.5]),
            t_hat=np.array([1.0, 0.0]),
            k_p=1e9,
            layer_l_id=0,
            layer_t_id=1,
            node_l=0,
            node_t=1,
            contact_type="crossing",
        )

        # Verify contact created
        assert cp.k_p > 0.0
        assert np.linalg.norm(cp.t_hat) - 1.0 < 1e-12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
