"""Integration tests for dissertation parity features.

Tests that the dissertation features integrate correctly with the production solver.

Reference: Dissertation 10.5445/IR/1000124842, Chapter 4
"""

import numpy as np
import pytest


def test_imports_dissertation_modules():
    """Test that all dissertation modules can be imported."""
    from xfem_clean.numerical_aspects import StabilizationParams, compute_node_removal_criterion
    from xfem_clean.reinforcement import ReinforcementLayer, ReinforcementSegment, ReinforcementState
    from xfem_clean.contact_rebar import RebarContactPoint, assemble_rebar_contact
    from xfem_clean.junction import CrackJunction, detect_crack_coalescence
    from xfem_clean.dof_mapping import project_dofs_l2, transfer_dofs_simple
    from xfem_clean.compression_damage import ConcreteCompressionModel, compute_equivalent_compressive_strain
    from xfem_clean.compression_damage_hook import apply_compression_damage_degradation

    # Verify classes can be instantiated
    stab = StabilizationParams()
    assert stab.use_dolbow_removal is True
    assert stab.tol_dolbow == 1e-4


def test_xfem_model_with_dissertation_fields():
    """Test that XFEMModel accepts dissertation parity fields."""
    from xfem_clean.xfem.model import XFEMModel

    model = XFEMModel(
        L=1.0,
        H=0.5,
        b=0.15,
        E=30e9,
        nu=0.2,
        ft=3e6,
        Gf=100.0,
        steel_A_total=3.14e-4,
        steel_E=200e9,
        # Dissertation features
        enable_reinforcement_heaviside=True,
        enable_rebar_contact=True,
        enable_junction_enrichment=False,  # Requires multi-crack
        enable_dof_projection=True,
        enable_dolbow_removal=True,
        enable_kinked_tips=True,
        enable_compression_damage=False,
        n_gauss_line=7,
        junction_merge_tolerance=0.01,
    )

    assert model.enable_reinforcement_heaviside is True
    assert model.enable_rebar_contact is True
    assert model.enable_dof_projection is True
    assert model.enable_dolbow_removal is True
    assert model.n_gauss_line == 7
    assert model.reinforcement_layers == []
    assert model.rebar_contact_points == []


def test_reinforcement_layer_creation():
    """Test creating a reinforcement layer."""
    from xfem_clean.reinforcement import create_straight_reinforcement_layer

    layer = create_straight_reinforcement_layer(
        x_start=np.array([0.1, 0.05]),
        x_end=np.array([0.9, 0.05]),
        A_s=1.57e-4,  # 14mm bar
        E_s=200e9,
        f_y=500e6,
        E_h=2e9,
        d_bar=0.014,
        layer_type="longitudinal",
        layer_id=0,
        n_segments=4,
    )

    assert layer.A_total == pytest.approx(1.57e-4)
    assert len(layer.segments) == 4
    assert layer.total_length() == pytest.approx(0.8)


def test_rebar_contact_point_creation():
    """Test creating a rebar contact point."""
    from xfem_clean.contact_rebar import RebarContactPoint

    cp = RebarContactPoint(
        X_c=np.array([0.5, 0.1]),
        t_hat=np.array([1.0, 0.0]),
        k_p=1e9,
        layer_l_id=0,
        layer_t_id=1,
        node_l=10,
        node_t=20,
        contact_type="crossing",
    )

    assert cp.k_p == 1e9
    assert np.allclose(cp.t_hat, [1.0, 0.0])
    assert cp.contact_type == "crossing"


def test_compression_damage_model():
    """Test compression damage model."""
    from xfem_clean.compression_damage import (
        get_default_compression_model,
        uniaxial_compression_test,
    )

    model = get_default_compression_model(f_c_mpa=30.0)

    assert model.f_c > 0
    assert model.eps_c1 > 0
    assert model.E_0 > 0

    # Run a uniaxial test
    eps_hist, sig_hist, dmg_hist = uniaxial_compression_test(
        model, eps_max=0.005, n_steps=20
    )

    assert len(eps_hist) == 20
    assert len(sig_hist) == 20
    assert len(dmg_hist) == 20
    assert sig_hist[-1] > 0  # Should have stress
    assert dmg_hist[-1] >= 0  # Should have some damage


def test_dolbow_criterion_computation():
    """Test Dolbow criterion computation."""
    from xfem_clean.numerical_aspects import compute_node_removal_criterion

    # Create simple mesh: 2x2 quad
    nodes = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.1, 0.1],
        [0.0, 0.1],
    ])
    elems = np.array([[0, 1, 2, 3]])

    # Mock crack object
    class MockCrack:
        def phi(self, x, y):
            return x - 0.05  # Vertical crack at x=0.05

    crack = MockCrack()

    # Test node 0 (left side)
    should_remove, eta = compute_node_removal_criterion(
        node_idx=0,
        nodes=nodes,
        elems=elems,
        crack=crack,
        tol_dolbow=1e-4,
    )

    assert isinstance(should_remove, bool)
    assert 0.0 <= eta <= 0.5


def test_assembly_with_dissertation_features():
    """Test that assembly can be called with dissertation parameters.

    This is a smoke test to ensure the API is correct.
    """
    from xfem_clean.xfem.assembly_single import assemble_xfem_system
    from xfem_clean.xfem.geometry import XFEMCrack
    from xfem_clean.xfem.dofs_single import build_xfem_dofs
    from xfem_clean.cohesive_laws import CohesiveLaw
    from xfem_clean.fem.mesh import structured_quad_mesh

    # Create simple mesh
    nodes, elems = structured_quad_mesh(L=1.0, H=0.5, nx=4, ny=2)

    # Inactive crack
    crack = XFEMCrack(x0=0.5, y0=0.0, tip_x=0.5, tip_y=0.0, stop_y=0.25, active=False)

    # Build DOFs
    dofs = build_xfem_dofs(
        nodes, elems, crack, H_region_ymax=0.0,
        tip_patch=(0.4, 0.6, 0.0, 0.1),
        rebar_segs=None,
        enable_bond_slip=False,
        enable_dolbow_removal=False,
    )

    # Elastic stiffness
    E = 30e9
    nu = 0.2
    c = E / (1.0 - nu * nu)
    C = np.array([
        [c, c * nu, 0.0],
        [c * nu, c, 0.0],
        [0.0, 0.0, c * (1.0 - nu) / 2.0]
    ])

    # Zero displacement
    q = np.zeros(dofs.ndof)

    # Cohesive law
    law = CohesiveLaw(Kn=1e12, ft=3e6, Gf=100.0)

    # Call assembly with dissertation features (all disabled for this test)
    K, fint, coh_updates, mp_updates, aux, bond_updates, reinf_updates, contact_updates = assemble_xfem_system(
        nodes=nodes,
        elems=elems,
        dofs=dofs,
        crack=crack,
        C=C,
        thickness=0.15,
        q=q,
        law=law,
        coh_states_comm={},
        tip_enr_radius=0.1,
        k_stab=1e-9,
        # Dissertation features
        reinforcement_layers=[],
        enable_reinforcement=False,
        n_gauss_line=7,
        reinforcement_states_comm=None,
        rebar_contact_points=[],
        enable_rebar_contact=False,
    )

    assert K.shape == (dofs.ndof, dofs.ndof)
    assert len(fint) == dofs.ndof
    assert reinf_updates is None  # Disabled
    assert contact_updates is None  # Disabled


def test_dof_projection_simple():
    """Test simple DOF transfer (fallback when projection disabled)."""
    from xfem_clean.dof_mapping import transfer_dofs_simple

    # Mock old DOFs
    class MockDofs:
        def __init__(self, ndof, nnode):
            self.ndof = ndof
            self.std = np.arange(2 * nnode).reshape(nnode, 2)
            self.H = -np.ones((nnode, 2), dtype=int)
            self.tip = -np.ones((nnode, 4, 2), dtype=int)

    dofs_old = MockDofs(ndof=20, nnode=10)
    dofs_new = MockDofs(ndof=20, nnode=10)

    q_old = np.random.rand(20)
    q_new = transfer_dofs_simple(q_old, dofs_old, dofs_new)

    # Standard DOFs should match
    assert np.allclose(q_new[:20], q_old[:20])


if __name__ == "__main__":
    # Run tests
    test_imports_dissertation_modules()
    test_xfem_model_with_dissertation_fields()
    test_reinforcement_layer_creation()
    test_rebar_contact_point_creation()
    test_compression_damage_model()
    test_dolbow_criterion_computation()
    test_assembly_with_dissertation_features()
    test_dof_projection_simple()
    print("All integration tests passed!")
