"""Unit tests for multi-layer bond (TASK 2).

This test verifies the BondLayer wiring and multi-layer reinforcement support.
"""

import sys
import os

# Add src and examples to path
repo_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(repo_root, 'src'))
sys.path.insert(0, repo_root)  # For 'examples' imports

import numpy as np
import pytest


def test_build_bond_layers_two_horizontal_rebars():
    """Test building two horizontal rebar layers at different y positions."""
    from examples.gutierrez_thesis.case_config import (
        RebarLayer, CEBFIPBondLaw, SteelConfig
    )
    from examples.gutierrez_thesis.solver_interface import build_bond_layers_from_case

    # Create simple mesh (10×4 elements, 0.1m × 0.1m)
    nx, ny = 10, 4
    L, H = 1.0, 0.4  # 1m × 0.4m
    nodes = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = i * L / nx
            y = j * H / ny
            nodes.append([x, y])
    nodes = np.array(nodes, dtype=float)

    # Simple bond law
    bond_law = CEBFIPBondLaw(
        s1=1.0,  # mm
        s2=2.0,
        s3=10.0,
        tau_max=12.0,  # MPa
        tau_f=6.0,
        alpha=0.4,
    )

    steel = SteelConfig(E=200e3, nu=0.3, f_y=500.0, f_u=600.0)

    # Create two rebar layers
    rebar1 = RebarLayer(
        diameter=12.0,  # mm
        y_position=50.0,  # mm (0.05m from bottom)
        n_bars=3,
        steel=steel,
        bond_law=bond_law,
        orientation_deg=0.0,  # Horizontal
    )

    rebar2 = RebarLayer(
        diameter=16.0,  # mm
        y_position=350.0,  # mm (0.35m from bottom)
        n_bars=2,
        steel=steel,
        bond_law=bond_law,
        orientation_deg=0.0,  # Horizontal
    )

    # Create minimal case mock object
    class MockCase:
        rebar_layers = [rebar1, rebar2]
        frp_sheets = []

    # Build bond layers
    bond_layers = build_bond_layers_from_case(MockCase(), nodes)

    # Verify we got two layers
    assert len(bond_layers) == 2, f"Expected 2 bond layers, got {len(bond_layers)}"

    # Layer 0: 12mm diameter, 3 bars
    layer0 = bond_layers[0]
    assert layer0.layer_id == "rebar_layer_0_orient0deg"
    assert layer0.segments.shape[0] > 0, "Layer 0 should have segments"

    # Verify EA calculation
    d0 = 0.012  # m
    A0 = np.pi * (d0/2)**2
    EA0_expected = 200e9 * 3 * A0  # Pa * m²
    assert np.isclose(layer0.EA, EA0_expected, rtol=1e-6), (
        f"Layer 0 EA mismatch: expected={EA0_expected/1e6:.2f} MN, got={layer0.EA/1e6:.2f} MN"
    )

    # Verify perimeter
    perimeter0_expected = 3 * np.pi * d0  # m
    assert np.isclose(layer0.perimeter, perimeter0_expected, rtol=1e-6)

    # Layer 1: 16mm diameter, 2 bars
    layer1 = bond_layers[1]
    assert layer1.layer_id == "rebar_layer_1_orient0deg"

    d1 = 0.016  # m
    A1 = np.pi * (d1/2)**2
    EA1_expected = 200e9 * 2 * A1
    assert np.isclose(layer1.EA, EA1_expected, rtol=1e-6)

    perimeter1_expected = 2 * np.pi * d1
    assert np.isclose(layer1.perimeter, perimeter1_expected, rtol=1e-6)

    print("✓ Two horizontal rebar layers built correctly")
    print(f"  Layer 0: {layer0.segments.shape[0]} segments, EA={layer0.EA/1e6:.2f} MN")
    print(f"  Layer 1: {layer1.segments.shape[0]} segments, EA={layer1.EA/1e6:.2f} MN")


def test_build_bond_layers_vertical_rebar(tmp_path):
    """Test building vertical rebar layer (orientation=90°)."""
    from examples.gutierrez_thesis.case_config import (
        CaseConfig, GeometryConfig, ConcreteConfig, SteelConfig,
        RebarLayer, CEBFIPBondLaw, OutputConfig
    )
    from examples.gutierrez_thesis.solver_interface import build_bond_layers_from_case

    # Create mesh with nodes at x=0.2 for vertical bars
    nx, ny = 5, 10
    L, H = 1.0, 1.0
    nodes = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = i * L / nx
            y = j * H / ny
            nodes.append([x, y])
    nodes = np.array(nodes, dtype=float)

    # Verify we have nodes at x=0.2 (column index i=1)
    nodes_at_x02 = nodes[np.abs(nodes[:, 0] - 0.2) < 1e-6]
    assert len(nodes_at_x02) == ny + 1, f"Should have {ny+1} nodes at x=0.2, got {len(nodes_at_x02)}"

    bond_law = CEBFIPBondLaw(s1=1.0, s2=2.0, s3=10.0, tau_max=12.0, tau_f=6.0, alpha=0.4)
    steel = SteelConfig(E=200e3, nu=0.3, f_y=500.0)

    # Vertical rebar at x=200mm
    rebar_vert = RebarLayer(
        diameter=12.0,  # mm
        y_position=200.0,  # mm (interpreted as x-position for vertical bars)
        n_bars=2,
        steel=steel,
        bond_law=bond_law,
        orientation_deg=90.0,  # Vertical!
    )

    # Minimal OutputConfig (all outputs disabled for unit test)
    outputs = OutputConfig(
        output_dir=str(tmp_path),
        case_name="test_vertical",
        save_load_displacement=False,
        save_crack_data=False,
        save_energy=False,
        save_crack_pattern=False,
        save_damage_field=False,
        save_deformed_shape=False,
        save_metrics=False,
        save_vtk=False,
        compute_crack_widths=False,
        compute_slip_profiles=False,
        compute_bond_profiles=False,
        compute_steel_forces=False,
    )

    case = CaseConfig(
        case_id="test_vertical",
        description="unit-test: vertical rebar bond layer builder",
        geometry=GeometryConfig(length=1000, height=1000, thickness=100, n_elem_x=nx, n_elem_y=ny),
        concrete=ConcreteConfig(E=30e3, nu=0.2, f_c=30.0, f_t=3.0, G_f=0.1),
        loading=None,
        outputs=outputs,
        rebar_layers=[rebar_vert],
        frp_sheets=[],
    )

    bond_layers = build_bond_layers_from_case(case, nodes)

    assert len(bond_layers) == 1
    layer = bond_layers[0]

    assert layer.layer_id == "rebar_layer_0_orient90deg"
    assert layer.segments.shape[0] == ny, f"Expected {ny} segments, got {layer.segments.shape[0]}"

    # Check that segments are vertical (cy ≈ 1, cx ≈ 0)
    for i in range(layer.segments.shape[0]):
        cx, cy = layer.segments[i, 3], layer.segments[i, 4]
        assert abs(cy) > 0.99, f"Segment {i} should be vertical: cy={cy}"
        assert abs(cx) < 0.01, f"Segment {i} should be vertical: cx={cx}"

    print(f"✓ Vertical rebar layer built correctly")
    print(f"  {layer.segments.shape[0]} vertical segments at x=0.2m")
    print(f"  Tangent vectors: cy ≈ {layer.segments[:, 4].mean():.3f} (should be ≈1)")


def test_segment_mask(tmp_path):
    """Test bond-disabled regions (segment masking)."""
    from examples.gutierrez_thesis.case_config import (
        CaseConfig, GeometryConfig, ConcreteConfig, SteelConfig,
        RebarLayer, CEBFIPBondLaw, OutputConfig
    )
    from examples.gutierrez_thesis.solver_interface import build_bond_layers_from_case

    nx, ny = 20, 4
    L, H = 2.0, 0.4
    nodes = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = i * L / nx
            y = j * H / ny
            nodes.append([x, y])
    nodes = np.array(nodes, dtype=float)

    bond_law = CEBFIPBondLaw(s1=1.0, s2=2.0, s3=10.0, tau_max=12.0, tau_f=6.0, alpha=0.4)
    steel = SteelConfig(E=200e3, nu=0.3, f_y=500.0)

    # Rebar with bond disabled in central region (0.8m to 1.2m)
    rebar = RebarLayer(
        diameter=12.0,
        y_position=50.0,
        n_bars=1,
        steel=steel,
        bond_law=bond_law,
        orientation_deg=0.0,
        bond_disabled_x_range=(800.0, 1200.0),  # mm
    )

    # Minimal OutputConfig (all outputs disabled for unit test)
    outputs = OutputConfig(
        output_dir=str(tmp_path),
        case_name="test_mask",
        save_load_displacement=False,
        save_crack_data=False,
        save_energy=False,
        save_crack_pattern=False,
        save_damage_field=False,
        save_deformed_shape=False,
        save_metrics=False,
        save_vtk=False,
        compute_crack_widths=False,
        compute_slip_profiles=False,
        compute_bond_profiles=False,
        compute_steel_forces=False,
    )

    case = CaseConfig(
        case_id="test_mask",
        description="unit-test: segment masking",
        geometry=GeometryConfig(length=2000, height=400, thickness=100, n_elem_x=nx, n_elem_y=ny),
        concrete=ConcreteConfig(E=30e3, nu=0.2, f_c=30.0, f_t=3.0, G_f=0.1),
        loading=None,
        outputs=outputs,
        rebar_layers=[rebar],
        frp_sheets=[],
    )

    bond_layers = build_bond_layers_from_case(case, nodes)

    assert len(bond_layers) == 1
    layer = bond_layers[0]

    # Should have segment_mask
    assert layer.segment_mask is not None, "Expected segment_mask for bond-disabled range"

    # Count disabled segments (should be in 0.8-1.2m range)
    n_disabled = np.sum(layer.segment_mask)
    n_total = layer.segments.shape[0]

    # Expect roughly 20% of segments disabled (0.4m out of 2.0m)
    disabled_fraction = n_disabled / n_total
    assert 0.1 < disabled_fraction < 0.3, (
        f"Expected ~20% disabled segments, got {disabled_fraction*100:.1f}%"
    )

    print(f"✓ Segment masking works correctly")
    print(f"  {n_disabled}/{n_total} segments masked ({disabled_fraction*100:.1f}%)")


if __name__ == "__main__":
    print("=" * 60)
    print("TASK 2: Multi-Layer Bond Tests")
    print("=" * 60)

    print("\n[1/3] Testing two horizontal rebar layers...")
    test_build_bond_layers_two_horizontal_rebars()

    print("\n[2/3] Testing vertical rebar layer (90°)...")
    test_build_bond_layers_vertical_rebar()

    print("\n[3/3] Testing segment masking...")
    test_segment_mask()

    print("\n" + "=" * 60)
    print("✓ All TASK 2 tests passed!")
    print("=" * 60)
