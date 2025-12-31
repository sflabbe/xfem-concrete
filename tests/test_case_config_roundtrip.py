"""
Test CaseConfig serialization/deserialization round-trip.

Verifies that CaseConfig.to_dict() -> from_dict() preserves all data.
"""

import pytest
from examples.gutierrez_thesis.case_config import (
    CaseConfig,
    GeometryConfig,
    ConcreteConfig,
    SteelConfig,
    CEBFIPBondLaw,
    RebarLayer,
    MonotonicLoading,
    CyclicLoading,
    OutputConfig,
)


def test_geometry_roundtrip():
    """Test GeometryConfig round-trip"""
    geom = GeometryConfig(
        length=500.0,
        height=200.0,
        thickness=150.0,
        n_elem_x=20,
        n_elem_y=10,
        element_type="Q4",
        notch_depth=10.0,
        notch_x=250.0,
        use_symmetry=False,
    )
    data = geom.to_dict()
    geom2 = GeometryConfig.from_dict(data)

    assert geom2.length == geom.length
    assert geom2.height == geom.height
    assert geom2.n_elem_x == geom.n_elem_x
    assert geom2.notch_depth == geom.notch_depth


def test_concrete_roundtrip():
    """Test ConcreteConfig round-trip"""
    concrete = ConcreteConfig(
        E=30000.0,
        nu=0.2,
        f_c=30.0,
        f_t=3.0,
        G_f=0.1,
        eps_cu=-0.003,
        model_type="cdp_full",
    )
    data = concrete.to_dict()
    concrete2 = ConcreteConfig.from_dict(data)

    assert concrete2.E == concrete.E
    assert concrete2.f_c == concrete.f_c
    assert concrete2.G_f == concrete.G_f


def test_monotonic_loading_roundtrip():
    """Test MonotonicLoading round-trip"""
    loading = MonotonicLoading(
        max_displacement=5.0,
        n_steps=50,
        load_x_center=250.0,
        load_halfwidth=50.0,
    )
    data = loading.to_dict()
    loading2 = MonotonicLoading.from_dict(data)

    assert loading2.max_displacement == loading.max_displacement
    assert loading2.n_steps == loading.n_steps
    assert loading2.load_x_center == loading.load_x_center


def test_cyclic_loading_roundtrip():
    """Test CyclicLoading round-trip"""
    loading = CyclicLoading(
        targets=[1.0, 2.0, 3.0],
        load_x_center=250.0,
        load_halfwidth=50.0,
        n_cycles_per_target=2,
        axial_load=1000.0,
    )
    data = loading.to_dict()
    loading2 = CyclicLoading.from_dict(data)

    assert loading2.targets == loading.targets
    assert loading2.n_cycles_per_target == loading.n_cycles_per_target
    assert loading2.axial_load == loading.axial_load


def test_rebar_layer_roundtrip():
    """Test RebarLayer round-trip"""
    steel = SteelConfig(E=200000.0, nu=0.3, f_y=500.0)
    bond_law = CEBFIPBondLaw(
        s1=0.6, s2=1.0, s3=2.0, tau_max=8.0, tau_f=2.0, alpha=0.4
    )
    rebar = RebarLayer(
        diameter=16.0,
        y_position=30.0,
        n_bars=2,
        steel=steel,
        bond_law=bond_law,
        orientation_deg=0.0,
    )
    data = rebar.to_dict()
    rebar2 = RebarLayer.from_dict(data)

    assert rebar2.diameter == rebar.diameter
    assert rebar2.steel.E == rebar.steel.E
    assert rebar2.bond_law.tau_max == rebar.bond_law.tau_max


def test_case_config_roundtrip_simple():
    """Test CaseConfig round-trip for simple monotonic case"""
    geometry = GeometryConfig(
        length=500.0, height=200.0, thickness=150.0, n_elem_x=20, n_elem_y=10
    )
    concrete = ConcreteConfig(
        E=30000.0, nu=0.2, f_c=30.0, f_t=3.0, G_f=0.1
    )
    loading = MonotonicLoading(
        max_displacement=5.0, n_steps=50, load_x_center=250.0, load_halfwidth=50.0
    )
    outputs = OutputConfig(output_dir="outputs", case_name="test_case")

    config = CaseConfig(
        case_id="test_01",
        description="Test case for round-trip",
        geometry=geometry,
        concrete=concrete,
        loading=loading,
        outputs=outputs,
    )

    # Round-trip
    data = config.to_dict()
    config2 = CaseConfig.from_dict(data)

    # Verify key fields
    assert config2.case_id == config.case_id
    assert config2.description == config.description
    assert config2.geometry.length == config.geometry.length
    assert config2.concrete.f_c == config.concrete.f_c
    assert config2.loading.max_displacement == config.loading.max_displacement
    assert config2.outputs.case_name == config.outputs.case_name
    assert config2.max_steps == config.max_steps


def test_case_config_with_rebar():
    """Test CaseConfig round-trip with rebar layers"""
    geometry = GeometryConfig(
        length=500.0, height=200.0, thickness=150.0, n_elem_x=20, n_elem_y=10
    )
    concrete = ConcreteConfig(
        E=30000.0, nu=0.2, f_c=30.0, f_t=3.0, G_f=0.1
    )
    steel = SteelConfig(E=200000.0, nu=0.3, f_y=500.0)
    bond_law = CEBFIPBondLaw(
        s1=0.6, s2=1.0, s3=2.0, tau_max=8.0, tau_f=2.0
    )
    rebar = RebarLayer(
        diameter=16.0, y_position=30.0, n_bars=2, steel=steel, bond_law=bond_law
    )
    loading = MonotonicLoading(
        max_displacement=5.0, n_steps=50, load_x_center=250.0, load_halfwidth=50.0
    )
    outputs = OutputConfig()

    config = CaseConfig(
        case_id="test_02_rebar",
        description="Test with rebar",
        geometry=geometry,
        concrete=concrete,
        loading=loading,
        outputs=outputs,
        rebar_layers=[rebar],
    )

    # Round-trip
    data = config.to_dict()
    config2 = CaseConfig.from_dict(data)

    assert len(config2.rebar_layers) == 1
    assert config2.rebar_layers[0].diameter == 16.0
    assert config2.rebar_layers[0].steel.E == 200000.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
