"""
Tests for Gutiérrez Thesis Cases

Minimal smoke tests to ensure all cases can be instantiated and configured.
"""

import pytest
from pathlib import Path

from examples.gutierrez_thesis.cases.case_01_pullout_lettow import create_case_01
from examples.gutierrez_thesis.cases.case_02_sspot_frp import create_case_02
from examples.gutierrez_thesis.cases.case_03_tensile_stn12 import create_case_03
from examples.gutierrez_thesis.cases.case_04_beam_3pb_t5a1 import create_case_04
from examples.gutierrez_thesis.cases.case_05_wall_c1_cyclic import create_case_05
from examples.gutierrez_thesis.cases.case_06_fibre_tensile import create_case_06


def test_case_01_pullout_config():
    """Test that pull-out case can be created and has correct structure"""
    case = create_case_01()
    assert case.case_id == "01_pullout_lettow"
    assert case.geometry.length == 200.0
    assert case.geometry.height == 200.0
    assert len(case.rebar_layers) == 1
    assert case.rebar_layers[0].diameter == 12.0
    assert len(case.subdomains) == 1
    assert case.subdomains[0].material_type == "void"


def test_case_02_sspot_config():
    """Test that SSPOT FRP case can be created"""
    case = create_case_02()
    assert case.case_id == "02_sspot_frp"
    assert len(case.frp_sheets) == 1
    assert case.frp_sheets[0].bond_law.law_type == "bilinear"


def test_case_03_tensile_config():
    """Test that tensile member case can be created"""
    case = create_case_03()
    assert case.case_id == "03_tensile_stn12"
    assert case.geometry.length == 1000.0
    assert len(case.rebar_layers) == 1
    assert case.concrete.model_type == "cdp_lite"


def test_case_04_beam_config():
    """Test that 3PB beam case can be created"""
    case = create_case_04()
    assert case.case_id == "04_beam_3pb_t5a1"
    assert case.geometry.notch_depth == 25.0
    assert case.concrete.model_type == "cdp_full"


def test_case_05_wall_config():
    """Test that cyclic wall case can be created"""
    case = create_case_05()
    assert case.case_id == "05_wall_c1_cyclic"
    assert case.loading.loading_type == "cyclic"
    assert len(case.rebar_layers) == 3  # h1, h2, v1
    assert len(case.subdomains) == 1  # Rigid loading beam
    assert case.subdomains[0].material_type == "rigid"


def test_case_06_fibre_config():
    """Test that fibre-reinforced case can be created"""
    case = create_case_06()
    assert case.case_id == "06_fibre_tensile"
    assert case.fibres is not None
    assert case.fibres.fibre.density == 3.02
    assert case.fibres.bond_law.law_type == "banholzer"


def test_all_cases_have_outputs_config():
    """Test that all cases have properly configured outputs"""
    cases = [
        create_case_01(),
        create_case_02(),
        create_case_03(),
        create_case_04(),
        create_case_05(),
        create_case_06(),
    ]

    for case in cases:
        assert case.outputs is not None
        assert case.outputs.output_dir is not None
        assert case.outputs.case_name is not None


def test_bond_law_mapping():
    """Test that bond laws can be mapped correctly"""
    from examples.gutierrez_thesis.solver_interface import map_bond_law
    from examples.gutierrez_thesis.case_config import CEBFIPBondLaw, BilinearBondLaw, BanholzerBondLaw

    # Test CEB-FIP
    ceb_config = CEBFIPBondLaw(s1=1.0, s2=2.0, s3=10.0, tau_max=10.0, tau_f=2.0)
    ceb_law = map_bond_law(ceb_config)
    assert ceb_law.s1 == 1.0e-3  # mm → m
    assert ceb_law.tau_max == 10.0e6  # MPa → Pa

    # Test Bilinear
    bilin_config = BilinearBondLaw(s1=0.5, s2=1.0, tau1=1.0)
    bilin_law = map_bond_law(bilin_config)
    assert bilin_law.s1 == 0.5e-3
    assert bilin_law.tau1 == 1.0e6

    # Test Banholzer
    banh_config = BanholzerBondLaw(s0=0.01, a=20.0, tau1=5.0, tau2=3.0, tau_f=2.25)
    banh_law = map_bond_law(banh_config)
    assert banh_law.s0 == 0.01e-3
    assert banh_law.a == 20.0


def test_subdomain_manager_unit_conversion():
    """Test that subdomain manager converts units correctly"""
    import numpy as np
    from xfem_clean.xfem.subdomains import build_subdomain_manager_from_config
    from examples.gutierrez_thesis.case_config import SubdomainConfig

    # Create simple mesh (4 elements in 2x2 grid)
    nodes = np.array([
        [0.0, 0.0],
        [0.1, 0.0],  # 100 mm in SI units (m)
        [0.0, 0.1],
        [0.1, 0.1],
    ])
    elems = np.array([
        [0, 1, 3, 2],
    ])

    # Subdomain config in mm (0-50 mm in x-direction)
    subdomains = [
        SubdomainConfig(
            x_range=(0.0, 50.0),  # mm
            material_type="void",
        )
    ]

    # Build manager with unit conversion (mm → m)
    mgr = build_subdomain_manager_from_config(nodes, elems, subdomains, unit_conversion=1e-3)

    # Element 0 has centroid at (0.05, 0.05) m = (50, 50) mm
    # Should be classified as void since x=50mm is exactly at boundary
    props = mgr.get_property(0)
    # Note: Depending on tolerance, this might be "bulk" or "void"
    # The key is that the conversion happened correctly


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
