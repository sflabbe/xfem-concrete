"""Regression guards for configuration, fibre units, work, and result schema."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from examples.gutierrez_thesis.case_config import (
    CaseConfigurationError,
    normalize_case_config,
)
from examples.gutierrez_thesis.cases.case_09_beam_4pb_fibres_sorelli import create_case_09
from examples.gutierrez_thesis.run import CASE_REGISTRY
from xfem_clean.fibre_bridging import (
    fibre_traction_tangent,
    fibre_config_from_case,
)
from xfem_clean.results import AnalysisResult


@pytest.mark.parametrize("case_id,factory", tuple(CASE_REGISTRY.items()))
def test_registered_case_normalizes(case_id, factory):
    case = normalize_case_config(factory(), source=f"factory:{case_id}")
    assert case.case_id == case_id
    assert case.schema_version == 1
    assert case.concrete.model_type in {"elastic", "dp", "cdp_lite", "cdp_full"}
    assert case.solver_engine in {"auto", "single", "multi"}


def test_unknown_material_fails_with_context():
    case = create_case_09()
    case.concrete.model_type = "mystery_material"
    with pytest.raises(CaseConfigurationError) as captured:
        normalize_case_config(case, source="unit-test.yml")
    message = str(captured.value)
    assert "concrete.model_type" in message
    assert "mystery_material" in message
    assert case.case_id in message
    assert "unit-test.yml" in message


def test_case09_fibre_density_boundary_and_sampling():
    source = create_case_09()
    assert source.fibres.fibre.density == 3.43
    normalized = normalize_case_config(source)
    cfg = fibre_config_from_case(normalized.fibres, normalized.geometry.thickness * 1e-3)

    assert cfg.density_m2 == 34_300.0
    assert cfg.random_seed == 42
    assert cfg.explicit_fraction == 0.5

    delta_l = 0.01
    patch_area = cfg.thickness * delta_l
    expected_explicit_count = max(
        1, int(round(cfg.density_m2 * patch_area * cfg.explicit_fraction))
    )
    assert expected_explicit_count == 7

    first = fibre_traction_tangent(1.0e-4, delta_l, cfg)
    second = fibre_traction_tangent(1.0e-4, delta_l, cfg)
    assert first == pytest.approx(second, rel=0.0, abs=0.0)
    assert first[0] > 0.0
    assert np.all(np.isfinite(first))


def test_single_and_multi_adapters_share_result_schema():
    common = {
        "nodes": np.zeros((2, 2)),
        "elems": np.zeros((1, 4), dtype=int),
        "u": np.zeros(4),
    }
    single = AnalysisResult.from_solver_bundle(
        {**common, "history": np.array([[1, 1e-4, 10.0]]), "crack": object()},
        engine="single", material="elastic", bond_layer_count=0,
        use_numba=False, compat_mode=False,
    )
    multi = AnalysisResult.from_solver_bundle(
        {**common, "history": [{"step": 1, "u": 1e-4, "P": 10.0}], "cracks": []},
        engine="multi", material="cdp_lite", bond_layer_count=1,
        use_numba=True, compat_mode=True,
    )

    assert single.steps == multi.steps
    canonical_keys = {
        "schema_version", "steps", "fields", "energies_or_work", "cracks",
        "reinforcement", "solver_meta", "warnings",
    }
    assert canonical_keys <= set(single)
    assert canonical_keys <= set(multi)
    assert single.schema_version == multi.schema_version == 1
    assert single.solver_meta["crack_path"] == "single"
    assert multi.solver_meta["crack_path"] == "multi"
    assert multi.solver_meta["compat_mode"] is True
    assert single.solver_meta["provenance"] == {}
    assert multi.solver_meta["provenance"] == {}
    with pytest.warns(DeprecationWarning):
        assert single.legacy_view["history"] is not None


def test_closed_linear_elastic_cycle_has_zero_net_interface_work():
    stiffness = 4.0e9
    openings = np.array([0.0, 2.0e-5, 0.0])
    tractions = stiffness * openings
    work = np.sum(
        0.5 * (tractions[:-1] + tractions[1:]) * np.diff(openings)
    )
    assert work == pytest.approx(0.0, abs=1.0e-15)
