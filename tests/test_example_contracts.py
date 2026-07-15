"""Small fail-closed contract tests for all canonical thesis examples."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import numpy as np

from examples.gutierrez_thesis.case_config import (
    CaseConfig,
    CaseConfigurationError,
    normalize_case_config,
)
from examples.gutierrez_thesis.catalog import (
    CASE_ALIASES,
    CASE_DEFINITIONS,
    CASE_METADATA,
    CASE_REGISTRY,
    evaluate_compatibility,
    resolve_case_id,
)
from examples.gutierrez_thesis.solver_interface import run_case_solver
from tests.process_utils import registered_process_count, run_process
from xfem_clean.results import AnalysisResult


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IDS = (
    "01_pullout_lettow",
    "02_sspot_frp",
    "03_tensile_stn12",
    "04_beam_3pb_t5a1",
    "04a_beam_3pb_t5a1_bosco",
    "04b_beam_3pb_t6a1_bosco",
    "05_wall_c1_cyclic",
    "06_fibre_tensile",
    "07_beam_4pb_jason_4pbt",
    "08_beam_3pb_vvbs3_cfrp",
    "09_beam_4pb_fibres_sorelli",
    "10_wall_c2_cyclic",
    "11_balcony_cantilever_sls",
)
SUPPORTED_IDS = {
    "01_pullout_lettow",
    "02_sspot_frp",
    "03_tensile_stn12",
    "06_fibre_tensile",
}


def _canonical_cases():
    for definition in CASE_DEFINITIONS:
        yield definition, normalize_case_config(
            definition.factory(), source=definition.factory_source
        )


def test_catalog_has_exactly_13_deterministic_cases_and_unique_outputs():
    assert tuple(CASE_REGISTRY) == EXPECTED_IDS
    cases = [case for _, case in _canonical_cases()]
    outputs = [case.outputs.output_dir for case in cases]
    assert len(outputs) == len(set(outputs))


@pytest.mark.parametrize("case_id", EXPECTED_IDS)
def test_registered_case_has_explicit_engine_and_classified_compatibility(case_id):
    definition = CASE_METADATA[case_id]
    case = normalize_case_config(definition.factory(), source=definition.factory_source)

    assert case.case_id == case_id
    assert case.solver_engine == "multi"
    assert case.schema_version == 1
    assert getattr(case, "_source") == definition.factory_source
    assert definition.status
    compatibility = evaluate_compatibility(case, use_numba=True)
    assert compatibility.supported is (case_id in SUPPORTED_IDS)


def test_alias_contract_is_exact_case_insensitive_and_has_no_partial_matching():
    assert len(CASE_ALIASES) == sum(len(item.aliases) for item in CASE_DEFINITIONS)
    for alias, expected in CASE_ALIASES.items():
        assert resolve_case_id(alias) == expected
        assert resolve_case_id(alias.upper()) == expected
    for case_id in EXPECTED_IDS:
        assert resolve_case_id(case_id.upper()) == case_id
    assert resolve_case_id("04a_beam") is None
    assert resolve_case_id("unknown") is None


def test_t5a1_alias_and_dry_run_are_explicitly_unsupported():
    assert resolve_case_id("t5a1") == "04a_beam_3pb_t5a1_bosco"
    result = run_process(
        [
            sys.executable, "-m", "examples.gutierrez_thesis.run",
            "--case", "t5a1", "--mesh", "coarse", "--dry-run",
        ],
        cwd=REPO_ROOT,
        timeout=15,
        check=True,
    )
    assert '"definition_status": "ambiguous-unsupported"' in result.stdout
    assert '"compatibility": "unsupported"' in result.stdout
    assert '"engine": "multi"' in result.stdout
    assert '"nx": 40' in result.stdout
    assert '"steps": 200' in result.stdout


def test_t5a1_solve_fails_before_mesh_or_newton_and_numba_is_not_silent():
    result = run_process(
        [
            sys.executable, "-m", "examples.gutierrez_thesis.run",
            "--case", "t5a1", "--mesh", "coarse", "--use-numba", "--no-post",
        ],
        cwd=REPO_ROOT,
        timeout=15,
    )
    assert result.returncode == 2
    combined = result.stdout + result.stderr
    assert "multicrack maps cdp_full to the CDP-lite Numba kernel" in combined
    assert "Override: use_numba = True" in combined
    assert '"support": "partial"' in combined
    assert "Built 2 bond layer" not in combined
    assert "[substep]" not in combined
    assert registered_process_count() == 0


def test_direct_t5a1_adapter_is_also_fail_closed():
    definition = CASE_METADATA["04a_beam_3pb_t5a1_bosco"]
    case = normalize_case_config(definition.factory(), source=definition.factory_source)
    with pytest.raises(CaseConfigurationError, match="engine_compatibility"):
        run_case_solver(case, mesh_factor=0.5, enable_postprocess=False)


def test_cdp_lite_multicrack_rejects_unfaithful_python_fallback():
    case = normalize_case_config(CASE_REGISTRY["03_tensile_stn12"]())
    compatibility = evaluate_compatibility(case, use_numba=False)
    assert not compatibility.supported
    assert "Python assembly fallback is linear elastic" in compatibility.reason


@pytest.mark.parametrize(
    "case_id,target_mm",
    (
        ("01_pullout_lettow", 0.01),
        ("02_sspot_frp", 0.01),
        ("03_tensile_stn12", 0.01),
        ("06_fibre_tensile", 0.001),
    ),
)
def test_supported_family_smoke_returns_finite_provenance_result(case_id, target_mm):
    definition = CASE_METADATA[case_id]
    case = normalize_case_config(definition.factory(), source=definition.factory_source)
    case.loading.max_displacement = target_mm
    case.loading.n_steps = 1

    result = run_case_solver(case, mesh_factor=0.25, enable_postprocess=False)

    assert isinstance(result, AnalysisResult)
    assert result.schema_version == 1
    assert result.steps
    assert result.solver_meta["engine"] == "multi"
    assert result.solver_meta["provenance"]["case_id"] == case_id
    assert result.solver_meta["provenance"]["case_source"] == definition.factory_source
    assert len(result.solver_meta["provenance"]["canonical_config_sha256"]) == 64
    assert all(
        np.isfinite(float(row[key]))
        for row in result.steps
        for key in ("u", "P")
    )
    assert np.isfinite(result.fields["displacement_m"]).all()


def test_all_13_cases_complete_one_canonical_dry_run_process():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = run_process(
        [
            sys.executable, "-m", "examples.gutierrez_thesis.run",
            "--case", "all", "--mesh", "coarse", "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=env,
        timeout=30,
        check=True,
    )
    assert result.stdout.count("Canonical configuration:") == 13
    assert result.stdout.count("canonical validation complete") == 13


def _diff_paths(left, right, path="") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else key
            if key not in left or key not in right:
                differences.append(child)
            else:
                differences.extend(_diff_paths(left[key], right[key], child))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        differences = []
        if len(left) != len(right):
            differences.append(f"{path}.length")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(_diff_paths(left_item, right_item, f"{path}[{index}]"))
        return differences
    return [] if left == right else [path]


@pytest.mark.parametrize(
    "legacy_name,expected_count",
    (("jason", 31), ("pullout", 26), ("t5a1", 5)),
)
def test_loadable_legacy_yaml_divergence_is_stable_and_documented(
    legacy_name, expected_count
):
    path = REPO_ROOT / "examples/gutierrez_thesis/configs/legacy" / f"{legacy_name}.yml"
    legacy = CaseConfig.load_yaml(str(path))
    canonical = normalize_case_config(CASE_REGISTRY[legacy.case_id]())
    differences = _diff_paths(legacy.to_dict(), canonical.to_dict())
    assert len(differences) == expected_count, "normalized diff:\n" + "\n".join(differences)
    assert "solver_engine" in differences
