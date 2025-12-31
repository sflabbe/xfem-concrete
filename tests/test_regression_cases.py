"""
Regression tests for Gutiérrez thesis cases.

Validates that solver produces physically reasonable results within expected ranges.
Compares against reference metrics defined in tests/regression/reference_cases.yml.

Usage:
    pytest tests/test_regression_cases.py -v
    pytest tests/test_regression_cases.py::test_case_01_coarse -v
"""

import pytest
from pathlib import Path

try:
    import numpy as np
    import yaml
    import sys
    import os

    # Add examples to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from examples.gutierrez_thesis.cases.case_01_pullout_lettow import create_case_01
    from examples.gutierrez_thesis.cases.case_02_sspot_frp import create_case_02
    from examples.gutierrez_thesis.cases.case_03_tensile_stn12 import create_case_03
    from examples.gutierrez_thesis.cases.case_04_beam_3pb_t5a1 import create_case_04
    from examples.gutierrez_thesis.cases.case_04a_beam_3pb_t5a1_bosco import create_case_04a
    from examples.gutierrez_thesis.cases.case_04b_beam_3pb_t6a1_bosco import create_case_04b
    from examples.gutierrez_thesis.cases.case_05_wall_c1_cyclic import create_case_05
    from examples.gutierrez_thesis.cases.case_06_fibre_tensile import create_case_06
    from examples.gutierrez_thesis.cases.case_07_beam_4pb_jason_4pbt import create_case_07
    from examples.gutierrez_thesis.cases.case_08_beam_3pb_vvbs3_cfrp import create_case_08
    from examples.gutierrez_thesis.cases.case_09_beam_4pb_fibres_sorelli import create_case_09
    from examples.gutierrez_thesis.cases.case_10_wall_c2_cyclic import create_case_10
    from examples.gutierrez_thesis.solver_interface import run_case_solver

except ImportError as e:
    pytest.skip(f"Import error: {e}", allow_module_level=True)


# Load reference values
REFERENCE_FILE = Path(__file__).parent / "regression" / "reference_cases.yml"


def load_reference_ranges():
    """Load reference ranges from YAML file."""
    if not REFERENCE_FILE.exists():
        pytest.skip(f"Reference file not found: {REFERENCE_FILE}")

    with open(REFERENCE_FILE, 'r') as f:
        return yaml.safe_load(f)


def extract_metrics(history, case_name):
    """
    Extract key metrics from solver history for regression comparison.

    Parameters
    ----------
    history : dict
        Solver history with keys: 'u', 'P', 'crack_history', etc.
    case_name : str
        Case identifier for specialized metric extraction

    Returns
    -------
    metrics : dict
        Dictionary of extracted metrics (P_max_kN, u_at_Pmax_mm, energy_J, etc.)
    """
    metrics = {}

    # Convert arrays
    u_arr = np.array(history.get('u', []))
    P_arr = np.array(history.get('P', []))

    if len(P_arr) == 0:
        return {'P_max_kN': 0.0, 'u_at_Pmax_mm': 0.0, 'energy_J': 0.0}

    # Peak force and corresponding displacement
    idx_max = np.argmax(np.abs(P_arr))
    P_max = np.abs(P_arr[idx_max])
    u_at_Pmax = np.abs(u_arr[idx_max])

    metrics['P_max_kN'] = P_max / 1e3  # N → kN
    metrics['u_at_Pmax_mm'] = u_at_Pmax * 1e3  # m → mm

    # Dissipated energy (∫ P du, trapezoidal rule)
    if len(u_arr) > 1:
        energy = np.trapz(np.abs(P_arr), u_arr)
        metrics['energy_J'] = abs(energy)
    else:
        metrics['energy_J'] = 0.0

    # Case-specific metrics
    if 'pullout' in case_name or 'frp' in case_name or 'sspot' in case_name:
        # Bond-slip cases: extract slip and bond stress
        if 'slip_history' in history and len(history['slip_history']) > 0:
            slip_hist = history['slip_history']
            if len(slip_hist) > 0 and len(slip_hist[-1]) > 0:
                slip_max = np.max(np.abs(slip_hist[-1]))
                metrics['slip_max_mm'] = slip_max * 1e3  # m → mm
            else:
                metrics['slip_max_mm'] = 0.0

        if 'tau_history' in history and len(history['tau_history']) > 0:
            tau_hist = history['tau_history']
            if len(tau_hist) > 0 and len(tau_hist[-1]) > 0:
                tau_max = np.max(np.abs(tau_hist[-1]))
                metrics['tau_max_MPa'] = tau_max / 1e6  # Pa → MPa
            else:
                metrics['tau_max_MPa'] = 0.0

    if 'crack' in case_name or 'tensile' in case_name or 'beam' in case_name or 'wall' in case_name:
        # Cracking cases: count cracks
        if 'crack_history' in history:
            num_cracks = len(history.get('crack_history', []))
            metrics['num_cracks'] = num_cracks

        # Crack opening (for tensile tests)
        if 'crack_openings' in history and len(history['crack_openings']) > 0:
            w_max = np.max(history['crack_openings'])
            metrics['crack_opening_mm'] = w_max * 1e3  # m → mm

    if 'wall' in case_name or 'cyclic' in case_name:
        # Cyclic cases: count cycles
        # Simple heuristic: count sign changes in P or u
        if len(P_arr) > 2:
            sign_changes = np.sum(np.diff(np.sign(P_arr)) != 0)
            metrics['num_cycles'] = max(1, sign_changes // 2)
        else:
            metrics['num_cycles'] = 1

    if 'fibre' in case_name:
        # Fibre cases: post-peak residual force
        # Take force at final step (after peak)
        if len(P_arr) > idx_max + 5:
            P_tail = np.mean(np.abs(P_arr[idx_max+5:]))
            metrics['P_tail_kN'] = P_tail / 1e3
        else:
            metrics['P_tail_kN'] = 0.0

    return metrics


def check_metrics_in_range(metrics, expected_ranges, case_id, mesh):
    """
    Check that extracted metrics fall within expected ranges.

    Parameters
    ----------
    metrics : dict
        Extracted metrics from solver
    expected_ranges : dict
        Expected ranges {metric_name: [min, max]}
    case_id : str
        Case identifier for error messages
    mesh : str
        Mesh density for error messages

    Returns
    -------
    failures : list of str
        List of failure messages (empty if all pass)
    """
    failures = []

    for metric, (min_val, max_val) in expected_ranges.items():
        if metric not in metrics:
            failures.append(f"{metric}: NOT EXTRACTED")
            continue

        value = metrics[metric]
        if not (min_val <= value <= max_val):
            failures.append(
                f"{metric}: {value:.3e} outside range [{min_val:.3e}, {max_val:.3e}]"
            )

    return failures


# Case creators mapping
CASE_CREATORS = {
    'case_01_pullout_lettow': create_case_01,
    'case_02_sspot_frp': create_case_02,
    'case_03_tensile_stn12': create_case_03,
    'case_04_beam_3pb_t5a1': create_case_04,
    'case_04a_beam_3pb_t5a1_bosco': create_case_04a,
    'case_04b_beam_3pb_t6a1_bosco': create_case_04b,
    'case_05_wall_c1_cyclic': create_case_05,
    'case_06_fibre_tensile': create_case_06,
    'case_07_beam_4pb_jason_4pbt': create_case_07,
    'case_08_beam_3pb_vvbs3_cfrp': create_case_08,
    'case_09_beam_4pb_fibres_sorelli': create_case_09,
    'case_10_wall_c2_cyclic': create_case_10,
}


# Parametrize tests for all cases and mesh densities
@pytest.mark.parametrize("case_id", [
    "case_01_pullout_lettow",
    "case_02_sspot_frp",
    "case_03_tensile_stn12",
    "case_04_beam_3pb_t5a1",
    "case_04a_beam_3pb_t5a1_bosco",
    "case_04b_beam_3pb_t6a1_bosco",
    "case_05_wall_c1_cyclic",
    "case_06_fibre_tensile",
    "case_07_beam_4pb_jason_4pbt",
    "case_08_beam_3pb_vvbs3_cfrp",
    "case_09_beam_4pb_fibres_sorelli",
    "case_10_wall_c2_cyclic",
])
@pytest.mark.parametrize("mesh", ["coarse"])  # Start with coarse only
@pytest.mark.slow  # Mark as slow test (can be skipped in CI)
def test_regression_case(case_id, mesh):
    """
    Run regression test for a specific case and mesh density.

    Executes solver, extracts metrics, and validates against reference ranges.
    """
    # Load reference ranges
    references = load_reference_ranges()

    if case_id not in references:
        pytest.skip(f"No reference data for {case_id}")

    if mesh not in references[case_id]:
        pytest.skip(f"No reference data for {case_id}/{mesh}")

    expected_ranges = references[case_id][mesh]

    # Create case configuration
    create_case = CASE_CREATORS[case_id]
    case_config = create_case()

    # Override mesh density
    case_config.geometry.mesh_density = mesh

    # Run solver (with reduced steps for speed)
    print(f"\n{'='*70}")
    print(f"Running {case_id} with mesh={mesh}")
    print(f"{'='*70}")

    try:
        result = run_case_solver(
            case_config,
            max_steps=20,  # Reduced for regression smoke test
            return_bundle=True,
            output_dir=None,  # No file output for regression tests
        )
    except Exception as e:
        pytest.fail(f"Solver failed with exception: {e}")

    # Extract metrics
    history = result.get('history', {})
    metrics = extract_metrics(history, case_id)

    print(f"\nExtracted metrics:")
    for key, val in metrics.items():
        print(f"  {key}: {val:.3e}")

    # Validate metrics
    failures = check_metrics_in_range(metrics, expected_ranges, case_id, mesh)

    if failures:
        failure_msg = f"\n{case_id}/{mesh} FAILED:\n" + "\n".join(f"  - {f}" for f in failures)
        pytest.fail(failure_msg)

    print(f"\n✓ {case_id}/{mesh} PASSED - all metrics within expected ranges")


# Individual test functions for easier selection
def test_case_01_coarse():
    """Regression test: Case 01 Pullout (coarse mesh)"""
    test_regression_case("case_01_pullout_lettow", "coarse")


def test_case_02_coarse():
    """Regression test: Case 02 FRP SSPOT (coarse mesh)"""
    test_regression_case("case_02_sspot_frp", "coarse")


def test_case_03_coarse():
    """Regression test: Case 03 Tensile STN12 (coarse mesh)"""
    test_regression_case("case_03_tensile_stn12", "coarse")


def test_case_04_coarse():
    """Regression test: Case 04 Beam 3PB (coarse mesh)"""
    test_regression_case("case_04_beam_3pb_t5a1", "coarse")


def test_case_05_coarse():
    """Regression test: Case 05 Wall Cyclic (coarse mesh)"""
    test_regression_case("case_05_wall_c1_cyclic", "coarse")


def test_case_06_coarse():
    """Regression test: Case 06 Fibre Tensile (coarse mesh)"""
    test_regression_case("case_06_fibre_tensile", "coarse")


def test_case_04a_coarse():
    """Regression test: Case 04a BOSCO T5A1 (coarse mesh)"""
    test_regression_case("case_04a_beam_3pb_t5a1_bosco", "coarse")


def test_case_04b_coarse():
    """Regression test: Case 04b BOSCO T6A1 (coarse mesh)"""
    test_regression_case("case_04b_beam_3pb_t6a1_bosco", "coarse")


def test_case_07_coarse():
    """Regression test: Case 07 Jason 4PB (coarse mesh)"""
    test_regression_case("case_07_beam_4pb_jason_4pbt", "coarse")


def test_case_08_coarse():
    """Regression test: Case 08 VVBS3 CFRP (coarse mesh)"""
    test_regression_case("case_08_beam_3pb_vvbs3_cfrp", "coarse")


def test_case_09_coarse():
    """Regression test: Case 09 Sorelli Fibres (coarse mesh)"""
    test_regression_case("case_09_beam_4pb_fibres_sorelli", "coarse")


def test_case_10_coarse():
    """Regression test: Case 10 Wall C2 (coarse mesh)"""
    test_regression_case("case_10_wall_c2_cyclic", "coarse")
