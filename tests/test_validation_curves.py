"""
Quantitative validation tests against experimental reference curves.

These tests compare simulation results with digitized experimental data
from the thesis and verify that errors are within acceptable tolerances:
- |ΔPmax| < 10%
- |ΔE| < 15%
- RMSE < 5% of peak load

Run with:
    pytest tests/test_validation_curves.py -v -m slow
    pytest tests/test_validation_curves.py::test_validate_t5a1 -v -s
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from validation.compare_curves import (
        load_simulation_curve,
        load_reference_curve,
        compute_error_metrics,
        generate_validation_report,
        save_validation_summary,
        is_placeholder_data,
    )
    from examples.gutierrez_thesis.cases.case_04a_beam_3pb_t5a1_bosco import create_case_04a
    from examples.gutierrez_thesis.cases.case_08_beam_3pb_vvbs3_cfrp import create_case_08
    from examples.gutierrez_thesis.cases.case_09_beam_4pb_fibres_sorelli import create_case_09
    from examples.gutierrez_thesis.solver_interface import run_case_solver

    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


# Tolerance thresholds (as per thesis parity requirements)
TOLERANCES = {
    'peak_error_pct': 10.0,      # ± 10%
    'energy_error_pct': 15.0,    # ± 15%
    'rmse_normalized_pct': 5.0,  # 5% of peak load
}


def check_placeholder_and_skip_if_needed(ref_df, case_name: str):
    """
    Check if reference data is placeholder and skip/xfail test if so.

    Parameters
    ----------
    ref_df : pd.DataFrame
        Reference curve dataframe
    case_name : str
        Case name for error messages

    Raises
    ------
    pytest.xfail
        If reference data is placeholder (with reason)
    """
    is_placeholder, reason = is_placeholder_data(ref_df)
    if is_placeholder:
        pytest.xfail(
            f"Reference curve for '{case_name}' is PLACEHOLDER data.\n"
            f"Reason: {reason}\n"
            f"Real digitized experimental data needed for meaningful validation.\n"
            f"See validation/reference_data/SOURCES.md for digitization guidelines."
        )


@pytest.mark.slow
@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Import failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
def test_validate_t5a1_coarse():
    """
    Validate T5A1 beam (Bosco) against experimental reference.

    Case: 3PB beam with 2Ø16 reinforcement
    Mesh: coarse (fast test)
    """
    case_id = "04a_beam_3pb_t5a1_bosco"
    ref_id = "t5a1"

    # Check if simulation output exists
    output_dir = Path(__file__).parent.parent / "examples" / "gutierrez_thesis" / "outputs" / f"case_{case_id}"

    if not (output_dir / "load_displacement.csv").exists():
        pytest.skip(
            f"Simulation output not found. Run first:\n"
            f"python -m examples.gutierrez_thesis.run --case t5a1 --mesh coarse"
        )

    # Load curves
    sim = load_simulation_curve(case_id, mesh="coarse")
    ref = load_reference_curve(ref_id)

    # Check if reference data is placeholder (xfail if so)
    check_placeholder_and_skip_if_needed(ref, ref_id)

    # Compute metrics
    metrics = compute_error_metrics(sim, ref)

    # Generate report
    report = generate_validation_report(case_id, metrics, tolerances=TOLERANCES)
    print(report)

    # Assertions
    assert metrics['peak_error_pct'] < TOLERANCES['peak_error_pct'], \
        f"Peak load error {metrics['peak_error_pct']:.2f}% exceeds tolerance {TOLERANCES['peak_error_pct']}%"

    assert metrics['energy_error_pct'] < TOLERANCES['energy_error_pct'], \
        f"Energy error {metrics['energy_error_pct']:.2f}% exceeds tolerance {TOLERANCES['energy_error_pct']}%"

    assert metrics['rmse_normalized_pct'] < TOLERANCES['rmse_normalized_pct'], \
        f"RMSE {metrics['rmse_normalized_pct']:.2f}% exceeds tolerance {TOLERANCES['rmse_normalized_pct']}%"

    # Check R² (should be > 0.9 for good fit)
    assert metrics['r_squared'] > 0.85, \
        f"R² = {metrics['r_squared']:.3f} is too low (expected > 0.85)"


@pytest.mark.slow
@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Import failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
def test_validate_vvbs3_coarse():
    """
    Validate VVBS3 beam (CFRP strengthened) against experimental reference.

    Case: 3PB beam with CFRP sheet
    Mesh: coarse (fast test)
    """
    case_id = "08_beam_3pb_vvbs3_cfrp"
    ref_id = "vvbs3"

    output_dir = Path(__file__).parent.parent / "examples" / "gutierrez_thesis" / "outputs" / f"case_{case_id}"

    if not (output_dir / "load_displacement.csv").exists():
        pytest.skip(
            f"Simulation output not found. Run first:\n"
            f"python -m examples.gutierrez_thesis.run --case vvbs3 --mesh coarse"
        )

    # Load curves
    sim = load_simulation_curve(case_id, mesh="coarse")
    ref = load_reference_curve(ref_id)

    # Check if reference data is placeholder (xfail if so)
    check_placeholder_and_skip_if_needed(ref, ref_id)

    # Compute metrics
    metrics = compute_error_metrics(sim, ref)

    # Generate report
    report = generate_validation_report(case_id, metrics, tolerances=TOLERANCES)
    print(report)

    # Assertions
    assert metrics['peak_error_pct'] < TOLERANCES['peak_error_pct'], \
        f"Peak load error {metrics['peak_error_pct']:.2f}% exceeds tolerance {TOLERANCES['peak_error_pct']}%"

    assert metrics['energy_error_pct'] < TOLERANCES['energy_error_pct'], \
        f"Energy error {metrics['energy_error_pct']:.2f}% exceeds tolerance {TOLERANCES['energy_error_pct']}%"

    assert metrics['rmse_normalized_pct'] < TOLERANCES['rmse_normalized_pct'], \
        f"RMSE {metrics['rmse_normalized_pct']:.2f}% exceeds tolerance {TOLERANCES['rmse_normalized_pct']}%"


@pytest.mark.slow
@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Import failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
def test_validate_sorelli_coarse():
    """
    Validate Sorelli fibre beam against experimental reference.

    Case: 4PB beam with steel fibres
    Mesh: coarse (fast test)
    """
    case_id = "09_beam_4pb_fibres_sorelli"
    ref_id = "sorelli"

    output_dir = Path(__file__).parent.parent / "examples" / "gutierrez_thesis" / "outputs" / f"case_{case_id}"

    if not (output_dir / "load_displacement.csv").exists():
        pytest.skip(
            f"Simulation output not found. Run first:\n"
            f"python -m examples.gutierrez_thesis.run --case sorelli --mesh coarse"
        )

    # Load curves
    sim = load_simulation_curve(case_id, mesh="coarse")
    ref = load_reference_curve(ref_id)

    # Check if reference data is placeholder (xfail if so)
    check_placeholder_and_skip_if_needed(ref, ref_id)

    # Compute metrics
    metrics = compute_error_metrics(sim, ref)

    # Generate report
    report = generate_validation_report(case_id, metrics, tolerances=TOLERANCES)
    print(report)

    # Assertions
    assert metrics['peak_error_pct'] < TOLERANCES['peak_error_pct'], \
        f"Peak load error {metrics['peak_error_pct']:.2f}% exceeds tolerance {TOLERANCES['peak_error_pct']}%"

    assert metrics['energy_error_pct'] < TOLERANCES['energy_error_pct'], \
        f"Energy error {metrics['energy_error_pct']:.2f}% exceeds tolerance {TOLERANCES['energy_error_pct']}%"


@pytest.mark.slow
@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Import failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
def test_generate_validation_summary():
    """
    Generate comprehensive validation summary CSV for all validated cases.

    This test collects metrics from all validation cases and saves a summary.
    """
    cases = [
        ("04a_beam_3pb_t5a1_bosco", "t5a1"),
        ("08_beam_3pb_vvbs3_cfrp", "vvbs3"),
        ("09_beam_4pb_fibres_sorelli", "sorelli"),
    ]

    results = {}

    for case_id, ref_id in cases:
        output_dir = Path(__file__).parent.parent / "examples" / "gutierrez_thesis" / "outputs" / f"case_{case_id}"

        if not (output_dir / "load_displacement.csv").exists():
            print(f"⚠ Skipping {case_id}: simulation not found")
            continue

        try:
            sim = load_simulation_curve(case_id, mesh="coarse")
            ref = load_reference_curve(ref_id)
            metrics = compute_error_metrics(sim, ref)
            results[case_id] = metrics
            print(f"✓ Processed {case_id}")
        except Exception as e:
            print(f"✗ Error processing {case_id}: {e}")

    if results:
        save_validation_summary(results, output_file="validation/summary_validation.csv")
        print(f"\n✓ Validation summary saved for {len(results)} cases")

        # Check that summary file was created
        summary_file = Path(__file__).parent.parent / "validation" / "summary_validation.csv"
        assert summary_file.exists(), "Summary file was not created"
    else:
        pytest.skip("No simulation outputs found. Run cases first.")


# ============================================================================
# Helper: Run simulation if needed
# ============================================================================

def run_case_if_needed(case_factory, case_id: str, mesh: str = "coarse"):
    """
    Run simulation if output doesn't exist (helper for development).

    NOT used in CI/automated tests (outputs should be pre-generated).
    """
    output_dir = Path(__file__).parent.parent / "examples" / "gutierrez_thesis" / "outputs" / f"case_{case_id}"

    if not (output_dir / "load_displacement.csv").exists():
        print(f"\n{'='*70}")
        print(f"Running simulation: {case_id} (mesh={mesh})")
        print(f"{'='*70}\n")

        case_config = case_factory()
        mesh_factor = 0.5 if mesh == "coarse" else 1.0
        results = run_case_solver(case_config, mesh_factor=mesh_factor)

        print(f"✓ Simulation completed: {output_dir}")
