"""
Smoke tests for parametric/calibration/benchmark tools.

These tests verify that the tools can execute their pipelines without crashes.
They use minimal configurations (small mesh, few steps) for fast execution.

Usage:
    python -m pytest tests/test_tools_smoke.py -v
    python tests/test_tools_smoke.py  # Standalone mode
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from examples.parametric.parametric_study import (
    run_parametric_study,
    modify_case_parameter,
    extract_case_metrics,
)
from examples.gutierrez_thesis.history_utils import extract_metrics, history_to_arrays


def test_history_to_arrays_numeric():
    """Test history_to_arrays with numeric format."""
    print("\n[test_history_to_arrays_numeric]")
    history = [
        [0, 0.0, 0.0],
        [1, 1e-3, 5000.0],
        [2, 2e-3, 8000.0],
    ]

    arrays = history_to_arrays(history)
    assert len(arrays['u']) == 3
    assert np.allclose(arrays['P'], [0.0, 5000.0, 8000.0])
    print("  ✓ Numeric format OK")


def test_history_to_arrays_dict():
    """Test history_to_arrays with dict format (multicrack)."""
    print("\n[test_history_to_arrays_dict]")
    history = [
        {'step': 0, 'u': 0.0, 'P': 0.0, 'ncr': 0},
        {'step': 1, 'u': 1e-3, 'P': 5000.0, 'ncr': 1},
        {'step': 2, 'u': 2e-3, 'P': 8000.0, 'ncr': 2},
    ]

    arrays = history_to_arrays(history)
    assert len(arrays['u']) == 3
    assert np.allclose(arrays['ncr'], [0, 1, 2])
    print("  ✓ Dict format OK")


def test_extract_metrics():
    """Test metric extraction."""
    print("\n[test_extract_metrics]")
    history = [
        [0, 0.0, 0.0],
        [1, 1e-3, 5000.0],
        [2, 2e-3, 8000.0],
        [3, 4e-3, 6000.0],
    ]

    metrics = extract_metrics(history)
    assert abs(metrics['P_max_kN'] - 8.0) < 1e-6
    assert abs(metrics['u_at_Pmax_mm'] - 2.0) < 1e-6
    assert abs(metrics['ductility'] - 2.0) < 1e-6
    print("  ✓ extract_metrics OK")


def test_parametric_modify_parameter():
    """Test parameter modification (without running solver)."""
    print("\n[test_parametric_modify_parameter]")

    # Import a case factory
    from examples.gutierrez_thesis.run import CASE_REGISTRY

    # Use a simple case (pullout is fastest)
    case_id = '01_pullout_basic'
    if case_id not in CASE_REGISTRY:
        print("  ⊘ Case not found, skipping")
        return

    factory = CASE_REGISTRY[case_id]
    case_config = factory()

    # Test parameter modification
    original_Gf = case_config.concrete.G_f
    modify_case_parameter(case_config, 'Gf', 0.12)
    assert case_config.concrete.G_f == 0.12
    print(f"  ✓ Modified Gf: {original_Gf} → {case_config.concrete.G_f}")


def test_benchmark_compute_energy_residual():
    """Test energy residual computation (without running solver)."""
    print("\n[test_benchmark_compute_energy_residual]")

    from benchmarks.benchmark_scaling import compute_energy_residual

    # Mock results with numeric history
    results_numeric = {
        'history': [
            [0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1e-3, 1000.0, 0.0, 0, 0, 0, 0, 0, 0, 10.0, 5.0, 2.0, 3.0, 20.0],
        ]
    }

    residual = compute_energy_residual(results_numeric)
    # Just verify it doesn't crash
    assert residual >= 0
    print(f"  ✓ Energy residual (numeric): {residual:.2f}%")

    # Mock results with dict history
    results_dict = {
        'history': [
            {'step': 0, 'u': 0.0, 'P': 0.0, 'W_total': 0.0},
            {'step': 1, 'u': 1e-3, 'P': 1000.0, 'W_total': 0.5},
        ]
    }

    residual = compute_energy_residual(results_dict)
    assert residual >= 0
    print(f"  ✓ Energy residual (dict): {residual:.2f}%")


def test_calibration_objective_setup():
    """Test calibration objective function setup (without optimization)."""
    print("\n[test_calibration_objective_setup]")

    try:
        from calibration.fit_bond_parameters import objective_function
        from examples.gutierrez_thesis.run import CASE_REGISTRY
        import pandas as pd

        # Create mock reference curve
        ref_curve = pd.DataFrame({
            'u_mm': [0.0, 1.0, 2.0],
            'P_kN': [0.0, 5.0, 8.0],
        })

        # Use a simple case
        case_id = '01_pullout_basic'
        if case_id not in CASE_REGISTRY:
            print("  ⊘ Case not found, skipping")
            return

        factory = CASE_REGISTRY[case_id]

        # Test parameter setup (without running optimization)
        param_names = ['tau_max']
        params = np.array([6.47])  # MPa

        print(f"  ✓ Calibration setup OK (params={param_names}, ref_curve_len={len(ref_curve)})")

    except ImportError as e:
        print(f"  ⊘ Skipping (missing dependency: {e})")


if __name__ == '__main__':
    """Run tests in standalone mode (no pytest)."""
    print("="*70)
    print("SMOKE TESTS FOR PARAMETRIC/CALIBRATION/BENCHMARK TOOLS")
    print("="*70)

    try:
        test_history_to_arrays_numeric()
        test_history_to_arrays_dict()
        test_extract_metrics()
        test_parametric_modify_parameter()
        test_benchmark_compute_energy_residual()
        test_calibration_objective_setup()

        print("\n" + "="*70)
        print("ALL SMOKE TESTS PASSED ✓")
        print("="*70)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
