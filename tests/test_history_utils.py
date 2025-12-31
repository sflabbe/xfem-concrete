"""
Unit tests for history_utils.py

Tests the unified history extraction interface for both numeric and dict formats.
"""

import pytest
import numpy as np
from examples.gutierrez_thesis.history_utils import (
    history_to_arrays,
    extract_metrics,
    get_P_u_curve,
)


def test_history_to_arrays_numeric():
    """Test history_to_arrays with numeric format."""
    # Numeric format: [step, u, P, M, ...]
    history = [
        [0, 0.0, 0.0, 0.0],
        [1, 1e-3, 5000.0, 100.0],
        [2, 2e-3, 8000.0, 200.0],
        [3, 3e-3, 6000.0, 150.0],
    ]

    arrays = history_to_arrays(history)

    assert len(arrays['step']) == 4
    assert np.allclose(arrays['step'], [0, 1, 2, 3])
    assert np.allclose(arrays['u'], [0.0, 1e-3, 2e-3, 3e-3])
    assert np.allclose(arrays['P'], [0.0, 5000.0, 8000.0, 6000.0])
    assert np.allclose(arrays['M'], [0.0, 100.0, 200.0, 150.0])
    assert np.allclose(arrays['ncr'], [0, 0, 0, 0])  # Not available in numeric format


def test_history_to_arrays_dict():
    """Test history_to_arrays with dict format (multicrack)."""
    # Dict format: list of dicts
    history = [
        {'step': 0, 'u': 0.0, 'P': 0.0, 'M': 0.0, 'ncr': 0},
        {'step': 1, 'u': 1e-3, 'P': 5000.0, 'M': 100.0, 'ncr': 1},
        {'step': 2, 'u': 2e-3, 'P': 8000.0, 'M': 200.0, 'ncr': 2},
        {'step': 3, 'u': 3e-3, 'P': 6000.0, 'M': 150.0, 'ncr': 2},
    ]

    arrays = history_to_arrays(history)

    assert len(arrays['step']) == 4
    assert np.allclose(arrays['step'], [0, 1, 2, 3])
    assert np.allclose(arrays['u'], [0.0, 1e-3, 2e-3, 3e-3])
    assert np.allclose(arrays['P'], [0.0, 5000.0, 8000.0, 6000.0])
    assert np.allclose(arrays['M'], [0.0, 100.0, 200.0, 150.0])
    assert np.allclose(arrays['ncr'], [0, 1, 2, 2])


def test_history_to_arrays_empty():
    """Test history_to_arrays with empty history."""
    history = []

    arrays = history_to_arrays(history)

    assert len(arrays['step']) == 0
    assert len(arrays['u']) == 0
    assert len(arrays['P']) == 0


def test_extract_metrics_numeric():
    """Test extract_metrics with numeric format."""
    # Simple monotonic loading: peak at step 2, then softening
    history = [
        [0, 0.0, 0.0],
        [1, 1e-3, 5000.0],
        [2, 2e-3, 8000.0],
        [3, 4e-3, 6000.0],
    ]

    metrics = extract_metrics(history)

    assert metrics['P_max_kN'] == pytest.approx(8.0, abs=1e-6)
    assert metrics['u_at_Pmax_mm'] == pytest.approx(2.0, abs=1e-6)
    assert metrics['u_final_mm'] == pytest.approx(4.0, abs=1e-6)
    assert metrics['ductility'] == pytest.approx(2.0, abs=1e-6)  # 4.0 / 2.0
    assert metrics['num_cracks'] == 0  # Not available in numeric format
    assert metrics['energy_kNmm'] > 0  # Positive work


def test_extract_metrics_dict():
    """Test extract_metrics with dict format (multicrack)."""
    history = [
        {'step': 0, 'u': 0.0, 'P': 0.0, 'ncr': 0},
        {'step': 1, 'u': 1e-3, 'P': 5000.0, 'ncr': 1},
        {'step': 2, 'u': 2e-3, 'P': 8000.0, 'ncr': 2},
        {'step': 3, 'u': 4e-3, 'P': 6000.0, 'ncr': 2},
    ]

    metrics = extract_metrics(history)

    assert metrics['P_max_kN'] == pytest.approx(8.0, abs=1e-6)
    assert metrics['u_at_Pmax_mm'] == pytest.approx(2.0, abs=1e-6)
    assert metrics['u_final_mm'] == pytest.approx(4.0, abs=1e-6)
    assert metrics['ductility'] == pytest.approx(2.0, abs=1e-6)
    assert metrics['num_cracks'] == 2  # Max ncr


def test_get_P_u_curve():
    """Test get_P_u_curve extraction."""
    history = [
        [0, 0.0, 0.0],
        [1, 1e-3, 5000.0],
        [2, 2e-3, 8000.0],
    ]

    u_mm, P_kN = get_P_u_curve(history)

    assert np.allclose(u_mm, [0.0, 1.0, 2.0])
    assert np.allclose(P_kN, [0.0, 5.0, 8.0])


def test_history_to_arrays_extra_keys():
    """Test that extra keys in dict format are captured."""
    history = [
        {'step': 0, 'u': 0.0, 'P': 0.0, 'custom_field': 123.0},
        {'step': 1, 'u': 1e-3, 'P': 5000.0, 'custom_field': 456.0},
    ]

    arrays = history_to_arrays(history)

    assert 'custom_field' in arrays['extras']
    assert np.allclose(arrays['extras']['custom_field'], [123.0, 456.0])


def test_history_to_arrays_extra_columns():
    """Test that extra columns in numeric format are captured."""
    # Numeric format with extra columns beyond [step, u, P, M]
    history = [
        [0, 0.0, 0.0, 0.0, 1.0, 2.0],
        [1, 1e-3, 5000.0, 100.0, 3.0, 4.0],
    ]

    arrays = history_to_arrays(history)

    assert 'col_4' in arrays['extras']
    assert 'col_5' in arrays['extras']
    assert np.allclose(arrays['extras']['col_4'], [1.0, 3.0])
    assert np.allclose(arrays['extras']['col_5'], [2.0, 4.0])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
