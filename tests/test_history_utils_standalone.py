"""
Standalone tests for history_utils.py (no pytest dependency).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from examples.gutierrez_thesis.history_utils import (
    history_to_arrays,
    extract_metrics,
    get_P_u_curve,
)


def test_numeric_format():
    """Test numeric format."""
    print("Testing numeric format...")
    history = [
        [0, 0.0, 0.0, 0.0],
        [1, 1e-3, 5000.0, 100.0],
        [2, 2e-3, 8000.0, 200.0],
    ]

    arrays = history_to_arrays(history)
    assert len(arrays['step']) == 3
    assert np.allclose(arrays['u'], [0.0, 1e-3, 2e-3])
    assert np.allclose(arrays['P'], [0.0, 5000.0, 8000.0])
    print("  ✓ Numeric format OK")


def test_dict_format():
    """Test dict format (multicrack)."""
    print("Testing dict format...")
    history = [
        {'step': 0, 'u': 0.0, 'P': 0.0, 'ncr': 0},
        {'step': 1, 'u': 1e-3, 'P': 5000.0, 'ncr': 1},
        {'step': 2, 'u': 2e-3, 'P': 8000.0, 'ncr': 2},
    ]

    arrays = history_to_arrays(history)
    assert len(arrays['step']) == 3
    assert np.allclose(arrays['ncr'], [0, 1, 2])
    print("  ✓ Dict format OK")


def test_extract_metrics():
    """Test metric extraction."""
    print("Testing extract_metrics...")
    history = [
        [0, 0.0, 0.0],
        [1, 1e-3, 5000.0],
        [2, 2e-3, 8000.0],
        [3, 4e-3, 6000.0],
    ]

    metrics = extract_metrics(history)
    assert abs(metrics['P_max_kN'] - 8.0) < 1e-6
    assert abs(metrics['u_at_Pmax_mm'] - 2.0) < 1e-6
    assert abs(metrics['u_final_mm'] - 4.0) < 1e-6
    assert abs(metrics['ductility'] - 2.0) < 1e-6
    print("  ✓ extract_metrics OK")


def test_get_P_u_curve():
    """Test P-u curve extraction."""
    print("Testing get_P_u_curve...")
    history = [
        [0, 0.0, 0.0],
        [1, 1e-3, 5000.0],
        [2, 2e-3, 8000.0],
    ]

    u_mm, P_kN = get_P_u_curve(history)
    assert np.allclose(u_mm, [0.0, 1.0, 2.0])
    assert np.allclose(P_kN, [0.0, 5.0, 8.0])
    print("  ✓ get_P_u_curve OK")


if __name__ == '__main__':
    print("="*60)
    print("HISTORY_UTILS UNIT TESTS")
    print("="*60)

    try:
        test_numeric_format()
        test_dict_format()
        test_extract_metrics()
        test_get_P_u_curve()

        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
