"""
Unified history data extraction utilities.

Handles both legacy numeric history format and new multicrack dict format,
providing a consistent interface for parametric studies, calibration, and benchmarks.

History formats supported:
- Numeric: np.ndarray or list of lists/tuples with columns [step, u, P, M, ...]
- Dict: list of dicts with keys {'step', 'u', 'P', 'M', 'ncr', ...}
"""

from typing import Dict, List, Any, Union, Optional
import numpy as np


def history_to_arrays(history: Union[List, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Convert history to standardized numpy arrays.

    Parameters
    ----------
    history : list or np.ndarray
        History data in either format:
        A) Numeric: shape (n, >=3) with columns [step, u, P, M, ...]
        B) Dict: iterable of dicts with keys {"step", "u", "P", ...}

    Returns
    -------
    arrays : dict
        Dictionary with numpy arrays:
        - 'step': Load step index (int)
        - 'u': Displacement (float, SI units: m)
        - 'P': Load (float, SI units: N)
        - 'M': Moment (float, SI units: N·m, or np.nan if not available)
        - 'ncr': Number of active cracks (int, 0 if not available)
        - 'extras': dict with additional columns/keys present in history

    Examples
    --------
    >>> # Numeric format
    >>> history_num = [[0, 0.0, 0.0], [1, 1e-3, 5000.0], [2, 2e-3, 8000.0]]
    >>> arrays = history_to_arrays(history_num)
    >>> arrays['P']  # [0., 5000., 8000.]

    >>> # Dict format
    >>> history_dict = [
    ...     {'step': 0, 'u': 0.0, 'P': 0.0, 'ncr': 0},
    ...     {'step': 1, 'u': 1e-3, 'P': 5000.0, 'ncr': 1},
    ... ]
    >>> arrays = history_to_arrays(history_dict)
    >>> arrays['ncr']  # [0, 1]
    """
    if not history or len(history) == 0:
        # Empty history
        return {
            'step': np.array([], dtype=int),
            'u': np.array([]),
            'P': np.array([]),
            'M': np.array([]),
            'ncr': np.array([], dtype=int),
            'extras': {},
        }

    # Detect format
    first_row = history[0]
    is_dict_format = isinstance(first_row, dict)

    if is_dict_format:
        # Dict format: extract keys
        n_rows = len(history)

        # Extract standard keys
        step = np.array([row.get('step', i) for i, row in enumerate(history)], dtype=int)
        u = np.array([row.get('u', 0.0) for row in history])
        P = np.array([row.get('P', 0.0) for row in history])
        M = np.array([row.get('M', np.nan) for row in history])
        ncr = np.array([row.get('ncr', 0) for row in history], dtype=int)

        # Collect extra keys (beyond standard set)
        standard_keys = {'step', 'u', 'P', 'M', 'ncr'}
        all_keys = set()
        for row in history:
            all_keys.update(row.keys())
        extra_keys = all_keys - standard_keys

        extras = {}
        for key in extra_keys:
            extras[key] = np.array([row.get(key, np.nan) for row in history])

    else:
        # Numeric format: columns [step, u, P, M?, ...]
        history_arr = np.array(history)
        n_rows, n_cols = history_arr.shape

        # Extract standard columns
        step = history_arr[:, 0].astype(int)  # Column 0: step
        u = history_arr[:, 1]                 # Column 1: displacement (m)
        P = history_arr[:, 2]                 # Column 2: load (N)

        # Column 3: moment (if present)
        if n_cols > 3:
            M = history_arr[:, 3]
        else:
            M = np.full(n_rows, np.nan)

        # Number of cracks: not available in legacy numeric format
        ncr = np.zeros(n_rows, dtype=int)

        # Collect extra columns (beyond first 4)
        extras = {}
        if n_cols > 4:
            for i in range(4, n_cols):
                extras[f'col_{i}'] = history_arr[:, i]

    return {
        'step': step,
        'u': u,
        'P': P,
        'M': M,
        'ncr': ncr,
        'extras': extras,
    }


def extract_metrics(history: Union[List, np.ndarray]) -> Dict[str, float]:
    """
    Extract standard metrics from history.

    Parameters
    ----------
    history : list or np.ndarray
        History data in either format

    Returns
    -------
    metrics : dict
        Dictionary with extracted metrics:
        - P_max_kN: Peak load (kN)
        - u_at_Pmax_mm: Displacement at peak load (mm)
        - u_final_mm: Final displacement (mm)
        - energy_kNmm: Total dissipated energy ∫P du (kN·mm)
        - num_cracks: Maximum number of active cracks
        - ductility: u_final / u_at_Pmax (post-peak ductility)

    Examples
    --------
    >>> history = [[0, 0.0, 0.0], [1, 1e-3, 5000.0], [2, 3e-3, 3000.0]]
    >>> metrics = extract_metrics(history)
    >>> metrics['P_max_kN']  # 5.0
    >>> metrics['ductility']  # 3.0
    """
    arrays = history_to_arrays(history)

    if len(arrays['u']) == 0:
        # Empty history
        return {
            'P_max_kN': 0.0,
            'u_at_Pmax_mm': 0.0,
            'u_final_mm': 0.0,
            'energy_kNmm': 0.0,
            'num_cracks': 0,
            'ductility': 0.0,
        }

    # Extract arrays
    u_m = arrays['u']    # m
    P_N = arrays['P']    # N
    ncr = arrays['ncr']  # int

    # Convert units
    u_mm = u_m * 1e3  # m → mm
    P_kN = P_N / 1e3  # N → kN

    # Peak load
    idx_max = np.argmax(np.abs(P_kN))
    P_max = np.abs(P_kN[idx_max])
    u_at_Pmax = np.abs(u_mm[idx_max])

    # Final displacement
    u_final = np.abs(u_mm[-1])

    # Dissipated energy (∫ P du, trapezoidal rule)
    energy = np.trapezoid(np.abs(P_kN), u_mm)  # kN·mm

    # Number of cracks (maximum active)
    num_cracks = int(np.max(ncr))

    # Ductility (u_final / u_at_Pmax)
    ductility = u_final / u_at_Pmax if u_at_Pmax > 1e-9 else 0.0

    metrics = {
        'P_max_kN': float(P_max),
        'u_at_Pmax_mm': float(u_at_Pmax),
        'u_final_mm': float(u_final),
        'energy_kNmm': float(energy),
        'num_cracks': num_cracks,
        'ductility': float(ductility),
    }

    return metrics


def get_P_u_curve(history: Union[List, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract P-u curve in engineering units.

    Parameters
    ----------
    history : list or np.ndarray
        History data in either format

    Returns
    -------
    u_mm : np.ndarray
        Displacement in mm
    P_kN : np.ndarray
        Load in kN

    Examples
    --------
    >>> history = [[0, 0.0, 0.0], [1, 1e-3, 5000.0]]
    >>> u_mm, P_kN = get_P_u_curve(history)
    >>> u_mm  # [0., 1.]
    >>> P_kN  # [0., 5.]
    """
    arrays = history_to_arrays(history)

    u_m = arrays['u']  # m
    P_N = arrays['P']  # N

    u_mm = u_m * 1e3  # m → mm
    P_kN = P_N / 1e3  # N → kN

    return u_mm, P_kN
