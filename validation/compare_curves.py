"""
Curve comparison and error metrics for thesis validation.

Functions for loading simulation results, reference experimental data,
and computing quantitative error metrics (RMSE, peak load error, energy error).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional
from scipy.interpolate import interp1d
import warnings
from xfem_clean.utils.numpy_compat import trapezoid


def check_monotonicity(u: np.ndarray) -> Tuple[bool, str]:
    """
    Check if displacement array is monotonically increasing.

    Parameters
    ----------
    u : np.ndarray
        Displacement array

    Returns
    -------
    is_monotonic : bool
        True if monotonically increasing
    message : str
        Warning message if not monotonic
    """
    if len(u) < 2:
        return True, ""

    # Check if strictly increasing
    diff = np.diff(u)
    if np.all(diff > 0):
        return True, ""

    # Check if non-decreasing (allowing equal values)
    if np.all(diff >= 0):
        return True, "Warning: Displacement has repeated values (non-strictly increasing)"

    # Not monotonic
    n_violations = np.sum(diff < 0)
    return False, f"Error: Displacement is not monotonic ({n_violations} violations)"


def detect_and_convert_units(df: pd.DataFrame, expected_u_unit: str = 'mm',
                              expected_P_unit: str = 'kN') -> pd.DataFrame:
    """
    Detect and convert units if necessary.

    Auto-detects if data is in wrong units (e.g., m instead of mm, N instead of kN)
    and converts to expected units.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'u_mm' and 'P_kN' columns
    expected_u_unit : str
        Expected displacement unit ('mm' or 'm')
    expected_P_unit : str
        Expected force unit ('kN' or 'N')

    Returns
    -------
    df_converted : pd.DataFrame
        DataFrame with converted units if necessary
    """
    df_out = df.copy()

    # Check displacement magnitude
    u_max = df['u_mm'].max()
    u_min = df['u_mm'].min()

    # If max displacement is < 0.1, likely in meters instead of mm
    if u_max < 0.1 and u_max > 0:
        warnings.warn(
            f"Displacement magnitude ({u_max:.4f}) suggests units are in meters, not mm. "
            f"Converting to mm (×1000)."
        )
        df_out['u_mm'] = df['u_mm'] * 1000.0

    # Check force magnitude
    P_max = df['P_kN'].max()

    # If max force is > 10000, likely in N instead of kN
    if P_max > 10000:
        warnings.warn(
            f"Force magnitude ({P_max:.0f}) suggests units are in N, not kN. "
            f"Converting to kN (÷1000)."
        )
        df_out['P_kN'] = df['P_kN'] / 1000.0

    return df_out


def is_placeholder_data(df: pd.DataFrame, threshold_points: int = 25,
                        smoothness_threshold: float = 0.01) -> Tuple[bool, str]:
    """
    Detect if reference data is likely a synthetic placeholder.

    Heuristics:
    - Too few data points (< threshold_points)
    - Perfectly smooth curve (low variance in 2nd derivative)
    - Evenly spaced points (constant Δu)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'u_mm' and 'P_kN' columns
    threshold_points : int
        Minimum expected points for real digitized data
    smoothness_threshold : float
        Threshold for 2nd derivative variance (lower = smoother)

    Returns
    -------
    is_placeholder : bool
        True if data appears to be synthetic
    reason : str
        Explanation if placeholder detected
    """
    n_points = len(df)
    u = df['u_mm'].values
    P = df['P_kN'].values

    reasons = []

    # Check 1: Too few points
    if n_points < threshold_points:
        reasons.append(f"Only {n_points} data points (expected ≥{threshold_points} for real data)")

    # Check 2: Perfectly evenly spaced
    if n_points > 2:
        du = np.diff(u)
        du_std = np.std(du)
        du_mean = np.mean(du)
        if du_std / du_mean < 0.01:  # Coefficient of variation < 1%
            reasons.append("Points are perfectly evenly spaced (unlikely for digitized data)")

    # Check 3: Suspiciously smooth (low 2nd derivative variance)
    if n_points > 3:
        d2P = np.diff(P, n=2)
        if np.std(d2P) < smoothness_threshold:
            reasons.append(f"Curve is suspiciously smooth (2nd derivative std={np.std(d2P):.4f})")

    if reasons:
        return True, "; ".join(reasons)
    else:
        return False, ""


def load_simulation_curve(case_id: str, mesh: str = "medium", output_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load simulation P-δ curve from CSV output.

    Parameters
    ----------
    case_id : str
        Case identifier (e.g., "04a_beam_3pb_t5a1_bosco", "08_beam_3pb_vvbs3_cfrp")
    mesh : str
        Mesh preset ("coarse", "medium", "fine")
    output_dir : str, optional
        Override output directory. If None, uses default location.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns ['u_mm', 'P_kN'] (displacement, force)

    Raises
    ------
    FileNotFoundError
        If simulation output file not found
    """
    if output_dir is None:
        # Default location
        base_dir = Path(__file__).parent.parent
        output_dir = base_dir / "examples" / "gutierrez_thesis" / "outputs" / f"case_{case_id}"

    output_dir = Path(output_dir)
    csv_file = output_dir / "load_displacement.csv"

    if not csv_file.exists():
        raise FileNotFoundError(
            f"Simulation output not found: {csv_file}\n"
            f"Run the case first: python -m examples.gutierrez_thesis.run --case {case_id} --mesh {mesh}"
        )

    # Load CSV (format: step, u_mm, P_kN, M_kNm, ...)
    df_full = pd.read_csv(csv_file)

    # Extract P-δ curve
    df = pd.DataFrame({
        'u_mm': df_full['u_mm'].values,
        'P_kN': df_full['P_kN'].values,
    })

    return df


def load_reference_curve(case_id: str, check_quality: bool = True) -> pd.DataFrame:
    """
    Load reference experimental P-δ curve from digitized data.

    Parameters
    ----------
    case_id : str
        Case identifier (e.g., "t5a1", "vvbs3", "sorelli")
    check_quality : bool
        If True, perform data quality checks (monotonicity, units, placeholder detection)

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns ['u_mm', 'P_kN']

    Raises
    ------
    FileNotFoundError
        If reference data file not found

    Warnings
    --------
    Emits warnings if data appears to be synthetic/placeholder or has quality issues.

    Notes
    -----
    Reference data files should be placed in validation/reference_data/<case_id>.csv
    These are digitized from thesis figures using tools like WebPlotDigitizer.
    See validation/reference_data/SOURCES.md for data provenance.
    """
    ref_dir = Path(__file__).parent / "reference_data"
    ref_file = ref_dir / f"{case_id}.csv"

    if not ref_file.exists():
        raise FileNotFoundError(
            f"Reference data not found: {ref_file}\n"
            f"Expected format: CSV with columns 'u_mm,P_kN'\n"
            f"See validation/reference_data/SOURCES.md for data provenance guidelines."
        )

    df = pd.read_csv(ref_file)

    # Validate columns
    required_cols = ['u_mm', 'P_kN']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Reference CSV must have columns: {required_cols}")

    if check_quality:
        # Check monotonicity
        is_monotonic, mono_msg = check_monotonicity(df['u_mm'].values)
        if not is_monotonic:
            warnings.warn(f"Reference data for '{case_id}': {mono_msg}")
        elif mono_msg:
            # Non-critical warning (e.g., repeated values)
            warnings.warn(f"Reference data for '{case_id}': {mono_msg}")

        # Auto-detect and convert units
        df = detect_and_convert_units(df)

        # Check if placeholder
        is_placeholder, placeholder_reason = is_placeholder_data(df)
        if is_placeholder:
            warnings.warn(
                f"\n{'='*70}\n"
                f"WARNING: Reference data for '{case_id}' appears to be SYNTHETIC/PLACEHOLDER\n"
                f"Reason: {placeholder_reason}\n"
                f"See validation/reference_data/SOURCES.md for digitization guidelines.\n"
                f"Validation results may not be meaningful until real experimental data is added.\n"
                f"{'='*70}"
            )

    return df


def compute_error_metrics(sim: pd.DataFrame, ref: pd.DataFrame,
                          interpolation_points: int = 100) -> Dict[str, float]:
    """
    Compute quantitative error metrics between simulation and reference curves.

    Metrics computed:
    - RMSE: Root mean square error over interpolated curve
    - peak_error_pct: Relative error in peak load |ΔPmax|/Pmax_ref × 100%
    - energy_error_pct: Relative error in dissipated energy |ΔE|/E_ref × 100%
    - u_at_peak_error_pct: Relative error in displacement at peak load

    Parameters
    ----------
    sim : pd.DataFrame
        Simulation curve with columns ['u_mm', 'P_kN']
    ref : pd.DataFrame
        Reference curve with columns ['u_mm', 'P_kN']
    interpolation_points : int
        Number of points for interpolation (default: 100)

    Returns
    -------
    metrics : dict
        Dictionary with error metrics
    """
    # Extract arrays
    u_sim = sim['u_mm'].values
    P_sim = sim['P_kN'].values
    u_ref = ref['u_mm'].values
    P_ref = ref['P_kN'].values

    # --- 1. Peak load error ---
    P_max_sim = np.max(P_sim)
    P_max_ref = np.max(P_ref)
    peak_error_pct = abs(P_max_sim - P_max_ref) / P_max_ref * 100.0

    # Displacement at peak
    u_at_peak_sim = u_sim[np.argmax(P_sim)]
    u_at_peak_ref = u_ref[np.argmax(P_ref)]
    u_at_peak_error_pct = abs(u_at_peak_sim - u_at_peak_ref) / u_at_peak_ref * 100.0 if u_at_peak_ref > 0 else 0.0

    # --- 2. Energy error (∫ P du) ---
    # Compute energies using trapezoidal rule
    energy_sim = trapezoid(P_sim, u_sim)
    energy_ref = trapezoid(P_ref, u_ref)
    energy_error_pct = abs(energy_sim - energy_ref) / energy_ref * 100.0 if energy_ref > 0 else 0.0

    # --- 3. RMSE over interpolated curves ---
    # Find common displacement range
    u_min = max(u_sim.min(), u_ref.min())
    u_max = min(u_sim.max(), u_ref.max())

    if u_max <= u_min:
        # No overlap
        rmse = np.inf
        mae = np.inf
        r_squared = 0.0
    else:
        # Create common grid
        u_common = np.linspace(u_min, u_max, interpolation_points)

        # Interpolate both curves
        try:
            interp_sim = interp1d(u_sim, P_sim, kind='linear', bounds_error=False, fill_value='extrapolate')
            interp_ref = interp1d(u_ref, P_ref, kind='linear', bounds_error=False, fill_value='extrapolate')

            P_sim_interp = interp_sim(u_common)
            P_ref_interp = interp_ref(u_common)

            # Compute RMSE
            rmse = np.sqrt(np.mean((P_sim_interp - P_ref_interp)**2))

            # Compute MAE (Mean Absolute Error)
            mae = np.mean(np.abs(P_sim_interp - P_ref_interp))

            # Compute R²
            ss_res = np.sum((P_ref_interp - P_sim_interp)**2)
            ss_tot = np.sum((P_ref_interp - np.mean(P_ref_interp))**2)
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        except Exception as e:
            print(f"Warning: Interpolation failed: {e}")
            rmse = np.inf
            mae = np.inf
            r_squared = 0.0

    # --- 4. Normalized RMSE (%) ---
    rmse_normalized_pct = (rmse / P_max_ref * 100.0) if P_max_ref > 0 else np.inf

    metrics = {
        'rmse_kN': rmse,
        'rmse_normalized_pct': rmse_normalized_pct,
        'mae_kN': mae,
        'r_squared': r_squared,
        'peak_error_pct': peak_error_pct,
        'energy_error_pct': energy_error_pct,
        'u_at_peak_error_pct': u_at_peak_error_pct,
        'P_max_sim_kN': P_max_sim,
        'P_max_ref_kN': P_max_ref,
        'energy_sim_kNmm': energy_sim,
        'energy_ref_kNmm': energy_ref,
    }

    return metrics


def generate_validation_report(case_id: str, metrics: Dict[str, float],
                                 tolerances: Optional[Dict[str, float]] = None) -> str:
    """
    Generate human-readable validation report.

    Parameters
    ----------
    case_id : str
        Case identifier
    metrics : dict
        Error metrics from compute_error_metrics()
    tolerances : dict, optional
        Tolerance thresholds (e.g., {'peak_error_pct': 10.0, 'energy_error_pct': 15.0})

    Returns
    -------
    report : str
        Formatted validation report
    """
    if tolerances is None:
        tolerances = {
            'peak_error_pct': 10.0,      # ± 10%
            'energy_error_pct': 15.0,    # ± 15%
            'rmse_normalized_pct': 5.0,  # 5% of peak load
        }

    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"VALIDATION REPORT: {case_id}")
    lines.append(f"{'='*70}\n")

    # Peak load
    peak_err = metrics['peak_error_pct']
    peak_pass = "✓" if peak_err <= tolerances['peak_error_pct'] else "✗"
    lines.append(f"{peak_pass} Peak Load Error: {peak_err:.2f}% (tolerance: {tolerances['peak_error_pct']}%)")
    lines.append(f"    Simulation: {metrics['P_max_sim_kN']:.2f} kN")
    lines.append(f"    Reference:  {metrics['P_max_ref_kN']:.2f} kN\n")

    # Energy
    energy_err = metrics['energy_error_pct']
    energy_pass = "✓" if energy_err <= tolerances['energy_error_pct'] else "✗"
    lines.append(f"{energy_pass} Energy Error: {energy_err:.2f}% (tolerance: {tolerances['energy_error_pct']}%)")
    lines.append(f"    Simulation: {metrics['energy_sim_kNmm']:.2f} kN·mm")
    lines.append(f"    Reference:  {metrics['energy_ref_kNmm']:.2f} kN·mm\n")

    # RMSE
    rmse_norm = metrics['rmse_normalized_pct']
    rmse_pass = "✓" if rmse_norm <= tolerances['rmse_normalized_pct'] else "✗"
    lines.append(f"{rmse_pass} RMSE (normalized): {rmse_norm:.2f}% (tolerance: {tolerances['rmse_normalized_pct']}%)")
    lines.append(f"    RMSE: {metrics['rmse_kN']:.3f} kN")
    lines.append(f"    MAE:  {metrics['mae_kN']:.3f} kN")
    lines.append(f"    R²:   {metrics['r_squared']:.4f}\n")

    # Displacement at peak
    lines.append(f"  Displacement at Peak Error: {metrics['u_at_peak_error_pct']:.2f}%\n")

    # Overall pass/fail
    all_pass = (
        peak_err <= tolerances['peak_error_pct'] and
        energy_err <= tolerances['energy_error_pct'] and
        rmse_norm <= tolerances['rmse_normalized_pct']
    )

    if all_pass:
        lines.append(f"{'='*70}")
        lines.append("✓✓✓ VALIDATION PASSED ✓✓✓")
        lines.append(f"{'='*70}\n")
    else:
        lines.append(f"{'='*70}")
        lines.append("✗✗✗ VALIDATION FAILED ✗✗✗")
        lines.append(f"{'='*70}\n")

    return "\n".join(lines)


def save_validation_summary(results: Dict[str, Dict[str, float]], output_file: str = "validation/summary_validation.csv"):
    """
    Save validation results to CSV file.

    Parameters
    ----------
    results : dict
        Dictionary mapping case_id → metrics dict
    output_file : str
        Output CSV path
    """
    rows = []
    for case_id, metrics in results.items():
        row = {'case_id': case_id}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✓ Validation summary saved to: {output_path}")
