"""
Bond parameter calibration via optimization.

Fits bond law parameters (τmax, s1, s2, Gf) to experimental P-δ curves
by minimizing RMSE + energy residual penalty.

Usage:
    python -m calibration.fit_bond_parameters --case vvbs3 --init tau_max=6.47,s1=0.02,s2=0.25
    python -m calibration.fit_bond_parameters --case t5a1 --params Gf,tau_max --bounds 0.08-0.12,12-18
"""

import argparse
import sys
import json
import time
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scipy.optimize import minimize, differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from validation.compare_curves import load_reference_curve, compute_error_metrics
from examples.gutierrez_thesis.run import CASE_REGISTRY, resolve_case_id
from examples.gutierrez_thesis.solver_interface import run_case_solver
from examples.gutierrez_thesis.history_utils import get_P_u_curve
from examples.parametric.parametric_study import modify_case_parameter, extract_case_metrics


# ============================================================================
# Objective function
# ============================================================================

def objective_function(
    params: np.ndarray,
    param_names: List[str],
    case_factory,
    ref_curve,
    mesh_factor: float = 0.5,
    lambda_energy: float = 0.1,
) -> float:
    """
    Objective function for optimization: RMSE + λ·(energy residual).

    Parameters
    ----------
    params : np.ndarray
        Parameter values to optimize (e.g., [tau_max, s1, s2])
    param_names : list of str
        Parameter names corresponding to params
    case_factory : callable
        Case factory function (e.g., create_case_08)
    ref_curve : pd.DataFrame
        Reference experimental curve with columns ['u_mm', 'P_kN']
    mesh_factor : float
        Mesh refinement factor (default: 0.5 for coarse, fast)
    lambda_energy : float
        Weight for energy residual penalty (default: 0.1)

    Returns
    -------
    cost : float
        Objective function value (lower is better)
    """
    # Create case config
    case_config = case_factory()

    # Apply parameters
    for name, value in zip(param_names, params):
        try:
            modify_case_parameter(case_config, name, value)
        except ValueError as e:
            print(f"Warning: {e}")
            return 1e9  # Invalid parameter

    # Run solver
    try:
        results = run_case_solver(case_config, mesh_factor=mesh_factor)
    except Exception as e:
        print(f"  Solver failed: {e}")
        return 1e9  # Penalize solver failures

    # Extract P-δ curve (handles both numeric and dict history formats)
    history = results.get('history', [])
    if len(history) == 0:
        return 1e9

    u_sim, P_sim = get_P_u_curve(history)  # Already in mm, kN

    # Create simulation dataframe
    import pandas as pd
    sim_curve = pd.DataFrame({'u_mm': u_sim, 'P_kN': P_sim})

    # Compute metrics
    metrics = compute_error_metrics(sim_curve, ref_curve)

    # Cost function: RMSE + λ·(energy error)
    rmse = metrics['rmse_kN']
    energy_error_pct = metrics['energy_error_pct']

    # Normalized penalty
    energy_penalty = energy_error_pct / 100.0  # Convert to fraction

    cost = rmse + lambda_energy * energy_penalty * metrics['P_max_ref_kN']

    return cost


# ============================================================================
# Calibration runner
# ============================================================================

def calibrate_bond_parameters(
    case_id: str,
    param_names: List[str],
    initial_values: Optional[List[float]] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
    method: str = 'L-BFGS-B',
    mesh: str = 'coarse',
    lambda_energy: float = 0.1,
    max_iterations: int = 50,
) -> Dict:
    """
    Calibrate bond parameters by fitting to experimental curve.

    Parameters
    ----------
    case_id : str
        Case identifier (e.g., "vvbs3", "t5a1")
    param_names : list of str
        Parameters to calibrate (e.g., ['tau_max', 's1', 's2'])
    initial_values : list of float, optional
        Initial parameter values. If None, uses defaults from case.
    bounds : list of tuples, optional
        Parameter bounds [(min1, max1), (min2, max2), ...]
        If None, uses ±30% of initial values.
    method : str
        Optimization method ('L-BFGS-B', 'differential_evolution', 'Nelder-Mead')
    mesh : str
        Mesh preset (default: 'coarse' for speed)
    lambda_energy : float
        Energy penalty weight (default: 0.1)
    max_iterations : int
        Maximum optimization iterations

    Returns
    -------
    result : dict
        Calibration results with keys:
        - optimal_params: Dict mapping param_name → optimal_value
        - cost: Final objective function value
        - iterations: Number of iterations
        - success: True if optimization converged
        - message: Optimization status message
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for calibration. Install with: pip install scipy")

    # Resolve case
    case_id_resolved = resolve_case_id(case_id)
    if not case_id_resolved:
        raise ValueError(f"Unknown case: {case_id}")

    factory = CASE_REGISTRY[case_id_resolved]

    # Load reference curve
    # Map case_id to reference file (simplified mapping)
    ref_map = {
        '04a_beam_3pb_t5a1_bosco': 't5a1',
        '04b_beam_3pb_t6a1_bosco': 't5a1',  # Similar geometry
        '08_beam_3pb_vvbs3_cfrp': 'vvbs3',
        '09_beam_4pb_fibres_sorelli': 'sorelli',
    }
    ref_id = ref_map.get(case_id_resolved, case_id)

    try:
        ref_curve = load_reference_curve(ref_id)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Reference curve not found for case {case_id}.\n"
            f"Add reference data to: validation/reference_data/{ref_id}.csv"
        )

    # Get initial values (if not provided, extract from case)
    if initial_values is None:
        case_config = factory()
        initial_values = []
        for param_name in param_names:
            # Extract default value
            param_lower = param_name.lower()
            if param_lower in ['tau_max', 'taumax']:
                val = case_config.rebar_layers[0].bond_law.tau_max
            elif param_lower == 's1':
                val = case_config.rebar_layers[0].bond_law.s1
            elif param_lower == 's2':
                val = case_config.rebar_layers[0].bond_law.s2
            elif param_lower in ['gf', 'g_f']:
                val = case_config.concrete.G_f
            elif param_lower in ['ft', 'f_t']:
                val = case_config.concrete.f_t
            else:
                val = 1.0  # Default fallback
            initial_values.append(val)

    x0 = np.array(initial_values)

    # Setup bounds (±30% if not provided)
    if bounds is None:
        bounds = [(0.7 * v, 1.3 * v) for v in x0]

    # Mesh factor
    from examples.gutierrez_thesis.run import MESH_PRESETS
    mesh_factor = MESH_PRESETS.get(mesh, 0.5)

    print(f"\n{'='*70}")
    print(f"CALIBRATION: {case_id_resolved}")
    print(f"Parameters: {param_names}")
    print(f"Initial values: {x0}")
    print(f"Bounds: {bounds}")
    print(f"Method: {method}")
    print(f"Mesh: {mesh}")
    print(f"{'='*70}\n")

    # Define objective wrapper
    iteration_count = [0]

    def obj_wrapper(params):
        iteration_count[0] += 1
        cost = objective_function(
            params,
            param_names,
            factory,
            ref_curve,
            mesh_factor=mesh_factor,
            lambda_energy=lambda_energy,
        )
        print(f"  Iteration {iteration_count[0]}: params={params}, cost={cost:.6f}")
        return cost

    # Run optimization
    t0 = time.time()

    if method.lower() == 'differential_evolution':
        # Global optimizer (slower but more robust)
        opt_result = differential_evolution(
            obj_wrapper,
            bounds,
            maxiter=max_iterations,
            atol=1e-3,
            tol=1e-3,
            workers=1,
        )
    else:
        # Local optimizer (faster)
        opt_result = minimize(
            obj_wrapper,
            x0,
            method=method,
            bounds=bounds,
            options={'maxiter': max_iterations},
        )

    elapsed = time.time() - t0

    # Extract results
    optimal_params = {
        name: float(val) for name, val in zip(param_names, opt_result.x)
    }

    result = {
        'case_id': case_id_resolved,
        'optimal_params': optimal_params,
        'cost': float(opt_result.fun),
        'iterations': iteration_count[0],
        'success': bool(opt_result.success),
        'message': str(opt_result.message),
        'elapsed_time_s': elapsed,
    }

    print(f"\n{'='*70}")
    print("CALIBRATION COMPLETE")
    print(f"{'='*70}")
    print(f"Iterations: {result['iterations']}")
    print(f"Final cost: {result['cost']:.6f}")
    print(f"Success: {result['success']}")
    print(f"Time: {elapsed:.1f}s\n")
    print("Optimal parameters:")
    for name, value in optimal_params.items():
        print(f"  {name} = {value:.6f}")
    print(f"{'='*70}\n")

    return result


def save_calibration_results(result: Dict, output_file: str):
    """
    Save calibration results to JSON file.

    Parameters
    ----------
    result : dict
        Calibration result from calibrate_bond_parameters()
    output_file : str
        Output JSON file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"✓ Calibration results saved to: {output_path}")


# ============================================================================
# CLI
# ============================================================================

def run_calibration_cli():
    """Command-line interface for calibration."""
    parser = argparse.ArgumentParser(
        description="Calibrate bond parameters to experimental curves",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m calibration.fit_bond_parameters --case vvbs3 --init tau_max=6.47,s1=0.02,s2=0.25
  python -m calibration.fit_bond_parameters --case t5a1 --params Gf,tau_max --bounds 0.08-0.12,12-18
  python -m calibration.fit_bond_parameters --case vvbs3 --params tau_max,s1 --method differential_evolution
        """
    )

    parser.add_argument(
        '--case',
        type=str,
        required=True,
        help='Case identifier (e.g., vvbs3, t5a1)'
    )
    parser.add_argument(
        '--params',
        type=str,
        default='tau_max,s1,s2',
        help='Comma-separated parameter names (default: tau_max,s1,s2)'
    )
    parser.add_argument(
        '--init',
        type=str,
        help='Initial values as key=value pairs (e.g., tau_max=6.5,s1=0.02)'
    )
    parser.add_argument(
        '--bounds',
        type=str,
        help='Bounds as min-max pairs (e.g., 5-8,0.01-0.05,0.2-0.3)'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='L-BFGS-B',
        choices=['L-BFGS-B', 'Nelder-Mead', 'differential_evolution'],
        help='Optimization method (default: L-BFGS-B)'
    )
    parser.add_argument(
        '--mesh',
        type=str,
        default='coarse',
        help='Mesh preset (default: coarse for speed)'
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        default=50,
        help='Maximum iterations (default: 50)'
    )
    parser.add_argument(
        '--lambda-energy',
        type=float,
        default=0.1,
        help='Energy penalty weight (default: 0.1)'
    )

    args = parser.parse_args()

    # Parse parameter names
    param_names = [p.strip() for p in args.params.split(',')]

    # Parse initial values
    initial_values = None
    if args.init:
        try:
            init_dict = {}
            for pair in args.init.split(','):
                key, val = pair.split('=')
                init_dict[key.strip()] = float(val.strip())

            # Map to param_names order
            initial_values = [init_dict[name] for name in param_names]
        except (ValueError, KeyError) as e:
            print(f"Error parsing initial values: {e}")
            sys.exit(1)

    # Parse bounds
    bounds = None
    if args.bounds:
        try:
            bounds = []
            for pair in args.bounds.split(','):
                min_val, max_val = pair.split('-')
                bounds.append((float(min_val), float(max_val)))

            if len(bounds) != len(param_names):
                print(f"Error: Number of bounds ({len(bounds)}) must match number of parameters ({len(param_names)})")
                sys.exit(1)
        except ValueError as e:
            print(f"Error parsing bounds: {e}")
            sys.exit(1)

    # Run calibration
    result = calibrate_bond_parameters(
        case_id=args.case,
        param_names=param_names,
        initial_values=initial_values,
        bounds=bounds,
        method=args.method,
        mesh=args.mesh,
        lambda_energy=args.lambda_energy,
        max_iterations=args.max_iter,
    )

    # Save results
    output_file = f"calibration/results_{args.case}.json"
    save_calibration_results(result, output_file)


if __name__ == '__main__':
    run_calibration_cli()
