"""
Generic parametric study driver for sensitivity analysis.

Allows sweeping over material/geometric parameters to study their influence
on structural response (peak load, ductility, energy dissipation, cracking).

Usage:
    python -m examples.parametric.parametric_study --case t5a1 --param Gf --values 0.05,0.1,0.2
    python -m examples.parametric.parametric_study --case vvbs3 --param tau_max --values 4.0,6.47,9.0 --mesh coarse
    python -m examples.parametric.parametric_study --case pullout --param n_bars --values 1,2,4 --plot
"""

import argparse
import sys
import time
import copy
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.gutierrez_thesis.run import CASE_REGISTRY, CASE_ALIASES, MESH_PRESETS, resolve_case_id
from examples.gutierrez_thesis.solver_interface import run_case_solver
from examples.gutierrez_thesis.history_utils import extract_metrics as extract_metrics_from_history


# ============================================================================
# Parameter modification functions
# ============================================================================

def modify_case_parameter(case_config, param_name: str, param_value: float):
    """
    Modify a parameter in the case configuration.

    Parameters
    ----------
    case_config : CaseConfig
        Case configuration object to modify (modified in-place)
    param_name : str
        Parameter name (e.g., 'Gf', 'tau_max', 'n_bars', 'E_concrete', 'f_t')
    param_value : float
        New parameter value

    Raises
    ------
    ValueError
        If parameter name is not recognized
    """
    param_lower = param_name.lower()

    # Concrete parameters
    if param_lower in ['gf', 'g_f']:
        case_config.concrete.G_f = param_value
        print(f"  → Modified concrete.G_f = {param_value} N/mm")

    elif param_lower in ['ft', 'f_t', 'tensile_strength']:
        case_config.concrete.f_t = param_value
        print(f"  → Modified concrete.f_t = {param_value} MPa")

    elif param_lower in ['fc', 'f_c', 'compressive_strength']:
        case_config.concrete.f_c = param_value
        print(f"  → Modified concrete.f_c = {param_value} MPa")

    elif param_lower in ['e', 'e_concrete', 'young_modulus']:
        case_config.concrete.E = param_value
        print(f"  → Modified concrete.E = {param_value} MPa")

    # Bond law parameters (applied to ALL rebar layers)
    elif param_lower in ['tau_max', 'taumax', 'bond_strength']:
        for i, rebar in enumerate(case_config.rebar_layers):
            if hasattr(rebar.bond_law, 'tau_max'):
                rebar.bond_law.tau_max = param_value
                print(f"  → Modified rebar_layers[{i}].bond_law.tau_max = {param_value} MPa")

    elif param_lower in ['s1', 'slip_s1']:
        for i, rebar in enumerate(case_config.rebar_layers):
            if hasattr(rebar.bond_law, 's1'):
                rebar.bond_law.s1 = param_value
                print(f"  → Modified rebar_layers[{i}].bond_law.s1 = {param_value} mm")

    elif param_lower in ['s2', 'slip_s2']:
        for i, rebar in enumerate(case_config.rebar_layers):
            if hasattr(rebar.bond_law, 's2'):
                rebar.bond_law.s2 = param_value
                print(f"  → Modified rebar_layers[{i}].bond_law.s2 = {param_value} mm")

    # FRP bond parameters (applied to ALL FRP sheets)
    elif param_lower in ['frp_tau_max', 'frp_bond_strength']:
        if case_config.frp_sheets:
            for i, frp in enumerate(case_config.frp_sheets):
                if hasattr(frp.bond_law, 'tau1'):
                    frp.bond_law.tau1 = param_value
                    print(f"  → Modified frp_sheets[{i}].bond_law.tau1 = {param_value} MPa")

    elif param_lower in ['frp_s1']:
        if case_config.frp_sheets:
            for i, frp in enumerate(case_config.frp_sheets):
                if hasattr(frp.bond_law, 's1'):
                    frp.bond_law.s1 = param_value
                    print(f"  → Modified frp_sheets[{i}].bond_law.s1 = {param_value} mm")

    # Reinforcement (NOTE: n_bars requires special handling - integer value)
    elif param_lower in ['n_bars', 'num_bars', 'bar_count']:
        n_bars_int = int(param_value)
        for i, rebar in enumerate(case_config.rebar_layers):
            rebar.n_bars = n_bars_int
            print(f"  → Modified rebar_layers[{i}].n_bars = {n_bars_int}")

    elif param_lower in ['diameter', 'bar_diameter']:
        for i, rebar in enumerate(case_config.rebar_layers):
            rebar.diameter = param_value
            print(f"  → Modified rebar_layers[{i}].diameter = {param_value} mm")

    # FRP sheet parameters
    elif param_lower in ['l_sheet', 'frp_length', 'bonded_length']:
        if case_config.frp_sheets:
            for i, frp in enumerate(case_config.frp_sheets):
                frp.bonded_length = param_value
                print(f"  → Modified frp_sheets[{i}].bonded_length = {param_value} mm")

    elif param_lower in ['t_frp', 'frp_thickness']:
        if case_config.frp_sheets:
            for i, frp in enumerate(case_config.frp_sheets):
                frp.thickness = param_value
                print(f"  → Modified frp_sheets[{i}].thickness = {param_value} mm")

    # Fibre parameters
    elif param_lower in ['rho_fibre', 'rho_f', 'fibre_content']:
        if case_config.fibres:
            case_config.fibres.rho_f = param_value
            print(f"  → Modified fibres.rho_f = {param_value} (volume fraction)")

    elif param_lower in ['l_f', 'fibre_length']:
        if case_config.fibres:
            case_config.fibres.L_f = param_value
            print(f"  → Modified fibres.L_f = {param_value} mm")

    else:
        raise ValueError(
            f"Unknown parameter '{param_name}'. Supported parameters:\n"
            f"  Concrete: Gf, f_t, f_c, E\n"
            f"  Bond: tau_max, s1, s2\n"
            f"  Rebar: n_bars, diameter\n"
            f"  FRP: frp_tau_max, frp_s1, l_sheet, t_frp\n"
            f"  Fibres: rho_fibre, l_f"
        )


# ============================================================================
# Metric extraction
# ============================================================================

def extract_case_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract key metrics from solver results.

    Parameters
    ----------
    results : dict
        Solver results dictionary with 'history' key

    Returns
    -------
    metrics : dict
        Dictionary with extracted metrics:
        - P_max_kN: Peak load
        - u_at_Pmax_mm: Displacement at peak load
        - energy_kNmm: Total dissipated energy (∫ P du)
        - num_cracks: Number of cracks at final step
        - ductility: u_final / u_at_Pmax (post-peak ductility)
    """
    history = results.get('history', [])

    # Use unified history extraction (handles both numeric and dict formats)
    metrics = extract_metrics_from_history(history)

    # Add u_final_mm for compatibility (already in extract_metrics_from_history)
    # No need to modify, already present

    return metrics


# ============================================================================
# Parametric study runner
# ============================================================================

def run_parametric_study(
    case_id: str,
    param_name: str,
    param_values: List[float],
    mesh: str = "coarse",
    output_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run parametric study for a single parameter.

    Parameters
    ----------
    case_id : str
        Case identifier (e.g., "t5a1", "vvbs3")
    param_name : str
        Parameter to vary (e.g., "Gf", "tau_max")
    param_values : list of float
        Parameter values to sweep
    mesh : str
        Mesh preset (default: "coarse" for speed)
    output_dir : str, optional
        Output directory for results (default: outputs/parametric/<case>_<param>/)

    Returns
    -------
    results : list of dict
        List of dictionaries with keys:
        - param_value: Parameter value
        - metrics: Extracted metrics (P_max, energy, etc.)
        - history: Full solver history
    """
    # Resolve case
    case_id_resolved = resolve_case_id(case_id)
    if not case_id_resolved:
        raise ValueError(f"Unknown case: {case_id}")

    # Get case factory
    factory = CASE_REGISTRY[case_id_resolved]

    # Setup output directory
    if output_dir is None:
        output_dir = f"outputs/parametric/{case_id_resolved}_{param_name}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"PARAMETRIC STUDY: {case_id_resolved}")
    print(f"Parameter: {param_name}")
    print(f"Values: {param_values}")
    print(f"Mesh: {mesh}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")

    # Get mesh factor
    mesh_factor = MESH_PRESETS.get(mesh, 1.0)

    # Run sweep
    results = []

    for i, value in enumerate(param_values):
        print(f"\n--- Run {i+1}/{len(param_values)}: {param_name} = {value} ---")

        # Create fresh case config
        case_config = factory()

        # Modify parameter
        try:
            modify_case_parameter(case_config, param_name, value)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

        # Modify output directory to avoid overwriting
        case_config.outputs.output_dir = str(output_path / f"run_{i+1:02d}_value_{value}")

        # Run solver
        t0 = time.time()
        try:
            solver_results = run_case_solver(case_config, mesh_factor=mesh_factor)
            elapsed = time.time() - t0

            # Extract metrics
            metrics = extract_case_metrics(solver_results)

            # Store results
            result_entry = {
                'param_value': value,
                'metrics': metrics,
                'history': solver_results.get('history', []),
                'elapsed_time_s': elapsed,
            }
            results.append(result_entry)

            print(f"  ✓ Completed in {elapsed:.1f}s")
            print(f"    P_max = {metrics['P_max_kN']:.2f} kN")
            print(f"    u@Pmax = {metrics['u_at_Pmax_mm']:.3f} mm")
            print(f"    Energy = {metrics['energy_kNmm']:.2f} kN·mm")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    return results


def save_parametric_results(
    results: List[Dict[str, Any]],
    param_name: str,
    case_id: str,
    output_file: str,
):
    """
    Save parametric study results to CSV.

    Parameters
    ----------
    results : list of dict
        Results from run_parametric_study()
    param_name : str
        Parameter name (for column header)
    case_id : str
        Case identifier
    output_file : str
        Output CSV file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            param_name,
            'P_max_kN',
            'u_at_Pmax_mm',
            'energy_kNmm',
            'num_cracks',
            'ductility',
            'u_final_mm',
            'elapsed_time_s',
        ])

        # Data rows
        for res in results:
            value = res['param_value']
            metrics = res['metrics']
            elapsed = res['elapsed_time_s']

            writer.writerow([
                value,
                metrics['P_max_kN'],
                metrics['u_at_Pmax_mm'],
                metrics['energy_kNmm'],
                metrics['num_cracks'],
                metrics['ductility'],
                metrics['u_final_mm'],
                elapsed,
            ])

    print(f"\n✓ Results saved to: {output_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Parametric study driver for sensitivity analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m examples.parametric.parametric_study --case t5a1 --param Gf --values 0.05,0.1,0.2
  python -m examples.parametric.parametric_study --case vvbs3 --param tau_max --values 4.0,6.47,9.0
  python -m examples.parametric.parametric_study --case pullout --param n_bars --values 1,2,4 --mesh medium
        """
    )

    parser.add_argument(
        '--case',
        type=str,
        required=True,
        help='Case identifier (e.g., t5a1, vvbs3, sorelli)'
    )
    parser.add_argument(
        '--param',
        type=str,
        required=True,
        help='Parameter to vary (e.g., Gf, tau_max, n_bars, rho_fibre)'
    )
    parser.add_argument(
        '--values',
        type=str,
        required=True,
        help='Comma-separated parameter values (e.g., 0.05,0.1,0.2)'
    )
    parser.add_argument(
        '--mesh',
        type=str,
        default='coarse',
        choices=list(MESH_PRESETS.keys()),
        help='Mesh preset (default: coarse for speed)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate plots (requires matplotlib)'
    )

    args = parser.parse_args()

    # Parse parameter values
    try:
        param_values = [float(v.strip()) for v in args.values.split(',')]
    except ValueError:
        print(f"Error: Invalid parameter values '{args.values}'")
        print("Expected comma-separated numbers (e.g., 0.05,0.1,0.2)")
        sys.exit(1)

    # Run study
    results = run_parametric_study(
        case_id=args.case,
        param_name=args.param,
        param_values=param_values,
        mesh=args.mesh,
        output_dir=args.output_dir,
    )

    # Save results
    case_id_resolved = resolve_case_id(args.case)
    output_file = f"outputs/parametric/{case_id_resolved}_{args.param}.csv"
    save_parametric_results(results, args.param, case_id_resolved, output_file)

    # Optional: Plot results
    if args.plot:
        try:
            import matplotlib.pyplot as plt

            # Extract data
            param_vals = [r['param_value'] for r in results]
            P_max_vals = [r['metrics']['P_max_kN'] for r in results]
            energy_vals = [r['metrics']['energy_kNmm'] for r in results]

            # Create plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Peak load vs parameter
            ax1.plot(param_vals, P_max_vals, 'o-', linewidth=2, markersize=8)
            ax1.set_xlabel(args.param, fontsize=12)
            ax1.set_ylabel('Peak Load (kN)', fontsize=12)
            ax1.set_title(f'{case_id_resolved}: Peak Load Sensitivity', fontsize=14)
            ax1.grid(True, alpha=0.3)

            # Energy vs parameter
            ax2.plot(param_vals, energy_vals, 's-', linewidth=2, markersize=8, color='C1')
            ax2.set_xlabel(args.param, fontsize=12)
            ax2.set_ylabel('Energy (kN·mm)', fontsize=12)
            ax2.set_title(f'{case_id_resolved}: Energy Sensitivity', fontsize=14)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_file = f"outputs/parametric/{case_id_resolved}_{args.param}.png"
            plt.savefig(plot_file, dpi=150)
            print(f"✓ Plot saved to: {plot_file}")

        except ImportError:
            print("Warning: matplotlib not available, skipping plot")


if __name__ == '__main__':
    main()
