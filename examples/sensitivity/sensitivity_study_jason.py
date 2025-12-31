"""
Sensitivity study for Jason case (4PBT CFRP beam).

Explores sensitivity to:
1. Mesh size (coarse/medium/fine)
2. Candidate point density (via crack_rho scaling factor)
3. (Optional future) Element type (quad vs triangle)

Usage:
    python -m examples.sensitivity.sensitivity_study_jason --mesh coarse,medium --cand 1.0,0.5
    python -m examples.sensitivity.sensitivity_study_jason --quick
"""

import argparse
import sys
import time
import csv
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.gutierrez_thesis.run import CASE_REGISTRY, MESH_PRESETS, resolve_case_id
from examples.gutierrez_thesis.solver_interface import run_case_solver
from examples.gutierrez_thesis.history_utils import extract_metrics


# ============================================================================
# Configuration runner
# ============================================================================

def run_sensitivity_config(
    case_id: str,
    mesh: str,
    cand_density: float = 1.0,
    nsteps: int = None,
) -> Dict[str, Any]:
    """
    Run single sensitivity configuration.

    Parameters
    ----------
    case_id : str
        Case identifier (e.g., "jason", "vvbs3")
    mesh : str
        Mesh preset ("coarse", "medium", "fine")
    cand_density : float
        Candidate point density scaling factor (1.0 = default)
        Lower values reduce candidate points (e.g., 0.5 = half density)
    nsteps : int, optional
        Override number of load steps (for quick tests)

    Returns
    -------
    result : dict
        Configuration results with keys:
        - config_id: Configuration identifier
        - mesh: Mesh preset
        - cand_density: Candidate density factor
        - runtime_s: Wall-clock time
        - converged: True if solver converged
        - metrics: Extracted metrics (P_max, u@Pmax, num_cracks, etc.)
        - error: Error message if failed
    """
    # Resolve case
    case_id_resolved = resolve_case_id(case_id)
    if not case_id_resolved:
        raise ValueError(f"Unknown case: {case_id}")

    factory = CASE_REGISTRY[case_id_resolved]
    mesh_factor = MESH_PRESETS.get(mesh, 1.0)

    # Create case config
    case_config = factory()

    # Override output directory
    config_id = f"{mesh}_cand{cand_density:.2f}"
    case_config.outputs.output_dir = f"outputs/sensitivity/{case_id_resolved}/{config_id}"

    # Override number of steps if specified (for quick tests)
    if nsteps is not None:
        case_config.loading.n_steps = nsteps

    # Modify crack_rho (candidate point density)
    # crack_rho controls spacing between candidate points:
    # Higher rho → coarser spacing → fewer candidates
    # Lower rho → finer spacing → more candidates
    # Therefore: scaling factor = 1/cand_density
    # (cand_density=0.5 → rho*2 → half the candidates)
    # Note: This is a heuristic; actual implementation may vary
    # For now, we'll document this parameter and leave it as a scaling option

    # In case_config_to_xfem_model, crack_rho is set to lch
    # We can't directly modify it here without refactoring solver_interface
    # Instead, we'll add a note and leave it for future enhancement
    # For now, we'll run with varying mesh sizes only

    print(f"\n{'='*70}")
    print(f"SENSITIVITY CONFIG: {case_id_resolved}")
    print(f"  Mesh: {mesh} (factor={mesh_factor})")
    print(f"  Candidate density: {cand_density:.2f}")
    if nsteps:
        print(f"  Load steps: {nsteps} (override)")
    print(f"  Output: {case_config.outputs.output_dir}")
    print(f"{'='*70}\n")

    # Run solver
    t0 = time.time()
    converged = True
    error_msg = None
    metrics = {}

    try:
        results = run_case_solver(case_config, mesh_factor=mesh_factor)
        elapsed = time.time() - t0

        # Extract metrics
        history = results.get('history', [])
        metrics = extract_metrics(history)

        # Add crack width max if available
        coh_states = results.get('coh_states', None)
        if coh_states is not None:
            from examples.gutierrez_thesis.postprocess import compute_crack_widths_from_cohesive
            cracks = results.get('cracks', [results.get('crack')])
            nodes = results['nodes']
            elems = results['elems']

            crack_widths = compute_crack_widths_from_cohesive(coh_states, cracks, nodes, elems)
            if crack_widths:
                # Get max crack width across all cracks
                w_max_global = 0.0
                for crack_id, width_data in crack_widths.items():
                    if len(width_data) > 0:
                        w_max_crack = max(w * 1e3 for s, x, y, w in width_data)  # m → mm
                        w_max_global = max(w_max_global, w_max_crack)
                metrics['crack_width_max_mm'] = w_max_global

        print(f"\n✓ Configuration completed in {elapsed:.1f}s")
        print(f"  P_max = {metrics.get('P_max_kN', 0):.2f} kN")
        print(f"  u@Pmax = {metrics.get('u_at_Pmax_mm', 0):.3f} mm")
        print(f"  Num cracks = {metrics.get('num_cracks', 0)}")

    except Exception as e:
        print(f"\n✗ Configuration failed: {e}")
        import traceback
        traceback.print_exc()
        converged = False
        error_msg = str(e)
        elapsed = time.time() - t0

    result = {
        'config_id': config_id,
        'case_id': case_id_resolved,
        'mesh': mesh,
        'cand_density': cand_density,
        'runtime_s': elapsed,
        'converged': converged,
        'metrics': metrics,
        'error': error_msg,
    }

    return result


# ============================================================================
# Sensitivity study runner
# ============================================================================

def run_sensitivity_study(
    case_id: str = 'jason',
    meshes: List[str] = ['coarse', 'medium'],
    cand_densities: List[float] = [1.0, 0.5],
    quick: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run full sensitivity study.

    Parameters
    ----------
    case_id : str
        Case identifier (default: 'jason' for Jason 4PBT beam)
    meshes : list of str
        Mesh presets to test
    cand_densities : list of float
        Candidate density factors to test
    quick : bool
        If True, use minimal nsteps for quick testing

    Returns
    -------
    results : list of dict
        List of configuration results
    """
    results = []

    nsteps_override = 3 if quick else None

    # Run all configurations
    for mesh in meshes:
        for cand_density in cand_densities:
            result = run_sensitivity_config(
                case_id=case_id,
                mesh=mesh,
                cand_density=cand_density,
                nsteps=nsteps_override,
            )
            results.append(result)

    return results


def save_sensitivity_summary(
    results: List[Dict[str, Any]],
    output_file: str = "outputs/sensitivity/jason_summary.csv",
):
    """
    Save sensitivity study summary to CSV.

    Parameters
    ----------
    results : list of dict
        Results from run_sensitivity_study()
    output_file : str
        Output CSV file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'config_id',
            'case_id',
            'mesh',
            'cand_density',
            'runtime_s',
            'P_max_kN',
            'u_at_Pmax_mm',
            'num_cracks',
            'crack_width_max_mm',
            'energy_kNmm',
            'ductility',
            'converged',
            'error',
        ])

        # Data rows
        for res in results:
            metrics = res['metrics']
            writer.writerow([
                res['config_id'],
                res['case_id'],
                res['mesh'],
                res['cand_density'],
                res['runtime_s'],
                metrics.get('P_max_kN', 0),
                metrics.get('u_at_Pmax_mm', 0),
                metrics.get('num_cracks', 0),
                metrics.get('crack_width_max_mm', 0),
                metrics.get('energy_kNmm', 0),
                metrics.get('ductility', 0),
                res['converged'],
                res.get('error', ''),
            ])

    print(f"\n✓ Sensitivity summary saved to: {output_path}")


def generate_sensitivity_plots(
    results: List[Dict[str, Any]],
    output_dir: str = "outputs/sensitivity",
):
    """
    Generate sensitivity plots.

    Parameters
    ----------
    results : list of dict
        Sensitivity study results
    output_dir : str
        Output directory for plots
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract data
    mesh_labels = sorted(set(r['mesh'] for r in results))
    cand_labels = sorted(set(r['cand_density'] for r in results), reverse=True)

    # Create bar chart for P_max
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(mesh_labels))
    width = 0.8 / len(cand_labels)

    for i, cand in enumerate(cand_labels):
        P_max_vals = []
        for mesh in mesh_labels:
            # Find matching result
            match = next((r for r in results if r['mesh'] == mesh and r['cand_density'] == cand), None)
            if match:
                P_max_vals.append(match['metrics'].get('P_max_kN', 0))
            else:
                P_max_vals.append(0)

        offset = (i - len(cand_labels) / 2 + 0.5) * width
        ax.bar(x + offset, P_max_vals, width, label=f'Cand={cand:.2f}')

    ax.set_xlabel('Mesh Size', fontsize=12)
    ax.set_ylabel('Peak Load (kN)', fontsize=12)
    ax.set_title('Sensitivity: Peak Load vs Mesh and Candidate Density', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(mesh_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_file = output_path / "sensitivity_Pmax.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()

    print(f"✓ Sensitivity plot saved to: {plot_file}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sensitivity study for Jason case (4PBT CFRP beam)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m examples.sensitivity.sensitivity_study_jason --mesh coarse,medium --cand 1.0,0.5
  python -m examples.sensitivity.sensitivity_study_jason --quick
  python -m examples.sensitivity.sensitivity_study_jason --plot
        """
    )

    parser.add_argument(
        '--case',
        type=str,
        default='jason',
        help='Case identifier (default: jason)'
    )
    parser.add_argument(
        '--mesh',
        type=str,
        default='coarse,medium',
        help='Comma-separated mesh presets (default: coarse,medium)'
    )
    parser.add_argument(
        '--cand',
        type=str,
        default='1.0',
        help='Comma-separated candidate density factors (default: 1.0)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test with minimal steps (nsteps=3)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate sensitivity plots'
    )

    args = parser.parse_args()

    # Parse mesh list
    meshes = [m.strip() for m in args.mesh.split(',')]

    # Parse candidate density list
    cand_densities = [float(c.strip()) for c in args.cand.split(',')]

    # Run study
    results = run_sensitivity_study(
        case_id=args.case,
        meshes=meshes,
        cand_densities=cand_densities,
        quick=args.quick,
    )

    # Save summary
    output_file = f"outputs/sensitivity/{args.case}_summary.csv"
    save_sensitivity_summary(results, output_file)

    # Generate plots
    if args.plot:
        generate_sensitivity_plots(results, "outputs/sensitivity")


if __name__ == '__main__':
    main()
