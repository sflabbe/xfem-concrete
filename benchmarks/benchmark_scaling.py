"""
Performance and scaling benchmarks for thesis cases.

Measures runtime, peak memory, and energy residual across different mesh sizes.
Validates that solver maintains energy conservation (E_residual < 1%).

Usage:
    python -m benchmarks.benchmark_scaling --case t5a1 --meshes coarse,medium,fine
    python -m benchmarks.benchmark_scaling --case all --meshes coarse,medium
    python -m benchmarks.benchmark_scaling --plot
"""

import argparse
import sys
import time
import csv
import tracemalloc
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.gutierrez_thesis.run import CASE_REGISTRY, MESH_PRESETS, resolve_case_id
from examples.gutierrez_thesis.solver_interface import run_case_solver
from examples.gutierrez_thesis.history_utils import history_to_arrays


# ============================================================================
# Benchmark runner
# ============================================================================

def compute_energy_residual(results: Dict) -> float:
    """
    Compute energy residual: |W_total - W_external| / W_external × 100%.

    Parameters
    ----------
    results : dict
        Solver results with 'history' key

    Returns
    -------
    residual_pct : float
        Energy residual as percentage (should be < 1% for accurate solver)

    Notes
    -----
    W_external = ∫ P du (external work by applied loads)
    W_total = W_plastic + W_damage_t + W_damage_c + W_cohesive (internal dissipation)

    For a converged simulation: W_external ≈ W_total
    """
    history = results.get('history', [])

    if len(history) == 0:
        return 0.0

    # Extract arrays (handles both numeric and dict history formats)
    arrays = history_to_arrays(history)
    u_arr = arrays['u']  # [m]
    P_arr = arrays['P']  # [N]

    # External work (∫ P du)
    if len(u_arr) == 0:
        return 0.0

    W_external = np.trapz(np.abs(P_arr), u_arr)  # [J]

    # Internal dissipation (try to extract from extras, otherwise fallback)
    extras = arrays['extras']

    # Try to get W_total directly (column 14 in old format, or key in dict format)
    W_total = None
    if 'W_total' in extras:
        W_total = extras['W_total'][-1]  # Final value
    elif 'col_14' in extras:
        W_total = extras['col_14'][-1]
    else:
        # Fallback: sum individual components
        W_plastic = extras.get('W_plastic', extras.get('col_10', np.array([0.0])))[-1]
        W_damage_t = extras.get('W_damage_t', extras.get('col_11', np.array([0.0])))[-1]
        W_damage_c = extras.get('W_damage_c', extras.get('col_12', np.array([0.0])))[-1]
        W_cohesive = extras.get('W_cohesive', extras.get('col_13', np.array([0.0])))[-1]
        W_total = W_plastic + W_damage_t + W_damage_c + W_cohesive

    if W_total is None:
        W_total = 0.0

    # Residual
    if W_external > 1e-12:
        residual_pct = abs(W_total - W_external) / W_external * 100.0
    else:
        residual_pct = 0.0

    return residual_pct


def run_single_benchmark(
    case_id: str,
    mesh: str,
) -> Dict:
    """
    Run single benchmark for one case + mesh combination.

    Parameters
    ----------
    case_id : str
        Case identifier (e.g., "04a_beam_3pb_t5a1_bosco")
    mesh : str
        Mesh preset ("coarse", "medium", "fine")

    Returns
    -------
    benchmark : dict
        Benchmark results with keys:
        - case_id: Case identifier
        - mesh: Mesh preset
        - n_elements: Total number of elements
        - n_steps: Number of load steps
        - runtime_s: Wall-clock time in seconds
        - peak_memory_mb: Peak memory usage in MB
        - energy_residual_pct: Energy residual percentage
        - converged: True if solver converged
    """
    case_id_resolved = resolve_case_id(case_id)
    if not case_id_resolved:
        raise ValueError(f"Unknown case: {case_id}")

    factory = CASE_REGISTRY[case_id_resolved]
    mesh_factor = MESH_PRESETS.get(mesh, 1.0)

    print(f"\n{'='*70}")
    print(f"BENCHMARK: {case_id_resolved} (mesh={mesh})")
    print(f"{'='*70}\n")

    # Create case config
    case_config = factory()

    # Calculate mesh size
    n_elem_x = int(case_config.geometry.n_elem_x * mesh_factor)
    n_elem_y = int(case_config.geometry.n_elem_y * mesh_factor)
    n_elements = n_elem_x * n_elem_y

    print(f"Mesh: {n_elem_x} × {n_elem_y} = {n_elements} elements")

    # Start memory tracking
    tracemalloc.start()

    # Run solver
    t0 = time.time()
    converged = True

    try:
        results = run_case_solver(case_config, mesh_factor=mesh_factor)
    except Exception as e:
        print(f"✗ Solver failed: {e}")
        converged = False
        results = {'history': []}

    runtime = time.time() - t0

    # Get peak memory
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_memory_mb = peak_mem / 1024 / 1024  # bytes → MB

    # Compute energy residual
    energy_residual = compute_energy_residual(results)

    # Extract metrics
    history = results.get('history', [])
    n_steps = len(history)

    benchmark = {
        'case_id': case_id_resolved,
        'mesh': mesh,
        'n_elements': n_elements,
        'n_steps': n_steps,
        'runtime_s': runtime,
        'peak_memory_mb': peak_memory_mb,
        'energy_residual_pct': energy_residual,
        'converged': converged,
    }

    # Print summary
    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS:")
    print(f"{'='*70}")
    print(f"  Elements:        {n_elements}")
    print(f"  Steps:           {n_steps}")
    print(f"  Runtime:         {runtime:.2f} s")
    print(f"  Peak memory:     {peak_memory_mb:.2f} MB")
    print(f"  Energy residual: {energy_residual:.4f} %")
    print(f"  Converged:       {converged}")

    # Validation check
    if energy_residual > 1.0:
        print(f"\n⚠ WARNING: Energy residual {energy_residual:.2f}% > 1% (accuracy issue)")
    else:
        print(f"\n✓ Energy residual within tolerance (< 1%)")

    print(f"{'='*70}\n")

    return benchmark


# ============================================================================
# Scaling benchmark runner
# ============================================================================

def run_scaling_benchmark(
    case_ids: List[str],
    meshes: List[str] = ['coarse', 'medium', 'fine'],
) -> List[Dict]:
    """
    Run scaling benchmarks across multiple cases and mesh sizes.

    Parameters
    ----------
    case_ids : list of str
        Case identifiers to benchmark
    meshes : list of str
        Mesh presets to test (default: ['coarse', 'medium', 'fine'])

    Returns
    -------
    benchmarks : list of dict
        List of benchmark results
    """
    benchmarks = []

    for case_id in case_ids:
        for mesh in meshes:
            try:
                benchmark = run_single_benchmark(case_id, mesh)
                benchmarks.append(benchmark)
            except Exception as e:
                print(f"\n✗ Benchmark failed for {case_id} ({mesh}): {e}\n")
                import traceback
                traceback.print_exc()

    return benchmarks


def save_scaling_summary(benchmarks: List[Dict], output_file: str = "benchmarks/scaling_summary.csv"):
    """
    Save benchmark results to CSV file.

    Parameters
    ----------
    benchmarks : list of dict
        Benchmark results from run_scaling_benchmark()
    output_file : str
        Output CSV file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'case_id',
            'mesh',
            'n_elements',
            'n_steps',
            'runtime_s',
            'peak_memory_mb',
            'energy_residual_pct',
            'converged',
        ])

        # Data rows
        for b in benchmarks:
            writer.writerow([
                b['case_id'],
                b['mesh'],
                b['n_elements'],
                b['n_steps'],
                b['runtime_s'],
                b['peak_memory_mb'],
                b['energy_residual_pct'],
                b['converged'],
            ])

    print(f"\n✓ Benchmark summary saved to: {output_path}")


def generate_scaling_report(benchmarks: List[Dict]) -> str:
    """
    Generate human-readable benchmark summary report.

    Parameters
    ----------
    benchmarks : list of dict
        Benchmark results

    Returns
    -------
    report : str
        Formatted report
    """
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append("SCALING BENCHMARK SUMMARY")
    lines.append(f"{'='*70}\n")

    # Group by case
    cases = {}
    for b in benchmarks:
        case_id = b['case_id']
        if case_id not in cases:
            cases[case_id] = []
        cases[case_id].append(b)

    for case_id, case_benchmarks in cases.items():
        lines.append(f"\nCase: {case_id}")
        lines.append("-" * 70)
        lines.append(f"{'Mesh':<12} {'Elements':<12} {'Steps':<8} {'Time (s)':<12} {'Memory (MB)':<15} {'E_res (%)':<12} {'OK':<5}")
        lines.append("-" * 70)

        for b in case_benchmarks:
            ok_flag = "✓" if b['converged'] and b['energy_residual_pct'] < 1.0 else "✗"
            lines.append(
                f"{b['mesh']:<12} {b['n_elements']:<12} {b['n_steps']:<8} "
                f"{b['runtime_s']:<12.2f} {b['peak_memory_mb']:<15.2f} "
                f"{b['energy_residual_pct']:<12.4f} {ok_flag:<5}"
            )

    lines.append(f"\n{'='*70}\n")

    return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Performance and scaling benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmarks.benchmark_scaling --case t5a1 --meshes coarse,medium,fine
  python -m benchmarks.benchmark_scaling --case all --meshes coarse,medium
  python -m benchmarks.benchmark_scaling --case vvbs3,sorelli --plot
        """
    )

    parser.add_argument(
        '--case',
        type=str,
        default='all',
        help='Case identifier or "all" (default: all)'
    )
    parser.add_argument(
        '--meshes',
        type=str,
        default='coarse,medium',
        help='Comma-separated mesh presets (default: coarse,medium)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate scaling plots (requires matplotlib)'
    )

    args = parser.parse_args()

    # Parse mesh list
    meshes = [m.strip() for m in args.meshes.split(',')]

    # Parse case list
    if args.case.lower() == 'all':
        case_ids = list(CASE_REGISTRY.keys())
    else:
        case_ids = [c.strip() for c in args.case.split(',')]

    # Run benchmarks
    benchmarks = run_scaling_benchmark(case_ids, meshes)

    # Save summary
    save_scaling_summary(benchmarks, "benchmarks/scaling_summary.csv")

    # Print report
    report = generate_scaling_report(benchmarks)
    print(report)

    # Optional: Generate plots
    if args.plot:
        try:
            import matplotlib.pyplot as plt

            # Group by case for plotting
            cases = {}
            for b in benchmarks:
                case_id = b['case_id']
                if case_id not in cases:
                    cases[case_id] = []
                cases[case_id].append(b)

            # Create log-log plot: elements vs runtime
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            for case_id, case_benchmarks in cases.items():
                n_elem = [b['n_elements'] for b in case_benchmarks]
                runtime = [b['runtime_s'] for b in case_benchmarks]
                energy_res = [b['energy_residual_pct'] for b in case_benchmarks]

                ax1.loglog(n_elem, runtime, 'o-', label=case_id, linewidth=2, markersize=8)
                ax2.semilogx(n_elem, energy_res, 's-', label=case_id, linewidth=2, markersize=8)

            ax1.set_xlabel('Number of Elements', fontsize=12)
            ax1.set_ylabel('Runtime (s)', fontsize=12)
            ax1.set_title('Scaling: Runtime vs Mesh Size', fontsize=14)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)

            ax2.set_xlabel('Number of Elements', fontsize=12)
            ax2.set_ylabel('Energy Residual (%)', fontsize=12)
            ax2.set_title('Energy Conservation Check', fontsize=14)
            ax2.axhline(1.0, color='r', linestyle='--', linewidth=2, label='1% threshold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_file = "benchmarks/scaling_plot.png"
            plt.savefig(plot_file, dpi=150)
            print(f"✓ Scaling plot saved to: {plot_file}")

        except ImportError:
            print("Warning: matplotlib not available, skipping plot")


if __name__ == '__main__':
    main()
