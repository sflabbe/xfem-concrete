"""
Appendix Generator for Gutiérrez Thesis

Automates generation of LaTeX appendix with figures and tables from simulation results.

Usage:
    python -m report.generate_appendix --cases t5a1,vvbs3 --mesh coarse --nsteps 10
    python -m report.generate_appendix --cases all --mesh medium --nsteps 50
"""

import argparse
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import csv
import time

# Import case registry
from examples.gutierrez_thesis.run import CASE_REGISTRY, CASE_ALIASES, MESH_PRESETS
from examples.gutierrez_thesis.solver_interface import run_case_solver


def resolve_case_ids(case_names: List[str]) -> List[str]:
    """
    Resolve case names to canonical IDs.

    Parameters
    ----------
    case_names : list of str
        Case names (can be aliases or canonical IDs)

    Returns
    -------
    case_ids : list of str
        Canonical case IDs
    """
    if len(case_names) == 1 and case_names[0].lower() == "all":
        return list(CASE_REGISTRY.keys())

    case_ids = []
    for name in case_names:
        # Direct match
        if name in CASE_REGISTRY:
            case_ids.append(name)
        # Alias match
        elif name in CASE_ALIASES:
            case_ids.append(CASE_ALIASES[name])
        else:
            # Partial match
            matches = [cid for cid in CASE_REGISTRY.keys() if name.lower() in cid.lower()]
            if len(matches) == 1:
                case_ids.append(matches[0])
            elif len(matches) > 1:
                print(f"Warning: Ambiguous case name '{name}'. Matches: {matches}")
                print(f"         Using first match: {matches[0]}")
                case_ids.append(matches[0])
            else:
                print(f"Warning: Unknown case '{name}', skipping")

    return case_ids


def run_case_for_appendix(case_id: str, mesh: str = "coarse", nsteps: Optional[int] = None,
                            dry_run: bool = False) -> Optional[Dict]:
    """
    Run a single case for appendix generation.

    Parameters
    ----------
    case_id : str
        Canonical case ID
    mesh : str
        Mesh preset
    nsteps : int, optional
        Number of steps (overrides case default)
    dry_run : bool
        If True, skip simulation

    Returns
    -------
    results : dict or None
        Simulation results, or None if dry run or failed
    """
    print(f"\n{'='*70}")
    print(f"Running case: {case_id}")
    print(f"  Mesh: {mesh}")
    print(f"  Steps: {nsteps if nsteps else 'default'}")
    print(f"{'='*70}")

    if dry_run:
        print("DRY RUN - skipping simulation")
        return None

    # Get case factory
    factory = CASE_REGISTRY[case_id]
    case_config = factory()

    # Override nsteps if provided
    if nsteps is not None and hasattr(case_config.loading, 'n_steps'):
        case_config.loading.n_steps = nsteps

    # Run simulation
    mesh_factor = MESH_PRESETS[mesh]

    try:
        t0 = time.time()
        results = run_case_solver(case_config, mesh_factor=mesh_factor)
        t_elapsed = time.time() - t0
        print(f"✓ Simulation completed in {t_elapsed:.1f} seconds")
        return results

    except Exception as e:
        import traceback
        print(f"✗ ERROR: {e}")
        traceback.print_exc()
        return None


def extract_metrics_from_history(history: List) -> Dict[str, float]:
    """
    Extract key metrics from simulation history for LaTeX table.

    Parameters
    ----------
    history : list
        Simulation history (format: [step, u, P, M, ...])

    Returns
    -------
    metrics : dict
        Dictionary with metrics (P_max_kN, u_at_Pmax_mm, energy_kNmm, etc.)
    """
    import numpy as np

    # Convert to arrays
    history_arr = np.array(history)
    if history_arr.ndim == 1:
        # Single row
        history_arr = history_arr.reshape(1, -1)

    u_m = history_arr[:, 1]
    P_N = history_arr[:, 2]

    # Convert units
    u_mm = u_m * 1e3
    P_kN = P_N / 1e3

    # Peak load
    idx_max = np.argmax(P_kN)
    P_max = P_kN[idx_max]
    u_at_Pmax = u_mm[idx_max]

    # Energy (∫ P du)
    energy = np.trapezoid(P_kN, u_mm)

    # Final displacement
    u_final = u_mm[-1]

    # Ductility (if available)
    ductility = u_final / u_at_Pmax if u_at_Pmax > 1e-6 else 0.0

    # Number of cracks (if available in history)
    num_cracks = 0
    if history_arr.shape[1] > 9:
        # Check if crack_active column exists
        crack_active = history_arr[:, 9]
        num_cracks = int(np.max(crack_active))

    metrics = {
        'P_max_kN': P_max,
        'u_at_Pmax_mm': u_at_Pmax,
        'u_final_mm': u_final,
        'energy_kNmm': energy,
        'ductility': ductility,
        'num_cracks': num_cracks,
    }

    return metrics


def generate_latex_table(case_id: str, metrics: Dict[str, float]) -> str:
    """
    Generate LaTeX table code for case metrics.

    Parameters
    ----------
    case_id : str
        Case ID
    metrics : dict
        Metrics dictionary

    Returns
    -------
    latex : str
        LaTeX table code
    """
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Simulation results for " + case_id.replace("_", "\\_") + "}")
    lines.append("\\label{tab:" + case_id + "}")
    lines.append("\\begin{tabular}{ll}")
    lines.append("\\toprule")
    lines.append("Metric & Value \\\\")
    lines.append("\\midrule")
    lines.append(f"Peak Load ($P_{{\\max}}$) & {metrics['P_max_kN']:.2f} kN \\\\")
    lines.append(f"Displacement at Peak ($u_{{P_{{\\max}}}}$) & {metrics['u_at_Pmax_mm']:.2f} mm \\\\")
    lines.append(f"Final Displacement ($u_{{\\text{{final}}}}$) & {metrics['u_final_mm']:.2f} mm \\\\")
    lines.append(f"Dissipated Energy & {metrics['energy_kNmm']:.2f} kN$\\cdot$mm \\\\")
    lines.append(f"Ductility ($\\mu$) & {metrics['ductility']:.2f} \\\\")
    if metrics['num_cracks'] > 0:
        lines.append(f"Number of Cracks & {metrics['num_cracks']} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def copy_outputs_to_report(case_id: str, output_dir: Path, report_dir: Path) -> None:
    """
    Copy simulation outputs to report directories.

    Parameters
    ----------
    case_id : str
        Case ID
    output_dir : Path
        Source output directory
    report_dir : Path
        Report root directory

    Notes
    -----
    Copies:
    - P-u curve figure to report/figures/{case_id}_P_u.pdf (or .png)
    - Other relevant figures
    """
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Look for P-u curve plot
    for ext in ['.pdf', '.png', '.jpg']:
        src = output_dir / f"P_u_curve{ext}"
        if src.exists():
            dst = figures_dir / f"{case_id}_P_u{ext}"
            shutil.copy(src, dst)
            print(f"  → Copied {src.name} to {dst}")
            break

    # Copy crack pattern if exists
    for ext in ['.pdf', '.png']:
        src = output_dir / f"crack_pattern{ext}"
        if src.exists():
            dst = figures_dir / f"{case_id}_crack_pattern{ext}"
            shutil.copy(src, dst)
            print(f"  → Copied {src.name} to {dst}")
            break


def generate_appendix_latex(cases_data: List[Dict], output_file: Path) -> None:
    """
    Generate LaTeX appendix file with all cases.

    Parameters
    ----------
    cases_data : list of dict
        List of dictionaries with keys: case_id, metrics
    output_file : Path
        Output .tex file path
    """
    lines = []
    lines.append("% Auto-generated appendix for Gutiérrez thesis")
    lines.append("% Generated by report/generate_appendix.py")
    lines.append("")
    lines.append("\\chapter{Simulation Results}")
    lines.append("\\label{app:simulation_results}")
    lines.append("")
    lines.append("This appendix presents the simulation results for the validation cases.")
    lines.append("")

    for data in cases_data:
        case_id = data['case_id']
        metrics = data['metrics']

        # Prepare LaTeX-safe case ID
        case_id_latex = case_id.replace('_', '\\_')
        case_title = case_id.replace('_', ' ').title()

        lines.append(f"\\section{{{case_title}}}")
        lines.append("")

        # Include figure if exists
        fig_path = Path("figures") / f"{case_id}_P_u"
        for ext in ['.pdf', '.png']:
            if (output_file.parent / "figures" / f"{case_id}_P_u{ext}").exists():
                lines.append("\\begin{figure}[htbp]")
                lines.append("\\centering")
                lines.append(f"\\includegraphics[width=0.7\\textwidth]{{{fig_path}}}")
                lines.append(f"\\caption{{Load-displacement curve for {case_id_latex}}}")
                lines.append(f"\\label{{fig:{case_id}_Pu}}")
                lines.append("\\end{figure}")
                lines.append("")
                break

        # Include table
        lines.append(generate_latex_table(case_id, metrics))
        lines.append("")
        lines.append("\\clearpage")
        lines.append("")

    # Write to file
    with open(output_file, 'w') as f:
        f.write("\n".join(lines))

    print(f"\n✓ LaTeX appendix saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX appendix from simulation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--cases",
        type=str,
        required=True,
        help="Comma-separated list of case IDs or 'all'"
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default="coarse",
        choices=list(MESH_PRESETS.keys()),
        help="Mesh preset (default: coarse for speed)"
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        help="Number of steps (overrides case defaults)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip simulations (only generate LaTeX from existing outputs)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="report/appendix_A.tex",
        help="Output LaTeX file path"
    )

    args = parser.parse_args()

    # Parse case list
    case_names = [c.strip() for c in args.cases.split(',')]
    case_ids = resolve_case_ids(case_names)

    if not case_ids:
        print("Error: No valid cases specified")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Generating appendix for {len(case_ids)} cases")
    print(f"  Cases: {', '.join(case_ids)}")
    print(f"  Mesh: {args.mesh}")
    print(f"  Output: {args.output}")
    print(f"{'='*70}")

    # Run simulations and collect data
    cases_data = []

    for case_id in case_ids:
        # Run simulation
        results = run_case_for_appendix(case_id, args.mesh, args.nsteps, args.dry_run)

        if results and 'history' in results:
            # Extract metrics
            metrics = extract_metrics_from_history(results['history'])

            # Save table to file
            report_dir = Path("report")
            tables_dir = report_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)

            table_file = tables_dir / f"{case_id}_metrics.tex"
            with open(table_file, 'w') as f:
                f.write(generate_latex_table(case_id, metrics))
            print(f"  → Table saved to {table_file}")

            # Copy outputs
            # Determine output directory (depends on case runner implementation)
            output_dir = Path("outputs") / case_id
            if output_dir.exists():
                copy_outputs_to_report(case_id, output_dir, report_dir)

            cases_data.append({
                'case_id': case_id,
                'metrics': metrics,
            })

        elif args.dry_run:
            print(f"  Skipped (dry run)")
        else:
            print(f"  Failed or no results")

    # Generate appendix
    if cases_data or args.dry_run:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if not args.dry_run:
            generate_appendix_latex(cases_data, output_file)
        else:
            print(f"\nDRY RUN - Would generate {output_file}")

        print(f"\n{'='*70}")
        print("✓ Appendix generation complete")
        print(f"{'='*70}\n")

    else:
        print("\nNo successful simulations. Appendix not generated.")
        sys.exit(1)


if __name__ == "__main__":
    main()
