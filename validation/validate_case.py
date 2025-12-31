#!/usr/bin/env python3
"""
Run validation for a specific case: simulate + compare with reference.

This script provides a convenient way to run a simulation and immediately
validate it against experimental reference data.

Usage:
    python validation/validate_case.py --case vvbs3 --mesh coarse
    python validation/validate_case.py --case t5a1 --mesh medium --skip-sim
    python validation/validate_case.py --case sorelli --mesh fine --output validation/reports/
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from examples.gutierrez_thesis.run import CASE_REGISTRY, CASE_ALIASES, MESH_PRESETS
    from examples.gutierrez_thesis.solver_interface import run_case_solver
    from validation.compare_curves import (
        load_simulation_curve,
        load_reference_curve,
        compute_error_metrics,
        generate_validation_report,
        is_placeholder_data,
    )
    IMPORTS_OK = True
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    IMPORTS_OK = False


def resolve_case_id(case_name: str) -> str:
    """Resolve case name to canonical ID."""
    # Direct match
    if case_name in CASE_REGISTRY:
        return case_name

    # Alias match
    if case_name in CASE_ALIASES:
        return CASE_ALIASES[case_name]

    # Partial match
    matches = [cid for cid in CASE_REGISTRY.keys() if case_name.lower() in cid.lower()]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"ERROR: Ambiguous case name '{case_name}'. Matches: {matches}")
        sys.exit(1)
    else:
        print(f"ERROR: Unknown case '{case_name}'")
        print(f"Available cases: {', '.join(CASE_REGISTRY.keys())}")
        sys.exit(1)


def run_simulation(case_id: str, mesh: str) -> bool:
    """
    Run simulation for the specified case.

    Returns
    -------
    success : bool
        True if simulation completed successfully
    """
    print(f"\n{'='*70}")
    print(f"Running simulation: {case_id}")
    print(f"Mesh: {mesh}")
    print(f"{'='*70}\n")

    factory = CASE_REGISTRY[case_id]
    case_config = factory()

    mesh_factor = MESH_PRESETS.get(mesh, 1.0)

    try:
        results = run_case_solver(case_config, mesh_factor=mesh_factor)
        print(f"\n✓ Simulation completed successfully")
        return True
    except Exception as e:
        print(f"\n✗ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_case(case_id: str, mesh: str, ref_id: str = None,
                  output_dir: str = None, tolerances: dict = None) -> dict:
    """
    Validate simulation results against reference data.

    Parameters
    ----------
    case_id : str
        Canonical case ID
    mesh : str
        Mesh preset
    ref_id : str, optional
        Reference data ID (defaults to case_id)
    output_dir : str, optional
        Directory for validation reports
    tolerances : dict, optional
        Tolerance thresholds

    Returns
    -------
    result : dict
        Validation results with keys: 'metrics', 'passed', 'report', 'is_placeholder'
    """
    if ref_id is None:
        # Try to infer reference ID from case ID
        # e.g., "04a_beam_3pb_t5a1_bosco" → "t5a1"
        if 't5a1' in case_id:
            ref_id = 't5a1'
        elif 'vvbs3' in case_id:
            ref_id = 'vvbs3'
        elif 'sorelli' in case_id:
            ref_id = 'sorelli'
        else:
            ref_id = case_id

    if tolerances is None:
        tolerances = {
            'peak_error_pct': 10.0,      # ± 10%
            'energy_error_pct': 15.0,    # ± 15%
            'rmse_normalized_pct': 5.0,  # 5% of peak load
        }

    print(f"\n{'='*70}")
    print(f"Validating: {case_id}")
    print(f"Reference: {ref_id}")
    print(f"Mesh: {mesh}")
    print(f"{'='*70}\n")

    # Load curves
    try:
        sim = load_simulation_curve(case_id, mesh=mesh)
        print(f"✓ Loaded simulation curve ({len(sim)} points)")
    except FileNotFoundError as e:
        print(f"✗ Simulation output not found: {e}")
        print(f"  Run simulation first: python -m examples.gutierrez_thesis.run --case {ref_id} --mesh {mesh}")
        return {'error': 'simulation_not_found'}

    try:
        ref = load_reference_curve(ref_id, check_quality=True)
        print(f"✓ Loaded reference curve ({len(ref)} points)")
    except FileNotFoundError as e:
        print(f"✗ Reference data not found: {e}")
        print(f"  See validation/reference_data/SOURCES.md for digitization guidelines")
        return {'error': 'reference_not_found'}

    # Check if reference is placeholder
    is_placeholder, placeholder_reason = is_placeholder_data(ref)
    if is_placeholder:
        print(f"\n⚠ WARNING: Reference data appears to be PLACEHOLDER")
        print(f"  Reason: {placeholder_reason}")
        print(f"  Validation results may not be meaningful.")
        print(f"  See validation/reference_data/SOURCES.md for digitization guidelines.\n")

    # Compute metrics
    metrics = compute_error_metrics(sim, ref)

    # Generate report
    report = generate_validation_report(case_id, metrics, tolerances=tolerances)
    print(report)

    # Check if passed
    passed = (
        metrics['peak_error_pct'] <= tolerances['peak_error_pct'] and
        metrics['energy_error_pct'] <= tolerances['energy_error_pct'] and
        metrics['rmse_normalized_pct'] <= tolerances['rmse_normalized_pct']
    )

    # Save report if output_dir specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_file = output_path / f"{case_id}_{mesh}_validation.json"

        report_data = {
            'case_id': case_id,
            'ref_id': ref_id,
            'mesh': mesh,
            'timestamp': datetime.now().isoformat(),
            'is_placeholder': is_placeholder,
            'placeholder_reason': placeholder_reason if is_placeholder else None,
            'metrics': metrics,
            'tolerances': tolerances,
            'passed': passed,
        }

        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\n✓ Validation report saved to: {report_file}")

    return {
        'metrics': metrics,
        'passed': passed,
        'report': report,
        'is_placeholder': is_placeholder,
    }


def main():
    if not IMPORTS_OK:
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Run case validation (simulate + compare with reference)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--case', '-c', required=True, help='Case ID or alias (e.g., vvbs3, t5a1, sorelli)')
    parser.add_argument('--mesh', '-m', default='coarse', choices=list(MESH_PRESETS.keys()),
                        help='Mesh preset (default: coarse)')
    parser.add_argument('--ref', '-r', help='Reference data ID (auto-detected if not specified)')
    parser.add_argument('--skip-sim', action='store_true', help='Skip simulation (use existing output)')
    parser.add_argument('--output', '-o', default='validation/reports',
                        help='Output directory for validation reports')
    parser.add_argument('--peak-tol', type=float, default=10.0, help='Peak load error tolerance (%%)')
    parser.add_argument('--energy-tol', type=float, default=15.0, help='Energy error tolerance (%%)')
    parser.add_argument('--rmse-tol', type=float, default=5.0, help='RMSE tolerance (%% of peak)')

    args = parser.parse_args()

    # Resolve case ID
    case_id = resolve_case_id(args.case)

    # Run simulation unless skipped
    if not args.skip_sim:
        success = run_simulation(case_id, args.mesh)
        if not success:
            print(f"\n✗ Simulation failed. Cannot proceed with validation.")
            sys.exit(1)
    else:
        print(f"\n⚠ Skipping simulation (using existing output)")

    # Tolerances
    tolerances = {
        'peak_error_pct': args.peak_tol,
        'energy_error_pct': args.energy_tol,
        'rmse_normalized_pct': args.rmse_tol,
    }

    # Validate
    result = validate_case(
        case_id=case_id,
        mesh=args.mesh,
        ref_id=args.ref,
        output_dir=args.output,
        tolerances=tolerances,
    )

    if 'error' in result:
        sys.exit(1)

    # Exit with appropriate code
    if result['is_placeholder']:
        print(f"\n⚠ VALIDATION INCOMPLETE: Reference data is placeholder")
        sys.exit(2)  # Special exit code for placeholder
    elif result['passed']:
        print(f"\n✓✓✓ VALIDATION PASSED ✓✓✓")
        sys.exit(0)
    else:
        print(f"\n✗✗✗ VALIDATION FAILED ✗✗✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
