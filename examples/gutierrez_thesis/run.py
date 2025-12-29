"""
Gutiérrez Thesis Case Suite Runner

CLI to run all thesis examples from Chapter 5 + Appendix A.

Usage:
    python -m examples.gutierrez_thesis.run --case STN12 --mesh fine
    python -m examples.gutierrez_thesis.run --case all --mesh coarse
    python -m examples.gutierrez_thesis.run --list
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Callable

# Import case factories
from examples.gutierrez_thesis.cases.case_01_pullout_lettow import create_case_01


# ============================================================================
# CASE REGISTRY
# ============================================================================

CASE_REGISTRY: Dict[str, Callable] = {
    "01_pullout_lettow": create_case_01,
    # Add more cases as implemented
    # "02_sspot_frp": create_case_02,
    # "03_tensile_stn12": create_case_03,
    # etc.
}

CASE_ALIASES = {
    "pullout": "01_pullout_lettow",
    "lettow": "01_pullout_lettow",
    # Add aliases for convenience
    # "sspot": "02_sspot_frp",
    # "stn12": "03_tensile_stn12",
}

MESH_PRESETS = {
    "coarse": 0.5,   # Coarsening factor
    "medium": 1.0,
    "fine": 1.5,
    "very_fine": 2.0,
}


# ============================================================================
# SOLVER INTERFACE
# ============================================================================

def run_case(case_config, mesh_factor: float = 1.0, dry_run: bool = False):
    """
    Run a single case with the XFEM solver.

    Args:
        case_config: CaseConfig instance
        mesh_factor: Mesh refinement factor (1.0 = default)
        dry_run: If True, only print configuration without running
    """
    print(f"\n{'='*70}")
    print(f"Running Case: {case_config.case_id}")
    print(f"Description: {case_config.description}")
    print(f"{'='*70}\n")

    # Apply mesh factor
    if mesh_factor != 1.0:
        case_config.geometry.n_elem_x = int(
            case_config.geometry.n_elem_x * mesh_factor
        )
        case_config.geometry.n_elem_y = int(
            case_config.geometry.n_elem_y * mesh_factor
        )
        print(f"Mesh adjusted: {case_config.geometry.n_elem_x} x "
              f"{case_config.geometry.n_elem_y} elements")

    # Print configuration summary
    print(f"Geometry: {case_config.geometry.length} x "
          f"{case_config.geometry.height} mm")
    print(f"Concrete: E={case_config.concrete.E} MPa, "
          f"ft={case_config.concrete.f_t} MPa")
    print(f"Rebar layers: {len(case_config.rebar_layers)}")
    print(f"FRP sheets: {len(case_config.frp_sheets)}")
    print(f"Fibres: {'Yes' if case_config.fibres else 'No'}")
    print(f"Loading: {case_config.loading.loading_type}")
    print(f"Output dir: {case_config.outputs.output_dir}\n")

    if dry_run:
        print("DRY RUN - Solver not executed.\n")
        return

    # Create output directory
    output_dir = Path(case_config.outputs.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run solver
    print("Starting solver...\n")
    t0 = time.time()

    try:
        # TODO: Call actual solver
        # result = run_xfem_solver(case_config)
        raise NotImplementedError(
            "Solver integration not yet complete. "
            "Need to implement:\n"
            "  1. Bond-slip laws (CEB-FIP, bilinear, Banholzer)\n"
            "  2. Subdomain support (void elements, rigid regions)\n"
            "  3. FRP sheet reinforcement\n"
            "  4. Fibre reinforcement\n"
            "  5. Cyclic loading\n"
            "  6. Post-processing utilities\n"
        )

    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        sys.exit(1)

    t_elapsed = time.time() - t0
    print(f"\n✓ Case completed in {t_elapsed:.1f} seconds")
    print(f"✓ Outputs saved to: {output_dir}\n")


# ============================================================================
# CLI
# ============================================================================

def list_cases():
    """Print available cases"""
    print("\nAvailable Cases:")
    print("=" * 70)
    for case_id, factory in CASE_REGISTRY.items():
        case = factory()
        print(f"  {case_id:30s} - {case.description}")
    print("=" * 70)
    print(f"\nTotal: {len(CASE_REGISTRY)} cases")
    print("\nMesh presets: coarse, medium (default), fine, very_fine\n")


def resolve_case_id(name: str) -> Optional[str]:
    """Resolve case name to canonical ID"""
    # Direct match
    if name in CASE_REGISTRY:
        return name

    # Alias match
    if name in CASE_ALIASES:
        return CASE_ALIASES[name]

    # Case-insensitive partial match
    name_lower = name.lower()
    matches = [
        cid for cid in CASE_REGISTRY.keys()
        if name_lower in cid.lower()
    ]

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"Ambiguous case name '{name}'. Matches: {matches}")
        return None
    else:
        print(f"Unknown case: '{name}'")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Gutiérrez Thesis Case Suite Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m examples.gutierrez_thesis.run --list
  python -m examples.gutierrez_thesis.run --case pullout
  python -m examples.gutierrez_thesis.run --case STN12 --mesh fine
  python -m examples.gutierrez_thesis.run --case all --mesh coarse
        """
    )

    parser.add_argument(
        "--case",
        type=str,
        help="Case ID or alias (or 'all' to run all cases)"
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default="medium",
        choices=list(MESH_PRESETS.keys()),
        help="Mesh preset (default: medium)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available cases and exit"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without running solver"
    )

    args = parser.parse_args()

    # List cases
    if args.list:
        list_cases()
        sys.exit(0)

    # Validate arguments
    if not args.case:
        parser.print_help()
        print("\nError: --case is required (or use --list)")
        sys.exit(1)

    # Get mesh factor
    mesh_factor = MESH_PRESETS[args.mesh]

    # Run cases
    if args.case.lower() == "all":
        # Run all cases
        print(f"\n{'='*70}")
        print(f"Running ALL {len(CASE_REGISTRY)} cases with mesh={args.mesh}")
        print(f"{'='*70}")

        for case_id, factory in CASE_REGISTRY.items():
            case_config = factory()
            run_case(case_config, mesh_factor, args.dry_run)

    else:
        # Run single case
        case_id = resolve_case_id(args.case)
        if not case_id:
            sys.exit(1)

        factory = CASE_REGISTRY[case_id]
        case_config = factory()
        run_case(case_config, mesh_factor, args.dry_run)


if __name__ == "__main__":
    main()
