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
from examples.gutierrez_thesis.cases.case_02_sspot_frp import create_case_02
from examples.gutierrez_thesis.cases.case_03_tensile_stn12 import create_case_03
from examples.gutierrez_thesis.cases.case_04_beam_3pb_t5a1 import create_case_04  # Legacy
from examples.gutierrez_thesis.cases.case_04a_beam_3pb_t5a1_bosco import create_case_04a
from examples.gutierrez_thesis.cases.case_04b_beam_3pb_t6a1_bosco import create_case_04b
from examples.gutierrez_thesis.cases.case_05_wall_c1_cyclic import create_case_05
from examples.gutierrez_thesis.cases.case_06_fibre_tensile import create_case_06
from examples.gutierrez_thesis.cases.case_07_beam_4pb_jason_4pbt import create_case_07
from examples.gutierrez_thesis.cases.case_08_beam_3pb_vvbs3_cfrp import create_case_08
from examples.gutierrez_thesis.cases.case_09_beam_4pb_fibres_sorelli import create_case_09
from examples.gutierrez_thesis.cases.case_10_wall_c2_cyclic import create_case_10
from examples.gutierrez_thesis.cases.case_11_balcony_cantilever_sls import create_case_11


# ============================================================================
# CASE REGISTRY
# ============================================================================

CASE_REGISTRY: Dict[str, Callable] = {
    "01_pullout_lettow": create_case_01,
    "02_sspot_frp": create_case_02,
    "03_tensile_stn12": create_case_03,
    "04_beam_3pb_t5a1": create_case_04,  # Legacy (keep for compatibility)
    "04a_beam_3pb_t5a1_bosco": create_case_04a,
    "04b_beam_3pb_t6a1_bosco": create_case_04b,
    "05_wall_c1_cyclic": create_case_05,
    "06_fibre_tensile": create_case_06,
    "07_beam_4pb_jason_4pbt": create_case_07,
    "08_beam_3pb_vvbs3_cfrp": create_case_08,
    "09_beam_4pb_fibres_sorelli": create_case_09,
    "10_wall_c2_cyclic": create_case_10,
    "11_balcony_cantilever_sls": create_case_11,
}

CASE_ALIASES = {
    # Pullout
    "pullout": "01_pullout_lettow",
    "lettow": "01_pullout_lettow",
    # FRP
    "sspot": "02_sspot_frp",
    "frp": "02_sspot_frp",
    # Tensile
    "stn12": "03_tensile_stn12",
    "tensile": "03_tensile_stn12",
    # Beams (legacy)
    "beam": "04_beam_3pb_t5a1",
    "3pb": "04_beam_3pb_t5a1",
    # BOSCO beams
    "t5a1": "04a_beam_3pb_t5a1_bosco",
    "bosco_t5a1": "04a_beam_3pb_t5a1_bosco",
    "t6a1": "04b_beam_3pb_t6a1_bosco",
    "bosco_t6a1": "04b_beam_3pb_t6a1_bosco",
    # Jason 4PB
    "jason": "07_beam_4pb_jason_4pbt",
    "4pb": "07_beam_4pb_jason_4pbt",
    # CFRP
    "vvbs3": "08_beam_3pb_vvbs3_cfrp",
    "cfrp": "08_beam_3pb_vvbs3_cfrp",
    # Sorelli fibres
    "sorelli": "09_beam_4pb_fibres_sorelli",
    # Walls
    "wall": "05_wall_c1_cyclic",
    "c1": "05_wall_c1_cyclic",
    "c2": "10_wall_c2_cyclic",
    "cyclic": "05_wall_c1_cyclic",
    # Fibres (tensile)
    "fibre": "06_fibre_tensile",
    "fiber": "06_fibre_tensile",
    # Balcony cantilever
    "balcony": "11_balcony_cantilever_sls",
    "cantilever": "11_balcony_cantilever_sls",
    "sls": "11_balcony_cantilever_sls",
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

def run_case(case_config, mesh_factor: float = 1.0, dry_run: bool = False, enable_postprocess: bool = True, cli_args=None):
    """
    Run a single case with the XFEM solver.

    Args:
        case_config: CaseConfig instance
        mesh_factor: Mesh refinement factor (1.0 = default)
        dry_run: If True, only print configuration without running
        cli_args: Optional CLI arguments for overrides
    """
    print(f"\n{'='*70}")
    print(f"Running Case: {case_config.case_id}")
    print(f"Description: {case_config.description}")
    print(f"{'='*70}\n")

    # mesh_factor is now applied only in solver_interface.run_case_solver()
    # to avoid double scaling (was being applied here AND in solver)
    if mesh_factor != 1.0:
        nx = int(case_config.geometry.n_elem_x * mesh_factor)
        ny = int(case_config.geometry.n_elem_y * mesh_factor)
        print(f"Mesh adjusted: {nx} x {ny} elements (factor={mesh_factor})")

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

    override_messages = []

    # Apply CLI overrides to case_config
    if cli_args and hasattr(cli_args, 'bulk') and cli_args.bulk is not None:
        case_config.concrete.model_type = cli_args.bulk
        override_messages.append(f"concrete.model_type = {cli_args.bulk}")

    if cli_args and hasattr(cli_args, 'nsteps') and cli_args.nsteps is not None:
        if hasattr(case_config.loading, 'n_steps'):
            case_config.loading.n_steps = cli_args.nsteps
            override_messages.append(f"n_steps = {cli_args.nsteps}")

    if cli_args and hasattr(cli_args, 'cycles') and cli_args.cycles is not None:
        if hasattr(case_config.loading, 'n_cycles_per_target'):
            case_config.loading.n_cycles_per_target = cli_args.cycles
            override_messages.append(f"n_cycles_per_target = {cli_args.cycles}")

    if cli_args and hasattr(cli_args, 'output_dir') and cli_args.output_dir is not None:
        case_config.outputs.output_dir = cli_args.output_dir
        override_messages.append(f"output_dir = {cli_args.output_dir}")

    if cli_args and hasattr(cli_args, 'solver') and cli_args.solver is not None:
        override_messages.append(f"solver = {cli_args.solver}")

    if cli_args and hasattr(cli_args, 'bond_slip') and cli_args.bond_slip is not None:
        override_messages.append(f"bond_slip = {cli_args.bond_slip}")

    if cli_args and hasattr(cli_args, 'use_numba') and cli_args.use_numba:
        override_messages.append("use_numba = True")
    elif cli_args and hasattr(cli_args, 'no_numba') and cli_args.no_numba:
        override_messages.append("use_numba = False")

    if override_messages:
        print("Active overrides:")
        for message in override_messages:
            print(f"  - {message}")
    else:
        print("Active overrides: none")

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
        from examples.gutierrez_thesis.solver_interface import run_case_solver

        results = run_case_solver(case_config, mesh_factor=mesh_factor, enable_postprocess=enable_postprocess, cli_args=cli_args)

        # Save results
        print("\nSaving results...")
        import csv
        history_file = output_dir / "load_displacement.csv"
        from collections.abc import Mapping

        def _get_value(row, key, idx=None, default=0.0):
            if isinstance(row, Mapping):
                return row.get(key, default)
            if idx is not None:
                return row[idx]
            return default

        with open(history_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'u_mm', 'P_kN', 'M_kNm', 'kappa', 'R',
                             'crack_tip_x', 'crack_tip_y', 'angle_deg', 'crack_active',
                             'W_plastic', 'W_damage_t', 'W_damage_c', 'W_cohesive', 'W_total'])
            for row in results['history']:
                # Convert units: m → mm, N → kN, J → J
                row_out = [
                    int(_get_value(row, "step", idx=0, default=0)),              # step
                    _get_value(row, "u", idx=1, default=0.0) * 1e3,              # u [mm]
                    _get_value(row, "P", idx=2, default=0.0) / 1e3,              # P [kN]
                    _get_value(row, "M", idx=3, default=0.0) / 1e3,              # M [kN·m]
                    _get_value(row, "kappa", idx=4, default=0.0),                # kappa
                    _get_value(row, "R", idx=5, default=0.0),                    # R
                    _get_value(row, "crack_tip_x", idx=6, default=0.0),          # crack_tip_x [m]
                    _get_value(row, "crack_tip_y", idx=7, default=0.0),          # crack_tip_y [m]
                    _get_value(row, "angle_deg", idx=8, default=0.0),            # angle [deg]
                    int(_get_value(row, "crack_active", idx=9, default=0)),      # crack_active
                    _get_value(row, "W_plastic", idx=10, default=0.0),           # W_plastic [J]
                    _get_value(row, "W_damage_t", idx=11, default=0.0),          # W_damage_t [J]
                    _get_value(row, "W_damage_c", idx=12, default=0.0),          # W_damage_c [J]
                    _get_value(row, "W_cohesive", idx=13, default=0.0),          # W_cohesive [J]
                    _get_value(row, "W_total", idx=14, default=0.0),             # W_total [J]
                ]
                writer.writerow(row_out)
        print(f"  → {history_file}")

    except Exception as e:
        import traceback
        print(f"\n❌ ERROR: {e}\n")
        print("Traceback:")
        traceback.print_exc()
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
  python -m examples.gutierrez_thesis.run --case t5a1 --bulk dp --dry-run
        """
    )

    parser.add_argument(
        "--case",
        type=str,
        help="Case ID or alias (or 'all' to run all cases)"
    )
    parser.add_argument(
        "--case-config",
        type=str,
        help="Path to YAML/JSON case configuration file (overrides --case)"
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
    parser.add_argument(
        "--nsteps",
        type=int,
        help="Override number of steps"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        help="Override number of cycles per target (for cyclic cases)"
    )
    parser.add_argument(
        "--mesh-factor",
        type=float,
        help="Override mesh factor (alternative to --mesh preset)"
    )
    parser.add_argument(
        "--bulk",
        choices=["elastic", "dp", "cdp_lite", "cdp_full"],
        help="Override concrete bulk material model_type (default: case config)"
    )
    parser.add_argument(
        "--solver",
        choices=["single", "multi"],
        help="Override solver selection (single or multi)"
    )
    parser.add_argument(
        "--bond-slip",
        dest="bond_slip",
        choices=["on", "off"],
        help="Toggle bond-slip behavior without editing case files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory"
    )
    parser.add_argument(
        "--no-post",
        action="store_true",
        help="Disable postprocessing (skip CSV/PNG generation)"
    )

    # Numba acceleration control (mutually exclusive)
    numba_group = parser.add_mutually_exclusive_group()
    numba_group.add_argument(
        "--use-numba",
        action="store_true",
        help="Force enable Numba acceleration (override model default)"
    )
    numba_group.add_argument(
        "--no-numba",
        action="store_true",
        help="Force disable Numba acceleration (override model default)"
    )

    args = parser.parse_args()

    # List cases
    if args.list:
        list_cases()
        sys.exit(0)

    # Validate arguments
    if not args.case and not args.case_config:
        parser.print_help()
        print("\nError: --case or --case-config is required (or use --list)")
        sys.exit(1)

    # Get mesh factor (CLI override takes precedence)
    if args.mesh_factor is not None:
        mesh_factor = args.mesh_factor
    else:
        mesh_factor = MESH_PRESETS[args.mesh]

    # Load from config file if provided (takes precedence over --case)
    if args.case_config:
        from examples.gutierrez_thesis.case_config import CaseConfig
        config_path = Path(args.case_config)

        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)

        print(f"Loading configuration from: {config_path}")

        if config_path.suffix in ['.yaml', '.yml']:
            case_config = CaseConfig.load_yaml(str(config_path))
        elif config_path.suffix == '.json':
            case_config = CaseConfig.load_json(str(config_path))
        else:
            print(f"Error: Unsupported config file format: {config_path.suffix}")
            print("Supported formats: .yaml, .yml, .json")
            sys.exit(1)

        enable_post = not args.no_post
        run_case(case_config, mesh_factor, args.dry_run, enable_postprocess=enable_post, cli_args=args)

    # Run cases from registry
    elif args.case.lower() == "all":
        # Run all cases
        print(f"\n{'='*70}")
        print(f"Running ALL {len(CASE_REGISTRY)} cases with mesh={args.mesh}")
        print(f"{'='*70}")

        for case_id, factory in CASE_REGISTRY.items():
            case_config = factory()
            enable_post = not args.no_post
            run_case(case_config, mesh_factor, args.dry_run, enable_postprocess=enable_post, cli_args=args)

    else:
        # Run single case
        case_id = resolve_case_id(args.case)
        if not case_id:
            sys.exit(1)

        factory = CASE_REGISTRY[case_id]
        case_config = factory()
        enable_post = not args.no_post
        run_case(case_config, mesh_factor, args.dry_run, enable_postprocess=enable_post, cli_args=args)


if __name__ == "__main__":
    main()
