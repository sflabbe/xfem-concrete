"""
Gutiérrez Thesis Case Suite Runner

CLI to run all thesis examples from Chapter 5 + Appendix A.

Usage:
    python -m examples.gutierrez_thesis.run --case STN12 --mesh fine
    python -m examples.gutierrez_thesis.run --case all --mesh coarse
    python -m examples.gutierrez_thesis.run --list
"""

import argparse
import json
import os
import sys

# Force local imports from this repository first.
# This avoids accidentally importing an older xfem_clean from site packages.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
import time
from pathlib import Path
from typing import Optional

from examples.gutierrez_thesis.case_config import (
    CaseConfigurationError,
    normalize_case_config,
)
from examples.gutierrez_thesis.catalog import (
    CASE_ALIASES,
    CASE_DEFINITIONS,
    CASE_METADATA,
    CASE_REGISTRY,
    evaluate_compatibility,
    resolve_case_id as resolve_catalog_case_id,
)


# ============================================================================
# CASE REGISTRY
# ============================================================================

MESH_PRESETS = {
    "coarse": 0.5,   # Coarsening factor
    "medium": 1.0,
    "fine": 1.5,
    "very_fine": 2.0,
}


# ============================================================================
# SOLVER INTERFACE
# ============================================================================

def _effective_numba(cli_args) -> bool:
    from xfem_clean.numba.utils import NUMBA_AVAILABLE

    if cli_args is not None and getattr(cli_args, "use_numba", False):
        return True
    if cli_args is not None and getattr(cli_args, "no_numba", False):
        return False
    return bool(NUMBA_AVAILABLE)


def _apply_cli_overrides(case_config, cli_args):
    """Apply the documented overrides to a normalized copy and record provenance."""
    case = normalize_case_config(case_config)
    overrides = []

    def apply(field: str, owner, attribute: str, value, option: str) -> None:
        previous = getattr(owner, attribute)
        setattr(owner, attribute, value)
        overrides.append((field, previous, value, option))

    if cli_args is not None:
        if getattr(cli_args, "bulk", None) is not None:
            apply("concrete.model_type", case.concrete, "model_type", cli_args.bulk, "--bulk")
        if getattr(cli_args, "nsteps", None) is not None:
            if not hasattr(case.loading, "n_steps"):
                raise CaseConfigurationError(
                    "loading.n_steps", cli_args.nsteps, ["monotonic loading only"],
                    case_id=case.case_id, source="CLI --nsteps",
                )
            apply("loading.n_steps", case.loading, "n_steps", cli_args.nsteps, "--nsteps")
        if getattr(cli_args, "max_displacement", None) is not None:
            if not hasattr(case.loading, "max_displacement"):
                raise CaseConfigurationError(
                    "loading.max_displacement", cli_args.max_displacement,
                    ["monotonic loading only"], case_id=case.case_id,
                    source="CLI --max-displacement",
                )
            apply(
                "loading.max_displacement_mm", case.loading, "max_displacement",
                cli_args.max_displacement, "--max-displacement",
            )
        if getattr(cli_args, "cycles", None) is not None:
            if not hasattr(case.loading, "n_cycles_per_target"):
                raise CaseConfigurationError(
                    "loading.n_cycles_per_target", cli_args.cycles,
                    ["cyclic loading only"], case_id=case.case_id, source="CLI --cycles",
                )
            apply(
                "loading.n_cycles_per_target", case.loading, "n_cycles_per_target",
                cli_args.cycles, "--cycles",
            )
        if getattr(cli_args, "output_dir", None) is not None:
            apply(
                "outputs.output_dir", case.outputs, "output_dir", cli_args.output_dir,
                "--output-dir",
            )
        if getattr(cli_args, "solver", None) is not None:
            apply("solver_engine", case, "solver_engine", cli_args.solver, "--solver")

    case = normalize_case_config(case)
    setattr(
        case,
        "_overrides",
        tuple(
            {
                "field": field,
                "previous": previous,
                "effective": effective,
                "source": f"CLI {option}",
            }
            for field, previous, effective, option in overrides
        ),
    )
    return case, tuple(overrides)


def _canonical_view(
    case, *, mesh_factor: float, compatibility, use_numba: bool,
    bond_slip_effective: bool,
) -> dict:
    metadata = CASE_METADATA.get(case.case_id)
    nx = int(case.geometry.n_elem_x * mesh_factor)
    ny = int(case.geometry.n_elem_y * mesh_factor)
    loading = {
        "type": case.loading.loading_type,
        "steps": getattr(case.loading, "n_steps", len(getattr(case.loading, "targets", ()))),
        "target_mm": getattr(
            case.loading,
            "max_displacement",
            max((abs(value) for value in getattr(case.loading, "targets", (0.0,))), default=0.0),
        ),
    }
    return {
        "case_id": case.case_id,
        "schema_version": case.schema_version,
        "source": getattr(case, "_source", None),
        "definition_status": metadata.status if metadata else "external-config",
        "validation_source": metadata.validation_source if metadata else None,
        "engine": case.solver_engine,
        "compatibility": compatibility.state,
        "compatibility_reason": compatibility.reason,
        "material": case.concrete.model_type,
        "geometry_mm": {
            "length": case.geometry.length,
            "height": case.geometry.height,
            "thickness": case.geometry.thickness,
        },
        "mesh": {"nx": nx, "ny": ny, "factor": mesh_factor},
        "loading": loading,
        "features": {
            "rebar_layers": len(case.rebar_layers),
            "frp_sheets": len(case.frp_sheets),
            "fibres": case.fibres is not None,
            "subdomains": len(case.subdomains),
            "bond_slip_effective": bond_slip_effective,
        },
        "solver_options": {
            "tolerance": case.tolerance,
            "line_search": case.use_line_search,
            "substepping": case.use_substepping,
        },
        "numba": {
            "requested_or_auto_effective": use_numba,
            "support": compatibility.numba_state,
        },
        "output_dir": case.outputs.output_dir,
        "result_contract": "AnalysisResult/schema-1",
    }


def run_case(
    case_config,
    mesh_factor: float = 1.0,
    dry_run: bool = False,
    enable_postprocess: bool = True,
    cli_args=None,
):
    """Validate and run one canonical case, returning its ``AnalysisResult``."""
    case_config, overrides = _apply_cli_overrides(case_config, cli_args)
    if not mesh_factor > 0.0:
        raise CaseConfigurationError(
            "mesh_factor", mesh_factor, ["a positive value"],
            case_id=case_config.case_id, source="CLI mesh preset/--mesh-factor",
        )
    nx = int(case_config.geometry.n_elem_x * mesh_factor)
    ny = int(case_config.geometry.n_elem_y * mesh_factor)
    if nx < 1 or ny < 1:
        raise CaseConfigurationError(
            "effective_mesh", (nx, ny), ["at least 1 x 1 elements"],
            case_id=case_config.case_id, source="CLI mesh preset/--mesh-factor",
        )

    use_numba = _effective_numba(cli_args)
    bond_slip_effective = bool(case_config.rebar_layers or case_config.frp_sheets)
    if cli_args is not None and getattr(cli_args, "bond_slip", None) is not None:
        bond_slip_effective = cli_args.bond_slip == "on"
    compatibility = evaluate_compatibility(case_config, use_numba=use_numba)

    print(f"\n{'='*70}")
    print(f"Running Case: {case_config.case_id}")
    print(f"Description: {case_config.description}")
    print(f"{'='*70}\n")

    if mesh_factor != 1.0:
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

    runtime_overrides = list(overrides)
    if cli_args is not None and getattr(cli_args, "bond_slip", None) is not None:
        runtime_overrides.append(("bond_slip", "case default", cli_args.bond_slip, "--bond-slip"))
    if cli_args is not None and getattr(cli_args, "use_numba", False):
        runtime_overrides.append(("use_numba", "auto", True, "--use-numba"))
    elif cli_args is not None and getattr(cli_args, "no_numba", False):
        runtime_overrides.append(("use_numba", "auto", False, "--no-numba"))

    if runtime_overrides:
        print("Active overrides:")
        for field, previous, effective, option in runtime_overrides:
            print(
                f"  - Override: {field} = {effective} "
                f"(previous={previous!r}, source=CLI {option})"
            )
    else:
        print("Active overrides: none")

    view = _canonical_view(
        case_config,
        mesh_factor=mesh_factor,
        compatibility=compatibility,
        use_numba=use_numba,
        bond_slip_effective=bond_slip_effective,
    )
    print("Canonical configuration:")
    print(json.dumps(view, indent=2, sort_keys=True, ensure_ascii=False))

    if dry_run:
        print("DRY RUN - canonical validation complete; solver not executed.\n")
        return case_config

    if not compatibility.supported:
        raise CaseConfigurationError(
            "engine_compatibility",
            f"{case_config.solver_engine}/{case_config.concrete.model_type}/numba={use_numba}",
            [compatibility.reason], case_id=case_config.case_id,
            source=getattr(case_config, "_source", None),
        )

    # Create output directory
    output_dir = Path(case_config.outputs.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run solver
    print("Starting solver...\n")
    t0 = time.time()

    from examples.gutierrez_thesis.solver_interface import run_case_solver
    from xfem_clean.results import AnalysisResult

    results = run_case_solver(
        case_config,
        mesh_factor=mesh_factor,
        enable_postprocess=enable_postprocess,
        cli_args=cli_args,
    )
    if not isinstance(results, AnalysisResult):
        raise TypeError(
            f"Case {case_config.case_id} returned {type(results).__name__}; "
            "expected AnalysisResult"
        )

    print("\nSaving results...")
    import csv
    history_file = output_dir / "load_displacement.csv"
    with open(history_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'u_mm', 'P_kN', 'M_kNm', 'kappa', 'R',
                         'crack_tip_x', 'crack_tip_y', 'angle_deg', 'crack_active',
                         'W_plastic', 'W_damage_t', 'W_damage_c', 'W_cohesive', 'W_total'])
        for row in results.steps:
            writer.writerow([
                int(row.get("step", 0)), row.get("u", 0.0) * 1e3,
                row.get("P", 0.0) / 1e3, row.get("M", 0.0) / 1e3,
                row.get("kappa", 0.0), row.get("R", 0.0),
                row.get("crack_tip_x", 0.0), row.get("crack_tip_y", 0.0),
                row.get("angle_deg", 0.0), int(row.get("crack_active", 0)),
                row.get("W_plastic", 0.0), row.get("W_damage_t", 0.0),
                row.get("W_damage_c", 0.0), row.get("W_cohesive", 0.0),
                row.get("W_total", 0.0),
            ])
    print(f"  → {history_file}")

    t_elapsed = time.time() - t0
    print(f"\n✓ Case completed in {t_elapsed:.1f} seconds")
    print(f"✓ Outputs saved to: {output_dir}\n")
    return results


# ============================================================================
# CLI
# ============================================================================

def list_cases():
    """Print available cases"""
    print("\nAvailable Cases:")
    print("=" * 70)
    for definition in CASE_DEFINITIONS:
        case = normalize_case_config(
            definition.factory(), source=definition.factory_source
        )
        aliases = ",".join(definition.aliases)
        print(
            f"  {definition.case_id:30s} engine={case.solver_engine:5s} "
            f"status={definition.status:31s} aliases={aliases}"
        )
    print("=" * 70)
    print(f"\nTotal: {len(CASE_REGISTRY)} cases")
    print("\nMesh presets: coarse, medium (default), fine, very_fine\n")


def resolve_case_id(name: str) -> Optional[str]:
    """Resolve exact case ID or alias without order-dependent partial matching."""
    resolved = resolve_catalog_case_id(name)
    if resolved is None:
        print(f"Unknown case: '{name}'")
    return resolved


def main() -> int:
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
        "--max-displacement",
        type=float,
        help="Override maximum monotonic displacement in mm"
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
        return 0

    # Validate arguments
    if not args.case and not args.case_config:
        parser.print_help()
        print("\nError: --case or --case-config is required (or use --list)")
        return 2

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
            return 2

        print(f"Loading configuration from: {config_path}")

        if config_path.suffix in ['.yaml', '.yml']:
            case_config = CaseConfig.load_yaml(str(config_path))
        elif config_path.suffix == '.json':
            case_config = CaseConfig.load_json(str(config_path))
        else:
            print(f"Error: Unsupported config file format: {config_path.suffix}")
            print("Supported formats: .yaml, .yml, .json")
            return 2

        enable_post = not args.no_post
        run_case(case_config, mesh_factor, args.dry_run, enable_postprocess=enable_post, cli_args=args)

    # Run cases from registry
    elif args.case.lower() == "all":
        # Run all cases
        print(f"\n{'='*70}")
        print(f"Running ALL {len(CASE_REGISTRY)} cases with mesh={args.mesh}")
        print(f"{'='*70}")

        for definition in CASE_DEFINITIONS:
            case_config = normalize_case_config(
                definition.factory(), source=definition.factory_source
            )
            enable_post = not args.no_post
            run_case(case_config, mesh_factor, args.dry_run, enable_postprocess=enable_post, cli_args=args)

    else:
        # Run single case
        case_id = resolve_case_id(args.case)
        if not case_id:
            return 2

        factory = CASE_REGISTRY[case_id]
        definition = CASE_METADATA[case_id]
        case_config = normalize_case_config(factory(), source=definition.factory_source)
        enable_post = not args.no_post
        run_case(case_config, mesh_factor, args.dry_run, enable_postprocess=enable_post, cli_args=args)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (CaseConfigurationError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
