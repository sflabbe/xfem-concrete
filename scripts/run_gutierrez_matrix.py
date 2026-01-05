"""Gutierrez example evaluation matrix runner."""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from examples.gutierrez_thesis.case_config import CyclicLoading, MonotonicLoading
from examples.gutierrez_thesis.run import CASE_REGISTRY
from examples.gutierrez_thesis.solver_interface import _should_use_multicrack, run_case_solver


@dataclass(frozen=True)
class MatrixConfig:
    cfg_id: str
    solver: str
    bulk: str | None
    bond_slip: str | None
    use_numba: str


@dataclass
class RunResult:
    case_id: str
    cfg_id: str
    solver: str
    bulk: str
    bond_slip: str
    use_numba: str
    status: str
    nsteps_done: int
    max_abs_P: float | None
    any_nonfinite: bool
    first_nonfinite_tag: str
    warnings: list[str]


def _apply_smoke_limits(case_config, max_steps: int) -> None:
    if isinstance(case_config.loading, MonotonicLoading):
        case_config.loading.n_steps = min(case_config.loading.n_steps, max_steps)
    elif isinstance(case_config.loading, CyclicLoading):
        case_config.loading.targets = case_config.loading.targets[:1]
        case_config.loading.n_cycles_per_target = min(case_config.loading.n_cycles_per_target, 1)


def _build_cli_args(config: MatrixConfig) -> SimpleNamespace:
    use_numba = config.use_numba == "on"
    no_numba = config.use_numba == "off"
    return SimpleNamespace(
        bulk=config.bulk,
        solver=None if config.solver == "auto" else config.solver,
        bond_slip=config.bond_slip,
        use_numba=use_numba,
        no_numba=no_numba,
    )


def _parse_nonfinite_tag(message: str) -> str:
    if "Non-finite values detected" not in message:
        return ""
    matches = re.findall(r"([A-Za-z0-9_]+): nan=(\d+), inf=(\d+)", message)
    for name, nan_count, inf_count in matches:
        if int(nan_count) > 0 or int(inf_count) > 0:
            return name
    return "nonfinite"


def _extract_failure_fields(message: str) -> tuple[str, str, str]:
    u0 = _match_or_na(message, r"u0=([\deE+\-.]+)")
    u1 = _match_or_na(message, r"u1=([\deE+\-.]+)")
    rhs = _match_or_na(message, r"last\|\|rhs\|\|=([^,\s]+)")
    return u0, u1, rhs


def _match_or_na(message: str, pattern: str) -> str:
    match = re.search(pattern, message)
    return match.group(1) if match else "n/a"


def _extract_history_metrics(history: Any) -> tuple[int, float | None, bool, str]:
    if history is None:
        return 0, None, False, ""

    nsteps = len(history)
    max_abs_p = None
    first_nonfinite_tag = ""

    if nsteps == 0:
        return 0, None, False, ""

    if isinstance(history, np.ndarray) and history.dtype != object:
        columns = {
            "u": history[:, 1],
            "P": history[:, 2],
            "M": history[:, 3],
            "kappa": history[:, 4],
            "W_plastic": history[:, 10],
            "W_damage_t": history[:, 11],
            "W_damage_c": history[:, 12],
            "W_cohesive": history[:, 13],
            "W_total": history[:, 14],
        }
        max_abs_p = float(np.nanmax(np.abs(columns["P"])))
        first_idx = None
        for name, data in columns.items():
            nonfinite = np.where(~np.isfinite(data))[0]
            if nonfinite.size > 0:
                idx = int(nonfinite[0])
                if first_idx is None or idx < first_idx:
                    first_idx = idx
                    first_nonfinite_tag = f"history.{name}"
        return nsteps, max_abs_p, first_nonfinite_tag != "", first_nonfinite_tag

    if isinstance(history, np.ndarray) and history.dtype == object:
        entries = history.tolist()
    else:
        entries = list(history)

    ps = []
    first_idx = None
    for idx, row in enumerate(entries):
        u_val = row.get("u") if isinstance(row, dict) else None
        p_val = row.get("P") if isinstance(row, dict) else None
        ps.append(p_val)
        for tag, value in (("u", u_val), ("P", p_val)):
            if value is not None and not np.isfinite(float(value)):
                if first_idx is None or idx < first_idx:
                    first_idx = idx
                    first_nonfinite_tag = f"history.{tag}"

    finite_ps = [abs(float(p)) for p in ps if p is not None and np.isfinite(float(p))]
    if finite_ps:
        max_abs_p = float(max(finite_ps))

    return nsteps, max_abs_p, first_nonfinite_tag != "", first_nonfinite_tag


def _inspect_outputs(results: dict[str, Any]) -> tuple[int, float | None, bool, str]:
    nsteps, max_abs_p, any_nonfinite, tag = _extract_history_metrics(results.get("history"))
    if not any_nonfinite:
        u_vec = results.get("u")
        if u_vec is not None and not np.isfinite(np.asarray(u_vec, dtype=float)).all():
            any_nonfinite = True
            tag = "u"
    return nsteps, max_abs_p, any_nonfinite, tag


def _build_matrix(case_id: str, case_config) -> list[MatrixConfig]:
    configs = [
        MatrixConfig(
            cfg_id="default",
            solver="auto",
            bulk=case_config.concrete.model_type,
            bond_slip=None,
            use_numba="auto",
        )
    ]

    if _should_use_multicrack(case_config):
        configs.append(
            MatrixConfig(
                cfg_id="bond-slip-off",
                solver="auto",
                bulk=case_config.concrete.model_type,
                bond_slip="off",
                use_numba="auto",
            )
        )

    if case_config.concrete.model_type in {"cdp_lite", "cdp_full"}:
        configs.append(
            MatrixConfig(
                cfg_id="bulk-elastic",
                solver="auto",
                bulk="elastic",
                bond_slip=None,
                use_numba="auto",
            )
        )

    if case_id in {"03_tensile_stn12", "08_beam_3pb_vvbs3_cfrp"}:
        configs.extend(
            [
                MatrixConfig(
                    cfg_id="numba-on",
                    solver="auto",
                    bulk=case_config.concrete.model_type,
                    bond_slip=None,
                    use_numba="on",
                ),
                MatrixConfig(
                    cfg_id="numba-off",
                    solver="auto",
                    bulk=case_config.concrete.model_type,
                    bond_slip=None,
                    use_numba="off",
                ),
            ]
        )

    return configs


def run_matrix(
    *,
    output_dir: Path,
    mesh_factor: float = 0.35,
    max_steps: int = 4,
    cases: Iterable[str] | None = None,
) -> list[RunResult]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    case_ids = sorted(cases) if cases is not None else sorted(CASE_REGISTRY.keys())
    results: list[RunResult] = []

    for case_id in case_ids:
        case_factory = CASE_REGISTRY[case_id]
        base_case = case_factory()
        configs = _build_matrix(case_id, base_case)

        for config in configs:
            case_config = case_factory()
            _apply_smoke_limits(case_config, max_steps=max_steps)
            case_config.outputs.output_dir = str(output_dir / case_id / config.cfg_id)

            cli_args = _build_cli_args(config)

            captured_warnings: list[str] = []
            stdout = io.StringIO()
            stderr = io.StringIO()
            status = "ok"
            any_nonfinite = False
            first_nonfinite_tag = ""
            nsteps_done = 0
            max_abs_p = None

            with warnings.catch_warnings(record=True) as warning_records:
                warnings.simplefilter("always")
                try:
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        run_results = run_case_solver(
                            case_config,
                            mesh_factor=mesh_factor,
                            enable_postprocess=False,
                            return_bundle=True,
                            cli_args=cli_args,
                        )
                    nsteps_done, max_abs_p, any_nonfinite, first_nonfinite_tag = _inspect_outputs(run_results)
                    if any_nonfinite:
                        status = "fail"
                except Exception as exc:  # noqa: BLE001 - deliberate summary capture
                    status = "fail"
                    message = str(exc)
                    first_nonfinite_tag = _parse_nonfinite_tag(message)
                    any_nonfinite = bool(first_nonfinite_tag)
                    u0, u1, rhs = _extract_failure_fields(message)
                    print(
                        f"{case_id} {config.cfg_id} u0={u0} u1={u1} ncr=n/a "
                        f"last||rhs||={rhs} first_nonfinite_tag={first_nonfinite_tag or 'n/a'}"
                    )

            for warning in warning_records:
                captured_warnings.append(f"{warning.category.__name__}: {warning.message}")

            results.append(
                RunResult(
                    case_id=case_id,
                    cfg_id=config.cfg_id,
                    solver=config.solver,
                    bulk=config.bulk or case_config.concrete.model_type,
                    bond_slip=config.bond_slip or "auto",
                    use_numba=config.use_numba,
                    status=status,
                    nsteps_done=nsteps_done,
                    max_abs_P=max_abs_p,
                    any_nonfinite=any_nonfinite,
                    first_nonfinite_tag=first_nonfinite_tag,
                    warnings=captured_warnings,
                )
            )

    _write_summary_files(results, output_dir)
    _print_summary_table(results)
    return results


def _write_summary_files(results: list[RunResult], output_dir: Path) -> None:
    json_path = output_dir / "gutierrez_matrix_summary.json"
    csv_path = output_dir / "gutierrez_matrix_summary.csv"

    json_payload = [result.__dict__ for result in results]
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_payload, handle, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_id",
                "cfg_id",
                "solver",
                "bulk",
                "bond_slip",
                "use_numba",
                "status",
                "nsteps_done",
                "max_abs_P",
                "any_nonfinite",
                "first_nonfinite_tag",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.case_id,
                    result.cfg_id,
                    result.solver,
                    result.bulk,
                    result.bond_slip,
                    result.use_numba,
                    result.status,
                    result.nsteps_done,
                    "" if result.max_abs_P is None else f"{result.max_abs_P:.6e}",
                    str(result.any_nonfinite).lower(),
                    result.first_nonfinite_tag,
                ]
            )


def _print_summary_table(results: list[RunResult]) -> None:
    headers = [
        "case_id",
        "cfg_id",
        "status",
        "steps",
        "max|P|",
        "nonfinite",
        "first_tag",
    ]
    rows = [
        [
            result.case_id,
            result.cfg_id,
            result.status,
            str(result.nsteps_done),
            "" if result.max_abs_P is None else f"{result.max_abs_P:.3e}",
            "yes" if result.any_nonfinite else "no",
            result.first_nonfinite_tag,
        ]
        for result in results
    ]

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def _format_row(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    divider = "-+-".join("-" * width for width in widths)

    print(_format_row(headers))
    print(divider)
    for row in rows:
        print(_format_row(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gutierrez matrix with diagnostics.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scripts/_gutierrez_matrix"),
        help="Directory for JSON/CSV summaries.",
    )
    parser.add_argument(
        "--mesh-factor",
        type=float,
        default=0.35,
        help="Mesh factor for all cases (default: 0.35).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=4,
        help="Max monotonic steps for smoke runs (default: 4).",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        help="Optional list of case IDs to run.",
    )
    args = parser.parse_args()

    run_matrix(
        output_dir=args.output_dir,
        mesh_factor=args.mesh_factor,
        max_steps=args.max_steps,
        cases=args.cases,
    )


if __name__ == "__main__":
    main()
