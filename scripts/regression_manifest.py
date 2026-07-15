"""Generate and compare the compact numerical/contract regression manifest."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from examples.gutierrez_thesis.case_config import normalize_case_config
from examples.gutierrez_thesis.cases.case_01_pullout_lettow import create_case_01
from examples.gutierrez_thesis.cases.case_02_sspot_frp import create_case_02
from examples.gutierrez_thesis.cases.case_03_tensile_stn12 import create_case_03
from examples.gutierrez_thesis.cases.case_09_beam_4pb_fibres_sorelli import create_case_09
from examples.gutierrez_thesis.solver_interface import (
    _should_use_multicrack,
    generate_cyclic_u_targets,
    map_bond_law,
)
from xfem_clean.cohesive_laws import CohesiveLaw
from xfem_clean.fibre_bridging import fibre_config_from_case, fibre_traction_tangent
from xfem_clean.results import AnalysisResult
from xfem_clean.xfem.analysis_single import run_analysis_xfem
from xfem_clean.xfem.model import XFEMModel
from xfem_clean.xfem.multicrack import run_analysis_xfem_multicrack

REFERENCE = ROOT / "tests" / "regression" / "canonical_manifest.json"


def _dependency_versions():
    found = {}
    for package in ("numpy", "scipy", "matplotlib", "numba", "pytest"):
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = None
    return found


def _collected_test_count():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    match = re.search(r"(\d+) tests? collected", completed.stdout + completed.stderr)
    return int(match.group(1)) if match else None


def _model():
    return XFEMModel(
        L=1.0, H=0.2, b=0.1, E=30e9, nu=0.2, ft=1e12, fc=30e6,
        Gf=100.0, steel_A_total=0.0, steel_E=200e9, steel_fy=500e6,
        steel_fu=600e6, steel_Eh=0.0, newton_maxit=10, max_subdiv=2,
        use_numba=False,
    )


def _solver_probes():
    single_model = _model()
    multi_model = _model()
    single_law = CohesiveLaw(Kn=1e12, ft=single_model.ft, Gf=single_model.Gf)
    multi_law = CohesiveLaw(Kn=1e12, ft=multi_model.ft, Gf=multi_model.Gf)
    with contextlib.redirect_stdout(io.StringIO()):
        single = run_analysis_xfem(
            single_model, 4, 2, 1, 1e-6, law=single_law, return_bundle=True,
        )
        multi = run_analysis_xfem_multicrack(
            multi_model, 4, 2, 1, 1e-6, law=multi_law, return_bundle=True,
        )
    single_result = AnalysisResult.from_solver_bundle(
        single, engine="single", material="elastic", bond_layer_count=0,
        use_numba=False, compat_mode=False,
    )
    multi_result = AnalysisResult.from_solver_bundle(
        multi, engine="multi", material="elastic", bond_layer_count=0,
        use_numba=False, compat_mode=False,
    )
    return {
        "elastic_single": {
            "load_N": single_result.steps[-1]["P"],
            "displacement_m": single_result.steps[-1]["u"],
            "step_count": len(single_result.steps),
            "converged": single_result.solver_meta["converged"],
        },
        "multicrack": {
            "load_N": multi_result.steps[-1]["P"],
            "displacement_m": multi_result.steps[-1]["u"],
            "crack_count": len(multi_result.cracks),
            "step_count": len(multi_result.steps),
            "converged": multi_result.solver_meta["converged"],
        },
    }


def build_manifest():
    probes = _solver_probes()

    pullout = normalize_case_config(create_case_01())
    pullout_law = map_bond_law(pullout.rebar_layers[0].bond_law, pullout.case_id)
    tau, tangent = pullout_law.tau_and_tangent(0.5e-3, 0.5e-3)
    probes["pullout_bond"] = {"slip_m": 0.5e-3, "stress_Pa": tau, "tangent_Pa_m": tangent}

    frp = normalize_case_config(create_case_02())
    frp_law = map_bond_law(frp.frp_sheets[0].bond_law, frp.case_id)
    tau, tangent = frp_law.tau_and_tangent(frp_law.s1, frp_law.s1)
    probes["frp"] = {"slip_m": frp_law.s1, "stress_Pa": tau, "tangent_Pa_m": tangent}

    fibre_case = normalize_case_config(create_case_09())
    fibre_cfg = fibre_config_from_case(
        fibre_case.fibres, fibre_case.geometry.thickness * 1e-3,
    )
    traction, tangent = fibre_traction_tangent(1e-4, 0.01, fibre_cfg)
    probes["fibres"] = {
        "density_fibres_m2": fibre_cfg.density_m2,
        "explicit_fraction": fibre_cfg.explicit_fraction,
        "seed": fibre_cfg.random_seed,
        "traction_Pa": traction,
        "tangent_Pa_m": tangent,
    }

    cyclic = generate_cyclic_u_targets([0.01], 1) * 1e-3
    force = 2e6 * cyclic
    probes["cyclic_reduced"] = {
        "step_count": len(cyclic),
        "net_elastic_work_J": float(np.trapz(force, cyclic)),
        "max_displacement_m": float(np.max(cyclic)),
        "min_displacement_m": float(np.min(cyclic)),
    }

    tensile = normalize_case_config(create_case_03())
    probes["configuration"] = {
        "multicrack_engine": "multi" if _should_use_multicrack(tensile) else "single",
        "material": tensile.concrete.model_type,
        "bond_layer_count": len(tensile.rebar_layers) + len(tensile.frp_sheets),
        "result_schema_version": 1,
    }

    with tempfile.TemporaryDirectory(prefix="xfem-manifest-") as temp_dir:
        output = Path(temp_dir) / "curve.csv"
        with output.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("u_m", "P_N"))
            writer.writerow((0.0, 0.0))
            writer.writerow((1e-6, probes["elastic_single"]["load_N"]))
        probes["csv_output"] = {
            "rows": 2,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }

    payload = {
        "environment": {
            "python": platform.python_version(),
            "dependencies": _dependency_versions(),
            "collected_tests": _collected_test_count(),
        },
        "probes": probes,
        "tolerances": {
            "solver_load_relative": 1e-8,
            "constitutive_relative": 1e-12,
            "work_absolute_J": 1e-15,
        },
    }
    payload["semantic_sha256"] = hashlib.sha256(
        json.dumps(probes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _compare(actual, expected, path="probes", *, rtol=1e-8, atol=1e-15):
    errors = []
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key not in actual:
                errors.append(f"{path}.{key}: missing")
            else:
                errors.extend(_compare(actual[key], value, f"{path}.{key}", rtol=rtol, atol=atol))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not np.isclose(actual, expected, rtol=rtol, atol=atol):
            errors.append(f"{path}: expected {expected!r}, got {actual!r}")
    elif actual != expected:
        errors.append(f"{path}: expected {expected!r}, got {actual!r}")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="print the generated manifest")
    args = parser.parse_args()
    manifest = build_manifest()
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    reference = json.loads(REFERENCE.read_text())
    errors = _compare(manifest["probes"], reference["probes"])
    if errors:
        print("Regression manifest mismatch:", file=sys.stderr)
        print("\n".join(f"- {item}" for item in errors), file=sys.stderr)
        return 1
    print(
        f"manifest OK: {len(manifest['probes'])} probes, "
        f"{manifest['environment']['collected_tests']} tests collected, "
        f"sha256={manifest['semantic_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
