"""Generate and compare the compact numerical/contract regression manifest."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import sys
import tempfile
from collections.abc import Mapping
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
from xfem_clean.utils.numpy_compat import trapezoid
from xfem_clean.xfem.analysis_single import run_analysis_xfem
from xfem_clean.xfem.model import XFEMModel
from xfem_clean.xfem.multicrack import run_analysis_xfem_multicrack
from tests.process_utils import run_process

REFERENCE = ROOT / "tests" / "regression" / "canonical_manifest.json"
SCHEMA_VERSION = 2
CANONICALIZATION_VERSION = 1
CANONICAL_FLOAT_SIGNIFICANT_DIGITS = 15

# Numerical values are compared directly, never after rounding. Every floating
# probe must be listed here; everything else is discrete metadata and exact.
NUMERIC_TOLERANCES = {
    "solver_load": {"rtol": 1e-8, "atol": 1e-12},
    "constitutive": {"rtol": 1e-12, "atol": 1e-15},
    "displacement": {"rtol": 1e-12, "atol": 1e-15},
    "work": {"rtol": 0.0, "atol": 1e-15},
    "exact_numeric": {"rtol": 0.0, "atol": 0.0},
}

NUMERIC_FIELDS = {
    "cyclic_reduced.max_displacement_m": "displacement",
    "cyclic_reduced.min_displacement_m": "displacement",
    "cyclic_reduced.net_elastic_work_J": "work",
    "csv_output.last_load_N": "solver_load",
    "csv_output.last_u_m": "displacement",
    "csv_output.origin_load_N": "exact_numeric",
    "csv_output.origin_u_m": "exact_numeric",
    "elastic_single.displacement_m": "displacement",
    "elastic_single.load_N": "solver_load",
    "fibres.density_fibres_m2": "exact_numeric",
    "fibres.explicit_fraction": "exact_numeric",
    "fibres.tangent_Pa_m": "constitutive",
    "fibres.traction_Pa": "constitutive",
    "frp.slip_m": "exact_numeric",
    "frp.stress_Pa": "constitutive",
    "frp.tangent_Pa_m": "constitutive",
    "multicrack.displacement_m": "displacement",
    "multicrack.load_N": "solver_load",
    "pullout_bond.slip_m": "exact_numeric",
    "pullout_bond.stress_Pa": "constitutive",
    "pullout_bond.tangent_Pa_m": "constitutive",
}

PROVENANCE = {
    "created": "2026-07-15",
    "case09_prior_results": "invalid: density was multiplied by 10000 twice",
    "numerical_policy": (
        "Floating probes use named rtol/atol comparisons; the canonical hash "
        "covers only discrete metadata and the numerical comparison contract."
    ),
}

CANONICALIZATION = {
    "version": CANONICALIZATION_VERSION,
    "format": "JSON",
    "encoding": "UTF-8",
    "newline": "LF",
    "field_order": "lexicographic",
    "record_order": "canonical-value",
    "float_policy": (
        "15 significant decimal digits; -0 is 0; NaN/+Inf/-Inf are tagged strings"
    ),
    "hash_scope": (
        "schema, provenance, discrete probes, metric-to-tolerance map, tolerances"
    ),
}


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
    completed = run_process(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT,
        env=env,
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
        "net_elastic_work_J": float(trapezoid(force, cyclic)),
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
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("u_m", "P_N"))
            writer.writerow((0.0, 0.0))
            writer.writerow((1e-6, probes["elastic_single"]["load_N"]))
        with output.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        probes["csv_output"] = {
            "columns": ["u_m", "P_N"],
            "newline": "LF",
            "rows": len(rows),
            "origin_u_m": float(rows[0]["u_m"]),
            "origin_load_N": float(rows[0]["P_N"]),
            "last_u_m": float(rows[-1]["u_m"]),
            "last_load_N": float(rows[-1]["P_N"]),
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "provenance": PROVENANCE,
        "canonicalization": CANONICALIZATION,
        "environment": {
            "python": platform.python_version(),
            "dependencies": _dependency_versions(),
            "collected_tests": _collected_test_count(),
        },
        "probes": probes,
        "tolerances": NUMERIC_TOLERANCES,
    }
    payload["semantic_sha256"] = semantic_hash(payload)
    return payload


def _canonicalize(value):
    """Return locale-independent JSON data for the discrete semantic hash."""
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        records = [_canonicalize(item) for item in value]
        return sorted(records, key=_canonical_sort_key)
    if isinstance(value, float):
        if math.isnan(value):
            token = "NaN"
        elif math.isinf(value):
            token = "+Inf" if value > 0 else "-Inf"
        elif value == 0.0:
            token = "0"
        else:
            token = format(value, f".{CANONICAL_FLOAT_SIGNIFICANT_DIGITS}g")
        return {"$float": token}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise TypeError(f"Unsupported canonical value: {type(value).__name__}")


def _canonical_sort_key(value):
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"),
    )


def canonical_json_bytes(value) -> bytes:
    """Serialize canonical data with fixed UTF-8 encoding and LF newline."""
    text = json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (text + "\n").encode("utf-8")


def _discrete_probes(probes):
    discrete = {}
    for case, metrics in probes.items():
        discrete[case] = {
            metric: value
            for metric, value in metrics.items()
            if f"{case}.{metric}" not in NUMERIC_FIELDS
        }
    return discrete


def semantic_hash(manifest) -> str:
    """Hash discrete metadata and the versioned numerical comparison contract."""
    material = {
        "schema_version": manifest["schema_version"],
        "provenance": manifest["provenance"],
        "canonicalization": manifest["canonicalization"],
        "discrete_probes": _discrete_probes(manifest["probes"]),
        "numeric_fields": NUMERIC_FIELDS,
        "tolerances": manifest["tolerances"],
    }
    return hashlib.sha256(canonical_json_bytes(material)).hexdigest()


def serialize_manifest(manifest) -> str:
    """Return the human-readable artifact using deterministic UTF-8/LF JSON."""
    return json.dumps(
        manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ) + "\n"


def _probe_path(path: str) -> str:
    return path.removeprefix("probes.")


def _relative_difference(actual: float, expected: float, absolute: float) -> float:
    scale = max(abs(actual), abs(expected))
    if scale == 0.0:
        return 0.0 if absolute == 0.0 else math.inf
    return absolute / scale


def _compare(actual, expected, path="probes"):
    errors = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected mapping, got {actual!r}"]
        for key, value in expected.items():
            if key not in actual:
                errors.append(f"{path}.{key}: missing")
            else:
                errors.extend(_compare(actual[key], value, f"{path}.{key}"))
        for key in actual.keys() - expected.keys():
            errors.append(f"{path}.{key}: unexpected")
    elif _probe_path(path) in NUMERIC_FIELDS:
        tolerance_name = NUMERIC_FIELDS[_probe_path(path)]
        tolerance = NUMERIC_TOLERANCES[tolerance_name]
        try:
            actual_number = float(actual)
            expected_number = float(expected)
        except (TypeError, ValueError):
            errors.append(f"{path}: expected numeric {expected!r}, got {actual!r}")
            return errors
        absolute = abs(actual_number - expected_number)
        relative = _relative_difference(actual_number, expected_number, absolute)
        finite = math.isfinite(actual_number) and math.isfinite(expected_number)
        matches = finite and bool(np.isclose(
            actual_number,
            expected_number,
            rtol=tolerance["rtol"],
            atol=tolerance["atol"],
        ))
        if not matches:
            case, metric = _probe_path(path).split(".", 1)
            errors.append(
                f"case={case}, metric={metric}: expected={expected_number!r}, "
                f"actual={actual_number!r}, abs_diff={absolute:.17g}, "
                f"rel_diff={relative:.17g}, tolerance={tolerance_name}"
                f"(rtol={tolerance['rtol']:.17g}, atol={tolerance['atol']:.17g})"
            )
    elif actual != expected:
        errors.append(f"{path}: expected {expected!r}, got {actual!r}")
    return errors


def compare_manifests(actual, expected):
    """Compare runtime-independent contract fields and all probes."""
    errors = []
    for field in ("schema_version", "provenance", "canonicalization", "tolerances"):
        errors.extend(_compare(actual.get(field), expected.get(field), field))
    errors.extend(_compare(actual["probes"], expected["probes"]))
    expected_hash = expected.get("semantic_sha256")
    actual_hash = semantic_hash(actual)
    if actual_hash != expected_hash:
        errors.append(
            f"semantic_sha256: expected {expected_hash!r}, got {actual_hash!r}"
        )
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="print the generated manifest")
    args = parser.parse_args()
    manifest = build_manifest()
    if args.json:
        print(serialize_manifest(manifest), end="")
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    errors = compare_manifests(manifest, reference)
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
