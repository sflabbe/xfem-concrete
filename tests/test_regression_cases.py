"""Regression guard for the compact canonical numerical manifest."""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path

import pytest

from scripts.regression_manifest import (
    REFERENCE,
    _compare,
    build_manifest,
    canonical_json_bytes,
    compare_manifests,
    semantic_hash,
    serialize_manifest,
)


@pytest.fixture(scope="module")
def reference_manifest():
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def generated_manifest():
    repo_root = Path(__file__).resolve().parents[1]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        manifest = build_manifest()
    own_deprecations = [
        warning
        for warning in caught
        if issubclass(warning.category, DeprecationWarning)
        and Path(warning.filename).resolve().is_relative_to(repo_root)
    ]
    assert own_deprecations == []
    return manifest


def test_canonical_regression_manifest(generated_manifest, reference_manifest):
    assert compare_manifests(generated_manifest, reference_manifest) == []


def test_machine_roundoff_does_not_break_regression(reference_manifest):
    actual = copy.deepcopy(reference_manifest)
    actual["probes"]["elastic_single"]["load_N"] += 1.2e-11
    actual["probes"]["multicrack"]["load_N"] += 1.8e-12

    assert _compare(actual["probes"], reference_manifest["probes"]) == []
    assert semantic_hash(actual) == reference_manifest["semantic_sha256"]


def test_difference_over_tolerance_has_actionable_error(reference_manifest):
    actual = copy.deepcopy(reference_manifest)
    expected = reference_manifest["probes"]["elastic_single"]["load_N"]
    actual["probes"]["elastic_single"]["load_N"] = expected * (1.0 + 2e-8)

    errors = _compare(actual["probes"], reference_manifest["probes"])

    assert len(errors) == 1
    assert "case=elastic_single, metric=load_N" in errors[0]
    assert "expected=" in errors[0]
    assert "actual=" in errors[0]
    assert "abs_diff=" in errors[0]
    assert "rel_diff=" in errors[0]
    assert "tolerance=solver_load(rtol=1e-08, atol=9.9999999999999998e-13)" in errors[0]


def test_discrete_metadata_change_fails_and_changes_hash(reference_manifest):
    actual = copy.deepcopy(reference_manifest)
    actual["probes"]["configuration"]["material"] = "different-material"

    errors = _compare(actual["probes"], reference_manifest["probes"])

    assert errors == [
        "probes.configuration.material: expected 'cdp_lite', got 'different-material'"
    ]
    assert semantic_hash(actual) != reference_manifest["semantic_sha256"]


def test_canonical_json_ignores_mapping_and_record_order():
    first = {"records": [{"case": "b", "value": 2}, {"case": "a", "value": 1}]}
    second = {"records": [{"value": 1, "case": "a"}, {"value": 2, "case": "b"}]}

    assert canonical_json_bytes(first) == canonical_json_bytes(second)


def test_canonical_float_policy_is_explicit_and_stable():
    assert canonical_json_bytes({"value": -0.0}) == canonical_json_bytes({"value": 0.0})
    encoded = canonical_json_bytes(
        {"nan": float("nan"), "positive": float("inf"), "negative": -float("inf")}
    )
    assert encoded.endswith(b"\n")
    assert b'"NaN"' in encoded
    assert b'"+Inf"' in encoded
    assert b'"-Inf"' in encoded


def test_generated_artifact_roundtrips_semantically(
    generated_manifest, reference_manifest, tmp_path,
):
    artifact = tmp_path / "manifest.json"
    artifact.write_text(serialize_manifest(generated_manifest), encoding="utf-8", newline="\n")

    parsed = json.loads(artifact.read_text(encoding="utf-8"))

    assert compare_manifests(parsed, reference_manifest) == []
    assert artifact.read_bytes().endswith(b"\n")
