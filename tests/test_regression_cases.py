"""Regression guard for the compact canonical numerical manifest."""

from __future__ import annotations

import copy
import json
import subprocess
import warnings
from pathlib import Path

import pytest

import scripts.regression_manifest as regression_manifest
from scripts.regression_manifest import (
    CollectionError,
    REFERENCE,
    _compare,
    _collected_test_count,
    build_manifest,
    canonical_json_bytes,
    compare_manifests,
    main,
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
    assert isinstance(generated_manifest["environment"]["collected_tests"], int)
    assert generated_manifest["environment"]["collected_tests"] > 0


def test_collection_count_parses_successful_pytest_output(monkeypatch):
    completed = subprocess.CompletedProcess(
        ["pytest"], 0, "controlled node ids\n17 tests collected in 0.12s\n", "",
    )
    monkeypatch.setattr(regression_manifest, "run_process", lambda *args, **kwargs: completed)

    assert _collected_test_count() == 17


def test_collection_nonzero_exit_preserves_diagnostics(monkeypatch):
    completed = subprocess.CompletedProcess(
        ["pytest"], 3, "collection-stdout", "collection-stderr",
    )
    monkeypatch.setenv("PYTHONWARNINGS", "error::DeprecationWarning")
    monkeypatch.setattr(regression_manifest, "run_process", lambda *args, **kwargs: completed)

    with pytest.raises(CollectionError) as caught:
        _collected_test_count()

    message = str(caught.value)
    assert "pytest collection failed" in message
    assert "returncode: 3" in message
    assert "collection-stdout" in message
    assert "collection-stderr" in message
    assert "PYTHONWARNINGS: error::DeprecationWarning" in message
    assert f"command: {regression_manifest.sys.executable} -m pytest --collect-only -q" in message


@pytest.mark.parametrize(
    ("stdout", "reason"),
    [
        ("collection finished without a summary", "no parseable count"),
        ("0 tests collected in 0.01s\n", "unexpectedly found zero tests"),
    ],
)
def test_collection_rejects_untrustworthy_count(monkeypatch, stdout, reason):
    completed = subprocess.CompletedProcess(["pytest"], 0, stdout, "")
    monkeypatch.setattr(regression_manifest, "run_process", lambda *args, **kwargs: completed)

    with pytest.raises(CollectionError, match=reason):
        _collected_test_count()


def test_collection_warning_promoted_to_error_is_not_swallowed(monkeypatch):
    completed = subprocess.CompletedProcess(
        ["pytest"],
        1,
        "",
        "DeprecationWarning: fatal collection warning",
    )
    monkeypatch.setenv("PYTHONWARNINGS", "error::DeprecationWarning")
    monkeypatch.setattr(regression_manifest, "run_process", lambda *args, **kwargs: completed)

    with pytest.raises(CollectionError, match="fatal collection warning"):
        _collected_test_count()


def test_cli_collection_failure_is_nonzero_and_never_prints_ok(monkeypatch, capsys):
    def fail_collection():
        raise CollectionError("returncode: 4\nstdout:\nbad-out\nstderr:\nbad-err")

    monkeypatch.setattr(regression_manifest, "build_manifest", fail_collection)

    assert main([]) == 2
    captured = capsys.readouterr()
    assert "manifest OK" not in captured.out + captured.err
    assert "bad-out" in captured.err
    assert "bad-err" in captured.err


def test_cli_success_requires_integer_count(
    monkeypatch, capsys, tmp_path, reference_manifest,
):
    manifest = copy.deepcopy(reference_manifest)
    manifest["environment"] = {"collected_tests": 17}
    reference = tmp_path / "reference.json"
    reference.write_text(serialize_manifest(reference_manifest), encoding="utf-8")
    monkeypatch.setattr(regression_manifest, "build_manifest", lambda: manifest)
    monkeypatch.setattr(regression_manifest, "REFERENCE", reference)

    assert main([]) == 0
    captured = capsys.readouterr()
    assert "manifest OK: 8 probes, 17 tests collected" in captured.out


def test_cli_comparison_failure_is_nonzero(
    monkeypatch, capsys, tmp_path, reference_manifest,
):
    manifest = copy.deepcopy(reference_manifest)
    manifest["environment"] = {"collected_tests": 17}
    manifest["probes"]["elastic_single"]["load_N"] *= 2.0
    reference = tmp_path / "reference.json"
    reference.write_text(serialize_manifest(reference_manifest), encoding="utf-8")
    monkeypatch.setattr(regression_manifest, "build_manifest", lambda: manifest)
    monkeypatch.setattr(regression_manifest, "REFERENCE", reference)

    assert main([]) == 1
    captured = capsys.readouterr()
    assert "manifest OK" not in captured.out + captured.err
    assert "Regression manifest mismatch" in captured.err


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
