"""Regression guard for the compact canonical numerical manifest."""

from __future__ import annotations

import json

from scripts.regression_manifest import REFERENCE, _compare, build_manifest


def test_canonical_regression_manifest():
    manifest = build_manifest()
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))

    assert _compare(manifest["probes"], reference["probes"]) == []
    assert manifest["semantic_sha256"] == reference["semantic_sha256"]
