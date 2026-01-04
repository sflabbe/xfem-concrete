"""CLI regression for Case 01 pullout (bond-slip mapping + nonzero load)."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_case01_pullout_cli_no_nans(tmp_path: Path) -> None:
    output_dir = tmp_path / "case01_pullout_cli"
    cmd = [
        sys.executable,
        "-m",
        "examples.gutierrez_thesis.run",
        "--case",
        "01_pullout_lettow",
        "--mesh",
        "coarse",
        "--nsteps",
        "3",
        "--no-post",
        "--no-numba",
        "--output-dir",
        str(output_dir),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    combined = (result.stdout or "") + (result.stderr or "")

    assert result.returncode == 0, (
        "CLI run failed.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    assert "invalid value encountered in matmul" not in combined
    assert "overflow encountered" not in combined
    assert "Invalid steel DOF mapping" not in combined

    history_csv = output_dir / "load_displacement.csv"
    assert history_csv.exists(), f"Missing CSV output at {history_csv}"

    with history_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        p_values = []
        for row in reader:
            if not row:
                continue
            p_values.append(float(row.get("P_kN", 0.0)))

    assert any(abs(p) > 1e-3 for p in p_values), (
        "Expected non-zero reaction force in at least one step; "
        f"got P_kN={p_values}"
    )
