"""CLI regression for Case 03 tensile STN12 (multicrack bond-slip stability)."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from math import isfinite
from pathlib import Path

import pytest


@pytest.mark.slow
@pytest.mark.case03
def test_case03_tensile_cli_multicrack(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "case03_tensile_cli"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    cmd = [
        sys.executable,
        "-m",
        "examples.gutierrez_thesis.run",
        "--case",
        "03_tensile_stn12",
        "--mesh",
        "coarse",
        "--nsteps",
        "5",
        "--no-post",
        "--no-numba",
        "--output-dir",
        str(output_dir),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
        env=env,
    )

    assert result.returncode == 0, (
        "CLI run failed.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert "Using default BondSlipModelCode2010" not in combined_output

    history_csv = output_dir / "load_displacement.csv"
    assert history_csv.exists(), f"Missing CSV output at {history_csv}"

    with history_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row]

    assert rows, "Expected load_displacement.csv to contain at least one row."
    for row in rows:
        for key, value in row.items():
            assert value is not None, f"Missing value for column {key}"
            parsed = float(value)
            assert isfinite(parsed), f"Non-finite value in {key}: {value}"
    last_row = rows[-1]
    p_kn = float(last_row.get("P_kN", 0.0))
    assert abs(p_kn) > 0.1, f"Expected non-zero reaction in last row, got P_kN={p_kn}"
