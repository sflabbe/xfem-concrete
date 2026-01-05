"""CLI regression for Bosco t5a1 multicrack saves without crashing."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_case04a_multicrack_saves(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "case04a_multicrack_saves"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    cmd = [
        sys.executable,
        "-m",
        "examples.gutierrez_thesis.run",
        "--case",
        "t5a1",
        "--mesh",
        "coarse",
        "--nsteps",
        "2",
        "--bond-slip",
        "off",
        "--solver",
        "multi",
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

    history_csv = output_dir / "load_displacement.csv"
    assert history_csv.exists(), f"Missing CSV output at {history_csv}"
