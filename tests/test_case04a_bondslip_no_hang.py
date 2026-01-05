"""CLI regression for bond-slip on t5a1 to avoid hangs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_case04a_bondslip_no_hang(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "case04a_bondslip_no_hang"
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
        "1",
        "--bond-slip",
        "on",
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
        timeout=120,
    )

    assert result.returncode == 0, (
        "CLI run failed.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
