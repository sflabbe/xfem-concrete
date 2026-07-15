"""CLI regression for bond-slip on t5a1 to avoid hangs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from tests.process_utils import run_process


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
        "--max-displacement",
        "0.1",
        "--bond-slip",
        "on",
        "--no-post",
        "--no-numba",
        "--output-dir",
        str(output_dir),
    ]

    result = run_process(
        cmd,
        cwd=repo_root,
        env=env,
        timeout=15,
    )

    assert result.returncode == 2
    combined = result.stdout + result.stderr
    assert "engine_compatibility" in combined
    assert "falls back to linear elastic assembly" in combined
    assert "Built 2 bond layer" not in combined
    assert "[substep]" not in combined
