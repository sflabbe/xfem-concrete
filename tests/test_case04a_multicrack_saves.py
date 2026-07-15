"""CLI regression for fail-closed Bosco T5A1 multicrack dispatch."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from tests.process_utils import run_process


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
        "--max-displacement",
        "0.1",
        "--bond-slip",
        "off",
        "--solver",
        "multi",
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
    assert "multicrack maps cdp_full to the CDP-lite Numba kernel" in combined
    assert "Built 2 bond layer" not in combined
    assert "[substep]" not in combined
    assert not (output_dir / "load_displacement.csv").exists()
