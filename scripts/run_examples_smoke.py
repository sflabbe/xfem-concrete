"""Smoke runner for examples and thesis cases."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_command(args: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    command = [sys.executable, *args]
    print(f"\n→ Running: {' '.join(command)}")
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )


def main() -> None:
    run_command([
        "-m",
        "examples.gutierrez_thesis.run",
        "--case",
        "all",
        "--mesh",
        "coarse",
        "--nsteps",
        "1",
        "--no-post",
        "--no-numba",
    ])

    run_command(["examples/pullout_minimal_working.py"])
    run_command(["examples/pullout_correct_bcs.py"])
    run_command(["examples/diagnose_stiffness_scale.py"])


if __name__ == "__main__":
    main()
