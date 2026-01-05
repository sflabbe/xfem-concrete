import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_command(args):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    command = [sys.executable, *args]
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )


@pytest.mark.slow
@pytest.mark.examples_smoke
def test_examples_smoke():
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
