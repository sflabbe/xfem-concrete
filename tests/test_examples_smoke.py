import os
import sys
from pathlib import Path

import pytest
from tests.process_utils import run_process


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_command(args):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    command = [sys.executable, *args]
    return run_process(
        command,
        cwd=REPO_ROOT,
        env=env,
        check=True,
        timeout=300,
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
        "--dry-run",
    ])

    run_command(["examples/pullout_minimal_working.py"])
    run_command(["examples/pullout_correct_bcs.py"])
    run_command(["examples/diagnose_stiffness_scale.py"])


@pytest.mark.slow
@pytest.mark.examples_smoke
def test_examples_dry_run_overrides():
    result = run_command([
        "-m",
        "examples.gutierrez_thesis.run",
        "--case",
        "03_tensile_stn12",
        "--dry-run",
        "--bulk",
        "elastic",
        "--solver",
        "multi",
    ])
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert "Active overrides:" in combined_output
    assert "concrete.model_type = elastic" in combined_output
    assert "solver_engine = multi" in combined_output
