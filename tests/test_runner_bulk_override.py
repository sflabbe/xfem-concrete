import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_runner(args):
    command = [sys.executable, "-m", "examples.gutierrez_thesis.run", *args]
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def test_bulk_override_reflected_in_dry_run():
    result = run_runner([
        "--case",
        "t5a1",
        "--mesh",
        "coarse",
        "--bulk",
        "elastic",
        "--dry-run",
    ])

    assert "Override: concrete.model_type = elastic" in result.stdout


def test_bulk_default_no_override_line():
    result = run_runner([
        "--case",
        "t5a1",
        "--mesh",
        "coarse",
        "--dry-run",
    ])

    assert "Override: concrete.model_type" not in result.stdout
