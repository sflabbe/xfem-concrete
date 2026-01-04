#!/usr/bin/env python
"""
Deterministic test runner for xfem-concrete.

Ensures pytest runs with the correct Python interpreter (the one running this script)
by using `python -m pytest` instead of calling pytest executable directly.

This prevents environment isolation issues where pytest entrypoint resolves to a
tool environment interpreter that doesn't have project dependencies installed.

Usage:
    python scripts/run_tests.py              # Run fast suite (default)
    python scripts/run_tests.py --runslow    # Run full suite including slow tests
    python scripts/run_tests.py -x -vv       # Pass debugging flags to pytest
"""

import sys
import subprocess


def main():
    """Run pytest using the current Python interpreter."""
    # Default arguments for fast suite
    default_args = ["-q", "-m", "not slow"]

    # Get any CLI arguments passed to this script
    passthrough_args = sys.argv[1:]

    # If user provides args, use those; otherwise use defaults
    pytest_args = passthrough_args if passthrough_args else default_args

    # Build command: use current interpreter to run pytest module
    cmd = [sys.executable, "-m", "pytest"] + pytest_args

    # Print command for transparency
    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    print(f"Python executable: {sys.executable}", file=sys.stderr)
    print("", file=sys.stderr)

    # Run pytest and return its exit code
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
