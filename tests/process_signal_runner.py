"""Standalone runner used to test cleanup when its own process is signalled."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tests.process_utils import run_process


CHILD_CODE = r"""
import json
import os
import pathlib
import subprocess
import sys
import time

readiness = pathlib.Path(sys.argv[1])
marker = sys.argv[2]
grandchild = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(120)", marker],
)
temporary = readiness.with_suffix(".tmp")
temporary.write_text(json.dumps({
    "runner_pid": os.getppid(),
    "child_pid": os.getpid(),
    "grandchild_pid": grandchild.pid,
    "marker": marker,
}), encoding="utf-8")
temporary.replace(readiness)
time.sleep(120)
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("readiness", type=Path)
    parser.add_argument("marker")
    args = parser.parse_args()
    run_process(
        [sys.executable, "-u", "-c", CHILD_CODE, str(args.readiness), args.marker],
        cwd=Path.cwd(),
        timeout=120,
        terminate_grace=0.2,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
