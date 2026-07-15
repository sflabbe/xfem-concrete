"""Subprocess helpers that cannot leave child process groups behind."""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from typing import Mapping, Sequence


def run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: float,
    env: Mapping[str, str] | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr) from exc

    completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if check and process.returncode:
        raise subprocess.CalledProcessError(
            process.returncode, command, output=stdout, stderr=stderr,
        )
    return completed
