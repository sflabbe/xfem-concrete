"""Lifecycle tests for every subprocess owned by the pytest harness."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

from tests.process_utils import (
    cleanup_processes,
    registered_process_count,
    run_process,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    status = Path(f"/proc/{pid}/status")
    if status.exists() and "\nState:\tZ" in status.read_text(errors="replace"):
        return False
    return True


def _wait_until(predicate, *, timeout: float, description: str) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail(f"Timed out waiting for {description}")
        time.sleep(0.01)


def _marker_pids(marker: str) -> list[int]:
    matches = []
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if marker in command:
            matches.append(int(cmdline.parent.name))
    return matches


def test_short_process_is_reaped():
    result = run_process(
        [sys.executable, "-c", "print('owned-ok')"],
        cwd=REPO_ROOT,
        timeout=5,
    )

    assert result.returncode == 0
    assert result.stdout == "owned-ok\n"
    assert result.stderr == ""
    assert registered_process_count() == 0


def test_failed_process_preserves_return_code_and_diagnostics():
    command = [
        sys.executable,
        "-c",
        "import sys; print('failure-out'); print('failure-err', file=sys.stderr); sys.exit(7)",
    ]

    with pytest.raises(subprocess.CalledProcessError) as caught:
        run_process(command, cwd=REPO_ROOT, timeout=5, check=True)

    assert caught.value.returncode == 7
    assert caught.value.cmd == command
    assert caught.value.stdout == "failure-out\n"
    assert caught.value.stderr == "failure-err\n"
    assert registered_process_count() == 0


@pytest.mark.skipif(os.name != "posix", reason="strong process-group cleanup is POSIX-only")
def test_internal_timeout_kills_child_and_grandchild():
    marker = f"xfem-timeout-{uuid.uuid4()}"
    payload = r"""
import json, os, subprocess, sys, time
grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", sys.argv[1]])
print(json.dumps({"child_pid": os.getpid(), "grandchild_pid": grandchild.pid}), flush=True)
time.sleep(120)
"""

    with pytest.raises(subprocess.TimeoutExpired) as caught:
        run_process(
            [sys.executable, "-u", "-c", payload, marker],
            cwd=REPO_ROOT,
            timeout=0.25,
            terminate_grace=0.2,
        )

    pids = json.loads(caught.value.stdout.strip())
    _wait_until(
        lambda: not _pid_exists(pids["child_pid"])
        and not _pid_exists(pids["grandchild_pid"]),
        timeout=3,
        description="timeout descendants to disappear",
    )
    assert _marker_pids(marker) == []
    assert registered_process_count() == 0


def test_cleanup_is_idempotent():
    cleanup_processes(terminate_grace=0.05)
    cleanup_processes(terminate_grace=0.05)
    assert registered_process_count() == 0


@pytest.mark.skipif(os.name != "posix", reason="signal integration is POSIX-only")
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT], ids=["sigterm", "sigint"])
def test_outer_runner_signal_kills_child_group(tmp_path: Path, signum: int):
    marker = f"xfem-signal-{signum}-{uuid.uuid4()}"
    readiness = tmp_path / "ready.json"
    command = [
        sys.executable,
        "-m",
        "tests.process_signal_runner",
        str(readiness),
        marker,
    ]
    outcome: dict[str, object] = {}

    def invoke_runner() -> None:
        try:
            outcome["result"] = run_process(
                command,
                cwd=REPO_ROOT,
                timeout=15,
                terminate_grace=0.2,
            )
        except BaseException as exc:  # surfaced in the main test thread below
            outcome["error"] = exc

    thread = threading.Thread(target=invoke_runner, daemon=True)
    thread.start()
    _wait_until(readiness.exists, timeout=5, description="runner readiness")
    pids = json.loads(readiness.read_text(encoding="utf-8"))

    os.kill(pids["runner_pid"], signum)
    thread.join(timeout=8)
    assert not thread.is_alive(), "outer runner did not preserve signal termination"
    assert "error" not in outcome, repr(outcome.get("error"))
    result = outcome["result"]
    assert isinstance(result, subprocess.CompletedProcess)
    assert result.returncode == -signum

    _wait_until(
        lambda: not _pid_exists(pids["child_pid"])
        and not _pid_exists(pids["grandchild_pid"]),
        timeout=3,
        description=f"descendants after signal {signum}",
    )
    assert _marker_pids(marker) == []
    assert registered_process_count() == 0
