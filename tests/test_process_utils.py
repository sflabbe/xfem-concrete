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
_TIMEOUT_PAYLOAD = r"""
import json, os, pathlib, subprocess, sys, time
readiness = pathlib.Path(sys.argv[1])
marker, stdout_marker, stderr_marker = sys.argv[2:5]
volume = int(sys.argv[5])
with_grandchild = sys.argv[6] == "1"
grandchild = None
if with_grandchild:
    grandchild = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)", marker]
    )
if stdout_marker:
    sys.stdout.write(stdout_marker + "\n")
if volume:
    sys.stdout.write("O" * volume)
sys.stdout.flush()
if stderr_marker:
    sys.stderr.write(stderr_marker + "\n")
if volume:
    sys.stderr.write("E" * volume)
sys.stderr.flush()
temporary = readiness.with_suffix(".tmp")
temporary.write_text(json.dumps({
    "child_pid": os.getpid(),
    "grandchild_pid": grandchild.pid if grandchild else None,
}), encoding="utf-8")
temporary.replace(readiness)
time.sleep(120)
"""


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


def _timeout_capture(
    tmp_path: Path,
    *,
    stdout_marker: str = "",
    stderr_marker: str = "",
    volume: int = 0,
    with_grandchild: bool = False,
):
    marker = f"xfem-timeout-{uuid.uuid4()}"
    readiness = tmp_path / f"{marker}.json"
    command = [
        sys.executable,
        "-u",
        "-c",
        _TIMEOUT_PAYLOAD,
        str(readiness),
        marker,
        stdout_marker,
        stderr_marker,
        str(volume),
        "1" if with_grandchild else "0",
    ]
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        run_process(
            command,
            cwd=REPO_ROOT,
            timeout=0.4,
            terminate_grace=0.2,
        )
    assert readiness.exists(), "child never reached the independent readiness channel"
    return caught.value, json.loads(readiness.read_text(encoding="utf-8")), marker


def test_short_process_is_reaped():
    result = run_process(
        [
            sys.executable,
            "-c",
            "import sys; print('owned-ok-ñ'); print('owned-err-λ', file=sys.stderr)",
        ],
        cwd=REPO_ROOT,
        timeout=5,
    )

    assert result.returncode == 0
    assert result.stdout == "owned-ok-ñ\n"
    assert result.stderr == "owned-err-λ\n"
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


def test_timeout_preserves_stdout_and_output_alias(tmp_path: Path):
    exc, _, _ = _timeout_capture(tmp_path, stdout_marker="salida-ñ-λ")

    assert exc.stdout == "salida-ñ-λ\n"
    assert exc.output == exc.stdout
    assert exc.stderr == ""
    assert isinstance(exc.stdout, str)


def test_timeout_preserves_stderr(tmp_path: Path):
    exc, _, _ = _timeout_capture(tmp_path, stderr_marker="diagnóstico-stderr")

    assert exc.stdout == ""
    assert exc.stderr == "diagnóstico-stderr\n"
    assert isinstance(exc.stderr, str)


def test_timeout_preserves_both_streams_without_pipe_deadlock(tmp_path: Path):
    volume = 128 * 1024
    exc, _, _ = _timeout_capture(
        tmp_path,
        stdout_marker="stdout-distinct",
        stderr_marker="stderr-distinct",
        volume=volume,
    )

    assert exc.stdout == "stdout-distinct\n" + "O" * volume
    assert exc.stderr == "stderr-distinct\n" + "E" * volume


@pytest.mark.skipif(os.name != "posix", reason="strong process-group cleanup is POSIX-only")
def test_internal_timeout_kills_child_and_grandchild(tmp_path: Path):
    exc, pids, marker = _timeout_capture(
        tmp_path,
        stdout_marker="diagnostic-not-readiness",
        stderr_marker="grandchild-diagnostic",
        with_grandchild=True,
    )

    assert exc.stdout == "diagnostic-not-readiness\n"
    assert exc.stderr == "grandchild-diagnostic\n"
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
