"""Owned subprocess helpers for the test harness.

POSIX children run in dedicated process groups. The registry is installed by
``tests.conftest`` before tests run, and ``run_process`` also installs it when
used by a standalone helper. This guarantees cleanup on normal return, timeout,
SIGINT, SIGTERM, and interpreter shutdown.
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import BinaryIO, Mapping, Sequence


DEFAULT_TERMINATE_GRACE = 1.0
OUTPUT_ENCODING = "utf-8"
OUTPUT_ERRORS = "strict"
_CLEANUP_SIGNALS = tuple(
    signum for signum in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None))
    if signum is not None
)


@dataclass(frozen=True)
class _OwnedProcess:
    process: subprocess.Popen[bytes]
    process_group: int
    command: tuple[str, ...]


class _ProcessRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._processes: dict[int, _OwnedProcess] = {}
        self._signal_handlers_installed = False
        self._previous_handlers: dict[int, object] = {}

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def register_locked(self, owned: _OwnedProcess) -> None:
        self._processes[owned.process.pid] = owned

    def unregister(self, owned: _OwnedProcess) -> None:
        if owned.process.poll() is None:
            raise RuntimeError(f"Cannot unregister live process {owned.process.pid}")
        with self._lock:
            self._processes.pop(owned.process.pid, None)

    def count(self) -> int:
        with self._lock:
            return len(self._processes)

    def install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        with self._lock:
            if self._signal_handlers_installed:
                return
            for signum in _CLEANUP_SIGNALS:
                self._previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle_signal)
            self._signal_handlers_installed = True

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        self.cleanup()
        previous = self._previous_handlers.get(signum, signal.SIG_DFL)
        signal.signal(signum, previous)
        if previous == signal.SIG_IGN:
            return
        if callable(previous):
            previous(signum, frame)
            return
        os.kill(os.getpid(), signum)

    def cleanup(self, terminate_grace: float = DEFAULT_TERMINATE_GRACE) -> None:
        with self._lock:
            owned_processes = list(self._processes.values())
        for owned in owned_processes:
            _terminate_owned(owned, terminate_grace=terminate_grace)
            if owned.process.poll() is not None:
                with self._lock:
                    self._processes.pop(owned.process.pid, None)


_REGISTRY = _ProcessRegistry()
atexit.register(_REGISTRY.cleanup)


def install_process_cleanup() -> None:
    """Install SIGINT/SIGTERM cleanup while preserving the runner's handlers."""
    _REGISTRY.install_signal_handlers()


def cleanup_processes(*, terminate_grace: float = DEFAULT_TERMINATE_GRACE) -> None:
    """Idempotently terminate and reap every process owned by this interpreter."""
    _REGISTRY.cleanup(terminate_grace=terminate_grace)


def registered_process_count() -> int:
    return _REGISTRY.count()


def _group_alive(process_group: int) -> bool:
    if os.name != "posix":
        return False
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_owned(owned: _OwnedProcess, signum: int) -> None:
    try:
        if os.name == "posix":
            os.killpg(owned.process_group, signum)
        elif owned.process.poll() is None:
            owned.process.send_signal(signum)
    except ProcessLookupError:
        pass


def _wait_direct_child(process: subprocess.Popen[bytes], timeout: float) -> None:
    try:
        process.wait(timeout=max(0.0, timeout))
    except subprocess.TimeoutExpired:
        pass


def _wait_group_gone(process_group: int, timeout: float) -> bool:
    if os.name != "posix":
        return True
    deadline = time.monotonic() + max(0.0, timeout)
    while _group_alive(process_group):
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
    return True


def _terminate_owned(
    owned: _OwnedProcess,
    *,
    terminate_grace: float = DEFAULT_TERMINATE_GRACE,
) -> None:
    """TERM, then KILL if needed, and reap the direct child."""
    process = owned.process
    direct_alive = process.poll() is None
    group_alive = _group_alive(owned.process_group) if os.name == "posix" else direct_alive
    if not direct_alive and not group_alive:
        return

    _signal_owned(owned, signal.SIGTERM)
    _wait_direct_child(process, terminate_grace)
    group_gone = _wait_group_gone(owned.process_group, terminate_grace)

    direct_alive = process.poll() is None
    if direct_alive or not group_gone:
        kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
        _signal_owned(owned, kill_signal)
        _wait_direct_child(process, terminate_grace)
        _wait_group_gone(owned.process_group, terminate_grace)

    if process.poll() is None:
        process.wait()


def _spawn_owned(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None,
    stdout: BinaryIO,
    stderr: BinaryIO,
) -> _OwnedProcess:
    install_process_cleanup()
    popen_kwargs: dict[str, object] = {}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    elif os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    old_mask = None
    if os.name == "posix" and threading.current_thread() is threading.main_thread():
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, _CLEANUP_SIGNALS)
    try:
        with _REGISTRY.lock:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=stdout,
                stderr=stderr,
                **popen_kwargs,
            )
            owned = _OwnedProcess(
                process=process,
                process_group=process.pid,
                command=tuple(os.fspath(part) for part in command),
            )
            _REGISTRY.register_locked(owned)
    finally:
        if old_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
    return owned


def _read_capture(stream: BinaryIO) -> str:
    stream.seek(0)
    return stream.read().decode(OUTPUT_ENCODING, errors=OUTPUT_ERRORS)


def run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: float,
    env: Mapping[str, str] | None = None,
    check: bool = False,
    terminate_grace: float = DEFAULT_TERMINATE_GRACE,
) -> subprocess.CompletedProcess[str]:
    """Run an owned command and capture UTF-8 diagnostics from its complete group."""
    with tempfile.TemporaryFile(mode="w+b") as stdout_capture, tempfile.TemporaryFile(
        mode="w+b"
    ) as stderr_capture:
        owned = _spawn_owned(
            command,
            cwd=cwd,
            env=env,
            stdout=stdout_capture,
            stderr=stderr_capture,
        )
        process = owned.process
        try:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                _terminate_owned(owned, terminate_grace=terminate_grace)
                stdout = _read_capture(stdout_capture)
                stderr = _read_capture(stderr_capture)
                raise subprocess.TimeoutExpired(
                    command,
                    timeout,
                    output=stdout,
                    stderr=stderr,
                ) from exc

            # A command that exits while leaving descendants behind still owns them.
            if os.name == "posix" and _group_alive(owned.process_group):
                _terminate_owned(owned, terminate_grace=terminate_grace)

            stdout = _read_capture(stdout_capture)
            stderr = _read_capture(stderr_capture)
            completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            if check and process.returncode:
                raise subprocess.CalledProcessError(
                    process.returncode, command, output=stdout, stderr=stderr,
                )
            return completed
        except BaseException:
            _terminate_owned(owned, terminate_grace=terminate_grace)
            raise
        finally:
            if process.poll() is not None:
                _REGISTRY.unregister(owned)
