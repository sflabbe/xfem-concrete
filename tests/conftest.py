"""
Pytest configuration for xfem-concrete tests.

Automatically adds src/ and repo root to sys.path so tests can import
xfem_clean and examples namespace without PYTHONPATH.

Implements fast/slow test split:
- Default (pytest -q): runs FAST suite only (unit + lightweight integration)
- With --runslow (pytest -q --runslow): runs EVERYTHING (including heavy solver analyses)
"""

# Early diagnostic guard: check for numpy before any imports that depend on it
try:
    import numpy as np
except Exception as e:
    import sys
    raise RuntimeError(
        "numpy is required to run tests but is not importable.\n"
        f"Python executable: {sys.executable}\n"
        "You are likely running pytest with the wrong interpreter.\n"
        "Fix:\n"
        "  1) Install deps: pip install -e '.[test]'\n"
        "  2) Run tests via: python -m pytest -q -m 'not slow'\n"
        "  or use the wrapper: python scripts/run_tests.py\n"
    ) from e

import sys
import os
import pytest

# Add repo root and src/ to path for module imports
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(repo_root, 'src')

# Add repo root first so 'examples' namespace is available
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Then add src/ for xfem_clean imports
if src_path not in sys.path:
    sys.path.insert(0, src_path)


def pytest_addoption(parser):
    """Add --runslow option to pytest command line."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests (full solver analyses, heavy integration tests)"
    )


def pytest_configure(config):
    """Register the 'slow' marker."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with -m 'not slow')"
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --runslow is specified."""
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# PHASE 1: Per-test telemetry for slow-suite performance debugging
# PHASE 3: Add matplotlib cleanup to prevent figure accumulation
import time
import gc
import resource


@pytest.fixture(autouse=True)
def _perf_guard(request):
    """
    Lightweight per-test telemetry: timing + RSS + GC cleanup.

    Also ensures matplotlib figures are closed to prevent accumulation
    in slow test suite (Fix #15).
    """
    t0 = time.perf_counter()
    rss_before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    yield

    dt = time.perf_counter() - t0
    rss_after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KB on Linux, bytes on macOS

    # Only report if test is marked slow or took >0.5s
    if "slow" in request.keywords or dt > 0.5:
        print(f"\n[PERF] {request.node.nodeid}")
        print(f"       dt={dt:.2f}s  rss_before={rss_before_kb/1024:.1f}MB  rss_after={rss_after_kb/1024:.1f}MB")

    # CRITICAL FIX #15: Close all matplotlib figures to prevent accumulation
    # Postprocessing creates plots; without cleanup, figures accumulate and
    # cause catastrophic slowdown (140x) in test suite execution.
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except (ImportError, Exception):
        # matplotlib not installed or other error - safe to ignore
        pass

    # Force GC to prevent accumulation
    gc.collect()
