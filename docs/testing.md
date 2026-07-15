# Testing

The default fast suite excludes tests marked `slow`; `--runslow` enables the
whole collection. Always apply an external timeout to the full suite:

```bash
python3 -m compileall -q src examples tests
python3 -m pytest --collect-only -q
timeout --signal=TERM --kill-after=15s 900s python3 -m pytest -m 'not slow' -q --durations=30
timeout --signal=TERM --kill-after=15s 1200s python3 -m pytest --runslow -q --durations=40
```

CLI tests use `tests.process_utils.run_process()`. On POSIX it registers each
child immediately in its own process group. Normal return, timeout, `SIGINT`,
`SIGTERM`, and `atexit` all follow the same TERM/grace/KILL/reap lifecycle.
Signal cleanup then delegates to pytest's prior handler, preserving Ctrl+C and
CI exit status. Windows uses a new process group but does not claim the same
descendant-tree guarantee. Do not call `subprocess.run` or `Popen` directly
from tests; deliberately unmanaged descendants exist only inside the isolated
process-lifecycle test payload.

The regression manifest uses schema 2. Floating probes are compared directly
with the named `rtol`/`atol` entries in the artifact. Its SHA-256 is computed
from canonical JSON containing schema, provenance, discrete probe metadata,
and the metric-to-tolerance contract; numerical probe values are intentionally
excluded from the hash. Canonical JSON uses UTF-8, LF, sorted fields/records,
15 significant digits for contract floats, tagged non-finite values, and
normalizes `-0.0` to `0.0`.

Plot-producing code closes its own figures. The suite deliberately has no
autouse matplotlib import and no global `gc.collect()` teardown.
