# Testing

The default fast suite excludes tests marked `slow`; `--runslow` enables the
whole collection. Always apply an external timeout to the full suite:

```bash
python3 -m compileall -q src examples tests
python3 -m pytest --collect-only -q
timeout --signal=TERM --kill-after=15s 900s python3 -m pytest -m 'not slow' -q --durations=30
timeout --signal=TERM --kill-after=15s 1200s python3 -m pytest --runslow -q --durations=40
```

CLI tests use `tests.process_utils.run_process()`. It starts a new session and,
on timeout, terminates and reaps the complete process group. Do not call
`subprocess.run` or `Popen` directly from tests.

Plot-producing code closes its own figures. The suite deliberately has no
autouse matplotlib import and no global `gc.collect()` teardown.
