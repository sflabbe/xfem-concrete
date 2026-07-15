# Case configuration

The maintained configuration source is the set of Python `CaseConfig` factories
in `../cases/`, registered by `../run.py`. Files under `legacy/` are historical
inputs and are not supported entrypoints.

JSON/YAML import remains available for files that conform to schema version 1.
Imported files pass through `normalize_case_config()` and fail explicitly on
unknown materials, engines, loading protocols, or law types.
