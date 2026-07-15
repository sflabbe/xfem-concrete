# Datasets

| Path | Class | Provenance status | Supported use |
|---|---|---|---|
| `validation/reference_data/*.csv` | Synthetic | Placeholder, documented in `SOURCES.md` | Pipeline regression only |
| `tests/regression/canonical_manifest.json` | Golden | Generated from documented reduced probes | Numerical regression |
| `examples/gutierrez_thesis/configs/legacy/*.yml` | Historical | Divergent or invalid schema | Archive only |

Experimental data must be labeled `REAL` only after source, figure/page, units,
digitization method, and review date are recorded. Derived datasets must name
their inputs and transformation. Golden files must name the code/environment
used to create them and their tolerance policy.
