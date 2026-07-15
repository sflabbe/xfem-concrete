# Validation

The files `validation/reference_data/t5a1.csv`, `vvbs3.csv`, and `sorelli.csv`
are synthetic placeholders. They exercise curve loading, metric calculation,
CSV generation, and regression plumbing. They do not establish experimental
validation and must not be cited as such.

Replacing one requires the exact publication/thesis version, figure and page,
digitization method, date, units, raw artifact provenance, and an explicit
review of peak, energy, and curve-shape tolerances. Until then, tests using
these files are pipeline regressions.

The compact numerical golden is `tests/regression/canonical_manifest.json`.
Schema 2 separates tolerant numerical probes from exact discrete metadata.
Numerical results are checked with named absolute and relative tolerances; the
canonical SHA-256 covers only exact metadata and that comparison contract, so
machine-roundoff differences cannot conflict with the semantic comparison.
It covers reduced single/multi solvers, bond, FRP, fibres, cyclic work, schema,
and CSV output. Case 09 results created before the 2026-07-15 density correction
are invalid because the density was multiplied by 10,000 twice.
