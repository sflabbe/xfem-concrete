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
It covers reduced single/multi solvers, bond, FRP, fibres, cyclic work, schema,
and CSV output. Case 09 results created before the 2026-07-15 density correction
are invalid because the density was multiplied by 10,000 twice.
