# Architecture

The maintained application is `examples/gutierrez_thesis/`; despite its path,
it is not disposable example code. Its Python factories are the canonical case
definitions and `solver_interface.py` is the adapter into the numerical kernel.

The numerical kernel remains split between the frozen single-crack path in
`xfem_clean.xfem.analysis_single` and the multicrack path in
`xfem_clean.xfem.multicrack`. This refactor does not merge or rewrite them.
Both are adapted at the public boundary to `xfem_clean.results.AnalysisResult`.

`AnalysisResult` schema version 1 exposes `steps`, `fields`,
`energies_or_work`, `cracks`, `reinforcement`, `solver_meta`, and structured
`warnings`. Large arrays remain referenced rather than copied. Historical dict
keys remain available through the Mapping adapter and `legacy_view` is
explicitly deprecated.

`xfem_beam.py` is frozen legacy. New code must not import it. `xfem_xfem.py`
remains the documented import shim for releases that used the old module path.
A future package move for `examples/gutierrez_thesis/` requires import shims and
is intentionally outside this phase.
