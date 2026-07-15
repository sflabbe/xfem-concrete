# Refactor Report

Date: 2026-07-15

## Scope And Evidence

`XFEM_AUDITORIA_INTEGRAL_2026-07-15.md` was not present in the checkout, any
parent directory searched, or the visible Git history. Every requested finding
was therefore verified directly against code. The initial worktree already had
six modified files; those changes were preserved and are not attributed to this
refactor:

- `examples/gutierrez_thesis/postprocess_comprehensive.py`
- `src/xfem_clean/compression_damage.py`
- `src/xfem_clean/xfem/assembly_single.py`
- `src/xfem_clean/xfem/dofs_single.py`
- `src/xfem_clean/xfem/geometry.py`
- `src/xfem_clean/xfem/material_factory.py`

No Newton iteration, FEM/XFEM assembly equation, constitutive law, cohesive
law, bond-slip law, crack propagation rule, physical factor, tolerance, sign,
or convergence criterion was deliberately changed.

## Initial Baseline

| Item | Initial result |
|---|---|
| Git | `main` at `63dc6dd`, six modified files above |
| `python` | Missing from PATH |
| `.venv/bin/python` | 3.13.5; numerical packages installed, pytest missing |
| `python3` | 3.11.2; pytest 7.2.1 |
| Numerical stack | NumPy 1.24.2, SciPy 1.10.1, Matplotlib 3.6.3, Numba 0.56.4 |
| Compile | Failed: two `IndentationError` debug examples |
| Collection | 237 tests, two dependency warnings |
| Fast suite | 183 passed, 1 failed, 53 deselected, 13 warnings, 85.25 s |
| Fast wall/RSS | 87.50 s; 270040 KiB maximum RSS |
| Fast failure | `benchmark_scaling.py` called NumPy-2-only `np.trapezoid` |
| Slow suite | External timeout at 1200 s, exit 124, 262032 KiB RSS; incomplete near 30% |
| Orphan check | No pytest or thesis runner remained after timeout |

The initial slow run had already printed three failures before timeout. Its
quiet output did not expose their node IDs before termination.

## Changes By Phase

### Baseline And Regression

`scripts/regression_manifest.py` runs eight compact probes with one command and
compares them to `tests/regression/canonical_manifest.json`. It records Python,
the numerical dependency versions, collected-test count, engine, normalized
material, bond-layer count, fibre seed, result schema, semantic values, CSV
hash, explicit tolerances, and a semantic SHA-256.

The probes cover reduced elastic single crack, multicrack, pullout/bond, FRP,
fibres, a five-point cyclic path, result/config provenance, and CSV creation.

### Fibre P0

- `case_09_beam_4pb_fibres_sorelli.py:create_case_09()` now stores `3.43`
  fibres/cm2, not `34300` in that field.
- `fibre_config_from_case()` still performs the single canonical `x1e4`
  conversion and now receives exactly `34300 fibres/m2`.
- `FibreConfig.explicit_fraction` transports the case's declared `0.5` instead
  of the adapter hard-code `0.1`.
- `test_case09_fibre_density_boundary_and_sampling` guards input, normalized
  value, generator config, seed 42, explicit count rule, and determinism.
- All previous case-09 fibre results are declared invalid in the golden and
  validation documentation.

### Work Semantics

Trapezoidal interface accumulators are now `interface_work_inc`,
`bond_work_inc`, and `dowel_work_inc`. Historical `D_coh_inc`, `D_bond_inc`,
and `D_dowel_inc` are compatibility aliases documented as deprecated. Plastic
and damage energies derived from internal variables retain dissipation names.
An elastic closed-cycle regression has net work tolerance `1e-15 J`.

### Configuration And Fallbacks

`normalize_case_config()` is the sole case normalization boundary. It returns a
validated copy, normalizes material/engine aliases, and raises
`CaseConfigurationError` with field, received value, valid values, case ID, and
source. `CaseConfig` has schema version 1, `solver_engine`, and `compat_mode`.

Unknown material no longer becomes elastic. Explicit single engine plus bond
no longer silently becomes multicrack. Bond slips are unambiguously mm. Bond
layer construction and CDP table parsing no longer catch broad exceptions and
continue with another model.

All 13 registered Python factories are parametrically normalized without a
solver run. Six YAML files had no consumers and all diverged: three were invalid
and three differed from factories in 4, 18, and 25 fields. They moved to
`examples/gutierrez_thesis/configs/legacy/` with an evidence table.

### Result Contract

`xfem_clean.results.AnalysisResult` schema version 1 provides `steps`, `fields`,
`energies_or_work`, `cracks`, `reinforcement`, `solver_meta`, and structured
`warnings`. It adapts both solver bundles without copying large arrays. The
deprecated Mapping view preserves dict consumers.

`run.py` CSV export, FRP integration checks, history utilities, and comprehensive
postprocessing consume canonical steps/fields. Single and multicrack internals
remain separate and unchanged.

### Test Reliability And Runtime

- `test_python_numba_parity_simple` no longer absorbs exceptions. Its invalid
  fixture was exposed and fixed: three concrete nodes require steel offset 6,
  not 4. Python/Numba parity now executes and passes.
- Six pytest returns became assertions/normal returns; no
  `PytestReturnNotNoneWarning` remains.
- Every test subprocess uses `tests.process_utils.run_process()` with explicit
  timeout, a new process group, TERM/KILL escalation, and `communicate()` reap.
- FRP validates scalar canonical step fields rather than `np.isfinite(dict)`.
- Two debug examples with unique Git history were repaired, not deleted.
- Matplotlib import/close and `gc.collect()` were removed from the autouse
  fixture. Plotting closes its own figures.
- The two case-04a subprocess tests now use an explicit `0.1 mm` CLI amplitude;
  they still exercise bond/multicrack/CSV paths and complete in 10.57 s instead
  of timing out after 120/180 s.
- The fibre integration uses two onset-scale CDP increments and canonical
  scalar checks; deterministic positive bridging traction is asserted locally.
- The Gutierrez matrix preserves the original increment size when truncating a
  smoke run and executes only the four cases its assertions inspect.
- The DP tangent test reports strain, FD perturbation `1e-8`, absolute error,
  relative error, and regime. Relative errors are about machine precision
  (elastic), 0.063 (transition), and 0.286 (plastic). The 0.30 plastic
  characterization limit remains pending domain review.

### Repository And Documentation

The empty tracked root file `xfem_clean` was removed. `.gitignore` now ignores
output directories rather than global CSV/PDF/PNG extensions. Status documents
were moved to `docs/archive/history/`. Canonical docs now cover architecture,
development, testing, validation, technical basis, units/signs, datasets, and a
recorded decision.

README and dataset docs now state that all three reference CSV files are
synthetic placeholders, not experimental validation evidence. Dependency
ranges and separate `test`, `export`, `numba`, and `cdp` extras are in
`pyproject.toml`.

`xfem_beam.py` is documented as frozen legacy and has an existing energy-HHT
characterization consumer. `xfem_xfem.py` remains the thin compatibility shim.

## Numerical Comparison

The reduced probes were run against Git `HEAD` in an isolated temporary tree
and against this patch.

| Probe | Before | After | Classification |
|---|---:|---:|---|
| Single load [N] | 2552.1731993725325 | 2552.1731993725325 | Exactly equal |
| Single displacement [m] | 1e-6 | 1e-6 | Exactly equal |
| Multi load [N] | 140.97910914191064 | 140.97910914191064 | Exactly equal |
| Multi steps / cracks | 1 / 0 | 1 / 0 | Exactly equal |
| Pullout stress [Pa] | 9675859.46729204 | 9675859.46729204 | Exactly equal |
| Pullout tangent [Pa/m] | 19351718934.58408 | same | Exactly equal |
| FRP stress [Pa] | 1000000.0 | 1000000.0 | Exactly equal |
| FRP tangent [Pa/m] | 2000000000.0 | same | Exactly equal |
| Case-09 density [fibres/m2] | 343000000 | 34300 | Required P0 bug fix |
| Case-09 explicit fraction | hard-coded 0.1 | case-declared 0.5 | Required P0 transport fix |

The old case-09 traction was intentionally not evaluated: the reduced patch
would sample 13.72 million explicit fibres and the result is invalid by unit
contract. No unexpected numerical difference was accepted.

## Final Verification

| Command / measurement | Final result |
|---|---|
| `python3 -m compileall -q src examples tests` | exit 0 |
| `python3 -m pytest --collect-only -q` | 224 tests |
| Fast suite | 195 passed, 29 deselected, zero warnings, 6.38 s |
| Fast wall/RSS | 7.37 s; 213956 KiB maximum RSS |
| Full `--runslow` suite | 219 passed, 5 skipped, zero failures, 117.45 s |
| Full wall/RSS | 118.51 s; 255708 KiB maximum RSS |
| Regression manifest | 8 probes; semantic SHA `4e6c1a892c69099eea344336deb38689fc728a703ebf4640bb769afef25912f6` |
| Process check | no pytest/thesis child remained after completion or forced timeout |

Relative to baseline, fast pytest time decreased by 78.87 s (92.5%), wall time
by 80.13 s (91.6%), and peak RSS by 56084 KiB (20.8%). The slow suite changed
from an external timeout at 1200 s near 30% to a complete 117.45 s run.

Four skips are input-dependent validation checks: the repository has no
provenance-complete experimental outputs and the bundled curves are synthetic
placeholders. The fifth is the already explicit junction-coalescence integration
gap, which requires a complete DOF rebuild with junction enrichment. None is a
solver exception or absorbed failure.

## Test Consolidation

| Removed/merged | Replacement | Preserved guarantee |
|---|---|---|
| Four tests in `test_history_utils_standalone.py` | `test_history_utils.py` | Numeric/dict histories, metrics, P-u curve |
| Three history tests in `test_tools_smoke.py` | `test_history_utils.py` | Numeric/dict conversion and metrics |
| Twelve wrapper functions in `test_regression_cases.py` | Existing parametrized calls | Exact duplicate invocations |
| Twelve broad synthetic range runs | One canonical eight-probe manifest test | Single/multi, bond, FRP, fibre, cyclic, config/result and CSV regression |
| All-case nonlinear execution in `test_examples_smoke.py` | All-case dry-run plus maintained executable scripts and family integrations | Entrypoint/config coverage without unasserted full solves |
| Unasserted 13-case matrix variants | Four explicitly asserted matrix cases plus 13-factory normalization | Matrix status, Python/Numba variants and complete config registry |

Thirty tests were removed or merged and seventeen contract guards were added,
reducing collection from 237 to 224 (-13, 5.5%). Test Python LOC decreased from
11649 to 11263 (-386, 3.3%); test files changed from 59 to 61 because the process
helper and contract module are new. Python LOC across source, examples, tests,
benchmarks, scripts, validation and calibration decreased from 49344 to 48905
(-439). Historical documentation moved to the archive accounts for most of the
larger repository-wide line reduction.

## Compatibility And Deferred Work

Preserved: single and multicrack solvers, `xfem_xfem` imports, dict result keys,
numeric/list histories, legacy work aliases, JSON/YAML schema-1 import, and all
existing case factory IDs/aliases.

Potential later removals and missing evidence:

| Candidate | Evidence still required |
|---|---|
| `xfem_beam.py` | External import/release telemetry and replacement characterization |
| Single-crack solver | Migration of cyclic and direct solver consumers plus golden parity |
| `xfem_xfem.py` | Supported-release deprecation window and external import audit |
| Archived YAML | Release artifact search and authoritative source comparison |
| Legacy result/history views | Downstream consumer migration and one release deprecation window |

Human decisions still pending:

1. Confirm the case-09 50% explicit sampling statement against the primary source.
2. Review the plane-stress DP tangent beyond the characterized plastic point.
3. Replace synthetic validation curves with provenance-complete experimental data.
4. Decide a supported release window for legacy result/config/history adapters.
5. Supply the missing `XFEM_AUDITORIA_INTEGRAL_2026-07-15.md` if it contains unique findings.

## Deliverables

- Full working-tree patch in the shared repository.
- Canonical report and documentation under `REFACTOR_REPORT.md` and `docs/`.
- Reproducible command: `PYTHONPATH=src python3 scripts/regression_manifest.py`.
- Clean ZIP: `/tmp/xfem-concrete-refactor-2026-07-15.zip` (792 KiB).
- ZIP integrity verified; `.git`, `.venv`, tool metadata, caches, build output,
  temporary tests and generated analysis outputs are excluded.

## Recommended Next PRs

1. Domain-reviewed DP plane-stress tangent and transition/plastic golden matrix.
2. Migrate remaining postprocessors/tools fully to `AnalysisResult`, then emit
   once-per-process deprecations for direct legacy keys.
3. Inventory external release consumers and design shims for moving the maintained
   thesis application out of `examples/`, without removing either solver.
