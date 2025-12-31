# XFEM for concrete

## Run (without install)

```bash
python examples/run_gutierrez_beam.py --umax-mm 10 --nsteps 30 --nx 120 --ny 20
```

## Install editable

```bash
pip install -e .
python examples/run_gutierrez_beam.py
```

## Key modules

- `src/xfem_clean/cohesive_laws.py` (extracted from stable solver)
- `src/xfem_clean/convergence.py`
- `src/xfem_clean/crack_criteria.py`
- `src/xfem_clean/xfem_xfem.py` (stable solver, now importing the modules above)

## Phase-1 Numba refactor (state arrays + patches)

This repo includes the first step toward Numba compilation: the solver no longer keeps
integration-point history in Python dicts that are copied every Newton iteration.

See:

- `src/xfem_clean/xfem/state_arrays.py` (`BulkStateArrays`, `CohesiveStateArrays` and their patch counterparts)
- `docs/ROADMAP_NUMBA.md` (phased plan for full Numba `nopython`)

Notes:

- In the single-crack solver, `run_analysis_xfem(..., return_states=True)` returns a `BulkStateArrays` instance.
- `nodal_average_state_fields()` accepts either the legacy dict state or `BulkStateArrays`.

## Phase-2 Numba kernels (constitutive/cohesive)

Phase 2 introduces stateless, value-based kernels that can be compiled by Numba.
They are **opt-in**.

Enable with:

```bash
python examples/run_gutierrez_beam.py --use-numba ...
```

Implemented kernels live in:

- `src/xfem_clean/numba/kernels_cohesive.py` (cohesive law update)
- `src/xfem_clean/numba/kernels_q4.py` (Q4 shape convenience)

Phase 2b adds **bulk constitutive** kernels (still opt-in):

- `src/xfem_clean/numba/kernels_bulk.py`
  - Linear elastic plane-stress.
  - Drucker–Prager (associated) + isotropic hardening (plane-stress enforced locally).

Notes:

- Bulk kernels are currently enabled only for `--bulk-material elastic` and `--bulk-material dp`.
- `--bulk-material cdp` still uses the Python constitutive model (next phase).

See `docs/ROADMAP_NUMBA.md` for the phased plan.

## Validation & Calibration (Thesis Parity)

This repository includes comprehensive validation tools to achieve **thesis parity** with experimental data:

### 1. Quantitative Validation (P–δ Curve Comparison)

Compare simulation results with experimental reference curves using RMSE, peak load error, and energy error metrics:

```bash
# Run validation tests (requires simulation outputs)
pytest tests/test_validation_curves.py -v -m slow

# Example: Validate T5A1 beam
python -m examples.gutierrez_thesis.run --case t5a1 --mesh coarse
pytest tests/test_validation_curves.py::test_validate_t5a1_coarse -v -s
```

**Tolerance thresholds:**
- Peak load error: |ΔPmax| < 10%
- Energy error: |ΔE| < 15%
- RMSE: < 5% of peak load

**Module:** `validation/compare_curves.py`

### 2. Parametric Studies (Sensitivity Analysis)

Perform parameter sweeps to study influence of material/geometric parameters:

```bash
# Sweep fracture energy Gf for T5A1 beam
python -m examples.parametric.parametric_study \
  --case t5a1 --param Gf --values 0.05,0.1,0.2 --mesh coarse

# Sweep bond strength τmax for VVBS3 beam
python -m examples.parametric.parametric_study \
  --case vvbs3 --param tau_max --values 4.0,6.47,9.0 --plot

# Available parameters: Gf, f_t, f_c, tau_max, s1, s2, n_bars, rho_fibre, etc.
```

**Output:** `outputs/parametric/<case>_<param>.csv` (table with Pmax, energy, ductility)

**Module:** `examples/parametric/parametric_study.py`

### 3. Bond Parameter Calibration (Experimental Fitting)

Calibrate bond law parameters (τmax, s1, s2) by fitting to experimental P–δ curves:

```bash
# Calibrate VVBS3 FRP bond parameters
python -m calibration.fit_bond_parameters \
  --case vvbs3 \
  --params tau_max,s1,s2 \
  --init tau_max=6.47,s1=0.02,s2=0.25 \
  --bounds 5-8,0.01-0.05,0.2-0.3 \
  --method L-BFGS-B

# Use global optimizer for robust fitting
python -m calibration.fit_bond_parameters \
  --case vvbs3 \
  --params tau_max,s1 \
  --method differential_evolution \
  --max-iter 100
```

**Output:** `calibration/results_<case>.json` (optimal parameters, convergence metrics)

**Module:** `calibration/fit_bond_parameters.py`

### 4. Performance Benchmarks (Scaling Analysis)

Measure runtime, memory, and energy residual across mesh sizes:

```bash
# Benchmark single case with multiple meshes
python -m benchmarks.benchmark_scaling \
  --case t5a1 --meshes coarse,medium,fine --plot

# Benchmark all cases (fast meshes only)
python -m benchmarks.benchmark_scaling \
  --case all --meshes coarse,medium
```

**Output:** `benchmarks/scaling_summary.csv` (runtime, memory, energy residual)

**Module:** `benchmarks/benchmark_scaling.py`

### 5. Sensitivity Study (Mesh and Solver Parameters)

Explore sensitivity to mesh size and solver parameters (candidate point density):

```bash
# Run sensitivity study for Jason case (4PBT CFRP beam)
python -m examples.sensitivity.sensitivity_study_jason \
  --mesh coarse,medium --cand 1.0,0.5 --plot

# Quick test with minimal steps
python -m examples.sensitivity.sensitivity_study_jason --quick
```

**Output:** `outputs/sensitivity/<case>_summary.csv` (configuration matrix with all metrics)

**Module:** `examples/sensitivity/sensitivity_study_jason.py`

### 6. Crack Width and Steel Force Postprocessing

All simulations now automatically export **crack width profiles** and **steel force distributions** when applicable:

**Crack width profiles** (from cohesive states):
- `crack_width_profile_crack{k}_final.csv` — (s, x, y, w) along crack path
- `crack_width_profile_crack{k}_final.png` — Plot of w(s)
- Metrics: w_max, w_avg

**Steel force profiles** (from bond-slip DOFs):
- `steel_force_profile_final.csv` — (x, N, sigma) along reinforcement
- `steel_force_profile_final.png` — Plots of N(x) and σ(x)

These outputs are generated automatically by `postprocess_comprehensive.py` when cohesive states and bond states are available.

**Module:** `examples/gutierrez_thesis/postprocess_comprehensive.py`, `postprocess.py`

### Robustness and Testing

All parametric, calibration, and benchmark tools are now **robust to both single-crack and multicrack history formats** (numeric arrays vs dicts). A unified history extraction interface is provided by `examples/gutierrez_thesis/history_utils.py`.

**Smoke tests** verify that all tools can execute without crashes:
```bash
# Run smoke tests (fast)
python tests/test_tools_smoke.py
```

### Reference Data

Experimental reference curves are located in `validation/reference_data/`:
- `t5a1.csv` - Bosco T5A1 beam (Chapter 5)
- `vvbs3.csv` - CFRP strengthened beam (Fig 5.30)
- `sorelli.csv` - Fibre-reinforced beam (Chapter 6)

These curves were digitized from thesis figures using WebPlotDigitizer.

### Thesis Case Suite

Run all 10 thesis cases:

```bash
# List available cases
python -m examples.gutierrez_thesis.run --list

# Run specific case
python -m examples.gutierrez_thesis.run --case t5a1 --mesh medium
python -m examples.gutierrez_thesis.run --case vvbs3 --mesh coarse
python -m examples.gutierrez_thesis.run --case sorelli --mesh fine

# Run all cases (coarse mesh for speed)
python -m examples.gutierrez_thesis.run --case all --mesh coarse
```

See `docs/thesis_mapping.md` for detailed mapping between thesis sections and repository cases.
