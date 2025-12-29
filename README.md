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
