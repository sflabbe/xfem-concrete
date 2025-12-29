# Roadmap: Numba-ready XFEM

This roadmap describes how to evolve the repository from the current Python implementation to a Numba-compileable (`nopython`) data-oriented solver, aligned with the Gutierrez thesis.

## Phase 1 — State arrays + sparse patches (DONE)

Objective: remove the Newton-iteration dictionary churn and prepare the codebase for later `@njit` kernels.

Implemented in this repository:

- `src/xfem_clean/xfem/state_arrays.py`
  - `BulkStateArrays` stores `MaterialPoint` history as flat NumPy arrays.
  - `BulkStatePatch` is a sparse update list applied only on convergence.
  - `CohesiveStateArrays` stores cohesive history as flat NumPy arrays with a primary dimension for multi-crack runs.
  - `CohesiveStatePatch` is the sparse update list for cohesive states.

- Single-crack solver
  - Assembly returns patches instead of mutating state dicts: `src/xfem_clean/xfem/assembly_single.py`.
  - Newton applies patches only when converged: `src/xfem_clean/xfem/analysis_single.py`.

- Multi-crack solver
  - Cohesive history is stored in `CohesiveStateArrays` (preallocated by `max_cracks`).
  - Assembly returns a `CohesiveStatePatch` (arrays mode) or a trial dict (legacy mode).
  - Newton applies the patch only on convergence: `src/xfem_clean/xfem/multicrack.py`.

- Post-processing
  - `nodal_average_state_fields()` now accepts either a dict or `BulkStateArrays`: `src/xfem_clean/xfem/post.py`.

Acceptance criteria for Phase 1:

- The examples run unchanged (or with minimal internal refactors) while using the new state containers.
- No full dict copies of cohesive/material-point history inside Newton loops.

## Phase 2 — Stateless `@njit` constitutive kernels (IN PROGRESS)

Objective: extract physics computations into pure functions operating on arrays (no Python objects) and compile them with Numba.

Tasks:

1. Replace `MaterialPoint` objects inside hot loops with raw arrays.
   - Introduce a set of `@njit` functions taking `(eps, state_vars, params)` and returning `(sigma, Ct, new_state_vars, energies)`.

2. Cohesive law as a pure kernel.
   - Implement `cohesive_update_values(delta_n, delta_max, damage, params)` returning `t_n, dt/ddelta, new_delta_max, new_damage`.

Implemented (Phase 2a):

- `src/xfem_clean/numba/kernels_cohesive.py`
  - `pack_cohesive_law_params(law)` packs a `CohesiveLaw` into a small float array.
  - `cohesive_update_values_numba(...)` is a stateless, value-based cohesive update for bilinear and Reinhardt laws.
- `src/xfem_clean/numba/kernels_q4.py`
  - `q4_shape_numba(...)` convenience kernel (used in Phase 3).
- Opt-in flag: `model.use_numba` and CLI `--use-numba` (Gutierrez runner).
- Assembly uses the value kernel automatically when:
  - `model.use_numba == True`, and
  - cohesive history is stored in `CohesiveStateArrays` (Phase 1), and
  - packed params are available.

Implemented (Phase 2b — bulk constitutive kernels):

- `src/xfem_clean/numba/kernels_bulk.py`
  - `elastic_integrate_plane_stress_numba(...)`.
  - `dp_integrate_plane_stress_numba(...)` (associated Drucker–Prager + isotropic hardening, with
    plane-stress enforced by a local Newton iteration on `eps_zz`).
  - `pack_bulk_params(model)` packs the bulk material settings into a small float array.
- Single-crack assembly (`src/xfem_clean/xfem/assembly_single.py`) uses the bulk kernels automatically when:
  - `model.use_numba == True`, and
  - bulk history is stored in `BulkStateArrays`, and
  - `bulk_material in {'elastic','dp'}`.

Notes:
- CDP return mapping is still executed via the Python constitutive model in this phase.

3. Preallocate integration-point buffers.
   - Move temporary per-element arrays to reusable buffers (avoid allocations in loops).

## Phase 3 — Numba assembly (triplets) (PLANNED)

Objective: compile the assembly loops for K and internal forces.

Tasks:

1. Replace Python list appends for sparse triplets with preallocated arrays.
   - Determine an NNZ upper bound per element and fill `(rows, cols, data)` with a counter.

2. Split topology from math.
   - Build “connectivity + geometry” arrays once.
   - Pass only arrays to `@njit` assembly kernels.

## Phase 4 — Missing thesis features (PARALLEL TRACK)

These are features missing or simplified relative to the Gutierrez thesis; they can be developed in parallel to the Numba work.

- Multi-crack + bulk nonlinearity (DP/CDP) support.
- Junction enrichment for crack coalescence (tip enrichment and junction DOFs).
- Transverse reinforcement interaction (stirrups / bar-to-bar contact).
