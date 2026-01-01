# RC Beam – CDP (Abaqus-like) + Modified Newton (Prototype)

This ZIP contains a **standalone Python prototype** for a 2D reinforced-concrete beam (plane-stress Q4),
driven by **displacement control** at midspan (top strip). It is intended for *calibration workflows*
(e.g., moment–curvature / plastic hinge tuning).

---

# Option B (recommended): XFEM real + cohesive crack (single dominant midspan crack)

This repo now includes a **real XFEM** prototype (straight crack, Heaviside + tip enrichment) with a
**bilinear cohesive traction–separation law** (Mode I) and **adaptive substepping** under
**displacement control**. This is meant to reproduce **one dominant flexural crack at midspan**
for validation, then **grow the tip up to mid-height** (to avoid contact/complexities).

## Run

```bash
python -m run_beam_xfem
```

Outputs:
- `results.csv` (step, u, P, M, curvature, radius, tip_y, crack_active)
- `mesh_undeformed.png`, `mesh_deformed.png` (scale=1, no artificial scaling)
- `disp_mag.png`, `pu_curve.png`
- `stress_sigma1.png`, `stress_von_mises.png`, `stress_tresca.png` (nodal averaged + crack-gap masked)

## Candidate points + debug

- Candidate points are generated at **bottom-edge element midpoints** (flexural cracking).
- `model.cand_mode`: `"dominant"` (window around midspan), `"three"` (three windows), `"full"` (whole span).
- `model.dominant_crack=True` keeps **one** dominant crack for validation (even if `cand_mode="three"`).
- Debug prints:
  - `model.debug_substeps=True`: prints every adaptive substep attempt.
  - `model.debug_newton=True`: prints when Newton converges inside each substep.

## Where the XFEM implementation lives

- `xfem_xfem.py`:
  - Q4 plane-stress bulk (linear elastic concrete)
  - Heaviside enrichment (displacement jump)
  - Tip enrichment (4 branch functions) in a patch around the crack column
  - Cohesive line integral along the crack (bilinear damage; Rankine-driven initiation)
  - Newton + line-search + adaptive bisection substepping

## Notes / current limitations (prototype)

- Crack is **straight vertical** (flexural crack), aligned with the structured mesh axis.
- Cohesive law is **Mode I only** (normal opening); shear/contact are not modeled (by design).
- Tip enrichment is applied in a **patch** (superset) and stabilized with a small diagonal `k_stab`.
- Concrete bulk is linear elastic in this XFEM path (CDP remains available in `xfem_beam.py`).



## What is implemented

- **Concrete Damaged Plasticity (CDP)** inspired by Abaqus:
  - Lubliner/Lee–Fenves type yield surface with a **Rankine** term using the maximum principal stress
  - Non-associated flow rule with **hyperbolic Drucker–Prager** potential (dilation angle `psi`, eccentricity `ecc`)
  - Separate damage variables in **tension** (`d_t`) and **compression** (`d_c`) with tabular laws
- **Damage tracking** (`d_t`, `d_c`) + **Rankine index** plots.
- **Rebar** as an embedded truss line with **bilinear B500A** (fy/fu + small hardening).
- **MODIFIED NEWTON**: a **secant bulk stiffness** is assembled once per load step from committed damage;
  line-search is used for robustness.

Concrete tension and compression tables are generated **offline** using the vendored `cdp_generator/`
package (provided by you).

## Run

```bash
python -m run_beam_xfem
```

Outputs:
- `results.csv`
- `mesh_undeformed.png`, `mesh_deformed.png`, `disp_mag.png`
- `stress_von_mises.png`, `stress_tresca.png`
- `damage_tension.png`, `damage_compression.png`, `rankine_index.png`
- `M_kappa.png`, `M_R.png`


## Acceleration (Numba)
If `numba` is installed, the prototype automatically JIT-compiles:
- Bulk secant stiffness assembly (Modified Newton)
- Rebar internal force + tangent (bilinear truss line)

This provides a noticeable speedup, especially for larger meshes.


## Integrator: HHT-α (quasi-static)
The default `run_analysis(...)` uses an **implicit HHT-α** pseudo-time stepper to obtain quasi-static response with numerical dissipation.
- Parameters live in `Model`: `hht_alpha`, `total_time`, `dt`, `rho`, `rayleigh_aM`, `rayleigh_aK`.
- The old incremental-static integrator is kept as `run_analysis_static(...)`.


## Troubleshooting HHT-α (quasi-static)
- If HHT fails to converge early, increase `total_time` (larger dt -> smaller inertia terms), and/or increase `rayleigh_aK`.
- `rho` is **pseudo-mass** here; reducing it also reduces inertia stiffness.
- This prototype updates the secant bulk stiffness **every Newton iteration** in HHT for robustness.


## Performance notes
- A large part of the original runtime was Python overhead in `element_B_detJ` for every Gauss point and iteration.
- This prototype now **precomputes** all `B` matrices + `detJ` once (`precompute_B_detJ`) and reuses them.
- Numba acceleration currently targets stiffness assembly + rebar tangent; CDP return-mapping is still Python.


## Adaptive substepping
The HHT-α quasi-static integrator uses **adaptive substepping**:
- Each nominal step (Δu, Δt) is attempted once.
- If Newton fails to converge, the interval is bisected: (Δu/2, Δt/2) + (Δu/2, Δt/2).
- This repeats up to `max_subdiv` times, with a guard `min_dt`.
This is critical for XFEM/CDP softening where snap-back/snap-through can occur.
