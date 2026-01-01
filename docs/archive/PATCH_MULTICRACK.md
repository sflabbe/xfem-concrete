# Patch: Multi-crack + diagonal (Gutierrez beam)

This patch adds an **experimental multi-crack solver** (Heaviside + cohesive) and extends `run_gutierrez_beam` with CLI options.

## New CLI options

```
python -m run_gutierrez_beam --umax-mm 10 --nsteps 30 --nx 120 --ny 20 --line-search \
  --max-cracks 8 --crack-mode option2 --diag-start-mm 2
```

- `--max-cracks`: maximum number of cracks.
- `--crack-mode`:
  - `single`: original single-crack logic (default)
  - `option1`: multiple **flexural** cracks only (mostly vertical)
  - `option2`: flexural cracks + allow **diagonal shear** crack initiation after `--diag-start-mm`
- `--diag-start-mm`: displacement threshold at which diagonal initiation is allowed (option2).

## Outputs

- `outputs_pu_gutierrez.csv`
  - For `single`: same as before (u, P, M, kappa, tip_x, tip_y)
  - For multi-crack: (step, u, P, M, kappa, ncr)
- `outputs_cracks_gutierrez.csv`: one row per crack (id, x0, y0, tip_x, tip_y, angle_deg)

## Notes / knobs

- Cohesive stiffness is controlled by `model.Kn_factor`.
- If Newton stagnates after crack activation, try smaller `Kn_factor` (softer) and/or increase `newton_maxit`.
- The multi-crack mode currently allows **at most one crack update per accepted step** (either initiation OR growth), to keep the scheme stable.


## Fix 2025-12-26 (runnable multi-crack)

- `assemble_xfem_system_multi`:
  - Removed the call to non-existent `assemble_rebar_contrib_at_gp`.
  - Added rebar contribution via existing `rebar_contrib` and embedded the 2*nnode rebar stiffness into the full XFEM matrix.

- `run_analysis_xfem_multicrack`:
  - Fixed Newton Dirichlet handling to use `apply_dirichlet(K, R, fixed_dofs, q)` (correct signature).
  - Implemented a simple backtracking line search inside the Newton loop (no missing helper).
  - Fixed nonlocal stress-bar calls (argument order + consistent use of `model.crack_rho` as physical length).
  - Added `angle_deg` field to `XFEMCrack` so diagonal cracks can propagate coherently.
