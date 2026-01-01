# XFEM v13 – Reinhardt cohesive law patch

This patch adds an optional **Reinhardt (Gutiérrez Eq. 3.24 / 3.25)** traction–separation law for Mode‑I cohesive cracks.

## Changes

### 1) `xfem_xfem.py`

- `CohesiveLaw`
  - new selector `law = "bilinear" | "reinhardt"`
  - Reinhardt parameters: `c1` (default 3.0) and `c2` (default 6.93)
  - critical opening `wcrit = 5.136 * Gf / ft` (Eq. 3.25)
  - `deltaf` now prints `wcrit` when `law="reinhardt"`

- `cohesive_update`
  - Reinhardt envelope (Eq. 3.24):

    t(w) = ft * ( (1 + (c1*w/wc)^3) * exp(-c2*w/wc) - (w/wc) * (1 + c1^3) * exp(-c2) )

    for 0 ≤ w ≤ wc, else t=0.
  - unloading/reloading is a straight line through the origin with secant stiffness at `w_max` (Eq. 3.29).
  - algorithmic stiffness is kept non‑negative (secant) with a small residual floor, and capped by `k_cap = cohesive_kcap_factor * Kn`.

- `XFEMModel`
  - new parameters: `cohesive_law`, `reinhardt_c1`, `reinhardt_c2`, `cohesive_kcap_factor`.

### 2) `run_gutierrez_beam.py`

New CLI flags:

- `--cohesive-law {bilinear,reinhardt}`
- `--reinhardt-c1 <float>`
- `--reinhardt-c2 <float>`
- `--cohesive-kcap-factor <float>`

Example:

```bash
python -u -m run_gutierrez_beam --umax-mm 4.5 --nsteps 10 --nx 60 --ny 10 \
  --max-cracks 8 --crack-mode option2 --cohesive-law reinhardt
```

## Patch file

Apply the patch with:

```bash
patch -p0 < xfem_reinhardt_patch.diff
```
