# XFEM Nonlinear Concrete: User Guide

## Overview

This guide documents the nonlinear material capabilities of the XFEM concrete solver, including:

- **Concrete Damage Plasticity (CDP)**: Full Lee-Fenves model with tension/compression damage
- **Bond-Slip**: Interface model for reinforcement (Model Code 2010)
- **Arc-Length Control**: Path-following for post-peak behavior
- **Energy Tracking**: Plastic, fracture, and cohesive dissipation
- **Damage Visualization**: VTK export for ParaView

---

## Table of Contents

1. [Material Models](#material-models)
2. [Bond-Slip Interface](#bond-slip-interface)
3. [Solver Control](#solver-control)
4. [Energy Analysis](#energy-analysis)
5. [Visualization](#visualization)
6. [Complete Examples](#complete-examples)
7. [Theory & Validation](#theory--validation)

---

## 1. Material Models

### 1.1 Concrete Damage Plasticity (CDP)

The CDP model combines:
- **Plasticity**: Drucker-Prager yield surface on effective stress
- **Damage**: Split scalar variables for tension (dt) and compression (dc)
- **Hardening**: Uniaxial stress-strain curves from cdp_generator

#### Basic Usage

```python
from xfem_clean.xfem_xfem import XFEMModel

model = XFEMModel(
    L=1.0, H=0.2, b=0.15,
    E=33e9,         # Elastic modulus [Pa]
    nu=0.2,         # Poisson's ratio
    ft=2.9e6,       # Tensile strength [Pa]
    fc=38e6,        # Mean compressive strength [Pa]
    Gf=120.0,       # Fracture energy [J/m²]

    # Activate CDP with automatic calibration
    bulk_material="cdp_real",
    cdp_use_generator=True,
)
```

#### Advanced: Manual CDP Parameters

For custom calibration (without cdp_generator):

```python
model = XFEMModel(
    ...
    bulk_material="cdp_real",
    cdp_use_generator=False,

    # Yield surface parameters
    cdp_fb0_fc0=1.16,      # Biaxial/uniaxial strength ratio
    cdp_Kc=0.667,          # Meridian ratio

    # Plastic potential
    cdp_psi_deg=30.0,      # Dilation angle [degrees]
    cdp_ecc=0.1,           # Eccentricity

    # Hardening tables (manually provided)
    cdp_w_tab_m=np.array([...]),         # Crack opening [m]
    cdp_sig_t_tab_pa=np.array([...]),    # Tension stress [Pa]
    cdp_dt_tab=np.array([...]),          # Tension damage [-]

    cdp_eps_in_c_tab=np.array([...]),    # Inelastic compression strain [-]
    cdp_sig_c_tab_pa=np.array([...]),    # Compression stress [Pa]
    cdp_dc_tab=np.array([...]),          # Compression damage [-]
)
```

#### State Variables at Integration Points

Each material point stores:

```python
from xfem_clean.material_point import MaterialPoint

mp = MaterialPoint()
mp.eps           # [3] Strain (xx, yy, xy)
mp.sigma         # [3] Stress
mp.eps_p         # [3] Plastic strain
mp.damage_t      # Tension damage [0,1]
mp.damage_c      # Compression damage [0,1]
mp.kappa         # Hardening variable
mp.w_plastic     # Plastic dissipation [J/m³]
mp.w_fract_t     # Tension fracture energy [J/m³]
mp.w_fract_c     # Compression crushing energy [J/m³]
```

---

## 2. Bond-Slip Interface

### 2.1 Model Code 2010 Bond-Slip Law

The bond-slip relationship follows fib Model Code 2010 (Section 6.1.2):

$$
\tau(s) = \begin{cases}
\tau_{\max} (s/s_1)^\alpha & 0 \leq s \leq s_1 \\
\tau_{\max} & s_1 < s \leq s_2 \\
\tau_{\max} - (\tau_{\max} - \tau_f)(s-s_2)/(s_3-s_2) & s_2 < s \leq s_3 \\
\tau_f & s > s_3
\end{cases}
$$

Where:
- τ_max: Maximum bond stress (depends on f_cm and bond condition)
- s1, s2, s3: Characteristic slips
- τ_f: Residual bond stress (0.15 × τ_max)
- α: Shape exponent (0.4)

### 2.2 Usage

```python
from xfem_clean.bond_slip import BondSlipModelCode2010, assemble_bond_slip

# Define bond-slip law
bond_law = BondSlipModelCode2010(
    f_cm=38e6,         # Mean concrete strength [Pa]
    d_bar=16e-3,       # Bar diameter [m]
    condition="good",  # "good" or "poor" bond
)

# During assembly (requires extended DOF structure)
f_bond, K_bond, bond_states_new = assemble_bond_slip(
    u_total=u,
    steel_segments=rebar_segments,
    steel_dof_offset=n_concrete_dofs,
    bond_law=bond_law,
    bond_states=bond_states,
    use_numba=True,
)
```

### 2.3 Dowel Action at Cracks

When a crack crosses a reinforcement bar:

```python
from xfem_clean.bond_slip import compute_dowel_springs, assemble_dowel_action

# Compute dowel springs
dowel_springs = compute_dowel_springs(
    crack_geom=crack,
    steel_segments=rebar_segments,
    nodes=nodes,
    E_steel=200e9,
    d_bar=16e-3,
)

# Assemble dowel stiffness
K_dowel = assemble_dowel_action(
    dowel_springs,
    steel_dof_offset=n_concrete_dofs,
    ndof_total=total_dofs,
)

K_total = K_bulk + K_bond + K_dowel
```

**Note**: Full integration of bond-slip with the main solver requires extending the DOF management. See `NONLINEAR_IMPLEMENTATION_PLAN.md` for integration details.

---

## 3. Solver Control

### 3.1 Displacement Control (Default)

Standard incremental-iterative Newton-Raphson:

```python
results = run_analysis_xfem(
    model,
    nx=120, ny=24,
    nsteps=50,
    umax=0.015,      # Maximum displacement [m]
    law=cohesive_law,
    use_numba=True,
)
```

### 3.2 Arc-Length Control

For problems with snap-back (softening, buckling):

```python
from xfem_clean.arc_length import ArcLengthSolver, arc_length_step

# Initialize arc-length solver
arc_solver = ArcLengthSolver(
    arc_length_initial=0.01,
    psi=1.0,
    max_iterations=25,
    tol_residual=1e-6,
    tol_displacement=1e-8,
    adaptive=True,
)

# Main analysis loop
for step in range(n_steps):
    converged, u_new, lambda_new, arc_new, n_iter = arc_solver.solve_step(
        K=K_current,
        f_int=f_int_current,
        P_ref=P_reference,
        u_n=u_current,
        lambda_n=lambda_current,
        assemble_system=assemble_fn,
        debug=True,
    )

    if converged:
        # Update state
        u_current = u_new
        lambda_current = lambda_new
        arc_solver.arc_length = arc_new
```

**Integration**: Arc-length requires modification of `run_analysis_xfem` to support load control. See `arc_length.py` for functional interface.

---

## 4. Energy Analysis

### 4.1 Compute Global Energies

```python
from xfem_clean.output.energy import compute_global_energies, EnergyBalance

energy_balance = compute_global_energies(
    mp_states=bulk_states,
    coh_states=cohesive_states,
    elems=elems,
    nodes=nodes,
    thickness=model.b,
    cohesive_law=law,
    P_ext=external_forces,
    u=displacements,
    K=tangent_stiffness,
)

print(energy_balance)
```

**Output**:
```
Energy Balance [J]:
  Plastic (bulk):        1.234567e+02
  Fracture (tension):    4.567890e+01
  Crushing (compression):2.345678e+01
  Cohesive (cracks):     7.890123e+01
  Steel plasticity:      0.000000e+00
  Bond-slip:             0.000000e+00
  ────────────────────────────────────
  Total dissipated:      2.702468e+02
  Elastic stored:        5.432109e+01
  External work:         3.245679e+02
  Balance error:         1.234567e-01
```

### 4.2 Energy Time Series

```python
from xfem_clean.output.energy import energy_time_series, plot_energy_evolution

# Collect energy at each step
energy_history = []
for step, (u, mp_states, coh_states) in enumerate(zip(u_hist, mp_hist, coh_hist)):
    eb = compute_global_energies(mp_states=mp_states, ...)
    energy_history.append(eb)

# Plot evolution
plot_energy_evolution(
    energy_history,
    time_or_steps=np.arange(len(energy_history)),
    filename="energy_evolution.png",
)
```

### 4.3 Energy Balance Validation

Check energy conservation:

```python
final_error = energy_history[-1].energy_error
final_total = energy_history[-1].W_total
relative_error = abs(final_error / final_total) * 100

if relative_error < 5.0:
    print("✓ Energy balance: EXCELLENT")
elif relative_error < 10.0:
    print("✓ Energy balance: OK")
else:
    print("⚠ Energy balance: WARNING - high imbalance")
```

---

## 5. Visualization

### 5.1 Damage Field Export (VTK)

Export damage fields for ParaView:

```python
from xfem_clean.output.vtk_export import export_damage_field, export_full_state

# Export final state
export_damage_field(
    filename="damage_field_final.vtk",
    nodes=nodes,
    elems=elems,
    mp_states=bulk_states_final,
)

# Export full state (including displacements)
export_full_state(
    filename="state_final.vtk",
    nodes=nodes,
    elems=elems,
    u=displacements,
    mp_states=bulk_states,
    coh_states=cohesive_states,
)
```

### 5.2 Time Series Export

For animation in ParaView:

```python
from xfem_clean.output.vtk_export import export_time_series

export_time_series(
    output_dir="output/vtk_series",
    nodes=nodes,
    elems=elems,
    u_history=displacement_history,
    mp_history=state_history,
    step_numbers=np.arange(len(displacement_history)),
)
```

This creates:
- Individual .vtk files for each step
- A .pvd file for ParaView series loading

### 5.3 ParaView Workflow

1. Open ParaView
2. File → Open → `damage_field_series.pvd`
3. Apply → Color by: `damage_compression`
4. Add filter: "Warp By Vector" (if displacement fields present)
5. Adjust color scale (0.0 to 1.0 for damage)
6. Play animation to see damage evolution

**Key Fields**:
- `damage_compression`: Crushing at top chord (red = failed)
- `damage_tension`: Cracking (red = fully cracked)
- `plastic_strain_magnitude`: ||ε_p||
- `energy_plastic`, `energy_fracture_tension`, `energy_fracture_compression`

---

## 6. Complete Examples

### 6.1 Four-Point Bending with CDP

```python
from xfem_clean.xfem_xfem import run_analysis_xfem, XFEMModel
from xfem_clean.cohesive_laws import CohesiveLaw
from xfem_clean.output.energy import compute_global_energies
from xfem_clean.output.vtk_export import export_damage_field

# Model setup
model = XFEMModel(
    L=1.0, H=0.2, b=0.15,
    E=33e9, nu=0.2, ft=2.9e6,
    fc=38e6, Gf=120.0,
    steel_A_total=4e-4, steel_E=200e9, steel_fy=500e6,
    bulk_material="cdp_real",
    cdp_use_generator=True,
    newton_tol_r=1e-4,
    line_search=True,
)

law = CohesiveLaw(Kn=1e12, ft=2.9e6, Gf=120.0, kind="bilinear")

# Run analysis
results = run_analysis_xfem(
    model, nx=120, ny=24, nsteps=50, umax=0.015,
    law=law, return_states=True, use_numba=True,
)

# Energy analysis
energy_balance = compute_global_energies(
    mp_states=results["mp_states_history"][-1],
    coh_states=results["coh_states_history"][-1],
    elems=results["elems"],
    nodes=results["nodes"],
    thickness=model.b,
)

print(energy_balance)

# Export damage field
export_damage_field(
    "damage_final.vtk",
    results["nodes"],
    results["elems"],
    results["mp_states_history"][-1],
)
```

### 6.2 Custom Material Calibration

```python
from cdp_generator import core

# Generate CDP parameters from concrete class
params = core.compute_cdp_parameters(
    f_cm=38e6,              # Mean strength [Pa]
    age_days=28,
    temperature_C=20,
    rh_percent=50,
    aggregate_type="quartzite",
)

# Extract tables
w_tab = params["w_tab_m"]
sig_t_tab = params["sig_t_tab_pa"]
dt_tab = params["dt_tab"]

eps_in_c_tab = params["eps_in_c_tab"]
sig_c_tab = params["sig_c_tab_pa"]
dc_tab = params["dc_tab"]

# Use in model
model = XFEMModel(
    ...
    bulk_material="cdp_real",
    cdp_use_generator=False,
    cdp_w_tab_m=w_tab,
    cdp_sig_t_tab_pa=sig_t_tab,
    cdp_dt_tab=dt_tab,
    cdp_eps_in_c_tab=eps_in_c_tab,
    cdp_sig_c_tab_pa=sig_c_tab,
    cdp_dc_tab=dc_tab,
)
```

---

## 7. Theory & Validation

### 7.1 CDP Formulation

**Yield Surface** (Lubliner/Lee-Fenves):

$$
F(\sigma, \kappa) = \frac{1}{1-\alpha}\left(\bar{p} + \frac{\sqrt{3J_2}}{\beta(\kappa)} - \gamma\langle\hat{\sigma}_{\max}\rangle\right) - \sigma_c(\kappa)
$$

Where:
- α = (fb0/fc0 - 1) / (2×fb0/fc0 - 1)
- β = σ_c / σ_t × (1-α) - (1+α)
- γ = 3(1-Kc) / (2×Kc - 1)

**Plastic Potential** (non-associative):

$$
G(\sigma) = \sqrt{(\epsilon\sigma_t)^2 + \bar{q}^2} - \bar{p}\tan\psi
$$

**Damage Operator**:

$$
\sigma = (1-d_t)\mathcal{P}_+ : \sigma_{\text{eff}} + (1-d_c)\mathcal{P}_- : \sigma_{\text{eff}}
$$

Where $\mathcal{P}_{\pm}$ projects onto tensile/compressive principal stress directions.

### 7.2 Validation Benchmarks

1. **Uniaxial Compression**:
   - Compare stress-strain curve with CEB-90
   - Expected peak: f_cm at ε ≈ 0.002
   - Softening branch validates damage evolution

2. **Uniaxial Tension**:
   - Peak stress: f_ctm
   - Fracture energy integration: ∫τ(w)dw = Gf

3. **Four-Point Bending**:
   - Crack initiation at mid-span (bottom)
   - Steel yielding before concrete crushing
   - Compression damage at top chord
   - Energy balance error < 5%

4. **Snap-Back Test** (arc-length):
   - Notched beam under displacement control
   - Load-displacement curve should capture softening
   - No premature divergence at peak load

### 7.3 Recommended Mesh Sizes

For crack band regularization (lch = element size):

| Concrete Class | Gf [J/m²] | Recommended h [mm] |
|----------------|-----------|-------------------|
| C20/25         | 80-100    | 5-10              |
| C30/37         | 100-130   | 8-15              |
| C40/50         | 120-150   | 10-20             |
| C50/60         | 140-170   | 12-25             |

**Rule of thumb**: h ≤ Gf / (ft × 5)

---

## 8. Troubleshooting

### 8.1 Newton Convergence Issues

**Symptom**: Iterations exceed `newton_maxit`

**Solutions**:
1. Enable line search: `model.line_search = True`
2. Reduce load step size: increase `nsteps`
3. Relax tolerance: `model.newton_tol_r = 1e-3`
4. Check mesh quality (avoid distorted elements)

### 8.2 Energy Imbalance

**Symptom**: `energy_error` > 10% of `W_total`

**Causes**:
- Incomplete state history (missing steps)
- Incorrect volume integration (check element areas)
- Adaptive substepping (energy not tracked during retries)

**Fix**: Ensure `return_states=True` and compute energy at every converged step.

### 8.3 Damage Saturation

**Symptom**: All elements show d ≈ 1.0 (fully damaged)

**Causes**:
- Excessive loading (umax too large)
- Insufficient fracture energy (Gf too low)
- Mesh too coarse (lch too large)

**Fix**: Increase Gf or refine mesh near expected crack path.

### 8.4 Bond-Slip Integration

**Current Status**: Bond-slip module is implemented but **not yet integrated** into the main solver.

**To integrate**:
1. Extend `XFEMDofs` to include steel DOFs
2. Modify assembly to call `assemble_bond_slip`
3. Update state management for `BondSlipStateArrays`
4. Add dowel action contributions at crack intersections

See `NONLINEAR_IMPLEMENTATION_PLAN.md` for detailed integration steps.

---

## 9. References

### Papers
1. Lee, J., & Fenves, G. L. (1998). "Plastic-damage model for cyclic loading of concrete structures." *J. Eng. Mech.*, 124(8), 892-900.
2. Lubliner, J., et al. (1989). "A plastic-damage model for concrete." *Int. J. Solids Struct.*, 25(3), 299-326.
3. Crisfield, M. A. (1981). "A fast incremental/iterative solution procedure that handles 'snap-through'." *Comput. Struct.*, 13(1-3), 55-62.

### Standards
- fib Model Code 2010 (Section 6.1.2: Bond and anchorage)
- Eurocode 2 (EN 1992-1-1: Concrete material properties)
- CEB-FIP Model Code 1990 (Compression behavior)

### Software
- ParaView: https://www.paraview.org/
- Numba: https://numba.pydata.org/

---

## 10. Future Developments

Planned features (not yet implemented):

1. **Cyclic Loading**: Unloading/reloading for earthquake simulations
2. **Temperature Effects**: Thermal degradation (ISO 834)
3. **Strain-Rate Dependency**: Dynamic loading (Malvar-Crawford)
4. **3D XFEM**: Extension to 3D cracking
5. **Multi-Physics**: Coupled thermo-mechanical analysis

---

*Last updated: 2025-12-29*
*XFEM Concrete Solver v3.0*
