# Energy Tracking in XFEM Dynamic Analysis

This document describes the energy tracking implementation for HHT-α time integration with physical dissipation decomposition.

## Overview

The XFEM solver uses HHT-α / Newmark time integration for pseudo-dynamic analysis. Energy tracking is implemented to:
1. Verify energy conservation in elastic problems (α=0, no damping)
2. Quantify algorithmic dissipation (α < 0)
3. Decompose physical dissipation into cohesive, bond-slip, and plastic contributions
4. Identify numerical dissipation as the remainder

## HHT-α Time Integration

### Kinematics

Given time step dt and parameter α ∈ [-1/3, 0]:
```
γ = 0.5 - α
β = 0.25 * (1 - α)²
```

Update relations:
```
a_{n+1} = a0 * (u_{n+1} - u_n) - a2 * v_n - a3 * a_n
v_{n+1} = v_n + dt * ((1-γ) * a_n + γ * a_{n+1})
```

where:
```
a0 = 1 / (β * dt²)
a2 = 1 / (β * dt)
a3 = 1/(2β) - 1
```

### Residual

Weighted residual at n+α:
```
g_n     = f_int(u_n)     + C * v_n     + M * a_n
g_{n+1} = f_int(u_{n+1}) + C * v_{n+1} + M * a_{n+1}

R = (1 + α) * g_{n+1} - α * g_n = 0
```

## Energy Components

### 1. Constraint Work (Energy Input)

Energy input through Dirichlet boundary conditions:
```
ΔW_dir = 0.5 * (λ_n + λ_{n+1}) · Δu_dir
```

where:
- `λ_n`, `λ_{n+1}` are reaction forces at constrained DOFs
- `Δu_dir = u_{n+1}[dir_dofs] - u_n[dir_dofs]`

Trapezoidal integration ensures consistency with HHT-α for α=0.

### 2. Mechanical Energy

Kinetic energy (lumped mass):
```
T = 0.5 * M_diag · v²
```

Bulk recoverable energy (elastic proxy):
```
Ψ_bulk = ∫_Ω 0.5 * σ : ε dV
```

Total mechanical energy:
```
E_mech = T + Ψ_bulk
```

### 3. Damping Dissipation

Rayleigh damping dissipation:
```
ΔD_damp = 0.5 * dt * (v_n · f_d_n + v_{n+1} · f_d_{n+1})
```

where `f_d = C * v` is the damping force.

**No extra matvecs**: Reuses `f_d` already computed in Newton solver.

### 4. Physical Dissipation

#### 4.1 Cohesive Dissipation

For each cohesive Gauss point with opening δ and traction t:
```
ΔD_coh_gp = 0.5 * (t_n + t_{n+1}) · (δ_{n+1} - δ_n) * w_gp * thickness
```

Total cohesive dissipation:
```
D_coh_inc = Σ_gp ΔD_coh_gp
```

**Notes**:
- Trapezoidal work integral (energy-consistent)
- Under monotonic fracture to zero traction: `D_coh_total ≈ Gf * crack_area`
- Can be negative during unloading (recoverable elastic energy)
- **Mixed-mode** (Mode I + Mode II):
  ```
  ΔD_coh_gp = 0.5 * [(t_n_old + t_n_new) * Δδ_n + (t_t_old + t_t_new) * Δδ_t] * w * thickness
  ```

#### 4.2 Bond-Slip Dissipation

For each bond segment with slip s and bond stress τ:
```
ΔD_bond_gp = 0.5 * (τ_n + τ_{n+1}) * (s_{n+1} - s_n) * perimeter * w_gp * J_seg
```

Total bond dissipation:
```
D_bond_inc = Σ_layers Σ_seg Σ_gp ΔD_bond_gp
```

where:
- `perimeter = π * d_bar` for circular rebar (or explicit value for FRP)
- `J_seg` accounts for mapping to physical length (e.g., `L_seg/2` for 2-pt Gauss)

**Important**:
- Masked segments (bond disabled) contribute zero dissipation but retain steel axial element
- τ_n is evaluated using **committed** bond state (do not mutate history)
- Supports multi-layer reinforcement (steel + FRP)

#### 4.3 Bulk Plastic Dissipation

For each element Gauss point:
```
ΔD_bulk_gp = σ : Δε_plastic * detJ * w_gp * thickness
```

Total bulk plastic dissipation:
```
D_bulk_plastic_inc = Σ_elem Σ_gp ΔD_bulk_gp
```

**Notes**:
- For purely elastic material: `Δε_plastic = 0` → `D_bulk_plastic = 0`
- For Drucker-Prager or CDP: Extract `Δε_p = ε_p_{n+1} - ε_p_n` from state update

#### 4.4 Total Physical Dissipation

```
D_physical_inc = D_coh_inc + D_bond_inc + D_bulk_plastic_inc
```

### 5. Algorithmic Dissipation

Computed as a remainder to satisfy energy balance:
```
ΔD_alg = ΔW_dir - (ΔE_mech + ΔD_damp)
```

**Expected behavior**:
- `α = 0` (Newmark average acceleration): `ΔD_alg ≈ 0` (roundoff only)
- `α < 0` (HHT-α with numerical damping): `ΔD_alg > 0`

### 6. Numerical Dissipation

Decompose algorithmic dissipation:
```
ΔD_numerical = ΔD_alg - ΔD_physical
```

This represents pure numerical dissipation (discretization error, Newton tolerance, etc.).

## Implementation

### Efficient Computation

**No extra global assemblies or sparse matvecs per accepted step.**

Dissipation is computed inside existing element/segment loops:
- Cohesive: During cohesive Gauss point loop (lines 682-711 in `assembly_single.py`)
- Bond: During bond segment loop (lines 1686-1726 in `bond_slip.py`)
- Bulk: During bulk Gauss point loop (to be implemented)

### API Usage

#### Assembly

```python
# Assemble with dissipation tracking
K, f, coh_states, mp_states, aux, bond_states, _, _ = assemble_xfem_system(
    ...,
    q_prev=q_n,               # Displacement at previous time step
    compute_dissipation=True,  # Enable dissipation computation
)

# Extract dissipation increments
D_coh_inc = aux["D_coh_inc"]          # Cohesive dissipation [J]
D_bond_inc = aux["D_bond_inc"]        # Bond-slip dissipation [J]
D_bulk_plastic_inc = aux["D_bulk_plastic_inc"]  # Bulk plastic dissipation [J]
```

#### Energy Tracking

```python
from xfem_clean.xfem.energy_hht import compute_step_energy

energy = compute_step_energy(
    step=i,
    t_n=t_n, t_np1=t_np1,
    Mdiag=M_diag,
    u_n=u_n, v_n=v_n,
    f_int_n=f_int_n, f_d_n=f_d_n, f_m_n=f_m_n,
    psi_bulk_n=psi_bulk_n,
    u_np1=u_np1, v_np1=v_np1,
    f_int_np1=f_int_np1, f_d_np1=f_d_np1, f_m_np1=f_m_np1,
    psi_bulk_np1=psi_bulk_np1,
    dir_dofs=dir_dofs,
    W_dir_cum_prev=W_dir_cum,
    D_damp_cum_prev=D_damp_cum,
    D_alg_cum_prev=D_alg_cum,
    # Physical dissipation (TASK 5)
    D_coh_inc=D_coh_inc,
    D_bond_inc=D_bond_inc,
    D_bulk_plastic_inc=D_bulk_plastic_inc,
    D_coh_cum_prev=D_coh_cum,
    D_bond_cum_prev=D_bond_cum,
    D_bulk_plastic_cum_prev=D_bulk_plastic_cum,
    D_physical_cum_prev=D_physical_cum,
    D_numerical_cum_prev=D_numerical_cum,
)
```

#### CSV Export

```python
from xfem_clean.xfem.energy_hht import write_energy_csv

energies = []  # List[StepEnergy] from each accepted step
write_energy_csv(energies, "energy.csv")
```

Output columns:
- `step`, `t`, `dt`
- `W_dir_inc`, `W_dir_cum` (constraint work)
- `T_n`, `T_np1` (kinetic energy)
- `Psi_bulk_n`, `Psi_bulk_np1` (recoverable energy)
- `D_damp_inc`, `D_damp_cum` (damping dissipation)
- `D_alg_inc`, `D_alg_cum` (algorithmic dissipation)
- `D_coh_inc`, `D_coh_cum` (cohesive dissipation)
- `D_bond_inc`, `D_bond_cum` (bond-slip dissipation)
- `D_bulk_plastic_inc`, `D_bulk_plastic_cum` (plastic dissipation)
- `D_physical_inc`, `D_physical_cum` (total physical dissipation)
- `D_numerical_inc`, `D_numerical_cum` (numerical dissipation)
- `E_mech_n`, `E_mech_np1` (mechanical energy)
- `balance_inc` (energy balance check, should be ~0)

## Validation

### Test Cases

1. **Elastic (α=0, no damping)**:
   - Expected: `|D_numerical| < 1e-6` (only roundoff error)
   - Verifies: Trapezoidal integration is energy-conserving

2. **Elastic (α<0, no damping)**:
   - Expected: `D_numerical > 0` (algorithmic dissipation)
   - Verifies: HHT-α numerical damping

3. **Cohesive fracture**:
   - Expected: `D_coh_total ≈ Gf * crack_area` (monotonic loading)
   - Verifies: Cohesive dissipation formula

4. **Bond-slip cyclic**:
   - Expected: `D_bond_total > 0` (hysteresis dissipation)
   - Verifies: Bond dissipation formula

5. **Plastic compression**:
   - Expected: `D_bulk_plastic > 0`
   - Verifies: Plastic dissipation formula

### Energy Balance

Global energy balance (should hold to machine precision):
```
ΔW_dir = ΔE_mech + ΔD_damp + ΔD_alg
       = ΔE_mech + ΔD_damp + ΔD_physical + ΔD_numerical
```

Check:
```
balance = ΔW_dir - (ΔE_mech + ΔD_damp + ΔD_alg)
assert |balance| < 1e-10
```

## Physical Interpretation

### Cohesive Dissipation

- **Monotonic opening**: Energy released as the cohesive zone opens from `δ=0` to `δ=δf`
- **Cyclic loading**: Can be negative during unloading (elastic energy recovery)
- **Final value**: `∫_0^{δf} t(δ) dδ = Gf` (fracture energy)

### Bond-Slip Dissipation

- **Monotonic slip**: Energy dissipated as the bond degrades
- **Cyclic loading**: Hysteresis loop area (irreversible energy loss)
- **Yielding reduction**: Reduced dissipation when steel yields (Ωy < 1)
- **Crack deterioration**: Reduced dissipation near cracks (Ωc < 1)

### Numerical Dissipation

- **α = 0**: Only discretization error and Newton tolerance
- **α < 0**: Intentional numerical damping to control high-frequency modes
- **Decomposition**: `D_numerical = D_alg - D_physical`

## References

1. Hilber, H. M., Hughes, T. J. R., & Taylor, R. L. (1977). Improved numerical dissipation for time integration algorithms in structural dynamics. *Earthquake Engineering & Structural Dynamics*, 5(3), 283-292.

2. Orlando Gutiérrez. PhD Thesis on reinforced concrete with XFEM and bond-slip (formulas for Ωy, Ωc, dissipation).

3. fib Model Code 2010 (bond-slip constitutive law).

## Implementation Status

- ✅ **Cohesive dissipation** (Mode I + mixed-mode): Complete (Python)
- ✅ **Bond dissipation** (Python): Complete
- ✅ **Bulk plastic dissipation** (Numba + Python): Complete
  - Elastic (bulk_kind=1): dW = 0 ✓
  - Drucker-Prager (bulk_kind=2): dW from return mapping ✓
  - CDP (bulk_kind=3): dW = wpl_new - wpl_old ✓
- ✅ **Energy framework integration**: Complete
- ✅ **Comprehensive tests**: Complete (formula validation + integration tests)
- 🔴 **Bond dissipation** (Numba): Deferred (Python path sufficient)

---

**Last updated**: 2026-01-03
**Author**: Claude (TASK 5 complete)
