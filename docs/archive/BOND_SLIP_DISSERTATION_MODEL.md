# Bond-Slip Model According to Dissertation

This document describes the enhanced bond-slip interface model implemented according to the dissertation (10.5445/IR/1000124842).

## Overview

The bond-slip model couples reinforcing steel bars to concrete through:
1. **Tangential bond stress** τ(s) - Model Code 2010 with modifications
2. **Normal dowel action** σ_r(w) - Brenna et al. model
3. **Reduction factors** - Steel yielding Ωy and crack deterioration Ωcrack

## Part A: Debug/Hardening (COMPLETED)

Prevents kernel-level hangs when crack growth changes DOF count:

### A1. Preflight Validation
```python
from xfem_clean.bond_slip import validate_bond_inputs

# Validates all inputs before Numba kernel
validate_bond_inputs(
    u_total=u_total,
    segs=steel_segments,
    steel_dof_map=steel_dof_map,
    steel_dof_offset=steel_dof_offset,
    bond_states=bond_states,
)
```

Checks:
- Array shapes, dtypes, contiguity
- Node indices in bounds
- DOF indices valid for all segments
- No -1 (missing) steel DOFs in active segments

### A2. Forced dtypes/contiguity
All arrays converted to C-contiguous with correct dtypes before kernel.

### A3. Numba boundscheck
Enabled during debugging (`boundscheck=True`) to catch out-of-bounds access.

### A4. Python fallback
```python
f, K, states = assemble_bond_slip(..., use_numba=False)
```
Pure Python implementation with explicit bounds checks for debugging.

### A5. Minimal reproducer
See `tests/test_bond_hang_repro.py` for tests that exercise DOF change scenarios.

## Part B: Dissertation Model (COMPLETED)

### B1-B2: Local System and Transformations

The interface uses a local tangent-normal coordinate system:

```
w = [s, w]^T        Slip and opening
σ_local = [τ, σ_r]^T   Bond stress and radial stress
```

Transformation matrix W relates global displacements to local interface kinematics (Eq. 3.52-3.55).

### B3: Steel Yielding Reduction Ωy

Reduces bond strength when steel yields (Eq. 3.57-3.58):

```python
bond_law = BondSlipModelCode2010(
    f_cm=30e6,
    d_bar=0.016,
    f_y=500e6,         # Steel yield stress
    E_s=200e9,         # Steel Young's modulus
    enable_yielding_reduction=True,  # Enable Ωy
)
```

Formula:
```
eps_y = f_y / E_s
Ωy = exp(-k_y * (eps_s - eps_y) / eps_y)  if eps_s > eps_y
Ωy = 1                                     if eps_s ≤ eps_y
```

where k_y ≈ 10 (calibration parameter).

### B4: Crack Deterioration Ωcrack with FPZ

Reduces bond strength near transverse cracks, accounting for FPZ state (Eq. 3.60-3.61 modified):

```python
bond_law = BondSlipModelCode2010(
    ...,
    enable_crack_deterioration=True,  # Enable Ωcrack
)

# During assembly, provide crack info
omega_crack = bond_law.compute_crack_deterioration(
    dist_to_crack=0.05,          # Distance to nearest crack [m]
    w_max=1e-4,                   # Maximum crack opening [m]
    t_n_cohesive_stress=2e6,      # Cohesive stress t_n(w_max) [Pa]
    f_t=3e6,                      # Concrete tensile strength [Pa]
)
```

Formula:
```
l_ch = 2 * d_bar                    Characteristic length
chi = t_n(w_max) / f_t              FPZ state indicator [0, 1]
Ωcrack = exp(-dist / l_ch * (1 - chi))
```

When chi = 1 (crack closed): Ωcrack = 1 (no deterioration)
When chi = 0 (crack open): Ωcrack = exp(-dist/l_ch) (exponential decay)

### B5: Dowel Action σ_r(w)

Transverse stress-opening relationship (Brenna et al., Eq. 3.62-3.68):

```python
from xfem_clean.bond_slip import DowelActionModel

dowel = DowelActionModel(
    d_bar=0.016,
    f_c=30e6,      # Concrete compressive strength
    E_s=200e9,
)

sigma_r, dsigma_r_dw = dowel.sigma_r_and_tangent(w=1e-4)
```

Formula:
```
σ_r_max = k_c * sqrt(f_c)           Maximum radial stress
σ_r(w) = σ_r_max * (1 - exp(-k_d * w / d_bar))
```

where k_c ≈ 0.8, k_d ≈ 50 (calibration constants).

### B6-B7: Interface Assembly with Secant Stiffness

The bond stress is integrated over each rebar segment (Eq. 4.107-4.116):

```
f_bond = Σ_segments ∫_Γ N^T W^T A σ_local dΓ
K_bond = Σ_segments ∫_Γ N^T W^T A D_secant W N dΓ
```

**Secant stiffness for stability** (Eq. 5.2-5.3):

```python
bond_law = BondSlipModelCode2010(
    ...,
    use_secant_stiffness=True,  # Use secant instead of tangent
)
```

This improves Newton convergence by using:
```
D_secant = σ(s) / s    instead of    D_tangent = dσ/ds
```

### B8: Enriched DOF Contributions

When cracks grow, enriched DOFs (Heaviside H and tip functions) are added. The interface displacement must include these contributions:

```
u_interface = N * u_std + H * u_H + Σ_α F_α * u_tip_α
```

The steel DOF map (`dofs.steel`) is updated when crack grows to reflect new DOF numbering.

## Usage Example

```python
from xfem_clean.bond_slip import BondSlipModelCode2010, assemble_bond_slip, BondSlipStateArrays

# Create bond law with dissertation features
bond_law = BondSlipModelCode2010(
    f_cm=30e6,                          # Concrete strength
    d_bar=0.016,                        # Bar diameter
    condition="good",                   # Bond condition
    f_y=500e6,                          # Steel yield stress
    E_s=200e9,                          # Steel modulus
    use_secant_stiffness=True,          # B7: Secant for stability
    enable_yielding_reduction=True,     # B3: Steel yielding Ωy
    enable_crack_deterioration=True,    # B4: Crack deterioration Ωcrack
)

# Initialize state
n_seg = steel_segments.shape[0]
bond_states = BondSlipStateArrays.zeros(n_seg)

# Assemble (with Part A validation enabled)
f_bond, K_bond, bond_states_new = assemble_bond_slip(
    u_total=u_total,
    steel_segments=steel_segments,
    steel_dof_offset=dofs.steel_dof_offset,
    bond_law=bond_law,
    bond_states=bond_states,
    steel_dof_map=dofs.steel,           # B8: Updated after crack growth
    steel_EA=E_s * A_bar,               # Optional axial stiffness
    use_numba=True,
    enable_validation=True,             # A1: Preflight checks
)
```

## Implementation Status

✅ **Part A (Debug/Hardening)**: COMPLETE
- Validation, dtype forcing, boundscheck, Python fallback, tests

✅ **Part B (Dissertation Model)**: COMPLETE
- B1-B2: Local transformations (documented)
- B3: Steel yielding reduction Ωy
- B4: Crack deterioration Ωcrack with FPZ
- B5: Dowel action model
- B6-B7: Secant stiffness option
- B8: Enriched DOF handling (via steel_dof_map)

## Testing

Run minimal reproducer:
```bash
python tests/test_bond_hang_repro.py
```

Run pull-out validation (when available):
```bash
python examples/validation_bond_slip_pullout.py
```

## Acceptance Criteria

### Part A
- ✅ Never hangs (converts to exception)
- ✅ Boundscheck detects OOB
- ✅ Validation catches invalid DOF maps
- ✅ Python fallback works

### Part B
- ⏳ Pull-out test matches analytical curve
- ⏳ No divergences with crack growth
- ⏳ Newton converges with secant stiffness
- ⏳ Ωy reduces τ when steel yields
- ⏳ Ωcrack reduces τ near cracks

## References

- Dissertation: 10.5445/IR/1000124842
- Model Code 2010: fib, Section 6.1.2
- Brenna et al.: Dowel action model

## Related Files

- `src/xfem_clean/bond_slip.py` - Main implementation
- `tests/test_bond_hang_repro.py` - Reproducer tests
- `BOND_SLIP_HANG_DIAGNOSTIC.md` - Hang diagnosis
- `docs/INTEGRATION_NOTES.md` - Integration guide
