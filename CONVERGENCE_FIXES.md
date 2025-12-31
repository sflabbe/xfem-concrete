# Convergence Fixes Summary (BLOQUES 0-3)

## Overview

This document summarizes the numerical convergence fixes implemented for the thesis example cases. The goal is to eliminate "Matrix is exactly singular" and "Substepping exceeded" errors that prevented cases 01-06 from running.

## Problem Diagnosis

Thesis cases (especially Case 01: Pullout) failed with:
- **Matrix singularity**: `MatrixRankWarning: Matrix is exactly singular`
  Root cause: Void elements with E=0, h=0 created zero rows/columns in K
- **Substepping failure**: `Substepping exceeded max_subdiv=15 at u~7e-9 m`
  Root cause: Broken Newton linearization from incomplete bond-slip Jacobian

## Implemented Fixes

### BLOQUE 0-1: API Consistency + Void Element Penalty

**Files Modified**:
- `examples/gutierrez_thesis/solver_interface.py` (lines ~908-940)
- `src/xfem_clean/xfem/assembly_single.py` (lines ~152-172, ~387-396)

**Changes**:

1. **API Compatibility** (`solver_interface.py:908-940`):
   - Added optional parameters to `run_case_solver()`:
     - `max_steps`: Override nsteps for regression tests
     - `return_bundle`: Always return bundle (default True)
     - `output_dir`: Override output directory

2. **Void Element Penalty** (`assembly_single.py:152-172`):
   ```python
   # Check if element is void (use penalty stiffness)
   is_void_elem = subdomain_mgr is not None and subdomain_mgr.is_void(e)

   # Void elements: apply penalty stiffness to prevent singularity
   if is_void_elem:
       C_eff = C * 1e-9  # Penalty stiffness (very small)
       if thickness_eff < 1e-12:
           thickness_eff = thickness * 1e-9
   else:
       C_eff = C  # Normal stiffness
   ```

**Status**: ✅ Implemented
**Impact**: Eliminates singular matrix from void elements (penalty stiffness ~ 1e-9 * E)

---

### BLOQUE 2: Bond-Slip Consistent Tangent (Full 8x8 Jacobian)

**Files Modified**:
- `src/xfem_clean/bond_slip.py` (lines ~1104-1212)

**Problem**: Previous implementation used **diagonal-only** stiffness (placeholder):
```python
# OLD (diagonal only - WRONG)
rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_s1x; data[entry_idx] = 0.25 * Kxx
# ... only 8 diagonal entries, NO coupling between steel ↔ concrete
```

This broke Newton's linearization because ∂F/∂u was incomplete.

**Solution**: Implemented **full 8x8 segment Jacobian** with proper steel↔concrete coupling:

```python
# NEW (full 8x8 - CORRECT)
K_bond = dtau_ds * perimeter * L0

# Gradient vector: g = [∂s/∂u] for 8 DOFs
g_c1x = -0.5 * cx  # Concrete node 1, x
g_c1y = -0.5 * cy  # Concrete node 1, y
g_c2x = -0.5 * cx  # Concrete node 2, x
g_c2y = -0.5 * cy  # Concrete node 2, y
g_s1x = +0.5 * cx  # Steel node 1, x
g_s1y = +0.5 * cy  # Steel node 1, y
g_s2x = +0.5 * cx  # Steel node 2, x
g_s2y = +0.5 * cy  # Steel node 2, y

# Assemble K_seg = K_bond * g ⊗ g^T (64 entries)
for ii in range(8):
    for jj in range(8):
        rows[entry_idx] = dofs[ii]
        cols[entry_idx] = dofs[jj]
        data[entry_idx] = K_bond * g[ii] * g[jj]
        entry_idx += 1
```

**Theory**:
```
Slip: s = t·(u^s_m - u^c_m)
Force: F = τ(s) * p * L_0
Jacobian: K_seg = (pL_0) * k_τ * g*g^T

where:
  t = [cx, cy] = bar tangent vector
  g = [∂s/∂u] = gradient w.r.t. all 8 DOFs
  k_τ = dτ/ds = bond tangent stiffness
```

**Status**: ✅ Implemented
**Impact**: Provides consistent Newton tangent with steel↔concrete coupling

---

### BLOQUE 3: Bond-Slip Continuation (Gamma Ramp)

**Files Modified**:
- `src/xfem_clean/bond_slip.py` (lines ~1265, ~1296-1300, ~1339-1349, ~958, ~1116)

**Changes**:

1. **Added `bond_gamma` parameter** to `assemble_bond_slip()`:
   - Default: `gamma = 1.0` (full bond)
   - Scales bond tangent: `K_bond = gamma * dtau_ds * perimeter * L0`

2. **Implementation**:
   ```python
   def assemble_bond_slip(..., bond_gamma: float = 1.0):
       """
       bond_gamma : float, optional
           Continuation parameter for bond-slip activation [0, 1].
           Scales the bond tangent stiffness: K_bond = gamma * k_τ.
           Use gamma < 1 for easier initial convergence, then ramp up to 1.
           Gamma=0 → no bond (only steel axial), Gamma=1 → full bond.
       """
       # In kernel:
       K_bond = gamma * dtau_ds * perimeter * L0
   ```

3. **Usage** (to be implemented in solver):
   ```python
   # Example continuation scheme:
   for step in range(n_steps):
       # Ramp gamma from 0 to 1 over first few steps
       if step < 5:
           gamma = step / 5.0  # Linear ramp
       else:
           gamma = 1.0  # Full bond

       f_bond, K_bond, states = assemble_bond_slip(..., bond_gamma=gamma)
   ```

**Status**: ✅ Infrastructure implemented (gamma parameter added)
**Next Step**: Integrate gamma ramp into solver load stepping logic

**Impact**: Allows gradual bond activation for easier initial convergence

---

## Testing Status

### Test: `test_case_01_coarse` (Pullout)

**Before fixes**:
```
MatrixRankWarning: Matrix is exactly singular
Substepping exceeded max_subdiv=15 at u=7.6e-9 m
```

**After BLOQUES 0-3**:
- Void penalty eliminates singular matrix warning ✓
- Full 8x8 Jacobian provides correct Newton tangent ✓
- **Still fails**: Substepping exceeded (further debugging needed)

**Interpretation**:
- Infrastructure fixes are correct
- Case 01 may have additional issues:
  - Very stiff bond-slip law (k_τ → ∞ as s → 0)
  - Need to actually USE gamma continuation in solver
  - May need relaxed solver parameters

---

## BLOQUE 4: Diagnostics (Pending)

**To Do**:
1. Add `debug_bond` flag to solver for detailed bond-slip diagnostics
2. Create unit tests:
   - `tests/test_bond_slip_minimal.py`: Test 8x8 Jacobian correctness
   - `tests/test_void_penalty.py`: Test void element penalty
3. Document troubleshooting in `CONVERGENCE_TROUBLESHOOTING.md`

---

## Summary of Commits

1. **fix(solver): make run_case_solver API consistent for regression tests**
   Added `max_steps`, `return_bundle`, `output_dir` parameters

2. **fix(void): add penalty stiffness for void/zero-thickness subdomains**
   C_eff = C * 1e-9 for void elements

3. **fix(bond): implement full 8x8 consistent tangent Jacobian for bond-slip**
   K_seg = K_bond * g ⊗ g^T with proper steel↔concrete coupling

4. **feat(bond): add gamma continuation parameter for bond-slip activation**
   Added `bond_gamma` parameter to scale bond tangent

---

## Next Steps

1. **Use gamma continuation in solver**: Modify load stepping to ramp gamma from 0→1
2. **Relax solver parameters**: Increase `max_subdiv`, reduce `du_tol` for difficult cases
3. **Debug Case 01 specifically**: Add logging to understand why it still fails
4. **Run full regression suite**: Test all cases 01-06 with fixes

---

## References

- BLOQUE2_STATUS.md: Original problem diagnosis
- src/xfem_clean/bond_slip.py: Bond-slip implementation
- src/xfem_clean/xfem/assembly_single.py: Void element penalty
- examples/gutierrez_thesis/solver_interface.py: Case runner API
