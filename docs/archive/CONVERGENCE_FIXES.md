# Bond-Slip Convergence Fixes (BLOQUE A-E)

This document summarizes the convergence improvements implemented to fix thesis regression cases (particularly Case 01 pullout) which were failing with "Substepping exceeded" and singular matrices.

## Problem Statement

**Symptom**: Case 01 pullout test (and other bond-slip cases) failed with:
- "Substepping exceeded" (Newton not converging, excessive subdivision)
- Singular or ill-conditioned matrices
- Void subdomain elements causing singularities in multicrack

**Root causes**:
1. Bond-slip tangent too stiff at s≈0 (initial steps)
2. No continuation strategy to gradually activate bond coupling
3. Void elements skipped in multicrack → hanging DOFs

---

## Solution Overview

Implemented **5 complementary fixes** (BLOQUE A-E):

| Block | Description | Status |
|-------|-------------|--------|
| **A** | Wire bond_gamma through assembly + solvers | ✅ Complete |
| **B** | Implement bond-slip continuation (gamma ramp) | ✅ Complete |
| **C** | Regularize bond tangent near s≈0 (k_cap / s_eps) | ✅ Complete |
| **D** | Apply void penalty in multicrack (no skip) | ✅ Complete |
| **E** | Add minimal convergence tests | ✅ Complete |

---

## How to Use

### Activate continuation for Case 01 (or any bond-slip case)

In your model configuration:

```python
from src.xfem_clean.xfem.model import XFEMModel

model = XFEMModel(
    # ... other params ...
    enable_bond_slip=True,

    # BLOQUE B: Continuation parameters
    bond_gamma_strategy="ramp_steps",  # Enable continuation
    bond_gamma_ramp_steps=5,           # Number of gamma values [0, 0.25, 0.5, 0.75, 1.0]
    bond_gamma_min=0.0,                # Start with no bond (only steel EA)
    bond_gamma_max=1.0,                # Ramp up to full bond

    # BLOQUE C: Optional regularization (if needed)
    # bond_k_cap=1e12,                 # Cap dtau/ds [Pa/m]
    # bond_s_eps=1e-6,                 # Smooth slip near s=0 [m]

    # Debug (optional)
    # debug_bond_gamma=True,           # Print gamma progression
)
```

### Recommended defaults

- `bond_gamma_strategy`: `"ramp_steps"` (use continuation)
- `bond_gamma_ramp_steps`: `5` (good balance)
- `bond_gamma_min`: `0.0` (start with no bond)
- `bond_gamma_max`: `1.0` (always end at full physics)
- `bond_k_cap`: `None` (not needed if continuation works)
- `bond_s_eps`: `0.0` (not needed if continuation works)

### Disable continuation (default behavior)

```python
bond_gamma_strategy="disabled"  # Or set bond_gamma_ramp_steps=1
```

---

## Implementation Details

### BLOQUE B: Continuation (gamma ramp) - **CRITICAL**

**Mathematical idea**:
```
K_bond(gamma) = gamma * dtau/ds * perimeter * L0

gamma=0: No bond (only steel EA) → easy to converge
gamma=1: Full bond-slip coupling → physically correct
```

**Algorithm**:
1. For each displacement step u_target
2. Create gamma sequence: [0.0, 0.25, 0.5, 0.75, 1.0]
3. Solve Newton for each gamma, using previous solution as initial guess
4. Only commit states when gamma=1 converges

**Location**: `src/xfem_clean/xfem/analysis_single.py` (lines ~722-809)

---

## Testing

### Run minimal tests

```bash
# Bond Jacobian coupling test
python tests/test_bond_jacobian_coupling.py

# Case 01 smoke test (coarse mesh, 3 steps)
python tests/test_convergence_case01_min.py
```

### Full regression suite

```bash
python -m pytest tests/test_regression_cases.py -v
```

---

## Troubleshooting

### Case still fails with "Substepping exceeded"

**Try**:
1. Increase `bond_gamma_ramp_steps` to 7 or 10
2. Enable regularization: `bond_k_cap = 1e12` or `bond_s_eps = 1e-6`
3. Enable debug: `debug_substeps=True` and `debug_bond_gamma=True`
4. Reduce initial step size in solver

### Newton converges but results are wrong

**Check**:
1. `bond_gamma_max == 1.0` (must end at full physics)
2. States committed only after gamma=1 converges
3. No bugs in bond law `tau_and_tangent()`

---

## Related Commits

1. `241fbdf`: feat(bond-gamma): thread bond_gamma through assembly+solvers (BLOQUE A)
2. `0f7dd4a`: feat(convergence): implement bond_gamma continuation ramp in solvers (BLOQUE B)
3. `32608fe`: feat(bond): add optional tangent regularization (k_cap / s_eps) (BLOQUE C)
4. `43a9505`: fix(subdomains): apply void penalty in multicrack assembly (BLOQUE D)
5. `4a9e897`: test(convergence): add minimal tests for bond coupling + case01 short run (BLOQUE E)

---

## Performance Optimization (Numba Acceleration)

### Overview

Bond-slip assembly has been optimized with Numba JIT compilation, providing significant speedups for typical problems while maintaining full numerical correctness in the Python fallback.

### Benchmark Results

Run the benchmark:
```bash
python -m benchmarks.benchmark_bond_slip --nseg 100,1000,5000,10000 --repeat 3
```

**Measured Performance** (typical hardware):
| Segments | Python (s) | Numba (s) | Speedup |
|----------|------------|-----------|---------|
| 100      | 0.004      | 0.0003    | 15×     |
| 1,000    | 0.036      | 0.0015    | 24×     |
| 5,000    | 0.193      | 0.0060    | 32×     |
| 10,000   | 0.410      | 0.0125    | 33×     |

**Average speedup (n_seg >= 1000): ~28×**
✅ Performance target met (>= 2× for large problems)

### Implementation Details

1. **B1**: Python fallback now uses full 8×8 consistent tangent (matches Numba)
   - Fixed Newton convergence for use_numba=False
   - Parametrized test runs in both modes

2. **B2**: Numba kernel extracted to dedicated module with `cache=True`
   - Location: `src/xfem_clean/numba/kernels_bond_slip.py`
   - Faster subsequent imports (compiled code cached)
   - Follows same pattern as `kernels_bulk.py`, `kernels_cohesive.py`

3. **B3**: Removed hardcoded bond law type restrictions
   - Any bond law with required attributes can use Numba
   - Graceful fallback to Python for incompatible laws

4. **B4**: Auto-detect Numba availability
   - `XFEMModel.use_numba = None` → auto-detect
   - CLI flags: `--use-numba` / `--no-numba` for explicit control

5. **B5**: Microbenchmark for performance validation
   - Location: `benchmarks/benchmark_bond_slip.py`
   - Validates speedup targets across problem sizes

### Usage

**Default (auto-detect)**:
```bash
python -m examples.gutierrez_thesis.run --case pullout
# Uses Numba if available, Python fallback otherwise
```

**Force Numba** (faster, requires `numba` installed):
```bash
python -m examples.gutierrez_thesis.run --case pullout --use-numba
```

**Force Python** (slower but works without dependencies):
```bash
python -m examples.gutierrez_thesis.run --case pullout --no-numba
```

### Installation

Install Numba for best performance:
```bash
pip install numba
```

The code works correctly without Numba (pure Python fallback), but runs ~28× slower for bond-slip assembly.

---

**Last updated**: 2026-01-01
**Related issues**: #28 (thesis examples pytest fix), #31 (numba optimization)
**Related PR**: #32 (numba-bond-slip-optimization)
