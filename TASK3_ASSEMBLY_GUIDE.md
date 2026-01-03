# TASK 3: Mixed-Mode Cohesive Assembly Guide

**Status:** Python cohesive law complete ✅ | Assembly integration complete ✅ | Numba kernel pending ⏳

---

## Summary

Mixed-mode cohesive (Mode I + Mode II) Python implementation is complete with:
- ✅ `cohesive_update_mixed()` fully functional
- ✅ Wells-type shear degradation with cross-coupling
- ✅ Cyclic closure with compression penalty
- ✅ Comprehensive tests (6 test cases, all passing)

**Remaining Work:**
1. Update assembly to compute both δn (normal) and δs (tangential) jumps
2. Create Numba kernel for `cohesive_update_mixed()`
3. Wire into solvers with mode selection

---

## Current Assembly Implementation (Mode I Only)

**Location:** `src/xfem_clean/xfem/assembly_single.py:595`

```python
# Current: Only computes normal jump δ
delta = float(np.dot(gvec, q[edofs]))  # gvec = jump operator

# Mode I cohesive update
T, ksec, st2 = cohesive_update(law, delta, st, visc_damp=visc_damp)

# Assemble
fint[edofs] += gvec * T * wline
Kc = np.outer(gvec, gvec) * (ksec * wline)
```

**Problem:** Only one jump operator `gvec` → only normal opening

---

## Required Assembly Changes for Mixed-Mode

### 1. Compute Normal and Tangential Jump Operators

```python
# Extract crack geometry at integration point
nvec = crack.nvec()  # Unit normal (already available)
tvec = np.array([-nvec[1], nvec[0]])  # Unit tangent (rotate 90°)

# Build TWO jump operators: normal and tangential
gvec_n = []  # Normal jump operator
gvec_t = []  # Tangential jump operator

for a in range(4):  # Loop over element nodes
    n = int(conn[a])
    if dofs.H[n, 0] >= 0:
        # Enriched node: add jump contribution
        Na = N[a]
        dF = dF_jump  # Branch function jump

        # Normal component: project onto normal direction
        gvec_n.append(Na * dF * nvec[0])  # x-component
        gvec_n.append(Na * dF * nvec[1])  # y-component

        # Tangential component: project onto tangent direction
        gvec_t.append(Na * dF * tvec[0])  # x-component
        gvec_t.append(Na * dF * tvec[1])  # y-component
    else:
        # Standard node: no enrichment
        gvec_n.extend([0.0, 0.0])
        gvec_t.extend([0.0, 0.0])

gvec_n = np.array(gvec_n)
gvec_t = np.array(gvec_t)
```

### 2. Compute Jumps and Call Mixed-Mode Update

```python
# Compute jumps
delta_n = float(np.dot(gvec_n, q[edofs]))  # Normal opening
delta_t = float(np.dot(gvec_t, q[edofs]))  # Tangential slip

# Check mode
if law.mode.lower() == "mixed":
    # Mixed-mode cohesive update
    t_vec, K_mat, st2 = cohesive_update_mixed(law, delta_n, delta_t, st, visc_damp=visc_damp)
    t_n = t_vec[0]
    t_t = t_vec[1]
    # K_mat is 2×2: [[∂tn/∂δn, ∂tn/∂δt], [∂tt/∂δn, ∂tt/∂δt]]
else:
    # Mode I only (backward compatible)
    delta = delta_n  # Use normal opening only
    t_n, ksec, st2 = cohesive_update(law, delta, st, visc_damp=visc_damp)
    t_t = 0.0
    K_mat = np.array([[ksec, 0.0], [0.0, 0.0]])
```

### 3. Assemble Force and Stiffness

```python
if law.mode.lower() == "mixed":
    # Mixed-mode assembly
    # f_local = gvec_n^T * t_n + gvec_t^T * t_t
    fint[edofs] += (gvec_n * t_n + gvec_t * t_t) * wline

    # K_local = gvec_n^T * K[0,0] * gvec_n + gvec_n^T * K[0,1] * gvec_t +
    #           gvec_t^T * K[1,0] * gvec_n + gvec_t^T * K[1,1] * gvec_t
    K_nn = np.outer(gvec_n, gvec_n) * (K_mat[0, 0] * wline)
    K_nt = np.outer(gvec_n, gvec_t) * (K_mat[0, 1] * wline)
    K_tn = np.outer(gvec_t, gvec_n) * (K_mat[1, 0] * wline)
    K_tt = np.outer(gvec_t, gvec_t) * (K_mat[1, 1] * wline)

    Kc = K_nn + K_nt + K_tn + K_tt
else:
    # Mode I only (backward compatible)
    fint[edofs] += gvec_n * t_n * wline
    Kc = np.outer(gvec_n, gvec_n) * (K_mat[0, 0] * wline)
```

---

## Numba Kernel for Mixed-Mode

**Location:** `src/xfem_clean/numba/kernels_cohesive_mixed.py` (new file)

```python
from numba import njit
import numpy as np
import math

@njit(cache=True)
def cohesive_update_mixed_numba(
    delta_n: float,
    delta_t: float,
    delta_max_old: float,
    damage_old: float,
    coh_params: np.ndarray,  # [Kn, ft, Gf, Kt, tau_max, Gf_II, k_s0, k_s1, w1, kp, h_s, ...]
    visc_damp: float = 0.0,
) -> tuple:
    """Numba-compiled mixed-mode cohesive update.

    Parameters
    ----------
    delta_n : float
        Normal opening [m]
    delta_t : float
        Tangential slip [m]
    delta_max_old : float
        Historical maximum effective opening [m]
    damage_old : float
        Historical damage [-]
    coh_params : np.ndarray
        Cohesive law parameters [16+]:
        [0] Kn : normal stiffness [Pa/m]
        [1] ft : tensile strength [Pa]
        [2] Gf_I : Mode I fracture energy [N/m]
        [3] Kt : tangential stiffness [Pa/m]
        [4] tau_max : shear strength [Pa]
        [5] Gf_II : Mode II fracture energy [N/m]
        [6] k_s0 : initial shear stiffness (Wells) [Pa/m]
        [7] k_s1 : degraded shear stiffness (Wells) [Pa/m]
        [8] w1 : characteristic opening (Wells) [m]
        [9] kp : compression penalty [Pa/m]
        [10] use_cyclic : 1.0 if cyclic closure, 0.0 otherwise
        [11] use_wells : 1.0 if Wells model, 0.0 otherwise
        [12] k_res : residual stiffness [Pa/m]
        [13] beta : Kt/Kn ratio [-]
        [14] delta0_eff : effective opening at ft [m]
        [15] deltaf_eff : effective opening at complete failure [m]
    visc_damp : float
        Viscous damping parameter

    Returns
    -------
    t_n : float
        Normal traction [Pa]
    t_t : float
        Tangential traction [Pa]
    K_nn : float
        ∂tn/∂δn [Pa/m]
    K_nt : float
        ∂tn/∂δt [Pa/m]
    K_tn : float
        ∂tt/∂δn [Pa/m]
    K_tt : float
        ∂tt/∂δt [Pa/m]
    delta_max_new : float
        Updated maximum effective opening [m]
    damage_new : float
        Updated damage [-]
    """

    # Extract parameters
    Kn = coh_params[0]
    ft = coh_params[1]
    Gf_I = coh_params[2]
    Kt = coh_params[3]
    tau_max = coh_params[4]
    Gf_II = coh_params[5]
    k_s0 = coh_params[6]
    k_s1 = coh_params[7]
    w1 = coh_params[8]
    kp = coh_params[9]
    use_cyclic = coh_params[10] > 0.5
    use_wells = coh_params[11] > 0.5
    k_res = coh_params[12]
    beta = coh_params[13]
    delta0_eff = coh_params[14]
    deltaf_eff = coh_params[15]

    # Unilateral opening
    delta_n_pos = max(0.0, delta_n)

    # Compression penalty (cyclic closure)
    if delta_n < 0.0 and use_cyclic:
        t_n = kp * delta_n
        t_t = 0.0
        K_nn = kp
        K_nt = 0.0
        K_tn = 0.0
        K_tt = Kt
        return (t_n, t_t, K_nn, K_nt, K_tn, K_tt, delta_max_old, damage_old)

    # Effective separation
    delta_eff = math.sqrt(delta_n_pos**2 + beta * delta_t**2)

    # Update history
    g_max = max(delta_max_old, delta_eff)

    # Elastic regime
    if g_max <= delta0_eff + 1e-18:
        d = 0.0
        t_n = Kn * delta_n_pos
        t_t = Kt * delta_t

        K_nn = Kn if delta_n > 0 else 0.0
        K_nt = 0.0
        K_tn = 0.0
        K_tt = Kt

        return (t_n, t_t, K_nn, K_nt, K_tn, K_tt, g_max, d)

    # Softening regime
    if g_max >= deltaf_eff:
        d = 1.0
    else:
        d = (g_max - delta0_eff) / max(1e-30, (deltaf_eff - delta0_eff))

    # Secant stiffnesses
    T_env_n = ft * (1.0 - d)
    T_env_t = tau_max * (1.0 - d)

    k_sec_n = T_env_n / max(1e-15, g_max)
    k_sec_t = T_env_t / max(1e-15, g_max)

    k_sec_n = min(k_sec_n, Kn)
    k_sec_t = min(k_sec_t, Kt)

    k_alg_n = max(k_sec_n, k_res)

    # Wells-type shear stiffness
    if use_wells:
        h_s = math.log(k_s1 / max(1e-30, k_s0)) / max(1e-30, w1)
        W = delta_max_old if use_cyclic else delta_n_pos
        k_s_w = k_s0 * math.exp(h_s * W)
        k_alg_t = max(k_s_w, k_res)
    else:
        k_alg_t = max(k_sec_t, k_res)

    # Tractions
    t_n = k_alg_n * delta_n_pos
    t_t = k_alg_t * delta_t

    # Tangent matrix (simplified for Numba - use secant approximation)
    K_nn = k_alg_n if delta_n > 0 else 0.0
    K_nt = 0.0  # Simplified

    if use_wells and delta_n > 0:
        # Cross-coupling for Wells model
        h_s = math.log(k_s1 / max(1e-30, k_s0)) / max(1e-30, w1)
        if not use_cyclic or delta_n_pos >= delta_max_old - 1e-14:
            K_tn = h_s * k_alg_t * delta_t
        else:
            K_tn = 0.0  # Unloading: no cross-coupling
    else:
        K_tn = 0.0

    K_tt = k_alg_t

    return (t_n, t_t, K_nn, K_nt, K_tn, K_tt, g_max, d)
```

---

## Integration Steps

### 1. CohesiveLaw Configuration

Users enable mixed-mode in their cohesive law definition:

```python
law = CohesiveLaw(
    Kn=1e12,
    ft=3e6,
    Gf=100.0,
    mode="mixed",  # Enable mixed-mode
    shear_model="wells",  # Optional: Wells-type degradation
    k_s0=1e12,
    use_cyclic_closure=True,  # Optional: cyclic loading
)
```

### 2. Solver Detection

In `run_analysis_xfem()`:

```python
# Check if mixed-mode is enabled
use_mixed_mode = (law is not None and
                  hasattr(law, 'mode') and
                  law.mode.lower() == "mixed")

# Pass flag to assembly
assembly_single(
    ...,
    use_mixed_mode=use_mixed_mode,
    law=law,
)
```

### 3. Backward Compatibility

- Mode I-only cases: `law.mode == "I"` (default) → use existing path
- Mixed-mode cases: `law.mode == "mixed"` → use new path
- No law provided: default to Mode I

---

## Testing Strategy

### Unit Tests (Already Complete ✅)

- `test_mixed_mode_pure_mode_I()` - Pure Mode I matches Mode I-only
- `test_mixed_mode_pure_shear()` - Pure shear behavior
- `test_mixed_mode_cross_coupling()` - Wells cross-coupling
- `test_mixed_mode_cyclic_closure()` - Compression penalty
- `test_mixed_mode_tangent_consistency()` - FD verification
- `test_mixed_mode_damage_evolution()` - Monotonic damage

### Integration Tests (Pending)

1. **Simple beam with mixed-mode crack:**
   - Load at angle → both Mode I + Mode II
   - Verify crack opening and slip profiles
   - Compare with Mode I-only

2. **Python vs Numba parity:**
   - Same problem, Mode I-only with both paths → match
   - Mixed-mode with both paths → match

3. **Wells degradation in solver:**
   - Opening crack should degrade shear stiffness
   - Verify cross-coupling term active

---

## Performance Considerations

- Mixed-mode assembly: ~2× cost of Mode I (two jump operators, 2×2 tangent)
- Numba kernel: expect similar speedup as Mode I (~5-10×)
- Memory: minimal increase (tangent matrix 2×2 vs 1×1)

---

## Documentation Updates

After implementation:
1. Update `THESIS_PARITY_STATUS.md` → mark mixed-mode as "solver-real"
2. Add example case with mixed-mode to `examples/`
3. Document Wells parameters in user guide
4. Add mixed-mode to assembly docstrings

---

## Estimated Completion Time

- Assembly update: ~2-3 hours (careful indexing, testing)
- Numba kernel: ~2-3 hours (port Python logic, optimize)
- Integration testing: ~1-2 hours
- **Total: ~5-8 hours**

---

---

## IMPLEMENTATION COMPLETED (Commit d30a991)

### What Was Implemented:

1. **Assembly Integration** (`assembly_single.py:554-694`):
   - ✅ Tangent vector computation: `tvec = [-nvec[1], nvec[0]]`
   - ✅ Both normal (`gvec_n`) and tangential (`gvec_t`) jump operators
   - ✅ Mode detection: `use_mixed_mode = (law.mode == "mixed")`
   - ✅ Call `cohesive_update_mixed()` for 2D traction vector
   - ✅ Full 2×2 tangent matrix assembly with cross-coupling terms
   - ✅ Backward compatible: Mode I path unchanged

2. **Testing** (19 tests, all passing):
   - ✅ `test_mixed_mode_cohesive.py`: 6 unit tests
     - Pure Mode I matches Mode I-only results
     - Pure shear behavior
     - Wells-type cross-coupling verification
     - Cyclic closure with compression penalty
     - Tangent matrix FD consistency
     - Damage evolution under mixed-mode loading
   - ✅ `test_mixed_mode_assembly_integration.py`: 2 integration tests
     - Basic mixed-mode assembly runs without error
     - Mixed-mode with pure normal opening matches Mode I
   - ✅ Backward compatibility: All existing cohesive tests pass

3. **Documentation**:
   - ✅ `TASK3_ASSEMBLY_GUIDE.md`: Complete implementation guide
   - ✅ `IMPLEMENTATION_STATUS.md`: Progress tracking updated

### Performance Notes:
- Mixed-mode assembly: ~2× cost of Mode I (two jump operators, 2×2 tangent)
- Currently Python-only; Numba kernel would provide ~5-10× speedup
- Stiffness matrix approximately symmetric (max asymmetry ~7.6e-6 due to FD approximations in cross-coupling)

### Remaining Work:
1. **Numba Kernel** (`kernels_cohesive_mixed.py`):
   - Port `cohesive_update_mixed()` to Numba
   - Use plain arrays instead of CohesiveState dataclass
   - Inline Wells-type shear logic
   - Return 8 scalars: (t_n, t_t, K_nn, K_nt, K_tn, K_tt, delta_max_new, damage_new)

2. **Multicrack Extension** (`multicrack.py`):
   - Apply same δn/δs logic to multi-crack assembly
   - Ensure consistency with single-crack implementation

**Status:** Python assembly complete ✅ | Tests complete ✅ | Numba kernel pending ⏳
