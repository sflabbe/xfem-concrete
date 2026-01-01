# Physics & Formulation Audit Findings
## XFEM Concrete Fracture Repository

**Date:** 2026-01-01
**Auditor:** Claude Code (Automated)
**Scope:** Full repository audit against self-contained model specification

---

## Executive Summary

**Overall Status:** ⚠️ **PARTIALLY CORRECT** - Critical physics bugs found

**Critical Issues Found:** 3 high-priority bugs that affect numerical correctness
**Medium Issues:** 2 formulation gaps (features exist but incomplete)
**Good Practices:** History management, bond-slip coupling, junction detection

### Priority Classification
- **P0 (CRITICAL)**: Must fix - causes incorrect physics/divergence
- **P1 (HIGH)**: Should fix - missing documented features
- **P2 (MEDIUM)**: Nice to have - improves capability
- **P3 (LOW)**: Optional enhancements

---

## P0: CRITICAL ISSUES (Must Fix)

### P0.1: Cohesive Law - Missing Unilateral Opening ❌

**Location:** `src/xfem_clean/cohesive_laws.py:105`, `src/xfem_clean/numba/kernels_cohesive.py:105`

**Model Spec Requirement:**
```
δ_n_pos = max(δ_n, 0)   # Only opening contributes to damage
t_n = (1-d) * K_n * δ_n_pos
```

**Current Implementation:**
```python
delta_abs = abs(delta)  # Line 105 in both Python and Numba
dm = max(dm_old, delta_abs)  # Compression contributes to damage!
```

**Why This is Wrong:**
1. **Compression contributes to damage**: When crack closes (δ < 0), `abs(delta)` makes it positive, increasing `delta_max` and damage
2. **Cohesive traction resists closing**: The function returns non-zero traction for compression, but cohesive zones should only resist opening
3. **Violates physical principle**: Cracks don't "heal" when compressed - they simply close with no cohesive resistance

**Impact:**
- Incorrect damage accumulation during cyclic loading
- Non-physical cohesive tractions during crack closure
- Can cause Newton convergence issues

**Minimal Safe Fix:**
```python
# In cohesive_update:
delta_pos = max(0.0, delta)  # Only positive opening
delta_abs = abs(delta_pos)   # Should be just delta_pos
dm = max(dm_old, delta_abs)

# Traction should be zero for compression
if delta <= 0.0:
    return 0.0, k_res, CohesiveState(delta_max=dm_old, damage=st.damage)
```

**Test Required:** `test_cohesive_unilateral_opening.py`
- Apply compression (δ < 0) → verify t = 0 and no damage
- Cycle: open → close → reopen → verify damage only from opening

---

### P0.2: Stagnation Detection - False Failures ❌

**Location:** `src/xfem_clean/xfem/analysis_single.py:519-522`

**Model Spec Requirement:**
```
DO NOT fail solely on tiny ||ΔU|| if ||R|| is converging.
Fail if residual reduction over last M iterations < r_min AND not converged.
```

**Current Implementation:**
```python
if norm_du < model.newton_tol_du:  # Line from convergence.py:19
    print(f"stagnated it={it+1:02d} ||du||={norm_du:.3e}")
    return False, q, coh_committed, ...
```

**Why This is Wrong:**
1. **Only checks ||ΔU||**: Doesn't verify if residual is still decreasing
2. **No grace period**: Fails immediately on first small ΔU
3. **False failures on low loads**: Early in loading, ||ΔU|| can be tiny even though Newton is healthy

**Impact:**
- Premature convergence failures
- Unnecessary substepping
- User frustration with "stagnation" that isn't real

**Minimal Safe Fix:**
```python
# Track residual history
if not hasattr(solve_step, 'r_hist'):
    solve_step.r_hist = []
solve_step.r_hist.append(norm_r)

# Only fail on stagnation if:
# 1. ΔU is small AND
# 2. Residual hasn't decreased significantly in last M iterations
if norm_du < model.newton_tol_du:
    M = 3
    if len(solve_step.r_hist) >= M:
        r_reduction = solve_step.r_hist[-M] / max(1e-30, norm_r)
        if r_reduction < 1.1:  # Less than 10% reduction in M steps
            print(f"stagnated (true)")
            return False, ...
    # Otherwise continue - might just be small step
```

**Test Required:** `test_convergence_stagnation_false_positive.py`
- Small load step with converging residual → should NOT stagnate
- Actual stagnation (flat residual) → should detect

---

### P0.3: Penalty Stiffness - No Caps ⚠️

**Location:** Multiple (cohesive `Kn`, bond `k_cap`, contact `k_p`)

**Model Spec Requirement:**
```
Apply caps to prevent ill-conditioning:
- Cohesive: kcap_factor (exists, default 1.0 - OK)
- Bond: k_cap parameter (exists in code)
- Contact: Should have max(k_p, E*1e3) or similar
```

**Current Status:**
- ✅ Cohesive: Has `kcap_factor` parameter (line 21 of cohesive_laws.py)
- ⚠️ Bond: Has `bond_k_cap` parameter but may not be enforced everywhere
- ❓ Contact: Need to verify penalty capping

**Impact:**
- Ill-conditioned tangent matrix → slow/failed Newton convergence
- Numerical precision loss

**Action:** Audit contact_rebar.py for penalty caps (lower priority than P0.1-P0.2)

---

## P1: HIGH PRIORITY (Missing Documented Features)

### P1.1: Dowel Action - Implemented but Not Wired ⚠️

**Location:**
- Implemented: `src/xfem_clean/bond_slip.py:672-740` (`DowelActionModel`)
- Functions: `compute_dowel_springs` (line 1383), `assemble_dowel_action` (line 1445)
- **NOT CALLED** anywhere in solvers

**Model Spec Requirement:**
```
Optional dowel action at crack-bar crossings:
w = (u_s - u_c)·n_cr
p = k_d * w (optionally unilateral)
Assembled like bond but along crack normal
```

**Current Status:**
- ✅ `DowelActionModel` class fully implemented with Brenna et al. equations (3.62-3.68)
- ✅ `sigma_r_and_tangent(w)` method provides stress-opening curve
- ❌ `assemble_dowel_action` exists but is **never called**
- ❌ No integration into crack insertion / assembly routines

**Impact:**
- Missing feature claimed in thesis/parity matrix
- Potential underprediction of crack resistance in cyclic wall cases

**Recommendation:**
**Option A (Safer):** Document as "not implemented" and remove from feature claims
**Option B:** Wire into multicrack solver:
1. Detect crack-bar intersections after crack insertion
2. Compute crack opening `w` at intersection
3. Call `assemble_dowel_action` to add springs
4. Add test case

**Action:** Choose safer Option A for now, defer Option B to future work

---

### P1.2: Junction/Coalescence - ALREADY WIRED! ✅

**Location:** `src/xfem_clean/xfem/multicrack.py:1611-1626`

**Status:** ✅ **IMPLEMENTED AND ACTIVE**

**Evidence:**
```python
# Line 1614
junctions = detect_crack_coalescence(cracks, nodes, elems, tol_merge=tol_merge)

if junctions:
    for junc in junctions:
        print(f"[junction] detected...")
        arrest_secondary_crack_at_junction(junc, cracks)
        print(f"[junction] arrested crack#{junc.secondary_crack_id+1}, added junction enrichment")
```

**Parity Matrix Status:** INCORRECT - claimed "not wired" but it IS active in multicrack solver

**Action:** Update parity matrix P0 status to ✅ PASS

---

## P2: MEDIUM PRIORITY (Feature Gaps)

### P2.1: Mixed-Mode Cohesive - Data Structure Exists, Logic Missing

**Location:** `src/xfem_clean/cohesive_laws.py:29-52`

**Model Spec Requirement:**
```
Effective separation:
δ_eff = sqrt( (δ_n_pos)² + β(δ_t)² )

Damage: d(δ_eff) with history g_max = max(g_old, δ_eff)

Tractions:
t_n = (1-d) K_n δ_n_pos
t_t = (1-d) K_t δ_t

Tangent (2x2 matrix with cross-coupling):
∂t_n/∂δ_t = - K_n δ_n_pos * (dd/dg) * (∂g/∂δ_t)
```

**Current Status:**
- ✅ Parameters exist: `mode`, `tau_max`, `Kt`, `Gf_II` (lines 29-36)
- ✅ Defaults set in `__post_init__` (lines 46-52)
- ❌ `cohesive_update` function **ignores mode** - only uses scalar `delta`
- ❌ No δ_t input, no effective separation calculation
- ❌ No 2x2 tangent matrix return

**Impact:**
- Mode I only (normal opening)
- Cannot model shear-dominated failure
- Missing thesis capability

**Minimal Safe Fix:**
Backward-compatible extension:
1. Add `cohesive_update_mixed(law, delta_n, delta_t, st)` new function
2. Keep existing `cohesive_update(law, delta, st)` for Mode I (backward compat)
3. Update assembly to call mixed version when `law.mode == "mixed"`
4. Return 2x2 tangent: `[[dtn_ddn, dtn_ddt], [dtt_ddn, dtt_ddt]]`

**Test Required:** `test_mixed_mode_cohesive_tangent.py`
- Pure Mode I (δ_t=0) → should match existing
- Pure Mode II (δ_n≤0, δ_t≠0) → shear softening
- Mixed (both) → cross-coupling in tangent

---

### P2.2: Compression Damage - ALREADY SELECTABLE! ✅

**Location:** `src/xfem_clean/xfem/material_factory.py:113-129`

**Status:** ✅ **IMPLEMENTED AND SELECTABLE**

**Evidence:**
```python
if bm == "compression-damage":
    # P1: Compression damage model per thesis Eq. (3.44-3.46)
    # Parabolic stress-strain up to peak, then constant plateau (no softening)
    f_c_mpa = float(model.fc) / 1e6
    ...
    return ConcreteCompressionModel(f_c=..., eps_c1=..., E_0=...)
```

**How to Use:**
```python
model.bulk_material = "compression-damage"
```

**Parity Matrix Status:** INCORRECT - claimed "not selectable" but option exists!

**Action:** Update parity matrix P1 status to ✅ PASS

---

## P3: LOW PRIORITY (Minor Issues)

### P3.1: Convergence Criteria - Mostly Correct ✅

**Location:** `src/xfem_clean/convergence.py:12-17`

**Model Spec Requirement:**
```
||R|| ≤ max(tol_R_abs, tol_R_rel * ||F_ext||)
||ΔU|| ≤ max(tol_U_abs, tol_U_rel * ||U||)
```

**Current Implementation:**
```python
def residual_tolerance(self, reaction_estimate: float) -> float:
    fscale = max(1.0, abs(float(reaction_estimate)))
    return float(self.newton_tol_r + self.newton_beta * fscale)  # Abs + Rel ✅
```

**Status:** ✅ **CORRECT** - uses abs+rel for residual

**Note:** ΔU criterion not explicitly coded but handled via stagnation check (which has P0.2 issue)

---

## ✅ VERIFIED CORRECT

### ✅ History Management (Cohesive, Bond, Bulk)

**Evidence:**
- `src/xfem_clean/xfem/assembly_single.py:99-100`: Returns "Trial states, do not mutate committed"
- Line 477: `coh_trial = coh_committed.copy()` - makes copy before applying updates
- Line 496: `return True, q, coh_trial, mp_trial, ...` - returns trial, not committed
- `src/xfem_clean/xfem/analysis_single.py:959, 966, 978`: Commit only after step acceptance

**Verdict:** ✅ **CORRECT** - history only committed after convergence, never during Newton

---

### ✅ Bond-Slip Coupling Matrix

**Evidence:**
- `src/xfem_clean/bond_slip.py:1332-1342`: Full 8×8 outer product assembly
```python
for a in range(8):
    for b in range(8):
        rows.append(dofs[a])
        cols.append(dofs[b])
        data.append(K_bond * g[a] * g[b])  # Outer product ✅
```

**Model Spec:** `K += ∫ B^T (dq/ds) B dl` (full coupling)

**Verdict:** ✅ **CORRECT** - implements full coupling, not just diagonal

---

### ✅ Compression Damage Model

**Evidence:**
- `src/xfem_clean/compression_damage.py:70-114`: Implements parabolic σ(ε) per Eq. 3.46
```python
if eps <= self.eps_c1:
    eta = eps / self.eps_c1
    sigma = self.f_c * (2.0 * eta - eta * eta)  # Parabolic ✅
    E_t = self.f_c / self.eps_c1 * (2.0 - 2.0 * eta)  # Tangent ✅
else:
    sigma = self.f_c  # Plateau (no softening) ✅
    E_t = 0.0
```

**Verdict:** ✅ **CORRECT** - matches spec exactly

---

## Summary of Findings

| Issue | Priority | Status | File(s) | Fix Complexity |
|-------|----------|--------|---------|----------------|
| Cohesive unilateral opening | P0 | ❌ **CRITICAL BUG** | cohesive_laws.py, kernels_cohesive.py | **Medium** (20-30 lines) |
| Stagnation false positives | P0 | ❌ **CRITICAL BUG** | analysis_single.py | **Small** (10 lines) |
| Penalty caps (contact) | P0 | ⚠️ Need audit | contact_rebar.py | **Small** (5 lines) |
| Dowel action not wired | P1 | ⚠️ Implemented but unused | bond_slip.py, multicrack.py | **Large** (50+ lines) OR document as N/A |
| Junction coalescence | P1 | ✅ **ALREADY FIXED** | multicrack.py | None - update docs |
| Mixed-mode cohesive | P2 | ⚠️ Partial (data only) | cohesive_laws.py, assembly | **Large** (100+ lines) |
| Compression damage selectable | P2 | ✅ **ALREADY AVAILABLE** | material_factory.py | None - update docs |
| History management | ✅ | ✅ CORRECT | assembly_single.py, analysis_single.py | None |
| Bond coupling matrix | ✅ | ✅ CORRECT | bond_slip.py | None |
| Convergence abs+rel | ✅ | ✅ CORRECT | convergence.py | None |

---

## Recommended Fix Order

### Phase 1: Critical Bugs (P0)
1. ✅ **P0.1: Cohesive unilateral opening** - Implement max(δ,0) logic (1-2 hours)
2. ✅ **P0.2: Stagnation detection** - Add residual decrease check (30 min)
3. ⏸️ **P0.3: Penalty caps** - Audit contact module (30 min, defer if low risk)

### Phase 2: Documentation Updates
4. ✅ **Update parity matrix** - Mark P1.1 (junction) and P2.2 (compression) as PASS (15 min)
5. ✅ **Document dowel action status** - Mark as "implemented but not active" (15 min)

### Phase 3: Feature Enhancements (Optional)
6. ⏸️ **P2.1: Mixed-mode cohesive** - Large effort, defer to future (4-6 hours)
7. ⏸️ **P1.1: Wire dowel action** - Medium effort, needs testing (2-3 hours)

---

## Test Coverage Recommendations

### Critical (Must Add)
- `test_cohesive_unilateral_compression.py`: δ < 0 → t = 0, no damage
- `test_cohesive_unilateral_cyclic.py`: open → close → reopen → damage only from opening
- `test_convergence_no_false_stagnation.py`: Small ΔU with decreasing R → should not fail

### High Priority (Should Add)
- `test_bond_coupling_matrix_symmetry.py`: Verify K = K^T
- `test_compression_damage_tangent_fd.py`: Finite difference check on E_t

### Medium Priority (Nice to Have)
- `test_mixed_mode_cohesive_pure_shear.py`: δ_n=0, δ_t≠0 → shear softening
- `test_dowel_action_model_standalone.py`: DowelActionModel.sigma_r_and_tangent()

---

## Audit Methodology

1. **Spec Comparison**: Each implementation audited against self-contained model spec (no external refs)
2. **Code Reading**: Systematic review of cohesive_laws.py, assembly, convergence, material_factory
3. **Grep Analysis**: Searched for key patterns (junction, dowel, compression, coupling)
4. **Call Graph**: Verified which functions are actually invoked vs defined-but-unused

**Tools Used:**
- `grep -r` for pattern matching
- `Read` tool for line-by-line analysis
- Cross-referencing between spec requirements and implementation

---

**Audit Completed:** 2026-01-01
**Next Action:** Implement P0.1 and P0.2 fixes, add tests, update parity matrix
