# Dowel Action Implementation Status

**Date:** 2026-01-01
**Status:** ⚠️ **IMPLEMENTED BUT NOT ACTIVE**
**Location:** `src/xfem_clean/bond_slip.py`

---

## Summary

Dowel action functionality is **fully implemented as data structures and functions** but is **NOT wired into the solver**. The code exists but is never called during analysis.

**Recommendation:** Document as "not active" (this document) rather than incomplete wiring (safer).

---

## What is Dowel Action?

From the model specification and dissertation (Eq. 3.62-3.68):

Dowel action provides transverse resistance at crack-rebar intersections:
```
w = (u_steel - u_concrete) · n_crack  # Normal opening at crossing
p = k_d * w                            # Dowel force (optionally unilateral)
```

This models the rebar acting as a dowel pin resisting crack opening perpendicular to the bar axis.

**Physics:** When a crack crosses a rebar at an angle, the bar provides normal stiffness like a dowel pin in a hole. This is distinct from:
- **Bond-slip**: Tangential resistance along bar axis
- **Cohesive**: Concrete-to-concrete traction
- **Contact**: Rebar-to-rebar interaction

---

## What's Implemented ✅

### 1. Constitutive Model

**File:** `src/xfem_clean/bond_slip.py:672-740`

```python
@dataclass
class DowelActionModel:
    """Dowel action model for transverse stress-opening relationship.

    Based on Brenna et al. model (Eqs. 3.62-3.68 in dissertation).
    """
    d_bar: float  # Bar diameter [m]
    f_c: float    # Concrete compressive strength [Pa]
    E_s: float    # Steel Young's modulus [Pa]

    def sigma_r_and_tangent(self, w: float) -> Tuple[float, float]:
        """Compute radial stress and tangent stiffness.

        σ_r = σ_r_max * (1 - exp(-k_d * w / d_bar))
        dσ_r/dw = σ_r_max * (k_d / d_bar) * exp(-k_d * w / d_bar)

        Where:
            σ_r_max = k_c * sqrt(f_c)  (k_c ≈ 0.8)
            k_d ≈ 50 (shape parameter)
        """
```

**Features:**
- ✅ Exponential stress-opening curve
- ✅ Consistent tangent stiffness
- ✅ Calibrated constants (k_c=0.8, k_d=50)
- ✅ Automatic σ_r_max calculation from f_c

### 2. Assembly Functions

**File:** `src/xfem_clean/bond_slip.py`

**Function 1:** `compute_dowel_springs` (line 1383)
- Computes dowel spring contributions for crack-bar crossings
- Returns forces and stiffness matrix

**Function 2:** `assemble_dowel_action` (line 1445)
- Assembles dowel action into global system
- Similar structure to bond-slip assembly

---

## What's NOT Wired ❌

### No Calls in Solvers

**Evidence:**
```bash
grep -r "assemble_dowel_action\|compute_dowel_springs" src/xfem_clean/ examples/
# Result: ZERO matches (only definitions, no calls)
```

The functions exist but are never invoked in:
- `analysis_single.py` - Single crack solver
- `multicrack.py` - Multi-crack solver
- Any example cases

### Missing Integration Points

To activate dowel action, the following would be needed:

1. **Crack-Bar Intersection Detection**
   - After crack insertion/growth
   - Identify which rebar segments cross which cracks
   - Compute intersection points in global coordinates

2. **Opening Calculation**
   - At each intersection, compute crack normal `n_crack`
   - Interpolate displacements: `u_steel`, `u_concrete`
   - Calculate `w = (u_steel - u_concrete) · n_crack`

3. **Assembly Call**
   - Pass intersection data to `compute_dowel_springs`
   - Add returned forces/stiffness to global system
   - Similar to how `assemble_bond_slip` is called

4. **State Management**
   - Track dowel opening history (if needed for unloading)
   - Commit/rollback during Newton iterations

**Estimated Effort:** ~100-150 lines + testing

---

## Why Not Active?

Possible reasons (speculation based on audit):

1. **Complexity vs Benefit:** Dowel action is a second-order effect compared to bond-slip and cohesive
2. **Numerical Stability:** Additional coupling may affect convergence
3. **Validation Data:** Limited experimental data for calibration
4. **Development Priority:** Other features took precedence

---

## Options for Future Work

### Option A: Document as "Not Active" (Current Choice) ✅

**Pros:**
- Honest about current status
- No risk of breaking existing functionality
- Preserves code for future activation

**Cons:**
- Feature not available for users
- Thesis parity gap remains

**Action:** This document serves as the documentation

### Option B: Wire Into Solver (Future Work)

**Pros:**
- Full thesis parity
- Enhanced physics for cyclic loading cases
- Use existing high-quality implementation

**Cons:**
- Non-trivial effort (~100-150 lines)
- Requires comprehensive testing
- May affect convergence behavior
- Need validation data

**Steps if pursued:**
1. Add `crack_bar_intersections()` function to detect crossings
2. Modify `solve_step()` to call `compute_dowel_springs` after assembly
3. Add dowel state arrays similar to bond-slip
4. Create synthetic test case (beam with angled crack crossing rebar)
5. Validate against literature if available

### Option C: Remove Code (Not Recommended)

**Pros:**
- Reduces code complexity
- No "dead code"

**Cons:**
- Loses high-quality implementation
- Would need to re-implement if needed later
- Thesis claims dowel action - removal creates disconnect

---

## Impact on Cases

**Cases NOT affected** (dowel action not critical):
- Pull-out tests (01): Aligned crack-bar, no transverse opening
- Tensile members (03): Axial loading, minimal dowel effect
- Beams without cyclic (04, 07, 08, 09): Cohesive + bond-slip dominate

**Cases POTENTIALLY affected** (if activated):
- **Cyclic walls (05, 10):** Diagonal shear cracks cross vertical/horizontal rebar at angles
  - Dowel action could improve crack opening resistance
  - May improve load-displacement hysteresis loops
  - Effect likely small compared to bond-slip

**Recommendation:** Current implementation without dowel action is adequate for most cases.

---

## Testing (If Wired in Future)

### Unit Test
```python
def test_dowel_action_model():
    """Test DowelActionModel constitutive law."""
    model = DowelActionModel(d_bar=12e-3, f_c=30e6, E_s=200e9)

    # Zero opening
    sigma_r, k = model.sigma_r_and_tangent(0.0)
    assert sigma_r == pytest.approx(0.0)

    # Positive opening
    w = 1e-3  # 1 mm
    sigma_r, k = model.sigma_r_and_tangent(w)
    assert sigma_r > 0
    assert k > 0
```

### Integration Test
```python
def test_dowel_action_assembly():
    """Test dowel springs at crack-bar crossing."""
    # Setup: beam with crack crossing rebar at 45°
    # Apply normal opening
    # Verify dowel force opposes opening
    # Check stiffness matrix symmetry
```

---

## References

**Dissertation:** 10.5445/IR/1000124842 (Rodrigo Gutiérrez)
- Equations 3.62-3.68: Brenna et al. dowel model
- Chapter 3, Section 3.4.5: Dowel action formulation

**Literature:**
- Brenna et al. (reference from dissertation)
- Dei Poli et al. (dowel action in RC)

---

## Conclusion

Dowel action is **implemented but not active**. The code quality is high (proper constitutive model, tangent, calibrated constants), but integration into the solver was not completed.

**Current Status:** Adequate for most use cases without dowel action
**Future Work:** Consider wiring if cyclic wall results need improvement
**Recommendation:** Leave as-is, document clearly (this file)

---

**Author:** Claude Code (Automated Audit)
**Last Updated:** 2026-01-01
