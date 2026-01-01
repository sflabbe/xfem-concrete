# P3: Dowel Action Status Report

**Priority**: Low (P3)
**Status**: ❓ UNCLEAR - Requires code audit and decision

---

## Background

The thesis (10.5445/IR/1000124842) claims that dowel action for embedded reinforcement bars is included in the model. Dowel action refers to the shear resistance provided by reinforcement bars when a crack crosses them at an angle, creating a normal force perpendicular to the bar axis.

---

## Code Search Results

Search for "dowel" in the codebase:

```bash
$ grep -ri "dowel" src/
```

Results:
- **No explicit dowel action implementation found** in main solver files
- **contact_rebar.py** exists and includes penalty contact for transverse bars
- **rebar.py** has basic bond-slip and axial stiffness contributions
- **No specific dowel stiffness calculation** at crack-bar crossings

---

## What IS Implemented

### 1. Reinforcement Heaviside Enrichment
- **Module**: `src/xfem_clean/reinforcement.py`
- **Status**: ✅ Fully implemented
- **Description**: Mesh-independent reinforcement via Heaviside enrichment per dissertation Eq. (4.92-4.103)
- **Features**:
  - Arbitrary bar orientation
  - Axial strain extraction from continuum
  - Elastic and bilinear plastic steel models
  - Line integral assembly

### 2. Transverse Reinforcement Contact
- **Module**: `src/xfem_clean/contact_rebar.py`
- **Status**: ✅ Implemented
- **Description**: Penalty contact between longitudinal and transverse bars per dissertation Eq. (4.120-4.129)
- **Features**:
  - Tangential gap calculation
  - Penalty law (unilateral or bilateral)
  - Contact forces and tangent stiffness
- **Limitation**: This is **bar-to-bar contact**, not crack-to-bar dowel action

### 3. Bond-Slip
- **Module**: `src/xfem_clean/bond_slip.py`
- **Status**: ✅ Fully implemented
- **Description**: Bond-slip interface between concrete and reinforcement
- **Features**:
  - Multiple bond laws (CEB-FIP, Bilinear, Banholzer)
  - Segment masking for unbonded regions
  - Yielding and crack deterioration

---

## What is MISSING

### Dowel Action at Crack-Bar Crossings

**Expected behavior** (per thesis):
When a crack crosses a reinforcement bar, the bar resists crack opening via:
1. **Axial tension** (already implemented via bond-slip + steel axial stiffness)
2. **Dowel action** (shear resistance from bar bending)

**Missing implementation**:
- No explicit calculation of crack-bar intersection geometry
- No additional stiffness contribution from bar bending/shear at crack crossings
- No normal force perpendicular to bar axis due to crack opening

**Possible locations where it should be**:
- In cohesive assembly, detect crack-bar crossings
- Add penalty/stiffness term: F_dowel = K_dowel * (u_crack · n_bar)
- Where n_bar is normal to bar axis

---

## Options

### Option A: Implement Dowel Action (RECOMMENDED)

**Effort**: Medium (2-3 days)

**Steps**:
1. Detect crack-bar intersection points during assembly
2. For each intersection:
   - Compute crack opening displacement (from XFEM enriched DOFs)
   - Project opening onto bar normal direction
   - Add penalty force: F_dowel = K_dowel * δ_normal
3. Calibrate K_dowel (typically 10-20% of axial stiffness)
4. Add unit test forcing crack-bar crossing
5. Verify dowel contribution is non-zero and finite

**Benefits**:
- Full thesis parity for cyclic wall cases (C1, C2)
- More accurate crack opening predictions
- Proper shear resistance at crack-bar intersections

---

### Option B: Document Omission (FALLBACK)

**Effort**: Low (1 hour)

**Steps**:
1. Explicitly document that dowel action is **not implemented**
2. Remove any thesis claims that it is included
3. Note in validation that cyclic wall results may differ due to missing dowel action
4. Add to "Future Work" section

**Drawbacks**:
- Thesis parity incomplete for cases relying on dowel action
- Cyclic wall hysteresis loops may be less accurate

---

## Decision

**RECOMMENDED**: Option A (implement dowel action properly)

**Rationale**:
- Cyclic walls (C1, C2) are key thesis validation cases
- Dowel action is explicitly mentioned in thesis
- Implementation is straightforward with existing crack/rebar infrastructure
- Improves accuracy for all reinforced concrete cases with inclined cracks

---

## Implementation Checklist (if Option A chosen)

- [ ] Create `src/xfem_clean/dowel_action.py` module
- [ ] Implement `detect_crack_bar_crossings(cracks, rebar_segs, nodes)`
- [ ] Implement `compute_dowel_stiffness(d_bar, E_s, I_bar)`
- [ ] Integrate into XFEM assembly (after cohesive forces)
- [ ] Add unit test: `tests/test_dowel_action.py`
  - [ ] Test crack-bar intersection detection
  - [ ] Test dowel force calculation
  - [ ] Test full assembly with dowel contribution
- [ ] Calibrate K_dowel against experimental data (if available)
- [ ] Update thesis parity matrix: P3 ✅ PASS

---

## References

- Thesis Section: Chapter 4 (reinforcement modeling)
- Relevant equations: (if explicit dowel equations exist in thesis, cite them)
- Contact rebar module: `src/xfem_clean/contact_rebar.py` (similar penalty approach)

---

**Last Updated**: 2026-01-01
**Author**: Claude (thesis parity verification task)
**Next Action**: Await user decision on Option A vs Option B
