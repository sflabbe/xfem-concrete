# XFEM Concrete Integration Notes
**Goal**: Bridge the gap between current solver and Gutierrez (2020) dissertation approach

---

## ✅ Phase 1: Non-Singular Tip Enrichment (COMPLETED)

### Implementation Summary
Implemented non-singular near-tip enrichment for cohesive cracks, replacing classical LEFM branch functions with smooth enrichment per Gutierrez (2020).

### Mathematical Background
- **Classical LEFM**: Uses branch functions `F = [√r sin(θ/2), √r cos(θ/2), ...]` (singular at tip)
- **Cohesive cracks**: Finite stresses at tip → smooth displacement field
- **New enrichment**: `F(r,θ) = r*sin(θ/2)` (non-singular, C⁰ continuous)

### Files Modified
```
src/xfem_clean/xfem/
├── geometry.py          (+52 lines) non_singular_cohesive_F_and_grad()
├── enrichment_single.py (+15 lines) tip_enrichment_type parameter
├── assembly_single.py   (+2 lines)  pass tip_enrichment_type
├── analysis_single.py   (+2 lines)  pass from model
├── model.py             (+4 lines)  add tip_enrichment_type field
└── post.py              (+2 lines)  retrocompat with getattr()
```

### Configuration API
```python
from xfem_clean import XFEMModel

# Default: non-singular enrichment for cohesive cracks
model = XFEMModel(..., tip_enrichment_type="non_singular_cohesive")

# Legacy: LEFM singular branch functions
model = XFEMModel(..., tip_enrichment_type="lefm_branch")
```

### Testing
- ✓ Function computes correctly (F₀ = r*sin(θ/2), F₁₋₃ = 0)
- ✓ Gradients analytical and verified
- ✓ Model accepts both modes
- ✓ Backward compatibility maintained

### References
- Gutierrez (2020), KIT dissertation, Section 3.3.2
- Moës et al. (1999) for classical branch functions

---

## ✅ Phase 2: Bond-Slip Integration (COMPLETED - Integration working, needs validation refinement)

### Implementation Status ✅
- **bond_slip.py**: ✅ Fully implemented (fib Model Code 2010)
- **dofs_single.py**: ✅ Extended XFEMDofs with steel DOF fields
- **model.py**: ✅ Added bond-slip parameters (enable_bond_slip, rebar_diameter, bond_condition)
- **assembly_single.py**: ✅ Integrated assemble_bond_slip() into solver
- **analysis_single.py**: ✅ State management (initialization + updates)
- **Steel DOF indexing**: ✅ FIXED - sparse DOF mapping implemented
- **validation example**: ⚠️  Created but needs tuning (convergence issue)

### Steel DOF Indexing Bug - ✅ RESOLVED
**Symptom**: `ValueError: axis 0 index 1393 exceeds matrix dimension 984`

**Root cause**: Mismatch between sparse DOF allocation and dense indexing assumption

**Solution implemented** (commit 4d4dbef):
- Modified `assemble_bond_slip()` to accept `steel_dof_map` parameter
- Updated `_bond_slip_assembly_numba()` to use `steel_dof_map[n, 0]` instead of `steel_dof_offset + 2*n`
- Updated `assembly_single.py` to pass `dofs.steel` as `steel_dof_map`
- Added backward compatibility: if `steel_dof_map=None`, generates dense mapping

**Files modified**:
- `src/xfem_clean/bond_slip.py`: Updated assembly functions
- `src/xfem_clean/xfem/assembly_single.py`: Pass dofs.steel
- `src/xfem_clean/xfem/analysis_single.py`: Fix missing bond_committed return

### Known Issue ⚠️: Pullout Test Convergence
**Symptom**: `RuntimeError: Substepping exceeded max_subdiv=12 (reason=newton) at u=1.46e-08 m`

**Likely causes**:
1. Boundary conditions for steel DOFs may be missing/incorrect
2. Initial stiffness matrix may be ill-conditioned
3. Pullout test geometry may need adjustment (simplified setup needed)

**Next steps**:
- Create simplified unit test (single element with bond-slip)
- Verify steel BC handling in apply_dirichlet()
- Add debug output for bond stiffness matrix conditioning

### Completed Implementation

#### 2.1 Extend DOF Structure
**File**: `src/xfem_clean/xfem/dofs_single.py`

```python
@dataclass
class XFEMDofs:
    std: np.ndarray              # Concrete DOFs (existing)
    H: np.ndarray                # Heaviside enrichment (existing)
    tip: np.ndarray              # Tip enrichment (existing)
    steel: np.ndarray            # NEW: Steel DOFs for bond-slip
    steel_dof_offset: int        # NEW: First steel DOF index
    ndof: int                    # Total DOFs (concrete + steel)
    H_nodes: np.ndarray
    tip_nodes: np.ndarray
    steel_nodes: np.ndarray      # NEW: Nodes with steel
```

**DOF allocation strategy**:
```python
# Concrete DOFs: 0 ... 2*nnode-1
# Enriched DOFs: 2*nnode ... (varies)
# Steel DOFs:    (after enriched) ... ndof-1
```

#### 2.2 Integrate into Assembly
**File**: `src/xfem_clean/xfem/assembly_single.py`

Add after cohesive integration (line ~565):
```python
# Bond-slip contribution (if reinforcement present)
if rebar_segs is not None and model.enable_bond_slip:
    f_bond, K_bond, bond_states_trial = assemble_bond_slip(
        u_total=q,
        steel_segments=rebar_segs,
        steel_dof_offset=dofs.steel_dof_offset,
        bond_law=bond_law,
        bond_states=bond_committed,
        use_numba=use_numba
    )
    # Add to global system
    fint += f_bond
    K_data += K_bond.data
    K_rows += K_bond.row
    K_cols += K_bond.col
```

**New imports**:
```python
from xfem_clean.bond_slip import (
    assemble_bond_slip,
    BondSlipModelCode2010,
    BondSlipStateArrays
)
```

#### 2.3 State Management
**File**: `src/xfem_clean/xfem/analysis_single.py`

**Initialize** (after crack initialization):
```python
if model.enable_bond_slip and rebar_segs is not None:
    n_seg = rebar_segs.shape[0]
    bond_committed = BondSlipStateArrays.zeros(n_seg)
    bond_law = BondSlipModelCode2010(
        f_cm=model.fc,
        d_bar=model.rebar_diameter,
        condition="good"  # or "poor"
    )
else:
    bond_committed = None
    bond_law = None
```

**Update** (after Newton convergence):
```python
if bond_committed is not None:
    bond_committed.update_from(bond_trial)
```

#### 2.4 Model Configuration
**File**: `src/xfem_clean/xfem/model.py`

```python
@dataclass
class XFEMModel:
    # ... existing fields ...

    # Bond-slip parameters
    enable_bond_slip: bool = False
    rebar_diameter: float = 0.016  # meters
    bond_condition: str = "good"   # "good" | "poor"

    # Dowel action (activated at crack-rebar intersection)
    enable_dowel: bool = False
    dowel_penalty_factor: float = 1.0
```

### Testing Strategy
Create `examples/bond_slip_pullout.py`:
```python
"""Pull-out test: single embedded bar with bond-slip"""
# Setup: bar length L_bar, embedment length L_emb
# Apply: displacement at free end
# Verify: Load-slip curve matches analytical/experimental
# Expected outputs: slip(x), bond_stress(x), energy_dissipated
```

---

## 📋 Phase 3: Crack Coalescence & Junction Enrichment (PLANNED)

### Current Status
- **multicrack.py**: Multiple cracks handled independently
- **Issue**: No detection when cracks merge
- **Issue**: No junction enrichment at merge points

### 3.1 Coalescence Detection
**File**: `src/xfem_clean/xfem/geometry.py`

Add geometric intersection test:
```python
def detect_tip_to_segment_intersection(
    tip: Tuple[float, float],
    tip_direction: Tuple[float, float],
    segment_p0: Tuple[float, float],
    segment_p1: Tuple[float, float],
    tolerance: float = 1e-3
) -> Optional[Tuple[float, float]]:
    """
    Check if crack tip intersects another crack segment.

    Returns
    -------
    intersection_point : (x, y) or None
    """
    # Line-segment intersection with tolerance
    # If distance(tip, segment) < tolerance: return closest point
```

**Call site**: In `multicrack.py::propagate_crack()`
```python
for k, crack_k in enumerate(cracks):
    if k == i:
        continue
    pt_isect = detect_tip_to_segment_intersection(
        tip=(cracks[i].tip_x, cracks[i].tip_y),
        tip_direction=...,
        segment_p0=crack_k.p0(),
        segment_p1=crack_k.pt(),
        tolerance=0.1 * dy
    )
    if pt_isect is not None:
        # Create junction entity
        return ("junction", i, k, pt_isect)
```

### 3.2 Junction Enrichment
**New file**: `src/xfem_clean/xfem/junction.py`

```python
@dataclass
class CrackJunction:
    """Junction between two coalesced cracks"""
    crack_1: int       # Index of first crack
    crack_2: int       # Index of second crack
    x: float           # Junction location
    y: float
    active: bool = True

def step_junction_enrichment(x, y, junction: CrackJunction) -> float:
    """
    Step-junction enrichment function (Gutierrez 2020, Section 3.4).

    Replaces tip enrichment near junction with Heaviside-like function
    to handle displacement discontinuity at junction.
    """
    # Simplified: return sign based on which side of junction line
    # Full implementation: distance-weighted blending
```

**DOF management**:
```python
@dataclass
class MultiXFEMDofsWithJunction:
    # ... existing fields ...
    junctions: list[CrackJunction]
    junction_dofs: list[np.ndarray]  # DOFs for each junction
```

### 3.3 DOF Transfer on Topology Change
**File**: `src/xfem_clean/xfem/multicrack.py`

When junction created:
```python
def transfer_solution_with_junction(
    q_old: np.ndarray,
    dofs_old: MultiXFEMDofs,
    dofs_new: MultiXFEMDofsWithJunction
) -> np.ndarray:
    """
    Transfer solution when enrichment changes (junction creation).

    Strategy:
    1. Standard DOFs: copy directly (always present)
    2. Heaviside DOFs: copy if still active
    3. Tip DOFs near junction: project to junction DOFs (least-squares)
    4. New junction DOFs: initialize via projection
    """
    q_new = np.zeros(dofs_new.ndof)
    q_new[:dofs_old.std.size] = q_old[:dofs_old.std.size]  # Standard
    # ... map H, tip, junction via (crack_id, node_id) lookup
    return q_new
```

### Testing Strategy
Create `examples/crack_coalescence.py`:
```python
"""Two-crack coalescence test"""
# Setup: beam with two initial notches (left and right)
# Load: displacement-controlled at top
# Expected: cracks propagate toward center and merge
# Verify: solver continues without NaN/divergence after merge
# Check: junction enrichment activated at merge point
```

---

## 📋 Phase 4: Mesh-Independent Reinforcement (PLANNED)

### Current Status
- **rebar.py**: Mesh-dependent (nodes = mesh nodes)
- **Limitation**: Perfect bond implicitly enforced
- **Limitation**: Cannot model arbitrary rebar layout

### 4.1 New Reinforcement Module
**New file**: `src/xfem_clean/reinforcement.py`

```python
from dataclasses import dataclass
from typing import Literal

ReinforcementType = Literal["rebar", "stirrup", "frp_strip", "fiber"]

@dataclass
class ReinforcementSegment:
    """1D reinforcement embedded in 2D concrete domain"""
    type: ReinforcementType
    x0: float             # Start point
    y0: float
    x1: float             # End point
    y1: float
    area: float           # Cross-sectional area
    E: float              # Young's modulus
    sigma_y: float        # Yield stress
    Eh: float             # Hardening modulus
    perimeter: float      # For bond-slip calculation

    # Bond-slip parameters
    bond_enabled: bool = True
    bond_tau_max: float = 2.5e6  # Pa (from Code Model 2010)
    bond_s1: float = 0.001       # m (slip at peak bond stress)
    bond_s2: float = 0.002       # m
    bond_s3: float = 0.016       # m (= d_bar default)

    def length(self) -> float:
        return math.sqrt((self.x1-self.x0)**2 + (self.y1-self.y0)**2)

    def tangent(self) -> Tuple[float, float]:
        L = self.length()
        return ((self.x1-self.x0)/L, (self.y1-self.y0)/L)

def create_longitudinal_rebars(
    y_location: float,
    x_start: float,
    x_end: float,
    n_bars: int,
    area_per_bar: float,
    E: float,
    sigma_y: float,
    diameter: float
) -> list[ReinforcementSegment]:
    """Helper to create evenly spaced horizontal bars"""
    spacing = (x_end - x_start) / (n_bars - 1)
    segments = []
    for i in range(n_bars):
        x = x_start + i * spacing
        seg = ReinforcementSegment(
            type="rebar",
            x0=x, y0=y_location,
            x1=x, y1=y_location,  # Point reinforcement (zero length)
            area=area_per_bar,
            E=E,
            sigma_y=sigma_y,
            perimeter=math.pi * diameter,
            # ... bond params
        )
        segments.append(seg)
    return segments
```

### 4.2 Heaviside Enrichment for Reinforcement
**Concept**: Activate Heaviside enrichment in elements containing reinforcement

```python
def detect_rebar_in_element(
    xe: np.ndarray,        # Element node coords (4,2)
    rebar: ReinforcementSegment
) -> bool:
    """Check if rebar segment intersects element"""
    xmin, xmax = xe[:,0].min(), xe[:,0].max()
    ymin, ymax = xe[:,1].min(), ye[:,1].max()

    seg_clipped = clip_segment_to_bbox(
        (rebar.x0, rebar.y0), (rebar.x1, rebar.y1),
        xmin, xmax, ymin, ymax
    )
    return seg_clipped is not None

def build_rebar_enriched_dofs(
    nodes: np.ndarray,
    elems: np.ndarray,
    rebars: list[ReinforcementSegment]
) -> dict:
    """
    For each rebar, identify enriched nodes.
    Similar to crack Heaviside enrichment.
    """
    rebar_enriched_nodes = []
    for rebar in rebars:
        enriched = np.zeros(len(nodes), dtype=bool)
        for e, conn in enumerate(elems):
            if detect_rebar_in_element(nodes[conn], rebar):
                enriched[conn] = True
        rebar_enriched_nodes.append(enriched)
    return rebar_enriched_nodes
```

**Integration**: Modify `dofs_single.py` or create `dofs_reinforced.py`

### 4.3 Assembly with Enriched Reinforcement
**Challenge**: Couple concrete DOFs (enriched) with steel DOFs (1D)

```python
def assemble_reinforcement_contribution(
    q: np.ndarray,
    nodes: np.ndarray,
    elems: np.ndarray,
    dofs: XFEMDofs,
    rebars: list[ReinforcementSegment],
    rebar_enriched_nodes: list[np.ndarray]
) -> Tuple[np.ndarray, sp.csr_matrix]:
    """
    Assemble f_rebar and K_rebar using Heaviside enrichment.

    For each rebar segment:
    1. Identify cut elements
    2. Gauss integration along rebar line
    3. Compute concrete displacement u_c at Gauss point (with enrichment)
    4. Compute steel displacement u_s at Gauss point
    5. Compute slip s = (u_s - u_c) · t
    6. Compute bond stress τ(s) and steel stress σ_s(ε_s)
    7. Assemble forces and tangent
    """
    # Placeholder: integrate along rebar using 1D Gauss quadrature
```

---

## 📋 Phase 5: Dowel Action (PLANNED)

### Concept
When a crack crosses a rebar at an angle, the bar resists opening via **dowel action** (transverse shear + bending).

### Detection
**File**: `src/xfem_clean/xfem/geometry.py`

```python
def detect_crack_rebar_intersection(
    crack: XFEMCrack,
    rebar: ReinforcementSegment
) -> Optional[Tuple[float, float, float]]:
    """
    Detect intersection between crack and rebar.

    Returns
    -------
    (x_isect, y_isect, angle) or None
        angle: angle between crack and rebar (radians)
    """
    # Line-line intersection
    # If |angle| > threshold (e.g., 15°): activate dowel
```

### Dowel Spring Assembly
**File**: `src/xfem_clean/bond_slip.py` (or new `dowel_action.py`)

```python
def assemble_dowel_springs(
    crack: XFEMCrack,
    rebars: list[ReinforcementSegment],
    nodes: np.ndarray,
    dofs: XFEMDofs,
    E_s: float,
    d_bar: float
) -> sp.csr_matrix:
    """
    Add penalty springs at crack-rebar intersections.

    Stiffness: k_dowel = E_s * I / L_e^3
    where L_e = 10*d_bar (empirical effective length)

    Couples:
    - Concrete DOFs at crack opening (normal direction)
    - Steel DOFs at rebar location (transverse direction)
    """
```

**Integration**: Call after bond-slip assembly in `assembly_single.py`

---

## 📋 Phase 6: Solution Transfer/Projection (PLANNED)

### Problem
When crack topology changes (new crack, growth, junction), the DOF structure changes:
- New enrichment DOFs added
- Old tip DOFs may become inactive
- Junction DOFs replace tip DOFs

**Current behavior**: Old solution discarded → Newton starts from elastic guess → poor convergence

### Solution: Projection
**File**: `src/xfem_clean/xfem/multicrack.py`

```python
def project_solution_to_new_dofs(
    q_old: np.ndarray,
    dofs_old: MultiXFEMDofs,
    dofs_new: MultiXFEMDofs,
    nodes: np.ndarray,
    elems: np.ndarray
) -> np.ndarray:
    """
    Project displacement field from old to new DOF structure.

    Strategy:
    1. Sample displacement field u_old(x,y) at Gauss points using q_old
    2. Construct least-squares system: min ||u_new(x,y) - u_old(x,y)||²
    3. Solve for q_new

    Sparse formulation (node-based):
    - Standard DOFs: direct copy
    - Enriched DOFs: project locally (element-wise)
    """
    q_new = np.zeros(dofs_new.ndof)

    # 1. Copy standard DOFs (always present)
    q_new[:2*len(nodes)] = q_old[:2*len(nodes)]

    # 2. Map enriched DOFs by (crack_id, node_id) lookup
    for k in range(len(dofs_new.H)):
        if k < len(dofs_old.H):
            # Crack k existed in old structure
            for n in range(len(nodes)):
                if dofs_new.H_nodes[k][n] and dofs_old.H_nodes[k][n]:
                    # Copy H DOFs
                    old_idx = dofs_old.H[k][n]
                    new_idx = dofs_new.H[k][n]
                    q_new[new_idx] = q_old[old_idx]
                    q_new[new_idx+1] = q_old[old_idx+1]

    # 3. Project tip DOFs (more complex: use local L2 projection)
    # ... element-wise least-squares solve

    return q_new
```

**Call site**: In `analysis_single.py` after DOF rebuild:
```python
if dofs_changed:
    q = project_solution_to_new_dofs(q, dofs_old, dofs_new, nodes, elems)
```

---

## 📋 Phase 7: Arc-Length Integration (PLANNED)

### Current Status
- **arc_length.py**: Fully implemented
- **Issue**: Main solver is displacement-controlled
- **Issue**: Cannot capture snap-back / post-peak softening

### Required Changes

#### 7.1 Solver Loop Redesign
**File**: `src/xfem_clean/xfem/analysis_single.py`

**Current**:
```python
for u_target in u_targets:
    # Solve with fixed displacement BC
```

**Proposed**:
```python
if model.control_mode == "displacement":
    # Existing displacement control
elif model.control_mode == "arc_length":
    from xfem_clean.arc_length import arc_length_step

    lambda_n = 0.0
    arc_len = model.arc_length_initial

    for step in range(n_steps):
        converged, u_new, lambda_new, n_iter = arc_length_step(
            K=K,
            f_int=fint,
            P_ref=P_reference,  # Load pattern
            u_n=u_n,
            lambda_n=lambda_n,
            arc_length=arc_len,
            assemble_fn=lambda q: assemble_xfem_system(...),
            fixed_dofs=fixed_dofs,
            max_iter=model.newton_maxit,
            tol_r=model.newton_tol_r
        )

        if converged:
            u_n = u_new
            lambda_n = lambda_new
            arc_len = adapt_arc_length(arc_len, n_iter)
        else:
            arc_len *= 0.5  # Reduce step size
```

#### 7.2 Load Pattern Definition
**File**: `src/xfem_clean/xfem/model.py`

```python
@dataclass
class XFEMModel:
    # ... existing ...

    # Arc-length control
    control_mode: str = "displacement"  # "displacement" | "arc_length"
    arc_length_initial: float = 1e-3
    arc_length_min: float = 1e-6
    arc_length_max: float = 1e-1

    # Reference load pattern (for arc-length)
    load_pattern: Optional[Dict[int, Tuple[float, float]]] = None
    # Example: {node_id: (fx, fy)}
```

#### 7.3 Boundary Condition Handling
**Arc-length predictor/corrector** needs to apply BCs during solve:

```python
def apply_bc_to_system(K: sp.csr_matrix, f: np.ndarray, fixed_dofs: dict):
    """Zero out rows/cols for Dirichlet BCs"""
    for dof, val in fixed_dofs.items():
        K[dof, :] = 0
        K[:, dof] = 0
        K[dof, dof] = 1.0
        f[dof] = val
```

**Already implemented** in `arc_length.py` (lines 381-396) ✓

---

## 📋 Testing & Validation

### Test Suite
```
examples/
├── validation_pullout.py           # Bond-slip: pull-out test
├── validation_sspot.py             # Bond-slip: FRP debonding (SSPOT)
├── validation_coalescence.py       # Junction: two-crack merge
├── validation_dowel.py             # Dowel action: inclined crack crossing rebar
├── validation_arc_length.py        # Post-peak softening (snap-back)
└── regression_gutierrez_beam.py    # Ensure existing example still works
```

### Acceptance Criteria
For each test:
- ✓ Converges without NaN/divergence
- ✓ Energy balance: `E_total = E_elastic + E_cohesive + E_bond + E_plastic`
- ✓ Results match analytical/literature values (±5%)
- ✓ Outputs: displacement, stress, damage, slip, bond stress, energy

---

## 🔧 Development Strategy

### Recommended Order
1. **Bond-slip integration** (Phase 2) — Most impactful, existing module ready
2. **Arc-length integration** (Phase 7) — Existing module, easier integration
3. **Solution transfer** (Phase 6) — Improves all crack propagation cases
4. **Crack coalescence** (Phase 3) — Requires junction enrichment (complex)
5. **Mesh-independent reinforcement** (Phase 4) — Major refactor
6. **Dowel action** (Phase 5) — Extension of bond-slip + crack detection

### Commit Strategy
Small, testable commits:
```bash
git commit -m "Extend DOF structure for bond-slip (steel DOFs)"
git commit -m "Integrate bond_slip assembly into solver"
git commit -m "Add bond-slip state management to analysis loop"
git commit -m "Test: pullout example validates bond-slip"
```

### Testing Workflow
For each phase:
1. Implement core functionality
2. Create validation example
3. Run regression tests (ensure nothing breaks)
4. Document in INTEGRATION_NOTES.md
5. Commit + push

---

## 📚 References

1. **Gutierrez (2020)**: "Two-dimensional modelling of fracture in reinforced concrete structures applying XFEM", KIT dissertation
   - Section 3.3.2: Non-singular enrichment
   - Section 3.4: Junction enrichment
   - Section 4: Bond-slip modeling

2. **Code Model 2010** (fib): Bond-slip constitutive law

3. **Moës et al. (1999)**: Classical XFEM branch functions

4. **Crisfield (1981)**: Arc-length method for nonlinear FEM

---

## 🚀 Next Steps

### Immediate (Phase 2)
- [ ] Extend `XFEMDofs` with steel DOFs
- [ ] Modify `assemble_xfem_system()` to call `assemble_bond_slip()`
- [ ] Add bond-slip state management to `run_analysis_xfem()`
- [ ] Create pullout test
- [ ] Validate Load-Slip curve

### Short-term
- [ ] Integrate arc-length (Phase 7)
- [ ] Implement solution transfer (Phase 6)

### Long-term
- [ ] Crack coalescence + junction (Phase 3)
- [ ] Mesh-independent reinforcement (Phase 4)
- [ ] Dowel action (Phase 5)

---

**Status**: Phase 1 complete ✅
**Last Updated**: 2025-12-29
**Next Milestone**: Bond-slip integration (Phase 2)
