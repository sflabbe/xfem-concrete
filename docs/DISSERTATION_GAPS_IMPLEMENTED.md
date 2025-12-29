# Dissertation Gaps Implemented

This document describes the features implemented to achieve parity between the solver and the dissertation model architecture (10.5445/IR/1000124842).

All implementations maintain backward compatibility with existing bond-slip features (validate_bond_inputs, boundscheck, python fallback, yielding/crack deterioration, dowel action).

---

## Phase 1: Mesh-Independent Reinforcement via Heaviside Enrichment

**Reference**: Dissertation Chapter 4, Section 4.5, Equations 4.92-4.103

### Implementation

**Module**: `src/xfem_clean/reinforcement.py`

### Key Equations

#### Heaviside Enrichment Function (Eq. 4.92)

```
H_r(x) = { 1  if φ_r(x) ≥ 0
         { 0  if φ_r(x) < 0

where φ_r(x) is the signed distance to reinforcement centerline
```

#### Displacement Approximation (Eq. 4.93-4.96)

```
u(x) = Σ N_i(x) u_i + Σ_{i∈N_r} N_i(x) H̃_r(x) a_i

where:
  H̃_r(x) = H_r(x) - H_r(x_i)  (shifted enrichment)
  N_r = nodes whose support intersects reinforcement
```

#### Reinforcement Internal Work (Eq. 4.101-4.103)

```
W_int^r = ∫_{Γ_r} ε_s σ_s A_s ds

where:
  ε_s = t^T ε t  (axial strain from continuum)
  Γ_r = reinforcement centerline
  A_s = cross-sectional area
```

### Features

1. **Geometric Independence**:
   - Reinforcement defined as polylines independent of mesh
   - Automatic detection of element-segment intersections
   - Line integral assembly along arbitrary paths

2. **Constitutive Models**:
   - Elastic steel: σ_s = E_s ε_s
   - Bilinear plastic: J2 plasticity with linear hardening
   - State management for plastic history

3. **Assembly**:
   - Line integral quadrature (default: 7-point Gauss)
   - B-matrix evaluation at arbitrary points via Newton iteration
   - Sparse global stiffness and force assembly

### Usage Flags

```python
from xfem_clean.reinforcement import create_straight_reinforcement_layer

layer = create_straight_reinforcement_layer(
    x_start=np.array([0.0, 0.0]),
    x_end=np.array([1.0, 0.5]),  # NOT aligned with mesh
    A_s=7.85e-5,                  # 10mm bar
    E_s=200e9,
    f_y=500e6,
    E_h=20e9,
    d_bar=0.010,
    layer_type="longitudinal",
)
```

### Validation

**Example**: `examples/ex_rebar_heaviside_angle.py`

- Single bar at 33° crossing multiple elements
- Demonstrates mesh independence
- Runtime: <10 seconds

---

## Phase 2: Transverse Reinforcement + Contact Penalty

**Reference**: Dissertation Chapter 4, Section 4.5.3, Equations 4.120-4.129

### Implementation

**Module**: `src/xfem_clean/contact_rebar.py`

### Key Equations

#### Tangential Gap (Eq. 4.120-4.126)

```
x_l = X_c + u_l
x_t = X_c + u_t
g = (u_l - u_t) · t̂

where:
  X_c = contact point (reference configuration)
  t̂ = contact direction (unit tangent)
```

#### Penalty Law (Eq. 4.127)

```
p(g) = { k_p(-g)  if g < 0  (penetration)
       { 0        if g ≥ 0  (separation)

For endpoint perfect contact (stirrup legs):
  p(g) = k_p(-g)  always (bilateral)
```

#### Contact Forces and Tangent (Eq. 4.128-4.129)

```
f_l = -p(g) N_l^T t̂
f_t = +p(g) N_t^T t̂

K_ll = k_p (t̂ ⊗ t̂)
K_tt = k_p (t̂ ⊗ t̂)
K_lt = -k_p (t̂ ⊗ t̂)
```

### Features

1. **Contact Types**:
   - `"crossing"`: Unilateral contact (only resists penetration)
   - `"endpoint"`: Bilateral constraint (perfect bonding at stirrup legs)

2. **Penalty Stiffness**:
   - Typically 10% of steel axial stiffness
   - User-configurable per contact point

3. **Integration**:
   - Compatible with bond-slip DOF mapping
   - Assembles global force/stiffness via sparse triplets

### Usage Flags

```python
from xfem_clean.contact_rebar import RebarContactPoint, assemble_rebar_contact

cp = RebarContactPoint(
    X_c=np.array([0.05, 0.05]),
    t_hat=np.array([1.0, 0.0]),
    k_p=1.57e6,  # ~10% of steel axial stiffness
    layer_l_id=0,
    layer_t_id=1,
    node_l=node_long,
    node_t=node_trans,
    contact_type="crossing",
)

f_contact, K_contact = assemble_rebar_contact(
    contact_points=[cp],
    u_total=q,
    dofs_map=dofs,
    ndof_total=ndof,
)
```

### Validation

**Example**: `examples/ex_transverse_contact.py`

- One longitudinal bar + one transverse bar
- Contact point with forced penetration
- Penalty activates, residual decreases
- Runtime: <10 seconds

---

## Phase 3: Crack Coalescence + Junction Enrichment + DOF Mapping

**Reference**: Dissertation Chapter 4, Sections 4.3.2-4.3.3, Equations 4.60-4.66

### Implementation

**Modules**:
- `src/xfem_clean/junction.py`
- `src/xfem_clean/dof_mapping.py`

### Key Equations

#### Junction Node Set (Eq. 4.64)

```
N_J = { i | supp(φ_i) ∩ X_j ≠ ∅ }

where X_j is the junction point
```

#### Junction Step Function (Eq. 4.65)

```
H_k(x) = sign(φ_k(x))  ∈ {-1, +1}

H̃_k(x) = H_k(x) - H_k(x_i)
```

#### Displacement Approximation (Eq. 4.66)

```
u(x) = Σ N_i u_i
     + Σ_{i∈N_J1} N_i H̃_1(x) a_i
     + Σ_{i∈N_J2} N_i H̃_2(x) b_i
```

#### DOF Projection (Eq. 4.60-4.63)

```
J(q_new) = (1/2) ∫_{Ω_p} |N_new q_new - N_old q_old|² dΩ

Normal equations:
  A q_new = b

where:
  A = ∫ N_new^T N_new dΩ
  b = ∫ N_new^T N_old q_old dΩ
```

### Features

1. **Coalescence Detection**:
   - Geometric proximity test (distance < tol_merge)
   - Element containment check
   - Automatic branch angle computation

2. **Topology Management**:
   - Arrest secondary crack at junction
   - Remove secondary tip enrichment
   - Add junction enrichment for both branches

3. **DOF Projection**:
   - L2 minimization for enriched DOFs
   - Standard DOFs copied directly
   - Reduces residual jump after topology change

### Usage Flags

```python
from xfem_clean.junction import detect_crack_coalescence, arrest_secondary_crack_at_junction

junctions = detect_crack_coalescence(
    cracks=cracks,
    nodes=nodes,
    elems=elems,
    tol_merge=0.005,  # 5mm tolerance
)

if junctions:
    arrest_secondary_crack_at_junction(junctions[0], cracks)
```

```python
from xfem_clean.dof_mapping import project_dofs_l2, transfer_dofs_simple

# After topology change:
q_new = project_dofs_l2(
    q_old=q,
    nodes_old=nodes,
    elems=elems,
    dofs_old=dofs_old,
    dofs_new=dofs_new,
)
```

### Validation

**Example**: `examples/ex_crack_coalescence_junction.py`

- Two cracks (vertical + horizontal)
- Coalescence detected when tips approach
- Junction enrichment created
- Runtime: <15 seconds

---

## Phase 4: Numerical Robustness Package

**Reference**: Dissertation Chapter 4 (various sections)

### Implementation

**Module**: `src/xfem_clean/numerical_aspects.py`

### Features

#### 1. Ill-Conditioning Node Removal (Dolbow Criterion)

**Equation**:
```
η_i = min(A⁺, A⁻) / (A⁺ + A⁻)

Remove node i from enrichment if η_i < 10⁻⁴
```

where A⁺, A⁻ are areas of node support on each side of crack.

**Usage**:
```python
from xfem_clean.numerical_aspects import remove_ill_conditioned_nodes

enriched_filtered = remove_ill_conditioned_nodes(
    nodes, elems, crack, enriched_nodes, tol_dolbow=1e-4
)
```

#### 2. Kinked Crack Tip Transformation

**Concept**: Near-tip functions assume straight crack. For kinked cracks, apply virtual rotation to align local coordinates with tip tangent.

**Usage**:
```python
from xfem_clean.numerical_aspects import tip_coords_kinked

r, theta = tip_coords_kinked(
    crack_polyline=[p0, p1, p_tip],
    tip_idx=2,
    x=evaluation_point,
)
```

#### 3. Nonlocal Stress Evaluation

**Reference**: Dissertation Eq. 4.134-4.135

**Equation**:
```
σ_nl(x₀) = Σ α(r) σ(x_gp) w_gp / Σ α(r) w_gp

where:
  α(r) = exp(-(r/ρ)²)  (Gaussian weight)
  ρ = nonlocal radius
```

**Features**:
- Gauss points within radius ρ
- Exclude points across crack (mesh-dependent)
- Symmetry mirroring for points beyond symmetry axis

#### 4. Quadrature Rules (Thesis Defaults)

**Module**: Configuration in model

**Defaults**:
- Quads: 9 pts with near-tip, 4 without
- Triangles: 7 pts with near-tip, 3 without
- Line integrals: 7 points

**Usage**:
```python
from xfem_clean.reinforcement import gauss_line_quadrature

xi, w = gauss_line_quadrature(n_points=7)
```

---

## Phase 5: Concrete Compression Damage Model

**Reference**: Dissertation Chapter 3, Section 3.2, Equations 3.44-3.46

### Implementation

**Module**: `src/xfem_clean/compression_damage.py`

### Key Equations

#### Stress-Strain Relation (Eq. 3.46)

```
σ_c(ε) = f_c [2(ε/ε_c1) - (ε/ε_c1)²]  for 0 ≤ ε ≤ ε_c1
σ_c(ε) = f_c                           for ε ≥ ε_c1

(parabolic up to peak, constant plateau beyond)
```

#### Damage from Secant Stiffness (Eq. 3.44-3.45)

```
E_sec(ε) = σ_c(ε) / ε

d_c = 1 - E_sec / E₀

where E₀ is initial elastic modulus
```

#### Equivalent Compressive Strain

```
ε_eq,c = max(0, -min(ε₁, ε₂))

where ε₁, ε₂ are principal strains
```

### Features

1. **Parabolic Model**:
   - No softening (plateau after peak)
   - Consistent with Model Code 2010 philosophy

2. **Damage Integration**:
   - Secant stiffness approach
   - Compatible with tension damage (split or combined)

3. **Calibration**:
   - Default parameters from f_c (Model Code 2010)
   - User-configurable

### Usage Flags

```python
from xfem_clean.compression_damage import ConcreteCompressionModel, get_default_compression_model

# Manual setup
model = ConcreteCompressionModel(
    f_c=30e6,       # 30 MPa
    eps_c1=0.002,   # Peak strain
    E_0=30e9,       # Initial modulus
)

# Or use defaults
model = get_default_compression_model(f_c_mpa=30.0)

# Compute stress and damage
sigma, E_t = model.sigma_epsilon_curve(eps=0.001)
d_c = model.compute_damage(eps=0.001)
```

### Validation

**Test**: `tests/test_dissertation_features.py::TestCompressionDamage::test_uniaxial_compression_test`

- Uniaxial compression simulation
- Parabolic curve verified
- Damage evolution monotonic
- No softening confirmed

---

## Regression Safety

All implementations maintain compatibility with existing features:

1. **Bond-Slip**:
   - `validate_bond_inputs` enabled by default in debug mode
   - Boundscheck active in Numba kernels
   - Python fallback available

2. **Substepping**:
   - All new assemblies respect rollback conventions
   - State arrays follow copy/commit pattern

3. **DOF Management**:
   - Enrichment changes trigger proper DOF rebuild
   - Steel DOFs compatible with new reinforcement

---

## Minimal Reproducible Examples

All examples located in `examples/` directory:

### 1. `ex_rebar_heaviside_angle.py`

**Purpose**: Single reinforcement bar at 33° crossing elements

**Features**:
- Mesh-independent reinforcement
- Bond-slip active (if integrated)
- NOT aligned with mesh

**Runtime**: <10 seconds

**Output**:
```
ndof (standard): 50
ndof (enriched): 72
Enrichment type: Heaviside (mesh-independent)
Bar angle: 33.0° (crosses 7 elements)
✓ PASS: Runtime < 30 seconds
```

### 2. `ex_transverse_contact.py`

**Purpose**: One longitudinal + one transverse bar with contact

**Features**:
- Penalty contact at crossing point
- Contact engages under penetration
- Newton reduces residual

**Runtime**: <10 seconds

**Output**:
```
ndof: 18
n_contact_points: 1
Contact active: YES
Penalty method: ✓
✓ PASS: Runtime < 30 seconds
```

### 3. `ex_crack_coalescence_junction.py`

**Purpose**: Two cracks coalescing, junction enrichment

**Features**:
- Geometric coalescence detection
- Junction enrichment triggers
- DOF topology change

**Runtime**: <15 seconds

**Output**:
```
n_cracks: 2
n_junctions: 1
Junction detected: ✓
Junction enrichment created: ✓
Topology update required: ✓
✓ PASS: Runtime < 30 seconds
```

---

## Testing

**Test Suite**: `tests/test_dissertation_features.py`

**Coverage**:
- Phase 1: 7 tests (reinforcement)
- Phase 2: 4 tests (contact)
- Phase 3: 1 test (junction)
- Phase 4: 2 tests (numerical aspects)
- Phase 5: 6 tests (compression damage)
- Integration: 1 test (combined)

**Run Tests**:
```bash
pytest tests/test_dissertation_features.py -v
```

---

## Summary

All dissertation gap features have been implemented with:

1. **Exact equations** from dissertation
2. **Validated examples** (<30s runtime each)
3. **Comprehensive tests** (20+ unit tests)
4. **Backward compatibility** with existing bond-slip
5. **Documentation** (this file + REINFORCEMENT_HEAVISIDE.md)

The solver now achieves **parity with the thesis model architecture and numerical robustness**.
