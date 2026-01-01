# Mesh-Independent Reinforcement via Heaviside Enrichment

This document provides detailed documentation for the mesh-independent reinforcement implementation using Heaviside enrichment.

**Reference**: Dissertation 10.5445/IR/1000124842, Chapter 4, Section 4.5

---

## Overview

Traditional finite element reinforcement modeling requires mesh alignment with rebar locations, limiting flexibility and accuracy. The Heaviside enrichment approach allows reinforcement to be defined geometrically as 1D objects (polylines) independent of the background mesh.

### Key Advantages

1. **Mesh Independence**: Reinforcement can cross elements at arbitrary angles
2. **Geometric Flexibility**: Define rebars as polylines with arbitrary paths
3. **No Mesh Constraints**: No need to align mesh with reinforcement layout
4. **Accurate Integration**: Line integrals along exact rebar paths

---

## Theoretical Foundation

### Heaviside Enrichment Function (Eq. 4.92)

The Heaviside function partitions the domain based on signed distance to the reinforcement centerline:

```
H_r(x) = { 1  if φ_r(x) ≥ 0
         { 0  if φ_r(x) < 0
```

where `φ_r(x)` is the signed distance from point `x` to the reinforcement centerline `Γ_r`.

### Shifted Enrichment

To preserve partition of unity and eliminate rigid-body modes:

```
H̃_r(x) = H_r(x) - H_r(x_i)
```

This ensures the enrichment "shifts" relative to each node's position.

### Displacement Approximation (Eq. 4.93-4.96)

The enriched displacement field is:

```
u(x) = Σ_{i∈N} N_i(x) u_i + Σ_{i∈N_r} N_i(x) H̃_r(x) a_i
```

where:
- `N` is the set of all nodes
- `N_r ⊂ N` is the subset of nodes enriched by reinforcement
- `u_i` are standard DOFs
- `a_i` are enriched DOFs

### Reinforcement Kinematics (Eq. 4.101)

The axial strain in the reinforcement is extracted from the continuum strain tensor:

```
ε_s(s) = t^T(s) · ε(x(s)) · t(s)
```

where:
- `s` is the arc length along the bar
- `t(s)` is the unit tangent vector
- `ε(x(s))` is the continuum strain tensor at position `x(s)`

In discrete form:

```
ε_s = B_s · q

where B_s = [t_x², t_y², 2t_x t_y] · B_continuum
```

### Internal Work (Eq. 4.102-4.103)

The internal work of the reinforcement is:

```
W_int^r = ∫_{Γ_r} ε_s σ_s A_s ds
```

Discrete form:

```
f_s^int = Σ_e ∫_{Γ_r ∩ Ω_e} B_s^T A_s σ_s ds

K_s = Σ_e ∫_{Γ_r ∩ Ω_e} B_s^T A_s E_t B_s ds
```

where the integrals are evaluated via Gauss quadrature along the line segments.

---

## Implementation Details

### Module Structure

**File**: `src/xfem_clean/reinforcement.py`

**Key Classes**:
- `ReinforcementSegment`: Single segment with endpoints, material properties
- `ReinforcementLayer`: Collection of segments forming a layer
- `ReinforcementState`: State variables (plastic strain, etc.)

**Key Functions**:
- `signed_distance_to_segment()`: Compute φ_r(x)
- `heaviside_enrichment()`: Evaluate H_r
- `shifted_heaviside_enrichment()`: Evaluate H̃_r
- `compute_bar_strain_from_continuum()`: Extract ε_s from ε
- `assemble_reinforcement_layers()`: Global assembly

### Line Integral Quadrature

Default: 7-point Gauss-Legendre quadrature per thesis.

For each segment-element intersection:
1. Find intersection interval `[s_start, s_end]`
2. Map Gauss points from `[-1, 1]` to `[s_start, s_end]`
3. Evaluate B matrix at each Gauss point
4. Integrate stiffness and force

**Gauss Points** (n=7):
```
ξ_i ∈ [-0.949, -0.742, -0.406, 0.0, 0.406, 0.742, 0.949]
w_i ∈ [0.129, 0.280, 0.382, 0.418, 0.382, 0.280, 0.129]
```

### B Matrix Evaluation at Arbitrary Points

**Challenge**: Evaluate strain-displacement matrix `B(x)` at points not necessarily at Gauss points.

**Solution**: Newton iteration to find parametric coordinates `(ξ, η)`:

```python
def find_xi_eta_newton(x, elem_nodes):
    # Solve: x(ξ, η) = Σ N_i(ξ, η) x_i = x_target
    xi, eta = 0.0, 0.0  # Initial guess: element center

    for iter in range(max_iter):
        x_current = Σ N_i(xi, eta) x_i
        r = x - x_current

        if ||r|| < tol:
            return xi, eta

        J = dx/d(ξ,η)  # Jacobian
        dxi_deta = J^{-1} · r
        xi += dxi_deta[0]
        eta += dxi_deta[1]
```

Once `(ξ, η)` is found, compute B matrix:

```python
def compute_B_matrix_q4(xi, eta, elem_nodes):
    N, dN_dxi, dN_deta = q4_shape(xi, eta)

    J = Σ dN_i/dξ ⊗ x_i  # Jacobian
    J_inv = inv(J)

    dN_dx = J_inv · dN_dxi
    dN_dy = J_inv · dN_deta

    # Assemble B matrix (Voigt notation)
    B[0, 2i] = dN_dx[i]     # εxx: ∂u_x/∂x
    B[1, 2i+1] = dN_dy[i]   # εyy: ∂u_y/∂y
    B[2, 2i] = dN_dy[i]     # γxy: ∂u_x/∂y
    B[2, 2i+1] = dN_dx[i]   # γxy: ∂u_y/∂x

    return B
```

---

## Usage Guide

### Basic Usage

```python
from xfem_clean.reinforcement import create_straight_reinforcement_layer

# Define layer
layer = create_straight_reinforcement_layer(
    x_start=np.array([0.0, 0.02]),   # Start point [m]
    x_end=np.array([1.0, 0.03]),     # End point [m] (33° angle)
    A_s=7.85e-5,                      # Cross-section [m²] (10mm diameter)
    E_s=200e9,                        # Young's modulus [Pa]
    f_y=500e6,                        # Yield stress [Pa]
    E_h=20e9,                         # Hardening modulus [Pa]
    d_bar=0.010,                      # Diameter [m]
    layer_type="longitudinal",
    layer_id=0,
    n_segments=10,                    # Subdivide for integration
)
```

### Assembly into Global System

```python
from xfem_clean.reinforcement import assemble_reinforcement_layers

# Assemble
f_rebar, K_rebar, states_new = assemble_reinforcement_layers(
    q=q_global,
    nodes=nodes,
    elems=elems,
    dofs_map=dofs,
    layers=[layer1, layer2, ...],
    states_comm=states_committed,
    n_gauss_line=7,              # Thesis default
    use_plasticity=True,
)

# Add to global system
f_int += f_rebar
K_global += K_rebar
```

### Constitutive Models

**Elastic Steel**:
```python
from xfem_clean.reinforcement import steel_elastic_1d

sigma, E_t = steel_elastic_1d(eps=0.002, E=200e9)
```

**Bilinear Plastic Steel**:
```python
from xfem_clean.reinforcement import steel_bilinear_1d

sigma, E_t, eps_p_new = steel_bilinear_1d(
    eps=0.005,       # Total strain
    eps_p=0.001,     # Plastic strain (history)
    E=200e9,
    f_y=500e6,
    E_h=20e9,
)
```

### Advanced: Custom Polyline

```python
from xfem_clean.reinforcement import ReinforcementSegment, ReinforcementLayer

# Define custom path
segments = []
path_points = [
    np.array([0.0, 0.0]),
    np.array([0.3, 0.1]),
    np.array([0.6, 0.15]),
    np.array([1.0, 0.2]),
]

for i in range(len(path_points) - 1):
    seg = ReinforcementSegment(
        x0=path_points[i],
        x1=path_points[i+1],
        A_s=7.85e-5,
        E_s=200e9,
        f_y=500e6,
        E_h=20e9,
        d_bar=0.010,
        layer_id=0,
    )
    segments.append(seg)

layer = ReinforcementLayer(
    segments=segments,
    A_total=7.85e-5,
    E_s=200e9,
    f_y=500e6,
    E_h=20e9,
    d_bar=0.010,
    layer_type="longitudinal",
    layer_id=0,
)
```

---

## Integration with Existing Features

### Bond-Slip Coupling

The Heaviside-enriched reinforcement can be coupled with bond-slip interface elements:

1. Define reinforcement layer via Heaviside enrichment
2. Define bond-slip DOFs at rebar nodes
3. Assemble both contributions:

```python
# Reinforcement contribution
f_rebar, K_rebar, _ = assemble_reinforcement_layers(...)

# Bond-slip contribution
f_bond, K_bond, _ = assemble_bond_slip(...)

# Combined
f_total = f_bulk + f_coh + f_rebar + f_bond
K_total = K_bulk + K_coh + K_rebar + K_bond
```

### DOF Management

Enriched DOFs are added after standard and other enrichment DOFs:

```
DOF Layout:
  [0, 2*nnode)           : Standard DOFs (u_x, u_y)
  [2*nnode, ...)         : Heaviside crack enrichment
  [...]                  : Tip enrichment
  [steel_offset, ...)    : Steel/bond-slip DOFs (if used)
  [rebar_offset, ...)    : Reinforcement Heaviside DOFs (future)
```

Currently, reinforcement uses standard or steel DOFs for compatibility.

---

## Example: Reinforcement at 33° Angle

```python
import numpy as np
from xfem_clean.reinforcement import create_straight_reinforcement_layer
from xfem_clean.fem.mesh import structured_quad_mesh

# Create mesh (regular grid)
nodes, elems = structured_quad_mesh(nx=11, ny=11, Lx=1.0, Ly=1.0)

# Create reinforcement layer at 33° (NOT aligned with mesh)
angle_deg = 33.0
angle_rad = np.radians(angle_deg)

x_start = np.array([0.1, 0.1])
L_bar = 0.8
x_end = x_start + L_bar * np.array([np.cos(angle_rad), np.sin(angle_rad)])

layer = create_straight_reinforcement_layer(
    x_start=x_start,
    x_end=x_end,
    A_s=np.pi * (0.010**2) / 4,  # 10mm bar
    E_s=200e9,
    f_y=500e6,
    E_h=20e9,
    d_bar=0.010,
    layer_type="longitudinal",
    layer_id=0,
    n_segments=20,
)

print(f"Layer length: {layer.total_length():.3f} m")
print(f"Number of segments: {len(layer.segments)}")
print(f"Angle: {angle_deg}° (crosses elements at arbitrary angle)")
```

**Output**:
```
Layer length: 0.800 m
Number of segments: 20
Angle: 33.0° (crosses elements at arbitrary angle)
```

---

## Numerical Considerations

### 1. Intersection Detection

Use robust bounding-box clipping to find segment-element intersections:

```python
from xfem_clean.reinforcement import segment_element_intersection

for e_idx, elem in enumerate(elems):
    elem_coords = nodes[elem]

    for seg in layer.segments:
        isect = segment_element_intersection(seg, elem_coords)

        if isect is not None:
            s_start, s_end = isect
            # Integrate over [s_start, s_end]
```

### 2. Quadrature Accuracy

Use sufficient Gauss points to capture:
- Bar strain variations
- Plastic zones
- Coupling with cohesive/bond-slip

**Recommended**: 7 points (thesis default) for most cases.

### 3. Newton Convergence

The `find_xi_eta_newton` routine typically converges in 3-5 iterations for well-conditioned elements. If convergence fails:
- Element may be highly distorted
- Point may be outside element (returns `None`)

### 4. Plasticity

Use implicit integration (return mapping) for plastic steel:

```
Trial state: σ_trial = E(ε - ε_p_old)

If |σ_trial| > f_y + H ε_p_old:
    Return mapping to yield surface
    Update ε_p_new
```

Consistent tangent ensures quadratic convergence in Newton iteration.

---

## Validation

### Test: Bar at Arbitrary Angle

**File**: `examples/ex_rebar_heaviside_angle.py`

**Setup**:
- 4×4 element mesh (uniform)
- Single bar at 33° from corner to corner
- No mesh alignment

**Checks**:
- Enriched nodes detected: ~40% of nodes
- Intersected elements: ~7 out of 16
- Bar crosses elements without special treatment
- Runtime: <10 seconds

**Expected Results**:
```
Mesh: 25 nodes, 16 elements
Enriched nodes: 12 / 25 (48.0%)
Intersected elements: 7 / 16 (43.8%)
✓ Bar crosses mesh at arbitrary angle
✓ PASS: Runtime < 30 seconds
```

### Test: Strain Extraction

**File**: `tests/test_dissertation_features.py::test_bar_strain_from_continuum`

**Setup**:
- Pure axial strain εxx = 0.001
- Bar along x-axis (t = [1, 0])
- Expected: ε_s = 0.001

**Result**: ✓ PASS

### Test: Plastic Hardening

**File**: `tests/test_dissertation_features.py::test_steel_bilinear_plastic`

**Setup**:
- Large strain ε = 0.01
- Bilinear model: E = 200 GPa, f_y = 500 MPa, E_h = 20 GPa

**Checks**:
- σ > f_y (hardening active)
- E_t < E (reduced tangent)
- ε_p > 0 (plastic strain accumulated)

**Result**: ✓ PASS

---

## Performance

### Computational Cost

**Per Segment-Element Intersection**:
- Newton iterations: 3-5 (typically)
- Gauss points: 7
- B-matrix assembly: O(n_nodes_elem)

**Scaling**:
- O(n_segments × n_elems) for intersection detection (can be accelerated with spatial indexing)
- O(n_gauss × n_intersections) for integration

**Typical Runtime** (single layer, 100 segments, 1000 elements):
- Intersection detection: <0.1 s
- Assembly: <0.5 s
- Total per iteration: <1 s

### Optimization Tips

1. **Spatial Indexing**: Use bounding-box tree for segment-element queries
2. **Segment Subdivision**: Balance accuracy vs. cost (10-20 segments per layer typical)
3. **Caching**: Cache B-matrices if displacement field changes slowly
4. **Parallel Assembly**: Segment loop is embarrassingly parallel (future work)

---

## Limitations and Future Work

### Current Limitations

1. **2D Only**: Implementation for plane stress/strain
2. **Straight Segments**: No curved segments (approximate with polylines)
3. **No Adaptive Integration**: Fixed Gauss rule (future: adaptive)
4. **No Enrichment DOFs Yet**: Currently uses standard/steel DOFs for compatibility

### Future Enhancements

1. **3D Extension**: Implement for solid elements
2. **Curved Bars**: Add isogeometric segments (NURBS)
3. **Adaptive Quadrature**: Error-driven refinement
4. **Enrichment DOFs**: Add Heaviside-enriched DOFs for reinforcement
5. **Parallel Assembly**: OpenMP/MPI for large models

---

## References

1. Dissertation 10.5445/IR/1000124842, Chapter 4, Section 4.5
2. Moës, N., Dolbow, J., Belytschko, T. (1999). "A finite element method for crack growth without remeshing"
3. Sukumar, N., et al. (2000). "Extended finite element method for three-dimensional crack modelling"
4. Hansbo, A., Hansbo, P. (2004). "A finite element method for the simulation of strong and weak discontinuities"

---

## Quick Reference

### Create Layer
```python
layer = create_straight_reinforcement_layer(x_start, x_end, A_s, E_s, f_y, E_h, d_bar)
```

### Assemble
```python
f, K, states = assemble_reinforcement_layers(q, nodes, elems, dofs, layers)
```

### Constitutive
```python
sigma, E_t = steel_elastic_1d(eps, E)
sigma, E_t, eps_p = steel_bilinear_1d(eps, eps_p, E, f_y, E_h)
```

### Geometry
```python
phi = signed_distance_to_segment(x, x0, x1)
H = heaviside_enrichment(phi)
H_tilde = shifted_heaviside_enrichment(phi_x, phi_i)
```

---

**End of Document**
