# XFEM Nonlinear Concrete Implementation Plan

## Executive Summary

This document outlines the implementation of advanced nonlinear features for the XFEM reinforced concrete solver, evolving from elastic-linear to fully nonlinear material behavior with plasticity, damage, bond-slip, and arc-length control.

## Current Status Assessment

### ✅ ALREADY IMPLEMENTED (Phase 1 & 2 Complete)

#### 1. Material Point Object-Oriented Structure
**Location**: `src/xfem_clean/material_point.py`

```python
@dataclass
class MaterialPoint:
    eps: np.ndarray           # [3] Strain (xx, yy, xy)
    sigma: np.ndarray         # [3] Stress
    eps_p: np.ndarray         # [3] Plastic strain
    damage_t: float           # Tension damage [0,1]
    damage_c: float           # Compression damage [0,1]
    kappa: float              # Hardening variable
    w_plastic: float          # Plastic dissipation [J/m³]
    w_fract_t: float          # Tension fracture [J/m³]
    w_fract_c: float          # Compression crushing [J/m³]
    extra: Dict[str, Any]     # Model-specific data
```

**Status**: ✅ Complete - Full state variable storage with energy tracking

#### 2. Separated Constitutive Laws
**Location**: `src/xfem_clean/constitutive.py` (780+ lines)

**Available Models**:
- `LinearElasticPlaneStress`: Basic elasticity with damage scaling
- `DruckerPrager`: Associative plasticity with consistent tangent
- `ConcreteCDP`: "CDP-lite" with frozen damage
- **`ConcreteCDPReal`**: Full Lee-Fenves implementation ✅

**Key Features**:
```python
# Consistent tangent matrix (line 985-1013)
Ct_eff = self.Ce - np.outer(Cm, B) / den

# Damage operator application (line 1016-1018)
M = _damage_operator_matrix(sig_eff, dt, dc)
sig_nom = M @ sig_eff
Ct = M @ Ct_eff
```

**Status**: ✅ Complete - Includes degradation formula from prompt:
$$D_{tang} = (1-d)D_{el} - \sigma_{eff} \otimes \frac{\partial F}{\partial \sigma}$$

#### 3. Concrete Damage Plasticity (CDP)
**Location**: `src/xfem_clean/constitutive.py:700-1059`

**Implementation**:
- ✅ Lee-Fenves yield surface with `fb0/fc0` and `Kc` parameters
- ✅ Non-associative hyperbolic potential (dilation angle `psi_deg`, eccentricity `ecc`)
- ✅ Uniaxial hardening tables from `cdp_generator` calibration
- ✅ Modified return mapping algorithm (frozen flow direction)
- ✅ Consistent algorithmic tangent
- ✅ Split scalar damage (tension/compression)
- ✅ Plane stress enforcement via local Newton
- ✅ Energy dissipation tracking:
  - `w_plastic`: Plastic work on effective stress
  - `w_fract_t`: Tension fracture energy
  - `w_fract_c`: Compression crushing energy

**Compression Model**: CEB-90 curves (Hognestad-type)
**Location**: `src/cdp_generator/compression.py`

**Status**: ✅ Complete - Full CDP with all features requested in prompt

#### 4. Numba Acceleration
**Location**: `src/xfem_clean/numba/kernels_bulk.py`

**Available Kernels**:
```python
@njit(cache=True)
def elastic_integrate_plane_stress_numba(...)
def dp_integrate_plane_stress_numba(...)
def cdp_integrate_plane_stress_numba(...)  # Phase 2b
```

**Status**: ✅ Complete - Numba kernels for all material models

#### 5. Reinforcement (Embedded Truss)
**Location**: `src/xfem_clean/rebar.py`

**Current Model**:
- Embedded truss elements (perfect bond)
- Bilinear steel constitutive law
- Numba-accelerated assembly
- Parameters: `E, fy, fu, Eh, A_total`

**Status**: ✅ Basic implementation complete, ❌ Bond-slip NOT implemented

#### 6. Newton Solver with Robustness
**Location**: `src/xfem_clean/xfem/analysis_single.py:300-450`

**Features**:
- Modified Newton (secant stiffness)
- Line search (optional)
- Adaptive substepping with bisection
- Convergence criteria:
  - Residual: `||rhs|| < tol_r + beta * ||reaction||`
  - Displacement: `||du|| < tol_du`

**Status**: ✅ Robust Newton complete, ❌ Arc-length NOT implemented

---

## 🔧 IMPLEMENTATION REQUIRED

### Task 1: Bond-Slip for Reinforcement ⏳

**Objective**: Replace perfect bond with slip-capable interface following Model Code 2010

#### 1.1 Duplicate DOFs for Slip

**Concept**:
```
Standard DOF layout:
  Node i: [u_x, u_y]

With bond-slip:
  Concrete node i: [u_cx, u_cy]
  Steel node i: [u_sx, u_sy]  (only tangential slip active)

  Slip: s = (u_s - u_c) · t_bar
```

**Implementation**:
1. Extend `XFEMDofs` to include steel DOFs:
   ```python
   class XFEMDofs:
       ndof: int
       std: np.ndarray        # Concrete DOFs [nnode, 2]
       steel: np.ndarray      # Steel DOFs [n_bar_nodes, 2]
       steel_nodes: np.ndarray  # Which nodes have steel
   ```

2. Modify DOF numbering:
   ```python
   # Concrete DOFs: 0 to 2*n_concrete_nodes - 1
   # Steel DOFs: 2*n_concrete_nodes to 2*n_concrete_nodes + 2*n_steel_nodes - 1
   ```

#### 1.2 1D Interface Element with τ(s) Law (Model Code 2010)

**Bond-Slip Constitutive Law**:
```python
class BondSlipModelCode2010:
    """Model Code 2010 bond-slip relationship for ribbed bars."""

    def __init__(self, f_cm: float, d_bar: float, condition: str = "good"):
        """
        f_cm: Mean concrete compressive strength [Pa]
        d_bar: Bar diameter [m]
        condition: "good" or "poor" bond conditions
        """
        # MC2010 parameters
        if condition == "good":
            self.tau_max = 2.5 * sqrt(f_cm)  # [Pa]
            self.s1 = 1.0e-3  # [m] - slip at tau_max
            self.s2 = 2.0e-3  # [m] - slip at start of plateau
            self.s3 = d_bar   # [m] - slip at start of residual
            self.alpha = 0.4  # rising branch exponent
        else:  # poor condition
            self.tau_max = 1.25 * sqrt(f_cm)
            self.s1 = 1.8e-3
            self.s2 = 3.6e-3
            self.s3 = d_bar
            self.alpha = 0.4

        self.tau_f = 0.15 * self.tau_max  # residual bond stress

    def tau_and_dtau(self, s: float, s_max_history: float) -> Tuple[float, float]:
        """Return bond stress and tangent stiffness.

        Piecewise law:
        - 0 ≤ s ≤ s1: τ = τ_max * (s/s1)^α
        - s1 < s ≤ s2: τ = τ_max (plateau)
        - s2 < s ≤ s3: τ = τ_max - (τ_max - τ_f) * (s - s2)/(s3 - s2)
        - s > s3: τ = τ_f (residual)

        Unloading/reloading: secant to s_max
        """
        s_abs = abs(s)
        sign = 1.0 if s >= 0 else -1.0

        # Loading envelope
        if s_abs >= s_max_history:
            if s_abs <= self.s1:
                tau = self.tau_max * (s_abs / self.s1) ** self.alpha
                dtau = self.tau_max * self.alpha / self.s1 * (s_abs / self.s1) ** (self.alpha - 1)
            elif s_abs <= self.s2:
                tau = self.tau_max
                dtau = 0.0
            elif s_abs <= self.s3:
                tau = self.tau_max - (self.tau_max - self.tau_f) * (s_abs - self.s2) / (self.s3 - self.s2)
                dtau = -(self.tau_max - self.tau_f) / (self.s3 - self.s2)
            else:
                tau = self.tau_f
                dtau = 0.0
        else:
            # Unloading: secant stiffness to s_max
            tau_max_reached, _ = self.tau_and_dtau(s_max_history, s_max_history)
            tau = tau_max_reached * (s_abs / s_max_history) if s_max_history > 1e-14 else 0.0
            dtau = tau_max_reached / max(s_max_history, 1e-14)

        return sign * tau, dtau
```

**Interface Element Assembly**:
```python
def assemble_bond_interface(
    nodes: np.ndarray,
    steel_segments: np.ndarray,
    u_concrete: np.ndarray,
    u_steel: np.ndarray,
    bond_law: BondSlipModelCode2010,
    perimeter: float,  # π * d_bar
    s_max_history: np.ndarray,  # [n_segments] history variable
) -> Tuple[np.ndarray, sparse.csr_matrix]:
    """
    Returns (f_bond, K_bond) contributions to global system.

    For each bar segment of length L:
      - Integrate bond stress along length
      - Contribute forces to both concrete and steel DOFs
    """
    pass  # Implementation in new file: bond_slip.py
```

#### 1.3 Dowel Action at Crack-Rebar Intersections

**Concept**: When a crack crosses a rebar, the bar provides shear resistance (dowel effect)

**Model**: Elastic foundation model
```
F_dowel = k_dowel * Δu_transverse

k_dowel = E_s * I_bar / L_e^3
where L_e is effective embedment length
```

**Implementation**:
```python
def compute_dowel_stiffness(
    crack: XFEMCrack,
    steel_segments: np.ndarray,
    nodes: np.ndarray,
    E_steel: float,
    d_bar: float,
) -> List[Tuple[int, int, float, float]]:
    """
    Find crack-rebar intersections and return dowel springs.

    Returns: list of (node_concrete, node_steel, k_normal, k_tangent)
    """
    I_bar = np.pi * d_bar**4 / 64  # moment of inertia
    L_e = 10 * d_bar  # effective length (empirical)
    k_dowel = E_steel * I_bar / L_e**3

    intersections = []
    # Find segments that cross crack
    for seg in steel_segments:
        if crack_crosses_segment(crack, seg, nodes):
            # Add transverse stiffness at intersection
            intersections.append((seg.n_concrete, seg.n_steel, k_dowel, 0.0))

    return intersections
```

**Assembly**: Add to global K during cohesive assembly

---

### Task 2: Arc-Length Control ⏳

**Objective**: Enable solver to pass through load-displacement peaks (snapback)

#### 2.1 Arc-Length Equation

**Standard Newton**:
```
K * du = λ * P_ext - f_int
```

**Arc-Length** (Crisfield):
```
K * du = λ * P_ext - f_int  (equilibrium)
du^T * du + ψ² * dλ² * P^T * P = Δl²  (constraint)
```

Where:
- `λ`: load factor
- `Δl`: arc-length step size (adaptive)
- `ψ`: scaling factor (typically `ψ = 1 / ||P||`)

#### 2.2 Implementation Strategy

**New file**: `src/xfem_clean/xfem/arc_length.py`

```python
def newton_arc_length_step(
    K: sparse.csr_matrix,
    f_int: np.ndarray,
    P_ref: np.ndarray,  # Reference load pattern
    lambda_n: float,    # Current load factor
    u_n: np.ndarray,    # Current displacement
    arc_length: float,  # Current arc-length
    fixed: Dict[int, float],
    model: XFEMModel,
) -> Tuple[bool, np.ndarray, float, float]:
    """
    Perform one arc-length controlled step.

    Returns:
        converged: bool
        u_new: np.ndarray
        lambda_new: float
        arc_length_new: float (adaptive)
    """

    # Predictor: tangent solution
    du_bar = spla.spsolve(K, P_ref)

    psi = 1.0 / max(np.linalg.norm(P_ref), 1e-12)

    # Initial increment (tangent predictor)
    denom = np.dot(du_bar, du_bar) + psi**2 * np.dot(P_ref, P_ref)
    dlambda_1 = arc_length / np.sqrt(denom)
    du_1 = dlambda_1 * du_bar

    u_try = u_n + du_1
    lambda_try = lambda_n + dlambda_1

    # Corrector iterations
    for it in range(model.newton_maxit):
        # Residual
        f_int_try = assemble_internal_forces(u_try, ...)
        r = lambda_try * P_ref - f_int_try

        # Solve two systems
        du_I = spla.spsolve(K, r)  # residual correction
        du_II = spla.spsolve(K, P_ref)  # load correction

        # Arc-length constraint
        a = np.dot(du_I, du_I) + psi**2 * np.dot(P_ref, P_ref)
        b = 2 * (np.dot(u_try - u_n, du_I) + psi**2 * (lambda_try - lambda_n) * np.dot(P_ref, P_ref))
        c = np.dot(u_try - u_n, u_try - u_n) + psi**2 * (lambda_try - lambda_n)**2 * np.dot(P_ref, P_ref) - arc_length**2

        # Solve quadratic for dlambda
        disc = b**2 - 4*a*c
        if disc < 0:
            return False, u_try, lambda_try, arc_length

        dlambda_a = (-b + np.sqrt(disc)) / (2*a)
        dlambda_b = (-b - np.sqrt(disc)) / (2*a)

        # Choose solution with positive load increment (forward stepping)
        dlambda = dlambda_a if abs(dlambda_a) < abs(dlambda_b) else dlambda_b

        du = du_I + dlambda * du_II

        # Update
        u_try += du
        lambda_try += dlambda

        # Check convergence
        if np.linalg.norm(r) < model.newton_tol_r and np.linalg.norm(du) < model.newton_tol_du:
            # Adaptive arc-length update
            arc_length_new = arc_length * min(2.0, float(model.newton_maxit) / max(it+1, 1))
            return True, u_try, lambda_try, arc_length_new

    # Reduce arc-length and retry
    return False, u_try, lambda_try, arc_length * 0.5
```

#### 2.3 Integration with Existing Solver

Modify `run_analysis_xfem` to add arc-length mode:

```python
def run_analysis_xfem(
    model: XFEMModel,
    ...,
    control_mode: str = "displacement",  # "displacement" or "arc_length"
    arc_length_initial: float = 0.01,
):
    ...

    if control_mode == "arc_length":
        # Use arc-length controlled stepping
        converged, q_new, lambda_new, arc_new = newton_arc_length_step(...)
    else:
        # Use displacement control (existing)
        converged, q_new, ... = newton_displacement_step(...)
```

---

### Task 3: Enhanced Output ⏳

#### 3.1 Energy Dissipation Tracking

**Current Status**: Energy densities computed in `MaterialPoint`:
- `w_plastic`: Plastic dissipation [J/m³]
- `w_fract_t`: Tension fracture [J/m³]
- `w_fract_c`: Compression crushing [J/m³]

**Required**: Global energy summation and export

**Implementation**:
```python
def compute_global_energies(
    mp_states: BulkStateArrays,
    coh_states: CohesiveStateArrays,
    elems: np.ndarray,
    volumes: np.ndarray,  # element volumes
    coh_areas: np.ndarray,  # cohesive element areas
) -> Dict[str, float]:
    """
    Compute global energy dissipation.

    Returns:
        {
            "W_plastic": float,  # Total plastic work [J]
            "W_fract_t": float,  # Total tension fracture [J]
            "W_fract_c": float,  # Total compression crushing [J]
            "W_cohesive": float, # Cohesive crack energy [J]
            "W_total": float,    # Total dissipation [J]
        }
    """
    W_plastic = 0.0
    W_fract_t = 0.0
    W_fract_c = 0.0

    for ie, elem in enumerate(elems):
        n_ip = mp_states.eps.shape[1]  # integration points per element
        for ip in range(n_ip):
            mp = mp_states.get_mp(ie, ip)
            dV = volumes[ie] / n_ip  # volume per IP

            W_plastic += mp.w_plastic * dV
            W_fract_t += mp.w_fract_t * dV
            W_fract_c += mp.w_fract_c * dV

    # Cohesive energy from crack
    W_cohesive = 0.0
    for ic in range(coh_states.damage.shape[0]):
        for ie in range(coh_states.damage.shape[1]):
            for ip in range(coh_states.damage.shape[2]):
                dmg = coh_states.damage[ic, ie, ip]
                delta_max = coh_states.delta_max[ic, ie, ip]
                # Integrate cohesive law
                W_cohesive += integrate_cohesive_law(dmg, delta_max) * coh_areas[ic, ie, ip]

    return {
        "W_plastic": W_plastic,
        "W_fract_t": W_fract_t,
        "W_fract_c": W_fract_c,
        "W_cohesive": W_cohesive,
        "W_total": W_plastic + W_fract_t + W_fract_c + W_cohesive,
    }
```

**Export**: Add to step-by-step results

#### 3.2 Damage Field Visualization

**Objective**: Export compression damage field to identify crushing zones (top chord of beam)

**Implementation**:
```python
def export_damage_fields(
    nodes: np.ndarray,
    elems: np.ndarray,
    mp_states: BulkStateArrays,
    filename: str = "damage_field.vtk",
):
    """
    Export damage fields to VTK for ParaView visualization.

    Fields:
        - damage_compression: dc at integration points (averaged to nodes)
        - damage_tension: dt at integration points
        - plastic_strain_mag: ||eps_p||
    """
    # Average IP values to nodes
    damage_c_nodes = np.zeros(len(nodes))
    damage_t_nodes = np.zeros(len(nodes))
    eps_p_mag_nodes = np.zeros(len(nodes))
    count_nodes = np.zeros(len(nodes))

    for ie, elem in enumerate(elems):
        n_ip = mp_states.eps.shape[1]
        for ip in range(n_ip):
            mp = mp_states.get_mp(ie, ip)
            # Average to element nodes
            for node in elem:
                damage_c_nodes[node] += mp.damage_c
                damage_t_nodes[node] += mp.damage_t
                eps_p_mag_nodes[node] += np.linalg.norm(mp.eps_p)
                count_nodes[node] += 1

    damage_c_nodes /= np.maximum(count_nodes, 1)
    damage_t_nodes /= np.maximum(count_nodes, 1)
    eps_p_mag_nodes /= np.maximum(count_nodes, 1)

    # Write VTK
    write_vtk_unstructured_grid(
        filename,
        nodes,
        elems,
        point_data={
            "damage_compression": damage_c_nodes,
            "damage_tension": damage_t_nodes,
            "plastic_strain_magnitude": eps_p_mag_nodes,
        },
    )
```

---

## Implementation Timeline

### Phase 3a: Bond-Slip (Priority: High)
1. ✅ Review existing rebar implementation
2. [ ] Implement `BondSlipModelCode2010` class
3. [ ] Extend DOF management for steel nodes
4. [ ] Implement interface element assembly
5. [ ] Add dowel action at crack intersections
6. [ ] Test with single bar pull-out
7. [ ] Test with beam under bending

### Phase 3b: Arc-Length (Priority: Medium)
1. [ ] Implement `newton_arc_length_step`
2. [ ] Add adaptive arc-length control
3. [ ] Integrate with `run_analysis_xfem`
4. [ ] Test with snap-back beam problem
5. [ ] Validate load-displacement curves

### Phase 3c: Enhanced Output (Priority: Low)
1. [ ] Implement global energy computation
2. [ ] Add VTK export for damage fields
3. [ ] Integrate with step-by-step output
4. [ ] Create visualization examples

---

## Testing & Validation

### Test Case 1: Four-Point Bending Beam
- Geometry: L=1.0m, H=0.2m, cover=0.02m
- Material: C30/37 concrete (fc=38 MPa, Ec=33 GPa)
- Steel: fy=500 MPa, Es=200 GPa
- Loading: Displacement control → Arc-length
- Expected:
  - Crack initiation at mid-span
  - Steel yielding
  - Compression crushing at top chord
  - Softening branch in load-displacement

### Test Case 2: Pull-Out Test
- Single bar embedded in concrete block
- Measure bond-slip relationship
- Compare with Model Code 2010 analytical solution

### Test Case 3: Dowel Action
- Crack crossing reinforcement bar
- Shear load application
- Validate dowel stiffness contribution

---

## References

1. **CDP Model**: Lee, J., & Fenves, G. L. (1998). "Plastic-damage model for cyclic loading of concrete structures." Journal of Engineering Mechanics.

2. **Bond-Slip**: fib Model Code 2010, Section 6.1.2 "Bond and anchorage"

3. **Arc-Length**: Crisfield, M. A. (1981). "A fast incremental/iterative solution procedure that handles 'snap-through'." Computers & Structures.

4. **Dowel Action**: Vintzēleou, E. N., & Tassios, T. P. (1987). "Mathematical models for dowel action under monotonic and cyclic conditions." Magazine of Concrete Research.

---

## File Structure (New Files)

```
src/xfem_clean/
├── bond_slip.py              # NEW: Bond-slip constitutive law and assembly
├── arc_length.py             # NEW: Arc-length solver
├── output/
│   ├── energy.py             # NEW: Global energy computation
│   └── vtk_export.py         # NEW: VTK damage field export
└── xfem/
    ├── analysis_single.py    # MODIFY: Add arc-length integration
    └── dofs_single.py        # MODIFY: Extend for steel DOFs
```

---

## Conclusion

The XFEM concrete solver has a solid foundation with complete CDP implementation. The remaining work focuses on:
1. **Bond-slip** for realistic reinforcement behavior
2. **Arc-length** for robust post-peak analysis
3. **Enhanced output** for better understanding of failure mechanisms

All proposed features follow the mathematical framework outlined in the original prompt and integrate seamlessly with the existing codebase.
