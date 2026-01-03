# TASK 5: Complete Physical Dissipation Tracking Guide

**Status:** Cohesive dissipation complete ✅ | Bond & bulk dissipation remaining ⏳

---

## Summary

Physical dissipation tracking decomposes the algorithmic dissipation `ΔD_alg` (computed as a remainder in `energy_hht.py`) into:
- **Physical dissipation**: Irreversible energy loss (fracture, plasticity, friction)
  - `ΔD_coh`: Cohesive zone dissipation (crack formation)
  - `ΔD_bond`: Bond-slip dissipation (rebar-concrete interface friction)
  - `ΔD_bulk`: Bulk plastic dissipation (concrete crushing, steel yielding)
- **Numerical dissipation**: HHT-α algorithmic damping (α < 0)

For energy-conserving schemes (α=0, no damping), we expect:
```
ΔD_numerical = ΔD_alg - ΔD_physical ≈ 0
```

For dissipative schemes (α<0), `ΔD_numerical > 0` provides controlled high-frequency damping.

---

## Completed: Cohesive Dissipation (Commit fe4a2b4)

### Implementation (`assembly_single.py:682-711`)

**Trapezoidal Rule**:
```python
if compute_dissipation and q_prev is not None:
    # Compute old opening
    delta_old = np.dot(gvec_n, q_prev[edofs])

    # Get old traction by evaluating law at old state
    T_old, _, _ = cohesive_update(law, delta_old, st_committed)

    # Dissipation increment (trapezoidal)
    dD = 0.5 * (T_old + T_new) * (delta_new - delta_old) * w_gp * thickness
    D_coh_inc += dD
```

**Mixed-Mode**:
```python
# For mixed-mode, integrate both normal and tangential work
d_delta_n = delta_n_new - delta_n_old
d_delta_t = delta_t_new - delta_t_old
dD = 0.5 * ((tn_old + tn_new) * d_delta_n + (tt_old + tt_new) * d_delta_t) * w_gp
```

### Validation

**Formula Test** (`test_dissipation_formula.py`):
- ✅ Elastic work: Exact match for linear regime
- ✅ **Complete fracture**: Total dissipation = 99.944 J/m vs Gf = 100.0 J/m (0.056% error)
- ✅ Monotonic positive dissipation in softening regime

**Physical Interpretation**:
- Elastic loading: Work is stored as recoverable energy
- Softening: Work is irreversibly dissipated (damage)
- Unloading: Elastic energy recovered, dissipation unchanged
- Cyclic loading: Dissipation accumulates monotonically

---

## Remaining: Bond-Slip Dissipation

### Physics

Bond-slip dissipation is the work done by bond shear stress τ over relative slip s:
```
ΔD_bond = Σ_segments Σ_GPs  0.5 * (τ_old + τ_new) * (s_new - s_old) * perimeter * L_gp
```

Where:
- `perimeter`: Rebar perimeter (π*d for circular bars)
- `L_gp`: Gauss point spacing along segment
- `τ`: Bond shear stress [Pa]
- `s`: Relative slip [m]

### Implementation Strategy

**Option 1: Modify `assemble_bond_slip()` (recommended for full integration)**

Add parameters to `bond_slip.py::assemble_bond_slip()`:
```python
def assemble_bond_slip(
    u_total: np.ndarray,
    # ... existing parameters ...
    u_total_prev: Optional[np.ndarray] = None,
    compute_dissipation: bool = False,
) -> Tuple[np.ndarray, sp.csr_matrix, BondSlipStateArrays, float]:
    """
    Returns
    -------
    f_bond, K_bond, bond_states_new, D_bond_inc  # Added dissipation
    """
```

During segment loop (around line 1400-1500):
```python
for iseg in range(n_seg):
    for igp in gauss_points:
        # Compute slip from current displacement
        s_new = compute_slip(u_total, segment, igp)

        # Evaluate bond law
        tau_new, dtau_ds, s_max_new = bond_update(s_new, s_max_old)

        if compute_dissipation and u_total_prev is not None:
            # Compute old slip
            s_old = compute_slip(u_total_prev, segment, igp)

            # Get old stress (evaluate at old slip with old state)
            tau_old, _, _ = bond_update(s_old, s_max_old)

            # Trapezoidal dissipation
            dD = 0.5 * (tau_old + tau_new) * (s_new - s_old) * perimeter * L_gp
            D_bond_inc += dD
```

**Option 2: Post-hoc computation in `assembly_single.py` (simpler, less integrated)**

After calling `assemble_bond_slip()` in `assembly_single.py:781-803`:
```python
# Call bond assembly
f_bond, K_bond, layer_updates = assemble_bond_slip(...)

# Compute dissipation post-hoc (if enabled)
if compute_dissipation and q_prev is not None:
    D_bond_inc += compute_bond_dissipation_layer(
        q, q_prev, layer, layer_states_comm, layer_updates
    )
```

Where `compute_bond_dissipation_layer()` is a helper function that:
1. Loops over segments and GPs
2. Computes old and new slip
3. Evaluates bond law at both
4. Accumulates trapezoidal dissipation

### Testing

Create `tests/test_bond_dissipation_tracking.py`:
```python
def test_bond_dissipation_monotonic_loading():
    """Test that monotonic bond slip produces positive dissipation."""
    # Setup: simple pullout test
    # Step n: s = 0.1 mm
    # Step n+1: s = 0.2 mm
    # Expected: D > 0 (friction work)

def test_bond_dissipation_cyclic_loading():
    """Test that loading-unloading cycle dissipates energy."""
    # Setup: load to s=0.2mm, unload to s=0
    # Expected: D > 0 (hysteresis loop area)

def test_bond_dissipation_total_matches_integral():
    """Test that incremental dissipation sums to total work."""
    # Integrate from s=0 to s=s_max in many small steps
    # Compare to analytical integral of bond law
```

### Estimated Time: 2-3 hours

---

## Remaining: Bulk Plastic Dissipation

### Physics

Bulk plastic dissipation is the work done by stress over plastic strain:
```
ΔD_bulk = Σ_elements Σ_GPs  σ : Δε_plastic * detJ * w_gp * thickness
```

For Concrete Damaged Plasticity (CDP):
- Tension: Damage evolution with cracking strain
- Compression: Plastic flow with hardening/softening
- Dissipation = stress × plastic strain increment

### Implementation Strategy

**Modify `bulk_internal_force()` in `xfem_beam.py` (or similar in your solver)**

Add parameter:
```python
def bulk_internal_force(
    nodes, elems, mat, u, states_committed, thickness,
    precomp=None,
    compute_psi_bulk: bool = False,
    u_prev: Optional[np.ndarray] = None,
    compute_dissipation: bool = False,
) -> Tuple[np.ndarray, List, np.ndarray, np.ndarray, float, float]:
    """
    Returns
    -------
    f_int, states_trial, gp_stress, gp_rankine, psi_bulk, D_bulk_inc
    """
```

During GP loop (around line 785-796):
```python
for igp, (xi, eta) in enumerate(gauss):
    eps_new = B @ ue_new
    sig_new, st_new, _ = cdp_update(eps_new, st_old, mat)

    if compute_dissipation and u_prev is not None:
        # Compute old strain
        ue_old = extract_element_dofs(u_prev, conn)
        eps_old = B @ ue_old

        # Get old stress (at old strain with old state)
        sig_old, _, _ = cdp_update(eps_old, st_old, mat)

        # Plastic strain increment
        d_eps_p = st_new.eps_p - st_old.eps_p

        # Dissipation (average stress × plastic strain)
        dD = 0.5 * (sig_old + sig_new) @ d_eps_p * detJ * w_gp * thickness
        D_bulk_inc += dD
```

**Note**: For elastic material (no plasticity), `Δε_p = 0` → `D_bulk = 0` ✓

### Alternative: Track Dissipation in Constitutive Update

Some constitutive models (e.g., plasticity) can directly compute dissipation:
```python
class CDPMaterial:
    def integrate(self, mp: MaterialPoint, eps: np.ndarray) -> float:
        """
        Returns dissipation increment for this integration step.
        """
        # Compute plastic strain increment
        d_eps_p = ...

        # Dissipation = σ : Δε_p
        D_inc = np.dot(mp.sig, d_eps_p)

        return D_inc
```

Then accumulate in assembly:
```python
for igp in range(n_gp):
    D_inc = material.integrate(mp, eps)
    D_bulk_inc += D_inc * detJ * w_gp * thickness
```

### Testing

Create `tests/test_bulk_dissipation_tracking.py`:
```python
def test_elastic_no_dissipation():
    """Test that elastic deformation has zero plastic dissipation."""
    # Linear elastic material
    # Load and unload
    # Expected: D_bulk = 0

def test_plastic_compression_dissipation():
    """Test that plastic compression dissipates energy."""
    # CDP material loaded beyond yield
    # Expected: D_bulk > 0

def test_damage_dissipation():
    """Test that damage evolution dissipates energy."""
    # Tension with softening
    # Expected: D_bulk = damage dissipation
```

### Estimated Time: 3-4 hours

---

## Integration with Energy Framework

Once all three dissipation components are implemented, integrate into `energy_hht.py`:

### Modify `compute_step_energy()`

Add parameter:
```python
def compute_step_energy(
    # ... existing parameters ...
    D_physical_inc: float = 0.0,  # Total physical dissipation
) -> StepEnergy:
```

Update algorithmic dissipation calculation:
```python
# Physical dissipation (from assembly)
ΔD_physical = D_physical_inc

# Algorithmic dissipation (remainder)
ΔD_alg = ΔW_dir - (ΔE_mech + ΔD_damp)

# Numerical dissipation (pure HHT-α damping)
ΔD_numerical = ΔD_alg - ΔD_physical
```

### Extend `StepEnergy` dataclass

```python
@dataclass
class StepEnergy:
    # ... existing fields ...

    # Physical dissipation breakdown
    D_coh_inc: float = 0.0
    D_bond_inc: float = 0.0
    D_bulk_inc: float = 0.0
    D_physical_inc: float = 0.0  # Sum of above

    # Numerical dissipation (HHT-α)
    D_numerical_inc: float = 0.0
```

### Solver Integration

In time integrator (e.g., `xfem_beam.py`), when calling assembly for final converged state:

```python
# Final assembly at converged n+1 with dissipation tracking
K_np1, f_np1, coh_states_np1, mp_states_np1, aux, bond_states_np1, _, _ = (
    assemble_xfem_system(
        # ... parameters ...
        q=u_np1,
        q_prev=u_n,  # Previous converged displacement
        compute_dissipation=True,  # Enable dissipation
    )
)

# Extract dissipation from aux
D_coh = aux["D_coh_inc"]
D_bond = aux["D_bond_inc"]
D_bulk = aux["D_bulk_plastic_inc"]
D_physical = D_coh + D_bond + D_bulk

# Pass to energy tracker
step_energy = compute_step_energy(
    # ... other parameters ...
    D_physical_inc=D_physical,
)
```

---

## Validation Strategy

### Energy Conservation Test (α=0, no damping)

```python
def test_energy_conservation_elastic():
    """Test perfect energy conservation for elastic system."""
    # Setup: α=0, no Rayleigh damping, elastic material
    # Run simulation
    # Check: |ΔD_numerical| < 1e-6  (numerical errors only)
```

### Physical Dissipation Test

```python
def test_physical_dissipation_vs_fracture_energy():
    """Test that cohesive dissipation matches Gf."""
    # Setup: Pure Mode I fracture
    # Open crack to complete failure
    # Check: D_coh ≈ Gf * crack_area
```

### Algorithmic Dissipation Test (α<0)

```python
def test_algorithmic_dissipation_hht():
    """Test that HHT-α produces controlled numerical dissipation."""
    # Setup: α=-0.1, elastic material
    # Apply transient load
    # Check: ΔD_numerical > 0 (high-freq damping)
    # Check: ΔD_physical = 0 (no physical damage)
```

---

## Summary of Remaining Work

| Task | Estimated Time | Priority | Notes |
|------|---------------|----------|-------|
| Bond dissipation implementation | 2-3h | Medium | Straightforward extension of cohesive approach |
| Bulk dissipation implementation | 3-4h | Medium | Requires CDP integration |
| Energy framework integration | 1-2h | High | Connect all pieces |
| Comprehensive testing | 2-3h | High | Validate energy balance |
| **Total** | **8-12h** | - | - |

---

## References

1. **Cohesive dissipation** (✅ Complete):
   - Formula validated: `test_dissipation_formula.py`
   - Total dissipation matches Gf within 0.056%
   - Trapezoidal integration exact for linear segments

2. **Energy framework** (`energy_hht.py`):
   - Already tracks kinetic, bulk recoverable, damping, constraint work
   - Computes `ΔD_alg` as remainder
   - Need to decompose into physical + numerical components

3. **HHT-α Theory**:
   - α=0: Newmark average acceleration (energy-conserving)
   - α∈[-1/3, 0): Controlled high-frequency dissipation
   - `ΔD_numerical = f(α, ω_high)` - dissipates high frequencies preferentially

---

**Implementation Status**: ~24h complete, ~8-12h remaining for TASK 5
