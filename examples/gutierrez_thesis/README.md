# Gutiérrez Thesis Case Suite

**Complete XFEM examples from Chapter 5 + Appendix A**

This suite provides reproducible implementations of all major test cases from the Gutiérrez thesis on XFEM modeling of reinforced concrete. Each case can be run from the CLI without needing access to the PDF, with all parameters, equations, and validation data coded directly.

---

## Quick Start

```bash
# List available cases
python -m examples.gutierrez_thesis.run --list

# Run a single case
python -m examples.gutierrez_thesis.run --case pullout --mesh medium

# Run with different mesh density
python -m examples.gutierrez_thesis.run --case STN12 --mesh fine

# Dry run (config only, no solver)
python -m examples.gutierrez_thesis.run --case pullout --dry-run
```

---

## Status Overview

### ✅ Completed Infrastructure

| Component | Status | Description |
|-----------|--------|-------------|
| **Case configuration system** | ✅ Complete | Dataclasses for geometry, materials, reinforcement, bond laws, loading |
| **Runner CLI** | ✅ Complete | Unified CLI with case registry, mesh presets, dry-run mode |
| **Bond laws** | ✅ Complete | CEB-FIP (custom params), Bilinear (FRP), Banholzer (fibres) |
| **Postprocessing** | ✅ Complete | CTOD, slip profiles, crack widths, steel forces, base moment |
| **Case 01 definition** | ✅ Complete | Pull-out Lettow with all parameters |

### 🚧 Remaining Work (Integration Required)

| Component | Status | Priority | Estimated Effort |
|-----------|--------|----------|------------------|
| **Solver integration** | 🔴 Not started | **Critical** | 3-5 days |
| **Subdomain support** | 🔴 Not started | High | 2-3 days |
| **FRP sheet reinforcement** | 🔴 Not started | Medium | 2-3 days |
| **Fibre generator** | 🔴 Not started | Medium | 2-3 days |
| **Cyclic loading** | 🔴 Not started | Low | 1-2 days |
| **Cases 02-10** | 🔴 Not started | High | 5-7 days |
| **Testing** | 🔴 Not started | High | 2-3 days |

**Total remaining effort estimate: 17-28 days**

---

## Architecture

### Directory Structure

```
examples/gutierrez_thesis/
├── __init__.py              # Package initialization
├── README.md                # This file
├── run.py                   # CLI runner
├── case_config.py           # Configuration dataclasses
├── postprocess.py           # Post-processing utilities
├── cases/                   # Case definitions
│   ├── case_01_pullout_lettow.py
│   ├── case_02_sspot_frp.py        # TODO
│   ├── case_03_tensile_stn12.py    # TODO
│   └── ...
├── outputs/                 # Output directory (generated)
│   ├── case_01/
│   │   ├── load_displacement.csv
│   │   ├── crack_pattern.png
│   │   ├── slip_profile.csv
│   │   └── metrics.json
│   └── ...
└── tests/                   # Smoke tests
    └── test_thesis_cases.py       # TODO
```

### Configuration System

Each case is defined using dataclasses:

```python
from examples.gutierrez_thesis.case_config import (
    CaseConfig, GeometryConfig, ConcreteConfig,
    CustomBondSlipLaw, RebarLayer, MonotonicLoading
)

# Define geometry
geometry = GeometryConfig(
    length=200.0,  # mm
    height=200.0,
    thickness=200.0,
    n_elem_x=10,
    n_elem_y=10,
)

# Define concrete
concrete = ConcreteConfig(
    E=26287.0,  # MPa
    nu=0.2,
    f_c=30.0,
    f_t=3.0,
    G_f=0.1,
)

# Define bond law (custom parameters)
bond_law = CustomBondSlipLaw(
    s1=0.77,  # mm
    s2=1.37,
    s3=7.5,
    tau_max=11.5,  # MPa
    tau_f=4.0,
    alpha=0.4,
)

# Assemble case
case = CaseConfig(
    case_id="01_pullout_lettow",
    description="Pull-out test",
    geometry=geometry,
    concrete=concrete,
    rebar_layers=[...],
    loading=loading,
    outputs=outputs,
)
```

---

## Bond-Slip Laws

### 1. CustomBondSlipLaw (CEB-FIP with custom parameters)

```python
from xfem_clean.bond_slip import CustomBondSlipLaw

bond = CustomBondSlipLaw(
    s1=0.77,    # End of rising branch (mm)
    s2=1.37,    # End of plateau (mm)
    s3=7.5,     # End of softening (mm)
    tau_max=11.5,  # Maximum bond stress (MPa)
    tau_f=4.0,     # Residual stress (MPa)
    alpha=0.4,     # Exponent in rising branch
    use_secant_stiffness=True,  # For convergence
)

# Compute bond stress and tangent
tau, dtau_ds = bond.tau_and_tangent(s=0.5, s_max_history=0.5)
```

**Equation:**
```
τ(s) =
  τ_max * (s/s1)^α                     if 0 ≤ s ≤ s1
  τ_max                                if s1 < s ≤ s2
  τ_max - (τ_max - τ_f)*(s-s2)/(s3-s2) if s2 < s ≤ s3
  τ_f                                  if s > s3
```

### 2. BilinearBondLaw (for FRP sheets)

```python
from xfem_clean.bond_slip import BilinearBondLaw

bond = BilinearBondLaw(
    s1=0.5,     # End of hardening (mm)
    s2=1.0,     # Complete debonding (mm)
    tau1=1.0,   # Peak stress (MPa)
    use_secant_stiffness=True,
)
```

**Equation:**
```
τ(s) =
  (τ1/s1)*s                    if 0 ≤ s ≤ s1
  τ1*(1 - (s-s1)/(s2-s1))      if s1 < s ≤ s2
  0                            if s > s2
```

### 3. BanholzerBondLaw (for fibres)

```python
from xfem_clean.bond_slip import BanholzerBondLaw

bond = BanholzerBondLaw(
    s0=0.01,    # End of rising (mm)
    a=20.0,     # Softening multiplier
    tau1=5.0,   # Peak in rising (MPa)
    tau2=3.0,   # Start of softening (MPa)
    tau_f=2.25, # Residual (MPa)
    use_secant_stiffness=True,
)
```

**Equation:**
```
τ(s) =
  (τ1/s0)*s                                  if 0 ≤ s ≤ s0
  τ2 - (τ2 - τ_f)*(s - s0)/((a-1)*s0)        if s0 < s ≤ a*s0
  τ_f                                        if s > a*s0
```

All laws include:
- **Unloading/reloading:** Secant path to historical maximum
- **Secant stiffness:** Use `tau/s` instead of `dtau/ds` for stability
- **History tracking:** State variable `s_max_history` tracks maximum slip

---

## Case Definitions

### Case 01: Pull-Out (Lettow)

**Geometry:** 200×200×200 mm block, Ø12 bar, 36 mm embedment

**Features:**
- Load element (displacement control)
- Empty element (E=0, bond disabled)
- Custom bond law (calibrated parameters)

**Parameters:**
```python
Concrete: E=26287 MPa, ν=0.2
Steel: E=200000 MPa, ν=0.3
Bond: s1=0.77, s2=1.37, s3=7.5, τ_max=11.5, τ_f=4.0 (mm, MPa)
```

**Expected Output:**
- Load-slip curve
- Slip profile along bar
- Bond stress profile
- Comparison with 1D analytical solution (optional)

**Status:** ✅ Configuration complete, 🔴 Solver integration pending

---

### Case 02: SSPOT FRP Sheet

**Geometry:** FRP sheet bonded to concrete block

**Features:**
- Externally bonded reinforcement (FRP)
- Bilinear bond law
- Debonding progression

**Parameters:**
```python
Sheet: 0.05×40 mm², E=100000 MPa, bonded length 150 mm
Bond: s1=0.5, s2=1.0, τ1=1.0 (mm, MPa)
```

**Status:** 🔴 Not implemented

---

### Case 03: Tensile Member STN12 (Wu et al.)

**Geometry:** 100×100×1100 mm prism, Ø12 bar

**Features:**
- Multiple cracking
- Tension stiffening
- Crack spacing analysis
- Half-symmetry model

**Parameters:**
```python
Concrete: E=22.4 GPa, f_c=21.6 MPa, f_t=2.04 MPa, G_f=0.051 N/mm
Steel: E=200 GPa, f_y=520 MPa
Bond: s1=1, s2=2, s3=5, τ_max=11.61, τ_f=4.64 (mm, MPa)
```

**Status:** 🔴 Not implemented

---

### Case 04: Uniaxial FRC (Sorelli)

**Geometry:** Specimen with notch

**Features:**
- Fibre reinforcement (explicit fibre generation)
- Banholzer bond law
- Orientation sensitivity (0°, 45°, 90°)
- Volume fraction study

**Parameters:**
```python
Concrete: E=25.1 GPa, f_c=28.3 MPa, f_t=2.6 MPa, G_f=0.1 N/mm
Fibres: E=210 GPa, density 3.02/cm²
Fibre bond: s0=0.01, a=20, τ1=5, τ2=3, τ_f=2.25 (mm, MPa)
```

**Status:** 🔴 Not implemented

---

### Cases 05-10

See full specifications in the original requirements. All parameter sets are documented.

---

## Implementation Roadmap

### Phase 1: Solver Integration (CRITICAL) ⚠️

**Goal:** Make Case 01 (pull-out) work end-to-end

**Tasks:**
1. Integrate bond-slip assembly into main solver loop
2. Extend DOF structure for steel slip DOFs
3. Implement subdomain support (void elements, bond masking)
4. Wire up postprocessing (slip profiles, bond profiles)
5. Validate against analytical solution

**Files to modify:**
- `src/xfem_clean/xfem/analysis_single.py` (main loop)
- `src/xfem_clean/xfem/model.py` (add subdomain config)
- `src/xfem_clean/bond_slip.py` (integrate custom laws)
- `examples/gutierrez_thesis/run.py` (wire up solver call)

**Acceptance criteria:**
- Case 01 runs without errors
- Load-slip curve matches expected trend
- Slip profile shows decay from load end

### Phase 2: Subdomain Infrastructure

**Goal:** Support element-level properties (void, rigid, material override)

**Tasks:**
1. Add `element_properties` dict to model
2. Modify assembly to check properties per element
3. Implement void elements (E=0 or h=0)
4. Implement rigid regions (E_override = 1e6)
5. Implement bond masking per segment

**New module:**
- `src/xfem_clean/xfem/subdomains.py`

### Phase 3: FRP Sheet Reinforcement

**Goal:** Support externally bonded reinforcement

**Tasks:**
1. Create 1D FRP element with own DOFs
2. Discretize sheet into segments
3. Couple to bulk via bond elements
4. Integrate BilinearBondLaw
5. Track debonding progression

**New module:**
- `src/xfem_clean/reinforcement/frp_sheet.py`

### Phase 4: Fibre Reinforcement

**Goal:** Explicit fibre generation and activation

**Tasks:**
1. Create fibre generator (deterministic seed)
2. Sample fibres in crack zone
3. Orientation distribution (von Mises, uniform)
4. Activate fibres near cracks only
5. Integrate BanholzerBondLaw

**New module:**
- `src/xfem_clean/reinforcement/fibres.py`

### Phase 5: Cyclic Loading

**Goal:** Support drift protocols (walls C1/C2)

**Tasks:**
1. Implement load program (list of targets)
2. Sign alternation (0 → +d → 0 → -d → 0)
3. Cycle repetition (n_cycles_per_drift)
4. Constant axial load superposition

**Files to modify:**
- `src/xfem_clean/xfem/analysis_single.py` (loading control)

### Phase 6: Remaining Cases

**Goal:** Implement Cases 02-10

**Tasks:**
1. Create case definitions (copy/adapt Case 01 template)
2. Add case-specific features (stirrups, transverse contact, etc.)
3. Validate each case against thesis results
4. Document acceptance criteria

### Phase 7: Testing

**Goal:** Smoke tests for all cases

**Tasks:**
1. Create `test_thesis_cases.py`
2. Each case runs with coarse mesh (<30s)
3. Check convergence (no crashes)
4. Check metrics (peak load within 20% of expected)

---

## Data Pack (Quick Reference)

### Pull-Out (Lettow)
```
Block: 200×200×200
Bar: Ø12, embedment 36
E_c=26287, ν_c=0.2
E_s=200000, ν_s=0.3
Bond: s1=0.77, s2=1.37, s3=7.5, τ_max=11.5, τ_f=4.0
```

### SSPOT (FRP)
```
Sheet: 0.05×40, bonded 150, h=40
E_c=26287, ν_c=0.2
E_sheet=100000, ν_sheet=0.3
Bond: s1=0.5, s2=1.0, τ1=1.0
```

### STN12 (Tensile Member)
```
Prism: 100×100×1100, bar Ø12
E_c=22.4 GPa, f_c=21.6, f_t=2.04, G_f=0.051
E_s=200 GPa, f_y=520
Bond: s1=1, s2=2, s3=5, τ_max=11.61, τ_f=4.64
```

### FRC (Sorelli)
```
Notched specimen, h=40
E_c=25.1 GPa, f_c=28.3, f_t=2.6, G_f=0.1
Fibres: E=210 GPa, density 3.02/cm²
Bond: s0=0.01, a=20, τ1=5, τ2=3, τ_f=2.25
```

### Beams T5A1/T6A1 (Bosco, 3PBT)
```
E_c=28.0 GPa, f_c=32, f_t=2.5, G_f=0.1
E_s=200 GPa, f_y=487
Bond (Ø12): s1=0.73, s2=0.73, s3=4.0, τ_max=15.51, τ_f=6.2
Bond (Ø10): s1=0.73, s2=0.73, s3=4.0, τ_max=14.07, τ_f=5.63
Bond (Ø6):  s1=1.0, s2=2.0, s3=2.4, τ_max=15.51, τ_f=6.2
```

### Beam 4PBT (Jason)
```
E_c=29.8 GPa, f_c=39, f_t=2.5, G_f=0.1
E_s=190 GPa, f_y=550, f_u=590
Bond (Ø12): s1=0.47, s2=0.47, s3=0.57, τ_max=11.58, τ_f=0.12
Bond (Ø8):  s1=0.87, s2=0.87, s3=1.05, τ_max=14.8, τ_f=0.15
Bond (Ø6):  s1=1.0, s2=2.0, s3=2.4, τ_max=15.61, τ_f=6.24
```

### Walls C1/C2 (Lu et al., Cyclic)
```
Axial: 290 kN (q=207.1 N/mm)
Drift: 0.2, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5 (%)

C1: E=26.0 GPa, f_c=38.5, f_t=2.88, G_f=0.077
    Bond (Ø10): s1=0.89, s2=0.89, s3=2.0, τ_max=14.79, τ_f=5.91
    Bond (Ø6):  s1=1.0, s2=2.0, s3=2.4, τ_max=15.51, τ_f=6.20

C2: E=27.5 GPa, f_c=34.5, f_t=2.53, G_f=0.071
    Bond (Ø10): s1=0.95, s2=0.95, s3=2.0, τ_max=14.38, τ_f=5.75
    Bond (Ø6):  s1=1.0, s2=2.0, s3=2.4, τ_max=14.68, τ_f=5.87
```

---

## Acceptance Criteria

Each case must satisfy:

1. **Convergence:** Runs to completion without hanging or diverging
2. **Load-displacement:** Curve shows expected shape (peak, softening)
3. **Crack pattern:** Number and orientation of cracks match thesis
4. **Energy balance:** Within 1-2% of total work
5. **Case-specific metrics:**
   - **Pull-out:** Slip profile matches analytical solution
   - **STN12:** ~5 cracks with reasonable spacing
   - **FRC:** Orientation sensitivity (90° ≈ plain concrete)
   - **Beams:** Crack pattern + slip/bond at failure
   - **Walls:** Moment-drift loops to ±1.5%

---

## Contributing

To add a new case:

1. **Create case definition:**
   ```python
   # examples/gutierrez_thesis/cases/case_XX_name.py
   def create_case_XX() -> CaseConfig:
       # Define geometry, materials, loading, outputs
       return CaseConfig(...)
   ```

2. **Register in runner:**
   ```python
   # examples/gutierrez_thesis/run.py
   from examples.gutierrez_thesis.cases.case_XX_name import create_case_XX

   CASE_REGISTRY = {
       ...
       "XX_name": create_case_XX,
   }
   ```

3. **Add smoke test:**
   ```python
   # examples/gutierrez_thesis/tests/test_thesis_cases.py
   def test_case_XX_runs():
       case = create_case_XX()
       # Modify to coarse mesh
       case.geometry.n_elem_x = 5
       case.geometry.n_elem_y = 5
       # Run solver
       result = run_case(case)
       assert result.converged
   ```

4. **Document:**
   - Add section to this README
   - Include expected outputs
   - List acceptance criteria

---

## Known Issues and Limitations

### 🔴 Critical Blockers

1. **Bond-slip not integrated:** The bond-slip code in `bond_slip.py` is fully implemented but not called from the main analysis loop. This requires:
   - Extending DOF structure to include steel slip DOFs
   - Modifying assembly to include bond stiffness and forces
   - Updating state management to track slip history

2. **No subdomain support:** Element-level material/thickness override not implemented. Required for:
   - Void elements (load element in pull-out)
   - Rigid loading beams (walls)
   - Bond masking (empty elements)

3. **No FRP/fibre reinforcement:** Externally bonded and fibre reinforcement types not implemented.

4. **No cyclic loading:** Drift protocol for walls C1/C2 not implemented.

### ⚠️ Limitations

- **T3 elements:** Triangular elements not implemented (Q4 only)
- **Stirrups:** Transverse reinforcement requires special handling (penalty contact)
- **Mesh generators:** Simple structured meshes only; complex geometries need external meshing

### 💡 Future Enhancements

- **Parallel execution:** Run all cases in batch mode
- **Parameter sweeps:** Automated sensitivity studies
- **Result comparison:** Overlay plots with thesis experimental data
- **Optimization:** Calibrate bond laws from pull-out tests

---

## References

1. **Gutiérrez Thesis (2004):** Chapter 5 + Appendix A
2. **Model Code 2010:** fib Model Code for Concrete Structures
3. **Lettow (2006):** Pull-out test data
4. **Wu et al. (STN12):** Tensile member test
5. **Sorelli:** FRC uniaxial and beam tests
6. **Bosco:** Beam tests T5A1/T6A1
7. **Jason:** 4-point bending beam
8. **Lu et al.:** Cyclic wall tests C1/C2

---

## Support

For questions or issues:
1. Check this README first
2. Review case configuration in `cases/`
3. Inspect bond laws in `src/xfem_clean/bond_slip.py`
4. Examine solver integration in `src/xfem_clean/xfem/analysis_single.py`

---

## License

Same as main repository.

---

**Last updated:** 2025-12-29

**Status:** Infrastructure complete, solver integration pending
