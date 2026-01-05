# XFEM for Concrete

Python implementation of Extended Finite Element Method (XFEM) for concrete structures with crack propagation, bond-slip, and material nonlinearity. Based on Gutierrez (2020) PhD thesis.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Thesis Case Suite](#thesis-case-suite)
- [Testing](#testing)
- [Convergence & Solver Parameters](#convergence--solver-parameters)
- [Tools](#tools)
- [Numba Performance](#numba-performance)
- [Repository Layout](#repository-layout)

---

## Installation

### Basic install (editable)

```bash
pip install -e .
```

### With Numba (recommended for performance)

```bash
pip install -e .
pip install numba
```

Numba provides ~28× speedup for bond-slip assembly and significant speedup for bulk constitutive kernels.

### Run without install

```bash
python examples/run_gutierrez_beam.py --umax-mm 10 --nsteps 30 --nx 120 --ny 20
```

---

## Quick Start

### List available cases

```bash
python -m examples.gutierrez_thesis.run --list
```

### Run a thesis case

```bash
# Run T5A1 beam (Bosco 3-point bending)
python -m examples.gutierrez_thesis.run --case t5a1 --mesh coarse

# Run VVBS3 CFRP strengthened beam
python -m examples.gutierrez_thesis.run --case vvbs3 --mesh medium

# Run Sorelli fibre-reinforced beam
python -m examples.gutierrez_thesis.run --case sorelli --mesh fine

# Run all cases (coarse mesh for speed)
python -m examples.gutierrez_thesis.run --case all --mesh coarse
```

### Enable Numba acceleration

```bash
# Auto-detect (default)
python -m examples.gutierrez_thesis.run --case t5a1

# Force enable
python -m examples.gutierrez_thesis.run --case t5a1 --use-numba

# Force disable
python -m examples.gutierrez_thesis.run --case t5a1 --no-numba
```

---

## Thesis Case Suite

## Recent Updates

- Bond-slip continuation (`bond_gamma`) now scales both interface forces and stiffness (including dowel contributions), keeping gamma ramps consistent.
- Multicrack linear solves now check linear residuals and fall back to LSMR/regularization for ill-conditioned steps, allowing substepping to recover instead of hard errors.

All cases from Gutierrez (KIT, 2020) PhD thesis are implemented with validation against experimental data.

### Case Overview

| Case ID | Thesis Section | Description | Features |
|---------|---------------|-------------|----------|
| **01** | §5.2 | Pullout test (Lettow) | Bond-slip, segment masking |
| **02** | §5.3 | FRP debonding (SSPOT) | FRP sheet, bilinear bond law |
| **03** | §5.4 | Direct tension (STN12) | Multicrack, distributed cracking |
| **04a** | §5.5.1 | 3PB beam T5A1 (Bosco) | Flexural cracking, validation |
| **04b** | §5.5.1 | 3PB beam T6A1 (Bosco) | Flexural cracking |
| **05** | §5.6.1 | Cyclic wall C1 | Cyclic loading, multicrack |
| **06** | §6.2 | Fibre tensile test | Fibre bridging, Banholzer law |
| **07** | §5.5.2 | 4PB beam (Jason 4PBT) | CFRP strengthening |
| **08** | §5.5.3 | CFRP beam (VVBS3) | IC-debonding, validation |
| **09** | §6.3 | Fibre beam (Sorelli) | Post-peak ductility, validation |
| **10** | §5.6.2 | Cyclic wall C2 | Cyclic loading |

### Key Parameters per Case

#### Case 04a: T5A1 Beam (Most Validated)

**Thesis Reference:**
- Section: §5.5.1 - Three-point bending tests
- Figure: Fig 5.20 - P–δ curve
- Table: Table 5.10 - Material properties

**Geometry:**
- Dimensions: 1500×250×120 mm, span 1400 mm
- Rebar: 2Ø16 mm (bottom), cover 30 mm

**Material:**
- Concrete: E = 31 GPa, fc = 35 MPa, ft = 3.2 MPa, Gf = 0.10 N/mm
- Bond law: τmax = 15.0 MPa, s1 = 1.0 mm, s2 = 2.0 mm

**Validation:**
- Reference data: `validation/reference_data/t5a1.csv`
- Test: `pytest tests/test_validation_curves.py::test_validate_t5a1_coarse`
- Tolerance: |ΔPmax| < 10%, |ΔE| < 15%

**Run:**
```bash
python -m examples.gutierrez_thesis.run --case t5a1 --mesh coarse
```

#### Case 08: VVBS3 CFRP Beam

**Thesis Reference:**
- Section: §5.5.3 - FRP strengthening
- Figure: Fig 5.30 - P–δ curve with CFRP debonding
- Tables: 5.17 (concrete), 5.18 (steel), 5.19 (FRP/bond)

**Geometry:**
- Dimensions: 4300×450×200 mm, span 3700 mm
- Rebar: 3Ø20 (bottom), 2Ø16 (top)
- CFRP: 100 mm width, 1.4 mm thickness, E = 170.8 GPa

**Material:**
- Concrete: E = 24 GPa, fc = 28.2 MPa, ft = 2.9 MPa, Gf = 0.052 N/mm
- FRP bond law (bilinear): τmax = 6.47 MPa, s1 = 0.02 mm, s2 = 0.25 mm

**Validation:**
- Reference data: `validation/reference_data/vvbs3.csv`
- Key phenomenon: IC-debonding (intermediate crack-induced debonding)

**Run:**
```bash
python -m examples.gutierrez_thesis.run --case vvbs3 --mesh medium
```

#### Case 09: Sorelli Fibre Beam

**Thesis Reference:**
- Section: §6.3 - Fibre-reinforced concrete
- Figure: Fig 6.10 - P–δ curve showing fibre bridging
- Table: Table 6.5 - Fibre properties

**Material:**
- Concrete: fc = 40 MPa, ft = 3.5 MPa
- Fibres: ρf = 1.0%, Lf/df = 60 (steel fibres)
- Fibre bond law: τmax = 8.0 MPa, slip-dependent bridging (Banholzer)

**Validation:**
- Reference data: `validation/reference_data/sorelli.csv`
- Key phenomenon: Post-peak ductility from fibre bridging

**Run:**
```bash
python -m examples.gutierrez_thesis.run --case sorelli --mesh fine
```

---

## Testing

### Fast tests (< 2 minutes)

Run smoke tests and fast regression tests:

```bash
# All non-slow tests
pytest -m "not slow" -q

# Specific smoke tests
pytest tests/test_smoke.py -v
pytest tests/test_tools_smoke.py -v

# Bond-slip convergence tests
pytest tests/test_bond_jacobian_coupling.py -v
```

### Slow tests (validation, full cases)

Full validation suite with experimental curve comparison:

```bash
# All validation tests (requires simulation outputs)
pytest tests/test_validation_curves.py -v -m slow

# Specific case validation
pytest tests/test_validation_curves.py::test_validate_t5a1_coarse -v -s
pytest tests/test_validation_curves.py::test_validate_vvbs3_coarse -v -s
pytest tests/test_validation_curves.py::test_validate_sorelli_coarse -v -s

# Full regression suite
pytest tests/test_regression_cases.py -v
```

### Validation metrics

Tests verify against experimental data using:

- **Peak load error:** |ΔPmax| < 10%
- **Energy error:** |ΔE| < 15%
- **RMSE:** < 5% of peak load
- **R²:** > 0.95 (goodness of fit)

### Test structure

- `tests/test_smoke.py` - Import smoke tests
- `tests/test_tools_smoke.py` - Tool execution tests (parametric, calibration, benchmarks)
- `tests/test_validation_curves.py` - Experimental curve comparison (slow)
- `tests/test_regression_cases.py` - Regression tests with reference ranges
- `tests/test_thesis_smoke.py` - Thesis case smoke tests
- `tests/regression/reference_cases.yml` - Expected ranges for regression

---

## Convergence & Solver Parameters

### Bond-slip continuation (critical for convergence)

Bond-slip cases can fail with "Substepping exceeded" due to stiff tangent at s≈0. The solution is **bond_gamma continuation** (gradual activation).

**Configuration:**

```python
from xfem_clean.xfem.model import XFEMModel

model = XFEMModel(
    # ... other params ...
    enable_bond_slip=True,

    # Bond-slip continuation (recommended)
    bond_gamma_strategy="ramp_steps",  # Enable continuation
    bond_gamma_ramp_steps=5,           # [0, 0.25, 0.5, 0.75, 1.0]
    bond_gamma_min=0.0,                # Start with no bond (only steel EA)
    bond_gamma_max=1.0,                # Ramp up to full bond
)
```

**How it works:**

1. For each displacement step `u_target`
2. Solve Newton for gamma sequence: [0.0, 0.25, 0.5, 0.75, 1.0]
3. Use previous gamma solution as initial guess
4. Only commit states when gamma=1 converges

**Mathematical formulation:**

```
K_bond(gamma) = gamma * (dtau/ds) * perimeter * L0

gamma=0: No bond (only steel EA) → easy to converge
gamma=1: Full bond-slip coupling → physically correct
```

### Optional tangent regularization

For extremely difficult cases, enable tangent capping and smoothing:

```python
model = XFEMModel(
    # ... continuation params above ...

    # Optional regularization (usually not needed with continuation)
    bond_k_cap=1e12,    # Cap dtau/ds [Pa/m] to prevent excessive stiffness
    bond_s_eps=1e-6,    # Smooth slip near s=0 [m] using sqrt(s² + ε²)
)
```

### Void penalty (multicrack)

Void elements (fully damaged regions) get penalty stiffness to prevent singular matrices:

```python
# In assembly (automatic)
if is_void_elem:
    C_eff = C * 1e-9  # Penalty factor
```

This prevents hanging DOFs in multicrack simulations.

### Convergence troubleshooting

**If case still fails with "Substepping exceeded":**

1. Increase ramp steps: `bond_gamma_ramp_steps = 7` or `10`
2. Enable regularization: `bond_k_cap = 1e12` or `bond_s_eps = 1e-6`
3. Enable debug output: `debug_substeps=True`, `debug_bond_gamma=True`
4. Reduce initial step size in solver

**If Newton converges but results are wrong:**

1. Verify `bond_gamma_max == 1.0` (must end at full physics)
2. Check states are committed only after gamma=1 converges
3. Verify bond law `tau_and_tangent()` implementation

**Related files:**

- Implementation: `src/xfem_clean/xfem/analysis_single.py` (lines 722-809)
- Tests: `tests/test_bond_jacobian_coupling.py`, `tests/test_convergence_case01_min.py`

---

## Tools

### 1. Parametric Studies (Sensitivity Analysis)

Sweep material/geometric parameters to study influence on response:

```bash
# Sweep fracture energy Gf for T5A1 beam
python -m examples.parametric.parametric_study \
  --case t5a1 --param Gf --values 0.05,0.1,0.2 --mesh coarse

# Sweep bond strength τmax for VVBS3 beam
python -m examples.parametric.parametric_study \
  --case vvbs3 --param tau_max --values 4.0,6.47,9.0 --plot

# Available parameters: Gf, f_t, f_c, tau_max, s1, s2, n_bars, rho_fibre, etc.
```

**Output:** `outputs/parametric/<case>_<param>.csv` (Pmax, energy, ductility)

**Module:** `examples/parametric/parametric_study.py`

### 2. Bond Parameter Calibration

Fit bond law parameters (τmax, s1, s2) to experimental P–δ curves:

```bash
# Calibrate VVBS3 FRP bond parameters
python -m calibration.fit_bond_parameters \
  --case vvbs3 \
  --params tau_max,s1,s2 \
  --init tau_max=6.47,s1=0.02,s2=0.25 \
  --bounds 5-8,0.01-0.05,0.2-0.3 \
  --method L-BFGS-B

# Use global optimizer for robust fitting
python -m calibration.fit_bond_parameters \
  --case vvbs3 \
  --params tau_max,s1 \
  --method differential_evolution \
  --max-iter 100
```

**Output:** `calibration/results_<case>.json` (optimal parameters, convergence)

**Module:** `calibration/fit_bond_parameters.py`

### 3. Performance Benchmarks

Measure runtime, memory, and scaling across mesh sizes:

```bash
# Benchmark single case with multiple meshes
python -m benchmarks.benchmark_scaling \
  --case t5a1 --meshes coarse,medium,fine --plot

# Benchmark all cases (fast meshes only)
python -m benchmarks.benchmark_scaling \
  --case all --meshes coarse,medium

# Bond-slip microbenchmark (Numba vs Python)
python -m benchmarks.benchmark_bond_slip \
  --nseg 100,1000,5000,10000 --repeat 3
```

**Output:** `benchmarks/scaling_summary.csv` (runtime, memory, energy residual)

**Module:** `benchmarks/benchmark_scaling.py`, `benchmarks/benchmark_bond_slip.py`

### 4. Validation Curve Comparison

Compare simulation results with experimental reference curves:

```bash
# Run validation tests (requires simulation outputs)
pytest tests/test_validation_curves.py -v -m slow

# Example: Validate T5A1 beam
python -m examples.gutierrez_thesis.run --case t5a1 --mesh coarse
pytest tests/test_validation_curves.py::test_validate_t5a1_coarse -v -s
```

**Module:** `validation/compare_curves.py`

### 5. Report Generation

Generate appendix with figures and tables (future work):

```bash
python -m report.generate_appendix
```

**Module:** `report/generate_appendix.py`

---

## Numba Performance

### Overview

Numba JIT compilation provides significant speedups for computationally intensive kernels while maintaining full numerical correctness in the Python fallback.

### Benchmark Results

**Bond-slip assembly speedup** (measured on typical hardware):

| Segments | Python (s) | Numba (s) | Speedup |
|----------|------------|-----------|---------|
| 100      | 0.004      | 0.0003    | 15×     |
| 1,000    | 0.036      | 0.0015    | 24×     |
| 5,000    | 0.193      | 0.0060    | 32×     |
| 10,000   | 0.410      | 0.0125    | 33×     |

**Average speedup (n_seg >= 1000): ~28×**

Bulk constitutive kernels (elastic, Drucker-Prager) also show 10-50× speedups depending on problem size and material model.

### Implementation Status

**Phase 1: State arrays** (DONE)
- `BulkStateArrays`, `CohesiveStateArrays` replace Python dict history
- Sparse patches (`BulkStatePatch`, `CohesiveStatePatch`) for Newton updates
- No dictionary copies inside Newton loops

**Phase 2: Stateless kernels** (DONE)
- Cohesive laws: `src/xfem_clean/numba/kernels_cohesive.py`
- Bulk constitutive: `src/xfem_clean/numba/kernels_bulk.py`
  - Linear elastic plane-stress
  - Drucker-Prager (associated) + isotropic hardening
- Bond-slip: `src/xfem_clean/numba/kernels_bond_slip.py`
- Q4 shape functions: `src/xfem_clean/numba/kernels_q4.py`

**Phase 3: Full assembly** (PLANNED)
- Numba-compiled assembly loops for K and internal forces
- Preallocated triplet arrays for sparse matrix construction

### Usage

**Auto-detect Numba** (default):

```bash
python -m examples.gutierrez_thesis.run --case pullout
# Uses Numba if available, Python fallback otherwise
```

**Force Numba:**

```bash
python -m examples.gutierrez_thesis.run --case pullout --use-numba
```

**Force Python fallback:**

```bash
python -m examples.gutierrez_thesis.run --case pullout --no-numba
```

**In code:**

```python
from xfem_clean.xfem.model import XFEMModel

# Auto-detect (recommended)
model = XFEMModel(..., use_numba=None)

# Force enable
model = XFEMModel(..., use_numba=True)

# Force disable
model = XFEMModel(..., use_numba=False)
```

### Installation

Install Numba for best performance:

```bash
pip install numba
```

The code works correctly without Numba (pure Python fallback), but runs significantly slower for bond-slip assembly and bulk constitutive updates.

### Notes

- **First run:** Slow (10-30 seconds) due to JIT compilation
- **Subsequent runs:** Fast (compiled code is cached)
- **Kernels:** Located in `src/xfem_clean/numba/` with `cache=True`
- **Fallback:** Python path uses full consistent tangent (8×8 for bond-slip)

---

## Repository Layout

```
xfem-concrete/
├── src/                              # Source code
│   ├── xfem_clean/                   # XFEM solver
│   │   ├── xfem/                     # Core XFEM modules
│   │   │   ├── analysis_single.py   # Single-crack solver
│   │   │   ├── assembly_single.py   # Single-crack assembly
│   │   │   ├── multicrack.py        # Multicrack solver
│   │   │   ├── model.py             # XFEMModel config dataclass
│   │   │   ├── state_arrays.py      # State containers (Phase 1)
│   │   │   ├── enrichment_single.py # Enrichment functions
│   │   │   ├── post.py              # Post-processing
│   │   │   └── ...
│   │   ├── numba/                    # Numba kernels (Phase 2)
│   │   │   ├── kernels_bulk.py      # Constitutive (elastic, DP)
│   │   │   ├── kernels_cohesive.py  # Cohesive laws
│   │   │   ├── kernels_bond_slip.py # Bond-slip assembly
│   │   │   └── kernels_q4.py        # Q4 shape functions
│   │   ├── fem/                      # FEM basics
│   │   │   ├── mesh.py              # Mesh generation
│   │   │   ├── bcs.py               # Boundary conditions
│   │   │   └── q4.py                # Q4 element
│   │   ├── cohesive_laws.py         # Cohesive law implementations
│   │   ├── bond_slip.py             # Bond-slip assembly
│   │   ├── convergence.py           # Convergence utilities
│   │   ├── crack_criteria.py        # Crack initiation criteria
│   │   ├── constitutive.py          # Bulk constitutive models
│   │   ├── compression_damage.py    # CDP implementation
│   │   └── ...
│   └── cdp_generator/                # CDP calibration tool
│       ├── cli.py                    # CLI for CDP generation
│       ├── core.py                   # Core CDP calibration
│       └── ...
├── examples/                         # Example scripts
│   ├── gutierrez_thesis/             # Thesis case suite
│   │   ├── run.py                    # CLI to run cases
│   │   ├── case_config.py            # CaseConfig dataclass
│   │   ├── solver_interface.py       # Solver dispatcher
│   │   ├── postprocess_comprehensive.py  # Post-processing
│   │   ├── history_utils.py          # History extraction
│   │   └── cases/                    # Individual case files
│   │       ├── case_01_pullout_lettow.py
│   │       ├── case_04a_beam_3pb_t5a1_bosco.py
│   │       ├── case_08_beam_3pb_vvbs3_cfrp.py
│   │       └── ...
│   ├── parametric/                   # Parametric studies
│   │   └── parametric_study.py
│   └── sensitivity/                  # Sensitivity studies
│       └── sensitivity_study_jason.py
├── tests/                            # Test suite
│   ├── test_smoke.py                 # Import smoke tests
│   ├── test_tools_smoke.py           # Tool execution tests
│   ├── test_validation_curves.py     # Experimental validation (slow)
│   ├── test_regression_cases.py      # Regression tests
│   ├── test_thesis_smoke.py          # Thesis case smoke tests
│   ├── test_bond_jacobian_coupling.py  # Bond convergence tests
│   └── regression/                   # Regression references
│       └── reference_cases.yml
├── validation/                       # Validation data
│   ├── compare_curves.py             # Curve comparison utilities
│   └── reference_data/               # Experimental curves
│       ├── t5a1.csv                  # Bosco T5A1 beam
│       ├── vvbs3.csv                 # VVBS3 CFRP beam
│       ├── sorelli.csv               # Sorelli fibre beam
│       └── SOURCES.md                # Data sources
├── calibration/                      # Bond parameter calibration
│   └── fit_bond_parameters.py
├── benchmarks/                       # Performance benchmarks
│   ├── benchmark_scaling.py          # Scaling benchmarks
│   └── benchmark_bond_slip.py        # Bond-slip microbenchmark
├── report/                           # Report generation
│   └── generate_appendix.py
├── pyproject.toml                    # Package metadata
└── README.md                         # This file
```

### Key modules

**Solver core:**
- `src/xfem_clean/cohesive_laws.py` - Cohesive law implementations
- `src/xfem_clean/convergence.py` - Convergence utilities
- `src/xfem_clean/crack_criteria.py` - Crack initiation criteria
- `src/xfem_clean/xfem_xfem.py` - Legacy stable solver

**State management:**
- `src/xfem_clean/xfem/state_arrays.py` - BulkStateArrays, CohesiveStateArrays

**Numba kernels:**
- `src/xfem_clean/numba/kernels_bulk.py` - Constitutive kernels
- `src/xfem_clean/numba/kernels_cohesive.py` - Cohesive kernels
- `src/xfem_clean/numba/kernels_bond_slip.py` - Bond-slip kernel

---

## Additional Information

### Validation Data Sources

Experimental reference curves in `validation/reference_data/` were digitized from thesis figures using WebPlotDigitizer. See `validation/reference_data/SOURCES.md` for details.

### Post-processing Outputs

Simulations automatically export:

- **Load-displacement:** `load_displacement.csv`, `load_displacement.png`
- **Crack width profiles:** `crack_width_profile_crack{k}_final.csv`, `.png`
- **Steel force profiles:** `steel_force_profile_final.csv`, `.png`
- **Bond stress profiles:** `bond_stress_profile_final.csv`, `.png`
- **VTK files:** `vtk/step_XXXX.vtk` (for ParaView)

Module: `examples/gutierrez_thesis/postprocess_comprehensive.py`

### Material Models

Supported bulk material models:

- **Linear elastic:** Plane stress
- **Drucker-Prager:** Associated plasticity + isotropic hardening
- **CDP (Concrete Damaged Plasticity):** Simplified (Numba) and table-based (Python)

Supported cohesive laws:

- **Bilinear:** Softening law (Reinhardt)
- **Exponential:** CEB-FIP Model Code 2010
- **Bilinear bond:** For FRP debonding
- **Banholzer:** 5-parameter fibre pullout law

### Citation

If you use this code, please cite:

> Gutierrez, M. A. (2020). Extended Finite Element Method for Reinforced Concrete Structures with Bond-Slip. PhD Thesis, Karlsruhe Institute of Technology (KIT).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Third-party Licenses

This repository includes third-party code with separate licenses:

- **CDP Generator**: MIT License (see [LICENSES/CDP_GENERATOR_LICENSE](LICENSES/CDP_GENERATOR_LICENSE))

---

For archived documentation, see `docs/archive/`.
