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

Numba accelerates selected bond, cohesive, and bulk constitutive kernels. The
element/Jacobian loops, sparse assembly, and linear solves remain Python/SciPy,
so end-to-end speedup is case-dependent and the CLI reports support as partial.

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

### Validate thesis definitions

```bash
# Validate all 13 aliases/factories without entering a solver
python -m examples.gutierrez_thesis.run --case all --mesh coarse --dry-run

# Supported elastic smoke families
python -m examples.gutierrez_thesis.run --case pullout --dry-run
python -m examples.gutierrez_thesis.run --case sspot --dry-run

# T5A1 is characterized but blocked before solve; this prints the reason
python -m examples.gutierrez_thesis.run --case t5a1 --mesh coarse --dry-run
```

### Enable Numba acceleration

```bash
# Show effective partial support without solving
python -m examples.gutierrez_thesis.run --case stn12 --use-numba --dry-run
```

See `docs/examples.md` for the compatibility matrix. `cdp_full` multicrack
cases currently fail closed because neither backend preserves that material
contract; they no longer start a long, misleading solve.

---

## Thesis Case Suite

## Recent Updates

- Bond-slip continuation (`bond_gamma`) now scales both interface forces and stiffness (including dowel contributions), keeping gamma ramps consistent.
- Multicrack linear solves now check linear residuals and fall back to LSMR/regularization for ill-conditioned steps, allowing substepping to recover instead of hard errors.

All registered cases from the Gutiérrez (KIT, 2020) thesis are implemented as maintained Python case factories. The bundled reference CSV files are synthetic placeholders for pipeline regression, not experimental validation evidence; see `docs/validation.md`.

### Case Overview

| Case ID | Thesis Section | Description | Features |
|---------|---------------|-------------|----------|
| **01** | §5.2 | Pullout test (Lettow) | Bond-slip, segment masking |
| **02** | §5.3 | FRP debonding (SSPOT) | FRP sheet, bilinear bond law |
| **03** | §5.4 | Direct tension (STN12) | Multicrack, distributed cracking |
| **04a** | §5.5.1 | 3PB beam T5A1 (Bosco) | Ambiguous definition; unsupported cdp_full/multi |
| **04b** | §5.5.1 | 3PB beam T6A1 (Bosco) | Experimental; unsupported cdp_full/multi |
| **05** | §5.6.1 | Cyclic wall C1 | Experimental; unsupported cdp_full/multi |
| **06** | §6.2 | Fibre tensile test | Fibre bridging, Banholzer law |
| **07** | §5.5.2 | 4PB beam (Jason 4PBT) | Experimental; unsupported cdp_full/multi |
| **08** | §5.5.3 | CFRP beam (VVBS3) | FRP defined; unsupported cdp_full/multi |
| **09** | §6.3 | Fibre beam (Sorelli) | Synthetic reference; unsupported cdp_full/multi |
| **10** | §5.6.2 | Cyclic wall C2 | Experimental; unsupported cdp_full/multi |

### Key Parameters per Case

#### Case 04a: T5A1 Beam (characterized, blocked)

**Thesis Reference:**
- Section: §5.5.1 - Three-point bending tests
- Figure: Fig 5.20 - P–δ curve
- Table: Table 5.10 - Material properties

The factory/YAML describe 4000×400×200 mm with 4Ø12 bottom and 2Ø10 top;
older README/reference placeholders describe 1500×250×120 mm with 2Ø16. The
CSV is explicitly synthetic. No physical values were changed without a primary
source. In addition, multicrack does not faithfully consume `cdp_full`.

**Run:**
```bash
python -m examples.gutierrez_thesis.run --case t5a1 --mesh coarse --dry-run
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
python -m examples.gutierrez_thesis.run --case vvbs3 --mesh medium --dry-run
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
python -m examples.gutierrez_thesis.run --case sorelli --mesh fine --dry-run
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

# Compact canonical regression manifest
PYTHONPATH=src python3 scripts/regression_manifest.py
```

### Validation metrics

Placeholder-curve pipeline tests currently calculate:

- **Peak load error:** |ΔPmax| < 10%
- **Energy error:** |ΔE| < 15%
- **RMSE:** < 5% of peak load
- **R²:** > 0.95 (goodness of fit)

### Test structure

- `tests/test_smoke.py` - Import smoke tests
- `tests/test_tools_smoke.py` - Tool execution tests (parametric, calibration, benchmarks)
- `tests/test_validation_curves.py` - Placeholder-curve pipeline checks (slow)
- `tests/test_regression_cases.py` - Canonical manifest comparison
- `tests/test_thesis_smoke.py` - Thesis case smoke tests
- `tests/regression/canonical_manifest.json` - Versioned numerical golden
- `tests/regression/reference_cases.yml` - Inactive legacy placeholder ranges

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
# Sweep bond strength on the supported elastic pullout path
python -m examples.parametric.parametric_study \
  --case pullout --param tau_max --values 8.0,10.0,12.0 --mesh coarse

# Available parameters: Gf, f_t, f_c, tau_max, s1, s2, n_bars, rho_fibre, etc.
```

**Output:** `outputs/parametric/<case>_<param>.csv` (Pmax, energy, ductility)

**Module:** `examples/parametric/parametric_study.py`

### 2. Bond Parameter Calibration

The historical calibration commands for VVBS3 are blocked until its
`cdp_full` multicrack contract is supported. The module is retained for future
use with provenance-bearing experimental data.

```bash
# Inspect the CLI only; do not calibrate against placeholder CSVs
python -m calibration.fit_bond_parameters --help
```

**Output:** `calibration/results_<case>.json` (optimal parameters, convergence)

**Module:** `calibration/fit_bond_parameters.py`

### 3. Performance Benchmarks

Measure runtime, memory, and scaling across mesh sizes:

```bash
# Benchmark a supported case with multiple meshes
python -m benchmarks.benchmark_scaling \
  --case pullout --meshes coarse,medium --plot

# Bond-slip microbenchmark (Numba vs Python)
python -m benchmarks.benchmark_bond_slip \
  --nseg 100,1000,5000,10000 --repeat 3
```

**Output:** `benchmarks/scaling_summary.csv` (runtime, memory, energy residual)

**Module:** `benchmarks/benchmark_scaling.py`, `benchmarks/benchmark_bond_slip.py`

### 4. Validation Curve Comparison

The bundled curves are synthetic placeholders. These tests validate the
comparison pipeline only and skip/xfail physical claims.

```bash
# Run validation tests (requires simulation outputs)
pytest tests/test_validation_curves.py -v -m slow

# Characterize T5A1 without solving an incompatible path
python -m examples.gutierrez_thesis.run --case t5a1 --mesh coarse --dry-run
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
│   ├── test_validation_curves.py     # Placeholder pipeline checks (slow)
│   ├── test_regression_cases.py      # Regression tests
│   ├── test_thesis_smoke.py          # Thesis case smoke tests
│   ├── test_bond_jacobian_coupling.py  # Bond convergence tests
│   └── regression/                   # Regression references
│       ├── canonical_manifest.json   # Numerical regression golden
│       └── reference_cases.yml       # Inactive legacy ranges
├── validation/                       # Validation pipeline and placeholder data
│   ├── compare_curves.py             # Curve comparison utilities
│   └── reference_data/               # Synthetic placeholder curves
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
