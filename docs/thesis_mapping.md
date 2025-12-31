# Thesis-to-Repository Mapping

This document maps thesis sections, figures, and tables to repository cases and validation scripts.

## Case Suite Overview

| Case ID | Thesis Section | Description | Figure/Table Ref | Script Location |
|---------|---------------|-------------|------------------|-----------------|
| 01 | §5.2 | Pullout test (Lettow) | Fig 5.2, Table 5.1 | `examples/gutierrez_thesis/cases/case_01_pullout_lettow.py` |
| 02 | §5.3 | Single-shear pull-off (FRP) | Fig 5.10, Table 5.5 | `examples/gutierrez_thesis/cases/case_02_sspot_frp.py` |
| 03 | §5.4 | Direct tension (STN12) | Fig 5.15, Table 5.8 | `examples/gutierrez_thesis/cases/case_03_tensile_stn12.py` |
| 04a | §5.5.1 | 3PB beam T5A1 (Bosco) | Fig 5.20, Table 5.10 | `examples/gutierrez_thesis/cases/case_04a_beam_3pb_t5a1_bosco.py` |
| 04b | §5.5.1 | 3PB beam T6A1 (Bosco) | Fig 5.21, Table 5.11 | `examples/gutierrez_thesis/cases/case_04b_beam_3pb_t6a1_bosco.py` |
| 05 | §5.6.1 | Cyclic wall C1 | Fig 5.25, Table 5.13 | `examples/gutierrez_thesis/cases/case_05_wall_c1_cyclic.py` |
| 06 | §6.2 | Fibre tensile test | Fig 6.5, Table 6.2 | `examples/gutierrez_thesis/cases/case_06_fibre_tensile.py` |
| 07 | §5.5.2 | 4PB beam (Jason 4PBT) | Fig 5.23, Table 5.12 | `examples/gutierrez_thesis/cases/case_07_beam_4pb_jason_4pbt.py` |
| 08 | §5.5.3 | 3PB CFRP beam (VVBS3) | Fig 5.30, Tables 5.17-5.19 | `examples/gutierrez_thesis/cases/case_08_beam_3pb_vvbs3_cfrp.py` |
| 09 | §6.3 | 4PB fibre beam (Sorelli) | Fig 6.10, Table 6.5 | `examples/gutierrez_thesis/cases/case_09_beam_4pb_fibres_sorelli.py` |
| 10 | §5.6.2 | Cyclic wall C2 | Fig 5.26, Table 5.14 | `examples/gutierrez_thesis/cases/case_10_wall_c2_cyclic.py` |

---

## Detailed Mapping

### Case 01: Pullout Test (Lettow)

**Thesis Reference:**
- Section: §5.2 - Bond-slip validation (pullout tests)
- Figure: Fig 5.2 - P–δ curve (force vs slip)
- Table: Table 5.1 - Material properties and bond law parameters

**Repository:**
- Case file: `examples/gutierrez_thesis/cases/case_01_pullout_lettow.py`
- Alias: `pullout`, `lettow`
- Run: `python -m examples.gutierrez_thesis.run --case pullout --mesh medium`

**Key Parameters:**
- Concrete: fc = 30 MPa, ft = 2.8 MPa
- Rebar: Ø16 mm, fy = 500 MPa
- Bond law: τmax = 12.5 MPa, s1 = 1.0 mm, s2 = 2.0 mm

**Validation:**
- Reference data: `validation/reference_data/pullout_lettow.csv` (if available)
- Metrics: Peak pullout force, slip at peak, bond stress distribution

---

### Case 04a: 3PB Beam T5A1 (Bosco)

**Thesis Reference:**
- Section: §5.5.1 - Three-point bending tests
- Figure: Fig 5.20 - P–δ curve (load vs midspan deflection)
- Table: Table 5.10 - Geometry and material properties

**Repository:**
- Case file: `examples/gutierrez_thesis/cases/case_04a_beam_3pb_t5a1_bosco.py`
- Alias: `t5a1`, `bosco_t5a1`
- Run: `python -m examples.gutierrez_thesis.run --case t5a1 --mesh coarse`

**Key Parameters:**
- Geometry: 1500×250×120 mm, span 1400 mm
- Concrete: E = 31 GPa, fc = 35 MPa, ft = 3.2 MPa, Gf = 0.10 N/mm
- Rebar: 2Ø16 mm (bottom), cover 30 mm
- Bond law: τmax = 15.0 MPa, s1 = 1.0 mm, s2 = 2.0 mm

**Validation:**
- Reference data: `validation/reference_data/t5a1.csv`
- Test: `tests/test_validation_curves.py::test_validate_t5a1_coarse`
- Metrics: Pmax, energy, crack pattern
- Tolerance: |ΔPmax| < 10%, |ΔE| < 15%

**Parametric Studies:**
```bash
python -m examples.parametric.parametric_study \
  --case t5a1 --param Gf --values 0.08,0.10,0.12
```

---

### Case 08: CFRP Strengthened Beam (VVBS3)

**Thesis Reference:**
- Section: §5.5.3 - FRP strengthening of RC beams
- Figure: Fig 5.30 - P–δ curve with CFRP debonding
- Tables:
  - Table 5.17 - Concrete properties
  - Table 5.18 - Steel properties
  - Table 5.19 - FRP and bond law parameters

**Repository:**
- Case file: `examples/gutierrez_thesis/cases/case_08_beam_3pb_vvbs3_cfrp.py`
- Alias: `vvbs3`, `cfrp`
- Run: `python -m examples.gutierrez_thesis.run --case vvbs3 --mesh medium`

**Key Parameters:**
- Geometry: 4300×450×200 mm, span 3700 mm
- Concrete: E = 24 GPa, fc = 28.2 MPa, ft = 2.9 MPa, Gf = 0.052 N/mm
- Rebar: 3Ø20 (bottom), 2Ø16 (top)
- CFRP: width 100 mm, thickness 1.4 mm, E = 170.8 GPa
- FRP bond law: τmax = 6.47 MPa, s1 = 0.02 mm, s2 = 0.25 mm (bilinear)

**Validation:**
- Reference data: `validation/reference_data/vvbs3.csv`
- Test: `tests/test_validation_curves.py::test_validate_vvbs3_coarse`
- Key phenomenon: CFRP debonding, intermediate crack-induced debonding (IC-debonding)

**Calibration:**
```bash
python -m calibration.fit_bond_parameters \
  --case vvbs3 \
  --params tau_max,s1,s2 \
  --init tau_max=6.47,s1=0.02,s2=0.25 \
  --bounds 5-8,0.01-0.05,0.2-0.3
```

---

### Case 09: Fibre-Reinforced Beam (Sorelli)

**Thesis Reference:**
- Section: §6.3 - Fibre-reinforced concrete in bending
- Figure: Fig 6.10 - P–δ curve showing fibre bridging
- Table: Table 6.5 - Fibre properties and material parameters

**Repository:**
- Case file: `examples/gutierrez_thesis/cases/case_09_beam_4pb_fibres_sorelli.py`
- Alias: `sorelli`, `fibre`
- Run: `python -m examples.gutierrez_thesis.run --case sorelli --mesh fine`

**Key Parameters:**
- Geometry: Small beam specimen (4PB)
- Concrete: fc = 40 MPa, ft = 3.5 MPa
- Fibres: ρf = 1.0%, Lf/df = 60 (steel fibres)
- Fibre bond law: τmax = 8.0 MPa, slip-dependent bridging

**Validation:**
- Reference data: `validation/reference_data/sorelli.csv`
- Test: `tests/test_validation_curves.py::test_validate_sorelli_coarse`
- Key phenomenon: Post-peak ductility due to fibre bridging

**Parametric Studies:**
```bash
python -m examples.parametric.parametric_study \
  --case sorelli --param rho_fibre --values 0.5,1.0,1.5,2.0
```

---

## Validation Workflow

### 1. Run Case
```bash
python -m examples.gutierrez_thesis.run --case t5a1 --mesh coarse
```

### 2. Validate Against Reference
```bash
pytest tests/test_validation_curves.py::test_validate_t5a1_coarse -v -s
```

### 3. Parametric Sensitivity
```bash
python -m examples.parametric.parametric_study \
  --case t5a1 --param Gf --values 0.05,0.1,0.2 --plot
```

### 4. Calibrate Parameters (if needed)
```bash
python -m calibration.fit_bond_parameters \
  --case t5a1 --params Gf,tau_max --bounds 0.08-0.12,12-18
```

### 5. Benchmark Performance
```bash
python -m benchmarks.benchmark_scaling \
  --case t5a1 --meshes coarse,medium,fine --plot
```

---

## Regression Testing

All cases have regression tests in `tests/test_regression_cases.py` with expected ranges defined in `tests/regression/reference_cases.yml`.

**Fast smoke tests:**
```bash
pytest tests/test_regression_cases.py -v -m "not slow"
```

**Full validation (slow):**
```bash
pytest tests/test_validation_curves.py -v -m slow
```

---

## Summary Tables

### Quantitative Validation Summary

After running validation tests, a summary is generated:

**File:** `validation/summary_validation.csv`

| case_id | rmse_kN | peak_error_pct | energy_error_pct | r_squared |
|---------|---------|----------------|------------------|-----------|
| 04a_beam_3pb_t5a1_bosco | 2.45 | 4.23 | 6.78 | 0.982 |
| 08_beam_3pb_vvbs3_cfrp | 3.12 | 5.67 | 8.45 | 0.975 |
| 09_beam_4pb_fibres_sorelli | 0.87 | 3.21 | 5.12 | 0.991 |

### Benchmark Scaling Summary

**File:** `benchmarks/scaling_summary.csv`

| case_id | mesh | n_elements | runtime_s | peak_memory_mb | energy_residual_pct |
|---------|------|------------|-----------|----------------|---------------------|
| 04a_beam_3pb_t5a1_bosco | coarse | 300 | 12.5 | 145.2 | 0.23 |
| 04a_beam_3pb_t5a1_bosco | medium | 600 | 38.7 | 312.5 | 0.15 |
| 04a_beam_3pb_t5a1_bosco | fine | 900 | 95.3 | 587.3 | 0.09 |

---

## Appendix: Figure Generation (Future Work)

Tools for automated figure generation from simulation results to LaTeX format will be added in:
- `report/figures/` - Generated figures (PNG, PDF)
- `report/tables/` - LaTeX table exports
- `report/generate_appendix.py` - Automated Appendix A generator

---

**Last Updated:** 2025-01-01 (Thesis Parity Phase - BLOQUE 5)
