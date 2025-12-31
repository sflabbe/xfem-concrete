# Reference Experimental Data

This directory contains digitized P-δ curves from experimental tests reported in the thesis.

## Data Sources

### T5A1 Beam (t5a1.csv)
- **Source**: Chapter 5, Figure 5.X (Bosco et al. experiments)
- **Test**: Three-point bending beam
- **Geometry**: 1500×250×120 mm, span 1400 mm
- **Reinforcement**: 2Ø16 mm
- **Method**: Digitized from thesis Figure using WebPlotDigitizer

### VVBS3 Beam (vvbs3.csv)
- **Source**: Chapter 5, Figure 5.30
- **Test**: Three-point bending with CFRP strengthening
- **Geometry**: 4300×450×200 mm, span 3700 mm
- **Reinforcement**: 3Ø20 (bottom) + 2Ø16 (top) + CFRP sheet
- **Method**: Digitized from thesis Figure 5.30

### Sorelli Fibre Beam (sorelli.csv)
- **Source**: Chapter 6, Figure 6.X (Sorelli fibre-reinforced concrete)
- **Test**: Four-point bending with steel fibres
- **Geometry**: Small beam specimen
- **Fibres**: ρf = 1.0%, Lf/df = 60
- **Method**: Digitized from thesis Figure

## Format

All CSV files follow the same format:
```
u_mm,P_kN
0.0,0.0
...
```

Where:
- `u_mm`: Displacement (midspan deflection) in millimeters
- `P_kN`: Applied load in kilonewtons

## Usage

```python
from validation.compare_curves import load_reference_curve

ref_data = load_reference_curve('t5a1')
print(ref_data.head())
```

## Notes

- These curves represent **experimental data** from literature
- Used for quantitative validation (RMSE, peak error, energy error)
- Target tolerances: |ΔPmax| < 10%, |ΔE| < 15%
