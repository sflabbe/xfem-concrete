# Reference Data Sources

This document provides provenance and traceability for validation reference data.

## Data Format

All CSV files should contain:
- **Required columns**: `u_mm` (displacement in mm), `P_kN` (load in kN)
- **Optional columns**: `source_figure`, `source_page`, `digitization_method`, `date_digitized`
- **Units**: Displacement in mm, Load in kN (consistent across all files)

## Data Quality Indicators

Each reference dataset is classified as:
- **REAL**: Digitized from experimental figures in published literature
- **SYNTHETIC**: Generated for testing purposes (placeholder)
- **PENDING**: Awaiting digitization from source

## Dataset Catalog

### t5a1.csv

**Status**: SYNTHETIC (Placeholder)
**Source**: Gutiérrez thesis, Chapter 5, Figure 5.17
**Experiment**: Three-point bending beam T5A1 (BOSCO)
**Specimen**: 4000×400×200 mm beam, 4Ø12 bottom reinforcement
**Loading**: Monotonic displacement control at midspan
**Units**: u_mm (displacement), P_kN (load)
**Digitization method**: PENDING - needs WebPlotDigitizer extraction
**Date digitized**: N/A
**Notes**: Current data is synthetic smooth curve for testing. Requires replacement with actual digitized experimental data from thesis figure.

### vvbs3.csv

**Status**: SYNTHETIC (Placeholder)
**Source**: Gutiérrez thesis, Chapter 5 (VVBS3 beam with CFRP)
**Experiment**: Three-point bending beam with CFRP strengthening
**Specimen**: VVBS3 beam configuration
**Loading**: Monotonic
**Units**: u_mm, P_kN
**Digitization method**: PENDING
**Date digitized**: N/A
**Notes**: Placeholder data. Needs digitization from actual experimental curve.

### sorelli.csv

**Status**: SYNTHETIC (Placeholder)
**Source**: Sorelli et al. fiber-reinforced concrete beam tests
**Experiment**: Four-point bending with steel fibers
**Specimen**: Fiber-reinforced concrete beam
**Loading**: Monotonic
**Units**: u_mm, P_kN
**Digitization method**: PENDING
**Date digitized**: N/A
**Notes**: Placeholder data. Requires extraction from Sorelli publication figures.

## Digitization Guidelines

When digitizing experimental curves:

1. **Use WebPlotDigitizer** (https://automeris.io/WebPlotDigitizer/)
2. **Extract at least 20-30 points** to capture curve shape accurately
3. **Record metadata**:
   - Exact figure number and page
   - Paper DOI or thesis chapter
   - Date of digitization
   - Digitization software/method
4. **Verify units**: Ensure u_mm and P_kN (convert if necessary)
5. **Check monotonicity**: Displacement should be monotonically increasing
6. **Update this file** with complete provenance information

## Data Validation

The `validation/compare_curves.py` script will:
- Warn if reference data is marked as SYNTHETIC
- Check monotonicity of displacement
- Auto-detect and convert units if necessary (mm vs m, kN vs N)
- Compute RMSE, peak error %, and energy error %

## Updating Reference Data

To replace synthetic data with real digitized curves:

1. Digitize experimental curve following guidelines above
2. Save as CSV with `u_mm,P_kN` columns
3. Update status in this file from SYNTHETIC to REAL
4. Add complete source metadata (figure, page, DOI, date)
5. Commit changes with message: `data(validation): replace [case] with digitized experimental data`

## Contact

For questions about reference data or digitization process, contact the thesis author or repository maintainer.
