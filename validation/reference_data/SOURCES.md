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
   - Upload figure image (PDF, PNG, JPG)
   - Set axes: define X (displacement) and Y (force) axes with units
   - Extract points: click along curve (aim for 25+ points, focus on peak and post-peak regions)
   - Export as CSV

2. **Import and clean with provided script**:
   ```bash
   cd validation/reference_data
   python import_webplotdigitizer.py --input raw_data/case.csv --output case.csv --case case_id --interactive
   ```
   The script will:
   - Auto-detect and convert units (m→mm, N→kN)
   - Remove duplicates and sort by displacement
   - Validate monotonicity
   - Generate metadata template for this SOURCES.md file

3. **Record metadata** (prompted by --interactive flag):
   - Exact figure number and page
   - Paper DOI or thesis chapter
   - Date of digitization
   - Digitization software/method

4. **Verify quality**:
   - Check that curve shape matches original figure
   - Ensure at least 20-30 points (preferably 30-50 for complex curves)
   - Verify displacement is monotonically increasing

5. **Update this file** with complete provenance information (copy from generated template)

## Data Validation

The `validation/compare_curves.py` script will:
- Warn if reference data is marked as SYNTHETIC
- Check monotonicity of displacement
- Auto-detect and convert units if necessary (mm vs m, kN vs N)
- Compute RMSE, peak error %, and energy error %

## Updating Reference Data

To replace synthetic data with real digitized curves:

1. **Digitize** experimental curve following guidelines above
2. **Import** using `import_webplotdigitizer.py` script (see Digitization Guidelines)
3. **Update** status in this file from SYNTHETIC to REAL
4. **Add** complete source metadata (figure, page, DOI, date) - use generated template
5. **Test** validation: `pytest tests/test_validation_curves.py::test_validate_<case>_coarse -v`
6. **Commit** changes: `git add <case>.csv && git commit -m "data(validation): replace <case> with digitized experimental data"`

## Import Pipeline Tools

Located in `validation/reference_data/`:

- **import_webplotdigitizer.py**: Main import script
  - Auto-detects units and converts to standard format (mm, kN)
  - Cleans data (removes duplicates, sorts, validates)
  - Generates metadata template
  - Usage: `python import_webplotdigitizer.py --input raw.csv --output case.csv --case case_id --interactive`

- **SOURCES.md** (this file): Provenance documentation for all reference datasets

## Contact

For questions about reference data or digitization process, contact the thesis author or repository maintainer.
