#!/usr/bin/env python3
"""
Import and clean digitized experimental curves from WebPlotDigitizer.

This script takes raw CSV exports from WebPlotDigitizer (or other digitization tools)
and normalizes them into the standard format for validation reference data.

Usage:
    # Import from WebPlotDigitizer CSV
    python import_webplotdigitizer.py --input raw_data/t5a1_digitized.csv --output t5a1.csv --case t5a1

    # Interactive mode (prompts for metadata)
    python import_webplotdigitizer.py --input raw.csv --interactive

Features:
- Auto-detect and convert units (m→mm, N→kN)
- Remove duplicate points
- Sort by displacement
- Validate monotonicity
- Remove outliers (optional)
- Generate metadata template for SOURCES.md
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Optional


def detect_units(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Auto-detect likely units based on data magnitude.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns x, y (or u, P, or similar)

    Returns
    -------
    u_unit : str
        Detected displacement unit ('m' or 'mm')
    P_unit : str
        Detected force unit ('N' or 'kN')
    """
    # Get first two columns (assume x=displacement, y=force)
    col_u = df.columns[0]
    col_P = df.columns[1]

    u_max = df[col_u].max()
    P_max = df[col_P].max()

    # Displacement: if max < 0.5, likely in meters
    u_unit = 'm' if u_max < 0.5 else 'mm'

    # Force: if max > 5000, likely in N
    P_unit = 'N' if P_max > 5000 else 'kN'

    return u_unit, P_unit


def convert_to_standard_units(df: pd.DataFrame, u_unit: str, P_unit: str) -> pd.DataFrame:
    """
    Convert displacement and force to standard units (mm, kN).

    Parameters
    ----------
    df : pd.DataFrame
        Raw data
    u_unit : str
        Current displacement unit ('m' or 'mm')
    P_unit : str
        Current force unit ('N' or 'kN')

    Returns
    -------
    df_converted : pd.DataFrame
        Data in standard units
    """
    df_out = df.copy()

    col_u = df.columns[0]
    col_P = df.columns[1]

    # Convert displacement
    if u_unit == 'm':
        print(f"  Converting displacement: {u_unit} → mm (×1000)")
        df_out[col_u] = df[col_u] * 1000.0
    elif u_unit != 'mm':
        raise ValueError(f"Unknown displacement unit: {u_unit}")

    # Convert force
    if P_unit == 'N':
        print(f"  Converting force: {P_unit} → kN (÷1000)")
        df_out[col_P] = df[col_P] / 1000.0
    elif P_unit != 'kN':
        raise ValueError(f"Unknown force unit: {P_unit}")

    # Rename columns to standard names
    df_out = df_out.rename(columns={col_u: 'u_mm', col_P: 'P_kN'})

    return df_out


def clean_data(df: pd.DataFrame, remove_duplicates: bool = True,
               remove_negative: bool = True) -> pd.DataFrame:
    """
    Clean and validate digitized data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with u_mm, P_kN columns
    remove_duplicates : bool
        Remove duplicate u values (keep first)
    remove_negative : bool
        Remove points with negative u or P

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned data
    """
    df_out = df.copy()

    n_original = len(df_out)

    # Remove negative values
    if remove_negative:
        df_out = df_out[(df_out['u_mm'] >= 0) & (df_out['P_kN'] >= 0)]
        n_removed_neg = n_original - len(df_out)
        if n_removed_neg > 0:
            print(f"  Removed {n_removed_neg} points with negative values")

    # Sort by displacement
    df_out = df_out.sort_values('u_mm').reset_index(drop=True)

    # Remove duplicates (same u value)
    if remove_duplicates:
        n_before = len(df_out)
        df_out = df_out.drop_duplicates(subset='u_mm', keep='first').reset_index(drop=True)
        n_removed_dup = n_before - len(df_out)
        if n_removed_dup > 0:
            print(f"  Removed {n_removed_dup} duplicate points (same u)")

    # Check monotonicity
    du = np.diff(df_out['u_mm'].values)
    if not np.all(du > 0):
        print(f"  WARNING: Displacement is not strictly monotonic ({np.sum(du <= 0)} violations)")
        print(f"           This may indicate digitization errors or cyclic loading")

    print(f"  Final dataset: {len(df_out)} points")

    return df_out


def generate_sources_entry(case_id: str, source_info: dict) -> str:
    """
    Generate SOURCES.md entry for the digitized dataset.

    Parameters
    ----------
    case_id : str
        Case identifier (e.g., 't5a1', 'vvbs3')
    source_info : dict
        Metadata dictionary with keys: 'figure', 'page', 'source', 'experiment', etc.

    Returns
    -------
    entry : str
        Formatted markdown entry for SOURCES.md
    """
    lines = []
    lines.append(f"### {case_id}.csv\n")
    lines.append(f"**Status**: REAL (Digitized from experimental data)")
    lines.append(f"**Source**: {source_info.get('source', 'PENDING')}")
    lines.append(f"**Figure**: {source_info.get('figure', 'PENDING')}")
    lines.append(f"**Page**: {source_info.get('page', 'PENDING')}")
    lines.append(f"**Experiment**: {source_info.get('experiment', 'PENDING')}")
    lines.append(f"**Specimen**: {source_info.get('specimen', 'PENDING')}")
    lines.append(f"**Loading**: {source_info.get('loading', 'Monotonic')}")
    lines.append(f"**Units**: u_mm (displacement), P_kN (load)")
    lines.append(f"**Digitization method**: {source_info.get('method', 'WebPlotDigitizer')}")
    lines.append(f"**Date digitized**: {source_info.get('date', 'YYYY-MM-DD')}")
    lines.append(f"**Notes**: {source_info.get('notes', 'Digitized from experimental curve.')}")
    lines.append("")

    return "\n".join(lines)


def import_curve(input_file: str, output_file: str, case_id: str,
                 u_unit: Optional[str] = None, P_unit: Optional[str] = None,
                 interactive: bool = False) -> None:
    """
    Import and clean digitized curve.

    Parameters
    ----------
    input_file : str
        Path to raw CSV from WebPlotDigitizer
    output_file : str
        Path to output standardized CSV
    case_id : str
        Case identifier
    u_unit : str, optional
        Displacement unit ('m' or 'mm'). Auto-detect if None.
    P_unit : str, optional
        Force unit ('N' or 'kN'). Auto-detect if None.
    interactive : bool
        If True, prompt user for metadata
    """
    print(f"\n{'='*70}")
    print(f"Importing: {input_file}")
    print(f"Output:    {output_file}")
    print(f"Case ID:   {case_id}")
    print(f"{'='*70}\n")

    # Load raw data
    try:
        df_raw = pd.read_csv(input_file)
    except Exception as e:
        print(f"ERROR: Failed to read {input_file}: {e}")
        sys.exit(1)

    print(f"Loaded {len(df_raw)} points from {input_file}")
    print(f"Columns: {list(df_raw.columns)}")

    # Check columns
    if len(df_raw.columns) < 2:
        print("ERROR: CSV must have at least 2 columns (displacement, force)")
        sys.exit(1)

    # Auto-detect units if not specified
    if u_unit is None or P_unit is None:
        u_unit_detected, P_unit_detected = detect_units(df_raw)
        u_unit = u_unit or u_unit_detected
        P_unit = P_unit or P_unit_detected
        print(f"Auto-detected units: u={u_unit}, P={P_unit}")

    # Convert to standard units
    df_std = convert_to_standard_units(df_raw, u_unit, P_unit)

    # Clean data
    df_clean = clean_data(df_std, remove_duplicates=True, remove_negative=True)

    # Save output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    print(f"\n✓ Saved cleaned data to: {output_path}")
    print(f"  Points: {len(df_clean)}")
    print(f"  u range: [{df_clean['u_mm'].min():.3f}, {df_clean['u_mm'].max():.3f}] mm")
    print(f"  P range: [{df_clean['P_kN'].min():.3f}, {df_clean['P_kN'].max():.3f}] kN")

    # Generate metadata template
    if interactive:
        print(f"\n{'='*70}")
        print("Please provide metadata for SOURCES.md:")
        print(f"{'='*70}")

        source_info = {}
        source_info['source'] = input("Source (thesis chapter, paper DOI): ") or "PENDING"
        source_info['figure'] = input("Figure number: ") or "PENDING"
        source_info['page'] = input("Page number: ") or "PENDING"
        source_info['experiment'] = input("Experiment description: ") or "PENDING"
        source_info['specimen'] = input("Specimen details: ") or "PENDING"
        source_info['loading'] = input("Loading type (Monotonic/Cyclic): ") or "Monotonic"
        source_info['method'] = input("Digitization method: ") or "WebPlotDigitizer"
        source_info['date'] = input("Date digitized (YYYY-MM-DD): ") or "YYYY-MM-DD"
        source_info['notes'] = input("Additional notes: ") or "Digitized from experimental curve."

        sources_entry = generate_sources_entry(case_id, source_info)

        print(f"\n{'='*70}")
        print("Add this entry to validation/reference_data/SOURCES.md:")
        print(f"{'='*70}\n")
        print(sources_entry)

        # Optionally save to file
        metadata_file = output_path.parent / f"{case_id}_metadata.txt"
        with open(metadata_file, 'w') as f:
            f.write(sources_entry)
        print(f"\n✓ Metadata template saved to: {metadata_file}")

    print(f"\n{'='*70}")
    print("Next steps:")
    print(f"{'='*70}")
    print(f"1. Review {output_path} to ensure data quality")
    print(f"2. Update validation/reference_data/SOURCES.md with provenance info")
    print(f"3. Run validation test: pytest tests/test_validation_curves.py::test_validate_{case_id}_coarse -v")
    print(f"4. Commit changes: git add {output_path} && git commit -m 'data(validation): add real digitized curve for {case_id}'")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Import and clean digitized experimental curves",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python import_webplotdigitizer.py --input raw/t5a1.csv --output t5a1.csv --case t5a1

  # With interactive metadata
  python import_webplotdigitizer.py --input raw/vvbs3.csv --output vvbs3.csv --case vvbs3 --interactive

  # Specify units explicitly
  python import_webplotdigitizer.py --input raw/sorelli.csv --output sorelli.csv --case sorelli --u-unit m --P-unit N
        """
    )

    parser.add_argument('--input', '-i', required=True, help='Input CSV file from WebPlotDigitizer')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file (standard format)')
    parser.add_argument('--case', '-c', required=True, help='Case ID (e.g., t5a1, vvbs3, sorelli)')
    parser.add_argument('--u-unit', choices=['m', 'mm'], help='Displacement unit (auto-detect if not specified)')
    parser.add_argument('--P-unit', choices=['N', 'kN'], help='Force unit (auto-detect if not specified)')
    parser.add_argument('--interactive', action='store_true', help='Prompt for metadata interactively')

    args = parser.parse_args()

    import_curve(
        input_file=args.input,
        output_file=args.output,
        case_id=args.case,
        u_unit=args.u_unit,
        P_unit=args.P_unit,
        interactive=args.interactive,
    )


if __name__ == "__main__":
    main()
