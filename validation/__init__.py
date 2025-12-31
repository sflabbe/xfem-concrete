"""
Validation module for quantitative comparison against thesis experimental data.

Provides tools for:
- Loading simulation and reference curves
- Computing error metrics (RMSE, peak errors, energy errors)
- Generating validation reports
"""

from .compare_curves import (
    load_simulation_curve,
    load_reference_curve,
    compute_error_metrics,
    generate_validation_report,
)

__all__ = [
    'load_simulation_curve',
    'load_reference_curve',
    'compute_error_metrics',
    'generate_validation_report',
]
