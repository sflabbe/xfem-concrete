"""
Parametric study module for sensitivity analysis.

Provides tools for:
- Running parameter sweeps (Gf, τmax, n_bars, etc.)
- Extracting key metrics (Pmax, u@Pmax, energy, num_cracks)
- Generating summary tables and plots
"""

from .parametric_study import run_parametric_study, extract_case_metrics

__all__ = ['run_parametric_study', 'extract_case_metrics']
