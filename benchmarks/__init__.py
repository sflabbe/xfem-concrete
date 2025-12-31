"""
Benchmarking module for performance and scaling analysis.

Provides tools for:
- Runtime and memory profiling across mesh sizes
- Energy residual tracking
- Scaling analysis (log-log plots of steps vs time)
"""

from .benchmark_scaling import run_scaling_benchmark, generate_scaling_report

__all__ = ['run_scaling_benchmark', 'generate_scaling_report']
