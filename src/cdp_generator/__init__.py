"""
CDP Generator - Concrete Damage Plasticity Model Input Parameter Generator

This package generates input parameters for the Concrete Damage Plasticity (CDP)
model in ABAQUS, with support for strain-rate and temperature-dependent properties.

Example usage:
    from cdp_generator import calculate_stress_strain, calculate_stress_strain_temp

    # For strain rate dependent analysis
    results = calculate_stress_strain(
        f_cm=28.0,              # Mean compressive strength [MPa]
        e_c1=0.0022,            # Strain at peak compressive strength [-]
        e_clim=0.0035,          # Ultimate strain [-]
        l_ch=1.0,               # Characteristic element length [mm]
        strain_rates=[0, 2, 30, 100]  # Strain rates [1/s]
    )

    # For temperature dependent analysis
    results = calculate_stress_strain_temp(
        f_cm=28.0,
        e_c1=0.0022,
        e_clim=0.0035,
        l_ch=1.0,
        verbose=True
    )
"""

__version__ = "1.0.0"
__author__ = "CDP Generator Contributors"

# Core calculation functions
from .core import calculate_stress_strain, calculate_stress_strain_temp

# Material property functions
from .material_properties import (
    calculate_concrete_strength_properties,
    calculate_elastic_modulus,
    calculate_poisson_ratios,
    calculate_cdp_parameters,
    calculate_fracture_energy,
    calculate_characteristic_length
)

# Strain rate functions
from .strain_rate import (
    apply_strain_rate_effects,
    apply_fracture_energy_rate_effects
)

# Temperature functions
from .temperature import (
    get_eurocode_temperature_table,
    apply_temperature_effects
)

# Compression functions
from .compression import (
    calculate_compression_behavior,
    calculate_inelastic_compression,
    calculate_compression_damage
)

# Tension functions
from .tension import (
    calculate_tension_bilinear,
    calculate_tension_power_law,
    calculate_tension_damage
)

# Lazy-loaded export functions to avoid pandas import at module load time
def export_to_excel(*args, **kwargs):
    """
    Export CDP results to Excel file.

    Lazy-loads pandas to avoid import-time side effects.
    Requires pandas to be installed: pip install pandas

    Args:
        results: Dictionary of CDP results from calculate_stress_strain or calculate_stress_strain_temp
        var: List of variable values (strain rates or temperatures)
        mode: 'strain_rate' or 'temperature' to determine column labels
        filename: Output Excel filename (default: 'CDP-Results.xlsx')

    Returns:
        str: Path to the created Excel file
    """
    try:
        from .export import export_to_excel as _export_to_excel
        return _export_to_excel(*args, **kwargs)
    except ImportError as e:
        if 'pandas' in str(e):
            raise ImportError(
                "Export functions require pandas. "
                "Install with: pip install pandas openpyxl"
            ) from e
        raise

def print_properties(*args, **kwargs):
    """
    Print material properties summary.

    Args:
        f_cm: Mean compressive strength [MPa]
        results: Dictionary of CDP results
    """
    from .export import print_properties as _print_properties
    return _print_properties(*args, **kwargs)

# Lazy-loaded plotting functions to avoid matplotlib import at module load time
def plot_curve(*args, **kwargs):
    """
    Plot a single curve with appropriate labeling.

    Lazy-loads matplotlib to avoid import-time side effects.
    Requires matplotlib to be installed: pip install matplotlib

    Args:
        x: X-axis data
        y: Y-axis data
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        var: Variable value (strain rate or temperature)
        mode: 'strain_rate' or 'temperature' to determine label format
        style: Optional style dictionary
    """
    try:
        from .plotting import plot_curve as _plot_curve
        return _plot_curve(*args, **kwargs)
    except ImportError as e:
        if 'matplotlib' in str(e):
            raise ImportError(
                "Plotting functions require matplotlib. "
                "Install with: pip install matplotlib"
            ) from e
        raise

def plot_multiple_curves(*args, **kwargs):
    """
    Create a figure with multiple curves.

    Lazy-loads matplotlib to avoid import-time side effects.
    Requires matplotlib to be installed: pip install matplotlib

    Args:
        x: X-axis data (single array or list of arrays)
        y: List of Y-axis data arrays
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        var: List of variable values (strain rates or temperatures)
        mode: 'strain_rate' or 'temperature' to determine label format
    """
    try:
        from .plotting import plot_multiple_curves as _plot_multiple_curves
        return _plot_multiple_curves(*args, **kwargs)
    except ImportError as e:
        if 'matplotlib' in str(e):
            raise ImportError(
                "Plotting functions require matplotlib. "
                "Install with: pip install matplotlib"
            ) from e
        raise

def plot_all_results(*args, **kwargs):
    """
    Plot all CDP results including compression, tension, and damage.

    Lazy-loads matplotlib to avoid import-time side effects.
    Requires matplotlib to be installed: pip install matplotlib

    Args:
        results: Dictionary of CDP results from calculate_stress_strain or calculate_stress_strain_temp
        var: List of variable values (strain rates or temperatures)
        mode: 'strain_rate' or 'temperature' to determine label format
    """
    try:
        from .plotting import plot_all_results as _plot_all_results
        return _plot_all_results(*args, **kwargs)
    except ImportError as e:
        if 'matplotlib' in str(e):
            raise ImportError(
                "Plotting functions require matplotlib. "
                "Install with: pip install matplotlib"
            ) from e
        raise

__all__ = [
    # Core functions
    'calculate_stress_strain',
    'calculate_stress_strain_temp',

    # Material properties
    'calculate_concrete_strength_properties',
    'calculate_elastic_modulus',
    'calculate_poisson_ratios',
    'calculate_cdp_parameters',
    'calculate_fracture_energy',
    'calculate_characteristic_length',

    # Strain rate
    'apply_strain_rate_effects',
    'apply_fracture_energy_rate_effects',

    # Temperature
    'get_eurocode_temperature_table',
    'apply_temperature_effects',

    # Compression
    'calculate_compression_behavior',
    'calculate_inelastic_compression',
    'calculate_compression_damage',

    # Tension
    'calculate_tension_bilinear',
    'calculate_tension_power_law',
    'calculate_tension_damage',

    # Plotting and export
    'plot_curve',
    'plot_multiple_curves',
    'plot_all_results',
    'export_to_excel',
    'print_properties',
]
