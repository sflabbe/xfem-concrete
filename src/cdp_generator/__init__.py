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

# Plotting and export functions
from .plotting import plot_curve, plot_multiple_curves, plot_all_results
from .export import export_to_excel, print_properties

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
