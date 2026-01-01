# CDP Generator

A modular Python package for generating input parameters for the Concrete Damage Plasticity (CDP) model in ABAQUS, with support for strain-rate and temperature-dependent properties.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for material properties, strain rate effects, temperature effects, compression/tension behavior, plotting, and export
- **Strain Rate Dependent Analysis**: Calculates CDP parameters for multiple strain rates
- **Temperature Dependent Analysis**: Computes temperature-dependent properties based on Eurocode
- **Multiple Models**: Supports both bilinear and power law tension softening models
- **Comprehensive Output**: Generates stress-strain curves, damage parameters, and material properties
- **Easy Integration**: Can be imported and used in other Python projects
- **Excel Export**: Automatically exports results to Excel for use in ABAQUS
- **Visualization**: Built-in plotting functions for all results

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/sflabbe/cdp-generator.git
cd cdp-generator

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### From PyPI (when published)

```bash
pip install cdp-generator
```

### For Use in Other Projects

```bash
# Install from GitHub
pip install git+https://github.com/sflabbe/cdp-generator.git

# Or add to requirements.txt
# git+https://github.com/sflabbe/cdp-generator.git
```

## Usage

### Command Line Interface

Run the interactive CLI:

```bash
cdp-generator
```

Or if installed in development mode:

```bash
python -m cdp_generator.cli
```

### As a Python Library

#### Strain Rate Dependent Analysis

```python
from cdp_generator import calculate_stress_strain, export_to_excel, plot_all_results

# Define input parameters
f_cm = 28.0              # Mean compressive strength [MPa]
e_c1 = 0.0022            # Strain at peak compressive strength [-]
e_clim = 0.0035          # Ultimate strain [-]
l_ch = 1.0               # Characteristic element length [mm]
strain_rates = [0, 2, 30, 100]  # Strain rates [1/s]

# Calculate stress-strain relationships
results = calculate_stress_strain(f_cm, e_c1, e_clim, l_ch, strain_rates)

# Access results
print(f"Tensile strength: {results['properties']['tensile strength']:.2f} MPa")
print(f"Dilation angle: {results['properties']['dilation angle']:.2f}°")
print(f"CDP Kc: {results['properties']['Kc']:.2f}")
print(f"CDP fb/fc: {results['properties']['fbfc']:.2f}")

# Plot results
plot_all_results(results, strain_rates, mode='strain_rate')

# Export to Excel
export_to_excel(results, strain_rates, mode='strain_rate', filename='CDP-Results.xlsx')
```

#### Temperature Dependent Analysis

```python
from cdp_generator import calculate_stress_strain_temp, export_to_excel
import numpy as np

# Define input parameters
f_cm = 28.0
e_c1 = 0.0022
e_clim = 0.0035
l_ch = 1.0

# Calculate temperature-dependent properties
results = calculate_stress_strain_temp(f_cm, e_c1, e_clim, l_ch, verbose=True)

# Eurocode temperatures are used by default
temperatures = np.array([20, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100])

# Export results
export_to_excel(results, temperatures, mode='temperature', filename='CDP-Results-Temp.xlsx')
```

#### Using Individual Functions

```python
from cdp_generator import (
    calculate_concrete_strength_properties,
    calculate_elastic_modulus,
    calculate_cdp_parameters,
    calculate_compression_behavior,
    calculate_tension_bilinear
)

# Calculate material properties
f_cm = 28.0
strength_props = calculate_concrete_strength_properties(f_cm)
print(f"f_ck: {strength_props['f_ck']:.2f} MPa")
print(f"f_ctm: {strength_props['f_ctm']:.2f} MPa")

elastic_props = calculate_elastic_modulus(f_cm)
print(f"E_ci: {elastic_props['E_ci']:.2f} MPa")
print(f"E_c: {elastic_props['E_c']:.2f} MPa")
```

### Integration with Other Projects

For use in other repositories (e.g., a main ABAQUS materials library):

```python
# In your main materials repository
from cdp_generator import calculate_stress_strain
import json

def generate_abaqus_material_card(concrete_grade):
    """Generate ABAQUS material input for a concrete grade."""

    # Define properties based on grade
    if concrete_grade == "C20/25":
        f_cm = 28.0
        e_c1 = 0.0022
    elif concrete_grade == "C30/37":
        f_cm = 38.0
        e_c1 = 0.0023
    # Add more grades...

    # Calculate CDP parameters
    results = calculate_stress_strain(
        f_cm=f_cm,
        e_c1=e_c1,
        e_clim=0.0035,
        l_ch=1.0,
        strain_rates=[0]
    )

    # Extract parameters for ABAQUS input
    props = results['properties']

    return {
        'name': f'Concrete_{concrete_grade}',
        'elasticity': props['elasticity'],
        'poisson': props['poisson'],
        'dilation_angle': props['dilation angle'],
        'eccentricity': 0.1,  # Default value
        'fb0_fc0': props['fbfc'],
        'K': props['Kc'],
        'viscosity': 0.0
    }
```

## Module Structure

```
cdp_generator/
├── __init__.py              # Public API exports
├── core.py                  # Main calculation functions
├── material_properties.py   # Basic material property calculations
├── strain_rate.py           # Strain rate effect functions
├── temperature.py           # Temperature effect functions
├── compression.py           # Compression behavior calculations
├── tension.py               # Tension behavior calculations
├── plotting.py              # Visualization functions
├── export.py                # Excel export functions
└── cli.py                   # Command-line interface
```

## Input Parameters

| Parameter | Description | Units | Default |
|-----------|-------------|-------|---------|
| `f_cm` | Mean compressive strength | MPa | 28 |
| `e_c1` | Strain at peak compressive strength | - | 0.0022 |
| `e_clim` | Ultimate strain | - | 0.0035 |
| `l_ch` | Characteristic element length | mm | 1 |
| `strain_rates` | List of strain rates (strain rate mode) | 1/s | [0, 2, 30, 100] |

## Output

The package provides:

1. **Material Properties**:
   - Elastic modulus
   - Poisson's ratio
   - Tensile strength
   - Fracture energy
   - CDP parameters (dilation angle, Kc, fb/fc)

2. **Stress-Strain Data**:
   - Compression stress-strain curves
   - Compression inelastic strain
   - Compression damage
   - Tension crack opening curves
   - Tension cracking strain
   - Tension damage

3. **Visualization**:
   - Multiple stress-strain plots
   - Damage evolution curves

4. **Excel Export**:
   - Ready-to-use data for ABAQUS input

## Theory and References

This package implements:

- **Material Properties**: Based on Eurocode 2, fib Model Code 2010
- **Compression Behavior**: CEB-90 model
- **Tension Behavior**: Bilinear and power law models (FIB2010)
- **Strain Rate Effects**: Dynamic Increase Factors (DIF)
- **Temperature Effects**: Eurocode temperature-dependent reduction factors

## Disclaimer

⚠️ This package is provided for research and engineering purposes. Users should:

- Verify all outputs for plausibility
- Understand the underlying assumptions and models
- Use appropriate safety factors for design
- Validate results against experimental data when possible

The authors accept no liability for the use of this software in personal, academic, or commercial applications.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

MIT License - see LICENSE file for details

## Future Development

This package is designed to be part of a larger ABAQUS materials library. Future enhancements may include:

- Additional concrete models (CDPM2, Microplane, etc.)
- Steel material models
- Composite material models
- Direct ABAQUS input file generation
- Database of standard concrete grades
- Web interface

## Contact

For questions, issues, or suggestions, please open an issue on GitHub.
