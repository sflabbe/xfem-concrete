"""
Command Line Interface Module

Interactive CLI for generating CDP parameters.
"""

import numpy as np
from .core import calculate_stress_strain, calculate_stress_strain_temp
from .plotting import plot_all_results
from .export import export_to_excel, print_properties


def main():
    """
    Main CLI function for interactive CDP parameter generation.
    """
    print("CDP Generator - Concrete Damage Plasticity Model Input Parameter Generator")
    print("=" * 80)

    # Get user inputs with defaults
    f_cm = input("Enter the compressive strength of the concrete (MPa) [Default: 28]: ")
    e_c1 = input("Enter the strain at maximum compressive strength c_i [Default: 0.0022]: ")
    e_clim = input("Enter the strain at ultimate state [Default: 0.0035]: ")
    l_ch = input("Enter the characteristic element length of the mesh (mm) [Default: 1]: ")
    e_rate = input("Enter the strain rates additional to 0/s, separated by a comma [Default: 2,30,100]: ")

    while True:
        temp_input = input("Temperature Dependent Data? (y/n) [Default: n]: ").strip().lower()
        if temp_input in ("y", "n", ""):
            is_strain_rate_mode = (temp_input != "y")
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    # Assign default values if no input is provided
    f_cm = float(f_cm.strip()) if f_cm.strip() else 28
    e_c1 = float(e_c1.strip()) if e_c1.strip() else 0.0022
    e_clim = float(e_clim.strip()) if e_clim.strip() else 0.0035
    l_ch = float(l_ch.strip()) if l_ch.strip() else 1
    strain_rates = [0] + list(map(float, e_rate.strip().split(','))) if e_rate.strip() else [0, 2, 30, 100]
    temperatures = np.array([20, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100])

    # Calculate stress-strain relationships
    if is_strain_rate_mode:
        results = calculate_stress_strain(f_cm, e_c1, e_clim, l_ch, strain_rates)
        var = strain_rates
        mode = 'strain_rate'
    else:
        results = calculate_stress_strain_temp(f_cm, e_c1, e_clim, l_ch)
        var = temperatures
        mode = 'temperature'

    # Plot results
    plot_all_results(results, var, mode)

    # Print properties
    print_properties(f_cm, results)

    # Export to Excel
    print('\nExporting results to Excel...')
    excel_file = export_to_excel(results, var, mode)
    print(f'Results exported successfully to {excel_file}')


if __name__ == "__main__":
    main()
