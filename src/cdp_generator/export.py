"""
Export Module

Functions for exporting CDP results to Excel and other formats.
"""

import numpy as np
import pandas as pd


def export_to_excel(results, var, mode='strain_rate', filename='CDP-Results.xlsx'):
    """
    Export CDP results to Excel file.

    Args:
        results: Dictionary of CDP results from calculate_stress_strain or calculate_stress_strain_temp
        var: List of variable values (strain rates or temperatures)
        mode: 'strain_rate' or 'temperature' to determine column labels
        filename: Output Excel filename (default: 'CDP-Results.xlsx')

    Returns:
        str: Path to the created Excel file
    """
    is_strain_rate_mode = (mode == 'strain_rate')
    var_label = 'Strain Rate [1/s]' if is_strain_rate_mode else 'Temperature [°C]'

    # Prepare export arrays
    var_export_el = np.array([])
    var_export = np.array([])
    compressive_stress_el = np.array([])
    compressive_stress = np.array([])
    compressive_strain_el = np.array([])
    compressive_strain = np.array([])
    var_export_tension = np.array([])
    tension_stress = np.array([])
    tension_strain = np.array([])
    tension_crack = np.array([])
    tension_stress_exp = np.array([])

    for i in range(len(var)):
        # For elastic compression
        if is_strain_rate_mode:
            # Strain rate mode: same strain array for all rates
            strain_length = len(results['compression']['strain'])
            var_export_el = np.concatenate((var_export_el, np.ones(strain_length) * var[i]))
            compressive_strain_el = np.concatenate((compressive_strain_el, results['compression']['strain']))
        else:
            # Temperature mode: different strain arrays
            strain_length = len(results['compression']['strain temp'][i])
            var_export_el = np.concatenate((var_export_el, np.ones(strain_length) * var[i]))
            compressive_strain_el = np.concatenate((compressive_strain_el, results['compression']['strain temp'][i]))

        compressive_stress_el = np.concatenate((compressive_stress_el, results['compression']['stress'][i]))

        # For inelastic compression
        var_export = np.concatenate((var_export, np.ones(len(results['compression']['inelastic strain'][0])) * var[i]))
        compressive_stress = np.concatenate((compressive_stress, results['compression']['inelastic stress'][i]))
        compressive_strain = np.concatenate((compressive_strain, results['compression']['inelastic strain'][i]))

        # For tension
        var_export_tension = np.concatenate((var_export_tension, np.ones(len(results['tension']['cracking strain'][0])) * var[i]))
        tension_stress = np.concatenate((tension_stress, results['tension']['stress'][i]))
        tension_strain = np.concatenate((tension_strain, results['tension']['cracking strain'][i]))
        tension_crack = np.concatenate((tension_crack, results['tension']['crack opening'][i]))
        tension_stress_exp = np.concatenate((tension_stress_exp, results['tension']['stress exponential'][i]))

    # Create DataFrames
    compression_strain_el_df = pd.DataFrame({
        'Compressive Stress [MPa]': compressive_stress_el,
        'Strain [-]': compressive_strain_el,
        var_label: var_export_el
    })

    compression_strain_df = pd.DataFrame({
        'Compressive Stress [MPa]': compressive_stress,
        'Inelastic Strain [-]': compressive_strain,
        var_label: var_export
    })

    compression_damage_df = pd.DataFrame({
        'Damage [-]': results['compression']['damage'],
        'Inelastic Strain [-]': results['compression']['inelastic strain'][0]
    })

    tension_cracking_df = pd.DataFrame({
        'Tension Stress [MPa]': tension_stress,
        'Crack Opening [mm]': tension_crack,
        var_label: var_export_tension
    })

    tension_cracking_strain_df = pd.DataFrame({
        'Tension Stress [MPa]': tension_stress,
        'Cracking Strain [-]': tension_strain,
        var_label: var_export_tension
    })

    tension_damage_df = pd.DataFrame({
        'Damage [-]': results['tension']['damage'],
        'Cracking Strain [-]': results['tension']['cracking strain'][0]
    })

    tension_cracking_power_df = pd.DataFrame({
        'Tension Stress [MPa]': tension_stress_exp,
        'Cracking Strain [-]': tension_strain,
        var_label: var_export_tension
    })

    tension_damage_power_df = pd.DataFrame({
        'Damage [-]': results['tension']['damage exponential'],
        'Cracking Strain [-]': results['tension']['cracking strain'][0]
    })

    # Write to Excel
    with pd.ExcelWriter(filename) as writer:
        compression_strain_el_df.to_excel(writer, sheet_name="Compression Stress-Strain", index=False)
        compression_strain_df.to_excel(writer, sheet_name="Compression Inl.Strain", index=False)
        compression_damage_df.to_excel(writer, sheet_name="Compression Damage", index=False)
        tension_cracking_df.to_excel(writer, sheet_name="Tension Cracking", index=False)
        tension_cracking_strain_df.to_excel(writer, sheet_name="Tension Cr.Strain", index=False)
        tension_damage_df.to_excel(writer, sheet_name="Tension Damage", index=False)
        tension_cracking_power_df.to_excel(writer, sheet_name="Tension Cracking Power", index=False)
        tension_damage_power_df.to_excel(writer, sheet_name="Tension Damage Power", index=False)

    return filename


def print_properties(f_cm, results):
    """
    Print material properties summary.

    Args:
        f_cm: Mean compressive strength [MPa]
        results: Dictionary of CDP results
    """
    print('\n' + '='*60)
    print('MATERIAL PROPERTIES SUMMARY')
    print('='*60)
    print(f'Compressive Strength: {f_cm} [MPa]')
    print(f'Tensile Strength: {results["properties"]["tensile strength"]:.2f} [MPa]')
    print(f'Elasticity Modulus: {results["properties"]["elasticity"]:.2f} [MPa]')
    print(f'Poisson: {results["properties"]["poisson"]:.2f} [-]')
    print(f'Shear Modulus: {results["properties"]["shear"]:.2f} [MPa]')
    print(f'Fracture Energy: {results["properties"]["fracture energy"]:.4f} [N/mm]')
    print(f'CDP Dilation angle: {results["properties"]["dilation angle"]:.2f} [°]')
    print(f'CDP fb/fc: {results["properties"]["fbfc"]:.2f} [-]')
    print(f'CDP Kc: {results["properties"]["Kc"]:.2f} [-]')
    print(f'Max. Mesh Size: {results["properties"]["l0"]:.2f} [m]')
    print('='*60)
