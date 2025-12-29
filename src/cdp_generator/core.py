"""
Core Module

Main calculation functions for stress-strain relationships.
"""

import numpy as np

from .material_properties import (
    calculate_concrete_strength_properties,
    calculate_elastic_modulus,
    calculate_poisson_ratios,
    calculate_cdp_parameters,
    calculate_fracture_energy,
    calculate_characteristic_length
)
from .strain_rate import apply_strain_rate_effects, apply_fracture_energy_rate_effects
from .temperature import get_eurocode_temperature_table, apply_temperature_effects
from .compression import (
    calculate_compression_behavior,
    calculate_inelastic_compression,
    calculate_compression_damage
)
from .tension import (
    calculate_tension_bilinear,
    calculate_tension_power_law,
    calculate_tension_damage
)


def calculate_stress_strain(f_cm, e_c1, e_clim, l_ch, strain_rates):
    """
    Calculate stress-strain relationships for different strain rates.

    Args:
        f_cm: Mean compressive strength [MPa]
        e_c1: Strain at peak compressive strength [-]
        e_clim: Ultimate strain [-]
        l_ch: Characteristic element length [mm]
        strain_rates: List of strain rates [1/s]

    Returns:
        dict: Complete results including properties and stress-strain data
    """
    n_points = 20

    # Base material properties
    strength_props = calculate_concrete_strength_properties(f_cm)
    f_ck = strength_props['f_ck']
    f_ctm = strength_props['f_ctm']

    elastic_props = calculate_elastic_modulus(f_cm)
    E_ci = elastic_props['E_ci']
    E_c = elastic_props['E_c']
    E_c1 = f_cm / e_c1

    poisson_props = calculate_poisson_ratios(f_cm, E_c, e_c1)
    v_c0 = poisson_props['v_c0']
    v_ce = poisson_props['v_ce']

    cdp_params = calculate_cdp_parameters(f_cm, E_c, e_c1, v_c0, v_ce)

    G_f = calculate_fracture_energy(f_cm)
    G_c = E_c / (2 * (1 + v_ce))
    l_0 = calculate_characteristic_length(E_c, G_f, f_ctm)

    # Storage for rate-dependent results
    compression_stress = []
    compression_inelastic_strain = []
    compression_inelastic_stress = []
    tension_crack_opening = []
    tension_stress_bilinear = []
    tension_stress_power = []
    tension_cracking_strain = []

    # Base properties for rate calculations
    base_props = {
        'f_cm': f_cm,
        'f_ctm': f_ctm,
        'e_c1': e_c1,
        'E_ci': E_ci
    }

    # Calculate for each strain rate
    for rate in strain_rates:
        # Apply strain rate effects
        rate_props = apply_strain_rate_effects(base_props, rate)
        f_cm_dyn = rate_props['f_cm_dyn']
        f_ctm_dyn = rate_props['f_ctm_dyn']
        e_c1_dyn = rate_props['e_c1_dyn']
        E_ci_dyn = rate_props['E_ci_dyn']
        E_c1_dyn = f_cm_dyn / e_c1_dyn

        # Compression behavior
        comp_result = calculate_compression_behavior(
            f_cm_dyn, e_c1_dyn, E_ci_dyn, E_c1_dyn,
            n_points, e_clim * 3
        )
        compression_stress.append(comp_result['stress'])

        # Inelastic compression
        inel_result = calculate_inelastic_compression(
            comp_result['strain'], comp_result['stress'],
            f_cm_dyn, E_c1_dyn
        )
        compression_inelastic_strain.append(inel_result['inelastic_strain'])
        compression_inelastic_stress.append(inel_result['inelastic_stress'])

        # Tension behavior
        G_f_dyn = apply_fracture_energy_rate_effects(G_f, rate, l_ch)

        # Bilinear tension model
        tension_result = calculate_tension_bilinear(f_ctm_dyn, G_f_dyn, n_points)
        tension_crack_opening.append(tension_result['crack_opening'])
        tension_stress_bilinear.append(tension_result['stress'])

        # Power law tension model
        stress_power = calculate_tension_power_law(
            f_ctm_dyn, G_f_dyn,
            tension_result['w_c'],
            tension_result['crack_opening']
        )
        tension_stress_power.append(stress_power)

        # Cracking strain
        cracking_strain = tension_result['crack_opening'] / l_ch
        tension_cracking_strain.append(cracking_strain)

    # Interpolate all curves to match first strain rate grid
    for i in range(1, len(strain_rates)):
        compression_inelastic_stress[i] = np.interp(
            compression_inelastic_strain[0],
            compression_inelastic_strain[i],
            compression_inelastic_stress[i]
        )
        compression_inelastic_strain[i] = compression_inelastic_strain[0]

        tension_stress_bilinear[i] = np.interp(
            tension_crack_opening[0],
            tension_crack_opening[i],
            tension_stress_bilinear[i]
        )
        tension_stress_power[i] = np.interp(
            tension_crack_opening[0],
            tension_crack_opening[i],
            tension_stress_power[i]
        )
        tension_cracking_strain[i] = tension_cracking_strain[0]
        tension_crack_opening[i] = tension_crack_opening[0]

    # Calculate damage (based on reference rate)
    damage_c = calculate_compression_damage(
        compression_inelastic_stress[0],
        compression_inelastic_strain[0],
        f_cm
    )

    damage_t = calculate_tension_damage(tension_stress_bilinear[0], f_ctm)
    damage_t_power = calculate_tension_damage(tension_stress_power[0], f_ctm)

    return {
        'properties': {
            'elasticity': E_c,
            'shear': G_c,
            'fracture energy': G_f,
            'tensile strength': f_ctm,
            'dilation angle': cdp_params['dilation_angle'],
            'poisson': v_ce,
            'Kc': cdp_params['K_c'],
            'fbfc': cdp_params['fbfc'],
            'l0': l_0
        },
        'compression': {
            'strain': comp_result['strain'],
            'stress': compression_stress,
            'inelastic strain': compression_inelastic_strain,
            'inelastic stress': compression_inelastic_stress,
            'damage': damage_c
        },
        'tension': {
            'crack opening': tension_crack_opening,
            'stress': tension_stress_bilinear,
            'stress exponential': tension_stress_power,
            'cracking strain': tension_cracking_strain,
            'cracking stress': tension_stress_bilinear,
            'damage': damage_t,
            'damage exponential': damage_t_power
        }
    }


def calculate_stress_strain_temp(f_cm, e_c1, e_clim, l_ch, verbose=True):
    """
    Calculate stress-strain relationships for different temperatures.

    Args:
        f_cm: Mean compressive strength [MPa]
        e_c1: Strain at peak compressive strength [-]
        e_clim: Ultimate strain [-]
        l_ch: Characteristic element length [mm]
        verbose: Print temperature-specific information (default: True)

    Returns:
        dict: Complete results including properties and stress-strain data
    """
    n_points = 40

    # Base material properties
    strength_props = calculate_concrete_strength_properties(f_cm)
    f_ck = strength_props['f_ck']
    f_ctm = strength_props['f_ctm']

    elastic_props = calculate_elastic_modulus(f_cm)
    E_ci = elastic_props['E_ci']
    E_c = elastic_props['E_c']
    E_c1 = f_cm / e_c1

    poisson_props = calculate_poisson_ratios(f_cm, E_c, e_c1)
    v_c0 = poisson_props['v_c0']
    v_ce = poisson_props['v_ce']

    cdp_params = calculate_cdp_parameters(f_cm, E_c, e_c1, v_c0, v_ce)

    G_f = calculate_fracture_energy(f_cm)
    G_c = E_c / (2 * (1 + v_ce))
    l_0 = calculate_characteristic_length(E_c, G_f, f_ctm)

    # Get Eurocode temperature table
    temp_table = get_eurocode_temperature_table()
    temperatures = temp_table[:, 0]

    # Storage for temperature-dependent results
    compression_strain_arrays = []
    compression_stress = []
    compression_inelastic_strain = []
    compression_inelastic_stress = []
    tension_crack_opening = []
    tension_stress_bilinear = []
    tension_stress_power = []
    tension_cracking_strain = []

    # Base properties for temperature calculations
    base_props = {
        'f_cm': f_cm,
        'f_ck': f_ck,
        'f_ctm': f_ctm,
        'E_c1': E_c1
    }

    # Calculate for each temperature
    for temp in temperatures:
        # Apply temperature effects
        temp_props = apply_temperature_effects(base_props, temp, temp_table)
        f_cm_temp = temp_props['f_cm_temp']
        f_ctm_temp = temp_props['f_ctm_temp']
        e_c1_temp = temp_props['e_c1_temp']
        eps_cu = temp_props['eps_cu']
        E_ci_temp = temp_props['E_ci_temp']
        E_c1_temp = temp_props['E_c1_temp']

        if verbose:
            print(f'\nTensile Strength [N/mm²], at {temp} [°C]: {f_ctm_temp:.3f}')
            print(f'Tensile Strength [N/mm²] (EC2), at {temp} [°C]: {temp_props["f_ctm_temp_EC"]:.3f}')
            print(f'E-Modul sekant [N/mm²], at {temp} [°C]: {E_c1_temp:.1f}')
            print(f'E-Modul tangent [N/mm²], at {temp} [°C]: {E_ci_temp:.1f}')

        # Compression behavior
        comp_result = calculate_compression_behavior(
            f_cm_temp, e_c1_temp, E_ci_temp, E_c1_temp,
            n_points, eps_cu * 2
        )
        compression_strain_arrays.append(comp_result['strain'])
        compression_stress.append(comp_result['stress'])

        # Inelastic compression
        inel_result = calculate_inelastic_compression(
            comp_result['strain'], comp_result['stress'],
            f_cm_temp, E_c1_temp
        )
        compression_inelastic_strain.append(inel_result['inelastic_strain'])
        compression_inelastic_stress.append(inel_result['inelastic_stress'])

        # Tension behavior
        G_f_temp = calculate_fracture_energy(f_cm_temp)

        # Bilinear tension model
        tension_result = calculate_tension_bilinear(f_ctm_temp, G_f_temp, n_points)
        tension_crack_opening.append(tension_result['crack_opening'])
        tension_stress_bilinear.append(tension_result['stress'])

        # Power law tension model
        stress_power = calculate_tension_power_law(
            f_ctm_temp, G_f_temp,
            tension_result['w_c'],
            tension_result['crack_opening']
        )
        tension_stress_power.append(stress_power)

        # Cracking strain
        cracking_strain = tension_result['crack_opening'] / l_ch
        tension_cracking_strain.append(cracking_strain)

    # Interpolate all curves to match first temperature grid
    for i in range(1, len(temperatures)):
        compression_inelastic_stress[i] = np.interp(
            compression_inelastic_strain[0],
            compression_inelastic_strain[i],
            compression_inelastic_stress[i]
        )
        compression_inelastic_strain[i] = compression_inelastic_strain[0]

        tension_stress_bilinear[i] = np.interp(
            tension_crack_opening[0],
            tension_crack_opening[i],
            tension_stress_bilinear[i]
        )
        tension_stress_power[i] = np.interp(
            tension_crack_opening[0],
            tension_crack_opening[i],
            tension_stress_power[i]
        )
        tension_cracking_strain[i] = tension_cracking_strain[0]
        tension_crack_opening[i] = tension_crack_opening[0]

    # Calculate damage (based on reference temperature)
    damage_c = calculate_compression_damage(
        compression_inelastic_stress[0],
        compression_inelastic_strain[0],
        f_cm
    )

    damage_t = calculate_tension_damage(tension_stress_bilinear[0], f_ctm)
    damage_t_power = calculate_tension_damage(tension_stress_power[0], f_ctm)

    return {
        'properties': {
            'elasticity': E_c,
            'shear': G_c,
            'fracture energy': G_f,
            'tensile strength': f_ctm,
            'dilation angle': cdp_params['dilation_angle'],
            'poisson': v_ce,
            'Kc': cdp_params['K_c'],
            'fbfc': cdp_params['fbfc'],
            'l0': l_0
        },
        'compression': {
            'strain': compression_strain_arrays[0],  # For compatibility
            'strain temp': compression_strain_arrays,  # Temperature-specific strains
            'stress': compression_stress,
            'inelastic strain': compression_inelastic_strain,
            'inelastic stress': compression_inelastic_stress,
            'damage': damage_c
        },
        'tension': {
            'crack opening': tension_crack_opening,
            'stress': tension_stress_bilinear,
            'stress exponential': tension_stress_power,
            'cracking strain': tension_cracking_strain,
            'cracking stress': tension_stress_bilinear,
            'damage': damage_t,
            'damage exponential': damage_t_power
        }
    }
