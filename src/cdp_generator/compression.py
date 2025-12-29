"""
Compression Behavior Module

Functions for calculating compression behavior and damage.
"""

import numpy as np


def calculate_compression_behavior(f_cm, e_c1, E_ci, E_c1, n_points, e_max):
    """
    Calculate compressive stress-strain behavior using CEB-90 model.

    Args:
        f_cm: Mean compressive strength [MPa]
        e_c1: Strain at peak stress [-]
        E_ci: Tangent modulus [MPa]
        E_c1: Secant modulus [MPa]
        n_points: Number of points in curve
        e_max: Maximum strain to calculate [-]

    Returns:
        dict: Contains strain and stress arrays
    """
    # Strain array
    strain = np.linspace(0, e_max, n_points)

    # CEB-90 parameters
    eta_E = E_ci / E_c1
    e_clim = e_c1 * (0.5 * (0.5 * eta_E + 1) +
                     (0.25 * ((0.5 * eta_E + 1) ** 2) - 0.5) ** 0.5)
    eta = strain / e_c1
    eta_lim = e_clim / e_c1

    xi = 4 * (eta_lim**2 * (eta_E - 2) + 2 * eta_lim - eta_E) / \
         ((eta_lim * (eta_E - 2) + 1) ** 2)

    # Calculate stress
    stress = np.zeros_like(strain)
    for i in range(len(strain)):
        if strain[i] <= e_clim:
            # Ascending branch
            stress[i] = (eta_E * strain[i] / e_c1 - (strain[i] / e_c1) ** 2) / \
                       (1 + (eta_E - 2) * (strain[i] / e_c1)) * f_cm
        else:
            # Descending branch
            stress[i] = f_cm / ((xi / eta_lim - 2 / (eta_lim**2)) *
                               ((strain[i] / e_c1) ** 2) +
                               (4 / eta_lim - xi) * strain[i] / e_c1)

    return {
        'strain': strain,
        'stress': stress,
        'e_clim': e_clim
    }


def calculate_inelastic_compression(strain, stress, f_cm, E_c1):
    """
    Calculate inelastic strain and stress for compression.

    Args:
        strain: Total strain array [-]
        stress: Total stress array [MPa]
        f_cm: Mean compressive strength [MPa]
        E_c1: Secant modulus [MPa]

    Returns:
        dict: Contains inelastic strain and stress arrays
    """
    f_cel = f_cm * 0.4  # Elastic limit

    inelastic_strain = []
    inelastic_stress = []
    first_point = False

    for i in range(len(strain)):
        if stress[i] > f_cel:
            if strain[i] - stress[i] / E_c1 > 0:
                if not first_point:
                    first_point = True
                    inelastic_strain.append(0)
                else:
                    inelastic_strain.append(strain[i] - stress[i] / E_c1)
                inelastic_stress.append(stress[i])
        elif strain[i] > strain[np.argmax(stress)]:
            inelastic_strain.append(strain[i] - stress[i] / E_c1)
            inelastic_stress.append(stress[i])

    return {
        'inelastic_strain': np.array(inelastic_strain),
        'inelastic_stress': np.array(inelastic_stress)
    }


def calculate_compression_damage(inelastic_stress, inelastic_strain, f_cm):
    """
    Calculate compression damage parameter.

    Args:
        inelastic_stress: Inelastic stress array [MPa]
        inelastic_strain: Inelastic strain array [-]
        f_cm: Mean compressive strength [MPa]

    Returns:
        ndarray: Damage array [-]
    """
    damage = np.zeros(len(inelastic_strain))

    for i in range(len(inelastic_strain)):
        if inelastic_stress[i] < f_cm:
            damage[i] = 1 - inelastic_stress[i] / f_cm

    # Ensure damage is monotonically increasing
    damage[np.gradient(damage) < 0] = 0

    return damage
