"""
Temperature Effects Module

Functions for applying temperature effects to material properties.
"""

import numpy as np


def get_eurocode_temperature_table():
    """
    Return Eurocode temperature-dependent reduction factors.

    Returns:
        ndarray: Temperature table [T, f_ratio, eps_c, eps_cu]
    """
    return np.array([
        [20,   1.00, 0.0025, 0.0200],
        [100,  1.00, 0.0040, 0.0225],
        [200,  0.95, 0.0055, 0.0250],
        [300,  0.85, 0.0070, 0.0275],
        [400,  0.75, 0.0100, 0.0300],
        [500,  0.60, 0.0150, 0.0325],
        [600,  0.45, 0.0250, 0.0350],
        [700,  0.30, 0.0250, 0.0375],
        [800,  0.15, 0.0250, 0.0400],
        [900,  0.08, 0.0250, 0.0425],
        [1000, 0.04, 0.0250, 0.0450],
        [1100, 0.01, 0.0250, 0.0475],
    ])


def apply_temperature_effects(base_props, temperature, temp_table):
    """
    Apply temperature effects to material properties based on Eurocode.

    Args:
        base_props: Dictionary of base material properties
        temperature: Temperature [°C]
        temp_table: Eurocode temperature table

    Returns:
        dict: Modified properties accounting for temperature
    """
    # Find index for this temperature
    temp_idx = np.where(temp_table[:, 0] == temperature)[0][0]

    f_ratio = temp_table[temp_idx, 1]
    eps_c = temp_table[temp_idx, 2]
    eps_cu = temp_table[temp_idx, 3]

    f_ck = base_props['f_ck']
    f_cm = base_props['f_cm']
    f_ctm = base_props['f_ctm']
    E_c1 = base_props['E_c1']

    # Compressive strength at temperature
    f_ck_temp = f_ratio * f_ck
    f_cm_temp = f_ratio * f_cm

    # Tensile strength at temperature (Eurocode approach)
    if temperature <= 100:
        f_ctm_temp_EC = f_ctm
    else:
        f_ctm_temp_EC = max(0, f_ctm * (1 - (temperature - 100) / 500))

    # Tensile strength (FIB approach)
    if f_ck > 50:
        f_ctm_temp = 2.12 * np.log(1 + 0.1 * f_cm_temp)
    else:
        f_ctm_temp = 0.3 * (f_ck_temp) ** (2/3)

    # Strain at peak stress
    e_c1_temp = eps_c

    # Elastic modulus at temperature
    E_c1_temp = (f_ratio / eps_c) / (f_ratio / temp_table[1, 2]) * E_c1

    # Tangent modulus
    k_temp = -0.0193 * f_ck_temp + 2.6408
    E_ci_temp = E_c1_temp * k_temp

    return {
        'f_cm_temp': f_cm_temp,
        'f_ck_temp': f_ck_temp,
        'f_ctm_temp': f_ctm_temp,
        'f_ctm_temp_EC': f_ctm_temp_EC,
        'e_c1_temp': e_c1_temp,
        'eps_cu': eps_cu,
        'E_c1_temp': E_c1_temp,
        'E_ci_temp': E_ci_temp
    }
