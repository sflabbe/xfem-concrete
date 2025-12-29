"""
Strain Rate Effects Module

Functions for applying strain rate effects to material properties.
"""

import numpy as np


def apply_strain_rate_effects(base_props, strain_rate):
    """
    Apply strain rate effects to material properties.

    Args:
        base_props: Dictionary of base material properties
        strain_rate: Strain rate [1/s]

    Returns:
        dict: Modified properties accounting for strain rate
    """
    f_cm = base_props['f_cm']
    f_ctm = base_props['f_ctm']
    e_c1 = base_props['e_c1']
    E_ci = base_props['E_ci']

    # No strain rate effects for static case
    if strain_rate == 0:
        return {
            'f_cm_dyn': f_cm,
            'f_ctm_dyn': f_ctm,
            'e_c1_dyn': e_c1,
            'E_ci_dyn': E_ci
        }

    # Compressive strength DIF (Dynamic Increase Factor)
    if strain_rate > 30:
        f_cm_dyn = 0.012 * f_cm * (strain_rate / 0.00003) ** (1/3)
    else:
        f_cm_dyn = f_cm * (strain_rate / 0.00003) ** 0.014

    # Tensile strength DIF
    if strain_rate > 10:
        f_ctm_dyn = 0.0062 * f_ctm * (strain_rate / 1e-6) ** (1/3)
    else:
        f_ctm_dyn = f_ctm * (strain_rate / 1e-6) ** 0.018

    # Strain at peak stress
    e_c1_dyn = e_c1 * (strain_rate / 0.00003) ** 0.02

    # Elastic modulus
    E_ci_dyn = E_ci * (strain_rate / 0.00003) ** 0.025

    return {
        'f_cm_dyn': f_cm_dyn,
        'f_ctm_dyn': f_ctm_dyn,
        'e_c1_dyn': e_c1_dyn,
        'E_ci_dyn': E_ci_dyn
    }


def apply_fracture_energy_rate_effects(G_f, strain_rate, l_ch):
    """
    Apply strain rate effects to fracture energy (Li et al.).

    Args:
        G_f: Base fracture energy [N/mm]
        strain_rate: Strain rate [1/s]
        l_ch: Characteristic element length [mm]

    Returns:
        float: Dynamic fracture energy [N/mm]
    """
    if strain_rate == 0:
        return G_f

    w_rate = strain_rate * l_ch

    if w_rate > 200:
        b_g = (200 / 0.01) ** (0.08 - 0.62)
        G_f_dyn = G_f * b_g * (w_rate / 0.01) ** 0.62
    else:
        G_f_dyn = G_f * (w_rate / 0.01) ** 0.08

    return G_f_dyn
