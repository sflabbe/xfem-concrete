"""
Material Properties Module

Functions for calculating basic concrete material properties.
"""

import numpy as np
import re



def ec2_concrete(concrete_class: str):
    """
    Convenience helper: return basic EC2-like properties from a concrete class string.

    Example: "C20/25" -> f_ck=20 MPa, f_cm=f_ck+8 MPa.

    Returns keys used by the beam prototype:
      - E_cm : secant modulus [MPa] (mapped to calculate_elastic_modulus(...)[E_c])
      - f_ctm: mean tensile strength [MPa]
      - G_f : fracture energy [N/mm]
    """
    m = re.match(r"\s*C\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$", concrete_class, flags=re.I)
    if not m:
        raise ValueError(f"Unsupported concrete class '{concrete_class}'. Expected e.g. 'C20/25'.")
    f_ck = float(m.group(1))  # MPa
    # Eurocode 2: mean compressive strength approx f_cm = f_ck + 8 MPa
    f_cm = f_ck + 8.0

    strength = calculate_concrete_strength_properties(f_cm)
    Eprops = calculate_elastic_modulus(f_cm)
    G_f = calculate_fracture_energy(f_cm)

    return {
        "class": concrete_class,
        "f_ck": f_ck,
        "f_cm": f_cm,
        "E_cm": float(Eprops["E_c"]),
        "f_ctm": float(strength["f_ctm"]),
        "G_f": float(G_f),
    }


def calculate_concrete_strength_properties(f_cm):
    """
    Calculate basic concrete strength properties based on mean compressive strength.

    Args:
        f_cm: Mean compressive strength [MPa]

    Returns:
        dict: Contains f_ck (characteristic strength) and f_ctm (tensile strength)
    """
    f_ck = f_cm - 8

    # Tensile strength according to FIB2010
    if f_ck > 50:
        f_ctm = 2.12 * np.log(1 + 0.1 * f_cm)
    else:
        f_ctm = 0.3 * (f_ck) ** (2/3)

    return {
        'f_ck': f_ck,
        'f_ctm': f_ctm
    }


def calculate_elastic_modulus(f_cm, alpha_E=1.0):
    """
    Calculate elastic modulus based on concrete strength.
    Assumes Quartzite aggregates by default.

    Args:
        f_cm: Mean compressive strength [MPa]
        alpha_E: Aggregate type factor (default 1.0 for Quartzite)

    Returns:
        dict: Contains E_ci (tangent modulus) and E_c (secant modulus)
    """
    E_ci = 21500 * alpha_E * (f_cm / 10) ** (1/3)
    alpha = min(0.8 + 0.2 * f_cm / 88, 1.0)
    E_c = alpha * E_ci

    return {
        'E_ci': E_ci,
        'E_c': E_c
    }


def calculate_poisson_ratios(f_cm, E_c, e_c1):
    """
    Calculate Poisson's ratios at different states.

    Args:
        f_cm: Mean compressive strength [MPa]
        E_c: Elastic modulus [MPa]
        e_c1: Strain at peak compressive strength [-]

    Returns:
        dict: Contains v_c0 (at peak) and v_ce (elastic)
    """
    v_c0 = 0.5  # Poisson's ratio at peak engineering stress
    v_ce = 8e-6 * f_cm**2 + 0.0002 * f_cm + 0.138  # Elastic Poisson's ratio

    return {
        'v_c0': v_c0,
        'v_ce': v_ce
    }


def calculate_cdp_parameters(f_cm, E_c, e_c1, v_c0, v_ce):
    """
    Calculate CDP (Concrete Damage Plasticity) model parameters.

    Args:
        f_cm: Mean compressive strength [MPa]
        E_c: Elastic modulus [MPa]
        e_c1: Strain at peak compressive strength [-]
        v_c0: Poisson's ratio at peak
        v_ce: Elastic Poisson's ratio

    Returns:
        dict: Contains dilation angle, fbfc, and Kc
    """
    # Dilation angle [degrees]
    dilation_angle = np.arctan(
        6 * (v_c0 - v_ce) / (3 * E_c * e_c1 / f_cm + 2 * (v_c0 - v_ce) - 3)
    ) * 180 / np.pi

    # Ratio of biaxial to uniaxial compressive strength
    fbfc = 1.57 * f_cm ** (-0.09)

    # Ratio of second stress invariant on tensile meridian to that on compressive meridian
    K_c = 0.71 * f_cm ** (-0.025)

    return {
        'dilation_angle': dilation_angle,
        'fbfc': fbfc,
        'K_c': K_c
    }


def calculate_fracture_energy(f_cm):
    """
    Calculate fracture energy for concrete.

    Args:
        f_cm: Mean compressive strength [MPa]

    Returns:
        float: Fracture energy [N/mm]
    """
    return 73 * f_cm ** 0.18 / 1000


def calculate_characteristic_length(E_c, G_f, f_ctm):
    """
    Calculate characteristic element length for mesh.

    Args:
        E_c: Elastic modulus [MPa]
        G_f: Fracture energy [N/mm]
        f_ctm: Tensile strength [MPa]

    Returns:
        float: Characteristic length [m]
    """
    return 0.4 * E_c * 1e6 * G_f * 1000 / (f_ctm * 1e6) ** 2
