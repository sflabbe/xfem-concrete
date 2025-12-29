"""CDP generator calibration.

This module bridges `cdp_generator` into the XFEM solver model configuration.

`apply_cdp_generator_calibration(model)` is intended to be called once at run start.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cdp_generator

from xfem_clean.xfem.model import XFEMModel


@dataclass
class CDPCalibration:
    dilation_angle_deg: float
    Kc: float
    fb0_fc0: float
    E_cm_mpa: float
    f_ctm_mpa: float
    G_f_nmm: float
    l0_m: float
    lch_mm_used: float
    ec1: float
    eclim: float
    strain_rate: float
    concrete_class: Optional[str] = None


def _default_ec2_strains(f_cm_mpa: float) -> tuple[float, float]:
    """Reasonable EC2-like defaults for (eps_c1, eps_cu) for normal strength concrete."""
    # For f_ck <= 50 MPa, EC2 gives eps_c1≈2.2‰ and eps_cu≈3.5‰.
    # For higher strengths, EC2 modifies these; we keep a conservative fallback.
    return 0.0022, 0.0035


def calibrate_from_cdp_generator(model: XFEMModel) -> CDPCalibration:
    """Run cdp_generator and return a calibration summary (also stores curves on the model)."""
    # Determine concrete mean compressive strength used by generator (MPa)
    if model.cdp_class:
        from cdp_generator.material_properties import ec2_concrete
        props = ec2_concrete(model.cdp_class)
        f_cm = float(props["f_cm"])
    else:
        f_cm = abs(float(model.fc)) / 1e6

    ec1_def, eclim_def = _default_ec2_strains(f_cm)
    ec1 = float(getattr(model, 'cdp_ec1', 0.0) or ec1_def)
    eclim = float(getattr(model, 'cdp_eclim', 0.0) or eclim_def)
    lch_mm = float(model.lch) * 1e3
    strain_rates = [float(model.cdp_strain_rate)]

    res = cdp_generator.calculate_stress_strain(
        f_cm=f_cm,
        e_c1=ec1,
        e_clim=eclim,
        l_ch=lch_mm,
        strain_rates=strain_rates,
    )

    props = res.get("properties", {})
    dilation = float(props.get("dilation angle", 36.0))
    Kc = float(props.get("Kc", 2.0 / 3.0))
    fbfc = float(props.get("fbfc", 1.16))
    E_cm_mpa = float(props.get("elasticity", 0.0))
    f_ctm_mpa = float(props.get("tensile strength", 0.0))
    G_f_nmm = float(props.get("fracture energy", 0.0))
    l0_m = float(props.get("l0", 0.0))

    def _first_rate(x):
        """Return the first strain-rate entry if `x` is a per-rate container.

        `cdp_generator` commonly returns either:
        - a numpy array (already the desired data), or
        - a list/tuple of arrays (one per strain rate).
        """
        import numpy as _np
        if isinstance(x, (list, tuple)) and len(x) > 0:
            x0 = x[0]
            if isinstance(x0, (list, tuple, _np.ndarray)):
                return x0
        return x

    # Store uniaxial tables for the solver (first strain-rate entry).
    # IMPORTANT: In current `cdp_generator`, the *stress* arrays are typically
    # returned per-rate as lists, while *damage* is often returned directly as an
    # array (not per-rate). We handle both.
    try:
        tens = res.get("tension", {})
        w_mm = _first_rate(tens.get("crack opening", []))
        sig_t_mpa = _first_rate(tens.get("stress", []))
        dt = _first_rate(tens.get("damage", []))

        comp = res.get("compression", {})
        eps_in_c = _first_rate(comp.get("inelastic strain", []))
        sig_c_mpa = _first_rate(comp.get("inelastic stress", []))
        dc = _first_rate(comp.get("damage", []))

        # Convert to SI
        model.cdp_w_tab_m = tuple(float(x) * 1e-3 for x in list(w_mm))
        model.cdp_sig_t_tab_pa = tuple(float(x) * 1e6 for x in list(sig_t_mpa))
        model.cdp_dt_tab = tuple(float(x) for x in list(dt))

        model.cdp_eps_in_c_tab = tuple(float(x) for x in list(eps_in_c))
        model.cdp_sig_c_tab_pa = tuple(float(x) * 1e6 for x in list(sig_c_mpa))
        model.cdp_dc_tab = tuple(float(x) for x in list(dc))
    except Exception:
        # Keep solver fallback curves if generator parsing fails.
        pass

    return CDPCalibration(
        dilation_angle_deg=dilation,
        Kc=Kc,
        fb0_fc0=fbfc,
        E_cm_mpa=E_cm_mpa,
        f_ctm_mpa=f_ctm_mpa,
        G_f_nmm=G_f_nmm,
        l0_m=l0_m,
        lch_mm_used=float(lch_mm),
        ec1=ec1,
        eclim=eclim,
        strain_rate=float(strain_rates[0]),
        concrete_class=model.cdp_class,
    )


def apply_cdp_generator_calibration(model: XFEMModel) -> CDPCalibration:
    """Apply cdp_generator coefficients to the model (in-place) and return them."""
    calib = calibrate_from_cdp_generator(model)

    model.cdp_calibrated = True
    model.cdp_dilation_angle = calib.dilation_angle_deg
    # Use generator dilation angle as CDP potential parameter psi unless user opted out
    if getattr(model, "cdp_use_dilation_angle", True):
        model.cdp_psi_deg = calib.dilation_angle_deg
    model.cdp_Kc = calib.Kc
    model.cdp_fbfc = calib.fb0_fc0
    model.cdp_E_mpa = calib.E_cm_mpa
    model.cdp_fctm_mpa = calib.f_ctm_mpa
    model.cdp_Gf_nmm = calib.G_f_nmm
    model.cdp_l0_m = calib.l0_m
    model.cdp_lch_mm_used = calib.lch_mm_used
    model.cdp_ec1 = calib.ec1
    model.cdp_eclim = calib.eclim

    # Optional overrides to keep the run fully consistent with the generator.
    # These are governed by model flags so old regression cases remain possible.
    if getattr(model, "cdp_override_E", True) and calib.E_cm_mpa > 0.0:
        model.E = float(calib.E_cm_mpa) * 1e6
    if getattr(model, "cdp_override_ft", True) and calib.f_ctm_mpa > 0.0:
        model.ft = float(calib.f_ctm_mpa) * 1e6
    if getattr(model, "cdp_override_Gf", True) and calib.G_f_nmm > 0.0:
        # N/mm -> N/m (J/m^2)
        model.Gf = float(calib.G_f_nmm) * 1000.0
    # Generator's f_cm (MPa) is mean compressive strength; keep fc magnitude aligned.
    try:
        if model.cdp_class:
            from cdp_generator.material_properties import ec2_concrete
            props = ec2_concrete(model.cdp_class)
            f_cm = float(props["f_cm"]) * 1e6
            model.fc = abs(float(f_cm))
    except Exception:
        pass

    return calib
