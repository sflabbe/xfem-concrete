"""Bulk material factory.

The analysis drivers select the bulk constitutive model using `XFEMModel.bulk_material`.
This module centralizes that mapping.
"""

from __future__ import annotations

import numpy as np

from xfem_clean.constitutive import (
    LinearElasticPlaneStress,
    DruckerPrager,
    ConcreteCDP,
    ConcreteCDPReal,
)
from xfem_clean.compression_damage import (
    ConcreteCompressionModel,
    get_default_compression_model,
)
from xfem_clean.xfem.model import XFEMModel
from xfem_clean.xfem.cdp_calibration import apply_cdp_generator_calibration


def make_bulk_material(model: XFEMModel):
    """Instantiate the bulk constitutive model selected in `model.bulk_material`."""
    bm = (model.bulk_material or "elastic").lower()

    if bm == "elastic":
        return LinearElasticPlaneStress(E=float(model.E), nu=float(model.nu))

    if bm == "dp":
        return DruckerPrager(
            E=float(model.E),
            nu=float(model.nu),
            phi_deg=float(model.dp_phi_deg),
            cohesion=float(model.dp_cohesion),
            H=float(model.dp_H),
        )

    if bm == "cdp":
        # Ensure generator calibration is applied if requested.
        if getattr(model, "cdp_use_generator", False) and not getattr(model, "cdp_calibrated", False):
            apply_cdp_generator_calibration(model)

        # Dilation angle for CDP potential (psi). Prefer calibrated dilation angle.
        psi = float(
            model.cdp_psi_deg
            if model.cdp_psi_deg is not None
            else (model.cdp_dilation_angle if model.cdp_dilation_angle is not None else 36.0)
        )
        ecc = float(getattr(model, "cdp_ecc", 0.1) or 0.1)
        fb0_fc0 = float(model.cdp_fbfc if model.cdp_fbfc is not None else 1.16)
        Kc = float(model.cdp_Kc if model.cdp_Kc is not None else (2.0 / 3.0))

        # Uniaxial tables (prefer generator output; otherwise build a minimal fallback)
        if model.cdp_w_tab_m and model.cdp_sig_t_tab_pa and model.cdp_dt_tab:
            w_tab = np.asarray(model.cdp_w_tab_m, dtype=float)
            sig_t_tab = np.asarray(model.cdp_sig_t_tab_pa, dtype=float)
            dt_tab = np.asarray(model.cdp_dt_tab, dtype=float)
        else:
            ft0 = float(model.ft)
            Gf = float(model.Gf)
            w1 = max(1e-12, Gf / max(1e-12, ft0))
            wc = 5.0 * w1
            w_tab = np.linspace(0.0, wc, 60)
            sig_t_tab = ft0 * np.maximum(0.0, 1.0 - w_tab / wc)
            dt_tab = np.clip(1.0 - sig_t_tab / max(1e-12, ft0), 0.0, 0.9999)

        if model.cdp_eps_in_c_tab and model.cdp_sig_c_tab_pa and model.cdp_dc_tab:
            eps_in_c_tab = np.asarray(model.cdp_eps_in_c_tab, dtype=float)
            sig_c_tab = np.asarray(model.cdp_sig_c_tab_pa, dtype=float)
            dc_tab = np.asarray(model.cdp_dc_tab, dtype=float)
        else:
            fc0 = float(model.fc)
            eps_in_c_tab = np.linspace(0.0, 0.01, 80)
            sig_c_tab = fc0 * np.maximum(0.1, 1.0 - (eps_in_c_tab / eps_in_c_tab[-1]) ** 1.2)
            dc_tab = np.clip(1.0 - sig_c_tab / max(1e-12, fc0), 0.0, 0.9999)

        return ConcreteCDPReal(
            E=float(model.E),
            nu=float(model.nu),
            psi_deg=psi,
            ecc=ecc,
            fb0_fc0=fb0_fc0,
            Kc=Kc,
            lch=float(model.lch),
            w_tab_m=w_tab,
            sig_t_tab_pa=sig_t_tab,
            dt_tab=dt_tab,
            eps_in_c_tab=eps_in_c_tab,
            sig_c_tab_pa=sig_c_tab,
            dc_tab=dc_tab,
            f_t0=float(model.ft),
            f_c0=float(model.fc),
        )

    if bm == "cdp-lite":
        # Old, simplified CDP-like response (kept for debugging / regression).
        if getattr(model, "cdp_use_generator", False) and not getattr(model, "cdp_calibrated", False):
            apply_cdp_generator_calibration(model)
        return ConcreteCDP(
            E=float(model.E),
            nu=float(model.nu),
            ft=float(model.ft),
            fc=float(model.fc),
            Gf_t=float(model.Gf),
            lch=float(model.lch),
            phi_deg=float(model.cdp_phi_deg),
            H=float(model.cdp_H),
        )

    if bm == "compression-damage":
        # P1: Compression damage model per thesis Eq. (3.44-3.46)
        # Lightweight compression-only bulk model.
        # Default curve: fib/EC2 style hyperbolic ascending branch with linear softening.

        # Peak strain eps_c1
        if hasattr(model, "compression_eps_c1") and model.compression_eps_c1 is not None:
            eps_c1 = float(model.compression_eps_c1)
        else:
            eps_c1 = 0.0022  # Reasonable default for normal strength concrete

        # Ultimate strain eps_cu
        if hasattr(model, "compression_eps_cu") and model.compression_eps_cu is not None:
            eps_cu = float(model.compression_eps_cu)
        else:
            eps_cu = 0.0035

        # Residual stress ratio at eps_cu
        if hasattr(model, "compression_alpha_residual") and model.compression_alpha_residual is not None:
            alpha_res = float(model.compression_alpha_residual)
        else:
            alpha_res = 0.20

        # Curve kind
        if hasattr(model, "compression_curve_kind") and model.compression_curve_kind is not None:
            curve_kind = str(model.compression_curve_kind)
        else:
            curve_kind = "fib_hyperbola"

        # Strain sign convention for compression
        # Default assumes compression corresponds to negative strains.
        if hasattr(model, "compression_strain_sign") and model.compression_strain_sign is not None:
            comp_sign = float(model.compression_strain_sign)
        else:
            comp_sign = -1.0

        return ConcreteCompressionModel(
            f_c=float(model.fc),
            eps_c1=eps_c1,
            E_0=float(model.E),
            eps_cu=eps_cu,
            alpha_residual=alpha_res,
            curve_kind=curve_kind,
            compression_strain_sign=comp_sign,
        )

    raise ValueError(f"Unknown bulk_material='{model.bulk_material}'")
