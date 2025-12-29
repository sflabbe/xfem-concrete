"""Run-time info printing utilities."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional

from xfem_clean.xfem.model import XFEMModel
from xfem_clean.numba.utils import numba_available


def _fmt_pa(x: float) -> str:
    x = float(x)
    if abs(x) >= 1e9:
        return f"{x/1e9:.3g} GPa"
    if abs(x) >= 1e6:
        return f"{x/1e6:.3g} MPa"
    if abs(x) >= 1e3:
        return f"{x/1e3:.3g} kPa"
    return f"{x:.3g} Pa"


def _fmt_float(x: Optional[float], fmt: str = "{:.3g}") -> str:
    if x is None:
        return "n/a"
    try:
        return fmt.format(float(x))
    except Exception:
        return "n/a"


def print_run_header(tag: str) -> None:
    # Use a stable timezone so logs are comparable across machines.
    ts = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"\n[run] {tag}  start={ts}")


def print_material_summary(model: XFEMModel) -> None:
    def _print_numba() -> None:
        req = bool(getattr(model, "use_numba", False))
        av = bool(numba_available())
        bm_local = (model.bulk_material or "elastic").lower()
        bulk_supported = bm_local in ("elastic", "dp")
        bulk_on = bool(req and av and bulk_supported)
        coh_on = bool(req and av)
        if req and not av:
            print("[numba] requested=yes  available=no  (falling back to pure Python)")
        else:
            print(
                f"[numba] requested={'yes' if req else 'no'}  available={'yes' if av else 'no'}"
                f"  cohesive_kernels={'yes' if coh_on else 'no'}"
                f"  bulk_kernels={'yes' if bulk_on else 'no'}"
            )

    bm = (model.bulk_material or "elastic").lower()
    if bm == "elastic":
        print(f"[material] (elastic) E={_fmt_pa(model.E)}  nu={model.nu:.3g}")
        _print_numba()
        return

    if bm == "dp":
        print(f"[material] (drucker-prager) E={_fmt_pa(model.E)}  nu={model.nu:.3g}")
        print(f"[material] phi={model.dp_phi_deg:.3g} deg  cohesion={_fmt_pa(model.dp_cohesion)}  H={_fmt_pa(model.dp_H)}")
        _print_numba()
        return

    if bm == "cdp":
        print(f"[material] (CDP) E={_fmt_pa(model.E)}  nu={model.nu:.3g}")
        print(f"[material] ft={_fmt_pa(model.ft)}  fc={_fmt_pa(model.fc)}  Gf={model.Gf:.3g} N/m  lch={model.lch:.4g} m")
        psi = getattr(model, "cdp_psi_deg", None)
        ecc = getattr(model, "cdp_ecc", None)
        print(f"[material] psi={_fmt_float(psi, '{:.3g}')} deg  ecc={_fmt_float(ecc, '{:.3g}')}")
        if getattr(model, "cdp_calibrated", False):
            cls = getattr(model, "cdp_class", None)
            cls = (cls.strip() if isinstance(cls, str) and cls.strip() else None)
            print("[cdp_generator] calibrated:", "yes" + (f" (class={cls})" if cls else ""))
            print(
                f"[cdp_generator] dilation angle={_fmt_float(model.cdp_dilation_angle, '{:.3g}')} deg"
                f"  Kc={_fmt_float(model.cdp_Kc, '{:.4f}')}  fbfc={_fmt_float(model.cdp_fbfc, '{:.4f}')}"
            )
            print(
                f"[cdp_generator] E_cm={_fmt_float(model.cdp_E_mpa, '{:.3f}')} MPa"
                f"  f_ctm={_fmt_float(model.cdp_fctm_mpa, '{:.3f}')} MPa"
                f"  G_f={_fmt_float(model.cdp_Gf_nmm, '{:.4f}')} N/mm"
            )
            print(
                f"[cdp_generator] l0={_fmt_float(model.cdp_l0_m, '{:.4g}')} m"
                f"  lch_used={_fmt_float(model.cdp_lch_mm_used, '{:.4g}')} mm"
                f"  ec1={_fmt_float(model.cdp_ec1, '{:.4g}')}  eclim={_fmt_float(model.cdp_eclim, '{:.4g}')}"
                f"  rate={_fmt_float(model.cdp_strain_rate, '{:.3g}')} 1/s"
            )
            nt = len(getattr(model, "cdp_w_tab_m", ()) or ())
            nc = len(getattr(model, "cdp_eps_in_c_tab", ()) or ())
            if nt and nc:
                print(f"[cdp_generator] curves: tension_pts={nt}  compression_pts={nc}")
            else:
                print("[cdp_generator] curves: (not stored)")
        else:
            print("[cdp_generator] calibrated: no")
        _print_numba()
        return

    if bm == "cdp-lite":
        print(f"[material] (CDP-lite) E={_fmt_pa(model.E)}  nu={model.nu:.3g}")
        print(f"[material] ft={_fmt_pa(model.ft)}  fc={_fmt_pa(model.fc)}  Gf={model.Gf:.3g} N/m  lch={model.lch:.4g} m")
        print(f"[material] phi={model.cdp_phi_deg:.3g} deg  H={_fmt_pa(model.cdp_H)}")
        _print_numba()
        return

    print(f"[material] (unknown bulk_material='{model.bulk_material}')")
    _print_numba()
