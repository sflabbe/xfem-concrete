"""Gutierrez-style simply-supported beam (3-point bending) demo.

This is a convenience runner to compare against the simply-supported beam
example from Gutierrez (Table/Figure around pp. 128+ in the provided thesis).

Usage:
  python -m run_gutierrez_beam
  python -m run_gutierrez_beam --umax-mm 10 --nsteps 30 --nx 120 --ny 20

Notes:
- Bulk constitutive model is selectable: elastic | dp | cdp (see --bulk-material).
- The goal is numerical / qualitative agreement (P-u, crack initiation) first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running examples without installing the package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import os
import math

import numpy as np

from xfem_clean.xfem_xfem import XFEMModel, run_analysis_xfem, run_analysis_xfem_multicrack
from xfem_clean.cohesive_laws import CohesiveLaw
from xfem_clean.xfem.cdp_calibration import apply_cdp_generator_calibration
from xfem_clean.utils.run_info import print_run_header, print_material_summary
from xfem_clean.xfem.post import nodal_average_state_fields



def _mm(x: float) -> float:
    return x * 1e-3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--umax-mm", type=float, default=10.0, help="Max imposed midspan disp [mm] (default: 10)")
    ap.add_argument("--nsteps", type=int, default=30, help="Number of global load steps (default: 30)")
    ap.add_argument("--nx", type=int, default=120, help="Elements along span (default: 120)")
    ap.add_argument("--ny", type=int, default=20, help="Elements along height (default: 20)")
    ap.add_argument("--kn-factor", type=float, default=0.1, help="Cohesive penalty factor Kn_factor (default: 0.1)")
    ap.add_argument("--cohesive-law", type=str, default="bilinear",
                choices=["bilinear", "reinhardt"],
                help="Cohesive traction-separation law")
    ap.add_argument("--reinhardt-c1", type=float, default=3.0, help="Reinhardt parameter c1 (default 3.0)")
    ap.add_argument("--reinhardt-c2", type=float, default=6.93, help="Reinhardt parameter c2 (default 6.93)")
    ap.add_argument("--reinhardt-wcrit-mm", type=float, default=0.0,
                help="Critical crack opening w_c [mm]. If <=0, computed from Gf/ft and (c1,c2).")
    ap.add_argument("--kcap-factor", type=float, default=1.0,
                help="Cap factor for secant stiffness (k_sec <= kcap_factor*Kn).")
    ap.add_argument("--newton-beta", type=float, default=1e-3,
                help="Relative Newton tolerance beta (Gutierrez Eq. 4.59), default 1e-3")
    ap.add_argument("--newton-tol-r", type=float, default=1e-6,
                help="Absolute Newton residual tolerance (force units), default 1e-6")
    ap.add_argument("--newton-tol-du", type=float, default=1e-8,
                help="Absolute Newton stagnation threshold on ||du||, default 1e-8")
    ap.add_argument("--visc", type=float, default=1e-4, help="Viscous regularization (default: 1e-4)")
    ap.add_argument("--newton-maxit", type=int, default=60, help="Newton max iterations (default: 60)")
    ap.add_argument("--line-search", action="store_true", help="Enable backtracking line-search")
    ap.add_argument("--max-cracks", type=int, default=1, help="Maximum number of cracks (>=1)")
    ap.add_argument("--crack-mode", choices=["single","option1","option2"], default="single", help="Crack initiation mode for multi-crack runs")
    ap.add_argument("--diag-start-mm", type=float, default=2.0, help="(option2) allow diagonal/shear cracks from this displacement")
    # Bulk material model
    ap.add_argument("--bulk-material", choices=["elastic", "dp", "cdp", "cdp-lite"], default="elastic",
                help="Bulk constitutive model: elastic | dp | cdp (default: elastic)")
    ap.add_argument("--dp-phi-deg", type=float, default=30.0, help="Drucker–Prager friction angle phi [deg]")
    ap.add_argument("--dp-cohesion-mpa", type=float, default=2.0, help="Drucker–Prager cohesion [MPa]")
    ap.add_argument("--dp-H-mpa", type=float, default=0.0, help="Drucker–Prager isotropic hardening modulus H [MPa]")
    ap.add_argument("--fc-mpa", type=float, default=32.0, help="Concrete compressive strength |fc| [MPa] (CDP)")
    ap.add_argument("--cdp-phi-deg", type=float, default=30.0, help="CDP-lite friction angle phi [deg]")
    ap.add_argument("--cdp-H-mpa", type=float, default=0.0, help="CDP-lite isotropic hardening modulus H [MPa]")
    ap.add_argument("--cdp-class", type=str, default="", help="Concrete class for cdp_generator calibration (e.g. C20/25). If set, overrides --fc-mpa for calibration.")
    ap.add_argument("--cdp-ec1", type=float, default=0.0022, help="Strain at peak compressive stress ec1 for cdp_generator (default 0.0022)")
    ap.add_argument("--cdp-eclim", type=float, default=0.0035, help="Ultimate compressive strain eclim for cdp_generator (default 0.0035)")
    ap.add_argument("--cdp-strain-rate", type=float, default=0.0, help="Strain rate [1/s] for cdp_generator (default 0.0)")
    ap.add_argument("--no-cdp-generator", action="store_true", help="Disable cdp_generator calibration (CDP bulk only)")
    ap.add_argument("--no-gen-override-E", action="store_true", help="Do not override E with generator value")
    ap.add_argument("--no-gen-override-ft", action="store_true", help="Do not override ft with generator value")
    ap.add_argument("--no-gen-override-Gf", action="store_true", help="Do not override Gf with generator value")
    ap.add_argument("--no-gen-use-dilation-angle", action="store_true", help="Do not set cdp_phi_deg to generator dilation angle")

    # Optional Numba kernels
    ap.add_argument("--use-numba", action="store_true", help="Enable Numba kernels (if installed)")


    args = ap.parse_args()

    print_run_header("XFEM Gutierrez beam")


    # Geometry (Gutierrez example): L=4.0m, H=0.4m, thickness b=0.2m
    L = 4.0
    H = 0.4
    b = 0.2

    # Concrete (Gutierrez table): E=28 GPa, nu=0.2, fc≈32 MPa, ft≈2.5 MPa, Gf≈0.1 N/mm
    E = 28.0e9
    nu = 0.20
    ft = 2.5e6
    Gf = 0.1 * 1000.0  # N/mm -> N/m

    # Rebar: 4Ø12 at bottom. (This runner uses a single bottom rebar layer.)
    steel_d = 12e-3
    steel_n = 4
    steel_A_total = steel_n * (math.pi * steel_d**2 / 4.0)

    # Reinforcement material (Gutierrez): E=200 GPa, fy=587 MPa, fu=672 MPa, hardening modulus ~829 MPa
    steel_E = 200.0e9
    steel_fy = 587.0e6
    steel_fu = 672.0e6
    steel_Eh = 829.0e6

    model = XFEMModel(
        L=L,
        H=H,
        b=b,
        E=E,
        nu=nu,
        ft=ft,
        Gf=Gf,
        cover=0.035,  # ~35 mm cover
        steel_A_total=steel_A_total,
        steel_E=steel_E,
        steel_fy=steel_fy,
        steel_fu=steel_fu,
        steel_Eh=steel_Eh,
        bulk_material=str(args.bulk_material),
        dp_phi_deg=float(args.dp_phi_deg),
        dp_cohesion=float(args.dp_cohesion_mpa) * 1e6,
        dp_H=float(args.dp_H_mpa) * 1e6,
        fc=float(args.fc_mpa) * 1e6,
        cdp_phi_deg=float(args.cdp_phi_deg),
        cdp_H=float(args.cdp_H_mpa) * 1e6,
        cdp_use_generator=not bool(args.no_cdp_generator),
        cdp_class=str(args.cdp_class).strip() or None,
        cdp_ec1=float(args.cdp_ec1),
        cdp_eclim=float(args.cdp_eclim),
        cdp_strain_rate=float(args.cdp_strain_rate),
        cdp_use_dilation_angle=not bool(args.no_gen_use_dilation_angle),
        cdp_override_E=not bool(args.no_gen_override_E),
        cdp_override_ft=not bool(args.no_gen_override_ft),
        cdp_override_Gf=not bool(args.no_gen_override_Gf),
        Kn_factor=float(args.kn_factor),
        visc_damp=float(args.visc),
        newton_maxit=int(args.newton_maxit),
        line_search=bool(args.line_search),
    )

    # Enable optional Numba kernels
    model.use_numba = bool(args.use_numba)
    model.newton_beta = float(args.newton_beta)
    model.newton_tol_r = float(args.newton_tol_r)
    model.newton_tol_du = float(args.newton_tol_du)

    # Gutierrez cracks are not artificially arrested at half height.
    # (Solver default keeps an arrest for validation; disable it here.)
    if hasattr(model, "arrest_at_half_height"):
        model.arrest_at_half_height = False

    # Mesh-dependent characteristic length and cohesive diagnostics
    dx = L / args.nx
    dy = H / args.ny
    model.lch = float(math.sqrt(dx * dy))

    # Apply load over a small strip at midspan to avoid a pure point constraint.
    # Use ~2 elements half-width by default.
    model.load_halfwidth = 2.0 * dx

    # Apply cdp_generator calibration now that lch is set (needed for the uniaxial softening curves).
    if model.bulk_material == "cdp" and model.cdp_use_generator:
        apply_cdp_generator_calibration(model)

    print_material_summary(model)

    Kn = model.Kn_factor * model.E / max(1e-12, model.lch)
    law = CohesiveLaw(Kn=Kn, ft=model.ft, Gf=model.Gf,
                 law=str(args.cohesive_law),
                 c1=float(args.reinhardt_c1), c2=float(args.reinhardt_c2),
                 wcrit=float(args.reinhardt_wcrit_mm) * 1e-3,
                 kcap_factor=float(args.kcap_factor))
    print(f"[gutierrez] L={L:.3f}m H={H:.3f}m b={b:.3f}m  nx={args.nx} ny={args.ny}")
    print(f"[steel] As_prov={steel_A_total*1e6:.1f} mm² ({steel_n}Ø{int(steel_d*1e3)})")
    # CohesiveLaw uses attribute name `deltaf` for the final opening.
    print(f"[cohesive] law={law.law}  lch={model.lch:.4f} m  Kn={Kn:.3e} Pa/m  delta0={law.delta0*1e6:.2f} µm  delta_f={law.deltaf*1e3:.3f} mm")

    umax = _mm(args.umax_mm)

    mp_states = None

    if args.max_cracks <= 1 or args.crack_mode == "single":
        nodes, elems, u, results, crack, mp_states = run_analysis_xfem(
            model,
            nx=int(args.nx),
            ny=int(args.ny),
            nsteps=int(args.nsteps),
            umax=umax,
            law=law,
            return_states=True,
        )
        cracks = [crack] if getattr(crack, "active", False) else []
    else:
        nodes, elems, u, results, cracks = run_analysis_xfem_multicrack(
            model,
            nx=int(args.nx),
            ny=int(args.ny),
            nsteps=int(args.nsteps),
            umax=umax,
            max_cracks=int(args.max_cracks),
            crack_mode=str(args.crack_mode),
            u_diag_mm=float(args.diag_start_mm),
            law=law,
        )

    # Save P-u curve to csv for quick comparison
    try:
        import csv

        with open("outputs_pu_gutierrez.csv", "w", newline="") as f:
            import numpy as _np
            if isinstance(results, _np.ndarray):
                # single-crack solver returns a numeric table
                base = ["step","u_m","P_N","M_Nm","kappa","R_m","tip_x_m","tip_y_m","angle_deg","active"]
                if results.ndim == 2 and results.shape[1] == 15:
                    fieldnames = base + [
                        "W_plastic_J",
                        "W_damage_t_J",
                        "W_damage_c_J",
                        "W_cohesive_J",
                        "W_diss_total_J",
                    ]
                elif results.ndim == 2 and results.shape[1] == len(base):
                    fieldnames = base
                else:
                    fieldnames = [f"col{i}" for i in range(int(results.shape[1]))]
                w = csv.writer(f)
                w.writerow(fieldnames)
                for row in results:
                    w.writerow(list(row))
            elif len(results) > 0 and isinstance(results[0], dict):
                fieldnames = list(results[0].keys())
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for row in results:
                    w.writerow(row)
            else:
                w = csv.writer(f)
                w.writerow(["step", "u_mm", "P_kN", "M_kNm", "kappa", "tip_x_m", "tip_y_m", "angle_deg", "crack_active"])
                for row in results:
                    step = int(row[0])
                    uu = float(row[1])
                    PP = float(row[2])
                    MM = float(row[3])
                    kappa = float(row[4])
                    tipx = float(row[6])
                    tipy = float(row[7])
                    ang = float(row[8])
                    active = int(round(float(row[9])))
                    w.writerow([step, uu * 1e3, PP / 1e3, MM / 1e3, kappa, tipx, tipy, ang, active])
        print("[output] wrote outputs_pu_gutierrez.csv")
    except Exception as e:
        print(f"[output] failed to write outputs_pu_gutierrez.csv: {e}")

    # Export nodal-averaged state fields (damage dt/dc etc.) for the last converged step
    if mp_states is not None and args.max_cracks <= 1:
        try:
            import csv

            flds = nodal_average_state_fields(nodes, elems, u, model, crack, mp_states=mp_states)
            with open("outputs_fields_last.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "node",
                        "x_m",
                        "y_m",
                        "sigma1_Pa",
                        "mises_Pa",
                        "tresca_Pa",
                        "damage_t",
                        "damage_c",
                        "kappa",
                    ]
                )
                for i, (x, y) in enumerate(nodes):
                    w.writerow(
                        [
                            i,
                            float(x),
                            float(y),
                            float(flds["sigma1"][i]),
                            float(flds["mises"][i]),
                            float(flds["tresca"][i]),
                            float(flds["damage_t"][i]),
                            float(flds["damage_c"][i]),
                            float(flds["kappa"][i]),
                        ]
                    )
            print("[output] wrote outputs_fields_last.csv")
        except Exception as e:
            print(f"[output] failed to write outputs_fields_last.csv: {e}")

    # Save crack summary
    if cracks:
        try:
            import csv
            with open("outputs_cracks_gutierrez.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "x0_m", "y0_m", "tip_x_m", "tip_y_m", "angle_deg", "active"])
                for c in cracks:
                    w.writerow([getattr(c, "cid", ""), c.x0, c.y0, c.tip_x, c.tip_y, c.angle_deg, int(getattr(c, "active", True))])
            print("[output] wrote outputs_cracks_gutierrez.csv")
        except Exception as e:
            print(f"[output] failed to write outputs_cracks_gutierrez.csv: {e}")


    # Plots (Gutierrez-style)
    try:
        from gutierrez_plots import plot_pu_gutierrez, plot_crack_pattern_gutierrez
        outdir = "outputs_gutierrez"
        os.makedirs(outdir, exist_ok=True)
        plot_pu_gutierrez(results, os.path.join(outdir, "pu_curve.png"), label="Model")
        plot_crack_pattern_gutierrez(nodes, u, cracks, model, os.path.join(outdir, "crack_pattern.png"))
        print("[output] wrote outputs_gutierrez/pu_curve.png and crack_pattern.png")
    except Exception as e:
        print(f"[output] plotting skipped: {e}")


if __name__ == "__main__":
    main()
