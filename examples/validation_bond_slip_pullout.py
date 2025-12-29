"""Pullout test for bond-slip validation (Phase 2).

Test setup:
- Simple 2D concrete block with embedded horizontal rebar
- Displacement-controlled pullout at free end
- Measures Load-Slip curve and bond stress distribution

Expected behavior:
- Load increases with slip up to peak bond stress
- Softening after peak (bond degradation)
- Slip profile: maximum at loaded end, zero at far end

Validation criteria:
- Peak load should match analytical estimate: P_max ~ τ_max * π * d * L_emb
- Slip should be non-uniform along embedment length
- Energy dissipation should be positive and monotonic
"""

from __future__ import annotations

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xfem_clean.xfem.model import XFEMModel
from xfem_clean.xfem.analysis_single import run_analysis_xfem
from xfem_clean.cohesive_laws import CohesiveLaw


def main():
    """Run pullout test with bond-slip enabled."""

    # Geometry (simple block)
    L = 0.40  # [m] length (embedment length)
    H = 0.10  # [m] height (cover depth)

    # Material properties (concrete C30/37)
    E = 33e9  # [Pa]
    nu = 0.2
    fc = 38e6  # [Pa] mean compressive strength
    ft = 2.9e6  # [Pa] tensile strength
    Gf = 120  # [N/m] fracture energy

    # Steel properties
    d_bar = 0.016  # [m] 16mm diameter
    A_bar = np.pi * (d_bar / 2) ** 2  # [m^2]
    E_steel = 200e9  # [Pa]
    fy_steel = 500e6  # [Pa]
    fu_steel = 600e6  # [Pa]
    Eh_steel = 0.01 * E_steel  # [Pa] hardening modulus

    # Bond-slip parameters (Model Code 2010)
    bond_condition = "good"  # "good" | "poor"

    # Mesh
    nx = 40  # Elements along length
    ny = 10  # Elements along height

    # Loading
    nsteps = 50  # Load steps
    u_max = 0.003  # [m] = 3mm pullout displacement

    # Model setup
    model = XFEMModel(
        L=L,
        H=H,
        b=0.15,  # [m] thickness (out-of-plane)
        E=E,
        nu=nu,
        ft=ft,
        Gf=Gf,
        fc=fc,
        # Steel properties
        steel_A_total=A_bar,
        steel_E=E_steel,
        steel_fy=fy_steel,
        steel_fu=fu_steel,
        steel_Eh=Eh_steel,
        # Bond-slip (Phase 2)
        enable_bond_slip=True,
        rebar_diameter=d_bar,
        bond_condition=bond_condition,
        # Rebar location
        cover=0.05,  # [m] 50mm cover
        # Solver
        newton_maxit=30,
        newton_tol_r=1e-5,
        newton_tol_rel=1e-8,
        newton_tol_du=1e-9,
        line_search=True,
        # XFEM crack controls (disable cracking for pure pullout test)
        crack_margin=1e6,  # Effectively disable crack initiation
        arrest_at_half_height=False,
        crack_tip_stop_y=H * 2,  # Allow crack to propagate past specimen
        # Enable Numba (required for bond-slip assembly)
        use_numba=True,
    )

    # Cohesive law (not used in pullout test, but required by solver)
    Kn = 0.1 * E / 0.05  # Dummy cohesive stiffness
    law = CohesiveLaw(Kn=Kn, ft=ft, Gf=Gf)

    print("=" * 70)
    print("BOND-SLIP PULLOUT TEST (Phase 2 Validation)")
    print("=" * 70)
    print(f"Geometry:      L = {L:.3f} m, H = {H:.3f} m")
    print(f"Concrete:      fc = {fc/1e6:.1f} MPa, ft = {ft/1e6:.2f} MPa")
    print(f"Steel:         d = {d_bar*1e3:.0f} mm, A = {A_bar*1e6:.2f} mm²")
    print(f"Bond:          condition = '{bond_condition}'")
    print(f"Mesh:          nx = {nx}, ny = {ny}")
    print(f"Loading:       u_max = {u_max*1e3:.2f} mm, nsteps = {nsteps}")
    print(f"Bond-slip:     {'ENABLED' if model.enable_bond_slip else 'DISABLED'}")
    print("=" * 70)

    try:
        # Run analysis
        results = run_analysis_xfem(
            model=model,
            nx=nx,
            ny=ny,
            nsteps=nsteps,
            umax=u_max,
            law=law,
            return_states=False,
        )

        print(f"\n✓ Analysis completed: {len(results)} load steps")

        # Extract results
        load_steps = [r["step"] for r in results]
        displacements = [r["u_applied"] for r in results]
        reactions = [r["reaction"] for r in results]

        # Plot Load-Slip curve
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            np.array(displacements) * 1e3,  # mm
            np.array(reactions) / 1e3,  # kN
            "b-o",
            linewidth=2,
            markersize=4,
            label="Pullout test (bond-slip)"
        )
        ax.set_xlabel("Slip [mm]", fontsize=12)
        ax.set_ylabel("Pullout load [kN]", fontsize=12)
        ax.set_title("Bond-Slip Pullout Test (Model Code 2010)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        # Analytical estimate for peak load
        # τ_max from MC2010: τ_max = 2.5 * sqrt(f_cm) [MPa]
        sqrt_fc = np.sqrt(fc / 1e6)  # MPa
        tau_max = 2.5 * sqrt_fc * 1e6  # Pa
        P_analytical = tau_max * np.pi * d_bar * L  # N
        ax.axhline(
            P_analytical / 1e3,
            color="r",
            linestyle="--",
            linewidth=1.5,
            label=f"Analytical peak: {P_analytical/1e3:.2f} kN"
        )
        ax.legend(fontsize=10)

        plt.tight_layout()

        # Save plot
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        plot_path = output_dir / "bond_slip_pullout_load_slip.png"
        plt.savefig(plot_path, dpi=150)
        print(f"\n✓ Plot saved: {plot_path}")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Peak load (numerical):   {max(reactions)/1e3:.2f} kN")
        print(f"Peak load (analytical):  {P_analytical/1e3:.2f} kN")
        print(f"Final slip:              {displacements[-1]*1e3:.2f} mm")
        print(f"τ_max (MC2010):          {tau_max/1e6:.2f} MPa")
        print("=" * 70)

        plt.show()

    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
