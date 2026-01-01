"""Simple test for bond-slip fixes (anti-hang debugging)."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import numpy as np
    from xfem_clean.xfem.model import XFEMModel
    from xfem_clean.xfem.analysis_single import run_analysis_xfem
    from xfem_clean.cohesive_laws import CohesiveLaw
except ImportError as e:
    pytest.skip(f"Import error: {e}", allow_module_level=True)


def main():
    """Run simple bond-slip test with anti-hang debugging."""
    # Minimal geometry
    L = 0.20  # [m] shorter beam
    H = 0.10  # [m]

    # Material (concrete)
    E = 33e9  # [Pa]
    nu = 0.2
    fc = 38e6  # [Pa]
    ft = 2.9e6  # [Pa]
    Gf = 120  # [N/m]

    # Steel
    d_bar = 0.016  # [m]
    A_bar = np.pi * (d_bar / 2) ** 2
    E_steel = 200e9
    fy_steel = 500e6
    fu_steel = 600e6
    Eh_steel = 0.01 * E_steel

    # Coarse mesh for quick test
    nx = 10
    ny = 5

    # Minimal loading
    nsteps = 5  # Only 5 steps
    u_max = 0.001  # [m] = 1mm

    # Model with aggressive anti-hang settings
    model = XFEMModel(
        L=L,
        H=H,
        b=0.15,
        E=E,
        nu=nu,
        ft=ft,
        Gf=Gf,
        fc=fc,
        steel_A_total=A_bar,
        steel_E=E_steel,
        steel_fy=fy_steel,
        steel_fu=fu_steel,
        steel_Eh=Eh_steel,
        enable_bond_slip=True,
        rebar_diameter=d_bar,
        bond_condition="good",
        cover=0.05,
        newton_maxit=20,
        newton_tol_r=1e-4,  # Relaxed tolerance
        newton_tol_du=1e-8,
        line_search=True,
        enable_diagonal_scaling=True,
        crack_margin=1e6,  # Disable cracking
        arrest_at_half_height=False,
        crack_tip_stop_y=H * 2,
        use_numba=True,
        max_subdiv=8,  # Reduced subdiv limit
        max_total_substeps=100,  # VERY AGGRESSIVE: abort if > 100 substeps
        debug_substeps=True,  # Enable debug output
        debug_newton=True,  # Enable Newton debug to see where it hangs
    )

    Kn = 0.1 * E / 0.05
    law = CohesiveLaw(Kn=Kn, ft=ft, Gf=Gf)

    print("=" * 70)
    print("SIMPLE BOND-SLIP TEST (Anti-hang debugging)")
    print("=" * 70)
    print(f"Geometry:      L = {L:.2f} m, H = {H:.2f} m")
    print(f"Mesh:          nx = {nx}, ny = {ny}")
    print(f"Loading:       u_max = {u_max*1e3:.2f} mm, nsteps = {nsteps}")
    print(f"Max substeps:  {model.max_total_substeps}")
    print("=" * 70)

    try:
        nodes, elems, q_final, results_array, crack = run_analysis_xfem(
            model=model,
            nx=nx,
            ny=ny,
            nsteps=nsteps,
            umax=u_max,
            law=law,
            return_states=False,
        )

        print("\n" + "=" * 70)
        print(f"✓ SUCCESS: Analysis completed with {results_array.shape[0]} steps")
        print(f"✓ No hanging detected!")
        print("=" * 70)

    except RuntimeError as e:
        if "Anti-hang guardrail" in str(e):
            print("\n" + "=" * 70)
            print("✗ ANTI-HANG GUARDRAIL TRIGGERED")
            print("=" * 70)
            print(f"Error: {e}")
            print("\nThis means the substepping is still excessive.")
            print("Possible causes:")
            print("- Bond-slip rollback not working correctly")
            print("- Newton convergence issues")
            print("- Incompatible states between substeps")
            sys.exit(1)
        else:
            raise

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
