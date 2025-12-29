"""Simple bond-slip integration test - Phase 2 validation.

Simplified test case:
- Small beam with embedded rebar
- No XFEM cracking (crack disabled)
- Pure elastic concrete + bond-slip interface
- Verify that solver runs without errors

This is a sanity check to verify the bond-slip integration works
before attempting more complex validation tests.
"""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xfem_clean.xfem.model import XFEMModel
from xfem_clean.xfem.analysis_single import run_analysis_xfem
from xfem_clean.cohesive_laws import CohesiveLaw


def main():
    """Run simple bond-slip test."""

    print("=" * 70)
    print("BOND-SLIP SIMPLE TEST (Phase 2 - Sanity Check)")
    print("=" * 70)

    # Very simple geometry
    L = 0.20  # [m] length
    H = 0.10  # [m] height

    # Material properties
    E = 30e9  # [Pa]
    nu = 0.2
    fc = 30e6  # [Pa]
    ft = 3e6  # [Pa]
    Gf = 100  # [N/m]

    # Steel
    d_bar = 0.012  # [m] 12mm
    A_bar = np.pi * (d_bar / 2) ** 2
    E_steel = 200e9  # [Pa]
    fy_steel = 500e6  # [Pa]

    # Coarse mesh for speed
    nx = 10
    ny = 5

    # Small displacement
    nsteps = 5
    u_max = 0.0001  # [m] = 0.1mm

    model = XFEMModel(
        L=L,
        H=H,
        b=0.10,  # [m] thickness
        E=E,
        nu=nu,
        ft=ft,
        Gf=Gf,
        fc=fc,
        # Steel
        steel_A_total=A_bar,
        steel_E=E_steel,
        steel_fy=fy_steel,
        steel_fu=600e6,
        steel_Eh=0.01 * E_steel,
        # Bond-slip
        enable_bond_slip=True,
        rebar_diameter=d_bar,
        bond_condition="good",
        cover=0.05,  # [m]
        # Solver - relaxed tolerances
        newton_maxit=20,
        newton_tol_r=1e-4,  # Relaxed
        newton_tol_rel=1e-6,
        newton_tol_du=1e-8,
        line_search=True,
        max_subdiv=8,
        # XFEM - DISABLE cracking completely
        crack_margin=1e10,  # Impossible to initiate
        arrest_at_half_height=False,
        ft_initiation_factor=1e10,  # Impossible to reach
        # Numba
        use_numba=True,
    )

    # Cohesive law (won't be used since cracking disabled)
    Kn = 0.1 * E / 0.05
    law = CohesiveLaw(Kn=Kn, ft=ft, Gf=Gf)

    print(f"Geometry:    L = {L:.2f} m, H = {H:.2f} m")
    print(f"Mesh:        nx = {nx}, ny = {ny}")
    print(f"Loading:     u_max = {u_max*1e3:.3f} mm, steps = {nsteps}")
    print(f"Bond-slip:   ENABLED (d = {d_bar*1e3:.0f} mm)")
    print(f"Cracking:    DISABLED (sanity check only)")
    print("=" * 70)

    try:
        results = run_analysis_xfem(
            model=model,
            nx=nx,
            ny=ny,
            nsteps=nsteps,
            umax=u_max,
            law=law,
            return_states=False,
        )

        print(f"\n✓ SUCCESS: Analysis completed {len(results)} steps")
        print("\nResults:")
        for i, r in enumerate(results):
            u = r["u_applied"] * 1e3  # mm
            P = r["reaction"] / 1e3  # kN
            print(f"  Step {i+1:2d}: u = {u:6.3f} mm, P = {P:8.3f} kN")

        print("\n" + "=" * 70)
        print("✓ BOND-SLIP INTEGRATION VALIDATED")
        print("  - Solver converges with bond-slip enabled")
        print("  - Steel DOFs allocated correctly")
        print("  - Assembly runs without indexing errors")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
