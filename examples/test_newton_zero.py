"""Test if Newton solver handles zero displacement correctly."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xfem_clean.xfem.model import XFEMModel
from xfem_clean.xfem.analysis_single import run_analysis_xfem
from xfem_clean.cohesive_laws import CohesiveLaw

# Very simple test
L = 0.20
H = 0.10
E = 30e9
nu = 0.2
d_bar = 0.012
A_bar = np.pi * (d_bar / 2) ** 2

model = XFEMModel(
    L=L,
    H=H,
    b=0.10,
    E=E,
    nu=nu,
    ft=3e6,
    Gf=100,
    fc=30e6,
    steel_A_total=A_bar,
    steel_E=200e9,
    steel_fy=500e6,
    steel_fu=600e6,
    steel_Eh=2e9,
    enable_bond_slip=False,  # DISABLE to test if problem is bond-slip specific
    rebar_diameter=d_bar,
    bond_condition="good",
    cover=0.05,
    newton_maxit=10,
    newton_tol_r=1e-3,  # Very relaxed
    newton_tol_rel=1e-6,
    newton_tol_du=1e-8,
    line_search=False,  # Disable line search
    max_subdiv=2,  # Minimal substepping
    crack_margin=1e10,  # No cracking
    ft_initiation_factor=1e10,
    use_numba=True,
)

law = CohesiveLaw(Kn=0.1 * E / 0.05, ft=3e6, Gf=100)

print("Testing Newton solver at u=0...")
print(f"Tolerances: tol_r={model.newton_tol_r}, tol_rel={model.newton_tol_rel}")

try:
    results = run_analysis_xfem(
        model=model,
        nx=10,
        ny=5,
        nsteps=1,  # Just one step
        umax=1e-7,  # Tiny displacement
        law=law,
        return_states=False,
    )
    print(f"✓ SUCCESS: Converged at {len(results)} steps")
    for i, r in enumerate(results):
        u = r["u_applied"] * 1e6  # microns
        P = r["reaction"] / 1e3  # kN
        print(f"  Step {i+1}: u = {u:.3f} μm, P = {P:.3f} kN")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
