"""Quick pullout test with reduced parameters."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xfem_clean.xfem.model import XFEMModel
from xfem_clean.xfem.analysis_single import run_analysis_xfem
from xfem_clean.cohesive_laws import CohesiveLaw

print("="*70)
print("QUICK BOND-SLIP PULLOUT TEST")
print("="*70)

# Geometry (smaller/simpler than full test)
L = 0.20  # [m] 200mm embedment
H = 0.05  # [m] 50mm height

# Material properties
E = 33e9  # [Pa]
nu = 0.2
fc = 38e6  # [Pa]
ft = 2.9e6  # [Pa]
Gf = 120  # [N/m]

# Steel
d_bar = 0.016  # [m] 16mm
A_bar = np.pi * (d_bar / 2) ** 2
E_steel = 200e9  # [Pa]
fy_steel = 500e6  # [Pa]
fu_steel = 600e6  # [Pa]
Eh_steel = 0.01 * E_steel

# Mesh (coarser than full test)
nx = 10  # Reduced from 40
ny = 5   # Reduced from 10

# Loading (reduced)
nsteps = 5  # Reduced from 50
u_max = 0.0005  # [m] = 0.5mm (reduced from 3mm)

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
    steel_EA_min=1e3,  # Minimum stiffness
    cover=0.025,
    newton_maxit=30,
    newton_tol_r=1e-5,
    newton_tol_rel=1e-8,
    newton_tol_du=1e-9,
    line_search=True,
    enable_diagonal_scaling=True,  # Enable our fix!
    crack_margin=1e6,  # Disable cracking
    use_numba=True,
    debug_newton=True,  # Show Newton iterations
)

law = CohesiveLaw(Kn=0.1 * E / 0.05, ft=ft, Gf=Gf)

print(f"Geometry:  L={L*1e3:.0f}mm, H={H*1e3:.0f}mm")
print(f"Mesh:      nx={nx}, ny={ny}")
print(f"Loading:   u_max={u_max*1e3:.2f}mm, nsteps={nsteps}")
print(f"Diagonal scaling: {'ENABLED' if model.enable_diagonal_scaling else 'DISABLED'}")
print("="*70)

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

    print(f"\n✓ Analysis completed: {len(results)} load steps")

    # Extract results
    displacements = [r["u_applied"] for r in results]
    reactions = [r["reaction"] for r in results]

    print(f"\nResults:")
    for i, (u, P) in enumerate(zip(displacements, reactions)):
        print(f"  Step {i+1}: u={u*1e6:.2f} μm, P={P/1e3:.3f} kN")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("="*70)
