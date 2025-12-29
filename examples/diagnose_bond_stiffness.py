"""Diagnose bond stiffness values during pullout."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xfem_clean.fem.mesh import structured_quad_mesh
from xfem_clean.rebar import prepare_rebar_segments
from xfem_clean.xfem.dofs_single import build_xfem_dofs
from xfem_clean.xfem.geometry import XFEMCrack
from xfem_clean.bond_slip import BondSlipModelCode2010

print("="*70)
print("BOND STIFFNESS DIAGNOSIS")
print("="*70)

# Geometry
L, H = 0.10, 0.05
nx, ny = 5, 2
nodes, elems = structured_quad_mesh(L, H, nx, ny)
rebar_segs = prepare_rebar_segments(nodes, cover=0.025)

# Bond law
d_bar = 0.012
fc = 30e6
bond_law = BondSlipModelCode2010(f_cm=fc, d_bar=d_bar, condition="good")

print(f"\nBond law parameters:")
print(f"  tau_max = {bond_law.tau_max/1e6:.2f} MPa")
print(f"  s1 = {bond_law.s1*1e6:.1f} μm")
print(f"  s_reg = {0.1*bond_law.s1*1e6:.1f} μm")
print(f"  alpha = {bond_law.alpha}")

# Typical segment length
L_seg = rebar_segs[0, 2]  # L0 from first segment
perimeter = np.pi * d_bar

print(f"\nSegment geometry:")
print(f"  Segment length L = {L_seg*1e3:.1f} mm")
print(f"  Bar perimeter = {perimeter*1e3:.1f} mm")
print(f"  Contact area = L*P = {L_seg*perimeter*1e6:.2f} mm²")

# Test slips from very small to above s_reg
slips_um = [0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]

print(f"\n{'s [μm]':<10} {'τ [MPa]':<10} {'dτ/ds [GPa/m]':<15} {'k_seg [N/m]':<15} {'Note':<20}")
print("-"*70)

for s_um in slips_um:
    s = s_um * 1e-6
    tau, dtau_ds = bond_law.tau_envelope(s)

    # Bond stiffness contribution from this segment
    k_seg = dtau_ds * perimeter * L_seg

    note = ""
    if s_um < 0.1 * bond_law.s1 * 1e6:
        note = "< s_reg (linear)"
    elif s_um < bond_law.s1 * 1e6:
        note = "ascending branch"
    elif s_um < bond_law.s2 * 1e6:
        note = "plateau"
    else:
        note = "descending"

    print(f"{s_um:<10.2f} {tau/1e6:<10.3f} {dtau_ds/1e9:<15.3e} {k_seg:<15.3e} {note:<20}")

# Compare with typical FEM stiffness
E = 30e9
t = 0.10
K_concrete_typical = E * t / L_seg

print(f"\nComparison with concrete stiffness:")
print(f"  K_concrete ~ E*t/L_seg = {K_concrete_typical:.2e} N/m")

# At s_reg
s_reg = 0.1 * bond_law.s1
tau_reg, dtau_reg = bond_law.tau_envelope(s_reg)
k_reg = dtau_reg * perimeter * L_seg

print(f"\nAt regularization threshold s_reg = {s_reg*1e6:.1f} μm:")
print(f"  k_bond = {k_reg:.2e} N/m")
print(f"  Ratio k_bond / k_concrete = {k_reg / K_concrete_typical:.2e}")

# Check what happens with 5 segments in series
n_seg = 5
k_total = k_reg / n_seg  # Springs in series: 1/k_total = sum(1/k_i)
print(f"\nWith {n_seg} segments in series:")
print(f"  k_total = k_seg / {n_seg} = {k_total:.2e} N/m")
print(f"  Ratio k_total / k_concrete = {k_total / K_concrete_typical:.2e}")

# Check steel stiffness
steel_EA_min = 1e6
k_steel = steel_EA_min / L_seg
print(f"\nSteel stiffness (with EA_min = {steel_EA_min:.0e}):")
print(f"  k_steel = EA_min / L_seg = {k_steel:.2e} N/m")
print(f"  Ratio k_steel / k_concrete = {k_steel / K_concrete_typical:.2e}")

print("="*70)
