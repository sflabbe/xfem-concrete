"""Test bond law regularization at small slips."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xfem_clean.bond_slip import BondSlipModelCode2010

print("="*70)
print("BOND LAW REGULARIZATION TEST")
print("="*70)

# Create bond law (Model Code 2010, good condition)
d_bar = 0.016  # [m] 16mm
f_cm = 30e6   # [Pa] 30 MPa
bond_law = BondSlipModelCode2010(f_cm=f_cm, d_bar=d_bar, condition="good")

print(f"\nBond law parameters:")
print(f"  tau_max = {bond_law.tau_max/1e6:.2f} MPa")
print(f"  s1 = {bond_law.s1*1e3:.3f} mm")
print(f"  s2 = {bond_law.s2*1e3:.3f} mm")
print(f"  s3 = {bond_law.s3*1e3:.3f} mm")
print(f"  alpha = {bond_law.alpha}")

# Test slips from very small to s1
slips_m = np.array([
    1e-9,   # 1 nm
    1e-8,   # 10 nm
    1e-7,   # 100 nm
    1e-6,   # 1 μm
    1e-5,   # 10 μm
    1e-4,   # 100 μm = 0.1 mm
    5e-4,   # 0.5 mm
    1e-3,   # 1 mm = s1
], dtype=float)

print(f"\n{'Slip [mm]':<12} {'τ [MPa]':<12} {'dτ/ds [GPa/m]':<18} {'k_bond*L [N/m]':<18}")
print("-"*70)

# Assume a typical segment length
L_seg = 0.02  # [m] 20mm
perimeter = np.pi * d_bar  # [m]

for s in slips_m:
    tau, dtau_ds = bond_law.tau_envelope(s)

    # Bond stiffness contribution to matrix
    k_bond = dtau_ds * perimeter * L_seg  # [N/m]

    print(f"{s*1e3:<12.6f} {tau/1e6:<12.3f} {dtau_ds/1e9:<18.3e} {k_bond:<18.3e}")

# Check regularization threshold
s_reg = 0.01 * bond_law.s1
print(f"\nRegularization threshold: s_reg = {s_reg*1e3:.4f} mm")

print(f"\nAt s = s_reg:")
tau_reg, dtau_reg = bond_law.tau_envelope(s_reg)
print(f"  τ = {tau_reg/1e6:.3f} MPa")
print(f"  dτ/ds = {dtau_reg/1e9:.3e} GPa/m")
print(f"  k_bond*L = {dtau_reg * perimeter * L_seg:.3e} N/m")

print(f"\nJust above s_reg (s = 1.001 * s_reg):")
tau_above, dtau_above = bond_law.tau_envelope(1.001 * s_reg)
print(f"  τ = {tau_above/1e6:.3f} MPa")
print(f"  dτ/ds = {dtau_above/1e9:.3e} GPa/m")
print(f"  Continuity check: |dτ_above - dτ_reg| / dτ_reg = {abs(dtau_above - dtau_reg)/dtau_reg:.3e}")

# Compare with typical concrete stiffness
E_concrete = 30e9  # Pa
t = 0.10  # m
L_elem = 0.02  # m
K_concrete_typical = E_concrete * t / L_elem  # N/m

print(f"\nComparison with concrete bulk stiffness:")
print(f"  K_concrete ~ E*t/L = {K_concrete_typical:.2e} N/m")
print(f"  K_bond at s_reg   = {dtau_reg * perimeter * L_seg:.2e} N/m")
print(f"  Ratio K_bond / K_concrete = {(dtau_reg * perimeter * L_seg) / K_concrete_typical:.2e}")

print("="*70)
