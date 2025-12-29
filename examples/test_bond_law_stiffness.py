"""Check bond-slip law tangent stiffness at s=0."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xfem_clean.bond_slip import BondSlipModelCode2010

# Bond law
f_cm = 30e6  # Pa
d_bar = 0.012  # m
bond_law = BondSlipModelCode2010(f_cm=f_cm, d_bar=d_bar, condition="good")

print("Bond-slip law parameters:")
print(f"  tau_max = {bond_law.tau_max/1e6:.2f} MPa")
print(f"  s1 = {bond_law.s1*1e3:.4f} mm")
print(f"  s2 = {bond_law.s2*1e3:.4f} mm")
print(f"  s3 = {bond_law.s3*1e3:.4f} mm")
print(f"  tau_f = {bond_law.tau_f/1e6:.2f} MPa")
print(f"  alpha = {bond_law.alpha}")

# Evaluate at s≈0
s_values = [1e-16, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3]

# Evaluate tangent stiffness manually WITH REGULARIZATION
tau_max = bond_law.tau_max
s1 = bond_law.s1
s2 = bond_law.s2
s3 = bond_law.s3
tau_f = bond_law.tau_f
alpha = bond_law.alpha

# Regularization parameters
s_reg = 0.01 * s1
tau_reg = tau_max * (s_reg / s1) ** alpha
K_reg = tau_reg / s_reg

print(f"\nRegularization:")
print(f"  s_reg = {s_reg*1e6:.2f} microns")
print(f"  tau_reg = {tau_reg/1e6:.3f} MPa")
print(f"  K_reg = {K_reg:.3e} Pa/m")

print("\nTangent stiffness dtau/ds (with regularization):")
for s in s_values:
    # Envelope for loading (s = s_max)
    if s <= s1:
        if s < s_reg:
            # Regularized linear branch
            dtau_ds = K_reg
        else:
            ratio = s / s1
            dtau_ds = tau_max * alpha / s1 * (ratio ** (alpha - 1.0))
    elif s <= s2:
        dtau_ds = 0.0
    elif s <= s3:
        dtau_ds = -(tau_max - tau_f) / (s3 - s2)
    else:
        dtau_ds = 0.0
    print(f"  s = {s:.2e} m → dtau/ds = {dtau_ds:.3e} Pa/m")

# Check what this means for bond stiffness
perimeter = np.pi * d_bar
L_seg = 0.02  # typical segment length

print(f"\nBond stiffness = dtau/ds * perimeter * L:")
print(f"  perimeter = {perimeter:.4f} m")
print(f"  L_seg = {L_seg:.4f} m")

for s in s_values:
    if s <= s1:
        if s < s_reg:
            dtau_ds = K_reg
        else:
            ratio = s / s1
            dtau_ds = tau_max * alpha / s1 * (ratio ** (alpha - 1.0))
    elif s <= s2:
        dtau_ds = 0.0
    elif s <= s3:
        dtau_ds = -(tau_max - tau_f) / (s3 - s2)
    else:
        dtau_ds = 0.0
    K_bond_seg = dtau_ds * perimeter * L_seg
    print(f"  s = {s:.2e} m → K_bond = {K_bond_seg:.3e} N/m")

# Compare to steel stiffness
steel_E = 200e9
A_bar = np.pi * (d_bar/2)**2
steel_EA = steel_E * A_bar
K_steel_seg = steel_EA / L_seg

print(f"\nSteel axial stiffness:")
print(f"  K_steel = EA/L = {K_steel_seg:.3e} N/m")

print(f"\nRatio K_bond / K_steel (with regularization):")
for s in s_values:
    if s <= s1:
        if s < s_reg:
            dtau_ds = K_reg
        else:
            ratio = s / s1
            dtau_ds = tau_max * alpha / s1 * (ratio ** (alpha - 1.0))
    elif s <= s2:
        dtau_ds = 0.0
    elif s <= s3:
        dtau_ds = -(tau_max - tau_f) / (s3 - s2)
    else:
        dtau_ds = 0.0
    K_bond_seg = dtau_ds * perimeter * L_seg
    ratio = K_bond_seg / K_steel_seg
    print(f"  s = {s:.2e} m → ratio = {ratio:.3e}")
