# Units And Signs

`CaseConfig` uses mm, MPa, N/mm, and fibres/cm2. The solver kernel uses SI:
m, Pa, N/m, N, J, and fibres/m2. `solver_interface.py` and
`fibre_config_from_case()` are the conversion boundary.

Bond slip fields `s1`, `s2`, and `s3` are unambiguously millimetres in the case
schema and are multiplied by `1e-3`. Fibre density is multiplied by `1e4` once.
Case 09 therefore stores `3.43 fibres/cm2` and reaches `34300 fibres/m2`.

Beam prescribed displacement is negative vertical; wall push is positive
horizontal. Concrete compression sign conventions remain model-specific and
are not normalized by this refactor.

Trapezoidal traction-displacement integrals are signed work. Public names are
`interface_work`, `bond_work`, and corresponding `*_work_inc`. Historical
`D_coh_inc`, `D_bond_inc`, and `D_dowel_inc` keys remain deprecated aliases and
must not be described as irreversible dissipation. Plastic and damage terms
computed from constitutive internal variables remain identified as dissipation.
