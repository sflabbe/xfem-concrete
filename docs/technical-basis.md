# Technical Basis

This table is a traceability index, not a reproduction of protected sources.

| Rule | Implementation | Source basis | Primary test |
|---|---|---|---|
| Q4 plane stress | `xfem/fem/q4.py`, `xfem/xfem/material.py` | Standard isoparametric FEM | `test_smoke.py` |
| Cohesive Mode I/mixed | `cohesive_laws.py` | Gutiérrez thesis cohesive formulation | `test_mixed_mode_cohesive.py` |
| CEB-FIP bond-slip | `bond_slip.py` | CEB-FIP/Model Code law stated in case docs | `test_bond_slip_tangent` |
| Drucker-Prager return map | `constitutive.py:DruckerPrager` | Associative DP, plane-stress condensation | `test_bulk_dp_tangent` |
| Fibre pullout | `fibre_bridging.py` | Banholzer-form adapter | `test_case09_fibre_density_boundary_and_sampling` |
| Crack initiation/growth | `xfem/analysis_single.py`, `xfem/multicrack.py` | Gutiérrez thesis implementation lineage | regression manifest |
| Dowel action | `bond_slip.py:DowelActionModel` | Equations cited in archived status notes | `test_dowel_action_and_masking.py` |

Algorithm equations, tolerances, signs, and convergence rules are frozen for
this refactor. The DP plastic tangent test documents a remaining domain review;
no constitutive correction was made without independent evidence.

At strain `[2.0e-4, -0.5e-4, 0]` with FD perturbation `1e-8`, the current
plane-stress DP tangent has about 0.286 relative error. The elastic regime is
machine-precision consistent and the transition point is about 0.063. The 0.30
plastic characterization threshold is retained; behavior deeper into plasticity
is not certified and requires a constitutive-domain review.
