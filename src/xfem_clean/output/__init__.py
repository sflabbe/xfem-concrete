"""Output and post-processing utilities for XFEM analysis."""

from xfem_clean.output.energy import compute_global_energies, EnergyBalance
from xfem_clean.output.vtk_export import export_damage_field, export_full_state

__all__ = [
    "compute_global_energies",
    "EnergyBalance",
    "export_damage_field",
    "export_full_state",
]
