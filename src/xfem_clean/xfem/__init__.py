"""XFEM subpackage.

This subpackage contains the refactored building blocks of the original
monolithic `xfem_xfem.py` prototype.
"""

from .model import XFEMModel
from .geometry import XFEMCrack
from .dofs_single import XFEMDofs, build_xfem_dofs, transfer_q_between_dofs
from .material import plane_stress_C
from .analysis_single import run_analysis_xfem
from .multicrack import run_analysis_xfem_multicrack
from .post import nodal_average_stress_fields, nodal_average_state_fields
from .state_arrays import BulkStateArrays, BulkStatePatch, CohesiveStateArrays, CohesiveStatePatch

__all__ = [
    "XFEMModel",
    "XFEMCrack",
    "XFEMDofs",
    "build_xfem_dofs",
    "transfer_q_between_dofs",
    "plane_stress_C",
    "run_analysis_xfem",
    "run_analysis_xfem_multicrack",
    "nodal_average_stress_fields",
    "nodal_average_state_fields",
    "BulkStateArrays",
    "BulkStatePatch",
    "CohesiveStateArrays",
    "CohesiveStatePatch",
]
