"""Legacy entrypoint kept for backwards compatibility.

Historically the project used a single, monolithic file `xfem_xfem.py` containing
all geometry, DOF bookkeeping, assembly and analysis routines.

The stable refactor moves those building blocks into the `xfem_clean.xfem`
subpackage. This file remains as a thin shim so older scripts keep working.
"""

from __future__ import annotations

from xfem_clean.cohesive_laws import CohesiveLaw, CohesiveState
from xfem_clean.xfem.model import XFEMModel
from xfem_clean.xfem.geometry import XFEMCrack
from xfem_clean.xfem.dofs_single import XFEMDofs
from xfem_clean.xfem.analysis_single import run_analysis_xfem
from xfem_clean.xfem.multicrack import run_analysis_xfem_multicrack
from xfem_clean.xfem.post import nodal_average_stress_fields, nodal_average_state_fields

__all__ = [
    "CohesiveLaw",
    "CohesiveState",
    "XFEMModel",
    "XFEMCrack",
    "XFEMDofs",
    "run_analysis_xfem",
    "run_analysis_xfem_multicrack",
    "nodal_average_stress_fields",
    "nodal_average_state_fields",
]
