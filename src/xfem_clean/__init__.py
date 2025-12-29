"""xfem_clean package (ordered stable prototype)."""

from .cohesive_laws import CohesiveLaw, CohesiveState, cohesive_update
from .convergence import NewtonConvergence
from .crack_criteria import principal_max_2d, principal_max_dir
from .material_point import MaterialPoint
from .constitutive import (
    LinearElasticPlaneStress,
    DruckerPrager,
    ConcreteCDP,
    DruckerPragerPlaceholder,
    CDPPlaceholder,
)
from .xfem import XFEMModel, XFEMCrack, XFEMDofs, run_analysis_xfem, run_analysis_xfem_multicrack

__all__ = [
    "CohesiveLaw", "CohesiveState", "cohesive_update",
    "NewtonConvergence",
    "principal_max_2d", "principal_max_dir",
    "MaterialPoint",
    "LinearElasticPlaneStress", "DruckerPrager", "ConcreteCDP",
    "DruckerPragerPlaceholder", "CDPPlaceholder",
    "XFEMModel", "XFEMCrack", "XFEMDofs",
    "run_analysis_xfem", "run_analysis_xfem_multicrack",
]
