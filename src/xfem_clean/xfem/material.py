"""Material helpers for the XFEM prototypes.

Currently this module only re-exports linear-elastic plane-stress stiffness.
It remains inside the `xfem_clean.xfem` subpackage for historical reasons,
but the implementation lives at top-level (`xfem_clean.linear_elastic`) to
avoid circular imports with constitutive models.
"""

from __future__ import annotations

from xfem_clean.linear_elastic import plane_stress_C

__all__ = ["plane_stress_C"]
