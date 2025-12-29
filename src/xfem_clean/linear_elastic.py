"""Linear-elastic helpers shared across the package.

Placed at top-level to avoid circular imports between `xfem_clean.constitutive`
and the `xfem_clean.xfem` subpackage.
"""

from __future__ import annotations

import numpy as np


def plane_stress_C(E: float, nu: float) -> np.ndarray:
    """Plane-stress constitutive matrix for isotropic linear elasticity.

    Parameters
    ----------
    E:
        Young's modulus.
    nu:
        Poisson's ratio.

    Returns
    -------
    C : (3,3) ndarray
        Plane-stress matrix in Voigt ordering [xx, yy, xy].
    """

    c = float(E) / (1.0 - float(nu) * float(nu))
    C = c * np.array(
        [[1.0, float(nu), 0.0], [float(nu), 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - float(nu))]],
        dtype=float,
    )
    return C
