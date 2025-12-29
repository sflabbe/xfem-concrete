"""Boundary condition helpers."""

from __future__ import annotations
from typing import Dict, Tuple, List
import numpy as np
import scipy.sparse as sp

def apply_dirichlet(K: sp.csr_matrix, r: np.ndarray, fixed: Dict[int, float], u: np.ndarray):
    """
    Nonlinear Dirichlet handling:
    - Enforce u[dof] = value directly on the iterate u
    - The Newton solve is done only on free dofs: K_ff * du_f = -r_f
      with du_fixed = 0.
    """
    ndof = K.shape[0]
    all_ids = np.arange(ndof, dtype=int)
    fixed_ids = np.array(sorted(fixed.keys()), dtype=int)
    free = np.setdiff1d(all_ids, fixed_ids)

    # Enforce values on current iterate BEFORE extracting free equations.
    # Note: in a residual-based Newton method, if the element assembly already
    # evaluated r(u) and K(u) using the iterate with prescribed values enforced,
    # then the free residual is simply r[free] and the Newton system is
    #   K_ff * du_f = -r_f
    # with du_fixed = 0.
    for dof in fixed_ids:
        u[dof] = fixed[dof]

    K_ff = K[free, :][:, free]
    r_f = r[free]
    return free, K_ff, r_f, fixed_ids


# -----------------------------
# High-level model and solver
# -----------------------------
