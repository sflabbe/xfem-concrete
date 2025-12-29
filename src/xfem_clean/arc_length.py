"""Arc-length control for nonlinear finite element analysis.

This module implements the arc-length (also called Riks or Crisfield) method for
path-following in nonlinear structural analysis. This method enables tracing
equilibrium paths past limit points (snap-through, snap-back).

The key idea is to replace displacement control with a constraint on the
arc-length in the combined displacement-load space:

.. math::
    \\Delta u^T \\Delta u + \\psi^2 \\Delta\\lambda^2 P^T P = \\Delta l^2

where:
  - Δu: incremental displacement
  - Δλ: incremental load factor
  - P: reference load pattern
  - ψ: scaling parameter
  - Δl: arc-length step size

References
----------
- Crisfield, M. A. (1981). "A fast incremental/iterative solution procedure
  that handles 'snap-through'." Computers & Structures, 13(1-3), 55-62.
- Riks, E. (1979). "An incremental approach to the solution of snapping and
  buckling problems." International Journal of Solids and Structures, 15(7), 529-551.
"""

from __future__ import annotations

from typing import Tuple, Dict, Optional, Callable, Any
import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class ArcLengthSolver:
    """Arc-length solver for quasi-static nonlinear analysis.

    Attributes
    ----------
    arc_length : float
        Current arc-length parameter [units consistent with problem]
    arc_length_initial : float
        Initial arc-length (user-specified)
    psi : float
        Load-displacement scaling factor
    lambda_current : float
        Current load factor
    max_iterations : int
        Maximum Newton iterations per step
    tol_residual : float
        Residual norm tolerance
    tol_displacement : float
        Displacement increment norm tolerance
    adaptive : bool
        Use adaptive arc-length adjustment
    """

    def __init__(
        self,
        arc_length_initial: float = 0.01,
        psi: float = 1.0,
        max_iterations: int = 25,
        tol_residual: float = 1e-6,
        tol_displacement: float = 1e-8,
        adaptive: bool = True,
        arc_length_min: float = 1e-6,
        arc_length_max: float = 1.0,
    ):
        """Initialize arc-length solver.

        Parameters
        ----------
        arc_length_initial : float
            Initial arc-length step size
        psi : float
            Scaling factor (typically 1.0 or 1/||P||)
        max_iterations : int
            Maximum Newton iterations
        tol_residual : float
            Convergence tolerance for residual norm
        tol_displacement : float
            Convergence tolerance for displacement increment norm
        adaptive : bool
            Enable adaptive arc-length adjustment
        arc_length_min, arc_length_max : float
            Bounds for adaptive arc-length
        """
        self.arc_length = float(arc_length_initial)
        self.arc_length_initial = float(arc_length_initial)
        self.psi = float(psi)
        self.lambda_current = 0.0
        self.max_iterations = int(max_iterations)
        self.tol_residual = float(tol_residual)
        self.tol_displacement = float(tol_displacement)
        self.adaptive = bool(adaptive)
        self.arc_length_min = float(arc_length_min)
        self.arc_length_max = float(arc_length_max)

    def solve_step(
        self,
        K: sp.csr_matrix,
        f_int: np.ndarray,
        P_ref: np.ndarray,
        u_n: np.ndarray,
        lambda_n: float,
        assemble_system: Callable[[np.ndarray], Tuple[sp.csr_matrix, np.ndarray]],
        apply_bc: Optional[Callable[[sp.csr_matrix, np.ndarray], Tuple[sp.csr_matrix, np.ndarray]]] = None,
        debug: bool = False,
    ) -> Tuple[bool, np.ndarray, float, float, int]:
        """Perform one arc-length controlled load step.

        Parameters
        ----------
        K : sparse.csr_matrix
            Current tangent stiffness matrix
        f_int : np.ndarray
            Current internal force vector
        P_ref : np.ndarray
            Reference load pattern (fixed)
        u_n : np.ndarray
            Displacement at start of step
        lambda_n : float
            Load factor at start of step
        assemble_system : Callable
            Function to reassemble (K, f_int) given u_trial
            Signature: (u) -> (K, f_int)
        apply_bc : Optional[Callable]
            Function to apply boundary conditions
            Signature: (K, f) -> (K_bc, f_bc)
        debug : bool
            Print debug information

        Returns
        -------
        converged : bool
            Whether the step converged
        u_new : np.ndarray
            Updated displacement
        lambda_new : float
            Updated load factor
        arc_length_new : float
            Updated arc-length (if adaptive)
        num_iterations : int
            Number of iterations taken
        """
        arc_len = self.arc_length
        psi = self.psi

        # Auto-compute psi if not set
        norm_P = float(np.linalg.norm(P_ref))
        if norm_P > 1e-12 and psi == 1.0:
            psi = 1.0 / norm_P

        # ======================================================================
        # Predictor: tangent step
        # ======================================================================

        # Solve: K * du_bar = P_ref
        if apply_bc is not None:
            K_bc, P_bc = apply_bc(K, P_ref)
        else:
            K_bc = K
            P_bc = P_ref

        try:
            du_bar = spla.spsolve(K_bc, P_bc)
        except Exception as e:
            if debug:
                print(f"[arc-length] Predictor solve failed: {e}")
            return False, u_n, lambda_n, arc_len, 0

        du_bar = np.asarray(du_bar).flatten()

        # Initial load factor increment (tangent predictor)
        denom = float(np.dot(du_bar, du_bar) + (psi ** 2) * (norm_P ** 2))
        if denom < 1e-20:
            if debug:
                print("[arc-length] Predictor denominator near zero")
            return False, u_n, lambda_n, arc_len, 0

        dlambda_0 = arc_len / math.sqrt(denom)

        # Initial predictor displacement
        du_0 = dlambda_0 * du_bar

        u_trial = u_n + du_0
        lambda_trial = lambda_n + dlambda_0

        if debug:
            print(f"[arc-length] Predictor: dλ={dlambda_0:.4e}, ||du||={np.linalg.norm(du_0):.4e}")

        # ======================================================================
        # Corrector iterations (modified Newton with arc-length constraint)
        # ======================================================================

        converged = False
        num_iter = 0

        for it in range(self.max_iterations):
            num_iter = it + 1

            # Reassemble at current trial state
            K_trial, f_int_trial = assemble_system(u_trial)

            # Residual: r = λ * P_ref - f_int
            r = lambda_trial * P_ref - f_int_trial

            # Check convergence
            norm_r = float(np.linalg.norm(r))

            if norm_r < self.tol_residual:
                converged = True
                if debug:
                    print(f"[arc-length] Converged at iteration {it+1}: ||r||={norm_r:.3e}")
                break

            # Solve two auxiliary systems:
            # (1) du_I: correction for residual
            # (2) du_II: correction for load increment

            if apply_bc is not None:
                K_bc, r_bc = apply_bc(K_trial, r)
                _, P_bc = apply_bc(K_trial, P_ref)
            else:
                K_bc = K_trial
                r_bc = r
                P_bc = P_ref

            try:
                du_I = spla.spsolve(K_bc, r_bc)
                du_II = spla.spsolve(K_bc, P_bc)
            except Exception as e:
                if debug:
                    print(f"[arc-length] Iteration {it+1} solve failed: {e}")
                break

            du_I = np.asarray(du_I).flatten()
            du_II = np.asarray(du_II).flatten()

            # Arc-length constraint (quadratic equation for dλ):
            # ||u_trial + du_I + dλ*du_II - u_n||^2 + ψ²(λ_trial + dλ - λ_n)²||P||^2 = Δl²

            delta_u = u_trial - u_n
            delta_lambda = lambda_trial - lambda_n

            # Quadratic coefficients: a*dλ² + b*dλ + c = 0
            a = float(np.dot(du_II, du_II) + (psi ** 2) * (norm_P ** 2))
            b = 2.0 * float(np.dot(delta_u + du_I, du_II) + (psi ** 2) * delta_lambda * (norm_P ** 2))
            c = float(np.dot(delta_u + du_I, delta_u + du_I) + (psi ** 2) * (delta_lambda ** 2) * (norm_P ** 2) - (arc_len ** 2))

            # Solve quadratic
            disc = b * b - 4.0 * a * c

            if disc < 0.0:
                if debug:
                    print(f"[arc-length] Iteration {it+1}: negative discriminant")
                break

            sqrt_disc = math.sqrt(disc)

            # Two roots
            dlambda_plus = (-b + sqrt_disc) / (2.0 * a + 1e-20)
            dlambda_minus = (-b - sqrt_disc) / (2.0 * a + 1e-20)

            # Choose root: prefer the one closer to predictor direction
            # (i.e., smaller in magnitude, or positive if starting from zero)
            if abs(dlambda_plus) < abs(dlambda_minus):
                dlambda = dlambda_plus
            else:
                dlambda = dlambda_minus

            # Update
            du_corr = du_I + dlambda * du_II
            u_trial += du_corr
            lambda_trial += dlambda

            norm_du = float(np.linalg.norm(du_corr))

            if debug:
                print(f"  it={it+1:02d}  ||r||={norm_r:.3e}  ||du||={norm_du:.3e}  dλ={dlambda:.4e}  λ={lambda_trial:.4f}")

            # Secondary convergence check (displacement stagnation)
            if norm_du < self.tol_displacement:
                if norm_r > 10.0 * self.tol_residual:
                    # Stagnated but not converged
                    if debug:
                        print(f"[arc-length] Stagnated at iteration {it+1}")
                    break
                else:
                    # Close enough
                    converged = True
                    if debug:
                        print(f"[arc-length] Converged (displacement) at iteration {it+1}")
                    break

        # ======================================================================
        # Adaptive arc-length adjustment
        # ======================================================================

        if self.adaptive and converged:
            # Desired iterations: aim for ~5-7 iterations
            desired_iters = 6
            scaling_factor = min(2.0, max(0.5, float(desired_iters) / max(num_iter, 1)))
            arc_len_new = arc_len * scaling_factor
            arc_len_new = max(self.arc_length_min, min(arc_len_new, self.arc_length_max))
        elif not converged:
            # Reduce arc-length for retry
            arc_len_new = arc_len * 0.5
            arc_len_new = max(self.arc_length_min, arc_len_new)
        else:
            arc_len_new = arc_len

        if converged:
            self.arc_length = arc_len_new
            self.lambda_current = lambda_trial

        return converged, u_trial, lambda_trial, arc_len_new, num_iter

    def reset(self):
        """Reset solver to initial state."""
        self.arc_length = self.arc_length_initial
        self.lambda_current = 0.0


# ==============================================================================
# Simplified Arc-Length Step (Functional Interface)
# ==============================================================================

def arc_length_step(
    K: sp.csr_matrix,
    f_int: np.ndarray,
    P_ref: np.ndarray,
    u_n: np.ndarray,
    lambda_n: float,
    arc_length: float,
    assemble_fn: Callable[[np.ndarray], Tuple[sp.csr_matrix, np.ndarray]],
    fixed_dofs: Optional[Dict[int, float]] = None,
    max_iter: int = 25,
    tol_r: float = 1e-6,
    tol_du: float = 1e-8,
    debug: bool = False,
) -> Tuple[bool, np.ndarray, float, int]:
    """Perform a single arc-length controlled step (functional interface).

    Parameters
    ----------
    K : sparse.csr_matrix
        Initial tangent stiffness
    f_int : np.ndarray
        Initial internal forces
    P_ref : np.ndarray
        Reference load pattern
    u_n : np.ndarray
        Current displacement
    lambda_n : float
        Current load factor
    arc_length : float
        Arc-length parameter
    assemble_fn : Callable
        Function to assemble (K, f_int) at given u
    fixed_dofs : Optional[Dict[int, float]]
        Fixed DOF constraints {dof_index: value}
    max_iter : int
        Max iterations
    tol_r, tol_du : float
        Convergence tolerances
    debug : bool
        Debug output

    Returns
    -------
    converged : bool
    u_new : np.ndarray
    lambda_new : float
    num_iterations : int
    """

    def apply_bc(K_in, f_in):
        """Apply Dirichlet boundary conditions."""
        if fixed_dofs is None or len(fixed_dofs) == 0:
            return K_in, f_in

        K_out = K_in.tolil()
        f_out = f_in.copy()

        for dof, val in fixed_dofs.items():
            dof = int(dof)
            # Zero out row
            K_out[dof, :] = 0.0
            K_out[dof, dof] = 1.0
            f_out[dof] = 0.0  # Homogeneous BC for incremental problem

        return K_out.tocsr(), f_out

    solver = ArcLengthSolver(
        arc_length_initial=arc_length,
        max_iterations=max_iter,
        tol_residual=tol_r,
        tol_displacement=tol_du,
        adaptive=False,  # Caller handles adaptation
    )

    converged, u_new, lambda_new, _, num_iter = solver.solve_step(
        K, f_int, P_ref, u_n, lambda_n,
        assemble_system=assemble_fn,
        apply_bc=apply_bc,
        debug=debug,
    )

    return converged, u_new, lambda_new, num_iter


# ==============================================================================
# Cylindrical Arc-Length (Alternative Constraint)
# ==============================================================================

def cylindrical_arc_length_step(
    K: sp.csr_matrix,
    f_int: np.ndarray,
    P_ref: np.ndarray,
    u_n: np.ndarray,
    lambda_n: float,
    arc_length: float,
    assemble_fn: Callable[[np.ndarray], Tuple[sp.csr_matrix, np.ndarray]],
    control_dof: int,
    fixed_dofs: Optional[Dict[int, float]] = None,
    max_iter: int = 25,
    tol_r: float = 1e-6,
    tol_du: float = 1e-8,
    debug: bool = False,
) -> Tuple[bool, np.ndarray, float, int]:
    """Cylindrical arc-length method (Crisfield variant).

    Uses constraint on displacement at a single control DOF:
        (u[control_dof] - u_n[control_dof])^2 = Δl^2

    This is more robust for problems with a dominant DOF (e.g., point load).

    Parameters
    ----------
    control_dof : int
        DOF index to control arc-length

    Returns
    -------
    converged, u_new, lambda_new, num_iterations
    """
    # Implementation left as exercise (similar structure to spherical arc-length)
    raise NotImplementedError("Cylindrical arc-length not yet implemented.")


# ==============================================================================
# Utilities
# ==============================================================================

def estimate_initial_arc_length(
    K: sp.csr_matrix,
    P_ref: np.ndarray,
    u_target_scale: float = 1e-3,
) -> float:
    """Estimate a reasonable initial arc-length based on problem scale.

    Parameters
    ----------
    K : sparse.csr_matrix
        Initial stiffness matrix
    P_ref : np.ndarray
        Reference load pattern
    u_target_scale : float
        Target displacement magnitude for first step

    Returns
    -------
    arc_length_initial : float
    """
    try:
        du = spla.spsolve(K, P_ref)
        du = np.asarray(du).flatten()
        norm_du = float(np.linalg.norm(du))
        if norm_du > 1e-12:
            lambda_scale = u_target_scale / norm_du
            arc_length = lambda_scale * math.sqrt(norm_du ** 2 + np.dot(P_ref, P_ref))
            return float(arc_length)
    except Exception:
        pass

    # Fallback
    return 0.01


def load_displacement_curve(
    lambdas: np.ndarray,
    u_history: np.ndarray,
    control_dof: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract load-displacement curve from arc-length analysis.

    Parameters
    ----------
    lambdas : np.ndarray
        Load factor history
    u_history : np.ndarray
        Displacement history [n_steps, n_dof]
    control_dof : int
        DOF to plot

    Returns
    -------
    loads : np.ndarray
        Load factors
    displacements : np.ndarray
        Displacements at control DOF
    """
    displacements = u_history[:, control_dof]
    return lambdas, displacements
