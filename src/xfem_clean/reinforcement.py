"""Mesh-independent reinforcement layers via Heaviside enrichment.

This module implements reinforcement as 1D objects superimposed on the background
mesh using Heaviside enrichment per dissertation Eq. (4.92-4.103).

The key innovation is that reinforcement layers are defined geometrically (as
polylines/segments) independent of the mesh, and their contribution is integrated
via line integrals along the reinforcement path.

Reference: Dissertation 10.5445/IR/1000124842, Chapter 4, Section 4.5
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, List, Optional, Literal

import numpy as np
import scipy.sparse as sp

# Optional Numba acceleration
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ReinforcementSegment:
    """Single segment of a reinforcement layer.

    Attributes
    ----------
    x0 : np.ndarray
        Start point [x, y]
    x1 : np.ndarray
        End point [x, y]
    A_s : float
        Cross-sectional area [m²]
    E_s : float
        Young's modulus [Pa]
    f_y : float
        Yield stress [Pa]
    E_h : float
        Hardening modulus [Pa]
    d_bar : float
        Bar diameter [m] (for bond-slip coupling)
    layer_id : int
        Parent layer identifier
    """
    x0: np.ndarray
    x1: np.ndarray
    A_s: float
    E_s: float
    f_y: float
    E_h: float
    d_bar: float
    layer_id: int

    def length(self) -> float:
        """Segment length."""
        dx = self.x1 - self.x0
        return float(np.linalg.norm(dx))

    def tangent(self) -> np.ndarray:
        """Unit tangent vector."""
        dx = self.x1 - self.x0
        L = self.length()
        if L < 1e-14:
            return np.array([1.0, 0.0])
        return (dx / L).astype(float)

    def point_at(self, s: float) -> np.ndarray:
        """Point at arc length s from x0."""
        return self.x0 + s * self.tangent()


ReinforcementType = Literal["longitudinal", "transverse", "fibre", "bonded_sheet"]


@dataclass
class ReinforcementLayer:
    """Collection of segments forming a reinforcement layer.

    Attributes
    ----------
    segments : List[ReinforcementSegment]
        List of segments forming the layer
    A_total : float
        Total cross-sectional area [m²]
    E_s : float
        Steel Young's modulus [Pa]
    f_y : float
        Yield stress [Pa]
    E_h : float
        Hardening modulus [Pa]
    d_bar : float
        Bar diameter [m]
    layer_type : ReinforcementType
        Type of reinforcement
    layer_id : int
        Unique layer identifier
    """
    segments: List[ReinforcementSegment]
    A_total: float
    E_s: float
    f_y: float
    E_h: float
    d_bar: float
    layer_type: ReinforcementType
    layer_id: int

    def total_length(self) -> float:
        """Total length of all segments."""
        return sum(seg.length() for seg in self.segments)


@dataclass
class ReinforcementState:
    """State variables for reinforcement elements.

    Attributes
    ----------
    n_integration_points : int
        Total number of integration points across all segments
    eps_s : np.ndarray
        Axial strain at each integration point [-]
    eps_s_p : np.ndarray
        Plastic strain at each integration point [-]
    eps_s_max : np.ndarray
        Maximum strain history (for hardening) [-]
    sigma_s : np.ndarray
        Axial stress at each integration point [Pa]
    """
    n_integration_points: int
    eps_s: np.ndarray
    eps_s_p: np.ndarray
    eps_s_max: np.ndarray
    sigma_s: np.ndarray

    @classmethod
    def zeros(cls, n_pts: int) -> ReinforcementState:
        """Initialize zero states."""
        return cls(
            n_integration_points=int(n_pts),
            eps_s=np.zeros(n_pts, dtype=float),
            eps_s_p=np.zeros(n_pts, dtype=float),
            eps_s_max=np.zeros(n_pts, dtype=float),
            sigma_s=np.zeros(n_pts, dtype=float),
        )

    def copy(self) -> ReinforcementState:
        """Deep copy of state."""
        return ReinforcementState(
            n_integration_points=int(self.n_integration_points),
            eps_s=self.eps_s.copy(),
            eps_s_p=self.eps_s_p.copy(),
            eps_s_max=self.eps_s_max.copy(),
            sigma_s=self.sigma_s.copy(),
        )


# =============================================================================
# Geometry Helpers
# =============================================================================

def signed_distance_to_segment(x: np.ndarray, x0: np.ndarray, x1: np.ndarray) -> float:
    """Signed distance from point x to line segment (x0, x1).

    Distance is measured perpendicular to the segment centerline.
    Sign convention: positive on the "left" side of x0→x1.

    Parameters
    ----------
    x : np.ndarray
        Query point [x, y]
    x0, x1 : np.ndarray
        Segment endpoints

    Returns
    -------
    phi : float
        Signed distance (positive on +n side)
    """
    dx = x1 - x0
    L = float(np.linalg.norm(dx))
    if L < 1e-14:
        return float(np.linalg.norm(x - x0))

    # Tangent and normal vectors
    t = dx / L
    n = np.array([-t[1], t[0]])  # Left-hand normal (90° CCW rotation)

    # Signed distance
    phi = float(np.dot(n, x - x0))
    return phi


def heaviside_enrichment(phi: float) -> float:
    """Heaviside function for reinforcement domain.

    Per Eq. (4.92):
        H_r(x) = 1 if phi_r(x) >= 0
                 0 if phi_r(x) < 0

    Parameters
    ----------
    phi : float
        Signed distance to reinforcement centerline

    Returns
    -------
    H : float
        Heaviside value (0 or 1)
    """
    return 1.0 if phi >= 0.0 else 0.0


def shifted_heaviside_enrichment(phi_x: float, phi_i: float) -> float:
    """Shifted Heaviside enrichment function.

    Per Eq. (4.92):
        H̃_r(x) = H_r(x) - H_r(x_i)

    This ensures partition of unity and eliminates rigid-body modes.

    Parameters
    ----------
    phi_x : float
        Signed distance at query point x
    phi_i : float
        Signed distance at node i

    Returns
    -------
    H_tilde : float
        Shifted Heaviside value
    """
    H_x = heaviside_enrichment(phi_x)
    H_i = heaviside_enrichment(phi_i)
    return H_x - H_i


# =============================================================================
# Kinematics
# =============================================================================

def compute_bar_strain_from_continuum(
    eps_continuum: np.ndarray,
    t_bar: np.ndarray,
) -> float:
    """Extract bar axial strain from continuum strain tensor.

    Per Eq. (4.101-4.103):
        ε_s = t^T · ε · t

    where ε is the 2D continuum strain tensor and t is the bar tangent.

    Parameters
    ----------
    eps_continuum : np.ndarray
        Continuum strain tensor [εxx, εyy, γxy] (Voigt notation)
    t_bar : np.ndarray
        Unit tangent vector [tx, ty]

    Returns
    -------
    eps_s : float
        Axial strain in bar direction
    """
    # Voigt to tensor form
    eps_xx = eps_continuum[0]
    eps_yy = eps_continuum[1]
    gamma_xy = eps_continuum[2]

    # ε_s = t_x^2 * ε_xx + t_y^2 * ε_yy + 2 * t_x * t_y * ε_xy
    # Note: γ_xy = 2 * ε_xy (engineering shear strain)
    eps_xy = gamma_xy / 2.0

    tx, ty = t_bar[0], t_bar[1]
    eps_s = tx * tx * eps_xx + ty * ty * eps_yy + 2.0 * tx * ty * eps_xy

    return float(eps_s)


def compute_B_s_operator(
    B_continuum: np.ndarray,
    t_bar: np.ndarray,
) -> np.ndarray:
    """Compute strain-displacement operator for bar strain.

    Per Eq. (4.101):
        ε_s = B_s · q
    where
        B_s = t^T · B_continuum

    In practice: ε_s = [tx^2, ty^2, tx*ty] · B_continuum

    Parameters
    ----------
    B_continuum : np.ndarray
        Continuum B matrix, shape (3, ndof_elem)
        Rows: [εxx, εyy, γxy]
    t_bar : np.ndarray
        Unit tangent [tx, ty]

    Returns
    -------
    B_s : np.ndarray
        Bar strain-displacement operator, shape (ndof_elem,)
    """
    tx, ty = t_bar[0], t_bar[1]

    # Weight vector for Voigt strain: [tx^2, ty^2, tx*ty]
    # (Note: γ_xy = 2*ε_xy, so we need tx*ty not 2*tx*ty)
    w = np.array([tx * tx, ty * ty, 2.0 * tx * ty], dtype=float)

    # B_s = w^T · B_continuum
    B_s = np.dot(w, B_continuum)

    return B_s


# =============================================================================
# Constitutive Models
# =============================================================================

def steel_elastic_1d(eps: float, E: float) -> Tuple[float, float]:
    """Elastic steel constitutive law.

    Parameters
    ----------
    eps : float
        Axial strain
    E : float
        Young's modulus [Pa]

    Returns
    -------
    sigma : float
        Stress [Pa]
    E_t : float
        Tangent modulus [Pa]
    """
    sigma = E * eps
    E_t = E
    return float(sigma), float(E_t)


def steel_bilinear_1d(
    eps: float,
    eps_p: float,
    E: float,
    f_y: float,
    E_h: float,
) -> Tuple[float, float, float]:
    """Bilinear elasto-plastic steel constitutive law (1D).

    Implements J2 plasticity in 1D with linear hardening.

    Parameters
    ----------
    eps : float
        Total axial strain
    eps_p : float
        Plastic strain (from previous step)
    E : float
        Young's modulus [Pa]
    f_y : float
        Yield stress [Pa]
    E_h : float
        Hardening modulus [Pa]

    Returns
    -------
    sigma : float
        Stress [Pa]
    E_t : float
        Consistent tangent modulus [Pa]
    eps_p_new : float
        Updated plastic strain
    """
    eps_y = f_y / E

    # Trial elastic stress
    eps_e_trial = eps - eps_p
    sigma_trial = E * eps_e_trial

    # Yield function
    f = abs(sigma_trial) - f_y

    if f <= 0.0:
        # Elastic
        sigma = sigma_trial
        E_t = E
        eps_p_new = eps_p
    else:
        # Plastic
        sign = 1.0 if sigma_trial >= 0.0 else -1.0

        # Return mapping
        d_eps_p = f / (E + E_h)
        eps_p_new = eps_p + sign * d_eps_p

        # Updated stress
        eps_e = eps - eps_p_new
        sigma = E * eps_e

        # Consistent tangent (algorithmic)
        E_t = E * E_h / (E + E_h)

    return float(sigma), float(E_t), float(eps_p_new)


# =============================================================================
# Line-Integral Assembly
# =============================================================================

def segment_element_intersection(
    seg: ReinforcementSegment,
    elem_nodes: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """Check if reinforcement segment intersects element.

    Parameters
    ----------
    seg : ReinforcementSegment
        Reinforcement segment
    elem_nodes : np.ndarray
        Element node coordinates, shape (4, 2) for Q4

    Returns
    -------
    s_start, s_end : float or None
        Arc length parameters where segment intersects element bbox,
        or None if no intersection
    """
    # Element bounding box
    xmin = float(elem_nodes[:, 0].min())
    xmax = float(elem_nodes[:, 0].max())
    ymin = float(elem_nodes[:, 1].min())
    ymax = float(elem_nodes[:, 1].max())

    # Segment endpoints
    x0, x1 = seg.x0, seg.x1
    t = seg.tangent()
    L = seg.length()

    # Parametric intersection with bbox
    # Segment: p(s) = x0 + s * t, s ∈ [0, L]
    s_min = 0.0
    s_max = L

    # Clip against x bounds
    if abs(t[0]) > 1e-14:
        s_xmin = (xmin - x0[0]) / t[0]
        s_xmax = (xmax - x0[0]) / t[0]
        s_min = max(s_min, min(s_xmin, s_xmax))
        s_max = min(s_max, max(s_xmin, s_xmax))

    # Clip against y bounds
    if abs(t[1]) > 1e-14:
        s_ymin = (ymin - x0[1]) / t[1]
        s_ymax = (ymax - x0[1]) / t[1]
        s_min = max(s_min, min(s_ymin, s_ymax))
        s_max = min(s_max, max(s_ymin, s_ymax))

    # Check for valid intersection
    if s_min >= s_max - 1e-12:
        return None

    return float(s_min), float(s_max)


def gauss_line_quadrature(n_points: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre quadrature rule for line integrals.

    Parameters
    ----------
    n_points : int
        Number of quadrature points (1, 2, 3, 4, 5, 6, 7, ...)

    Returns
    -------
    xi : np.ndarray
        Quadrature points in [-1, 1]
    w : np.ndarray
        Quadrature weights
    """
    # Standard Gauss-Legendre rules
    if n_points == 1:
        xi = np.array([0.0])
        w = np.array([2.0])
    elif n_points == 2:
        xi = np.array([-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)])
        w = np.array([1.0, 1.0])
    elif n_points == 3:
        xi = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
        w = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
    elif n_points == 7:
        # 7-point rule (thesis default)
        xi = np.array([
            -0.9491079123427585,
            -0.7415311855993945,
            -0.4058451513773972,
            0.0,
            0.4058451513773972,
            0.7415311855993945,
            0.9491079123427585,
        ])
        w = np.array([
            0.1294849661688697,
            0.2797053914892766,
            0.3818300505051189,
            0.4179591836734694,
            0.3818300505051189,
            0.2797053914892766,
            0.1294849661688697,
        ])
    else:
        # Fallback: use numpy/scipy
        from numpy.polynomial.legendre import leggauss
        xi, w = leggauss(n_points)

    return xi.astype(float), w.astype(float)


def assemble_reinforcement_layers(
    q: np.ndarray,
    nodes: np.ndarray,
    elems: np.ndarray,
    dofs_map: object,  # XFEMDofs or similar
    layers: List[ReinforcementLayer],
    states_comm: Optional[List[ReinforcementState]] = None,
    n_gauss_line: int = 7,
    use_plasticity: bool = False,
) -> Tuple[np.ndarray, sp.csr_matrix, List[ReinforcementState]]:
    """Assemble reinforcement contribution via line integrals.

    Per Eq. (4.101-4.103):
        f_s^int = ∫_{Γ_r} B_s^T · A_s · σ_s ds
        K_s = ∫_{Γ_r} B_s^T · A_s · E_t · B_s ds

    Parameters
    ----------
    q : np.ndarray
        Global displacement vector
    nodes : np.ndarray
        Node coordinates, shape (nnode, 2)
    elems : np.ndarray
        Element connectivity, shape (nelem, 4)
    dofs_map : object
        DOF mapping (must have .std for standard DOFs)
    layers : List[ReinforcementLayer]
        List of reinforcement layers
    states_comm : List[ReinforcementState], optional
        Committed states for each layer
    n_gauss_line : int
        Number of Gauss points for line integration (thesis default: 7)
    use_plasticity : bool
        Use elasto-plastic steel model

    Returns
    -------
    f_rebar : np.ndarray
        Internal force vector
    K_rebar : sp.csr_matrix
        Stiffness matrix
    states_new : List[ReinforcementState]
        Updated states (trial)
    """
    ndof = len(q)
    f_rebar = np.zeros(ndof, dtype=float)

    rows = []
    cols = []
    data = []

    states_new = []

    # Gauss quadrature rule
    xi_gauss, w_gauss = gauss_line_quadrature(n_gauss_line)

    for layer_idx, layer in enumerate(layers):
        # Initialize or get committed state
        if states_comm is not None and layer_idx < len(states_comm):
            state_comm = states_comm[layer_idx]
        else:
            n_pts = len(layer.segments) * n_gauss_line
            state_comm = ReinforcementState.zeros(n_pts)

        state_trial = state_comm.copy()
        gp_idx = 0

        for seg in layer.segments:
            # Find elements intersected by this segment
            for e_idx in range(elems.shape[0]):
                elem_conn = elems[e_idx]
                elem_nodes_coords = nodes[elem_conn]

                isect = segment_element_intersection(seg, elem_nodes_coords)
                if isect is None:
                    continue

                s_start, s_end = isect
                L_int = s_end - s_start

                if L_int < 1e-14:
                    continue

                # Integrate along intersected portion
                t_bar = seg.tangent()

                for gp in range(n_gauss_line):
                    # Map Gauss point from [-1, 1] to [s_start, s_end]
                    s_local = s_start + 0.5 * (1.0 + xi_gauss[gp]) * L_int
                    w_local = 0.5 * L_int * w_gauss[gp]

                    # Point on segment
                    x_gp = seg.point_at(s_local)

                    # Evaluate B matrix at this point
                    # (Requires shape functions and derivatives at x_gp)
                    # For now, use element-averaged B (simplified)
                    # Full implementation would compute B(x_gp) properly

                    # TODO: Implement proper B evaluation at arbitrary x_gp
                    # For prototype, use element center approximation
                    B_elem = compute_element_B_matrix_at_point(
                        x_gp, elem_nodes_coords, elem_conn, dofs_map
                    )

                    if B_elem is None:
                        continue

                    # Extract continuum strain
                    eps_continuum = np.dot(B_elem, q)  # [εxx, εyy, γxy]

                    # Bar strain
                    eps_s = compute_bar_strain_from_continuum(eps_continuum, t_bar)

                    # Constitutive update
                    if use_plasticity:
                        eps_p_old = state_trial.eps_s_p[gp_idx]
                        sigma_s, E_t, eps_p_new = steel_bilinear_1d(
                            eps_s, eps_p_old, layer.E_s, layer.f_y, layer.E_h
                        )
                        state_trial.eps_s_p[gp_idx] = eps_p_new
                    else:
                        sigma_s, E_t = steel_elastic_1d(eps_s, layer.E_s)

                    state_trial.eps_s[gp_idx] = eps_s
                    state_trial.sigma_s[gp_idx] = sigma_s

                    # B_s operator
                    B_s = compute_B_s_operator(B_elem, t_bar)

                    # Integrate force: f += B_s^T · A_s · σ_s · w
                    f_rebar += B_s.T * seg.A_s * sigma_s * w_local

                    # Integrate stiffness: K += B_s^T · A_s · E_t · B_s · w
                    K_local = np.outer(B_s, B_s) * seg.A_s * E_t * w_local

                    # Extract DOF indices from B matrix structure
                    # (Assumes B_elem corresponds to element DOFs)
                    elem_dofs = get_element_dofs(elem_conn, dofs_map)

                    for i, dof_i in enumerate(elem_dofs):
                        for j, dof_j in enumerate(elem_dofs):
                            rows.append(dof_i)
                            cols.append(dof_j)
                            data.append(K_local[i, j])

                    gp_idx += 1

        states_new.append(state_trial)

    # Build sparse matrix
    K_rebar = sp.csr_matrix((data, (rows, cols)), shape=(ndof, ndof))

    return f_rebar, K_rebar, states_new


def compute_element_B_matrix_at_point(
    x: np.ndarray,
    elem_nodes: np.ndarray,
    elem_conn: np.ndarray,
    dofs_map: object,
) -> Optional[np.ndarray]:
    """Compute B matrix at arbitrary point x within element.

    Uses Newton iteration to find (ξ, η) corresponding to x, then
    computes B matrix via isoparametric mapping.

    Parameters
    ----------
    x : np.ndarray
        Point coordinates [x, y]
    elem_nodes : np.ndarray
        Element node coordinates, shape (4, 2)
    elem_conn : np.ndarray
        Element connectivity (node indices)
    dofs_map : object
        DOF mapping

    Returns
    -------
    B : np.ndarray or None
        B matrix [3, ndof_elem] or None if x outside element
    """
    # Find (ξ, η) via Newton iteration
    xi_eta = find_xi_eta_newton(x, elem_nodes)

    if xi_eta is None:
        return None

    xi, eta = xi_eta

    # Check if point is inside element (with tolerance)
    tol = 0.1  # Allow slight extrapolation for numerical stability
    if abs(xi) > 1.0 + tol or abs(eta) > 1.0 + tol:
        return None

    # Compute B matrix at (ξ, η)
    B = compute_B_matrix_q4(xi, eta, elem_nodes)

    return B


def find_xi_eta_newton(
    x: np.ndarray,
    elem_nodes: np.ndarray,
    max_iter: int = 10,
    tol: float = 1e-10,
) -> Optional[Tuple[float, float]]:
    """Find (ξ, η) corresponding to physical point x via Newton iteration.

    Solves the nonlinear system:
        x(ξ, η) = Σ N_i(ξ, η) · x_i = x_target

    Parameters
    ----------
    x : np.ndarray
        Target point [x, y]
    elem_nodes : np.ndarray
        Element node coordinates, shape (4, 2)
    max_iter : int
        Maximum Newton iterations
    tol : float
        Convergence tolerance

    Returns
    -------
    (xi, eta) : tuple or None
        Parametric coordinates or None if failed to converge
    """
    from xfem_clean.fem.q4 import q4_shape

    # Initial guess: element center
    xi = 0.0
    eta = 0.0

    for _ in range(max_iter):
        # Evaluate shape functions and derivatives
        N, dN_dxi, dN_deta = q4_shape(xi, eta)

        # Current physical position
        x_current = np.zeros(2, dtype=float)
        for i in range(4):
            x_current += N[i] * elem_nodes[i]

        # Residual
        r = x - x_current

        # Check convergence
        if np.linalg.norm(r) < tol:
            return float(xi), float(eta)

        # Jacobian: dx/dξ
        J = np.zeros((2, 2), dtype=float)
        for i in range(4):
            J[0, 0] += dN_dxi[i] * elem_nodes[i, 0]
            J[0, 1] += dN_deta[i] * elem_nodes[i, 0]
            J[1, 0] += dN_dxi[i] * elem_nodes[i, 1]
            J[1, 1] += dN_deta[i] * elem_nodes[i, 1]

        # Check singularity
        det_J = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det_J) < 1e-14:
            return None

        # Newton update: Δξ = J^{-1} · r
        J_inv = np.array([
            [J[1, 1], -J[0, 1]],
            [-J[1, 0], J[0, 0]]
        ]) / det_J

        d_xi_eta = np.dot(J_inv, r)

        xi += d_xi_eta[0]
        eta += d_xi_eta[1]

    # Failed to converge
    return None


def compute_B_matrix_q4(
    xi: float,
    eta: float,
    elem_nodes: np.ndarray,
) -> np.ndarray:
    """Compute B matrix for Q4 element at (ξ, η).

    Returns B matrix in Voigt notation:
        ε = [εxx, εyy, γxy]^T = B · q_elem

    Parameters
    ----------
    xi, eta : float
        Parametric coordinates
    elem_nodes : np.ndarray
        Element node coordinates, shape (4, 2)

    Returns
    -------
    B : np.ndarray
        Strain-displacement matrix, shape (3, 8)
    """
    from xfem_clean.fem.q4 import q4_shape

    N, dN_dxi, dN_deta = q4_shape(xi, eta)

    # Jacobian
    J = np.zeros((2, 2), dtype=float)
    for i in range(4):
        J[0, 0] += dN_dxi[i] * elem_nodes[i, 0]
        J[0, 1] += dN_deta[i] * elem_nodes[i, 0]
        J[1, 0] += dN_dxi[i] * elem_nodes[i, 1]
        J[1, 1] += dN_deta[i] * elem_nodes[i, 1]

    # Inverse Jacobian
    det_J = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    J_inv = np.array([
        [J[1, 1], -J[0, 1]],
        [-J[1, 0], J[0, 0]]
    ]) / det_J

    # Derivatives in physical space
    dN_dx = np.zeros(4, dtype=float)
    dN_dy = np.zeros(4, dtype=float)

    for i in range(4):
        dN_dx[i] = J_inv[0, 0] * dN_dxi[i] + J_inv[0, 1] * dN_deta[i]
        dN_dy[i] = J_inv[1, 0] * dN_dxi[i] + J_inv[1, 1] * dN_deta[i]

    # Build B matrix (3 x 8)
    B = np.zeros((3, 8), dtype=float)

    for i in range(4):
        # Node i has DOFs [u_x, u_y] at indices [2*i, 2*i+1]
        B[0, 2*i] = dN_dx[i]       # εxx: ∂u_x/∂x
        B[1, 2*i+1] = dN_dy[i]     # εyy: ∂u_y/∂y
        B[2, 2*i] = dN_dy[i]       # γxy: ∂u_x/∂y + ∂u_y/∂x
        B[2, 2*i+1] = dN_dx[i]

    return B


def get_element_dofs(elem_conn: np.ndarray, dofs_map: object) -> np.ndarray:
    """Extract global DOF indices for an element.

    Parameters
    ----------
    elem_conn : np.ndarray
        Element connectivity (node indices)
    dofs_map : object
        DOF mapping (must have .std attribute)

    Returns
    -------
    dofs : np.ndarray
        Global DOF indices for element
    """
    dofs = []
    for node in elem_conn:
        dofs.append(int(dofs_map.std[node, 0]))  # u_x
        dofs.append(int(dofs_map.std[node, 1]))  # u_y
    return np.array(dofs, dtype=int)


# =============================================================================
# Utilities
# =============================================================================

def create_straight_reinforcement_layer(
    x_start: np.ndarray,
    x_end: np.ndarray,
    A_s: float,
    E_s: float,
    f_y: float,
    E_h: float,
    d_bar: float,
    layer_type: ReinforcementType = "longitudinal",
    layer_id: int = 0,
    n_segments: int = 1,
) -> ReinforcementLayer:
    """Create a straight reinforcement layer (convenience function).

    Parameters
    ----------
    x_start : np.ndarray
        Start point [x, y]
    x_end : np.ndarray
        End point [x, y]
    A_s : float
        Cross-sectional area [m²]
    E_s : float
        Young's modulus [Pa]
    f_y : float
        Yield stress [Pa]
    E_h : float
        Hardening modulus [Pa]
    d_bar : float
        Bar diameter [m]
    layer_type : ReinforcementType
        Type of reinforcement
    layer_id : int
        Unique identifier
    n_segments : int
        Number of segments to subdivide layer into

    Returns
    -------
    layer : ReinforcementLayer
    """
    segments = []

    for i in range(n_segments):
        alpha0 = i / n_segments
        alpha1 = (i + 1) / n_segments

        x0 = x_start + alpha0 * (x_end - x_start)
        x1 = x_start + alpha1 * (x_end - x_start)

        seg = ReinforcementSegment(
            x0=x0,
            x1=x1,
            A_s=A_s / n_segments,  # Distribute area
            E_s=E_s,
            f_y=f_y,
            E_h=E_h,
            d_bar=d_bar,
            layer_id=layer_id,
        )
        segments.append(seg)

    layer = ReinforcementLayer(
        segments=segments,
        A_total=A_s,
        E_s=E_s,
        f_y=f_y,
        E_h=E_h,
        d_bar=d_bar,
        layer_type=layer_type,
        layer_id=layer_id,
    )

    return layer
