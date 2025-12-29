"""Numerical robustness features for XFEM.

This module implements:
1. Ill-conditioning node removal (Dolbow criterion)
2. Kinked crack tip coordinate transformation
3. Other numerical stabilization techniques

Reference: Dissertation 10.5445/IR/1000124842, Chapter 4
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np


# =============================================================================
# Dolbow Node Removal Criterion
# =============================================================================

def compute_node_removal_criterion(
    node_idx: int,
    nodes: np.ndarray,
    elems: np.ndarray,
    crack: object,
    tol_dolbow: float = 1e-4,
) -> Tuple[bool, float]:
    """Compute Dolbow criterion for node removal.

    A node is removed from enrichment if its support is almost entirely
    on one side of the crack, leading to ill-conditioning.

    Criterion:
        η_i = min(A⁺, A⁻) / (A⁺ + A⁻)

    where A⁺, A⁻ are areas of node support on positive/negative crack sides.

    If η_i < tol (typically 1e-4), remove enrichment at this node.

    Parameters
    ----------
    node_idx : int
        Node index
    nodes : np.ndarray
        Node coordinates
    elems : np.ndarray
        Element connectivity
    crack : object
        Crack geometry (must have phi() method)
    tol_dolbow : float
        Removal tolerance (thesis uses 1e-4)

    Returns
    -------
    should_remove : bool
        True if node should be removed from enrichment
    eta : float
        Criterion value (0 = fully on one side, 0.5 = balanced)
    """
    # Find elements containing this node (node support)
    support_elements = []
    for e_idx in range(elems.shape[0]):
        if node_idx in elems[e_idx]:
            support_elements.append(e_idx)

    if len(support_elements) == 0:
        return False, 0.5

    # Compute areas on each side
    A_plus = 0.0
    A_minus = 0.0

    for e_idx in support_elements:
        elem_conn = elems[e_idx]
        elem_coords = nodes[elem_conn]

        # Subdivide element and compute partial areas
        A_p, A_m = compute_element_partial_areas(elem_coords, crack)

        A_plus += A_p
        A_minus += A_m

    # Compute criterion
    A_total = A_plus + A_minus

    if A_total < 1e-14:
        return False, 0.5

    eta = min(A_plus, A_minus) / A_total

    should_remove = (eta < tol_dolbow)

    return should_remove, float(eta)


def compute_element_partial_areas(
    elem_coords: np.ndarray,
    crack: object,
    n_subdivisions: int = 4,
) -> Tuple[float, float]:
    """Compute element areas on each side of crack.

    Subdivides element into triangles and sums areas based on
    signed distance to crack.

    Parameters
    ----------
    elem_coords : np.ndarray
        Element node coordinates, shape (4, 2)
    crack : object
        Crack geometry
    n_subdivisions : int
        Number of subdivisions per direction

    Returns
    -------
    A_plus : float
        Area on positive side of crack
    A_minus : float
        Area on negative side of crack
    """
    # Simple triangulation: split quad into 2 triangles
    # Triangle 1: nodes 0-1-2
    # Triangle 2: nodes 0-2-3

    triangles = [
        [elem_coords[0], elem_coords[1], elem_coords[2]],
        [elem_coords[0], elem_coords[2], elem_coords[3]],
    ]

    A_plus = 0.0
    A_minus = 0.0

    for tri in triangles:
        # Compute triangle area
        area_tri = compute_triangle_area(tri[0], tri[1], tri[2])

        # Check centroid position relative to crack
        centroid = (tri[0] + tri[1] + tri[2]) / 3.0
        phi_c = crack.phi(centroid[0], centroid[1])

        # Assign area based on centroid side
        if phi_c >= 0.0:
            A_plus += area_tri
        else:
            A_minus += area_tri

    return float(A_plus), float(A_minus)


def compute_triangle_area(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> float:
    """Compute area of triangle.

    Uses cross product formula:
        A = 0.5 * |(p1 - p0) × (p2 - p0)|

    Parameters
    ----------
    p0, p1, p2 : np.ndarray
        Triangle vertices

    Returns
    -------
    area : float
    """
    v1 = p1 - p0
    v2 = p2 - p0

    # 2D cross product magnitude
    cross = v1[0] * v2[1] - v1[1] * v2[0]

    area = 0.5 * abs(cross)

    return float(area)


def remove_ill_conditioned_nodes(
    nodes: np.ndarray,
    elems: np.ndarray,
    crack: object,
    enriched_nodes: np.ndarray,
    tol_dolbow: float = 1e-4,
) -> np.ndarray:
    """Remove ill-conditioned nodes from enrichment set.

    Parameters
    ----------
    nodes : np.ndarray
        Node coordinates
    elems : np.ndarray
        Element connectivity
    crack : object
        Crack geometry
    enriched_nodes : np.ndarray
        Boolean array of nodes currently enriched
    tol_dolbow : float
        Removal tolerance

    Returns
    -------
    enriched_nodes_filtered : np.ndarray
        Boolean array with ill-conditioned nodes removed
    """
    enriched_nodes_filtered = enriched_nodes.copy()

    for node_idx in np.where(enriched_nodes)[0]:
        should_remove, eta = compute_node_removal_criterion(
            node_idx, nodes, elems, crack, tol_dolbow
        )

        if should_remove:
            enriched_nodes_filtered[node_idx] = False

    return enriched_nodes_filtered


# =============================================================================
# Kinked Crack Tip Coordinates
# =============================================================================

def compute_kinked_tip_coordinates(
    x: np.ndarray,
    crack_polyline: List[np.ndarray],
    tip_idx: int,
) -> Tuple[float, float]:
    """Compute local (r, θ) coordinates for kinked crack tip.

    For kinked cracks, the tip enrichment functions assume a locally
    straight crack. This function applies a virtual rotation to align
    the local coordinate system with the crack tip tangent.

    Per dissertation discussion:
        1. Compute tip tangent from last two segments
        2. Rotate coordinates to align with tip tangent
        3. Return (r, θ) in aligned system

    Parameters
    ----------
    x : np.ndarray
        Evaluation point [x, y]
    crack_polyline : List[np.ndarray]
        Crack path as list of points
    tip_idx : int
        Index of crack tip in polyline

    Returns
    -------
    r : float
        Distance from tip
    theta : float
        Angle in tip-aligned coordinates [radians]
    """
    if tip_idx < 1 or tip_idx >= len(crack_polyline):
        raise ValueError("Invalid tip_idx")

    # Tip position
    x_tip = crack_polyline[tip_idx]

    # Compute tip tangent from last segment
    x_prev = crack_polyline[tip_idx - 1]
    t_tip = x_tip - x_prev
    t_norm = np.linalg.norm(t_tip)

    if t_norm < 1e-14:
        # Degenerate segment, use previous one
        if tip_idx >= 2:
            x_prev = crack_polyline[tip_idx - 2]
            t_tip = x_tip - x_prev
            t_norm = np.linalg.norm(t_tip)

    if t_norm < 1e-14:
        # Fully degenerate, use default direction
        t_tip = np.array([0.0, 1.0])
        t_norm = 1.0

    t_tip = t_tip / t_norm

    # Normal vector (90° CCW rotation)
    n_tip = np.array([-t_tip[1], t_tip[0]])

    # Vector from tip to evaluation point
    r_vec = x - x_tip

    # Project onto tip-aligned coordinates
    x_local = np.dot(r_vec, t_tip)
    y_local = np.dot(r_vec, n_tip)

    # Polar coordinates in aligned system
    r = math.sqrt(x_local**2 + y_local**2)
    theta = math.atan2(y_local, x_local)

    return float(r), float(theta)


def tip_coords_kinked(
    crack_polyline: List[np.ndarray],
    tip_idx: int,
    x: np.ndarray,
) -> Tuple[float, float]:
    """Alias for compute_kinked_tip_coordinates (for API consistency).

    Parameters
    ----------
    crack_polyline : List[np.ndarray]
        Crack path points
    tip_idx : int
        Tip index
    x : np.ndarray
        Evaluation point

    Returns
    -------
    r, theta : float
        Polar coordinates in tip-aligned system
    """
    return compute_kinked_tip_coordinates(x, crack_polyline, tip_idx)


# =============================================================================
# Blending Function for Enrichment Transitions
# =============================================================================

def compute_blending_function(
    x: np.ndarray,
    x_center: np.ndarray,
    radius: float,
    blend_type: str = "linear",
) -> float:
    """Compute blending function for smooth enrichment transitions.

    Used to smoothly transition enrichment on/off in a patch to avoid
    discontinuities.

    Parameters
    ----------
    x : np.ndarray
        Evaluation point
    x_center : np.ndarray
        Center of blending region
    radius : float
        Blending radius
    blend_type : str
        Type of blending: "linear", "quadratic", "cubic"

    Returns
    -------
    alpha : float
        Blending weight [0, 1]
    """
    r = np.linalg.norm(x - x_center)

    if r >= radius:
        return 0.0

    # Normalized distance
    xi = r / radius

    if blend_type == "linear":
        alpha = 1.0 - xi
    elif blend_type == "quadratic":
        alpha = (1.0 - xi) ** 2
    elif blend_type == "cubic":
        # C1 continuous cubic
        alpha = 1.0 - 3.0 * xi**2 + 2.0 * xi**3
    else:
        alpha = 1.0 - xi

    return float(alpha)


# =============================================================================
# Conditioning Diagnostics
# =============================================================================

def compute_enrichment_conditioning(
    nodes: np.ndarray,
    elems: np.ndarray,
    dofs: object,
    crack: object,
) -> dict:
    """Compute conditioning diagnostics for enriched system.

    Returns
    -------
    diag : dict
        Dictionary with:
        - n_enriched_nodes: total number of enriched nodes
        - n_ill_conditioned: number of nodes failing Dolbow criterion
        - eta_min: minimum eta value (0 = worst)
        - eta_mean: mean eta value
    """
    if not hasattr(dofs, 'H_nodes'):
        return {
            'n_enriched_nodes': 0,
            'n_ill_conditioned': 0,
            'eta_min': 1.0,
            'eta_mean': 1.0,
        }

    enriched_indices = np.where(dofs.H_nodes)[0]
    n_enriched = len(enriched_indices)

    if n_enriched == 0:
        return {
            'n_enriched_nodes': 0,
            'n_ill_conditioned': 0,
            'eta_min': 1.0,
            'eta_mean': 1.0,
        }

    eta_values = []
    n_ill = 0

    for node_idx in enriched_indices:
        should_remove, eta = compute_node_removal_criterion(
            node_idx, nodes, elems, crack, tol_dolbow=1e-4
        )
        eta_values.append(eta)
        if should_remove:
            n_ill += 1

    return {
        'n_enriched_nodes': n_enriched,
        'n_ill_conditioned': n_ill,
        'eta_min': float(np.min(eta_values)) if len(eta_values) > 0 else 1.0,
        'eta_mean': float(np.mean(eta_values)) if len(eta_values) > 0 else 1.0,
    }


# =============================================================================
# Stabilization Parameters
# =============================================================================

@dataclass
class StabilizationParams:
    """Parameters for numerical stabilization.

    Attributes
    ----------
    use_dolbow_removal : bool
        Enable ill-conditioning node removal
    tol_dolbow : float
        Dolbow criterion tolerance (1e-4 recommended)
    use_kinked_tips : bool
        Use kinked crack tip transformation
    use_blending : bool
        Use blending functions at enrichment boundaries
    blend_radius : float
        Blending radius [m]
    """
    use_dolbow_removal: bool = True
    tol_dolbow: float = 1e-4
    use_kinked_tips: bool = True
    use_blending: bool = False
    blend_radius: float = 0.05
