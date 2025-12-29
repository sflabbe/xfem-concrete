"""Crack junction enrichment for crack coalescence.

This module implements junction enrichment when a propagating crack overlaps
or intersects another crack inside an element per dissertation Eq. (4.64-4.66).

Reference: Dissertation 10.5445/IR/1000124842, Chapter 4, Section 4.3.3
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class CrackJunction:
    """Junction between two cracks.

    Attributes
    ----------
    junction_point : np.ndarray
        Intersection point in global coordinates [x, y]
    main_crack_id : int
        ID of the main (existing) crack
    secondary_crack_id : int
        ID of the secondary (approaching) crack
    element_id : int
        Element containing the junction
    branch_angles : List[float]
        Angles of crack branches at junction [radians]
    active : bool
        Whether junction enrichment is active
    """
    junction_point: np.ndarray
    main_crack_id: int
    secondary_crack_id: int
    element_id: int
    branch_angles: List[float]
    active: bool = True


# =============================================================================
# Junction Detection (Geometry)
# =============================================================================

def detect_crack_coalescence(
    cracks: List[object],  # List of XFEMCrack or similar
    nodes: np.ndarray,
    elems: np.ndarray,
    tol_merge: float = 0.01,
) -> List[CrackJunction]:
    """Detect crack coalescence events.

    A coalescence occurs when:
    1. Tip of crack B reaches within tol_merge of crack A
    2. Intersection point lies inside the same element (or patch)

    Parameters
    ----------
    cracks : List[object]
        List of crack objects (must have .tip_x, .tip_y, and path geometry)
    nodes : np.ndarray
        Node coordinates [nnode, 2]
    elems : np.ndarray
        Element connectivity [nelem, 4]
    tol_merge : float
        Merge tolerance [m]

    Returns
    -------
    junctions : List[CrackJunction]
        List of detected junctions
    """
    junctions = []

    for i, crack_a in enumerate(cracks):
        if not crack_a.active:
            continue

        for j, crack_b in enumerate(cracks):
            if i == j or not crack_b.active:
                continue

            # Check if tip of crack_b is near crack_a path
            tip_b = np.array([crack_b.tip_x, crack_b.tip_y])

            # Compute distance from tip_b to crack_a centerline
            dist, nearest_point = distance_to_crack_path(tip_b, crack_a)

            if dist < tol_merge:
                # Find element containing the intersection
                elem_id = find_element_containing_point(nearest_point, nodes, elems)

                if elem_id is not None:
                    # Compute branch angles at junction
                    angle_a = get_crack_angle_at_point(crack_a, nearest_point)
                    angle_b = get_crack_angle_at_point(crack_b, tip_b)

                    junction = CrackJunction(
                        junction_point=nearest_point,
                        main_crack_id=i,
                        secondary_crack_id=j,
                        element_id=elem_id,
                        branch_angles=[angle_a, angle_b],
                        active=True,
                    )

                    junctions.append(junction)

    return junctions


def distance_to_crack_path(
    point: np.ndarray,
    crack: object,
) -> Tuple[float, np.ndarray]:
    """Compute minimum distance from point to crack path.

    Parameters
    ----------
    point : np.ndarray
        Query point [x, y]
    crack : object
        Crack object (must have p0(), pt(), and phi() methods)

    Returns
    -------
    dist : float
        Minimum distance
    nearest_point : np.ndarray
        Nearest point on crack path
    """
    # For straight crack: use perpendicular distance
    p0 = crack.p0()
    pt = crack.pt()

    # Project point onto crack line
    v = pt - p0
    L = np.linalg.norm(v)

    if L < 1e-14:
        return float(np.linalg.norm(point - p0)), p0

    t_crack = v / L
    n_crack = np.array([-t_crack[1], t_crack[0]])

    # Projection parameter
    s = np.dot(point - p0, t_crack)

    # Clamp to crack segment
    s = max(0.0, min(L, s))

    # Nearest point
    nearest = p0 + s * t_crack

    # Distance
    dist = float(np.linalg.norm(point - nearest))

    return dist, nearest


def find_element_containing_point(
    point: np.ndarray,
    nodes: np.ndarray,
    elems: np.ndarray,
) -> Optional[int]:
    """Find element containing a given point.

    Parameters
    ----------
    point : np.ndarray
        Query point [x, y]
    nodes : np.ndarray
        Node coordinates
    elems : np.ndarray
        Element connectivity

    Returns
    -------
    elem_id : int or None
        Element index or None if not found
    """
    for e_idx in range(elems.shape[0]):
        elem_conn = elems[e_idx]
        elem_nodes_coords = nodes[elem_conn]

        # Simple bounding box test
        xmin = elem_nodes_coords[:, 0].min()
        xmax = elem_nodes_coords[:, 0].max()
        ymin = elem_nodes_coords[:, 1].min()
        ymax = elem_nodes_coords[:, 1].max()

        if (xmin - 1e-6 <= point[0] <= xmax + 1e-6 and
            ymin - 1e-6 <= point[1] <= ymax + 1e-6):
            return e_idx

    return None


def get_crack_angle_at_point(crack: object, point: np.ndarray) -> float:
    """Get crack tangent angle at a given point.

    Parameters
    ----------
    crack : object
        Crack object
    point : np.ndarray
        Point on crack

    Returns
    -------
    angle : float
        Angle in radians (0 = horizontal, π/2 = vertical)
    """
    # For straight crack, use overall direction
    t = crack.tvec()
    angle = math.atan2(t[1], t[0])
    return float(angle)


# =============================================================================
# Junction Enrichment Functions (Eq. 4.64-4.66)
# =============================================================================

def compute_junction_heaviside(
    x: np.ndarray,
    junction: CrackJunction,
    crack_k: object,
) -> float:
    """Compute Heaviside function for branch k at junction.

    Per Eq. (4.64):
        H_k(x) = sign(φ_k(x))

    where φ_k is the signed distance to branch k.

    Parameters
    ----------
    x : np.ndarray
        Evaluation point [x, y]
    junction : CrackJunction
        Junction data
    crack_k : object
        Crack object for branch k

    Returns
    -------
    H_k : float
        Heaviside value (+1 or -1)
    """
    phi_k = crack_k.phi(x[0], x[1])
    H_k = 1.0 if phi_k >= 0.0 else -1.0
    return float(H_k)


def compute_shifted_junction_heaviside(
    x: np.ndarray,
    x_i: np.ndarray,
    junction: CrackJunction,
    crack_k: object,
) -> float:
    """Compute shifted Heaviside enrichment for junction.

    Per Eq. (4.65):
        H̃_k(x) = H_k(x) - H_k(x_i)

    Parameters
    ----------
    x : np.ndarray
        Evaluation point
    x_i : np.ndarray
        Node position
    junction : CrackJunction
    crack_k : object

    Returns
    -------
    H_tilde_k : float
        Shifted Heaviside value
    """
    H_x = compute_junction_heaviside(x, junction, crack_k)
    H_i = compute_junction_heaviside(x_i, junction, crack_k)

    return H_x - H_i


def get_junction_enriched_nodes(
    junction: CrackJunction,
    nodes: np.ndarray,
    elems: np.ndarray,
    patch_radius: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Identify nodes enriched by junction.

    Per Eq. (4.64):
        N_J1 = nodes whose support contains the junction (for branch 1)
        N_J2 = nodes whose support contains the junction (for branch 2)

    Parameters
    ----------
    junction : CrackJunction
    nodes : np.ndarray
    elems : np.ndarray
    patch_radius : float
        Radius for junction patch [m]

    Returns
    -------
    nodes_j1 : np.ndarray
        Node indices for branch 1 enrichment
    nodes_j2 : np.ndarray
        Node indices for branch 2 enrichment
    """
    # Find nodes within patch around junction
    X_j = junction.junction_point

    distances = np.linalg.norm(nodes - X_j[np.newaxis, :], axis=1)
    nodes_in_patch = np.where(distances <= patch_radius)[0]

    # For simplicity, use same node set for both branches
    # Full implementation might split based on crack orientation
    nodes_j1 = nodes_in_patch
    nodes_j2 = nodes_in_patch

    return nodes_j1, nodes_j2


# =============================================================================
# Junction Topology Management
# =============================================================================

def arrest_secondary_crack_at_junction(
    junction: CrackJunction,
    cracks: List[object],
) -> None:
    """Arrest secondary crack at junction and disable its tip enrichment.

    When coalescence occurs:
    1. Stop propagation of secondary crack
    2. Remove/disable its tip enrichment in patch
    3. Add junction enrichment

    Parameters
    ----------
    junction : CrackJunction
    cracks : List[object]
        List of crack objects (will be modified)
    """
    sec_crack = cracks[junction.secondary_crack_id]

    # Move tip to junction point
    sec_crack.tip_x = float(junction.junction_point[0])
    sec_crack.tip_y = float(junction.junction_point[1])

    # Mark crack as arrested (prevent further propagation)
    if hasattr(sec_crack, 'arrested'):
        sec_crack.arrested = True

    # Note: Actual tip enrichment removal happens during DOF rebuild


def compute_junction_displacement_approximation(
    x: np.ndarray,
    node_coords: np.ndarray,
    q_global: np.ndarray,
    junction: CrackJunction,
    crack_1: object,
    crack_2: object,
    dofs_junction: object,
) -> np.ndarray:
    """Compute displacement field with junction enrichment.

    Per Eq. (4.66):
        u(x) = Σ N_i u_i
             + Σ_{i∈N_J1} N_i H̃_1(x) a_i
             + Σ_{i∈N_J2} N_i H̃_2(x) b_i

    Parameters
    ----------
    x : np.ndarray
        Evaluation point
    node_coords : np.ndarray
        Node coordinates for element
    q_global : np.ndarray
        Global DOF vector
    junction : CrackJunction
    crack_1, crack_2 : object
        Crack objects for branches
    dofs_junction : object
        DOF mapping with junction enrichment

    Returns
    -------
    u : np.ndarray
        Displacement at x
    """
    # Placeholder for full implementation
    # Requires proper shape functions and DOF extraction
    u = np.zeros(2, dtype=float)

    # TODO: Implement full junction displacement field

    return u


# =============================================================================
# Utilities
# =============================================================================

def visualize_junction(
    junction: CrackJunction,
    cracks: List[object],
) -> dict:
    """Create visualization data for junction.

    Parameters
    ----------
    junction : CrackJunction
    cracks : List[object]

    Returns
    -------
    viz_data : dict
        Dictionary with visualization info
    """
    return {
        'junction_point': junction.junction_point.tolist(),
        'main_crack_id': junction.main_crack_id,
        'secondary_crack_id': junction.secondary_crack_id,
        'branch_angles_deg': [math.degrees(a) for a in junction.branch_angles],
        'element_id': junction.element_id,
    }
