"""Penalty contact between longitudinal and transverse reinforcement.

This module implements contact at predefined points between reinforcement layers
using the penalty method per dissertation Eq. (4.120-4.129).

Reference: Dissertation 10.5445/IR/1000124842, Chapter 4, Section 4.5.3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import scipy.sparse as sp


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class RebarContactPoint:
    """Contact point between two reinforcement layers.

    Attributes
    ----------
    X_c : np.ndarray
        Contact point position in reference configuration [x, y]
    t_hat : np.ndarray
        Contact direction (unit tangent for slip) [tx, ty]
    k_p : float
        Penalty stiffness [N/m]
    layer_l_id : int
        Longitudinal (or first) layer ID
    layer_t_id : int
        Transverse (or second) layer ID
    node_l : int
        Node index on longitudinal layer (for DOF lookup)
    node_t : int
        Node index on transverse layer (for DOF lookup)
    contact_type : str
        Type of contact: "crossing" (normal contact) or "endpoint" (perfect bonding)
    """
    X_c: np.ndarray
    t_hat: np.ndarray
    k_p: float
    layer_l_id: int
    layer_t_id: int
    node_l: int
    node_t: int
    contact_type: str = "crossing"

    def __post_init__(self):
        """Validate inputs."""
        # Normalize contact direction
        norm = np.linalg.norm(self.t_hat)
        if norm < 1e-14:
            raise ValueError("Contact direction vector must be non-zero")
        self.t_hat = (self.t_hat / norm).astype(float)

        if self.k_p <= 0.0:
            raise ValueError("Penalty stiffness must be positive")


# =============================================================================
# Kinematics (Eq. 4.120-4.126)
# =============================================================================

def compute_tangential_gap(
    u_l: np.ndarray,
    u_t: np.ndarray,
    t_hat: np.ndarray,
) -> float:
    """Compute tangential gap between two reinforcement layers.

    Per Eq. (4.120-4.126):
        x_l = X_c + u_l
        x_t = X_c + u_t
        g = (u_l - u_t) · t̂

    Parameters
    ----------
    u_l : np.ndarray
        Displacement at contact point on longitudinal layer [ux, uy]
    u_t : np.ndarray
        Displacement at contact point on transverse layer [ux, uy]
    t_hat : np.ndarray
        Contact direction (unit tangent) [tx, ty]

    Returns
    -------
    g : float
        Tangential gap (positive = separation in +t direction)
    """
    du = u_l - u_t
    g = float(np.dot(du, t_hat))
    return g


def compute_gap_derivative(
    t_hat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute derivatives of gap w.r.t. displacements.

    δg = (δu_l - δu_t) · t̂

    Parameters
    ----------
    t_hat : np.ndarray
        Contact direction [tx, ty]

    Returns
    -------
    dg_dul : np.ndarray
        ∂g/∂u_l = t̂
    dg_dut : np.ndarray
        ∂g/∂u_t = -t̂
    """
    dg_dul = t_hat.copy()
    dg_dut = -t_hat.copy()
    return dg_dul, dg_dut


# =============================================================================
# Penalty Law (Eq. 4.127)
# =============================================================================

def penalty_contact_law(
    g: float,
    k_p: float,
    contact_type: str = "crossing",
) -> Tuple[float, float]:
    """Penalty contact law.

    Per Eq. (4.127):
        p(g) = k_p * (-g)  if g < 0 (penetration)
               0           if g ≥ 0 (separation)

    For endpoint perfect contact (stirrup legs):
        p(g) = k_p * (-g)  always (bilateral constraint)

    Parameters
    ----------
    g : float
        Tangential gap
    k_p : float
        Penalty stiffness
    contact_type : str
        "crossing" for unilateral contact, "endpoint" for bilateral

    Returns
    -------
    p : float
        Contact pressure
    dp_dg : float
        Derivative dp/dg
    """
    if contact_type == "endpoint":
        # Perfect bonding (bilateral constraint)
        p = -k_p * g
        dp_dg = -k_p
    else:
        # Unilateral contact (only resists penetration)
        if g < 0.0:
            p = -k_p * g  # Positive pressure for negative gap
            dp_dg = -k_p
        else:
            p = 0.0
            dp_dg = 0.0

    return float(p), float(dp_dg)


# =============================================================================
# Assembly (Eq. 4.128-4.129)
# =============================================================================

def assemble_contact_point(
    cp: RebarContactPoint,
    u_total: np.ndarray,
    dof_l: np.ndarray,
    dof_t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble contact contribution for a single contact point.

    Per Eq. (4.128-4.129):
        f_l = -p(g) · N_l^T · t̂
        f_t = +p(g) · N_t^T · t̂

    For simplicity, assume N_l = N_t = 1 (concentrated contact at nodes).

    Tangent matrix (active only if dp/dg ≠ 0):
        K_ll = k_p · (t̂ ⊗ t̂)
        K_tt = k_p · (t̂ ⊗ t̂)
        K_lt = -k_p · (t̂ ⊗ t̂)
        K_tl = -k_p · (t̂ ⊗ t̂)

    Parameters
    ----------
    cp : RebarContactPoint
        Contact point data
    u_total : np.ndarray
        Global displacement vector
    dof_l : np.ndarray
        Global DOF indices for longitudinal point [dof_x, dof_y]
    dof_t : np.ndarray
        Global DOF indices for transverse point [dof_x, dof_y]

    Returns
    -------
    f_local : np.ndarray
        Local force vector [f_l_x, f_l_y, f_t_x, f_t_y]
    K_local : np.ndarray
        Local stiffness matrix (4 x 4)
    """
    # Extract displacements
    u_l = np.array([u_total[dof_l[0]], u_total[dof_l[1]]], dtype=float)
    u_t = np.array([u_total[dof_t[0]], u_total[dof_t[1]]], dtype=float)

    # Compute gap
    g = compute_tangential_gap(u_l, u_t, cp.t_hat)

    # Penalty law
    p, dp_dg = penalty_contact_law(g, cp.k_p, cp.contact_type)

    # Forces (Eq. 4.128)
    # f_l = -p · t̂
    # f_t = +p · t̂
    f_l = -p * cp.t_hat
    f_t = +p * cp.t_hat

    f_local = np.concatenate([f_l, f_t])

    # Tangent (Eq. 4.129)
    # K = dp/dg · (∂g/∂u_l) ⊗ (∂g/∂u_l)
    #   = dp/dg · t̂ ⊗ t̂
    t_outer = np.outer(cp.t_hat, cp.t_hat)
    K_ll = dp_dg * t_outer
    K_tt = dp_dg * t_outer
    K_lt = -dp_dg * t_outer
    K_tl = -dp_dg * t_outer

    # Assemble local 4x4 matrix
    K_local = np.block([
        [K_ll, K_lt],
        [K_tl, K_tt]
    ])

    return f_local, K_local


def assemble_rebar_contact(
    contact_points: List[RebarContactPoint],
    u_total: np.ndarray,
    dofs_map: object,
    ndof_total: int,
) -> Tuple[np.ndarray, sp.csr_matrix]:
    """Assemble global contact force vector and stiffness matrix.

    Parameters
    ----------
    contact_points : List[RebarContactPoint]
        List of contact points
    u_total : np.ndarray
        Global displacement vector
    dofs_map : object
        DOF mapping object (must support indexing for reinforcement DOFs)
    ndof_total : int
        Total number of DOFs

    Returns
    -------
    f_contact : np.ndarray
        Global contact force vector
    K_contact : sp.csr_matrix
        Global contact stiffness matrix
    """
    f_contact = np.zeros(ndof_total, dtype=float)

    rows = []
    cols = []
    data = []

    for cp in contact_points:
        # Get global DOFs for contact point
        # Assumes dofs_map has steel or reinforcement DOF mapping
        # For now, use simplified node-based mapping
        dof_l = get_rebar_dofs_at_node(cp.node_l, dofs_map)
        dof_t = get_rebar_dofs_at_node(cp.node_t, dofs_map)

        if dof_l is None or dof_t is None:
            continue  # Skip if DOFs not available

        # Assemble local contribution
        f_local, K_local = assemble_contact_point(cp, u_total, dof_l, dof_t)

        # Add to global force
        f_contact[dof_l[0]] += f_local[0]
        f_contact[dof_l[1]] += f_local[1]
        f_contact[dof_t[0]] += f_local[2]
        f_contact[dof_t[1]] += f_local[3]

        # Add to global stiffness (sparse triplets)
        global_dofs = np.concatenate([dof_l, dof_t])

        for i in range(4):
            for j in range(4):
                rows.append(global_dofs[i])
                cols.append(global_dofs[j])
                data.append(K_local[i, j])

    # Build sparse matrix
    if len(data) > 0:
        K_contact = sp.csr_matrix((data, (rows, cols)), shape=(ndof_total, ndof_total))
    else:
        K_contact = sp.csr_matrix((ndof_total, ndof_total), dtype=float)

    return f_contact, K_contact


def get_rebar_dofs_at_node(
    node_idx: int,
    dofs_map: object,
) -> Optional[np.ndarray]:
    """Get reinforcement DOF indices at a node.

    Parameters
    ----------
    node_idx : int
        Node index
    dofs_map : object
        DOF mapping object (should have .steel or .rebar attribute)

    Returns
    -------
    dofs : np.ndarray or None
        [dof_x, dof_y] or None if not available
    """
    # Try steel DOFs first (for bond-slip compatibility)
    if hasattr(dofs_map, 'steel') and dofs_map.steel is not None:
        if dofs_map.steel[node_idx, 0] >= 0:
            return np.array([
                dofs_map.steel[node_idx, 0],
                dofs_map.steel[node_idx, 1]
            ], dtype=int)

    # Try standard DOFs as fallback
    if hasattr(dofs_map, 'std'):
        return np.array([
            dofs_map.std[node_idx, 0],
            dofs_map.std[node_idx, 1]
        ], dtype=int)

    return None


# =============================================================================
# Utilities
# =============================================================================

def create_stirrup_contact_points(
    long_nodes: np.ndarray,
    trans_segments: List[Tuple[int, int]],
    nodes: np.ndarray,
    k_p: float,
    layer_l_id: int = 0,
    layer_t_id: int = 1,
) -> List[RebarContactPoint]:
    """Create contact points for stirrup-longitudinal bar interactions.

    Automatically detects crossings and creates contact points.

    Parameters
    ----------
    long_nodes : np.ndarray
        Node indices of longitudinal bars
    trans_segments : List[Tuple[int, int]]
        Transverse (stirrup) segments as (n1, n2) pairs
    nodes : np.ndarray
        Node coordinates [nnode, 2]
    k_p : float
        Penalty stiffness
    layer_l_id, layer_t_id : int
        Layer identifiers

    Returns
    -------
    contact_points : List[RebarContactPoint]
    """
    contact_points = []

    for trans_seg in trans_segments:
        n_t1, n_t2 = trans_seg
        x_t1 = nodes[n_t1]
        x_t2 = nodes[n_t2]

        # Transverse segment direction
        t_trans = x_t2 - x_t1
        t_trans = t_trans / np.linalg.norm(t_trans)

        # Check crossings with longitudinal bars
        for n_l in long_nodes:
            x_l = nodes[n_l]

            # Check if longitudinal bar crosses transverse segment
            # (Simplified: assume stirrup is vertical/horizontal)
            # Full implementation would do geometric intersection

            # For now, check if longitudinal node is near transverse segment
            # (within tolerance)
            dist = point_to_segment_distance(x_l, x_t1, x_t2)

            if dist < 0.01:  # 1 cm tolerance
                # Create contact point
                # Use transverse direction as contact direction
                cp = RebarContactPoint(
                    X_c=x_l.copy(),
                    t_hat=t_trans.copy(),
                    k_p=k_p,
                    layer_l_id=layer_l_id,
                    layer_t_id=layer_t_id,
                    node_l=n_l,
                    node_t=n_t1,  # Use first node of transverse segment
                    contact_type="crossing",
                )
                contact_points.append(cp)

        # Add endpoint constraints (stirrup leg endpoints)
        # Per thesis: stirrup legs have "perfect contact" with longitudinal bars
        for endpoint in [n_t1, n_t2]:
            x_ep = nodes[endpoint]

            # Find nearest longitudinal bar node
            min_dist = float('inf')
            nearest_long = None

            for n_l in long_nodes:
                x_l = nodes[n_l]
                dist = np.linalg.norm(x_ep - x_l)
                if dist < min_dist:
                    min_dist = dist
                    nearest_long = n_l

            if nearest_long is not None and min_dist < 0.02:  # 2 cm tolerance
                # Create endpoint perfect contact
                cp = RebarContactPoint(
                    X_c=x_ep.copy(),
                    t_hat=t_trans.copy(),
                    k_p=k_p * 10.0,  # Higher penalty for perfect bonding
                    layer_l_id=layer_l_id,
                    layer_t_id=layer_t_id,
                    node_l=nearest_long,
                    node_t=endpoint,
                    contact_type="endpoint",
                )
                contact_points.append(cp)

    return contact_points


def point_to_segment_distance(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """Compute minimum distance from point to line segment.

    Parameters
    ----------
    p : np.ndarray
        Point [x, y]
    a, b : np.ndarray
        Segment endpoints

    Returns
    -------
    dist : float
        Minimum distance
    """
    ab = b - a
    ap = p - a

    L2 = np.dot(ab, ab)

    if L2 < 1e-14:
        return float(np.linalg.norm(ap))

    # Parameter t ∈ [0, 1] for projection onto segment
    t = np.dot(ap, ab) / L2
    t = max(0.0, min(1.0, t))

    # Projection point
    proj = a + t * ab

    # Distance
    dist = float(np.linalg.norm(p - proj))

    return dist


# =============================================================================
# Diagnostics
# =============================================================================

def compute_contact_diagnostics(
    contact_points: List[RebarContactPoint],
    u_total: np.ndarray,
    dofs_map: object,
) -> dict:
    """Compute diagnostic information about contact state.

    Parameters
    ----------
    contact_points : List[RebarContactPoint]
    u_total : np.ndarray
    dofs_map : object

    Returns
    -------
    diag : dict
        Dictionary with diagnostic info:
        - n_active: number of active (penetrating) contacts
        - max_gap: maximum gap (absolute value)
        - max_pressure: maximum contact pressure
    """
    n_active = 0
    max_gap = 0.0
    max_pressure = 0.0

    for cp in contact_points:
        dof_l = get_rebar_dofs_at_node(cp.node_l, dofs_map)
        dof_t = get_rebar_dofs_at_node(cp.node_t, dofs_map)

        if dof_l is None or dof_t is None:
            continue

        u_l = np.array([u_total[dof_l[0]], u_total[dof_l[1]]])
        u_t = np.array([u_total[dof_t[0]], u_total[dof_t[1]]])

        g = compute_tangential_gap(u_l, u_t, cp.t_hat)
        p, _ = penalty_contact_law(g, cp.k_p, cp.contact_type)

        if abs(p) > 1e-9:
            n_active += 1

        max_gap = max(max_gap, abs(g))
        max_pressure = max(max_pressure, abs(p))

    return {
        'n_active': n_active,
        'max_gap': max_gap,
        'max_pressure': max_pressure,
    }
