"""Energy dissipation computation and tracking for XFEM concrete analysis.

This module provides tools to compute and track energy dissipation through:
  - Plastic deformation (bulk material)
  - Tension fracture (bulk material smeared cracking)
  - Compression crushing (bulk material damage)
  - Cohesive crack opening (discrete cracks)
  - Steel reinforcement plasticity

These energy measures are essential for:
  - Validating numerical accuracy
  - Understanding failure mechanisms
  - Comparing fracture vs plastic contributions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import math

import numpy as np

try:
    from xfem_clean.xfem.state_arrays import BulkStateArrays
    from xfem_clean.cohesive_laws import CohesiveLaw
except ImportError:
    # Graceful degradation if imports fail
    BulkStateArrays = Any
    CohesiveLaw = Any


@dataclass
class EnergyBalance:
    """Energy balance for a single analysis step.

    All energies in Joules [J].

    Attributes
    ----------
    W_plastic : float
        Plastic dissipation in bulk material
    W_fract_tension : float
        Tension fracture energy (smeared cracking in bulk)
    W_fract_compression : float
        Compression crushing energy (bulk damage)
    W_cohesive : float
        Cohesive crack opening energy (discrete cracks)
    W_steel_plastic : float
        Plastic dissipation in steel reinforcement
    W_bond_slip : float
        Bond-slip interface dissipation
    W_total : float
        Total dissipated energy
    W_external : float
        External work (load × displacement)
    W_elastic : float
        Stored elastic energy
    energy_error : float
        Energy balance error: W_external - (W_elastic + W_total)
    """

    W_plastic: float = 0.0
    W_fract_tension: float = 0.0
    W_fract_compression: float = 0.0
    W_cohesive: float = 0.0
    W_steel_plastic: float = 0.0
    W_bond_slip: float = 0.0
    W_total: float = 0.0
    W_external: float = 0.0
    W_elastic: float = 0.0
    energy_error: float = 0.0

    def compute_total(self) -> float:
        """Compute total dissipated energy."""
        self.W_total = (
            self.W_plastic
            + self.W_fract_tension
            + self.W_fract_compression
            + self.W_cohesive
            + self.W_steel_plastic
            + self.W_bond_slip
        )
        return self.W_total

    def compute_error(self) -> float:
        """Compute energy balance error."""
        self.energy_error = self.W_external - (self.W_elastic + self.W_total)
        return self.energy_error

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for export."""
        return {
            "W_plastic": self.W_plastic,
            "W_fract_tension": self.W_fract_tension,
            "W_fract_compression": self.W_fract_compression,
            "W_cohesive": self.W_cohesive,
            "W_steel_plastic": self.W_steel_plastic,
            "W_bond_slip": self.W_bond_slip,
            "W_total": self.W_total,
            "W_external": self.W_external,
            "W_elastic": self.W_elastic,
            "energy_error": self.energy_error,
        }

    def __repr__(self) -> str:
        """Pretty-print energy balance."""
        lines = [
            "Energy Balance [J]:",
            f"  Plastic (bulk):        {self.W_plastic:12.6e}",
            f"  Fracture (tension):    {self.W_fract_tension:12.6e}",
            f"  Crushing (compression):{self.W_fract_compression:12.6e}",
            f"  Cohesive (cracks):     {self.W_cohesive:12.6e}",
            f"  Steel plasticity:      {self.W_steel_plastic:12.6e}",
            f"  Bond-slip:             {self.W_bond_slip:12.6e}",
            f"  ────────────────────────────────────",
            f"  Total dissipated:      {self.W_total:12.6e}",
            f"  Elastic stored:        {self.W_elastic:12.6e}",
            f"  External work:         {self.W_external:12.6e}",
            f"  Balance error:         {self.energy_error:12.6e}",
        ]
        return "\n".join(lines)


def compute_bulk_energies(
    mp_states: BulkStateArrays,
    elems: np.ndarray,
    nodes: np.ndarray,
    thickness: float = 1.0,
) -> Dict[str, float]:
    """Compute bulk material energy dissipation from integration point states.

    Parameters
    ----------
    mp_states : BulkStateArrays
        Material point state arrays
    elems : np.ndarray
        Element connectivity [n_elem, n_nodes_per_elem]
    nodes : np.ndarray
        Node coordinates [n_nodes, ndim]
    thickness : float
        Out-of-plane thickness [m]

    Returns
    -------
    energies : dict
        {
            "W_plastic": float,
            "W_fract_t": float,
            "W_fract_c": float,
        }
    """
    n_elem = mp_states.eps.shape[0]
    n_ip_max = mp_states.eps.shape[1]

    W_plastic = 0.0
    W_fract_t = 0.0
    W_fract_c = 0.0

    for ie in range(n_elem):
        # Compute element volume (simplified: assume Q4 elements)
        elem_nodes = elems[ie]
        coords = nodes[elem_nodes]

        # Area via shoelace formula (assumes convex quad)
        if len(coords) == 4:
            x = coords[:, 0]
            y = coords[:, 1]
            area = 0.5 * abs(
                (x[0] - x[2]) * (y[1] - y[3]) - (x[1] - x[3]) * (y[0] - y[2])
            )
        else:
            # Fallback: approximate
            area = 0.0
            for i in range(len(coords)):
                j = (i + 1) % len(coords)
                area += coords[i, 0] * coords[j, 1]
                area -= coords[j, 0] * coords[i, 1]
            area = abs(area) / 2.0

        volume_elem = area * thickness

        # Integrate over IPs (equal weight assumption for simplicity)
        for ip in range(n_ip_max):
            mp = mp_states.get_mp(ie, ip)

            # Energy densities [J/m³]
            w_pl = float(getattr(mp, "w_plastic", 0.0))
            w_ft = float(getattr(mp, "w_fract_t", 0.0))
            w_fc = float(getattr(mp, "w_fract_c", 0.0))

            # Volume per IP (uniform distribution)
            dV = volume_elem / float(n_ip_max)

            W_plastic += w_pl * dV
            W_fract_t += w_ft * dV
            W_fract_c += w_fc * dV

    return {
        "W_plastic": float(W_plastic),
        "W_fract_t": float(W_fract_t),
        "W_fract_c": float(W_fract_c),
    }


def compute_cohesive_energy(
    coh_states: Any,  # CohesiveStateArrays
    cohesive_law: Optional[CohesiveLaw] = None,
    crack_lengths: Optional[np.ndarray] = None,
    thickness: float = 1.0,
) -> float:
    """Compute energy dissipated in cohesive crack opening.

    Parameters
    ----------
    coh_states : CohesiveStateArrays
        Cohesive state arrays (damage, delta_max)
    cohesive_law : CohesiveLaw
        Cohesive traction-separation law (to integrate dissipation)
    crack_lengths : np.ndarray
        Length of each cohesive segment [n_coh_segments]
    thickness : float
        Out-of-plane thickness [m]

    Returns
    -------
    W_cohesive : float
        Total cohesive energy [J]
    """
    if not hasattr(coh_states, "damage") or not hasattr(coh_states, "delta_max"):
        return 0.0

    W_cohesive = 0.0

    # Iterate over all cohesive integration points
    # Shape: (n_cracks, n_elements, n_gauss_points)
    damage_arr = coh_states.damage
    delta_max_arr = coh_states.delta_max

    n_cracks = damage_arr.shape[0]
    n_elem = damage_arr.shape[1] if damage_arr.ndim > 1 else 1
    n_gp = damage_arr.shape[2] if damage_arr.ndim > 2 else 1

    for ic in range(n_cracks):
        for ie in range(n_elem):
            for ip in range(n_gp):
                dmg = float(damage_arr[ic, ie, ip]) if damage_arr.ndim == 3 else float(damage_arr[ic, ie])
                delta_max = float(delta_max_arr[ic, ie, ip]) if delta_max_arr.ndim == 3 else float(delta_max_arr[ic, ie])

                # Energy per unit area (assuming bilinear law)
                if cohesive_law is not None and hasattr(cohesive_law, "Gf"):
                    # Exact: integrate traction-separation
                    Gf = float(cohesive_law.Gf)
                    psi_surf = dmg * Gf  # Energy per unit area [J/m²]
                else:
                    # Fallback: use damage as proxy (assumes Gf ~ 1)
                    psi_surf = delta_max * dmg * 1e6  # Rough estimate

                # Area of cohesive segment (requires crack geometry)
                if crack_lengths is not None and len(crack_lengths) > ic:
                    dL = float(crack_lengths[ic]) / max(n_elem, 1)
                else:
                    dL = 0.01  # Fallback: assume 1cm segments

                dA = dL * thickness
                W_cohesive += psi_surf * dA

    return float(W_cohesive)


def compute_external_work(
    P_ext: np.ndarray,
    u: np.ndarray,
) -> float:
    """Compute external work done by applied loads.

    Parameters
    ----------
    P_ext : np.ndarray
        External force vector [n_dof]
    u : np.ndarray
        Displacement vector [n_dof]

    Returns
    -------
    W_ext : float
        External work [J]
    """
    return float(np.dot(P_ext, u))


def compute_elastic_energy(
    K: Any,  # sparse matrix or ndarray
    u: np.ndarray,
) -> float:
    """Compute stored elastic energy.

    Parameters
    ----------
    K : sparse matrix or ndarray
        Stiffness matrix
    u : np.ndarray
        Displacement vector

    Returns
    -------
    W_elastic : float
        Elastic strain energy [J]
    """
    Ku = K @ u
    return 0.5 * float(np.dot(u, Ku))


def compute_global_energies(
    mp_states: Optional[BulkStateArrays] = None,
    coh_states: Optional[Any] = None,
    elems: Optional[np.ndarray] = None,
    nodes: Optional[np.ndarray] = None,
    thickness: float = 1.0,
    cohesive_law: Optional[CohesiveLaw] = None,
    crack_lengths: Optional[np.ndarray] = None,
    P_ext: Optional[np.ndarray] = None,
    u: Optional[np.ndarray] = None,
    K: Optional[Any] = None,
) -> EnergyBalance:
    """Compute comprehensive energy balance for XFEM analysis.

    Parameters
    ----------
    mp_states : BulkStateArrays
        Bulk material point states
    coh_states : CohesiveStateArrays
        Cohesive crack states
    elems : np.ndarray
        Element connectivity
    nodes : np.ndarray
        Node coordinates
    thickness : float
        Out-of-plane thickness
    cohesive_law : CohesiveLaw
        Cohesive law for energy integration
    crack_lengths : np.ndarray
        Crack segment lengths
    P_ext : np.ndarray
        External force vector
    u : np.ndarray
        Displacement vector
    K : sparse matrix
        Tangent stiffness matrix

    Returns
    -------
    energy_balance : EnergyBalance
        Complete energy balance
    """
    eb = EnergyBalance()

    # Bulk material energies
    if mp_states is not None and elems is not None and nodes is not None:
        bulk_e = compute_bulk_energies(mp_states, elems, nodes, thickness)
        eb.W_plastic = bulk_e["W_plastic"]
        eb.W_fract_tension = bulk_e["W_fract_t"]
        eb.W_fract_compression = bulk_e["W_fract_c"]

    # Cohesive energies
    if coh_states is not None:
        eb.W_cohesive = compute_cohesive_energy(
            coh_states, cohesive_law, crack_lengths, thickness
        )

    # External work
    if P_ext is not None and u is not None:
        eb.W_external = compute_external_work(P_ext, u)

    # Elastic energy
    if K is not None and u is not None:
        eb.W_elastic = compute_elastic_energy(K, u)

    # Compute totals
    eb.compute_total()
    eb.compute_error()

    return eb


def energy_time_series(
    energy_history: list[EnergyBalance],
) -> Dict[str, np.ndarray]:
    """Convert energy balance history to time series arrays.

    Parameters
    ----------
    energy_history : list of EnergyBalance
        Energy balance at each analysis step

    Returns
    -------
    time_series : dict
        {
            "W_plastic": np.ndarray [n_steps],
            "W_fract_tension": np.ndarray,
            ...
        }
    """
    n_steps = len(energy_history)
    keys = [
        "W_plastic",
        "W_fract_tension",
        "W_fract_compression",
        "W_cohesive",
        "W_steel_plastic",
        "W_bond_slip",
        "W_total",
        "W_external",
        "W_elastic",
        "energy_error",
    ]

    time_series = {key: np.zeros(n_steps) for key in keys}

    for i, eb in enumerate(energy_history):
        for key in keys:
            time_series[key][i] = getattr(eb, key, 0.0)

    return time_series


def plot_energy_evolution(
    energy_history: list[EnergyBalance],
    time_or_steps: Optional[np.ndarray] = None,
    filename: Optional[str] = None,
):
    """Plot energy evolution over analysis steps.

    Parameters
    ----------
    energy_history : list of EnergyBalance
    time_or_steps : np.ndarray
        Time or step indices (x-axis)
    filename : str
        Save figure to file (requires matplotlib)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available. Cannot plot energy evolution.")
        return

    ts = energy_time_series(energy_history)
    n_steps = len(energy_history)

    if time_or_steps is None:
        time_or_steps = np.arange(n_steps)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Dissipation components
    ax1.plot(time_or_steps, ts["W_plastic"], label="Plastic", linewidth=2)
    ax1.plot(time_or_steps, ts["W_fract_tension"], label="Tension Fracture", linewidth=2)
    ax1.plot(time_or_steps, ts["W_fract_compression"], label="Compression Crushing", linewidth=2)
    ax1.plot(time_or_steps, ts["W_cohesive"], label="Cohesive Crack", linewidth=2)
    ax1.plot(time_or_steps, ts["W_total"], "k--", label="Total Dissipated", linewidth=2)
    ax1.set_ylabel("Energy [J]")
    ax1.set_title("Energy Dissipation Evolution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Energy balance
    ax2.plot(time_or_steps, ts["W_external"], label="External Work", linewidth=2)
    ax2.plot(time_or_steps, ts["W_elastic"], label="Elastic Stored", linewidth=2)
    ax2.plot(time_or_steps, ts["W_total"], label="Dissipated", linewidth=2)
    ax2.plot(time_or_steps, ts["energy_error"], "r:", label="Balance Error", linewidth=2)
    ax2.set_xlabel("Step / Time")
    ax2.set_ylabel("Energy [J]")
    ax2.set_title("Energy Balance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Energy plot saved to: {filename}")
    else:
        plt.show()
