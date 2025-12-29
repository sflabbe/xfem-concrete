"""
Post-processing utilities for Gutiérrez thesis cases.

Provides standard metrics and plots for all thesis examples:
- CTOD (crack tip opening displacement)
- Crack widths along cracks
- Slip profiles along reinforcement
- Bond stress profiles along reinforcement
- Axial force in rebars
- Base moment and reactions (for walls)
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json
import csv


@dataclass
class CTODMeasurement:
    """Crack tip opening displacement measurement"""
    node_pair: Tuple[int, int]  # Node IDs on either side of crack
    ctod: float  # Opening displacement (mm)
    x: float  # x-coordinate of measurement
    y: float  # y-coordinate of measurement


@dataclass
class CrackWidth:
    """Crack width measurement along a crack"""
    crack_id: int
    x: float
    y: float
    width: float  # Opening (mm)
    angle_deg: float  # Crack orientation


@dataclass
class SlipProfile:
    """Slip distribution along a reinforcement element"""
    rebar_id: int
    x_coords: np.ndarray  # Positions along bar
    slip_values: np.ndarray  # Slip at each position (mm)
    tau_values: np.ndarray  # Bond stress at each position (MPa)


@dataclass
class SteelForce:
    """Axial force distribution in rebar"""
    rebar_id: int
    x_coords: np.ndarray
    force_values: np.ndarray  # Axial force (N)
    stress_values: np.ndarray  # Axial stress (MPa)


@dataclass
class ThesisMetrics:
    """Standard metrics for thesis case validation"""
    # Global response
    peak_load: float  # Maximum load (N)
    peak_displacement: float  # Displacement at peak (mm)
    ultimate_displacement: float  # Displacement at failure (mm)

    # Energy dissipation
    total_energy: float  # Total work (N·mm)
    fracture_energy: float  # Crack energy (N·mm)
    plastic_energy: float  # Plastic dissipation (N·mm)
    bond_slip_energy: Optional[float] = None  # Bond-slip dissipation (N·mm)

    # Crack pattern
    n_cracks: int = 0  # Number of cracks
    total_crack_length: float = 0.0  # Total crack length (mm)
    average_crack_spacing: Optional[float] = None  # For multiple cracks (mm)

    # Additional metrics (case-specific)
    ctod_at_peak: Optional[float] = None  # CTOD at peak load (mm)
    base_moment_peak: Optional[float] = None  # Base moment at peak (N·mm, for walls)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export"""
        return {
            "peak_load": self.peak_load,
            "peak_displacement": self.peak_displacement,
            "ultimate_displacement": self.ultimate_displacement,
            "total_energy": self.total_energy,
            "fracture_energy": self.fracture_energy,
            "plastic_energy": self.plastic_energy,
            "bond_slip_energy": self.bond_slip_energy,
            "n_cracks": self.n_cracks,
            "total_crack_length": self.total_crack_length,
            "average_crack_spacing": self.average_crack_spacing,
            "ctod_at_peak": self.ctod_at_peak,
            "base_moment_peak": self.base_moment_peak,
        }

    def save_json(self, filepath: str):
        """Save metrics to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# CTOD Computation
# =============================================================================

def compute_ctod(
    node_coords: np.ndarray,
    displacement: np.ndarray,
    node_pair: Tuple[int, int],
) -> CTODMeasurement:
    """
    Compute crack tip opening displacement between two nodes.

    Parameters
    ----------
    node_coords : ndarray, shape (n_nodes, 2)
        Node coordinates
    displacement : ndarray, shape (n_nodes, 2)
        Displacement field
    node_pair : tuple of int
        (node_id_left, node_id_right) on either side of crack tip

    Returns
    -------
    CTODMeasurement
        CTOD measurement
    """
    n1, n2 = node_pair

    # Current positions
    x1 = node_coords[n1] + displacement[n1]
    x2 = node_coords[n2] + displacement[n2]

    # Distance between nodes
    ctod = np.linalg.norm(x2 - x1)

    # Measurement location (midpoint)
    x_mid = 0.5 * (x1 + x2)

    return CTODMeasurement(
        node_pair=node_pair,
        ctod=ctod,
        x=x_mid[0],
        y=x_mid[1],
    )


def compute_ctod_time_series(
    node_coords: np.ndarray,
    displacement_history: List[np.ndarray],
    node_pair: Tuple[int, int],
) -> np.ndarray:
    """
    Compute CTOD time series.

    Returns
    -------
    ctod_series : ndarray, shape (n_steps,)
        CTOD at each load step
    """
    ctod_series = []
    for disp in displacement_history:
        measurement = compute_ctod(node_coords, disp, node_pair)
        ctod_series.append(measurement.ctod)

    return np.array(ctod_series)


# =============================================================================
# Crack Width Measurement
# =============================================================================

def compute_crack_widths(
    crack_coords: List[np.ndarray],
    displacement: np.ndarray,
    n_sample_points: int = 10,
) -> List[CrackWidth]:
    """
    Compute crack opening widths along crack paths.

    Parameters
    ----------
    crack_coords : list of ndarray
        List of crack polylines, each shape (n_points, 2)
    displacement : ndarray
        Displacement field
    n_sample_points : int
        Number of sample points along each crack

    Returns
    -------
    widths : list of CrackWidth
        Crack width measurements

    Notes
    -----
    This is a simplified implementation. In practice, you would need to:
    - Identify nodes on either side of the crack
    - Project displacements onto crack normal
    - Handle enriched DOFs properly
    """
    # TODO: Implement full crack width computation
    # Requires knowledge of enriched nodes and crack discontinuity
    raise NotImplementedError(
        "Crack width computation requires integration with XFEM enrichment"
    )


# =============================================================================
# Slip and Bond Stress Profiles
# =============================================================================

def compute_slip_profile(
    rebar_nodes: np.ndarray,
    slip_dofs: np.ndarray,
    n_sample: int = 50,
) -> SlipProfile:
    """
    Compute slip profile along a rebar.

    Parameters
    ----------
    rebar_nodes : ndarray, shape (n_rebar_nodes, 2)
        Node coordinates along rebar
    slip_dofs : ndarray, shape (n_rebar_nodes,)
        Slip DOF values
    n_sample : int
        Number of sample points for smooth profile

    Returns
    -------
    SlipProfile
        Slip and bond stress distribution

    Notes
    -----
    Bond stress τ(x) can be computed from slip via the bond law,
    or from equilibrium: dN/dx = p·τ where p is perimeter.
    """
    # Compute arc length along rebar
    rebar_length = np.cumsum(
        np.sqrt(np.sum(np.diff(rebar_nodes, axis=0) ** 2, axis=1))
    )
    rebar_length = np.concatenate([[0.0], rebar_length])

    # Interpolate slip onto uniform grid
    x_uniform = np.linspace(0, rebar_length[-1], n_sample)
    slip_uniform = np.interp(x_uniform, rebar_length, slip_dofs)

    # Compute bond stress from slip (requires bond law)
    # For now, return slip only
    tau_uniform = np.zeros_like(slip_uniform)  # TODO: Apply bond law

    return SlipProfile(
        rebar_id=0,
        x_coords=x_uniform,
        slip_values=slip_uniform,
        tau_values=tau_uniform,
    )


# =============================================================================
# Steel Force Computation
# =============================================================================

def compute_steel_force_profile(
    rebar_nodes: np.ndarray,
    steel_strain: np.ndarray,
    A_s: float,
    E_s: float,
) -> SteelForce:
    """
    Compute axial force profile in rebar.

    Parameters
    ----------
    rebar_nodes : ndarray, shape (n_nodes, 2)
        Rebar node coordinates
    steel_strain : ndarray, shape (n_nodes,)
        Axial strain along bar
    A_s : float
        Bar cross-sectional area (mm²)
    E_s : float
        Steel Young's modulus (MPa)

    Returns
    -------
    SteelForce
        Force and stress distribution
    """
    # Compute arc length
    seg_lengths = np.sqrt(np.sum(np.diff(rebar_nodes, axis=0) ** 2, axis=1))
    x_coords = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    # Stress = E * epsilon
    stress_values = E_s * steel_strain

    # Force = A * sigma
    force_values = A_s * stress_values

    return SteelForce(
        rebar_id=0,
        x_coords=x_coords,
        force_values=force_values,
        stress_values=stress_values,
    )


# =============================================================================
# Base Reactions and Moment (for walls)
# =============================================================================

def compute_base_moment(
    reaction_forces: np.ndarray,
    base_node_coords: np.ndarray,
    reference_point: np.ndarray,
) -> float:
    """
    Compute moment at base of wall.

    Parameters
    ----------
    reaction_forces : ndarray, shape (n_base_nodes, 2)
        Reaction forces at base nodes (Fx, Fy)
    base_node_coords : ndarray, shape (n_base_nodes, 2)
        Coordinates of base nodes
    reference_point : ndarray, shape (2,)
        Reference point for moment calculation (typically base centroid)

    Returns
    -------
    moment : float
        Base moment (N·mm)
    """
    # Moment = sum( r × F ) where r is moment arm
    moment = 0.0
    for i, (x, y) in enumerate(base_node_coords):
        rx = x - reference_point[0]
        ry = y - reference_point[1]
        Fx, Fy = reaction_forces[i]

        # M_z = rx * Fy - ry * Fx (2D)
        moment += rx * Fy - ry * Fx

    return moment


# =============================================================================
# Global Metrics Extraction
# =============================================================================

def extract_thesis_metrics(
    load_history: np.ndarray,
    disp_history: np.ndarray,
    energy_dict: Dict[str, float],
    n_cracks: int = 0,
    total_crack_length: float = 0.0,
) -> ThesisMetrics:
    """
    Extract standard metrics from simulation results.

    Parameters
    ----------
    load_history : ndarray
        Load time series (N)
    disp_history : ndarray
        Displacement time series (mm)
    energy_dict : dict
        Dictionary with energy components
    n_cracks : int
        Number of cracks
    total_crack_length : float
        Total crack length (mm)

    Returns
    -------
    ThesisMetrics
        Extracted metrics
    """
    # Peak load and displacement
    idx_peak = np.argmax(load_history)
    peak_load = load_history[idx_peak]
    peak_displacement = disp_history[idx_peak]

    # Ultimate displacement (last step)
    ultimate_displacement = disp_history[-1]

    # Total energy (work done)
    # W = integral(P dδ) ≈ trapz
    total_energy = np.trapz(load_history, disp_history)

    # Extract energy components
    fracture_energy = energy_dict.get("W_fract_tension", 0.0)
    plastic_energy = energy_dict.get("W_plastic", 0.0)
    bond_slip_energy = energy_dict.get("W_bond_slip", None)

    # Average crack spacing (if multiple cracks)
    if n_cracks > 1:
        # Simplified: assume cracks span similar regions
        average_crack_spacing = total_crack_length / n_cracks
    else:
        average_crack_spacing = None

    return ThesisMetrics(
        peak_load=peak_load,
        peak_displacement=peak_displacement,
        ultimate_displacement=ultimate_displacement,
        total_energy=total_energy,
        fracture_energy=fracture_energy,
        plastic_energy=plastic_energy,
        bond_slip_energy=bond_slip_energy,
        n_cracks=n_cracks,
        total_crack_length=total_crack_length,
        average_crack_spacing=average_crack_spacing,
    )


# =============================================================================
# CSV Output
# =============================================================================

def save_load_displacement_csv(
    filepath: str,
    displacement: np.ndarray,
    load: np.ndarray,
    step: Optional[np.ndarray] = None,
):
    """Save load-displacement curve to CSV."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        if step is not None:
            writer.writerow(['step', 'displacement_mm', 'load_N'])
            for s, d, p in zip(step, displacement, load):
                writer.writerow([s, d, p])
        else:
            writer.writerow(['displacement_mm', 'load_N'])
            for d, p in zip(displacement, load):
                writer.writerow([d, p])


def save_slip_profile_csv(
    filepath: str,
    profile: SlipProfile,
):
    """Save slip profile to CSV."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x_mm', 'slip_mm', 'tau_MPa'])
        for x, s, tau in zip(profile.x_coords, profile.slip_values, profile.tau_values):
            writer.writerow([x, s, tau])


def save_steel_force_csv(
    filepath: str,
    force_profile: SteelForce,
):
    """Save steel force profile to CSV."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x_mm', 'force_N', 'stress_MPa'])
        for x, f, s in zip(
            force_profile.x_coords,
            force_profile.force_values,
            force_profile.stress_values
        ):
            writer.writerow([x, f, s])


# =============================================================================
# Plotting Utilities (optional, requires matplotlib)
# =============================================================================

def plot_load_displacement(
    filepath: str,
    displacement: np.ndarray,
    load: np.ndarray,
    title: str = "Load-Displacement Curve",
):
    """Create load-displacement plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available, skipping plot")
        return

    plt.figure(figsize=(8, 6))
    plt.plot(displacement, load, 'b-', linewidth=2)
    plt.xlabel('Displacement (mm)', fontsize=12)
    plt.ylabel('Load (N)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def plot_slip_profile(
    filepath: str,
    profile: SlipProfile,
    title: str = "Slip and Bond Stress Profile",
):
    """Create slip and bond stress profile plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available, skipping plot")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Slip profile
    ax1.plot(profile.x_coords, profile.slip_values, 'b-', linewidth=2)
    ax1.set_xlabel('Position along bar (mm)')
    ax1.set_ylabel('Slip (mm)')
    ax1.set_title('Slip Profile')
    ax1.grid(True, alpha=0.3)

    # Bond stress profile
    ax2.plot(profile.x_coords, profile.tau_values, 'r-', linewidth=2)
    ax2.set_xlabel('Position along bar (mm)')
    ax2.set_ylabel('Bond Stress (MPa)')
    ax2.set_title('Bond Stress Profile')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
