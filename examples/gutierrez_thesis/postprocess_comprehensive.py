"""
Comprehensive Postprocessing for Gutiérrez Thesis Cases

Computes slip profiles, bond stress profiles, steel forces, crack patterns,
and exports results to CSV, PNG, and VTK formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# SLIP AND BOND PROFILES
# =============================================================================

def compute_slip_profile(
    nodes: np.ndarray,
    rebar_segs: np.ndarray,
    bond_states,  # BondSlipStateArrays
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute slip(x) profile along reinforcement.

    Parameters
    ----------
    nodes : np.ndarray
        Node coordinates [nnode, 2] (m)
    rebar_segs : np.ndarray
        Rebar segments [n_seg, 5]: [n1, n2, L0, cx, cy]
    bond_states : BondSlipStateArrays
        Bond-slip states with s_current

    Returns
    -------
    x_profile : np.ndarray
        x-coordinates along bar (m)
    slip_profile : np.ndarray
        Slip values (m)
    """
    n_seg = rebar_segs.shape[0]
    x_profile = []
    slip_profile = []

    for i in range(n_seg):
        n1 = int(rebar_segs[i, 0])
        n2 = int(rebar_segs[i, 1])

        # Segment midpoint
        x_mid = 0.5 * (nodes[n1, 0] + nodes[n2, 0])
        s = bond_states.s_current[i]

        x_profile.append(x_mid)
        slip_profile.append(s)

    return np.array(x_profile), np.array(slip_profile)


def compute_bond_stress_profile(
    nodes: np.ndarray,
    rebar_segs: np.ndarray,
    bond_states,  # BondSlipStateArrays
    bond_law,  # Bond law object
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute τ(x) profile along reinforcement.

    Parameters
    ----------
    nodes : np.ndarray
        Node coordinates [nnode, 2] (m)
    rebar_segs : np.ndarray
        Rebar segments [n_seg, 5]
    bond_states : BondSlipStateArrays
        Bond-slip states
    bond_law : BondSlipLaw
        Bond law for computing τ from slip

    Returns
    -------
    x_profile : np.ndarray
        x-coordinates (m)
    tau_profile : np.ndarray
        Bond stress (Pa)
    """
    n_seg = rebar_segs.shape[0]
    x_profile = []
    tau_profile = []

    for i in range(n_seg):
        n1 = int(rebar_segs[i, 0])
        n2 = int(rebar_segs[i, 1])

        x_mid = 0.5 * (nodes[n1, 0] + nodes[n2, 0])
        s = bond_states.s_current[i]
        s_max = bond_states.s_max[i]

        # Compute bond stress from law
        tau, _ = bond_law.tau_and_tangent(s, s_max)

        x_profile.append(x_mid)
        tau_profile.append(tau)

    return np.array(x_profile), np.array(tau_profile)


def compute_steel_force_profile(
    nodes: np.ndarray,
    rebar_segs: np.ndarray,
    u: np.ndarray,
    steel_dof_map: np.ndarray,
    steel_E: float,
    steel_A: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute axial force N(x) in reinforcement.

    Parameters
    ----------
    nodes : np.ndarray
        Node coordinates [nnode, 2] (m)
    rebar_segs : np.ndarray
        Rebar segments [n_seg, 5]
    u : np.ndarray
        Displacement vector (m)
    steel_dof_map : np.ndarray
        [nnode, 2] steel DOF mapping
    steel_E : float
        Steel Young's modulus (Pa)
    steel_A : float
        Steel area (m²)

    Returns
    -------
    x_profile : np.ndarray
        x-coordinates (m)
    N_profile : np.ndarray
        Axial force (N)
    """
    n_seg = rebar_segs.shape[0]
    x_profile = []
    N_profile = []

    for i in range(n_seg):
        n1 = int(rebar_segs[i, 0])
        n2 = int(rebar_segs[i, 1])
        L0 = rebar_segs[i, 2]
        cx = rebar_segs[i, 3]
        cy = rebar_segs[i, 4]

        # Get steel displacements
        dof_s1x = int(steel_dof_map[n1, 0])
        dof_s1y = int(steel_dof_map[n1, 1])
        dof_s2x = int(steel_dof_map[n2, 0])
        dof_s2y = int(steel_dof_map[n2, 1])

        u_s1x = u[dof_s1x]
        u_s1y = u[dof_s1y]
        u_s2x = u[dof_s2x]
        u_s2y = u[dof_s2y]

        # Axial strain (engineering)
        du_x = u_s2x - u_s1x
        du_y = u_s2y - u_s1y
        dL = du_x * cx + du_y * cy  # Projection onto bar axis
        epsilon = dL / L0

        # Axial force
        N = steel_E * steel_A * epsilon

        x_mid = 0.5 * (nodes[n1, 0] + nodes[n2, 0])
        x_profile.append(x_mid)
        N_profile.append(N)

    return np.array(x_profile), np.array(N_profile)


# =============================================================================
# CRACK PATTERN EXTRACTION
# =============================================================================

def extract_crack_pattern(
    crack,  # XFEMCrack object
    cohesive_states,  # CohesiveStateArrays
    nodes: np.ndarray,
) -> Dict[str, Any]:
    """
    Extract crack pattern data for visualization.

    Parameters
    ----------
    crack : XFEMCrack
        Crack geometry
    cohesive_states : CohesiveStateArrays
        Cohesive zone states (δ_max at Gauss points)
    nodes : np.ndarray
        Node coordinates

    Returns
    -------
    crack_data : dict
        Dictionary with:
        - 'tip': (x, y) crack tip location
        - 'base': (x, y) crack base location
        - 'active': bool
        - 'segments': List[((x1,y1), (x2,y2))] crack segments
        - 'openings': np.ndarray, crack openings at GP
    """
    crack_data = {
        'tip': (crack.tip_x, crack.tip_y),
        'base': (crack.x0, crack.y0),
        'active': crack.active,
        'segments': [],
        'openings': [],
    }

    if crack.active:
        # Simple straight crack (base to tip)
        crack_data['segments'].append(
            ((crack.x0, crack.y0), (crack.tip_x, crack.tip_y))
        )

        # Extract openings from cohesive states
        # cohesive_states.delta_max[crack_idx, gp]
        if hasattr(cohesive_states, 'delta_max') and cohesive_states.delta_max.shape[0] > 0:
            openings = cohesive_states.delta_max[0, :]  # Primary crack
            crack_data['openings'] = openings

    return crack_data


# =============================================================================
# VTK EXPORT
# =============================================================================

def export_vtk_step(
    output_dir: Path,
    step: int,
    nodes: np.ndarray,
    elems: np.ndarray,
    u: np.ndarray,
    crack_data: Optional[Dict] = None,
):
    """
    Export VTK file for a single step (simplified ASCII format).

    Parameters
    ----------
    output_dir : Path
        Output directory
    step : int
        Step number
    nodes : np.ndarray
        Node coordinates [nnode, 2] (m)
    elems : np.ndarray
        Element connectivity [nelem, 4]
    u : np.ndarray
        Displacement vector (m)
    crack_data : dict, optional
        Crack pattern data
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    vtk_file = output_dir / f"step_{step:04d}.vtk"

    nnode = nodes.shape[0]
    nelem = elems.shape[0]

    # Deformed coordinates
    coords_def = nodes.copy()
    coords_def[:, 0] += u[0::2]
    coords_def[:, 1] += u[1::2]

    with open(vtk_file, 'w') as f:
        # Header
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"XFEM Step {step}\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        # Points
        f.write(f"POINTS {nnode} float\n")
        for i in range(nnode):
            f.write(f"{coords_def[i, 0]:.6e} {coords_def[i, 1]:.6e} 0.0\n")

        # Cells (quads)
        f.write(f"\nCELLS {nelem} {nelem * 5}\n")
        for e in range(nelem):
            f.write(f"4 {elems[e, 0]} {elems[e, 1]} {elems[e, 2]} {elems[e, 3]}\n")

        f.write(f"\nCELL_TYPES {nelem}\n")
        for e in range(nelem):
            f.write("9\n")  # VTK_QUAD

        # Point data (displacements)
        f.write(f"\nPOINT_DATA {nnode}\n")
        f.write("VECTORS displacement float\n")
        for i in range(nnode):
            ux = u[2 * i]
            uy = u[2 * i + 1]
            f.write(f"{ux:.6e} {uy:.6e} 0.0\n")

    print(f"  VTK exported: {vtk_file.name}")


# =============================================================================
# PLOT GENERATION
# =============================================================================

def plot_load_displacement(
    history: List,
    output_dir: Path,
    xlabel: str = "Displacement [mm]",
    ylabel: str = "Load [kN]",
):
    """
    Generate load-displacement curve.

    Parameters
    ----------
    history : list
        History rows [step, u, P, ...]
    output_dir : Path
        Output directory
    xlabel, ylabel : str
        Axis labels
    """
    history_arr = np.array(history)
    u_mm = history_arr[:, 1]
    P_kN = history_arr[:, 2]

    plt.figure(figsize=(8, 6))
    plt.plot(u_mm, P_kN, 'b-', linewidth=2)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title("Load-Displacement Curve", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_file = output_dir / "load_displacement.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()

    print(f"  Plot saved: {plot_file.name}")


def plot_slip_profile(
    x_mm: np.ndarray,
    slip_mm: np.ndarray,
    output_dir: Path,
    step: int,
):
    """
    Generate slip(x) profile plot.

    Parameters
    ----------
    x_mm : np.ndarray
        x-coordinates (mm)
    slip_mm : np.ndarray
        Slip values (mm)
    output_dir : Path
        Output directory
    step : int
        Step number
    """
    plt.figure(figsize=(10, 4))
    plt.plot(x_mm, slip_mm, 'ro-', markersize=4, linewidth=1.5)
    plt.xlabel("Position along bar [mm]", fontsize=12)
    plt.ylabel("Slip [mm]", fontsize=12)
    plt.title(f"Bond Slip Profile (Step {step})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_file = output_dir / f"slip_profile_step_{step:04d}.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()


def plot_bond_stress_profile(
    x_mm: np.ndarray,
    tau_MPa: np.ndarray,
    output_dir: Path,
    step: int,
):
    """
    Generate τ(x) profile plot.

    Parameters
    ----------
    x_mm : np.ndarray
        x-coordinates (mm)
    tau_MPa : np.ndarray
        Bond stress (MPa)
    output_dir : Path
        Output directory
    step : int
        Step number
    """
    plt.figure(figsize=(10, 4))
    plt.plot(x_mm, tau_MPa, 'bs-', markersize=4, linewidth=1.5)
    plt.xlabel("Position along bar [mm]", fontsize=12)
    plt.ylabel("Bond stress τ [MPa]", fontsize=12)
    plt.title(f"Bond Stress Profile (Step {step})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_file = output_dir / f"bond_stress_profile_step_{step:04d}.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()


# =============================================================================
# MAIN POSTPROCESS FUNCTION
# =============================================================================

def postprocess_results(
    results: Dict[str, Any],
    case_config,  # CaseConfig
    output_dir: Path,
):
    """
    Comprehensive postprocessing for thesis case.

    Parameters
    ----------
    results : dict
        Solver results with keys: nodes, elems, u, history, crack, bond_states, etc.
    case_config : CaseConfig
        Case configuration
    output_dir : Path
        Output directory
    """
    print("\n" + "="*70)
    print("POSTPROCESSING")
    print("="*70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes = results['nodes']
    elems = results['elems']
    u = results['u']
    history = results['history']
    crack = results.get('crack', None)
    bond_states = results.get('bond_states', None)
    bond_law = results.get('bond_law', None)
    subdomain_mgr = results.get('subdomain_mgr', None)

    # Load-displacement plot
    if case_config.outputs.save_load_displacement:
        plot_load_displacement(history, output_dir)

    # Crack pattern extraction
    if crack is not None and crack.active and case_config.outputs.save_crack_data:
        crack_data = extract_crack_pattern(crack, results.get('coh_states', None), nodes)
        print(f"  Crack tip: ({crack_data['tip'][0]*1e3:.1f}, {crack_data['tip'][1]*1e3:.1f}) mm")

    # Bond-slip profiles (if applicable)
    if bond_states is not None and bond_law is not None:
        rebar_segs = results.get('rebar_segs', None)

        if rebar_segs is not None and len(rebar_segs) > 0:
            # Slip profile
            if case_config.outputs.compute_slip_profiles:
                x_m, slip_m = compute_slip_profile(nodes, rebar_segs, bond_states)
                x_mm = x_m * 1e3
                slip_mm = slip_m * 1e3

                # Save to CSV
                csv_file = output_dir / "slip_profile_final.csv"
                np.savetxt(csv_file, np.column_stack([x_mm, slip_mm]),
                           header="x_mm,slip_mm", delimiter=",", comments='')
                print(f"  Slip profile saved: {csv_file.name}")

                # Plot
                final_step = int(history[-1][0])
                plot_slip_profile(x_mm, slip_mm, output_dir, final_step)

            # Bond stress profile
            if case_config.outputs.compute_bond_profiles:
                x_m, tau_Pa = compute_bond_stress_profile(nodes, rebar_segs, bond_states, bond_law)
                x_mm = x_m * 1e3
                tau_MPa = tau_Pa / 1e6

                csv_file = output_dir / "bond_stress_profile_final.csv"
                np.savetxt(csv_file, np.column_stack([x_mm, tau_MPa]),
                           header="x_mm,tau_MPa", delimiter=",", comments='')
                print(f"  Bond stress profile saved: {csv_file.name}")

                # Plot
                final_step = int(history[-1][0])
                plot_bond_stress_profile(x_mm, tau_MPa, output_dir, final_step)

    # VTK export (final step)
    if case_config.outputs.save_vtk:
        final_step = int(history[-1][0])
        export_vtk_step(output_dir / "vtk", final_step, nodes, elems, u)

    print("\n✓ Postprocessing completed")
    print(f"✓ Outputs saved to: {output_dir}\n")
