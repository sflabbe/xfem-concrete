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
import csv


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
        Displacement vector (m) - full DOF vector including enrichment and steel DOFs
    crack_data : dict, optional
        Crack pattern data
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    vtk_file = output_dir / f"step_{step:04d}.vtk"

    nnode = nodes.shape[0]
    nelem = elems.shape[0]

    # Extract concrete node displacements from full DOF vector
    # u_total includes: concrete (2*nnode), enrichment, steel
    # We only need the first 2*nnode entries for VTK export
    u_conc = u[: 2*nnode]

    # Deformed coordinates
    coords_def = nodes.copy()
    coords_def[:, 0] += u_conc[0::2]
    coords_def[:, 1] += u_conc[1::2]

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
            ux = u_conc[2 * i]
            uy = u_conc[2 * i + 1]
            f.write(f"{ux:.6e} {uy:.6e} 0.0\n")

    print(f"  VTK exported: {vtk_file.name}")


# =============================================================================
# PLOT GENERATION
# =============================================================================

def plot_load_displacement(
    history,
    output_dir: Path,
    xlabel: str = "Displacement [mm]",
    ylabel: str = "Load [kN]",
):
    """
    Generate load-displacement curve and save CSV.

    Parameters
    ----------
    history : list or np.ndarray
        History rows [step, u, P, ...] (single-crack) or list of dicts (multicrack)
    output_dir : Path
        Output directory
    xlabel, ylabel : str
        Axis labels
    """
    # Handle different history formats
    if isinstance(history, list) and len(history) > 0 and isinstance(history[0], dict):
        # Multicrack format: list of dicts with keys 'u', 'P'
        u_mm = np.array([h['u'] for h in history]) * 1e3  # m → mm
        P_kN = np.array([h['P'] for h in history]) / 1e3  # N → kN
    else:
        # Single-crack format: np.ndarray with columns [step, u, P, ...]
        history_arr = np.array(history)
        u_mm = history_arr[:, 1] * 1e3  # Assume m → mm
        P_kN = history_arr[:, 2] / 1e3  # Assume N → kN

    # Save CSV
    csv_file = output_dir / "load_displacement.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['u_mm', 'P_kN'])
        for i in range(len(u_mm)):
            writer.writerow([u_mm[i], P_kN[i]])
    print(f"  CSV saved: {csv_file.name}")

    # Plot
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

def postprocess_case(case, results: Dict[str, Any]):
    """
    Unified postprocessing entry point for solver_interface.

    Parameters
    ----------
    case : CaseConfig
        Case configuration
    results : dict
        Solver results from run_case_solver
    """
    from pathlib import Path
    output_dir = Path(case.outputs.output_dir) / case.outputs.case_name
    postprocess_results(results, case, output_dir)


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
    if case_config.outputs.save_load_displacement and len(history) > 0:
        plot_load_displacement(history, output_dir)

    # Crack pattern extraction (handle both single crack and multicrack)
    cracks_list = results.get('cracks', [crack] if crack is not None else [])
    if cracks_list and case_config.outputs.save_crack_data:
        for i, c in enumerate(cracks_list):
            if c is not None and c.active:
                crack_data = extract_crack_pattern(c, results.get('coh_states', None), nodes)
                print(f"  Crack #{i+1} tip: ({crack_data['tip'][0]*1e3:.1f}, {crack_data['tip'][1]*1e3:.1f}) mm")

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

    # Crack width profiles (if cohesive states available)
    coh_states = results.get('coh_states', None)
    if coh_states is not None and cracks_list:
        from examples.gutierrez_thesis.postprocess import compute_crack_widths_from_cohesive

        crack_widths = compute_crack_widths_from_cohesive(coh_states, cracks_list, nodes, elems)

        # Export crack width profiles
        for crack_id, width_data in crack_widths.items():
            if len(width_data) == 0:
                continue

            # Convert to mm
            s_mm = np.array([s * 1e3 for s, x, y, w in width_data])
            x_mm = np.array([x * 1e3 for s, x, y, w in width_data])
            y_mm = np.array([y * 1e3 for s, x, y, w in width_data])
            w_mm = np.array([w * 1e3 for s, x, y, w in width_data])

            # Save to CSV
            csv_file = output_dir / f"crack_width_profile_crack{crack_id}_final.csv"
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['s_mm', 'x_mm', 'y_mm', 'w_mm'])
                for i in range(len(s_mm)):
                    writer.writerow([s_mm[i], x_mm[i], y_mm[i], w_mm[i]])
            print(f"  Crack width profile #{crack_id} saved: {csv_file.name}")

            # Plot
            if len(w_mm) > 0:
                plt.figure(figsize=(10, 4))
                plt.plot(s_mm, w_mm, 'ro-', markersize=4, linewidth=1.5)
                plt.xlabel("Position along crack [mm]", fontsize=12)
                plt.ylabel("Crack width [mm]", fontsize=12)
                plt.title(f"Crack Width Profile (Crack #{crack_id})", fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                plot_file = output_dir / f"crack_width_profile_crack{crack_id}_final.png"
                plt.savefig(plot_file, dpi=150)
                plt.close()

                # Summary metrics
                w_max = np.max(w_mm)
                w_avg = np.mean(w_mm)
                print(f"    w_max = {w_max:.4f} mm, w_avg = {w_avg:.4f} mm")

    # Steel force profiles (if bond states and rebar segments available)
    if bond_states is not None:
        rebar_segs = results.get('rebar_segs', None)
        dofs = results.get('dofs', None)
        model = results.get('model', None)

        if rebar_segs is not None and dofs is not None and model is not None:
            # Compute steel strain from displacement field
            # For each rebar segment: eps = (u_s(j) - u_s(i)) / L
            n_seg = rebar_segs.shape[0]
            x_coords = []
            N_vals = []
            sigma_vals = []

            # Get steel properties
            E_s = model.steel_E  # Pa
            # Compute cross-sectional area from model
            if hasattr(model, 'steel_A_total'):
                A_s = model.steel_A_total  # m^2
            else:
                # Fallback: assume 12mm bar
                d_bar = 0.012  # m
                A_s = np.pi * (d_bar / 2)**2

            for i in range(n_seg):
                n1 = int(rebar_segs[i, 0])
                n2 = int(rebar_segs[i, 1])
                L0 = rebar_segs[i, 2]  # Segment length (m)
                cx = rebar_segs[i, 3]  # Center x (m)

                # Get steel DOFs for nodes n1, n2
                if hasattr(dofs, 'steel_node_to_idx'):
                    # Map node to steel DOF index
                    if n1 in dofs.steel_node_to_idx and n2 in dofs.steel_node_to_idx:
                        idx1 = dofs.steel_node_to_idx[n1]
                        idx2 = dofs.steel_node_to_idx[n2]
                        dof1 = dofs.steel_dof_offset + 2 * idx1  # ux_steel
                        dof2 = dofs.steel_dof_offset + 2 * idx2

                        # Steel displacements
                        if dof1 < len(u) and dof2 < len(u):
                            u1 = u[dof1]
                            u2 = u[dof2]

                            # Axial strain
                            eps = (u2 - u1) / L0 if L0 > 1e-12 else 0.0

                            # Axial stress
                            sigma = E_s * eps  # Pa

                            # Axial force
                            N = sigma * A_s  # N

                            x_coords.append(cx)
                            N_vals.append(N)
                            sigma_vals.append(sigma)

            if len(x_coords) > 0:
                # Convert to engineering units
                x_mm = np.array(x_coords) * 1e3  # m → mm
                N_kN = np.array(N_vals) / 1e3  # N → kN
                sigma_MPa = np.array(sigma_vals) / 1e6  # Pa → MPa

                # Save to CSV
                csv_file = output_dir / "steel_force_profile_final.csv"
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['x_mm', 'N_kN', 'sigma_MPa'])
                    for i in range(len(x_mm)):
                        writer.writerow([x_mm[i], N_kN[i], sigma_MPa[i]])
                print(f"  Steel force profile saved: {csv_file.name}")

                # Plot
                plt.figure(figsize=(10, 6))

                # Subplot 1: Force distribution
                plt.subplot(2, 1, 1)
                plt.plot(x_mm, N_kN, 'bo-', markersize=4, linewidth=1.5)
                plt.ylabel("Axial Force [kN]", fontsize=12)
                plt.title("Steel Force Distribution", fontsize=14)
                plt.grid(True, alpha=0.3)

                # Subplot 2: Stress distribution
                plt.subplot(2, 1, 2)
                plt.plot(x_mm, sigma_MPa, 'ro-', markersize=4, linewidth=1.5)
                plt.xlabel("Position along bar [mm]", fontsize=12)
                plt.ylabel("Axial Stress [MPa]", fontsize=12)
                plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plot_file = output_dir / "steel_force_profile_final.png"
                plt.savefig(plot_file, dpi=150)
                plt.close()

    # VTK export (final step)
    if case_config.outputs.save_vtk:
        if len(history) > 0:
            # Handle both numeric and dict history formats
            if isinstance(history[0], dict):
                final_step = history[-1].get('step', len(history)-1)
            else:
                final_step = int(history[-1][0])
            export_vtk_step(output_dir / "vtk", final_step, nodes, elems, u)

    print("\n✓ Postprocessing completed")
    print(f"✓ Outputs saved to: {output_dir}\n")
