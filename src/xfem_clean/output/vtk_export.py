"""VTK export for XFEM damage field visualization.

This module provides utilities to export XFEM analysis results to VTK format
for visualization in ParaView or other VTK-compatible tools.

Key features:
  - Damage fields (compression and tension)
  - Plastic strain magnitude
  - Stress and strain tensors
  - Crack geometry
  - Reinforcement visualization
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from pathlib import Path

import numpy as np

try:
    from xfem_clean.xfem.state_arrays import BulkStateArrays
except ImportError:
    BulkStateArrays = Any


def write_vtk_unstructured_grid(
    filename: str,
    nodes: np.ndarray,
    elems: np.ndarray,
    point_data: Optional[Dict[str, np.ndarray]] = None,
    cell_data: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Write VTK unstructured grid file (legacy ASCII format).

    Parameters
    ----------
    filename : str
        Output .vtk filename
    nodes : np.ndarray
        Node coordinates [n_nodes, ndim]
    elems : np.ndarray
        Element connectivity [n_elem, n_nodes_per_elem]
    point_data : dict
        Nodal data {field_name: values[n_nodes]}
    cell_data : dict
        Element data {field_name: values[n_elem]}
    """
    n_nodes = nodes.shape[0]
    n_elem = elems.shape[0]
    ndim = nodes.shape[1]

    # Ensure 3D coordinates (pad with zeros if 2D)
    if ndim == 2:
        nodes_3d = np.column_stack([nodes, np.zeros(n_nodes)])
    else:
        nodes_3d = nodes

    with open(filename, "w") as f:
        # Header
        f.write("# vtk DataFile Version 3.0\n")
        f.write("XFEM Damage Field\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        # Points
        f.write(f"POINTS {n_nodes} float\n")
        for node in nodes_3d:
            f.write(f"{node[0]:.6e} {node[1]:.6e} {node[2]:.6e}\n")

        # Cells
        n_nodes_per_elem = elems.shape[1]
        cell_size = n_elem * (1 + n_nodes_per_elem)
        f.write(f"\nCELLS {n_elem} {cell_size}\n")
        for elem in elems:
            f.write(f"{n_nodes_per_elem}")
            for node_id in elem:
                f.write(f" {int(node_id)}")
            f.write("\n")

        # Cell types (9 = VTK_QUAD for quads)
        f.write(f"\nCELL_TYPES {n_elem}\n")
        cell_type = 9 if n_nodes_per_elem == 4 else 5  # QUAD or TRIANGLE
        for _ in range(n_elem):
            f.write(f"{cell_type}\n")

        # Point data (nodal fields)
        if point_data:
            f.write(f"\nPOINT_DATA {n_nodes}\n")
            for field_name, values in point_data.items():
                values = np.asarray(values).flatten()
                if len(values) != n_nodes:
                    print(f"Warning: {field_name} has wrong size ({len(values)} != {n_nodes}), skipping")
                    continue

                f.write(f"SCALARS {field_name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for val in values:
                    f.write(f"{float(val):.6e}\n")

        # Cell data (element fields)
        if cell_data:
            f.write(f"\nCELL_DATA {n_elem}\n")
            for field_name, values in cell_data.items():
                values = np.asarray(values).flatten()
                if len(values) != n_elem:
                    print(f"Warning: {field_name} has wrong size ({len(values)} != {n_elem}), skipping")
                    continue

                f.write(f"SCALARS {field_name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for val in values:
                    f.write(f"{float(val):.6e}\n")

    print(f"VTK file written: {filename}")


def average_ip_to_nodes(
    mp_states: BulkStateArrays,
    elems: np.ndarray,
    n_nodes: int,
    field_name: str,
) -> np.ndarray:
    """Average integration point values to nodes.

    Parameters
    ----------
    mp_states : BulkStateArrays
        Material point state arrays
    elems : np.ndarray
        Element connectivity [n_elem, n_nodes_per_elem]
    n_nodes : int
        Total number of nodes
    field_name : str
        Field to extract from MaterialPoint:
        "damage_c", "damage_t", "eps_p_mag", "w_plastic", etc.

    Returns
    -------
    nodal_values : np.ndarray
        Averaged values at nodes [n_nodes]
    """
    nodal_sum = np.zeros(n_nodes, dtype=float)
    nodal_count = np.zeros(n_nodes, dtype=float)

    n_elem = mp_states.eps.shape[0]
    n_ip = mp_states.eps.shape[1]

    for ie in range(n_elem):
        for ip in range(n_ip):
            mp = mp_states.get_mp(ie, ip)

            # Extract field
            if field_name == "damage_c":
                val = float(mp.damage_c)
            elif field_name == "damage_t":
                val = float(mp.damage_t)
            elif field_name == "eps_p_mag":
                val = float(np.linalg.norm(mp.eps_p))
            elif field_name == "w_plastic":
                val = float(mp.w_plastic)
            elif field_name == "w_fract_t":
                val = float(mp.w_fract_t)
            elif field_name == "w_fract_c":
                val = float(mp.w_fract_c)
            elif field_name == "kappa":
                val = float(mp.kappa)
            else:
                val = 0.0

            # Distribute to element nodes
            for node in elems[ie]:
                nodal_sum[node] += val
                nodal_count[node] += 1.0

    # Average
    nodal_values = np.divide(
        nodal_sum,
        nodal_count,
        out=np.zeros_like(nodal_sum),
        where=nodal_count > 0,
    )

    return nodal_values


def export_damage_field(
    filename: str,
    nodes: np.ndarray,
    elems: np.ndarray,
    mp_states: BulkStateArrays,
) -> None:
    """Export damage fields to VTK for visualization.

    Parameters
    ----------
    filename : str
        Output .vtk filename
    nodes : np.ndarray
        Node coordinates
    elems : np.ndarray
        Element connectivity
    mp_states : BulkStateArrays
        Material point states containing damage variables
    """
    n_nodes = nodes.shape[0]

    # Average IP fields to nodes
    damage_compression = average_ip_to_nodes(mp_states, elems, n_nodes, "damage_c")
    damage_tension = average_ip_to_nodes(mp_states, elems, n_nodes, "damage_t")
    plastic_strain_mag = average_ip_to_nodes(mp_states, elems, n_nodes, "eps_p_mag")
    w_plastic = average_ip_to_nodes(mp_states, elems, n_nodes, "w_plastic")
    w_fract_t = average_ip_to_nodes(mp_states, elems, n_nodes, "w_fract_t")
    w_fract_c = average_ip_to_nodes(mp_states, elems, n_nodes, "w_fract_c")

    point_data = {
        "damage_compression": damage_compression,
        "damage_tension": damage_tension,
        "plastic_strain_magnitude": plastic_strain_mag,
        "energy_plastic": w_plastic,
        "energy_fracture_tension": w_fract_t,
        "energy_fracture_compression": w_fract_c,
    }

    write_vtk_unstructured_grid(filename, nodes, elems, point_data=point_data)


def export_full_state(
    filename: str,
    nodes: np.ndarray,
    elems: np.ndarray,
    u: np.ndarray,
    mp_states: BulkStateArrays,
    coh_states: Optional[Any] = None,
) -> None:
    """Export full analysis state including displacements, damage, and energies.

    Parameters
    ----------
    filename : str
        Output .vtk filename
    nodes : np.ndarray
        Node coordinates
    elems : np.ndarray
        Element connectivity
    u : np.ndarray
        Displacement vector [n_dof]
    mp_states : BulkStateArrays
        Material point states
    coh_states : CohesiveStateArrays
        Cohesive crack states (optional)
    """
    n_nodes = nodes.shape[0]

    # Displacements (reshape to nodal vectors)
    u_x = u[0::2]
    u_y = u[1::2]
    u_mag = np.sqrt(u_x**2 + u_y**2)

    # Damage fields
    damage_c = average_ip_to_nodes(mp_states, elems, n_nodes, "damage_c")
    damage_t = average_ip_to_nodes(mp_states, elems, n_nodes, "damage_t")
    eps_p_mag = average_ip_to_nodes(mp_states, elems, n_nodes, "eps_p_mag")
    w_pl = average_ip_to_nodes(mp_states, elems, n_nodes, "w_plastic")
    w_ft = average_ip_to_nodes(mp_states, elems, n_nodes, "w_fract_t")
    w_fc = average_ip_to_nodes(mp_states, elems, n_nodes, "w_fract_c")
    kappa = average_ip_to_nodes(mp_states, elems, n_nodes, "kappa")

    point_data = {
        "displacement_x": u_x,
        "displacement_y": u_y,
        "displacement_magnitude": u_mag,
        "damage_compression": damage_c,
        "damage_tension": damage_t,
        "plastic_strain_magnitude": eps_p_mag,
        "hardening_kappa": kappa,
        "energy_plastic": w_pl,
        "energy_fracture_tension": w_ft,
        "energy_fracture_compression": w_fc,
    }

    write_vtk_unstructured_grid(filename, nodes, elems, point_data=point_data)


def export_crack_geometry(
    filename: str,
    crack: Any,  # XFEMCrack
    n_points: int = 100,
) -> None:
    """Export crack geometry as VTK polyline.

    Parameters
    ----------
    filename : str
        Output .vtk filename
    crack : XFEMCrack
        Crack geometry object
    n_points : int
        Number of points to sample along crack
    """
    if not hasattr(crack, "x0") or not crack.active:
        print("Crack not active or missing geometry, skipping export")
        return

    # Generate points along crack path (vertical line for simplicity)
    x0 = float(crack.x0)
    y0 = float(crack.y0)
    y_tip = float(crack.tip_y)

    y_vals = np.linspace(y0, y_tip, n_points)
    x_vals = np.full_like(y_vals, x0)
    z_vals = np.zeros_like(y_vals)

    points = np.column_stack([x_vals, y_vals, z_vals])

    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("XFEM Crack Geometry\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        # Points
        f.write(f"POINTS {n_points} float\n")
        for pt in points:
            f.write(f"{pt[0]:.6e} {pt[1]:.6e} {pt[2]:.6e}\n")

        # Lines (connect consecutive points)
        f.write(f"\nLINES 1 {n_points+1}\n")
        f.write(f"{n_points}")
        for i in range(n_points):
            f.write(f" {i}")
        f.write("\n")

    print(f"Crack geometry exported: {filename}")


def export_time_series(
    output_dir: str,
    nodes: np.ndarray,
    elems: np.ndarray,
    u_history: list[np.ndarray],
    mp_history: list[BulkStateArrays],
    step_numbers: Optional[np.ndarray] = None,
) -> None:
    """Export time series of damage fields as separate VTK files.

    Parameters
    ----------
    output_dir : str
        Output directory for VTK files
    nodes : np.ndarray
        Node coordinates
    elems : np.ndarray
        Element connectivity
    u_history : list of np.ndarray
        Displacement vectors at each step
    mp_history : list of BulkStateArrays
        Material point states at each step
    step_numbers : np.ndarray
        Step indices (for naming)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_steps = len(u_history)

    if step_numbers is None:
        step_numbers = np.arange(n_steps)

    for i, (u, mp_states) in enumerate(zip(u_history, mp_history)):
        step_num = int(step_numbers[i])
        filename = output_path / f"damage_field_step_{step_num:04d}.vtk"

        export_full_state(
            str(filename),
            nodes,
            elems,
            u,
            mp_states,
        )

    # Write ParaView series file (.pvd)
    pvd_filename = output_path / "damage_field_series.pvd"
    with open(pvd_filename, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1">\n')
        f.write('  <Collection>\n')

        for i, step_num in enumerate(step_numbers):
            vtk_file = f"damage_field_step_{int(step_num):04d}.vtk"
            f.write(f'    <DataSet timestep="{step_num}" file="{vtk_file}"/>\n')

        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')

    print(f"Time series exported to: {output_dir}")
    print(f"ParaView series file: {pvd_filename}")
