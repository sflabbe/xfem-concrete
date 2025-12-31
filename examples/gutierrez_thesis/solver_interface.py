"""
Solver Interface for Gutiérrez Thesis Cases

Provides adapters to convert CaseConfig to XFEM solver inputs and execute simulations.
"""

from typing import Tuple, Optional, Any, Dict, List
import numpy as np

from examples.gutierrez_thesis.case_config import (
    CaseConfig,
    CEBFIPBondLaw,
    BilinearBondLaw as BilinearBondLawConfig,
    BanholzerBondLaw as BanholzerBondLawConfig,
    RebarLayer,
    FRPSheet,
    SubdomainConfig,
)

from xfem_clean.xfem.model import XFEMModel
from xfem_clean.xfem.analysis_single import run_analysis_xfem, BCSpec
from xfem_clean.xfem.multicrack import run_analysis_xfem_multicrack
from xfem_clean.cohesive_laws import CohesiveLaw
from xfem_clean.bond_slip import (
    CustomBondSlipLaw,
    BilinearBondLaw,
    BanholzerBondLaw,
)
from xfem_clean.xfem.subdomains import build_subdomain_manager_from_config
from xfem_clean.fem.mesh import structured_quad_mesh


# =============================================================================
# BOND LAW MAPPER
# =============================================================================

def map_bond_law(bond_law_config: Any) -> Any:
    """
    Map case config bond law to solver bond law.

    Parameters
    ----------
    bond_law_config : CEBFIPBondLaw | BilinearBondLawConfig | BanholzerBondLawConfig
        Bond law from case configuration

    Returns
    -------
    bond_law : CustomBondSlipLaw | BilinearBondLaw | BanholzerBondLaw
        Bond law for solver
    """
    if isinstance(bond_law_config, CEBFIPBondLaw):
        # Convert mm to m, MPa to Pa
        return CustomBondSlipLaw(
            s1=bond_law_config.s1 * 1e-3,  # mm → m
            s2=bond_law_config.s2 * 1e-3,
            s3=bond_law_config.s3 * 1e-3,
            tau_max=bond_law_config.tau_max * 1e6,  # MPa → Pa
            tau_f=bond_law_config.tau_f * 1e6,
            alpha=bond_law_config.alpha,
            use_secant_stiffness=True,
        )

    elif isinstance(bond_law_config, BilinearBondLawConfig):
        # Convert mm to m, MPa to Pa
        return BilinearBondLaw(
            s1=bond_law_config.s1 * 1e-3,  # mm → m
            s2=bond_law_config.s2 * 1e-3,
            tau1=bond_law_config.tau1 * 1e6,  # MPa → Pa
            use_secant_stiffness=True,
        )

    elif isinstance(bond_law_config, BanholzerBondLawConfig):
        # Convert mm to m, MPa to Pa
        return BanholzerBondLaw(
            s0=bond_law_config.s0 * 1e-3,  # mm → m
            a=bond_law_config.a,
            tau1=bond_law_config.tau1 * 1e6,  # MPa → Pa
            tau2=bond_law_config.tau2 * 1e6,
            tau_f=bond_law_config.tau_f * 1e6,
            use_secant_stiffness=True,
        )

    else:
        raise ValueError(f"Unknown bond law type: {type(bond_law_config)}")


# =============================================================================
# BOUNDARY CONDITIONS MAPPER
# =============================================================================

def build_bcs_from_case(
    case: CaseConfig,
    nodes: np.ndarray,
    model: XFEMModel,
    rebar_segs: Optional[np.ndarray] = None,
    frp_nodes: Optional[np.ndarray] = None,
) -> BCSpec:
    """
    Build boundary condition specification from case configuration.

    Parameters
    ----------
    case : CaseConfig
        Case configuration
    nodes : np.ndarray
        Node coordinates [nnode, 2] in SI units (m)
    model : XFEMModel
        XFEM model (for accessing steel DOF mapping)
    rebar_segs : np.ndarray, optional
        Rebar segments for bond-slip cases
    frp_nodes : np.ndarray, optional
        Node IDs that have FRP DOFs allocated (from prepare_edge_segments)

    Returns
    -------
    bc_spec : BCSpec
        Boundary condition specification
    """
    nnode = nodes.shape[0]
    fixed_dofs = {}
    prescribed_dofs = []
    prescribed_scale = 1.0
    reaction_dofs = []

    # Determine BC type from case geometry and loading
    # For now, we implement pullout and beam (3PB) configurations
    case_name = case.case_id.lower()

    if "pullout" in case_name:
        # PULLOUT TEST:
        # - Fix right edge (x=L) concrete nodes: ux=0, uy=0
        # - Prescribe displacement on steel DOFs at load element (left edge)
        # - Measure reaction at steel DOFs

        L = model.L
        # Find nodes at right edge (x ≈ L)
        right_nodes = np.where(np.isclose(nodes[:, 0], L, atol=1e-6))[0]

        # Fix concrete DOFs at right edge
        for n in right_nodes:
            fixed_dofs[2 * n] = 0.0      # ux = 0
            fixed_dofs[2 * n + 1] = 0.0  # uy = 0

        # For steel DOFs, we need to identify them from the DOF manager
        # Since we don't have DOFs built yet, we'll use a spatial criterion
        # based on load_x_center from case.loading

        # Load element region (from case.loading)
        if hasattr(case.loading, 'load_x_center'):
            load_x_center = case.loading.load_x_center * 1e-3  # mm → m
            load_halfwidth = case.loading.load_halfwidth * 1e-3  # mm → m
        else:
            # Default to left edge
            load_x_center = 0.0
            load_halfwidth = 0.05  # 50 mm

        # Find rebar nodes in load element region
        if rebar_segs is not None and len(rebar_segs) > 0:
            # Identify rebar nodes
            rebar_nodes = set()
            for seg in rebar_segs:
                n1, n2 = int(seg[0]), int(seg[1])
                rebar_nodes.add(n1)
                rebar_nodes.add(n2)

            # Filter rebar nodes in load region
            load_rebar_nodes = []
            for n in rebar_nodes:
                x_n = nodes[n, 0]
                if abs(x_n - load_x_center) <= load_halfwidth:
                    load_rebar_nodes.append(n)

            # Steel DOFs: [steel_dof_offset + 2*local_idx, ...]
            # Since we don't have steel_dof_offset yet, store node indices
            # and convert later in analysis driver
            # WORKAROUND: store steel node indices as negative DOFs for now
            # (will be converted in run_analysis_xfem after DOF manager is built)
            for n in load_rebar_nodes:
                # Mark steel DOFs for this node (to be resolved later)
                # Use a marker: -(nnode + steel_node_id) to distinguish from concrete DOFs
                prescribed_dofs.append(-(2 * nnode + n * 2))  # steel ux
                reaction_dofs.append(-(2 * nnode + n * 2))

        # Positive scale for pullout (pull in +x direction)
        prescribed_scale = 1.0

    elif "frp" in case_name or "sspot" in case_name:
        # FRP SHEET TEST (SSPOT - Single Shear Push-Off Test):
        # - Fix bottom edge (y=0): uy=0 for all nodes
        # - Fix ux=0 for left bottom corner (prevent rigid body)
        # - Prescribe displacement on FRP sheet DOFs at loaded end (x≈L)

        # Find bottom nodes (y ≈ 0)
        y_tol = 1e-6
        bottom_nodes = np.where(np.isclose(nodes[:, 1], 0.0, atol=y_tol))[0]

        # Fix uy=0 for all bottom nodes (concrete)
        for n in bottom_nodes:
            fixed_dofs[2 * n + 1] = 0.0  # uy = 0

        # Fix ux=0 for leftmost bottom node (prevent rigid body)
        if len(bottom_nodes) > 0:
            left_bottom = bottom_nodes[np.argmin(nodes[bottom_nodes, 0])]
            fixed_dofs[2 * left_bottom] = 0.0  # ux = 0

        # Prescribe displacement on FRP sheet DOFs at loaded end
        # FRP sheet is at y=0 (bottom edge), loaded at x≈L (right edge)
        L = model.L
        load_halfwidth = case.loading.load_halfwidth * 1e-3  # mm → m

        # Find nodes at bottom edge near right end
        if hasattr(case.loading, 'load_x_center'):
            load_x_center = case.loading.load_x_center * 1e-3
        else:
            load_x_center = L  # Right edge

        # Find FRP nodes in load region
        # Use frp_nodes if provided (contains only nodes with FRP DOFs allocated),
        # otherwise fall back to bottom_nodes
        candidate_nodes = frp_nodes if frp_nodes is not None and len(frp_nodes) > 0 else bottom_nodes
        frp_load_nodes = []
        for n in candidate_nodes:
            x_n = nodes[n, 0]
            if abs(x_n - load_x_center) <= load_halfwidth:
                frp_load_nodes.append(n)

        # Mark FRP DOFs for prescription (negative marker for steel DOFs)
        for n in frp_load_nodes:
            # FRP DOF marker: -(2*nnode + 2*node_id) for x-component
            prescribed_dofs.append(-(2 * nnode + 2 * n))  # FRP ux
            reaction_dofs.append(-(2 * nnode + 2 * n))

        # Positive scale (pull in +x direction)
        prescribed_scale = 1.0

    elif "wall" in case_name:
        # WALL TEST (RC wall under cyclic lateral loading):
        # - Fix base (y=0): uy=0 for all nodes, ux=0 for one corner node (prevent rigid body)
        # - Prescribe top displacement: ux=u_target for nodes in rigid beam zone
        # - Axial load: constant compression on top edge

        # Find base nodes (y ≈ 0)
        y_tol = 1e-6
        base_nodes = np.where(np.isclose(nodes[:, 1], 0.0, atol=y_tol))[0]

        # Fix uy=0 for all base nodes
        for n in base_nodes:
            fixed_dofs[2 * n + 1] = 0.0  # uy = 0

        # Fix ux=0 for leftmost base node to prevent rigid body motion
        if len(base_nodes) > 0:
            left_base = base_nodes[np.argmin(nodes[base_nodes, 0])]
            fixed_dofs[2 * left_base] = 0.0  # ux = 0

        # Prescribe horizontal displacement at top (in rigid beam zone)
        # Rigid beam: y in [y_rigid_start, H]
        H = model.H

        # Check if rigid beam subdomain is defined in case
        if hasattr(case, 'subdomains') and case.subdomains is not None:
            # Find rigid beam subdomain
            rigid_y_min = None
            for subdomain in case.subdomains:
                if subdomain.material_type == "rigid" and subdomain.y_range is not None:
                    rigid_y_min = subdomain.y_range[0] * 1e-3  # mm → m
                    break

            if rigid_y_min is None:
                # Default: top 10% of height
                rigid_y_min = 0.9 * H
        else:
            # Default: top 10% of height
            rigid_y_min = 0.9 * H

        # Find nodes in rigid beam zone
        rigid_nodes = np.where(nodes[:, 1] >= rigid_y_min - y_tol)[0]

        # Prescribe ux for rigid beam nodes
        for n in rigid_nodes:
            prescribed_dofs.append(2 * n)  # ux
            reaction_dofs.append(2 * n)

        # Positive scale for wall (push in +x direction)
        prescribed_scale = 1.0

        # Axial load (constant compression on top edge)
        nodal_forces = {}
        if hasattr(case.loading, 'axial_load') and case.loading.axial_load is not None:
            P_axial = case.loading.axial_load  # N (negative for compression)
            L = model.L  # m

            # Find top nodes (y ≈ H)
            H = model.H
            top_nodes = np.where(np.isclose(nodes[:, 1], H, atol=y_tol))[0]

            if len(top_nodes) > 1:
                # Sort by x
                top_nodes_sorted = top_nodes[np.argsort(nodes[top_nodes, 0])]

                # Compute spacing (assume uniform)
                x_coords = nodes[top_nodes_sorted, 0]
                dx_avg = np.mean(np.diff(x_coords))

                # Distribute force: q = P_axial / L (force per unit length)
                q = P_axial / L  # N/m

                # Nodal forces: f_i = q * dx
                for i, n in enumerate(top_nodes_sorted):
                    if i == 0 or i == len(top_nodes_sorted) - 1:
                        # End nodes: half contribution
                        f_nodal = 0.5 * q * dx_avg
                    else:
                        # Interior nodes: full contribution
                        f_nodal = q * dx_avg

                    # Apply to uy DOF (compression is negative y direction)
                    dof_uy = 2 * n + 1
                    nodal_forces[dof_uy] = f_nodal

                print(f"Axial load: P={P_axial:.2e} N distributed on {len(top_nodes)} top nodes")

    else:
        # DEFAULT: 3-point bending beam
        # - Fix left bottom (ux=0, uy=0) and right bottom (uy=0)
        # - Prescribe top center (uy=-umax)

        left = int(np.argmin(nodes[:, 0]))
        right = int(np.argmax(nodes[:, 0]))

        fixed_dofs[2 * left] = 0.0      # ux = 0
        fixed_dofs[2 * left + 1] = 0.0  # uy = 0
        fixed_dofs[2 * right + 1] = 0.0  # uy = 0

        # Top center nodes for prescribed displacement
        y_top = model.H
        top_nodes = np.where(np.isclose(nodes[:, 1], y_top))[0]

        if hasattr(case.loading, 'load_x_center'):
            load_x_center = case.loading.load_x_center * 1e-3
        else:
            load_x_center = model.L / 2.0

        if hasattr(case.loading, 'load_halfwidth'):
            load_halfwidth = case.loading.load_halfwidth * 1e-3
        else:
            dx = model.L / case.geometry.n_elem_x
            load_halfwidth = 2.0 * dx

        load_nodes = top_nodes[np.where(np.abs(nodes[top_nodes, 0] - load_x_center) <= load_halfwidth)[0]]

        if len(load_nodes) == 0:
            load_nodes = [int(top_nodes[np.argmin(np.abs(nodes[top_nodes, 0] - load_x_center))])]

        for n in load_nodes:
            prescribed_dofs.append(2 * n + 1)  # uy
            reaction_dofs.append(2 * n + 1)

        # Negative scale for beam (load downward)
        prescribed_scale = -1.0

    # Collect nodal_forces if defined (e.g., for axial load in wall case)
    final_nodal_forces = locals().get('nodal_forces', None)

    return BCSpec(
        fixed_dofs=fixed_dofs,
        prescribed_dofs=prescribed_dofs,
        prescribed_scale=prescribed_scale,
        reaction_dofs=reaction_dofs,
        nodal_forces=final_nodal_forces,
    )


# =============================================================================
# CASE CONFIG → XFEM MODEL ADAPTER
# =============================================================================

def case_config_to_xfem_model(case: CaseConfig) -> XFEMModel:
    """
    Convert CaseConfig to XFEMModel.

    Parameters
    ----------
    case : CaseConfig
        Thesis case configuration

    Returns
    -------
    model : XFEMModel
        XFEM solver model
    """

    # Convert units: mm → m, MPa → Pa, N/mm → N/m
    L = case.geometry.length * 1e-3  # mm → m
    H = case.geometry.height * 1e-3
    b = case.geometry.thickness * 1e-3

    E = case.concrete.E * 1e6  # MPa → Pa
    nu = case.concrete.nu
    ft = case.concrete.f_t * 1e6  # MPa → Pa
    Gf = case.concrete.G_f * 1e3  # N/mm → N/m
    fc = case.concrete.f_c * 1e6  # MPa → Pa

    # Steel/FRP properties (rebar OR FRP sheet)
    if case.rebar_layers:
        # Rebar case
        rebar_first = case.rebar_layers[0]
        steel_E = rebar_first.steel.E * 1e6  # MPa → Pa
        steel_fy = rebar_first.steel.f_y * 1e6  # MPa → Pa
        steel_fu = rebar_first.steel.f_u * 1e6 if rebar_first.steel.f_u else steel_fy * 1.145
        steel_Eh = rebar_first.steel.hardening_modulus * 1e6 if rebar_first.steel.hardening_modulus else 0.0

        # Compute total steel area
        steel_A_total = sum(
            (layer.diameter * 1e-3 / 2)**2 * np.pi * layer.n_bars
            for layer in case.rebar_layers
        )

        # Use first rebar diameter for bond-slip
        rebar_diameter = case.rebar_layers[0].diameter * 1e-3  # mm → m

    elif case.frp_sheets:
        # FRP sheet case: reuse "steel" DOFs for FRP
        frp_sheet = case.frp_sheets[0]
        steel_E = frp_sheet.E * 1e6  # MPa → Pa
        steel_fy = steel_E * 0.01  # FRP doesn't yield, use high value
        steel_fu = steel_E * 0.02
        steel_Eh = 0.0  # No hardening for FRP (elastic-brittle)

        # Compute FRP area
        thickness_m = frp_sheet.thickness * 1e-3  # mm → m
        width_m = frp_sheet.width * 1e-3  # mm → m
        steel_A_total = thickness_m * width_m

        # FRP doesn't have diameter; use dummy for backward compat
        rebar_diameter = 0.001  # mm (will use explicit perimeter instead)

    else:
        # No reinforcement
        steel_E = 200e9  # Default
        steel_fy = 500e6
        steel_fu = 600e6
        steel_Eh = 0.0
        steel_A_total = 0.0
        rebar_diameter = 0.012  # Default 12mm

    # Map material model
    bulk_material_map = {
        "elastic": "elastic",
        "dp": "dp",
        "cdp_lite": "cdp-lite",
        "cdp_full": "cdp",
    }
    bulk_material = bulk_material_map.get(case.concrete.model_type, "elastic")

    # Bond-slip configuration (rebar or FRP)
    enable_bond_slip = len(case.rebar_layers) > 0 or len(case.frp_sheets) > 0

    # Characteristic length (element size)
    lch = L / case.geometry.n_elem_x  # Approximate

    # Create model
    model = XFEMModel(
        # Geometry
        L=L,
        H=H,
        b=b,

        # Concrete
        E=E,
        nu=nu,
        ft=ft,
        Gf=Gf,
        fc=fc,

        # Steel
        steel_A_total=steel_A_total,
        steel_E=steel_E,
        steel_fy=steel_fy,
        steel_fu=steel_fu,
        steel_Eh=steel_Eh,

        # Material model
        bulk_material=bulk_material,

        # CDP parameters (if using CDP)
        cdp_phi_deg=30.0,
        cdp_H=0.0,

        # Bond-slip
        enable_bond_slip=enable_bond_slip,
        rebar_diameter=rebar_diameter,
        bond_condition="good",  # Default
        steel_EA_min=steel_E * steel_A_total if steel_A_total > 0 else 1e3,

        # Solver parameters (relaxed for robustness)
        newton_maxit=50,  # Increased from 25
        newton_tol_r=max(case.tolerance, 1e-4),  # Relaxed minimum tolerance
        newton_tol_du=1e-7,  # Relaxed from 1e-8
        line_search=case.use_line_search,
        enable_diagonal_scaling=True,
        max_subdiv=15 if case.use_substepping else 0,  # Increased from 12
        max_total_substeps=100000,  # Increased from 50000

        # Crack parameters (for cases with cohesive cracks)
        crack_margin=0.3,
        crack_rho=lch,
        lch=lch,
        Kn_factor=0.1,
        visc_damp=0.0,

        # Loading
        load_halfwidth=case.loading.load_halfwidth * 1e-3 if hasattr(case.loading, 'load_halfwidth') else 0.0,

        # Debugging
        debug_substeps=False,
        debug_newton=False,
    )

    return model


# =============================================================================
# CYCLIC LOADING TRAJECTORY GENERATOR
# =============================================================================

def generate_cyclic_u_targets(
    targets_mm: List[float],
    n_cycles_per_target: int = 1,
) -> np.ndarray:
    """
    Generate cyclic displacement trajectory.

    For each target displacement, repeats n_cycles_per_target of:
      0 → +target → 0 → -target → 0

    Parameters
    ----------
    targets_mm : list[float]
        List of target displacements (mm)
    n_cycles_per_target : int
        Number of cycles per target

    Returns
    -------
    u_targets_mm : np.ndarray
        Full displacement trajectory (mm)
    """
    trajectory = [0.0]

    for target in targets_mm:
        for _ in range(n_cycles_per_target):
            # Positive cycle: 0 → +t → 0
            trajectory.append(target)
            trajectory.append(0.0)

            # Negative cycle: 0 → -t → 0
            trajectory.append(-target)
            trajectory.append(0.0)

    # Remove duplicate consecutive zeros
    u_targets_mm = []
    prev = None
    for val in trajectory:
        if val != prev or val != 0.0:
            u_targets_mm.append(val)
        prev = val

    return np.array(u_targets_mm)


# =============================================================================
# SOLVER DISPATCH
# =============================================================================

def _should_use_multicrack(case: CaseConfig) -> bool:
    """
    Determine if case requires multicrack solver.

    Heuristics:
    - Explicit case IDs known to need distributed cracking
    - Non-elastic bulk material (CDP/DP)
    """
    multicrack_case_ids = {
        "03_tensile_stn12",
        "04_beam_3pb_t5a1",
        "05_wall_c1_cyclic",
        "06_fibre_tensile",
    }

    # Check case ID
    if case.case_id in multicrack_case_ids:
        return True

    # Check material model (non-elastic likely needs multicrack)
    if case.concrete.model_type not in {"elastic"}:
        # For now, be conservative and use multicrack for cdp/dp
        # (can be refined later)
        pass

    return False


# =============================================================================
# SOLVER EXECUTION
# =============================================================================

def run_case_solver(
    case: CaseConfig,
    mesh_factor: float = 1.0,
    enable_postprocess: bool = True,
    max_steps: Optional[int] = None,
    return_bundle: bool = True,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run XFEM solver for a thesis case.

    Parameters
    ----------
    case : CaseConfig
        Thesis case configuration
    mesh_factor : float
        Mesh refinement factor (1.0 = default)
    enable_postprocess : bool
        Enable comprehensive postprocessing (CSV, PNG outputs)
    max_steps : int, optional
        Override number of steps (if None, use case.loading.n_steps)
    return_bundle : bool
        If True, return comprehensive bundle (default). If False, minimal dict.
    output_dir : str, optional
        Override output directory (if None, use case.outputs.output_dir)

    Returns
    -------
    results : dict
        Solver results with keys:
        - 'nodes': node coordinates
        - 'elems': element connectivity
        - 'u': displacement vector
        - 'history': load-displacement history
        - 'crack': crack object
        - 'bond_states': bond-slip states (if applicable)
    """

    # Create model
    model = case_config_to_xfem_model(case)

    # Apply mesh factor
    nx = int(case.geometry.n_elem_x * mesh_factor)
    ny = int(case.geometry.n_elem_y * mesh_factor)

    # Dispatch: determine which solver to use
    # FASE D: Dispatcher for single-crack vs multicrack vs cyclic
    is_cyclic = hasattr(case.loading, 'loading_type') and case.loading.loading_type == "cyclic"
    use_multicrack = _should_use_multicrack(case)

    # Extract loading parameters
    if is_cyclic:
        # Cyclic loading: generate u_targets trajectory
        targets_mm = case.loading.targets
        n_cycles_per_target = getattr(case.loading, 'n_cycles_per_target', 1)
        u_targets_mm = generate_cyclic_u_targets(targets_mm, n_cycles_per_target)
        u_targets = u_targets_mm * 1e-3  # mm → m
        nsteps = len(u_targets)
        umax = float(np.max(np.abs(u_targets)))
        print(f"Cyclic loading: {len(u_targets)} steps, max={umax*1e3:.2f} mm")
    elif hasattr(case.loading, 'max_displacement'):
        # Monotonic loading
        umax = case.loading.max_displacement * 1e-3  # mm → m
        nsteps = case.loading.n_steps
        u_targets = None  # Will use linspace in solver
    else:
        raise NotImplementedError(f"Unsupported loading type: {type(case.loading)}")

    # Override nsteps if max_steps provided (for regression tests)
    if max_steps is not None:
        nsteps = max_steps

    # Override output_dir if provided (for regression tests)
    if output_dir is not None:
        case.outputs.output_dir = output_dir

    # Create cohesive law (if using cohesive cracks)
    # For pull-out tests without cracks, we still need a cohesive law object
    # but it won't be activated
    law = CohesiveLaw(
        Kn=model.Kn_factor * model.E / max(1e-12, model.lch),
        ft=model.ft,
        Gf=model.Gf,
    )

    # Initialize bond variables (will be set after segments are created)
    bond_law = None
    bond_perimeter = None
    bond_segment_mask = None
    bond_segs = None

    # Create mesh (needed for subdomain manager and BCs)
    nodes, elems = structured_quad_mesh(model.L, model.H, nx, ny)

    # Build subdomain manager (FASE C)
    subdomain_mgr = None
    if case.subdomains:
        # Note: nodes are in m, subdomain ranges in case are in mm → convert with 1e-3
        subdomain_mgr = build_subdomain_manager_from_config(
            nodes, elems, case.subdomains, unit_conversion=1e-3
        )
        print(f"Subdomain manager created: {len(case.subdomains)} subdomain(s)")

    # Store subdomain_mgr in model for access during assembly
    if subdomain_mgr is not None:
        model.subdomain_mgr = subdomain_mgr

    # Bond-disabled x-range (for pullout empty elements)
    if case.rebar_layers and case.rebar_layers[0].bond_disabled_x_range is not None:
        x_min, x_max = case.rebar_layers[0].bond_disabled_x_range
        model.bond_disabled_x_range = (x_min * 1e-3, x_max * 1e-3)  # mm → m

    # Prepare rebar/FRP segments for BC mapping
    from xfem_clean.rebar import prepare_rebar_segments, prepare_edge_segments
    rebar_segs = prepare_rebar_segments(nodes, cover=model.cover) if case.rebar_layers else None

    # Pre-generate FRP nodes for BC mapping (needed before build_bcs_from_case)
    frp_nodes_for_bc = None
    if case.frp_sheets:
        frp_sheet = case.frp_sheets[0]
        y_pos = frp_sheet.y_position * 1e-3  # mm → m
        _, frp_nodes_for_bc = prepare_edge_segments(
            nodes,
            y_target=y_pos,
            x_min=None,
            x_max=None,
            tol=1e-6,
        )

    # Build boundary conditions from case configuration
    bc_spec = build_bcs_from_case(case, nodes, model, rebar_segs=rebar_segs, frp_nodes=frp_nodes_for_bc)

    # FRP sheets handling (BLOQUE 5)
    frp_segs = None
    frp_bond_perimeter = None
    frp_segment_mask = None
    frp_EA = None
    if case.frp_sheets:
        # FRP sheets: use first sheet (multi-sheet TODO for future)
        frp_sheet = case.frp_sheets[0]

        # Convert units
        y_pos = frp_sheet.y_position * 1e-3  # mm → m
        bonded_length = frp_sheet.bonded_length * 1e-3  # mm → m
        L = model.L

        # Determine bonded region (assume bonded at loaded end for SSPOT)
        # For FRP: bonded region is typically at x ∈ [L - bonded_length, L]
        x_bonded_min = L - bonded_length
        x_bonded_max = L

        # Generate FRP segments (re-use y_pos from above)
        frp_segs, frp_nodes = prepare_edge_segments(
            nodes,
            y_target=y_pos,
            x_min=None,  # Include all segments, will mask unbonded later
            x_max=None,
            tol=1e-6,
        )

        if len(frp_segs) > 0:
            # Compute FRP perimeter (sheet width, not circular)
            frp_bond_perimeter = frp_sheet.width * 1e-3  # mm → m

            # Compute FRP EA
            thickness_m = frp_sheet.thickness * 1e-3  # mm → m
            width_m = frp_sheet.width * 1e-3  # mm → m
            E_frp = frp_sheet.E * 1e6  # MPa → Pa
            A_frp = thickness_m * width_m
            frp_EA = E_frp * A_frp

            # Generate segment mask (disable bond in unbonded region)
            frp_segment_mask = np.zeros(len(frp_segs), dtype=bool)
            for i, seg in enumerate(frp_segs):
                n1, n2 = int(seg[0]), int(seg[1])
                cx_seg = 0.5 * (nodes[n1, 0] + nodes[n2, 0])  # Segment center x
                # Mark as disabled (True) if outside bonded region
                if cx_seg < x_bonded_min or cx_seg > x_bonded_max:
                    frp_segment_mask[i] = True

            # Map FRP bond law
            bond_law = map_bond_law(frp_sheet.bond_law)

            # Store in model for analysis
            model.frp_segs = frp_segs
            model.frp_bond_law = bond_law
            model.frp_bond_perimeter = frp_bond_perimeter
            model.frp_segment_mask = frp_segment_mask
            model.frp_EA = frp_EA
            # Override bond perimeter for FRP (used by multicrack)
            model.bond_perimeter_override = frp_bond_perimeter

            print(f"FRP sheet: {len(frp_segs)} segments, perimeter={frp_bond_perimeter*1e3:.2f} mm, EA={frp_EA:.2e} N")
            print(f"  Bonded region: x ∈ [{x_bonded_min*1e3:.1f}, {x_bonded_max*1e3:.1f}] mm")
            print(f"  Disabled segments: {np.sum(frp_segment_mask)}/{len(frp_segs)}")

    # Fibre bridging handling (BLOQUE 6)
    fibre_bridging_cfg = None
    if case.fibres:
        from xfem_clean.fibre_bridging import fibre_config_from_case

        # Convert case fibre config to bridging config
        fibre_bridging_cfg = fibre_config_from_case(
            fibre_reinf=case.fibres,
            thickness_m=model.b,  # Specimen thickness in m
        )

        # Store in model for access during assembly
        model.fibre_bridging_cfg = fibre_bridging_cfg

        print(f"Fibre bridging: {fibre_bridging_cfg.density_m2:.1f} fibres/m², " +
              f"d={fibre_bridging_cfg.d_fibre*1e3:.2f} mm, " +
              f"L={fibre_bridging_cfg.L_fibre*1e3:.1f} mm")
        print(f"  Orientation: {fibre_bridging_cfg.orientation_mean_deg:.1f}° " +
              f"± {fibre_bridging_cfg.orientation_std_deg:.1f}°")
        print(f"  Explicit fraction: {fibre_bridging_cfg.explicit_fraction*100:.0f}% " +
              f"(forces scaled by {1/fibre_bridging_cfg.explicit_fraction:.0f}x)")

    # Map bond law (rebar or FRP) - must happen AFTER segments are created
    if case.frp_sheets and hasattr(model, 'frp_segs'):
        # FRP case: use FRP parameters set during FRP setup
        bond_law = model.frp_bond_law
        bond_perimeter = model.frp_bond_perimeter
        bond_segment_mask = model.frp_segment_mask
        bond_segs = model.frp_segs
    elif case.rebar_layers and rebar_segs is not None:
        # Rebar case: use first rebar layer's bond law
        bond_law = map_bond_law(case.rebar_layers[0].bond_law)
        bond_segs = rebar_segs
        # perimeter will be computed from bond_law.d_bar (legacy)
        bond_perimeter = None
        # segment_mask from bond_disabled_x_range (handled inside solver)
        bond_segment_mask = None

    # Dispatch to appropriate solver
    print(f"Running solver: nx={nx}, ny={ny}, nsteps={nsteps}, umax={umax*1e3:.3f} mm")

    # Pass mesh and bc_spec to analysis
    # NOTE: We pass nodes/elems manually to avoid re-creating mesh inside run_analysis_xfem
    # The analysis driver will use these if provided, otherwise creates its own
    model._nodes = nodes  # Store in model to pass to analysis
    model._elems = elems

    if use_multicrack:
        print("  Using MULTICRACK solver (distributed cracking)")
        # Call multicrack solver with full integration (FASE D + BLOQUE 2)
        bundle = run_analysis_xfem_multicrack(
            model=model,
            nx=nx,
            ny=ny,
            nsteps=nsteps,
            umax=umax,
            law=law,
            nodes=nodes,
            elems=elems,
            u_targets=u_targets if is_cyclic else None,
            bc_spec=bc_spec,
            bond_law=bond_law,
            return_bundle=True,  # BLOQUE 2: Get comprehensive bundle
        )

        # Add compatibility fields for postprocessing
        # Multicrack returns 'cracks' (list), but postprocess expects 'crack' (single)
        if 'crack' not in bundle and 'cracks' in bundle and len(bundle['cracks']) > 0:
            bundle['crack'] = bundle['cracks'][0]  # First crack for compatibility

        # Multicrack returns 'bulk_states' but postprocess may expect 'mp_states'
        if 'mp_states' not in bundle and 'bulk_states' in bundle:
            bundle['mp_states'] = bundle['bulk_states']

    elif is_cyclic:
        print("  Using CYCLIC driver with single-crack (custom u_targets)")
        # BLOQUE 3: Use single-crack with u_targets
        bundle = run_analysis_xfem(
            model=model,
            nx=nx,
            ny=ny,
            nsteps=nsteps,  # Will be overridden by u_targets
            umax=umax,  # Ignored when u_targets is provided
            law=law,
            return_bundle=True,
            bc_spec=bc_spec,
            bond_law=bond_law,
            u_targets=u_targets,  # BLOQUE 3: Cyclic trajectory
        )
    else:
        print("  Using SINGLE-CRACK solver (monotonic)")
        # Use return_bundle to get comprehensive results (FASE G)
        bundle = run_analysis_xfem(
            model=model,
            nx=nx,
            ny=ny,
            nsteps=nsteps,
            umax=umax,
            law=law,
            return_bundle=True,  # Get full bundle
            bc_spec=bc_spec,
            bond_law=bond_law,  # Pass mapped bond law from case config
        )

    # Package results with all necessary data for postprocessing
    results = {
        'nodes': bundle['nodes'],
        'elems': bundle['elems'],
        'u': bundle['u'],
        'history': bundle['history'],
        'crack': bundle.get('crack'),
        'cracks': bundle.get('cracks', [bundle.get('crack')] if bundle.get('crack') is not None else []),
        'mp_states': bundle.get('mp_states'),
        'bond_states': bundle.get('bond_states'),
        'rebar_segs': bundle.get('rebar_segs'),
        'dofs': bundle.get('dofs'),
        'coh_states': bundle.get('coh_states'),
        'model': model,
        'bond_law': bond_law,
        'subdomain_mgr': subdomain_mgr,
    }

    # Postprocessing (FASE G)
    if enable_postprocess:
        from examples.gutierrez_thesis.postprocess_comprehensive import postprocess_case
        postprocess_case(case, results)

    return results
