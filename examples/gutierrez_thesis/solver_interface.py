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

    elif "wall" in case_name:
        # WALL TEST (RC wall under cyclic lateral loading):
        # - Fix base (y=0): uy=0 for all nodes, ux=0 for one corner node (prevent rigid body)
        # - Prescribe top displacement: ux=u_target for nodes in rigid beam zone
        # - Axial load: TODO - implement as constant fext term

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

    return BCSpec(
        fixed_dofs=fixed_dofs,
        prescribed_dofs=prescribed_dofs,
        prescribed_scale=prescribed_scale,
        reaction_dofs=reaction_dofs,
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

    # Steel properties (use first rebar layer if available)
    if case.rebar_layers:
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
    else:
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

    # Bond-slip configuration
    enable_bond_slip = len(case.rebar_layers) > 0

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

        # Solver parameters
        newton_maxit=25,
        newton_tol_r=case.tolerance,
        newton_tol_du=1e-8,
        line_search=case.use_line_search,
        enable_diagonal_scaling=True,
        max_subdiv=12 if case.use_substepping else 0,
        max_total_substeps=50000,

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

    # Create cohesive law (if using cohesive cracks)
    # For pull-out tests without cracks, we still need a cohesive law object
    # but it won't be activated
    law = CohesiveLaw(
        Kn=model.Kn_factor * model.E / max(1e-12, model.lch),
        ft=model.ft,
        Gf=model.Gf,
    )

    # Map bond law (if rebar present)
    bond_law = None
    if case.rebar_layers:
        # Use first rebar layer's bond law
        bond_law = map_bond_law(case.rebar_layers[0].bond_law)

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

    # Prepare rebar segments for BC mapping
    from xfem_clean.rebar import prepare_rebar_segments
    rebar_segs = prepare_rebar_segments(nodes, cover=model.cover) if case.rebar_layers else None

    # Build boundary conditions from case configuration
    bc_spec = build_bcs_from_case(case, nodes, model, rebar_segs=rebar_segs)

    # TODO: Handle FRP sheets (FASE E)
    # TODO: Handle fibres (FASE E)

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
