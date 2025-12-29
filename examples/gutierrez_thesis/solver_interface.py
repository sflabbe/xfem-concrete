"""
Solver Interface for Gutiérrez Thesis Cases

Provides adapters to convert CaseConfig to XFEM solver inputs and execute simulations.
"""

from typing import Tuple, Optional, Any, Dict
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
from xfem_clean.xfem.analysis_single import run_analysis_xfem
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
# SOLVER EXECUTION
# =============================================================================

def run_case_solver(
    case: CaseConfig,
    mesh_factor: float = 1.0,
) -> Dict[str, Any]:
    """
    Run XFEM solver for a thesis case.

    Parameters
    ----------
    case : CaseConfig
        Thesis case configuration
    mesh_factor : float
        Mesh refinement factor (1.0 = default)

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

    # Extract loading parameters
    if hasattr(case.loading, 'max_displacement'):
        # Monotonic loading
        umax = case.loading.max_displacement * 1e-3  # mm → m
        nsteps = case.loading.n_steps
    else:
        raise NotImplementedError("Only monotonic loading supported for now")

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

    # Build subdomain manager (FASE C)
    subdomain_mgr = None
    if case.subdomains:
        # Create mesh first (needed for subdomain assignment)
        nodes, elems = structured_quad_mesh(model.L, model.H, nx, ny)
        subdomain_mgr = build_subdomain_manager_from_config(nodes, elems, case.subdomains)
        print(f"Subdomain manager created: {len(case.subdomains)} subdomain(s)")
    else:
        nodes, elems = None, None

    # TODO: Handle FRP sheets (FASE E)
    # TODO: Handle fibres (FASE E)
    # TODO: Handle cyclic loading (FASE F)

    # Run solver
    print(f"Running solver: nx={nx}, ny={ny}, nsteps={nsteps}, umax={umax*1e3:.3f} mm")

    # Note: run_analysis_xfem creates mesh internally if nodes/elems not passed
    # We need to modify it to accept pre-created mesh and subdomain_mgr
    # For now, we'll store subdomain_mgr in model for access during assembly
    if subdomain_mgr is not None:
        model.subdomain_mgr = subdomain_mgr

    nodes_out, elems_out, u, history, crack = run_analysis_xfem(
        model=model,
        nx=nx,
        ny=ny,
        nsteps=nsteps,
        umax=umax,
        law=law,
        return_states=False,
    )

    # Package results
    results = {
        'nodes': nodes_out,
        'elems': elems_out,
        'u': u,
        'history': history,
        'crack': crack,
        'model': model,
        'bond_law': bond_law,
        'subdomain_mgr': subdomain_mgr,
    }

    return results
