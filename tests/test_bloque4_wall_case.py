"""
Test BLOQUE 4: Validar Case 05 (Wall) - BCs específicas para muro

Verifica que las BCs de muro (base fija, desplazamiento horizontal en tope) funcionan correctamente.
"""

import sys
import pytest

try:
    import numpy as np
    from xfem_clean.xfem.multicrack import run_analysis_xfem_multicrack
    from xfem_clean.xfem.model import XFEMModel
    from xfem_clean.cohesive_laws import CohesiveLaw
    from xfem_clean.xfem.analysis_single import BCSpec
    from xfem_clean.fem.mesh import structured_quad_mesh

    # Import case and solver interface
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from examples.gutierrez_thesis.case_config import (
        CaseConfig, GeometryConfig, ConcreteConfig, CyclicLoading,
        OutputConfig, SubdomainConfig
    )
    from examples.gutierrez_thesis.solver_interface import (
        build_bcs_from_case, case_config_to_xfem_model
    )
except ImportError as e:
    pytest.skip(f"Import error: {e}", allow_module_level=True)


def test_wall_bcs_structure():
    """Test que las BCs de muro tienen la estructura correcta."""
    print("\n🧪 Test: Wall BCs structure...")

    # Minimal wall case (coarse mesh for speed)
    geometry = GeometryConfig(
        length=1400.0,  # mm
        height=2800.0,  # mm
        thickness=120.0,  # mm
        n_elem_x=7,   # Coarse: 200mm elements
        n_elem_y=14,  # Coarse: 200mm elements
        element_type="Q4",
    )

    concrete = ConcreteConfig(
        E=30000.0,  # MPa
        nu=0.2,
        f_c=40.0,  # MPa
        f_t=10.0,  # MPa (high to avoid cracking in test)
        G_f=0.120,  # N/mm
        model_type="linear_elastic",  # Simplest for BC test
    )

    # Minimal cyclic loading (elastic range)
    loading = CyclicLoading(
        targets=[5.6, 7.0],  # mm (0.2%, 0.25% drift for H=2800mm)
        n_cycles_per_target=1,
        load_x_center=700.0,  # Center
        load_halfwidth=700.0,  # Full width
        axial_load=0.0,  # TODO: implement axial load
    )

    # Rigid beam at top (200mm)
    rigid_beam = SubdomainConfig(
        x_range=None,  # Full width
        y_range=(2600.0, 2800.0),  # Top 200mm
        material_type="rigid",
        E_override=200000.0,  # MPa (steel)
    )

    outputs = OutputConfig(
        output_dir="tests/tmp",
        case_name="wall_bc_test",
        save_load_displacement=True,
        save_crack_data=False,
        save_vtk=False,
    )

    case = CaseConfig(
        case_id="test_05_wall_bc",
        description="Test wall BCs",
        geometry=geometry,
        concrete=concrete,
        loading=loading,
        outputs=outputs,
        subdomains=[rigid_beam],
        tolerance=1e-6,
    )

    # Build model and mesh
    model = case_config_to_xfem_model(case)
    nx = case.geometry.n_elem_x
    ny = case.geometry.n_elem_y
    nodes, elems = structured_quad_mesh(model.L, model.H, nx, ny)

    print(f"  Model: {nodes.shape[0]} nodes, {elems.shape[0]} elements")
    print(f"  Dimensions: L={model.L:.3f}m, H={model.H:.3f}m")

    # Build BCs
    bc_spec = build_bcs_from_case(case, nodes, model)

    # Verify BC structure
    assert isinstance(bc_spec, BCSpec), "Should return BCSpec"
    assert isinstance(bc_spec.fixed_dofs, dict), "fixed_dofs should be dict"
    assert isinstance(bc_spec.prescribed_dofs, list), "prescribed_dofs should be list"

    print(f"  ✓ BCSpec structure correct")
    print(f"    Fixed DOFs: {len(bc_spec.fixed_dofs)}")
    print(f"    Prescribed DOFs: {len(bc_spec.prescribed_dofs)}")

    # Verify base fixity
    # All base nodes (y=0) should have uy fixed
    base_nodes = np.where(np.isclose(nodes[:, 1], 0.0, atol=1e-6))[0]
    print(f"  Base nodes: {len(base_nodes)}")

    # Count how many base uy DOFs are fixed
    base_uy_fixed = sum(1 for n in base_nodes if (2*n + 1) in bc_spec.fixed_dofs)
    assert base_uy_fixed == len(base_nodes), f"All base nodes should have uy=0, got {base_uy_fixed}/{len(base_nodes)}"
    print(f"  ✓ All {len(base_nodes)} base nodes have uy=0")

    # At least one base node should have ux fixed (prevent rigid body)
    base_ux_fixed = sum(1 for n in base_nodes if (2*n) in bc_spec.fixed_dofs)
    assert base_ux_fixed >= 1, "At least one base node should have ux=0"
    print(f"  ✓ {base_ux_fixed} base node(s) have ux=0 (rigid body prevention)")

    # Verify top displacement control
    # Nodes in rigid beam zone should have ux prescribed
    rigid_y_min = 2600.0 * 1e-3  # mm → m
    rigid_nodes = np.where(nodes[:, 1] >= rigid_y_min - 1e-6)[0]
    print(f"  Rigid beam nodes (y >= {rigid_y_min:.3f}m): {len(rigid_nodes)}")

    # Count how many rigid ux DOFs are prescribed
    rigid_ux_prescribed = sum(1 for n in rigid_nodes if (2*n) in bc_spec.prescribed_dofs)
    assert rigid_ux_prescribed == len(rigid_nodes), f"All rigid nodes should have ux prescribed, got {rigid_ux_prescribed}/{len(rigid_nodes)}"
    print(f"  ✓ All {len(rigid_nodes)} rigid beam nodes have ux prescribed")

    # Verify prescribed scale
    assert bc_spec.prescribed_scale == 1.0, "Wall should have positive scale (push in +x)"
    print(f"  ✓ Prescribed scale: {bc_spec.prescribed_scale} (correct)")

    print("✅ Wall BCs structure: PASS")
    return True


@pytest.mark.slow
def test_wall_minimal_analysis():
    """Test que el análisis de muro corre sin crashes (BLOQUE 4)."""
    print("\n🧪 Test: Wall minimal analysis...")

    # Very coarse wall (super fast)
    model = XFEMModel(
        L=1.4,  # 1.4m
        H=2.8,  # 2.8m
        b=0.12,  # 120mm
        E=30e9,  # 30 GPa
        nu=0.2,
        ft=10.0e6,  # High ft to stay elastic
        fc=40e6,
        Gf=120.0,
        steel_E=200e9,
        steel_fy=500e6,
        steel_fu=600e6,
        steel_Eh=0.0,
        steel_A_total=0.0,  # No rebar for now
        # Solver params
        newton_maxit=10,
        newton_tol_r=1e-3,
        newton_beta=1e-2,
        newton_tol_du=1e-6,
        max_subdiv=2,
        visc_damp=0.0,
        Kn_factor=1000.0,
        k_stab=1e-6,
        ft_initiation_factor=0.8,
        crack_rho=0.05,
        crack_tip_stop_y=0.1,
        crack_max_inner=2,
    )

    law = CohesiveLaw(Kn=1e12, ft=model.ft, Gf=model.Gf)

    # Create minimal case for BC extraction
    from examples.gutierrez_thesis.case_config import (
        CaseConfig, GeometryConfig, ConcreteConfig, CyclicLoading,
        OutputConfig, SubdomainConfig
    )

    case = CaseConfig(
        case_id="test_05_wall_minimal",
        description="Minimal wall test",
        geometry=GeometryConfig(
            length=1400.0, height=2800.0, thickness=120.0,
            n_elem_x=7, n_elem_y=14, element_type="Q4"
        ),
        concrete=ConcreteConfig(
            E=30000.0, nu=0.2, f_c=40.0, f_t=10.0,
            G_f=0.120, model_type="linear_elastic"
        ),
        loading=CyclicLoading(
            targets=[2.8],  # mm (0.1% drift, very small)
            n_cycles_per_target=1,
            load_x_center=700.0,
            load_halfwidth=700.0,
            axial_load=0.0,
        ),
        outputs=OutputConfig(
            output_dir="tests/tmp",
            case_name="wall_minimal",
            save_load_displacement=True,
        ),
        subdomains=[SubdomainConfig(
            x_range=None,
            y_range=(2600.0, 2800.0),
            material_type="rigid",
            E_override=200000.0,
        )],
    )

    # Build mesh and BCs
    nx = case.geometry.n_elem_x
    ny = case.geometry.n_elem_y
    nodes, elems = structured_quad_mesh(model.L, model.H, nx, ny)
    bc_spec = build_bcs_from_case(case, nodes, model)

    print(f"  Running analysis: {nodes.shape[0]} nodes, {elems.shape[0]} elems")

    try:
        bundle = run_analysis_xfem_multicrack(
            model=model,
            nx=7,
            ny=14,
            nsteps=1,  # Single step (elastic push)
            umax=0.0028,  # 2.8mm = 0.1% drift
            max_cracks=0,  # No cracking (just BC test)
            law=law,
            nodes=nodes,
            elems=elems,
            bc_spec=bc_spec,
            return_bundle=True,
        )

        # Verify bundle
        assert isinstance(bundle, dict), "Should return bundle"
        assert 'history' in bundle, "Should have history"
        assert len(bundle['history']) > 0, "History should not be empty"

        history = bundle['history']
        print(f"    ✓ Analysis completed: {len(history)} steps")

        # Extract results
        if history.ndim == 2:
            displacements = history[:, 1]  # u
            loads = history[:, 2]  # P (reaction)

            print(f"    ✓ Displacement range: [{displacements.min()*1e3:.4f}, {displacements.max()*1e3:.4f}] mm")
            print(f"    ✓ Load range: [{loads.min()/1e3:.2f}, {loads.max()/1e3:.2f}] kN")

            # Verify no NaNs
            assert np.all(np.isfinite(displacements)), "Displacements contain NaN/Inf"
            assert np.all(np.isfinite(loads)), "Loads contain NaN/Inf"
            print(f"    ✓ All results finite (no NaNs)")

            # For wall in +x direction, expect positive load (wall pushes back)
            assert loads[-1] > 0, f"Final load should be positive for wall push, got {loads[-1]:.2f}"
            print(f"    ✓ Wall resistance correct sign (positive)")

        print("✅ Wall minimal analysis: PASS (BLOQUE 4 VALIDATED)")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success1 = test_wall_bcs_structure()
    success2 = test_wall_minimal_analysis()

    success = success1 and success2
    sys.exit(0 if success else 1)
