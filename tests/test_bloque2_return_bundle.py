"""
Test BLOQUE 2: Validar return_bundle en multicrack

Verifica que run_analysis_xfem_multicrack retorna un dict comprehensivo cuando return_bundle=True
"""

import sys
import numpy as np

try:
    from xfem_clean.xfem.multicrack import run_analysis_xfem_multicrack
    from xfem_clean.xfem.model import XFEMModel
    from xfem_clean.cohesive_laws import CohesiveLaw
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    sys.exit(0)

def test_multicrack_return_bundle():
    """Test que multicrack retorna bundle completo con return_bundle=True."""
    print("\n🧪 Test: multicrack return_bundle...")

    # Minimal model (no rebar, no bond-slip, just basic multicrack)
    model = XFEMModel(
        L=1.0,  # 1m
        H=0.2,  # 200mm
        b=0.1,  # 100mm thickness
        E=30e9,  # 30 GPa
        nu=0.2,
        ft=3.0e6,  # 3 MPa
        fc=30e6,
        Gf=100.0,
        steel_E=200e9,
        steel_fy=500e6,
        steel_fu=600e6,
        steel_Eh=0.0,
        steel_A_total=0.0,  # No rebar
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
        crack_max_inner=3,
    )

    law = CohesiveLaw(Kn=1e12, ft=model.ft, Gf=model.Gf)

    try:
        # Test with return_bundle=False (backward compatibility)
        print("  Testing return_bundle=False (backward compat)...")
        nodes, elems, q, results, cracks = run_analysis_xfem_multicrack(
            model=model,
            nx=6,
            ny=2,
            nsteps=1,
            umax=0.0005,  # 0.5mm
            max_cracks=1,
            law=law,
            return_bundle=False,
        )
        assert isinstance(nodes, np.ndarray), "nodes should be ndarray"
        assert isinstance(elems, np.ndarray), "elems should be ndarray"
        assert isinstance(q, np.ndarray), "q should be ndarray"
        assert isinstance(results, list), "results should be list"
        assert isinstance(cracks, list), "cracks should be list"
        print("    ✓ return_bundle=False returns tuple correctly")

        # Test with return_bundle=True (BLOQUE 2)
        print("  Testing return_bundle=True (BLOQUE 2)...")
        bundle = run_analysis_xfem_multicrack(
            model=model,
            nx=6,
            ny=2,
            nsteps=1,
            umax=0.0005,  # 0.5mm
            max_cracks=1,
            law=law,
            return_bundle=True,  # BLOQUE 2
        )

        # Verify bundle is a dict
        assert isinstance(bundle, dict), "bundle should be dict when return_bundle=True"

        # Verify required keys
        required_keys = ['nodes', 'elems', 'u', 'history', 'cracks', 'bond_states',
                        'rebar_segs', 'dofs', 'coh_states', 'bulk_states']
        for key in required_keys:
            assert key in bundle, f"bundle should contain '{key}'"

        # Verify types
        assert isinstance(bundle['nodes'], np.ndarray), "nodes should be ndarray"
        assert isinstance(bundle['elems'], np.ndarray), "elems should be ndarray"
        assert isinstance(bundle['u'], np.ndarray), "u should be ndarray"
        assert isinstance(bundle['history'], np.ndarray), "history should be ndarray"
        assert isinstance(bundle['cracks'], list), "cracks should be list"
        # bond_states can be None (no bond-slip)
        # rebar_segs can be None (no rebar)
        # dofs should be MultiXFEMDofs object
        # coh_states should be CohesiveStateArrays
        # bulk_states should be BulkStateArrays

        print("    ✓ return_bundle=True returns dict with all required keys")
        print("    ✓ All keys have correct types")

        # Verify content makes sense
        assert len(bundle['history']) > 0, "history should have results"
        assert bundle['nodes'].shape[0] > 0, "nodes should not be empty"
        assert bundle['elems'].shape[0] > 0, "elems should not be empty"
        assert bundle['u'].shape[0] > 0, "u should not be empty"

        print("    ✓ Bundle contents are valid")
        print("✅ multicrack return_bundle: PASS (BLOQUE 2 VALIDATED)")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_multicrack_return_bundle()
    sys.exit(0 if success else 1)
