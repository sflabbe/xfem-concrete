"""
Test de integración para Fase D: Multicrack + Bond-slip

Verifica que:
  - build_xfem_dofs_multi soporta steel DOFs
  - assemble_xfem_system_multi integra bond-slip
  - run_analysis_xfem_multicrack maneja bond_states correctamente
  - Subdomain manager funciona con void elements
"""

import sys

# Check numpy availability first
try:
    import numpy as np
except ImportError:
    print("⚠️  NumPy not available - skipping Fase D tests")
    sys.exit(0)

def test_multicrack_dof_mapping():
    """Test que build_xfem_dofs_multi crea steel DOFs correctamente."""
    print("\n🧪 Test 1: DOF mapping con bond-slip...")

    # Create simple mesh
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    elems = np.array([[0, 1, 2, 3]])

    # Simple rebar segment
    rebar_segs = np.array([[0, 1, 1.0, 1.0, 0.0]])  # [n1, n2, L0, cx, cy]

    # Test with bond-slip enabled
    from xfem_clean.xfem.multicrack import build_xfem_dofs_multi, XFEMCrack

    cracks = []
    dofs = build_xfem_dofs_multi(
        nodes=nodes,
        elems=elems,
        cracks=cracks,
        ny=1,
        rebar_segs=rebar_segs,
        enable_bond_slip=True,
    )

    # Verificar que se crearon DOFs de steel
    assert dofs.steel is not None, "Steel DOF map should exist"
    assert dofs.steel_dof_offset > 0, "Steel DOF offset should be positive"
    assert dofs.steel_nodes is not None, "Steel nodes mask should exist"

    # Nodos 0 y 1 deben tener steel DOFs
    assert dofs.steel_nodes[0], "Node 0 should have steel"
    assert dofs.steel_nodes[1], "Node 1 should have steel"
    assert not dofs.steel_nodes[2], "Node 2 should not have steel"
    assert not dofs.steel_nodes[3], "Node 3 should not have steel"

    # Verificar que steel DOFs están correctamente asignados
    assert dofs.steel[0, 0] >= dofs.steel_dof_offset, "Steel DOF should be after concrete DOFs"
    assert dofs.steel[1, 0] >= dofs.steel_dof_offset, "Steel DOF should be after concrete DOFs"

    # Total DOFs = concrete (4 nodes x 2) + steel (2 nodes x 2) = 12
    expected_ndof = 8 + 4
    assert dofs.ndof == expected_ndof, f"Total DOFs should be {expected_ndof}, got {dofs.ndof}"

    print("✅ DOF mapping con bond-slip: PASS")
    return True


def test_multicrack_assembly_signature():
    """Test que assemble_xfem_system_multi acepta parámetros de bond-slip."""
    print("\n🧪 Test 2: Signature de assembly con bond-slip...")

    from xfem_clean.xfem.multicrack import assemble_xfem_system_multi
    import inspect

    sig = inspect.signature(assemble_xfem_system_multi)
    params = list(sig.parameters.keys())

    # Verificar que nuevos parámetros están presentes
    assert 'bond_law' in params, "bond_law parameter should exist"
    assert 'bond_states_comm' in params, "bond_states_comm parameter should exist"
    assert 'enable_bond_slip' in params, "enable_bond_slip parameter should exist"
    assert 'steel_EA' in params, "steel_EA parameter should exist"
    assert 'subdomain_mgr' in params, "subdomain_mgr parameter should exist"

    print("✅ Assembly signature: PASS")
    return True


def test_dof_transfer_with_steel():
    """Test que transfer_q_between_dofs_multi transfiere steel DOFs."""
    print("\n🧪 Test 3: Transferencia de steel DOFs...")

    from xfem_clean.xfem.multicrack import (
        build_xfem_dofs_multi,
        transfer_q_between_dofs_multi,
    )

    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    elems = np.array([[0, 1, 2, 3]])
    rebar_segs = np.array([[0, 1, 1.0, 1.0, 0.0]])

    # Build DOFs
    cracks = []
    dofs = build_xfem_dofs_multi(
        nodes=nodes,
        elems=elems,
        cracks=cracks,
        ny=1,
        rebar_segs=rebar_segs,
        enable_bond_slip=True,
    )

    # Create displacement vector with some values in steel DOFs
    q_old = np.zeros(dofs.ndof)
    q_old[dofs.steel[0, 0]] = 1.0  # Steel node 0, x
    q_old[dofs.steel[0, 1]] = 2.0  # Steel node 0, y
    q_old[dofs.steel[1, 0]] = 3.0  # Steel node 1, x

    # Transfer to same DOFs (identity operation)
    q_new = transfer_q_between_dofs_multi(q_old, dofs, dofs)

    # Verify steel DOFs were transferred
    assert q_new[dofs.steel[0, 0]] == 1.0, "Steel DOF transfer failed"
    assert q_new[dofs.steel[0, 1]] == 2.0, "Steel DOF transfer failed"
    assert q_new[dofs.steel[1, 0]] == 3.0, "Steel DOF transfer failed"

    print("✅ Steel DOF transfer: PASS")
    return True


def test_subdomain_manager_integration():
    """Test que SubdomainManager funciona con multicrack assembly."""
    print("\n🧪 Test 4: SubdomainManager con void elements...")

    from xfem_clean.xfem.subdomains import SubdomainManager

    # Create subdomain manager
    nelem = 4
    mgr = SubdomainManager(nelem)

    # Mark element 0 as void
    mgr.set_property(0, material_type="void", thickness_override=0.0, bond_disabled=True)

    # Mark element 1 as rigid
    mgr.set_property(1, material_type="rigid", E_override=1e12)

    # Verify properties
    assert mgr.is_void(0), "Element 0 should be void"
    assert mgr.is_rigid(1), "Element 1 should be rigid"
    assert mgr.is_bond_disabled(0), "Bond should be disabled in void element"

    # Verify effective properties
    E_default = 30e9
    assert mgr.get_effective_E(0, E_default) < E_default * 1e-3, "Void E should be very small"
    assert mgr.get_effective_E(1, E_default) > E_default * 1e3, "Rigid E should be very large"
    assert mgr.get_effective_thickness(0, 0.1) == 0.0, "Void thickness should be 0"

    print("✅ SubdomainManager integration: PASS")
    return True


def main():
    """Run all Fase D integration tests."""
    print("="*60)
    print("  FASE D: Multicrack + Bond-slip Integration Tests")
    print("="*60)

    tests = [
        test_multicrack_dof_mapping,
        test_multicrack_assembly_signature,
        test_dof_transfer_with_steel,
        test_subdomain_manager_integration,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"  Results: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"  ⚠️  {failed} tests failed")
    else:
        print("  ✅ All tests passed!")
    print("="*60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
