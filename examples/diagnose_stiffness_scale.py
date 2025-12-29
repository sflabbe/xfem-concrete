"""Diagnose stiffness scaling bug in Q4 bulk assembly.

PRIORITY #0: Identify why diag(K_bulk) ~ 6 N/m instead of O(1e11) N/m.

Expected stiffness for a single Q4 element:
    K ~ E * t / L

With E=30 GPa, t=0.15 m, L~0.01 m:
    K_expected ~ 30e9 * 0.15 / 0.01 = 4.5e11 N/m

This script:
1. Creates a single Q4 element
2. Computes K_elem analytically
3. Verifies units and scaling factors
4. Reports diagnostic information
"""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xfem_clean.fem.q4 import q4_shape
from xfem_clean.linear_elastic import plane_stress_C


def single_q4_stiffness_analytical():
    """Compute stiffness of a single Q4 element analytically."""

    # Material properties (typical concrete)
    E = 30e9  # [Pa] = 30 GPa
    nu = 0.2
    t = 0.15  # [m] thickness

    # Element geometry (square element)
    L = 0.01  # [m] = 10 mm side length

    # Nodes (counter-clockwise from bottom-left)
    nodes = np.array([
        [0.0, 0.0],
        [L,   0.0],
        [L,   L],
        [0.0, L]
    ], dtype=float)

    print("=" * 70)
    print("STIFFNESS SANITY CHECK: Single Q4 Element")
    print("=" * 70)
    print(f"Material:  E = {E/1e9:.1f} GPa, ν = {nu}")
    print(f"Geometry:  L = {L*1e3:.1f} mm, t = {t*1e3:.0f} mm")
    print("=" * 70)

    # Constitutive matrix
    C = plane_stress_C(E, nu)
    print(f"\nConstitutive matrix C (Voigt ordering [ε_xx, ε_yy, γ_xy]):")
    print(f"C =\n{C}")
    print(f"\nC[0,0] = {C[0,0]:.3e} Pa  (should be E/(1-ν²) = {E/(1-nu**2):.3e} Pa)")

    # Gauss quadrature (2x2)
    gp = 1.0 / np.sqrt(3.0)
    gauss_pts = [
        (-gp, -gp),
        (+gp, -gp),
        (+gp, +gp),
        (-gp, +gp),
    ]
    weights = [1.0, 1.0, 1.0, 1.0]

    # Assemble element stiffness
    Ke = np.zeros((8, 8), dtype=float)  # 4 nodes × 2 DOF/node

    for (xi, eta), w in zip(gauss_pts, weights):
        # Shape functions and derivatives
        N, dN_dxi, dN_deta = q4_shape(xi, eta)

        # Jacobian: J = [dx/dξ, dy/dξ; dx/dη, dy/dη]
        # For a rectangular element: J = [L/2, 0; 0, L/2]
        J = np.zeros((2, 2), dtype=float)
        for a in range(4):
            J[0, 0] += dN_dxi[a] * nodes[a, 0]   # dx/dξ
            J[0, 1] += dN_dxi[a] * nodes[a, 1]   # dy/dξ
            J[1, 0] += dN_deta[a] * nodes[a, 0]  # dx/dη
            J[1, 1] += dN_deta[a] * nodes[a, 1]  # dy/dη

        detJ = np.linalg.det(J)
        print(f"\nGauss point (ξ={xi:+.3f}, η={eta:+.3f}):")
        print(f"  J =\n{J}")
        print(f"  det(J) = {detJ:.6e}  [m²] (expected L²/4 = {(L**2/4):.6e})")

        if abs(detJ) < 1e-12:
            raise ValueError(f"Singular Jacobian: det(J) = {detJ}")

        # Inverse Jacobian
        Jinv = np.linalg.inv(J)

        # Shape function derivatives in physical coordinates
        dN_dx = np.zeros(4, dtype=float)
        dN_dy = np.zeros(4, dtype=float)
        for a in range(4):
            dN_dx[a] = Jinv[0, 0] * dN_dxi[a] + Jinv[0, 1] * dN_deta[a]
            dN_dy[a] = Jinv[1, 0] * dN_dxi[a] + Jinv[1, 1] * dN_deta[a]

        # B-matrix (strain-displacement): ε = B u
        # For plane stress: ε = [ε_xx, ε_yy, γ_xy]ᵀ
        B = np.zeros((3, 8), dtype=float)
        for a in range(4):
            B[0, 2*a]   = dN_dx[a]  # ε_xx = ∂u/∂x
            B[1, 2*a+1] = dN_dy[a]  # ε_yy = ∂v/∂y
            B[2, 2*a]   = dN_dy[a]  # γ_xy = ∂u/∂y + ∂v/∂x
            B[2, 2*a+1] = dN_dx[a]

        # Integration weight
        wgt = w * detJ * t  # [m³]
        print(f"  weight = w × det(J) × t = {w} × {detJ:.3e} × {t:.3e} = {wgt:.6e} [m³]")

        # Element stiffness contribution: Ke += Bᵀ C B × wgt
        Ke += (B.T @ C @ B) * wgt

        # Check B matrix scale
        print(f"  B[0,0] (∂N1/∂x) = {B[0,0]:.3e} [1/m]  (expected ~1/L = {1/L:.3e})")

    print("\n" + "=" * 70)
    print("ELEMENT STIFFNESS MATRIX Ke (8×8)")
    print("=" * 70)
    print(f"Ke diagonal entries:")
    for i in range(8):
        dof_type = 'x' if i % 2 == 0 else 'y'
        node_id = i // 2
        print(f"  Ke[{i},{i}] (node {node_id}, DOF {dof_type}) = {Ke[i,i]:.6e} [N/m]")

    # Expected order of magnitude
    K_expected = E * t / L
    print(f"\n" + "=" * 70)
    print("ORDER OF MAGNITUDE CHECK")
    print("=" * 70)
    print(f"K_expected ~ E × t / L = {E:.2e} × {t:.2e} / {L:.2e}")
    print(f"           = {K_expected:.2e} N/m")
    print(f"K_actual   = {np.median(np.diag(Ke)):.2e} N/m (median diagonal)")
    print(f"Ratio      = {np.median(np.diag(Ke)) / K_expected:.3f}")

    if np.median(np.diag(Ke)) < 1e6:
        print("\n⚠️  WARNING: Stiffness is ABSURDLY LOW!")
        print("    Expected O(1e11) N/m, got O(1e0) N/m")
        print("    Possible causes:")
        print("    - Units mismatch (E in MPa instead of Pa?)")
        print("    - Missing thickness factor in assembly")
        print("    - Jacobian not computed correctly")
        print("    - B-matrix scaled incorrectly")
    elif abs(np.median(np.diag(Ke)) / K_expected - 1.0) > 0.5:
        print(f"\n⚠️  WARNING: Stiffness differs from expected by {abs(np.median(np.diag(Ke)) / K_expected - 1.0)*100:.0f}%")
    else:
        print("\n✓ Stiffness scale looks correct!")

    print("=" * 70)

    return Ke, nodes


def test_assembly_routine():
    """Test the actual assembly routine from assembly_single.py."""

    print("\n" + "=" * 70)
    print("TEST: Actual Assembly Routine")
    print("=" * 70)

    from xfem_clean.xfem.model import XFEMModel
    from xfem_clean.xfem.dofs_single import build_xfem_dofs
    from xfem_clean.xfem.geometry import XFEMCrack
    from xfem_clean.xfem.assembly_single import assemble_xfem_system
    from xfem_clean.cohesive_laws import CohesiveLaw

    # Simple 1×1 element mesh
    L = 0.01  # [m]
    nodes = np.array([
        [0.0, 0.0],
        [L,   0.0],
        [L,   L],
        [0.0, L]
    ], dtype=float)
    elems = np.array([[0, 1, 2, 3]], dtype=int)

    # Material properties
    E = 30e9  # [Pa]
    nu = 0.2
    thickness = 0.15  # [m]

    C = plane_stress_C(E, nu)

    # DOFs (no enrichment, no steel)
    dofs = build_xfem_dofs(
        nodes=nodes,
        enriched_nodes=[],
        tip_enriched_nodes=[],
        rebar_segs=None,
    )

    # No crack
    crack = XFEMCrack(x0=0.0, y0=0.0, angle=0.0, length=0.0)
    crack.active = False

    # Zero displacement
    q = np.zeros(dofs.ndof, dtype=float)

    # Dummy cohesive law
    law = CohesiveLaw(Kn=1e10, ft=2.9e6, Gf=120)

    # Assemble
    K, fint, _, _, _, _ = assemble_xfem_system(
        nodes=nodes,
        elems=elems,
        dofs=dofs,
        crack=crack,
        C=C,
        thickness=thickness,
        q=q,
        law=law,
        coh_states_comm={},
        tip_enr_radius=0.0,
        use_numba=False,
    )

    K_dense = K.toarray()

    print(f"Assembly result:")
    print(f"  ndof = {dofs.ndof}")
    print(f"  K.shape = {K.shape}")
    print(f"  nnz(K) = {K.nnz}")
    print(f"\nK diagonal entries:")
    for i in range(min(8, dofs.ndof)):
        dof_type = 'x' if i % 2 == 0 else 'y'
        node_id = i // 2
        print(f"  K[{i},{i}] (node {node_id}, DOF {dof_type}) = {K_dense[i,i]:.6e} [N/m]")

    K_expected = E * thickness / L
    print(f"\nExpected diagonal ~ {K_expected:.2e} N/m")
    print(f"Actual median     = {np.median(np.diag(K_dense)[:8]):.2e} N/m")
    print(f"Ratio             = {np.median(np.diag(K_dense)[:8]) / K_expected:.3f}")

    if np.median(np.diag(K_dense)[:8]) < 1e6:
        print("\n⚠️  BUG CONFIRMED: Assembly produces absurdly low stiffness!")
        print("    This is the root cause of Newton failure at u≈78 nm.")
    else:
        print("\n✓ Assembly stiffness looks reasonable.")

    print("=" * 70)

    return K_dense


def main():
    """Run stiffness diagnostics."""

    print("\n" * 2)

    # Test 1: Analytical single element
    Ke_analytical, _ = single_q4_stiffness_analytical()

    # Test 2: Actual assembly routine
    K_assembly = test_assembly_routine()

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Check if thickness is applied correctly in assembly")
    print("2. Verify detJ calculation (should be in m², not mm²)")
    print("3. Check if E is in Pa (not MPa)")
    print("4. Verify B-matrix construction (dN/dx in 1/m)")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
