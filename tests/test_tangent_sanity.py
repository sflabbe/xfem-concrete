
import pytest
import numpy as np
from xfem_clean.cohesive_laws import CohesiveLaw, CohesiveState, cohesive_update_mixed
from xfem_clean.bond_slip import BondSlipModelCode2010
from xfem_clean.constitutive import DruckerPrager
from xfem_clean.material_point import MaterialPoint

def finite_difference_gradient(func, x0, eps=1e-7):
    """Compute FD gradient of vector-valued function func at x0."""
    x0 = np.asarray(x0, dtype=float)
    f0 = np.asarray(func(x0))
    n = len(x0)
    m = len(f0)
    J = np.zeros((m, n))
    for i in range(n):
        x_plus = x0.copy()
        x_plus[i] += eps
        f_plus = np.asarray(func(x_plus))
        J[:, i] = (f_plus - f0) / eps
    return J

def test_cohesive_mixed_mode_tangent():
    """Verify mixed-mode cohesive tangent using FD."""
    law = CohesiveLaw(
        Kn=1e14, # penalty stiffness
        ft=3.0e6,
        Gf=100.0,
        mode="mixed",
        tau_max=4.0e6,
        Gf_II=200.0,
        Kt=1e14,
    )
    
    # State point in softening branch
    delta0 = np.array([0.02e-3, 0.01e-3]) # 0.02mm normal, 0.01mm tangential
    state = CohesiveState()
    
    # Pre-damage it slightly
    base_state = CohesiveState()
    _, _, base_state = cohesive_update_mixed(law, 0.01e-3, 0.005e-3, base_state)
    
    def get_traction(delta):
        # We must use the SAME base_state for all evaluations to check consistency
        st = base_state.copy()
        t, _, _ = cohesive_update_mixed(law, delta[0], delta[1], st)
        return t
        
    t_ana, K_ana, _ = cohesive_update_mixed(law, delta0[0], delta0[1], base_state.copy())
    
    K_fd = finite_difference_gradient(get_traction, delta0, eps=1e-9)
    
    # Tolerance: mixed mode laws can be tricky, allow 1% relative error
    err = np.linalg.norm(K_fd - K_ana) / np.linalg.norm(K_ana)
    print(f"\n[Cohesive] FD Error: {err:.2e}")
    print(f"K_ana:\n{K_ana}")
    print(f"K_fd:\n{K_fd}")
    
    assert err < 1e-2, f"Cohesive tangent mismatch. Rel Error: {err:.2e}"


def test_bond_slip_tangent():
    """Verify bond-slip tangent using FD."""
    law = BondSlipModelCode2010(
        f_cm=30e6,
        d_bar=0.012, # 12mm
        condition="good",
        tangent_mode="consistent"
        # s1, s2, s3, tau_max are derived from f_cm and condition
    )
    # Check derived params matches expectation roughly or set custom
    # Actually CEBFIPBondLaw derives everything. Let's just use it as is.
    
    # Test points in different regimes
    # s1=1mm, s2=2mm for good bond
    slips = [
        0.5e-3, # Ascending (curved)
        1.5e-3, # Plateau
        5.0e-3, # Softening (s3=d_bar=12mm)
    ]
    
    for s_val in slips:
        s0 = np.array([s_val])
        
        def get_tau(s):
            # Testing monotonic envelope tangent
            t, _ = law.tau_and_tangent(float(s[0]), float(abs(s[0])))
            return np.array([t])
            
        tau_ana, D_ana = law.tau_and_tangent(s_val, abs(s_val))
        D_check = finite_difference_gradient(get_tau, s0, eps=1e-8)[0,0]
        
        # Check scalar tangent
        if abs(D_ana) > 1e-5:
            err = abs(D_check - D_ana) / abs(D_ana)
        else:
            err = abs(D_check - D_ana)
            
        print(f"[Bond] s={s_val*1e3:.3f}mm, tau={tau_ana/1e6:.2f}MPa, D_ana={D_ana:.2e}, D_fd={D_check:.2e}, err={err:.2e}")
        assert err < 1e-2, f"Bond tangent mismatch at s={s_val}. Err={err:.2e}"



def test_bulk_dp_tangent():
    """Verify Drucker-Prager plane stress tangent."""
    mat = DruckerPrager(E=30e9, nu=0.2, cohesion=3e6, phi_deg=30.0, plane_stress_tol=1e-12)
    
    # Strain state causing plasticity
    eps0 = np.array([2.0e-4, -0.5e-4, 0.0]) # Plane stress: [exx, eyy, 2exy]
    
    mp0 = MaterialPoint()
    
    # Function for FD
    def get_stress(eps):
        mp = mp0.copy_shallow()
        sig, _ = mat.integrate(mp, eps)
        return sig
    
    # 1. Elastic check (small strain)
    eps_el = np.array([1e-5, -0.3e-5, 0.0])
    sig_el, C_el = mat.integrate(MaterialPoint(), eps_el)
    
    def get_stress_el(eps):
        mp = MaterialPoint()
        sig, _ = mat.integrate(mp, eps)
        return sig
        
    C_el_fd = finite_difference_gradient(get_stress_el, eps_el, eps=1e-8)
    err_el = np.linalg.norm(C_el_fd - C_el) / (np.linalg.norm(C_el) + 1e-10)
    print(f"\n[Bulk DP] Elastic FD Error: {err_el:.2e}")
    assert err_el < 1e-4, f"Bulk DP elastic tangent mismatch. Rel Error: {err_el:.2e}"

    # 2. Plastic check (large strain)
    sig_ana, C_ana = mat.integrate(mp0.copy_shallow(), eps0)
    
    C_fd = finite_difference_gradient(get_stress, eps0, eps=1e-8)
    
    # Check 3x3 tangent
    err = np.linalg.norm(C_fd - C_ana) / (np.linalg.norm(C_ana) + 1e-10)
    print(f"[Bulk DP] Plastic FD Error: {err:.2e}")
    
    if err > 5e-2:
        pytest.fail(f"Bulk DP plastic tangent mismatch. Rel Error: {err:.2e} (Potential Bug)", pytrace=False)
    # assert err < 5e-2, f"Bulk DP tangent mismatch. Rel Error: {err:.2e}"

