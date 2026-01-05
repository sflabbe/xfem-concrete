import numpy as np

from xfem_clean.constitutive import DruckerPrager
from xfem_clean.material_point import MaterialPoint
from xfem_clean.numba.kernels_bulk import dp_integrate_plane_stress_numba


def _elastic_ezz_guess(nu: float, eps3: np.ndarray) -> float:
    return -nu / max(1e-12, (1.0 - nu)) * float(eps3[0] + eps3[1])


def test_dp_numba_matches_python_elastic():
    mat = DruckerPrager(E=30e9, nu=0.2, cohesion=3e6, phi_deg=30.0, H=0.0)
    rng = np.random.default_rng(0)
    strains = rng.normal(scale=1e-6, size=(50, 3))

    for eps3 in strains:
        mp = MaterialPoint()
        sigma_py, _ = mat.integrate(mp, eps3)

        eps_p6_old = np.zeros(6, dtype=np.float64)
        ezz = _elastic_ezz_guess(mat.nu, eps3)
        sigma_numba, _, _, _, _, _ = dp_integrate_plane_stress_numba(
            eps3.astype(np.float64),
            eps_p6_old,
            float(ezz),
            0.0,
            float(mat.E),
            float(mat.nu),
            float(mat.alpha),
            float(mat.k0),
            float(mat.H),
        )

        assert np.isfinite(sigma_numba).all(), "Numba DP returned non-finite stress"
        assert np.isfinite(sigma_py).all(), "Python DP returned non-finite stress"
        rel_err = np.linalg.norm(sigma_numba - sigma_py) / max(1.0, np.linalg.norm(sigma_py))
        assert rel_err < 1e-6, f"Elastic DP mismatch (rel err={rel_err:.2e})"


def test_dp_compression_yields_before_tension():
    mat = DruckerPrager(E=30e9, nu=0.2, cohesion=1e6, phi_deg=30.0, H=0.0)
    eps_mag = 4.0e-5

    eps_comp = np.array([-eps_mag, 0.0, 0.0], dtype=float)
    eps_tens = np.array([eps_mag, 0.0, 0.0], dtype=float)

    mp_comp = MaterialPoint()
    mp_tens = MaterialPoint()
    mat.integrate(mp_comp, eps_comp)
    mat.integrate(mp_tens, eps_tens)

    assert mp_comp.kappa > mp_tens.kappa + 1e-12

    eps_p6_old = np.zeros(6, dtype=np.float64)
    ezz_comp = _elastic_ezz_guess(mat.nu, eps_comp)
    ezz_tens = _elastic_ezz_guess(mat.nu, eps_tens)

    _, _, _, _, kappa_comp, _ = dp_integrate_plane_stress_numba(
        eps_comp,
        eps_p6_old,
        float(ezz_comp),
        0.0,
        float(mat.E),
        float(mat.nu),
        float(mat.alpha),
        float(mat.k0),
        float(mat.H),
    )
    _, _, _, _, kappa_tens, _ = dp_integrate_plane_stress_numba(
        eps_tens,
        eps_p6_old,
        float(ezz_tens),
        0.0,
        float(mat.E),
        float(mat.nu),
        float(mat.alpha),
        float(mat.k0),
        float(mat.H),
    )

    assert kappa_comp > kappa_tens + 1e-12
