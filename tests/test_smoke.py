import numpy as np

from xfem_clean.cohesive_laws import CohesiveLaw, CohesiveState, cohesive_update
from xfem_clean.crack_criteria import principal_max_2d, principal_max_dir


def test_principal_max():
    sig = np.array([1.0, -2.0, 0.0])
    assert principal_max_2d(sig) == 1.0
    s1, v1 = principal_max_dir(sig)
    assert abs(s1 - 1.0) < 1e-12
    assert v1.shape == (2,)


def test_bilinear_cohesive_monotone():
    E = 30e9
    lch = 0.02
    Kn = E / lch
    ft = 2.5e6
    Gf = 100.0
    law = CohesiveLaw(Kn=Kn, ft=ft, Gf=Gf, law="bilinear")
    st = CohesiveState(delta_max=0.0, damage=0.0)
    t, kt, st2 = cohesive_update(law, 0.1*law.delta0, st)
    assert t > 0.0 and kt > 0.0
    t2, kt2, st3 = cohesive_update(law, 1.2*law.delta0, st2)
    assert t2 <= ft * 1.05
    assert st3.delta_max >= st2.delta_max
