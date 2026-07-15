import pytest

from examples.gutierrez_thesis.case_config import CEBFIPBondLaw, CaseConfigurationError
from examples.gutierrez_thesis.solver_interface import (
    map_bond_law,
    _reset_warned_cases_for_tests,
)


def _cfg(s1: float, s2: float, s3: float) -> CEBFIPBondLaw:
    return CEBFIPBondLaw(
        s1=s1,
        s2=s2,
        s3=s3,
        tau_max=10.0,
        tau_f=2.0,
        alpha=0.4,
    )


def test_bond_slip_schema_is_unambiguously_mm() -> None:
    law = map_bond_law(_cfg(1.0, 2.0, 12.0), case_id="case_X")
    assert law.s1 == pytest.approx(1.0e-3)
    assert law.s2 == pytest.approx(2.0e-3)
    assert law.s3 == pytest.approx(12.0e-3)


def test_invalid_bond_slip_order_is_rejected() -> None:
    with pytest.raises(CaseConfigurationError, match="bond_law.slip_mm"):
        map_bond_law(_cfg(2.0, 1.0, 12.0), case_id="case_bad")
