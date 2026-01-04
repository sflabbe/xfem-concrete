import warnings

from examples.gutierrez_thesis.case_config import CEBFIPBondLaw
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


def _ambiguous_warnings(warnings_list: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [
        warning
        for warning in warnings_list
        if issubclass(warning.category, RuntimeWarning)
        and "Ambiguous bond-slip units" in str(warning.message)
    ]


def test_ambiguous_units_warn_once_per_case() -> None:
    _reset_warned_cases_for_tests()
    cfg1 = _cfg(1.0, 2.0, 12.0)
    cfg2 = _cfg(0.9, 1.8, 10.0)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        map_bond_law(cfg1, case_id="case_X")
        map_bond_law(cfg2, case_id="case_X")

    assert len(_ambiguous_warnings(captured)) == 1


def test_ambiguous_units_warns_for_different_cases() -> None:
    _reset_warned_cases_for_tests()
    cfg1 = _cfg(1.0, 2.0, 12.0)
    cfg2 = _cfg(0.9, 1.8, 10.0)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        map_bond_law(cfg1, case_id="case_A")
        map_bond_law(cfg2, case_id="case_B")

    assert len(_ambiguous_warnings(captured)) == 2
