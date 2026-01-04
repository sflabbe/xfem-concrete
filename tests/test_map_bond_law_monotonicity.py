from examples.gutierrez_thesis.cases.case_10_wall_c2_cyclic import create_case_10
from examples.gutierrez_thesis.solver_interface import map_bond_law


def test_map_bond_law_enforces_monotonicity():
    case = create_case_10()
    bond_law_config = case.rebar_layers[0].bond_law

    bond_law = map_bond_law(bond_law_config, case_id=case.case_id)

    assert bond_law.s1 > 0.0
    assert bond_law.s1 < bond_law.s2 < bond_law.s3
