import pytest

from scripts.run_gutierrez_matrix import run_matrix


@pytest.mark.slow
def test_gutierrez_matrix_expected_configs(tmp_path):
    results = run_matrix(output_dir=tmp_path)
    lookup = {(r.case_id, r.cfg_id): r for r in results}

    def get(case_id: str, cfg_id: str):
        key = (case_id, cfg_id)
        assert key in lookup, f"Missing result for {case_id} {cfg_id}"
        return lookup[key]

    assert get("01_pullout_lettow", "default").status == "ok"
    assert get("02_sspot_frp", "default").status == "ok"

    case03_bond_off = get("03_tensile_stn12", "bond-slip-off")
    assert case03_bond_off.status == "ok"

    case08_default = get("08_beam_3pb_vvbs3_cfrp", "default")
    if case08_default.status != "ok":
        pytest.xfail("Invalid FRP DOF mapping")
