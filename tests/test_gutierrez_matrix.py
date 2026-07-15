import pytest

from scripts.run_gutierrez_matrix import run_matrix


@pytest.mark.slow
def test_gutierrez_matrix_expected_configs(tmp_path):
    results = run_matrix(
        output_dir=tmp_path,
        mesh_factor=0.25,
        max_steps=2,
        cases=(
            "01_pullout_lettow",
            "02_sspot_frp",
            "03_tensile_stn12",
            "08_beam_3pb_vvbs3_cfrp",
        ),
    )
    lookup = {(r.case_id, r.cfg_id): r for r in results}

    def get(case_id: str, cfg_id: str):
        key = (case_id, cfg_id)
        assert key in lookup, f"Missing result for {case_id} {cfg_id}"
        return lookup[key]

    assert get("01_pullout_lettow", "default").status == "ok"
    assert get("02_sspot_frp", "default").status == "ok"

    case03_bond_off = get("03_tensile_stn12", "bond-slip-off")
    assert case03_bond_off.status == "ok"

    assert get("03_tensile_stn12", "numba-off").status == "fail"
    assert get("08_beam_3pb_vvbs3_cfrp", "bulk-elastic").status == "ok"
    failed = {
        (result.case_id, result.cfg_id)
        for result in results
        if result.status != "ok"
    }
    assert failed == {
        ("03_tensile_stn12", "numba-off"),
        ("08_beam_3pb_vvbs3_cfrp", "default"),
        ("08_beam_3pb_vvbs3_cfrp", "bond-slip-off"),
        ("08_beam_3pb_vvbs3_cfrp", "numba-on"),
        ("08_beam_3pb_vvbs3_cfrp", "numba-off"),
    }
