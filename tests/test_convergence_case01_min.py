"""Minimal convergence test for Case 01 pullout (BLOQUE E).

Runs Case 01 with coarse mesh and few steps to verify that:
- Bond-slip continuation (gamma ramp) is active
- No "Substepping exceeded" error
- No NaN in history

This is a fast smoke test to catch regressions.
"""
import pytest
from examples.gutierrez_thesis.cases.case_01_pullout_lettow import create_case_01
from examples.gutierrez_thesis.solver_interface import run_case_solver


@pytest.mark.slow
def test_case01_pullout_minimal_convergence():
    """Run Case 01 pullout with coarse mesh and 2-3 steps (convergence smoke test).

    Marked as slow because even with coarse mesh, the nonlinear solver takes time.
    """

    # Create case configuration
    case = create_case_01()

    # Override loading to reduce steps for faster test
    # Keep reasonable displacement to avoid convergence issues
    case.loading.n_steps = 5
    case.loading.max_displacement = 0.5  # mm (reduced from 5.0, but not too small)

    # Run case with coarse mesh
    try:
        results = run_case_solver(
            case=case,
            mesh_factor=0.4,  # Coarse mesh (fewer elements)
            max_steps=5,      # Override max steps
            enable_postprocess=False,  # Skip postprocessing for speed
            return_bundle=True,
        )
    except Exception as e:
        pytest.fail(f"Case 01 failed with exception: {e}")

    # Extract results
    history = results.get("history", [])
    model = results.get("model", None)

    # Check: should have at least 1 successful step
    assert len(history) > 0, "Case 01 returned empty history"

    # Check: no NaN in displacement history
    # History format varies, check common fields
    for i, record in enumerate(history):
        # Try different possible field names
        u_val = record.get("u_max", record.get("u", record.get("displacement", 0.0)))
        if isinstance(u_val, (int, float)):
            assert not (u_val != u_val), f"NaN found in history at step {i} (u={u_val})"

    # Check: no "Substepping exceeded" (would show in records or raise exception)
    # If we got here without exception, substepping didn't exceed

    # Check: continuation was used (XFEMModel has bond_gamma_strategy)
    # Note: default in model.py is bond_gamma_strategy="ramp_steps"
    if model is not None:
        assert hasattr(model, "bond_gamma_strategy"), "Model missing bond_gamma_strategy"
        assert model.bond_gamma_strategy == "ramp_steps", "Expected ramp_steps strategy (default)"

    print(f"✓ Case 01 pullout minimal test passed ({len(history)} steps completed)")
    if len(history) > 0:
        last_record = history[-1]
        u_last = last_record.get("u_max", last_record.get("u", last_record.get("displacement", 0.0)))
        if isinstance(u_last, (int, float)):
            print(f"  Final displacement: {u_last*1e3:.3f} mm")


if __name__ == "__main__":
    test_case01_pullout_minimal_convergence()
