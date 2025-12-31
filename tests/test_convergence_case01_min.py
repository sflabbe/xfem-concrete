"""Minimal convergence test for Case 01 pullout (BLOQUE E).

Runs Case 01 with coarse mesh and few steps to verify that:
- Bond-slip continuation (gamma ramp) is active
- No "Substepping exceeded" error
- No NaN in history

This is a fast smoke test to catch regressions.
"""
import sys
from pathlib import Path
import pytest

# Allow importing examples module
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from gutierrez_thesis.case_01_pullout import run_case_01


def test_case01_pullout_minimal_convergence():
    """Run Case 01 pullout with coarse mesh and 2-3 steps (fast smoke test)."""

    # Override parameters for speed
    overrides = {
        "mesh_density": "coarse",  # Coarse mesh (faster)
        "max_steps": 3,            # Only 3 steps (just check initial convergence)
        "u_total_mm": 0.03,        # Reduce total displacement (3 steps × 0.01 mm)
        "enable_bond_slip": True,
        # Enable continuation
        "bond_gamma_strategy": "ramp_steps",
        "bond_gamma_ramp_steps": 5,
        "bond_gamma_min": 0.0,
        "bond_gamma_max": 1.0,
        # Debug flags
        "debug_substeps": False,
        "debug_bond_gamma": False,
    }

    # Run case
    try:
        history, model, _ = run_case_01(**overrides)
    except Exception as e:
        pytest.fail(f"Case 01 failed with exception: {e}")

    # Check: should have at least 1 successful step
    assert len(history) > 0, "Case 01 returned empty history"

    # Check: no NaN in displacement history
    for i, record in enumerate(history):
        u_val = record.get("u", 0.0)
        assert not (u_val != u_val), f"NaN found in history at step {i} (u={u_val})"

    # Check: no "Substepping exceeded" (would show in records or raise exception)
    # If we got here without exception, substepping didn't exceed

    # Check: continuation was used (should see gamma ramp in debug output if enabled)
    # For now, just verify that model has the continuation parameters
    assert hasattr(model, "bond_gamma_strategy"), "Model missing bond_gamma_strategy"
    assert model.bond_gamma_strategy == "ramp_steps", "Expected ramp_steps strategy"

    print(f"✓ Case 01 pullout minimal test passed ({len(history)} steps completed)")
    print(f"  Final displacement: {history[-1].get('u', 0.0)*1e3:.3f} mm")


if __name__ == "__main__":
    test_case01_pullout_minimal_convergence()
