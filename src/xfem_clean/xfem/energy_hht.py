"""
Efficient energy tracking for HHT-α time integration scheme.

This module provides energy tracking consistent with the HHT-α/Newmark
time integration used in xfem_beam.py, with a focus on computational efficiency:
- No extra global assemblies
- Reuse quantities computed by Newton solver
- Vectorized operations
- Minimal memory allocations

The scheme uses:
  alpha ∈ [-1/3, 0]
  gamma = 0.5 - alpha
  beta = 0.25*(1 - alpha)^2

Residual:
  g_prev = f_int_n + C_n v_n + M a_n
  r = (1+alpha)*(f_int_{n+1} + C_{n+1} v_{n+1} + M a_{n+1}) - alpha*g_prev
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp


@dataclass
class StepEnergy:
    """Energy ledger for a single converged time step n→n+1."""

    step: int           # accepted step number
    t: float            # current pseudo-time
    dt: float           # time step size

    # Constraint work (energy input through Dirichlet BCs)
    W_dir_inc: float    # increment for this step
    W_dir_cum: float    # cumulative

    # Kinetic energy (lumped mass)
    T_n: float          # at time n
    T_np1: float        # at time n+1

    # Bulk recoverable energy (cheap elastic-like proxy)
    Psi_bulk_n: float   # at time n
    Psi_bulk_np1: float # at time n+1

    # Damping dissipation (Rayleigh)
    D_damp_inc: float   # increment for this step
    D_damp_cum: float   # cumulative

    # Algorithmic dissipation (computed as remainder)
    D_alg_inc: float    # increment for this step
    D_alg_cum: float    # cumulative

    # TASK 5: Physical dissipation components
    D_coh_inc: float = 0.0      # cohesive dissipation increment
    D_coh_cum: float = 0.0      # cohesive dissipation cumulative
    D_bond_inc: float = 0.0     # bond-slip dissipation increment
    D_bond_cum: float = 0.0     # bond-slip dissipation cumulative
    D_bulk_plastic_inc: float = 0.0  # bulk plastic dissipation increment
    D_bulk_plastic_cum: float = 0.0  # bulk plastic dissipation cumulative
    D_physical_inc: float = 0.0 # total physical dissipation increment
    D_physical_cum: float = 0.0 # total physical dissipation cumulative
    D_numerical_inc: float = 0.0  # numerical dissipation (D_alg - D_physical)
    D_numerical_cum: float = 0.0  # numerical dissipation cumulative

    # Mechanical energy (for checking balance)
    E_mech_n: float = 0.0     # T_n + Psi_bulk_n
    E_mech_np1: float = 0.0   # T_np1 + Psi_bulk_np1

    # Balance check (should be ~0)
    balance_inc: float = 0.0  # ΔW_dir - (ΔE_mech + ΔD_damp + ΔD_alg)


def kinetic_energy(Mdiag: np.ndarray, v: np.ndarray) -> float:
    """
    Compute kinetic energy for lumped mass system.

    T = 0.5 * sum(Mdiag * v^2)

    Args:
        Mdiag: diagonal mass vector (ndof,)
        v: velocity vector (ndof,)

    Returns:
        Kinetic energy (scalar)
    """
    return 0.5 * float(np.dot(Mdiag, v * v))


def constraint_work_trapezoidal(
    f_int_n: np.ndarray,
    f_int_np1: np.ndarray,
    f_d_n: np.ndarray,
    f_d_np1: np.ndarray,
    f_m_n: np.ndarray,
    f_m_np1: np.ndarray,
    u_n: np.ndarray,
    u_np1: np.ndarray,
    dir_dofs: List[int]
) -> float:
    """
    Compute constraint work increment using trapezoidal rule.

    This is the energy-consistent formula for α=0 (Newmark average acceleration).
    For α<0, the algorithmic dissipation will be captured as a remainder.

    Dynamic nodal force vectors:
      g_n = f_int_n + f_d_n + f_m_n
      g_np1 = f_int_np1 + f_d_np1 + f_m_np1

    Constraint forces (reactions):
      λ_n = g_n[dir_dofs]
      λ_np1 = g_np1[dir_dofs]

    Displacement increment:
      Δu_dir = u_np1[dir_dofs] - u_n[dir_dofs]

    Work increment (trapezoidal):
      ΔW_dir = 0.5 * (λ_n + λ_np1) · Δu_dir

    Sign convention: positive when energy flows into the system
    (i.e., for monotone imposed displacement in same direction as reaction).

    Args:
        f_int_n, f_int_np1: internal force vectors at n and n+1
        f_d_n, f_d_np1: damping force vectors (C@v) at n and n+1
        f_m_n, f_m_np1: inertia force vectors (M@a) at n and n+1
        u_n, u_np1: displacement vectors at n and n+1
        dir_dofs: list of Dirichlet-constrained DOFs

    Returns:
        Constraint work increment (scalar)
    """
    if len(dir_dofs) == 0:
        return 0.0

    # Dynamic nodal force vectors (equilibrium)
    g_n = f_int_n + f_d_n + f_m_n
    g_np1 = f_int_np1 + f_d_np1 + f_m_np1

    # Extract constrained DOFs
    dir_dofs_arr = np.array(dir_dofs, dtype=int)
    lambda_n = g_n[dir_dofs_arr]
    lambda_np1 = g_np1[dir_dofs_arr]

    # Displacement increment at constrained DOFs
    du_dir = u_np1[dir_dofs_arr] - u_n[dir_dofs_arr]

    # Trapezoidal integration: work = average force · displacement increment
    ΔW_dir = 0.5 * float(np.dot(lambda_n + lambda_np1, du_dir))

    return ΔW_dir


def damping_dissipation_trapezoidal(
    dt: float,
    v_n: np.ndarray,
    v_np1: np.ndarray,
    f_d_n: np.ndarray,
    f_d_np1: np.ndarray
) -> float:
    """
    Compute damping dissipation increment using trapezoidal rule.

    NO EXTRA MATVECS: reuses f_d = C@v already computed in Newton iterations.

    Damping power: P_damp = v^T f_d
    Trapezoidal integration:
      ΔD_damp = 0.5 * dt * (v_n · f_d_n + v_np1 · f_d_np1)

    This is always >= 0 for positive semi-definite C.

    Args:
        dt: time step size
        v_n, v_np1: velocity vectors at n and n+1
        f_d_n, f_d_np1: damping force vectors (C@v) at n and n+1

    Returns:
        Damping dissipation increment (scalar, should be >= 0)
    """
    # Trapezoidal integration of damping power
    power_n = float(np.dot(v_n, f_d_n))
    power_np1 = float(np.dot(v_np1, f_d_np1))
    ΔD_damp = 0.5 * dt * (power_n + power_np1)

    return ΔD_damp


def compute_step_energy(
    step: int,
    t_n: float,
    t_np1: float,
    Mdiag: np.ndarray,
    u_n: np.ndarray,
    v_n: np.ndarray,
    f_int_n: np.ndarray,
    f_d_n: np.ndarray,
    f_m_n: np.ndarray,
    psi_bulk_n: float,
    u_np1: np.ndarray,
    v_np1: np.ndarray,
    f_int_np1: np.ndarray,
    f_d_np1: np.ndarray,
    f_m_np1: np.ndarray,
    psi_bulk_np1: float,
    dir_dofs: List[int],
    W_dir_cum_prev: float,
    D_damp_cum_prev: float,
    D_alg_cum_prev: float,
    # TASK 5: Physical dissipation components
    D_coh_inc: float = 0.0,
    D_bond_inc: float = 0.0,
    D_bulk_plastic_inc: float = 0.0,
    D_coh_cum_prev: float = 0.0,
    D_bond_cum_prev: float = 0.0,
    D_bulk_plastic_cum_prev: float = 0.0,
    D_physical_cum_prev: float = 0.0,
    D_numerical_cum_prev: float = 0.0,
) -> StepEnergy:
    """
    Compute complete energy ledger for step n→n+1 using trapezoidal formulas.

    This is the main interface for energy tracking. It uses trapezoidal integration
    which is energy-consistent for α=0 (Newmark average acceleration) and captures
    algorithmic dissipation for α<0 as a remainder.

    Args:
        step: step number
        t_n, t_np1: times at n and n+1
        Mdiag: diagonal mass vector
        u_n, v_n: displacement, velocity at n
        f_int_n: internal force at n
        f_d_n: damping force (C@v) at n
        f_m_n: inertia force (M@a) at n
        psi_bulk_n: bulk recoverable energy at n
        u_np1, v_np1: displacement, velocity at n+1
        f_int_np1: internal force at n+1
        f_d_np1: damping force (C@v) at n+1
        f_m_np1: inertia force (M@a) at n+1
        psi_bulk_np1: bulk recoverable energy at n+1
        dir_dofs: Dirichlet-constrained DOFs
        W_dir_cum_prev: cumulative constraint work from previous steps
        D_damp_cum_prev: cumulative damping dissipation from previous steps
        D_alg_cum_prev: cumulative algorithmic dissipation from previous steps

    Returns:
        StepEnergy dataclass with all energy quantities
    """
    dt = t_np1 - t_n

    # Kinetic energy
    T_n = kinetic_energy(Mdiag, v_n)
    T_np1 = kinetic_energy(Mdiag, v_np1)

    # Mechanical energy
    E_mech_n = T_n + psi_bulk_n
    E_mech_np1 = T_np1 + psi_bulk_np1
    ΔE_mech = E_mech_np1 - E_mech_n

    # Constraint work (trapezoidal)
    ΔW_dir = constraint_work_trapezoidal(
        f_int_n, f_int_np1,
        f_d_n, f_d_np1,
        f_m_n, f_m_np1,
        u_n, u_np1,
        dir_dofs
    )

    # Damping dissipation (trapezoidal, no extra matvecs)
    ΔD_damp = damping_dissipation_trapezoidal(
        dt,
        v_n, v_np1,
        f_d_n, f_d_np1
    )

    # Algorithmic dissipation (remainder)
    ΔD_alg = ΔW_dir - (ΔE_mech + ΔD_damp)

    # TASK 5: Physical dissipation components
    ΔD_physical = D_coh_inc + D_bond_inc + D_bulk_plastic_inc
    ΔD_numerical = ΔD_alg - ΔD_physical

    # Balance check (should be ~0 by construction)
    balance_inc = ΔW_dir - (ΔE_mech + ΔD_damp + ΔD_alg)

    # Cumulative quantities
    W_dir_cum = W_dir_cum_prev + ΔW_dir
    D_damp_cum = D_damp_cum_prev + ΔD_damp
    D_alg_cum = D_alg_cum_prev + ΔD_alg

    # TASK 5: Cumulative physical dissipation
    D_coh_cum = D_coh_cum_prev + D_coh_inc
    D_bond_cum = D_bond_cum_prev + D_bond_inc
    D_bulk_plastic_cum = D_bulk_plastic_cum_prev + D_bulk_plastic_inc
    D_physical_cum = D_physical_cum_prev + ΔD_physical
    D_numerical_cum = D_numerical_cum_prev + ΔD_numerical

    return StepEnergy(
        step=step,
        t=t_np1,
        dt=dt,
        W_dir_inc=ΔW_dir,
        W_dir_cum=W_dir_cum,
        T_n=T_n,
        T_np1=T_np1,
        Psi_bulk_n=psi_bulk_n,
        Psi_bulk_np1=psi_bulk_np1,
        D_damp_inc=ΔD_damp,
        D_damp_cum=D_damp_cum,
        D_alg_inc=ΔD_alg,
        D_alg_cum=D_alg_cum,
        # TASK 5: Physical dissipation components
        D_coh_inc=D_coh_inc,
        D_coh_cum=D_coh_cum,
        D_bond_inc=D_bond_inc,
        D_bond_cum=D_bond_cum,
        D_bulk_plastic_inc=D_bulk_plastic_inc,
        D_bulk_plastic_cum=D_bulk_plastic_cum,
        D_physical_inc=ΔD_physical,
        D_physical_cum=D_physical_cum,
        D_numerical_inc=ΔD_numerical,
        D_numerical_cum=D_numerical_cum,
        E_mech_n=E_mech_n,
        E_mech_np1=E_mech_np1,
        balance_inc=balance_inc
    )


def energy_to_dict(energy: StepEnergy) -> Dict[str, float]:
    """Convert StepEnergy to dictionary for CSV export."""
    return {
        'step': energy.step,
        't': energy.t,
        'dt': energy.dt,
        'W_dir_inc': energy.W_dir_inc,
        'W_dir_cum': energy.W_dir_cum,
        'T_n': energy.T_n,
        'T_np1': energy.T_np1,
        'Psi_bulk_n': energy.Psi_bulk_n,
        'Psi_bulk_np1': energy.Psi_bulk_np1,
        'D_damp_inc': energy.D_damp_inc,
        'D_damp_cum': energy.D_damp_cum,
        'D_alg_inc': energy.D_alg_inc,
        'D_alg_cum': energy.D_alg_cum,
        # TASK 5: Physical dissipation components
        'D_coh_inc': energy.D_coh_inc,
        'D_coh_cum': energy.D_coh_cum,
        'D_bond_inc': energy.D_bond_inc,
        'D_bond_cum': energy.D_bond_cum,
        'D_bulk_plastic_inc': energy.D_bulk_plastic_inc,
        'D_bulk_plastic_cum': energy.D_bulk_plastic_cum,
        'D_physical_inc': energy.D_physical_inc,
        'D_physical_cum': energy.D_physical_cum,
        'D_numerical_inc': energy.D_numerical_inc,
        'D_numerical_cum': energy.D_numerical_cum,
        'E_mech_n': energy.E_mech_n,
        'E_mech_np1': energy.E_mech_np1,
        'balance_inc': energy.balance_inc
    }


def write_energy_csv(energies: List[StepEnergy], filename: str):
    """Write energy history to CSV file."""
    import csv

    if len(energies) == 0:
        return

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=energy_to_dict(energies[0]).keys())
        writer.writeheader()
        for energy in energies:
            writer.writerow(energy_to_dict(energy))
