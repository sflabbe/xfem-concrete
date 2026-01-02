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

    # Mechanical energy (for checking balance)
    E_mech_n: float     # T_n + Psi_bulk_n
    E_mech_np1: float   # T_np1 + Psi_bulk_np1

    # Balance check (should be ~0)
    balance_inc: float  # ΔW_dir - (ΔE_mech + ΔD_damp + ΔD_alg)


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


def constraint_work_hht(
    alpha: float,
    dt: float,
    f_int_n: np.ndarray,
    f_int_np1: np.ndarray,
    C_n: sp.csr_matrix,
    C_np1: sp.csr_matrix,
    v_n: np.ndarray,
    v_np1: np.ndarray,
    a_n: np.ndarray,
    a_np1: np.ndarray,
    Mdiag: np.ndarray,
    dir_dofs: List[int]
) -> float:
    """
    Compute constraint work increment using α-weighted power.

    The constraint forces (reactions) are computed from the equilibrium
    at the α-evaluation point:
      g_alpha = (1+alpha)*g_{n+1} + (-alpha)*g_n
    where
      g_n = f_int_n + C_n v_n + M a_n
      g_{n+1} = f_int_{n+1} + C_{n+1} v_{n+1} + M a_{n+1}

    The constraint work is:
      ΔW_dir = dt * (lambda_alpha · v_dir_alpha)
    where
      lambda_alpha = g_alpha[dir_dofs]
      v_dir_alpha = (1+alpha)*v_{n+1}[dir_dofs] + (-alpha)*v_n[dir_dofs]

    Sign convention: positive when energy flows into the system
    (i.e., for monotone imposed displacement in same direction as reaction).

    Args:
        alpha: HHT-α parameter
        dt: time step size
        f_int_n, f_int_np1: internal force vectors at n and n+1
        C_n, C_np1: damping matrices at n and n+1
        v_n, v_np1: velocity vectors at n and n+1
        a_n, a_np1: acceleration vectors at n and n+1
        Mdiag: diagonal mass vector
        dir_dofs: list of Dirichlet-constrained DOFs

    Returns:
        Constraint work increment (scalar)
    """
    if len(dir_dofs) == 0:
        return 0.0

    # Weights for α-evaluation
    w1 = 1.0 + alpha
    w0 = -alpha

    # Equilibrium vectors
    g_n = f_int_n + (C_n @ v_n) + (Mdiag * a_n)
    g_np1 = f_int_np1 + (C_np1 @ v_np1) + (Mdiag * a_np1)

    # α-weighted equilibrium (reaction-like forces at constrained DOFs)
    g_alpha = w1 * g_np1 + w0 * g_n

    # α-weighted velocity at constrained DOFs
    v_alpha = w1 * v_np1 + w0 * v_n

    # Extract constrained DOFs
    dir_dofs_arr = np.array(dir_dofs, dtype=int)
    lambda_alpha = g_alpha[dir_dofs_arr]
    v_dir_alpha = v_alpha[dir_dofs_arr]

    # Power = force · velocity (integrated over time step)
    # Sign: for monotone imposed displacement, this should be positive
    ΔW_dir = dt * float(np.dot(lambda_alpha, v_dir_alpha))

    return ΔW_dir


def damping_dissipation_hht(
    alpha: float,
    dt: float,
    aM: float,
    aK: float,
    Mdiag: np.ndarray,
    K_bulk_n: sp.csr_matrix,
    K_bulk_np1: sp.csr_matrix,
    v_n: np.ndarray,
    v_np1: np.ndarray
) -> float:
    """
    Compute damping dissipation increment efficiently.

    Rayleigh damping: C = aM*M + aK*K

    Damping dissipation for α-weighted scheme:
      ΔD_damp = dt * (v_alpha^T C_alpha v_alpha)
    where
      v_alpha = (1+alpha)*v_{n+1} + (-alpha)*v_n
      C_alpha = (1+alpha)*C_{n+1} + (-alpha)*C_n  (approximately)

    For efficiency, we compute this in two parts:
    - Mass-proportional: aM * dt * sum(Mdiag * v_alpha^2)
    - Stiffness-proportional: aK * dt * (v_alpha^T K_alpha v_alpha)
      where K_alpha ≈ (1+alpha)*K_{n+1} + (-alpha)*K_n

    Args:
        alpha: HHT-α parameter
        dt: time step size
        aM: mass-proportional damping coefficient
        aK: stiffness-proportional damping coefficient
        Mdiag: diagonal mass vector
        K_bulk_n, K_bulk_np1: bulk stiffness matrices at n and n+1
        v_n, v_np1: velocity vectors at n and n+1

    Returns:
        Damping dissipation increment (scalar, should be >= 0)
    """
    if aM == 0.0 and aK == 0.0:
        return 0.0

    # Weights for α-evaluation
    w1 = 1.0 + alpha
    w0 = -alpha

    # α-weighted velocity
    v_alpha = w1 * v_np1 + w0 * v_n

    ΔD_damp = 0.0

    # Mass-proportional damping (cheap: diagonal)
    if aM != 0.0:
        ΔD_damp_mass = aM * dt * float(np.dot(Mdiag, v_alpha * v_alpha))
        ΔD_damp += ΔD_damp_mass

    # Stiffness-proportional damping (one sparse matvec + dot)
    if aK != 0.0:
        # Use α-weighted stiffness for accuracy
        K_alpha = w1 * K_bulk_np1 + w0 * K_bulk_n
        Kv = K_alpha @ v_alpha
        ΔD_damp_stiff = aK * dt * float(np.dot(v_alpha, Kv))
        ΔD_damp += ΔD_damp_stiff

    return ΔD_damp


def compute_step_energy(
    step: int,
    t_n: float,
    t_np1: float,
    alpha: float,
    aM: float,
    aK: float,
    Mdiag: np.ndarray,
    u_n: np.ndarray,
    v_n: np.ndarray,
    a_n: np.ndarray,
    f_int_n: np.ndarray,
    psi_bulk_n: float,
    K_bulk_n: sp.csr_matrix,
    C_n: sp.csr_matrix,
    u_np1: np.ndarray,
    v_np1: np.ndarray,
    a_np1: np.ndarray,
    f_int_np1: np.ndarray,
    psi_bulk_np1: float,
    K_bulk_np1: sp.csr_matrix,
    C_np1: sp.csr_matrix,
    dir_dofs: List[int],
    W_dir_cum_prev: float,
    D_damp_cum_prev: float,
    D_alg_cum_prev: float
) -> StepEnergy:
    """
    Compute complete energy ledger for step n→n+1.

    This is the main interface for energy tracking. It computes all energy
    quantities consistently with the HHT-α scheme.

    Args:
        step: step number
        t_n, t_np1: times at n and n+1
        alpha: HHT-α parameter
        aM, aK: Rayleigh damping coefficients
        Mdiag: diagonal mass vector
        u_n, v_n, a_n: displacement, velocity, acceleration at n
        f_int_n: internal force at n
        psi_bulk_n: bulk recoverable energy at n
        K_bulk_n: bulk stiffness at n
        C_n: damping matrix at n
        u_np1, v_np1, a_np1: displacement, velocity, acceleration at n+1
        f_int_np1: internal force at n+1
        psi_bulk_np1: bulk recoverable energy at n+1
        K_bulk_np1: bulk stiffness at n+1
        C_np1: damping matrix at n+1
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

    # Constraint work
    ΔW_dir = constraint_work_hht(
        alpha, dt,
        f_int_n, f_int_np1,
        C_n, C_np1,
        v_n, v_np1,
        a_n, a_np1,
        Mdiag, dir_dofs
    )

    # Damping dissipation
    ΔD_damp = damping_dissipation_hht(
        alpha, dt,
        aM, aK,
        Mdiag,
        K_bulk_n, K_bulk_np1,
        v_n, v_np1
    )

    # Algorithmic dissipation (remainder)
    ΔD_alg = ΔW_dir - (ΔE_mech + ΔD_damp)

    # Balance check (should be ~0 by construction)
    balance_inc = ΔW_dir - (ΔE_mech + ΔD_damp + ΔD_alg)

    # Cumulative quantities
    W_dir_cum = W_dir_cum_prev + ΔW_dir
    D_damp_cum = D_damp_cum_prev + ΔD_damp
    D_alg_cum = D_alg_cum_prev + ΔD_alg

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
