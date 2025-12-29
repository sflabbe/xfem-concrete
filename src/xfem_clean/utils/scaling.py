"""Diagonal scaling/equilibration utilities for ill-conditioned systems.

Reference: todo.md Section D (Diagonal scaling / equilibration)

The diagonal scaling improves conditioning by transforming:
    K x = r   →   K̃ x̃ = r̃

where:
    K̃ = D^(-1/2) K D^(-1/2)
    r̃ = D^(-1/2) r
    D = diag(|K|) + ε

After solving K̃ x̃ = r̃, recover x via:
    x = D^(-1/2) x̃
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def diagonal_equilibration(
    K: sp.spmatrix,
    r: np.ndarray,
    eps: float = 1e-12,
) -> tuple[sp.spmatrix, np.ndarray, np.ndarray]:
    """Apply diagonal equilibration to a linear system K x = r.

    Parameters
    ----------
    K : sparse matrix
        System matrix (typically stiffness matrix)
    r : ndarray
        Right-hand side vector (typically residual)
    eps : float, optional
        Regularization to avoid division by zero (default 1e-12)

    Returns
    -------
    K_scaled : sparse matrix
        Scaled system matrix K̃ = D^(-1/2) K D^(-1/2)
    r_scaled : ndarray
        Scaled RHS r̃ = D^(-1/2) r
    D_sqrt_inv : ndarray
        Diagonal scaling factors D^(-1/2) for recovering solution

    Notes
    -----
    After solving K̃ x̃ = r̃ for x̃, recover x via:
        x = D^(-1/2) * x̃  (element-wise product)

    Example
    -------
    >>> K_scaled, r_scaled, D_inv = diagonal_equilibration(K, r)
    >>> x_tilde = spla.spsolve(K_scaled, r_scaled)
    >>> x = D_inv * x_tilde
    """

    # Extract diagonal
    if sp.issparse(K):
        diag = K.diagonal()
    else:
        diag = np.diag(K)

    # Compute scaling: D = diag(|K|) + ε
    D = np.abs(diag) + eps

    # Compute D^(-1/2)
    D_sqrt_inv = 1.0 / np.sqrt(D)

    # Scale the system: K̃ = D^(-1/2) K D^(-1/2)
    if sp.issparse(K):
        # For sparse matrices, use diagonal matrix multiplication
        D_mat = sp.diags(D_sqrt_inv, format=K.format)
        K_scaled = D_mat @ K @ D_mat
    else:
        # For dense matrices
        K_scaled = D_sqrt_inv[:, None] * K * D_sqrt_inv[None, :]

    # Scale RHS: r̃ = D^(-1/2) r
    r_scaled = D_sqrt_inv * r

    return K_scaled, r_scaled, D_sqrt_inv


def unscale_solution(x_scaled: np.ndarray, D_sqrt_inv: np.ndarray) -> np.ndarray:
    """Recover unscaled solution from equilibrated system.

    Parameters
    ----------
    x_scaled : ndarray
        Solution x̃ from scaled system K̃ x̃ = r̃
    D_sqrt_inv : ndarray
        Diagonal scaling factors from diagonal_equilibration()

    Returns
    -------
    x : ndarray
        Unscaled solution x = D^(-1/2) x̃
    """
    return D_sqrt_inv * x_scaled


def check_conditioning_improvement(
    K_original: sp.spmatrix,
    K_scaled: sp.spmatrix,
    sample_size: int = 1000,
) -> dict:
    """Check conditioning improvement from diagonal scaling (diagnostic).

    Parameters
    ----------
    K_original : sparse matrix
        Original system matrix
    K_scaled : sparse matrix
        Scaled system matrix
    sample_size : int, optional
        Number of random vectors for condition number estimate

    Returns
    -------
    info : dict
        Diagnostic information with keys:
        - 'diag_range_original': (min, max) of |diag(K_original)|
        - 'diag_range_scaled': (min, max) of |diag(K_scaled)|
        - 'diag_ratio_original': max/min ratio before scaling
        - 'diag_ratio_scaled': max/min ratio after scaling

    Notes
    -----
    For large systems, computing exact condition number is expensive.
    This function reports diagonal ranges as a proxy.
    """

    diag_orig = np.abs(K_original.diagonal())
    diag_scaled = np.abs(K_scaled.diagonal())

    info = {
        'diag_range_original': (np.min(diag_orig), np.max(diag_orig)),
        'diag_range_scaled': (np.min(diag_scaled), np.max(diag_scaled)),
        'diag_ratio_original': np.max(diag_orig) / np.max([np.min(diag_orig), 1e-16]),
        'diag_ratio_scaled': np.max(diag_scaled) / np.max([np.min(diag_scaled), 1e-16]),
    }

    return info
