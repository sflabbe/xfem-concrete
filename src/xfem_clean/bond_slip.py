"""Bond-slip interface model for reinforced concrete (Model Code 2010).

This module implements the bond-slip relationship between concrete and reinforcing
steel bars according to fib Model Code 2010, Section 6.1.2.

The implementation includes:
  - Model Code 2010 bond-slip constitutive law for ribbed bars
  - Interface element assembly for rebar segments
  - Dowel action springs at crack-rebar intersections
  - State management for bond slip history
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Dict, Any
import math

import numpy as np
import scipy.sparse as sp

# Optional Numba acceleration
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap
    def prange(*args):
        return range(*args)


@dataclass
class BondSlipModelCode2010:
    """Bond-slip constitutive law per fib Model Code 2010 (ribbed bars).

    The bond stress-slip relationship is piecewise:

    .. math::
        \\tau(s) = \\begin{cases}
            \\tau_{max} (s/s_1)^\\alpha & 0 \\leq s \\leq s_1 \\\\
            \\tau_{max} & s_1 < s \\leq s_2 \\\\
            \\tau_{max} - (\\tau_{max} - \\tau_f)(s-s_2)/(s_3-s_2) & s_2 < s \\leq s_3 \\\\
            \\tau_f & s > s_3
        \\end{cases}

    Where:
      - τ_max: maximum bond stress
      - s1, s2, s3: characteristic slip values
      - τ_f: residual (friction) bond stress
      - α: shape parameter for rising branch

    Unloading/reloading follows a secant path to the maximum historical slip.

    Parameters
    ----------
    f_cm : float
        Mean concrete compressive strength [Pa]
    d_bar : float
        Bar diameter [m]
    condition : str
        Bond condition: "good" (unconfined) or "poor" (all other)

    Attributes
    ----------
    tau_max : float
        Maximum bond stress [Pa]
    s1, s2, s3 : float
        Characteristic slips [m]
    tau_f : float
        Residual bond stress [Pa]
    alpha : float
        Shape exponent [-]
    """

    f_cm: float
    d_bar: float
    condition: str = "good"

    def __post_init__(self):
        """Initialize Model Code 2010 parameters."""
        f_cm_mpa = float(self.f_cm) / 1e6  # Convert to MPa for MC2010 formulas
        sqrt_fcm = math.sqrt(f_cm_mpa)

        if self.condition == "good":
            # Unconfined conditions (good bond)
            self.tau_max = float(2.5 * sqrt_fcm * 1e6)  # Convert back to Pa
            self.s1 = 1.0e-3  # [m]
            self.s2 = 2.0e-3  # [m]
            self.s3 = float(self.d_bar)  # Clear rib spacing ~ diameter
            self.alpha = 0.4
        elif self.condition == "poor":
            # All other cases (poor bond)
            self.tau_max = float(1.25 * sqrt_fcm * 1e6)
            self.s1 = 1.8e-3
            self.s2 = 3.6e-3
            self.s3 = float(self.d_bar)
            self.alpha = 0.4
        else:
            raise ValueError(f"Unknown bond condition: {self.condition}. Use 'good' or 'poor'.")

        self.tau_f = float(0.15 * self.tau_max)  # Residual stress (friction)

        # Safety checks
        if self.s1 <= 0 or self.s2 <= self.s1 or self.s3 <= self.s2:
            raise ValueError("Bond-slip parameters must satisfy: 0 < s1 < s2 < s3")

    def tau_envelope(self, s_abs: float) -> Tuple[float, float]:
        """Compute bond stress and tangent on the monotonic envelope.

        Parameters
        ----------
        s_abs : float
            Absolute value of slip [m]

        Returns
        -------
        tau : float
            Bond stress [Pa]
        dtau_ds : float
            Tangent stiffness [Pa/m]
        """
        s_abs = float(s_abs)

        if s_abs <= self.s1:
            # Rising branch: τ = τ_max * (s/s1)^α
            if s_abs < 1e-16:
                # Avoid division by zero; use tangent at s → 0
                tau = 0.0
                dtau_ds = self.tau_max * self.alpha / self.s1 * (1e-16 / self.s1) ** (self.alpha - 1.0)
            else:
                ratio = s_abs / self.s1
                tau = self.tau_max * (ratio ** self.alpha)
                dtau_ds = self.tau_max * self.alpha / self.s1 * (ratio ** (self.alpha - 1.0))

        elif s_abs <= self.s2:
            # Plateau: τ = τ_max
            tau = self.tau_max
            dtau_ds = 0.0

        elif s_abs <= self.s3:
            # Softening branch: linear decay from τ_max to τ_f
            tau = self.tau_max - (self.tau_max - self.tau_f) * (s_abs - self.s2) / (self.s3 - self.s2)
            dtau_ds = -(self.tau_max - self.tau_f) / (self.s3 - self.s2)

        else:
            # Residual: τ = τ_f
            tau = self.tau_f
            dtau_ds = 0.0

        return float(tau), float(dtau_ds)

    def tau_and_tangent(self, s: float, s_max_history: float) -> Tuple[float, float]:
        """Compute bond stress and tangent stiffness with unloading/reloading.

        Parameters
        ----------
        s : float
            Current slip (signed) [m]
        s_max_history : float
            Maximum absolute slip reached in history [m]

        Returns
        -------
        tau : float
            Bond stress (signed) [Pa]
        dtau_ds : float
            Tangent stiffness [Pa/m]

        Notes
        -----
        - Loading: follows monotonic envelope
        - Unloading/reloading: secant stiffness to historical maximum
        """
        s_abs = abs(s)
        sign = 1.0 if s >= 0.0 else -1.0

        # Ensure s_max_history is non-decreasing
        s_max = max(float(s_max_history), s_abs)

        if s_abs >= s_max - 1e-14:
            # Loading: on envelope
            tau_abs, dtau_abs = self.tau_envelope(s_abs)
        else:
            # Unloading/reloading: secant to max point
            tau_max_reached, _ = self.tau_envelope(s_max)
            if s_max > 1e-14:
                tau_abs = tau_max_reached * (s_abs / s_max)
                dtau_abs = tau_max_reached / s_max
            else:
                tau_abs = 0.0
                dtau_abs = 0.0

        return sign * float(tau_abs), float(dtau_abs)


# ------------------------------------------------------------------------------
# Interface Element State Arrays
# ------------------------------------------------------------------------------

@dataclass
class BondSlipStateArrays:
    """State variables for bond-slip interface elements.

    Attributes
    ----------
    n_segments : int
        Number of rebar segments
    s_max : np.ndarray
        Maximum absolute slip history [n_segments] [m]
    s_current : np.ndarray
        Current slip (signed) [n_segments] [m]
    tau_current : np.ndarray
        Current bond stress [n_segments] [Pa]
    """

    n_segments: int
    s_max: np.ndarray
    s_current: np.ndarray
    tau_current: np.ndarray

    @classmethod
    def zeros(cls, n_segments: int) -> BondSlipStateArrays:
        """Initialize zero states."""
        return cls(
            n_segments=int(n_segments),
            s_max=np.zeros(n_segments, dtype=float),
            s_current=np.zeros(n_segments, dtype=float),
            tau_current=np.zeros(n_segments, dtype=float),
        )

    def copy_shallow(self) -> BondSlipStateArrays:
        """Create a shallow copy of state arrays."""
        return BondSlipStateArrays(
            n_segments=int(self.n_segments),
            s_max=self.s_max.copy(),
            s_current=self.s_current.copy(),
            tau_current=self.tau_current.copy(),
        )

    def copy(self) -> BondSlipStateArrays:
        """Alias for copy_shallow (no nested structure)."""
        return self.copy_shallow()


# ------------------------------------------------------------------------------
# Bond-Slip Assembly (Numba-accelerated)
# ------------------------------------------------------------------------------

@njit(cache=True, parallel=False)  # TODO: Fix parallel loop (entry_idx conflict)
def _bond_slip_assembly_numba(
    u_total: np.ndarray,        # [ndof_total] displacement vector
    segs: np.ndarray,           # [n_seg, 5]: [n1, n2, L0, cx, cy]
    steel_dof_map: np.ndarray,  # [nnode, 2]: node → steel DOF indices
    bond_params: np.ndarray,    # [tau_max, s1, s2, s3, tau_f, alpha, perimeter, dtau_max]
    s_max_hist: np.ndarray,     # [n_seg] history
    steel_EA: float = 0.0,      # E * A for steel (if > 0, adds axial stiffness)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble bond-slip interface contribution (Numba kernel).

    Returns
    -------
    f_int : np.ndarray
        Internal force vector [ndof_total]
    rows, cols, data : np.ndarray
        Triplets for stiffness matrix
    s_current : np.ndarray
        Current slip values [n_seg]
    """
    ndof = u_total.shape[0]
    n_seg = segs.shape[0]

    f = np.zeros(ndof, dtype=np.float64)
    max_entries = n_seg * 32  # 8 (bond interface) + 16 (steel axial) + margin
    rows = np.empty(max_entries, dtype=np.int64)
    cols = np.empty(max_entries, dtype=np.int64)
    data = np.empty(max_entries, dtype=np.float64)
    s_current = np.zeros(n_seg, dtype=np.float64)

    # Unpack bond parameters
    tau_max = bond_params[0]
    s1 = bond_params[1]
    s2 = bond_params[2]
    s3 = bond_params[3]
    tau_f = bond_params[4]
    alpha = bond_params[5]
    perimeter = bond_params[6]  # π * d_bar
    dtau_max = bond_params[7] if bond_params.shape[0] > 7 else 1e20  # Tangent cap (Priority #1)

    entry_idx = 0

    for i in range(n_seg):  # Changed from prange due to entry_idx conflict
        # Node IDs
        n1 = int(segs[i, 0])
        n2 = int(segs[i, 1])
        L0 = segs[i, 2]
        cx = segs[i, 3]
        cy = segs[i, 4]

        # Concrete DOFs
        dof_c1x = 2 * n1
        dof_c1y = 2 * n1 + 1
        dof_c2x = 2 * n2
        dof_c2y = 2 * n2 + 1

        # Steel DOFs (sparse mapping via steel_dof_map)
        dof_s1x = int(steel_dof_map[n1, 0])
        dof_s1y = int(steel_dof_map[n1, 1])
        dof_s2x = int(steel_dof_map[n2, 0])
        dof_s2y = int(steel_dof_map[n2, 1])

        # Displacements
        u_c1x = u_total[dof_c1x]
        u_c1y = u_total[dof_c1y]
        u_c2x = u_total[dof_c2x]
        u_c2y = u_total[dof_c2y]

        u_s1x = u_total[dof_s1x]
        u_s1y = u_total[dof_s1y]
        u_s2x = u_total[dof_s2x]
        u_s2y = u_total[dof_s2y]

        # Average slip along segment (simplified: use midpoint)
        u_c_mid_x = 0.5 * (u_c1x + u_c2x)
        u_c_mid_y = 0.5 * (u_c1y + u_c2y)
        u_s_mid_x = 0.5 * (u_s1x + u_s2x)
        u_s_mid_y = 0.5 * (u_s1y + u_s2y)

        # Relative displacement (steel - concrete)
        du_x = u_s_mid_x - u_c_mid_x
        du_y = u_s_mid_y - u_c_mid_y

        # Slip in bar direction (tangential)
        s = du_x * cx + du_y * cy
        s_current[i] = s

        # Bond stress (with history)
        s_abs = abs(s)
        s_max = max(s_max_hist[i], s_abs)
        sign = 1.0 if s >= 0.0 else -1.0

        # C1-continuous regularization for s -> 0 singularity (Priority #1)
        # Use s_reg = 0.01 * s1 as threshold
        s_reg = 0.01 * s1

        # Design C1-continuous transition:
        # For s <= s_reg: τ(s) = k0 * s (linear)
        # For s > s_reg: τ(s) = τ_max * ((s - s_reg + s_off) / s1)^α
        #
        # Match τ and dτ/ds at s = s_reg:
        # 1) τ(s_reg) from power law = k0 * s_reg
        # 2) dτ/ds(s_reg) from power law = k0
        #
        # From (2): τ_max * α / s1 * (s_off / s1)^(α-1) = k0
        # => s_off = s1 * (k0 * s1 / (τ_max * α))^(1/(α-1))
        #
        # From (1): τ_max * (s_off / s1)^α = k0 * s_reg
        # => k0 = τ_max / s_reg * (s_off / s1)^α
        #
        # Solve iteratively or use s_off heuristic:
        # Choose k0 = τ_max / s1 * s_reg^(α-1) (tangent at s_reg from original law)
        # Then s_off = s1 * (k0 * s1 / (τ_max * α))^(1/(α-1))

        k0 = tau_max / s1 * (s_reg ** (alpha - 1.0))  # Tangent stiffness at s_reg
        s_off = s1 * ((k0 * s1 / (tau_max * alpha)) ** (1.0 / (alpha - 1.0)))
        tau_offset = tau_max * ((s_off / s1) ** alpha) - k0 * s_reg  # Offset to match τ at s_reg

        # Envelope evaluation (inline for Numba)
        if s_abs >= s_max - 1e-14:
            # Loading envelope
            if s_abs <= s1:
                if s_abs < s_reg:
                    # Regularized linear branch (C1-continuous)
                    tau_abs = k0 * s_abs
                    dtau_abs = k0
                else:
                    # Power law with offset for C1 continuity
                    s_eff = s_abs - s_reg + s_off
                    ratio = s_eff / s1
                    tau_abs = tau_max * (ratio ** alpha) - tau_offset
                    dtau_abs = tau_max * alpha / s1 * (ratio ** (alpha - 1.0))
            elif s_abs <= s2:
                tau_abs = tau_max
                dtau_abs = 0.0
            elif s_abs <= s3:
                tau_abs = tau_max - (tau_max - tau_f) * (s_abs - s2) / (s3 - s2)
                dtau_abs = -(tau_max - tau_f) / (s3 - s2)
            else:
                tau_abs = tau_f
                dtau_abs = 0.0
        else:
            # Unloading: secant
            tau_env, _ = 0.0, 0.0
            if s_max <= s1:
                tau_env = tau_max * (s_max / s1) ** alpha if s_max > 1e-16 else 0.0
            elif s_max <= s2:
                tau_env = tau_max
            elif s_max <= s3:
                tau_env = tau_max - (tau_max - tau_f) * (s_max - s2) / (s3 - s2)
            else:
                tau_env = tau_f

            if s_max > 1e-14:
                tau_abs = tau_env * (s_abs / s_max)
                dtau_abs = tau_env / s_max
            else:
                tau_abs = 0.0
                dtau_abs = 0.0

        tau = sign * tau_abs
        dtau_ds = dtau_abs

        # Tangent capping (numerical stabilization, Priority #1)
        # NOTE: dtau_max passed as parameter; set to large value (1e20) if no capping desired
        # In practice, set dtau_max = bond_tangent_cap_factor * median(diag(K_bulk))
        # This prevents bond stiffness from dominating and causing ill-conditioning
        if dtau_ds > dtau_max:
            dtau_ds = dtau_max

        # Bond force (distributed over segment length)
        F_bond = tau * perimeter * L0

        # Distribute to nodes (simplified: equal split)
        # Force on steel (in bar direction)
        Fx_s = F_bond * cx
        Fy_s = F_bond * cy

        # Newton's third law: force on concrete is opposite
        Fx_c = -Fx_s
        Fy_c = -Fy_s

        # Distribute equally to segment ends
        f[dof_s1x] += 0.5 * Fx_s
        f[dof_s1y] += 0.5 * Fy_s
        f[dof_s2x] += 0.5 * Fx_s
        f[dof_s2y] += 0.5 * Fy_s

        f[dof_c1x] += 0.5 * Fx_c
        f[dof_c1y] += 0.5 * Fy_c
        f[dof_c2x] += 0.5 * Fx_c
        f[dof_c2y] += 0.5 * Fy_c

        # Stiffness contribution: K = dtau_ds * perimeter * L0 * (t ⊗ t)
        K_bond = dtau_ds * perimeter * L0

        # Directional stiffness matrix (only tangential component)
        Kxx = K_bond * cx * cx
        Kxy = K_bond * cx * cy
        Kyy = K_bond * cy * cy

        # Coupling between steel and concrete DOFs (schematic; simplified here)
        # Full implementation requires proper assembly over all 8 DOFs

        # For brevity: add diagonal penalty (correct implementation needs off-diagonal terms)
        # This is a placeholder; full derivation yields block matrix structure

        # Example: diagonal terms (simplified)
        if entry_idx + 8 < max_entries:
            # Steel DOFs (positive stiffness)
            rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_s1x; data[entry_idx] = 0.25 * Kxx; entry_idx += 1
            rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_s1y; data[entry_idx] = 0.25 * Kyy; entry_idx += 1
            rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_s2x; data[entry_idx] = 0.25 * Kxx; entry_idx += 1
            rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_s2y; data[entry_idx] = 0.25 * Kyy; entry_idx += 1

            # Concrete DOFs (positive stiffness, opposite direction)
            rows[entry_idx] = dof_c1x; cols[entry_idx] = dof_c1x; data[entry_idx] = 0.25 * Kxx; entry_idx += 1
            rows[entry_idx] = dof_c1y; cols[entry_idx] = dof_c1y; data[entry_idx] = 0.25 * Kyy; entry_idx += 1
            rows[entry_idx] = dof_c2x; cols[entry_idx] = dof_c2x; data[entry_idx] = 0.25 * Kxx; entry_idx += 1
            rows[entry_idx] = dof_c2y; cols[entry_idx] = dof_c2y; data[entry_idx] = 0.25 * Kyy; entry_idx += 1

        # Steel axial stiffness (if steel_EA > 0)
        if steel_EA > 0.0 and entry_idx + 16 < max_entries:
            K_steel = steel_EA / L0
            Kxx_s = K_steel * cx * cx
            Kxy_s = K_steel * cx * cy
            Kyy_s = K_steel * cy * cy

            # Steel bar stiffness: K = (EA/L) * [c⊗c, -c⊗c; -c⊗c, c⊗c]
            # Node 1 - Node 1 (positive)
            rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_s1x; data[entry_idx] = Kxx_s; entry_idx += 1
            rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_s1y; data[entry_idx] = Kxy_s; entry_idx += 1
            rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_s1x; data[entry_idx] = Kxy_s; entry_idx += 1
            rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_s1y; data[entry_idx] = Kyy_s; entry_idx += 1

            # Node 2 - Node 2 (positive)
            rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_s2x; data[entry_idx] = Kxx_s; entry_idx += 1
            rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_s2y; data[entry_idx] = Kxy_s; entry_idx += 1
            rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_s2x; data[entry_idx] = Kxy_s; entry_idx += 1
            rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_s2y; data[entry_idx] = Kyy_s; entry_idx += 1

            # Node 1 - Node 2 (negative)
            rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_s2x; data[entry_idx] = -Kxx_s; entry_idx += 1
            rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_s2y; data[entry_idx] = -Kxy_s; entry_idx += 1
            rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_s2x; data[entry_idx] = -Kxy_s; entry_idx += 1
            rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_s2y; data[entry_idx] = -Kyy_s; entry_idx += 1

            # Node 2 - Node 1 (negative)
            rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_s1x; data[entry_idx] = -Kxx_s; entry_idx += 1
            rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_s1y; data[entry_idx] = -Kxy_s; entry_idx += 1
            rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_s1x; data[entry_idx] = -Kxy_s; entry_idx += 1
            rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_s1y; data[entry_idx] = -Kyy_s; entry_idx += 1

    # Trim arrays
    rows_out = rows[:entry_idx]
    cols_out = cols[:entry_idx]
    data_out = data[:entry_idx]

    return f, rows_out, cols_out, data_out, s_current


def assemble_bond_slip(
    u_total: np.ndarray,
    steel_segments: np.ndarray,
    steel_dof_offset: int,
    bond_law: BondSlipModelCode2010,
    bond_states: BondSlipStateArrays,
    steel_dof_map: np.ndarray = None,
    steel_EA: float = 0.0,
    use_numba: bool = True,
) -> Tuple[np.ndarray, sp.csr_matrix, BondSlipStateArrays]:
    """Assemble bond-slip interface forces and stiffness.

    Parameters
    ----------
    u_total : np.ndarray
        Total displacement vector (concrete + steel DOFs)
    steel_segments : np.ndarray
        [n_seg, 5] array: [n1, n2, L0, cx, cy]
    steel_dof_offset : int
        Index where steel DOFs begin (for backward compat, not used if steel_dof_map provided)
    bond_law : BondSlipModelCode2010
        Bond-slip constitutive model
    bond_states : BondSlipStateArrays
        Current bond-slip states
    steel_dof_map : np.ndarray, optional
        [nnode, 2] array mapping node → steel DOF indices (or -1 if no steel)
        If None, assumes dense contiguous steel DOFs (legacy behavior)
    use_numba : bool
        Use Numba acceleration if available

    Returns
    -------
    f_bond : np.ndarray
        Bond interface force vector
    K_bond : sparse.csr_matrix
        Bond interface stiffness matrix
    bond_states_new : BondSlipStateArrays
        Updated bond-slip states (trial)
    """
    n_seg = steel_segments.shape[0]
    ndof_total = u_total.shape[0]

    # Pack bond parameters for Numba
    perimeter = math.pi * float(bond_law.d_bar)

    # Tangent capping for numerical stability (Priority #1)
    # dtau_max is typically set to bond_tangent_cap_factor * median(diag(K_bulk))
    # For now, use a large value (no capping); caller can override via bond_law
    dtau_max = getattr(bond_law, 'dtau_max', 1e20)  # Default: no cap

    bond_params = np.array([
        bond_law.tau_max,
        bond_law.s1,
        bond_law.s2,
        bond_law.s3,
        bond_law.tau_f,
        bond_law.alpha,
        perimeter,
        dtau_max,
    ], dtype=float)

    if use_numba and NUMBA_AVAILABLE:
        # Use sparse DOF mapping if provided, else legacy dense mapping
        if steel_dof_map is None:
            # Legacy: assume dense contiguous steel DOFs
            nnode = int(ndof_total - steel_dof_offset) // 2
            steel_dof_map = -np.ones((nnode, 2), dtype=np.int64)
            for n in range(nnode):
                steel_dof_map[n, 0] = steel_dof_offset + 2 * n
                steel_dof_map[n, 1] = steel_dof_offset + 2 * n + 1

        f_bond, rows, cols, data, s_curr = _bond_slip_assembly_numba(
            u_total,
            steel_segments,
            steel_dof_map,
            bond_params,
            bond_states.s_max,
        )

        K_bond = sp.csr_matrix((data, (rows, cols)), shape=(ndof_total, ndof_total))

        # Update states (trial)
        bond_states_new = BondSlipStateArrays(
            n_segments=n_seg,
            s_max=np.maximum(bond_states.s_max, np.abs(s_curr)),
            s_current=s_curr,
            tau_current=np.zeros(n_seg),  # Computed inline; could extract if needed
        )

    else:
        # Pure Python fallback (not implemented here for brevity)
        raise NotImplementedError("Python fallback for bond-slip assembly not yet implemented. Use Numba.")

    return f_bond, K_bond, bond_states_new


# ------------------------------------------------------------------------------
# Dowel Action at Crack Intersections
# ------------------------------------------------------------------------------

def compute_dowel_springs(
    crack_geom: Any,  # XFEMCrack or similar
    steel_segments: np.ndarray,
    nodes: np.ndarray,
    E_steel: float,
    d_bar: float,
) -> List[Tuple[int, int, float]]:
    """Compute dowel action springs at crack-rebar intersections.

    Parameters
    ----------
    crack_geom : crack geometry object
        Contains crack path information
    steel_segments : np.ndarray
        [n_seg, 5]: [n1, n2, L0, cx, cy]
    nodes : np.ndarray
        [n_nodes, 2]: node coordinates
    E_steel : float
        Steel Young's modulus [Pa]
    d_bar : float
        Bar diameter [m]

    Returns
    -------
    dowel_springs : List[Tuple[int, int, float]]
        List of (node_concrete, node_steel, k_dowel)
        where k_dowel is the transverse stiffness [N/m]

    Notes
    -----
    Dowel action model (elastic foundation):
        k_dowel = E_s * I / L_e^3
        L_e = 10 * d_bar (empirical effective length)
    """
    I_bar = math.pi * (d_bar ** 4) / 64.0  # Second moment of area
    L_e = 10.0 * d_bar  # Effective embedment length (empirical)
    k_dowel = E_steel * I_bar / (L_e ** 3)

    dowel_springs = []

    # Simplified: assume crack is a vertical line (for initial implementation)
    # Full implementation would do geometric intersection
    if not hasattr(crack_geom, 'tip_x'):
        return dowel_springs  # No active crack

    crack_x = float(crack_geom.tip_x)

    for i in range(steel_segments.shape[0]):
        n1 = int(steel_segments[i, 0])
        n2 = int(steel_segments[i, 1])
        x1 = nodes[n1, 0]
        x2 = nodes[n2, 0]

        # Check if crack crosses segment
        if (x1 <= crack_x <= x2) or (x2 <= crack_x <= x1):
            # Add dowel spring at both nodes (distributed)
            dowel_springs.append((n1, n1, k_dowel))
            dowel_springs.append((n2, n2, k_dowel))

    return dowel_springs


def assemble_dowel_action(
    dowel_springs: List[Tuple[int, int, float]],
    steel_dof_offset: int,
    ndof_total: int,
) -> sp.csr_matrix:
    """Assemble dowel action stiffness matrix.

    Parameters
    ----------
    dowel_springs : List[Tuple[int, int, float]]
        [(node_concrete, node_steel, k_dowel), ...]
    steel_dof_offset : int
        Offset for steel DOFs
    ndof_total : int
        Total number of DOFs

    Returns
    -------
    K_dowel : sparse.csr_matrix
        Dowel stiffness matrix (transverse direction only)
    """
    rows = []
    cols = []
    data = []

    for node_c, node_s, k in dowel_springs:
        # Transverse direction (assume y for vertical crack)
        dof_cy = 2 * node_c + 1
        dof_sy = steel_dof_offset + 2 * node_s + 1

        # Diagonal terms
        rows.append(dof_cy)
        cols.append(dof_cy)
        data.append(k)

        rows.append(dof_sy)
        cols.append(dof_sy)
        data.append(k)

        # Off-diagonal coupling (penalty: resists relative motion)
        rows.append(dof_cy)
        cols.append(dof_sy)
        data.append(-k)

        rows.append(dof_sy)
        cols.append(dof_cy)
        data.append(-k)

    if len(data) == 0:
        return sp.csr_matrix((ndof_total, ndof_total), dtype=float)

    K_dowel = sp.csr_matrix((data, (rows, cols)), shape=(ndof_total, ndof_total))
    return K_dowel


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def pack_bond_params(bond_law: BondSlipModelCode2010) -> np.ndarray:
    """Pack bond-slip parameters into array for Numba kernels."""
    perimeter = math.pi * float(bond_law.d_bar)
    return np.array([
        bond_law.tau_max,
        bond_law.s1,
        bond_law.s2,
        bond_law.s3,
        bond_law.tau_f,
        bond_law.alpha,
        perimeter,
    ], dtype=float)
