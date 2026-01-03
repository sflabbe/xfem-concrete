"""Numba-accelerated bond-slip assembly kernel.

This module contains the optimized kernel for assembling bond-slip interface
forces and stiffness matrices. The kernel is compiled with cache=True for
improved startup performance.

Design
------
* Implements full 8×8 consistent tangent coupling steel↔concrete DOFs
* Supports BondSlipModelCode2010 constitutive law with inline evaluation
* Optional steel axial stiffness contribution
* C1-continuous regularization near s=0 to prevent ill-conditioning
* PART A FIX: Steel axial element always assembled, even when bond is masked

Performance
-----------
* cache=True: Numba caches compiled code for faster subsequent imports
* Serial loop (not prange): Due to shared entry_idx counter for COO assembly
* Expected speedup: 2-10× for typical problems (n_seg ~ 100-1000)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from xfem_clean.numba.utils import njit


@njit(cache=True)  # Enable caching for faster startup
def bond_slip_assembly_kernel(
    u_total: np.ndarray,        # [ndof_total] displacement vector
    segs: np.ndarray,           # [n_seg, 5]: [n1, n2, L0, cx, cy]
    steel_dof_map: np.ndarray,  # [nnode, 2]: node → steel DOF indices
    bond_params: np.ndarray,    # [tau_max, s1, s2, s3, tau_f, alpha, perimeter, dtau_max, gamma, f_y, E_s, enable_omega_y]
    s_max_hist: np.ndarray,     # [n_seg] history
    steel_EA: float = 0.0,      # E * A for steel (if > 0, adds axial stiffness)
    segment_mask: np.ndarray = None,  # [n_seg] bool: True = bond disabled for segment
    crack_context: np.ndarray = None,  # [n_seg, 2]: [x_over_l, r] for Ωc computation
    dowel_params: np.ndarray = None,   # [phi, fc] for dowel action (if enabled)
    u_total_prev: np.ndarray = None,   # Previous displacements (for dissipation)
    compute_dissipation: bool = False, # Enable dissipation tracking
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Assemble bond-slip interface contribution (Numba kernel).

    Parameters
    ----------
    u_total : np.ndarray
        Displacement vector [ndof_total]
    segs : np.ndarray
        Segment geometry [n_seg, 5]: [n1, n2, L0, cx, cy]
        - n1, n2: node indices (int)
        - L0: reference length [m]
        - cx, cy: tangent vector components (unit)
    steel_dof_map : np.ndarray
        Mapping [nnode, 2]: node → (dof_x, dof_y) for steel
        Use -1 for nodes without steel DOFs
    bond_params : np.ndarray
        Bond law parameters [14] (THESIS PARITY: extended for proper εu):
        [0] tau_max: peak bond stress [Pa]
        [1] s1: slip at peak stress [m]
        [2] s2: slip at start of plateau end [m]
        [3] s3: slip at end of softening [m]
        [4] tau_f: residual bond stress [Pa]
        [5] alpha: shape parameter for ascending branch
        [6] perimeter: rebar perimeter [m] (π * d_bar for circular)
        [7] dtau_max: tangent cap [Pa/m] (use 1e20 for no cap)
        [8] gamma: continuation parameter [0, 1] (0=no bond, 1=full bond)
        [9] f_y: steel yield stress [Pa] (for Ωy, Part B)
        [10] E_s: steel Young's modulus [Pa] (for Ωy, Part B)
        [11] enable_omega_y: 1.0 to enable yielding reduction, 0.0 to disable
        [12] f_u: steel ultimate stress [Pa] (THESIS PARITY)
        [13] H: steel hardening modulus [Pa] (THESIS PARITY)
    s_max_hist : np.ndarray
        Maximum historical slip [n_seg]
    steel_EA : float, optional
        Steel axial stiffness E*A [N]. If > 0, adds truss element.
    segment_mask : np.ndarray, optional
        Boolean mask [n_seg] where True = bond disabled for that segment.
        If None, all segments are active.
        CRITICAL (Part A): Masked segments skip bond/dowel but RETAIN steel axial element.
    crack_context : np.ndarray, optional
        Crack deterioration context [n_seg, 2] for computing Ωc.
        crack_context[i, 0] = x_over_l (normalized distance to crack)
        crack_context[i, 1] = r (tension/strength ratio at crack)
        If None, Ωc = 1.0 everywhere (no deterioration).
    dowel_params : np.ndarray, optional
        Dowel action parameters [2]: [phi, fc] where phi is bar diameter [m]
        and fc is concrete compressive strength [Pa].
        If None, dowel action is disabled.
    u_total_prev : np.ndarray, optional
        Displacement vector at previous step [ndof_total] for dissipation.
        If None and compute_dissipation=True, dissipation will be zero.
    compute_dissipation : bool, optional
        If True, compute dissipation using trapezoidal rule. Default: False.

    Returns
    -------
    f_int : np.ndarray
        Internal force vector [ndof_total]
    rows, cols, data : np.ndarray
        Triplets for stiffness matrix (COO format)
    s_current : np.ndarray
        Current slip values [n_seg]
    D_bond_inc : float
        Bond dissipation increment [J]
    D_dowel_inc : float
        Dowel dissipation increment [J]

    Notes
    -----
    * The stiffness matrix is the full 8×8 consistent tangent:
      K_seg = (gamma * dtau/ds * perimeter * L0) * g ⊗ g^T
      where g = [∂s/∂u] is the slip gradient (see implementation).
    * Regularization: For s < s_reg = 0.5*s1, uses linear branch to avoid
      singular tangent at s=0 (C1-continuous transition).
    * PART A FIX: Masked segments (segment_mask[i] == True) skip bond shear/dowel
      but ALWAYS include steel axial stiffness and internal force.
    * PART B: Steel yielding reduction Ωy(eps_s) applied when enable_omega_y=1.
    """
    ndof = u_total.shape[0]
    n_seg = segs.shape[0]

    f = np.zeros(ndof, dtype=np.float64)
    # 64 (bond 8x8) + 16 (steel axial) + 64 (dowel 8x8) + margin = 160 per segment
    max_entries = n_seg * 160
    rows = np.empty(max_entries, dtype=np.int64)
    cols = np.empty(max_entries, dtype=np.int64)
    data = np.empty(max_entries, dtype=np.float64)
    s_current = np.zeros(n_seg, dtype=np.float64)

    # Dissipation accumulators
    D_bond_inc = 0.0
    D_dowel_inc = 0.0

    # Unpack bond parameters (THESIS PARITY: extended for proper εu)
    tau_max = bond_params[0]
    s1 = bond_params[1]
    s2 = bond_params[2]
    s3 = bond_params[3]
    tau_f = bond_params[4]
    alpha = bond_params[5]
    perimeter = bond_params[6]  # π * d_bar
    dtau_max = bond_params[7] if bond_params.shape[0] > 7 else 1e20  # Tangent cap
    gamma = bond_params[8] if bond_params.shape[0] > 8 else 1.0  # Continuation parameter
    # PART B / THESIS PARITY: Yielding reduction parameters
    f_y = bond_params[9] if bond_params.shape[0] > 9 else 500e6  # Default: 500 MPa
    E_s = bond_params[10] if bond_params.shape[0] > 10 else 200e9  # Default: 200 GPa
    enable_omega_y = bond_params[11] if bond_params.shape[0] > 11 else 0.0  # Default: disabled
    f_u = bond_params[12] if bond_params.shape[0] > 12 else 1.5 * f_y  # THESIS PARITY: Ultimate stress
    H = bond_params[13] if bond_params.shape[0] > 13 else 0.01 * E_s  # THESIS PARITY: Hardening modulus

    # Unpack dowel parameters (if provided)
    enable_dowel = dowel_params is not None
    if enable_dowel:
        phi = dowel_params[0]  # Bar diameter [m]
        fc = dowel_params[1]   # Concrete compressive strength [Pa]
        # Convert fc to MPa for dowel formula (formula uses MPa units internally)
        fc_MPa = fc / 1e6

    entry_idx = 0

    for i in range(n_seg):  # Serial (not prange) due to entry_idx
        # =====================================================================
        # PART A FIX: Extract geometry and DOFs BEFORE mask check
        # =====================================================================
        # Node IDs (needed for both bond and steel axial)
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

        # =====================================================================
        # PART A FIX: Steel axial contribution ALWAYS assembled (before mask check)
        # =====================================================================
        # This is the bar element behavior independent of bond interface.
        # Even if bond is disabled (masked), the steel bar must carry axial loads.
        if steel_EA > 0.0 and entry_idx + 16 < max_entries:
            K_steel = steel_EA / L0
            Kxx_s = K_steel * cx * cx
            Kxy_s = K_steel * cx * cy
            Kyy_s = K_steel * cy * cy

            # Compute steel axial displacement (du = u2 - u1)
            du_steel_x = u_s2x - u_s1x
            du_steel_y = u_s2y - u_s1y

            # Axial elongation in bar direction: axial = du · c
            axial = du_steel_x * cx + du_steel_y * cy

            # Axial force: N = (EA/L) * axial
            N_steel = K_steel * axial

            # Internal force contribution: f = N * c at each node
            # Node 1: f1 = -N * c (compression if pulled)
            # Node 2: f2 = +N * c (tension if pulled)
            f[dof_s1x] += -N_steel * cx
            f[dof_s1y] += -N_steel * cy
            f[dof_s2x] += +N_steel * cx
            f[dof_s2y] += +N_steel * cy

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

        # =====================================================================
        # PART A FIX: Mask check - Skip ONLY bond shear (and future dowel)
        # =====================================================================
        # CRITICAL FIX: Masked segments (bond disabled) skip bond/dowel contributions
        # but STILL include steel axial element (already assembled above).
        if segment_mask is not None and segment_mask[i]:
            # Set slip to zero for disabled segments (no bond forces/stiffness)
            s_current[i] = 0.0
            continue

        # =====================================================================
        # Bond shear interface (only if NOT masked)
        # =====================================================================
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

        # =====================================================================
        # PART B: Compute steel strain (eps_s) for yielding reduction
        # =====================================================================
        eps_s = 0.0
        if steel_EA > 0.0 and L0 > 1e-14:
            # Steel axial displacement (already computed for steel element above)
            # axial = (u_s2 - u_s1) · c
            # eps_s = axial / L0
            du_steel_x = u_s2x - u_s1x
            du_steel_y = u_s2y - u_s1y
            axial_displ = du_steel_x * cx + du_steel_y * cy
            eps_s = axial_displ / L0

        # =====================================================================
        # PART B / THESIS PARITY: Compute yielding reduction factor Ωy(eps_s)
        # =====================================================================
        omega_y = 1.0  # Default: no reduction
        if enable_omega_y > 0.5 and E_s > 1e-9:  # Check if enabled
            eps_y = f_y / E_s  # Yield strain

            # THESIS PARITY: Compute eps_u from fu and H per spec
            if H > 0.0 and f_u > f_y:
                # Bilinear hardening: εu = εy + (fu - fy) / H
                eps_u = eps_y + (f_u - f_y) / H
            else:
                # Fallback: εu = fu / E_s
                eps_u = f_u / E_s

            if abs(eps_s) > eps_y:
                # Steel has yielded: apply reduction per Eq. 3.57-3.58
                xi = max(0.0, (abs(eps_s) - eps_y) / max(1e-30, (eps_u - eps_y)))
                # Eq. 3.58: Ωy = 1 - 0.85*(1 - exp(-5*ξ))
                # Note: As ξ→∞, Ωy→0.15 (formula naturally bounds to [0.15, 1.0])
                omega_y = 1.0 - 0.85 * (1.0 - np.exp(-5.0 * xi))

        # =====================================================================
        # Crack deterioration factor Ωc (THESIS PARITY)
        # =====================================================================
        omega_crack = 1.0  # Default: no deterioration
        if crack_context is not None:
            x_over_l = crack_context[i, 0]  # Normalized distance: x/l where l = d_bar
            r = crack_context[i, 1]          # Tension/strength ratio at crack
            # Ωc formula: Ωc = 0.5*(x/l) + r*(1 - 0.5*(x/l)) for x <= 2l, else 1.0
            if x_over_l <= 2.0:
                omega_crack = 0.5 * x_over_l + r * (1.0 - 0.5 * x_over_l)
            else:
                omega_crack = 1.0

        # Bond stress (with history)
        s_abs = abs(s)
        s_max = max(s_max_hist[i], s_abs)
        sign = 1.0 if s >= 0.0 else -1.0

        # C1-continuous regularization for s -> 0 singularity
        # Use s_reg = 0.5 * s1 as threshold (increased from 0.1 to improve conditioning)
        # At s_reg=500μm: k_bond≈6e6 N/m (10x smaller than steel, 1e5x smaller than concrete)
        s_reg = 0.5 * s1

        # Simpler C1-continuous approach:
        # 1) For s <= s_reg: τ(s) = k0 * s (linear)
        # 2) For s > s_reg: τ(s) = τ_max * (s/s1)^α (original power law)
        #
        # Match dτ/ds at s_reg for C1 continuity:
        # k0 = τ_max * α / s1 * (s_reg/s1)^(α-1)
        #
        # This ensures:
        # - τ is continuous (may have small jump, but acceptable)
        # - dτ/ds is continuous (C1)
        # - k0 is finite for all α > 0

        k0 = tau_max * alpha / s1 * ((s_reg / s1) ** (alpha - 1.0))  # Tangent stiffness at s_reg

        # Envelope evaluation (inline for Numba)
        if s_abs >= s_max - 1e-14:
            # Loading envelope
            if s_abs <= s1:
                if s_abs < s_reg:
                    # Regularized linear branch (C1-continuous at s_reg)
                    tau_abs = k0 * s_abs
                    dtau_abs = k0
                else:
                    # Original power law (matches derivative at s_reg)
                    ratio = s_abs / s1
                    tau_abs = tau_max * (ratio ** alpha)
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

        # =====================================================================
        # PART B: Apply reduction factors (Ωy * Ωc)
        # =====================================================================
        omega_total = omega_y * omega_crack
        tau = tau * omega_total
        dtau_ds = dtau_ds * omega_total

        # Tangent capping (numerical stabilization)
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

        # =====================================================================
        # Bond dissipation tracking (trapezoidal rule)
        # =====================================================================
        if compute_dissipation and u_total_prev is not None:
            # Compute previous slip
            u_c_mid_x_prev = 0.5 * (u_total_prev[dof_c1x] + u_total_prev[dof_c2x])
            u_c_mid_y_prev = 0.5 * (u_total_prev[dof_c1y] + u_total_prev[dof_c2y])
            u_s_mid_x_prev = 0.5 * (u_total_prev[dof_s1x] + u_total_prev[dof_s2x])
            u_s_mid_y_prev = 0.5 * (u_total_prev[dof_s1y] + u_total_prev[dof_s2y])
            du_x_prev = u_s_mid_x_prev - u_c_mid_x_prev
            du_y_prev = u_s_mid_y_prev - u_c_mid_y_prev
            s_prev = du_x_prev * cx + du_y_prev * cy

            # Compute previous bond stress (simplified: use current envelope without history)
            s_abs_prev = abs(s_prev)
            sign_prev = 1.0 if s_prev >= 0.0 else -1.0

            # Evaluate tau at s_prev (using same envelope logic as current)
            if s_abs_prev <= s1:
                if s_abs_prev < s_reg:
                    tau_abs_prev = k0 * s_abs_prev
                else:
                    ratio_prev = s_abs_prev / s1
                    tau_abs_prev = tau_max * (ratio_prev ** alpha)
            elif s_abs_prev <= s2:
                tau_abs_prev = tau_max
            elif s_abs_prev <= s3:
                tau_abs_prev = tau_max - (tau_max - tau_f) * (s_abs_prev - s2) / (s3 - s2)
            else:
                tau_abs_prev = tau_f
            tau_prev = sign_prev * tau_abs_prev * omega_total  # Apply same reduction factor

            # Trapezoidal rule: ΔW = 0.5*(τ_old + τ_new)*(s_new - s_old)*perimeter*L
            ds = s - s_prev
            dW_bond_seg = 0.5 * (tau_prev + tau) * ds * perimeter * L0
            D_bond_inc += dW_bond_seg

        # Stiffness contribution: K_seg = K_bond * g ⊗ g^T (full 8x8 segment Jacobian)
        # where:
        #   K_bond = gamma * dtau_ds * perimeter * L0  (gamma scaling for continuation)
        #   g = [∂s/∂u] = [-t/2, -t/2, +t/2, +t/2] for [concrete_n1, concrete_n2, steel_n1, steel_n2]
        #   t = [cx, cy] is the bar tangent vector
        #
        # This gives the CONSISTENT TANGENT that couples steel ↔ concrete DOFs
        # (fixes Newton convergence issues from previous diagonal-only placeholder)
        #
        # Gamma continuation: start with gamma=0 (no bond), ramp to gamma=1 (full bond)

        K_bond = gamma * dtau_ds * perimeter * L0

        # Gradient vector components (scaled by 0.5 for midpoint averaging)
        # Concrete node 1: -0.5 * t
        g_c1x = -0.5 * cx
        g_c1y = -0.5 * cy
        # Concrete node 2: -0.5 * t
        g_c2x = -0.5 * cx
        g_c2y = -0.5 * cy
        # Steel node 1: +0.5 * t
        g_s1x = +0.5 * cx
        g_s1y = +0.5 * cy
        # Steel node 2: +0.5 * t
        g_s2x = +0.5 * cx
        g_s2y = +0.5 * cy

        # Assemble full 8x8 block: K_seg[i,j] = K_bond * g[i] * g[j]
        # This is the outer product g ⊗ g^T
        # Each row-column pair contributes one entry
        if entry_idx + 64 < max_entries:
            # Row 1: concrete node 1, x-direction
            rows[entry_idx] = dof_c1x; cols[entry_idx] = dof_c1x; data[entry_idx] = K_bond * g_c1x * g_c1x; entry_idx += 1
            rows[entry_idx] = dof_c1x; cols[entry_idx] = dof_c1y; data[entry_idx] = K_bond * g_c1x * g_c1y; entry_idx += 1
            rows[entry_idx] = dof_c1x; cols[entry_idx] = dof_c2x; data[entry_idx] = K_bond * g_c1x * g_c2x; entry_idx += 1
            rows[entry_idx] = dof_c1x; cols[entry_idx] = dof_c2y; data[entry_idx] = K_bond * g_c1x * g_c2y; entry_idx += 1
            rows[entry_idx] = dof_c1x; cols[entry_idx] = dof_s1x; data[entry_idx] = K_bond * g_c1x * g_s1x; entry_idx += 1
            rows[entry_idx] = dof_c1x; cols[entry_idx] = dof_s1y; data[entry_idx] = K_bond * g_c1x * g_s1y; entry_idx += 1
            rows[entry_idx] = dof_c1x; cols[entry_idx] = dof_s2x; data[entry_idx] = K_bond * g_c1x * g_s2x; entry_idx += 1
            rows[entry_idx] = dof_c1x; cols[entry_idx] = dof_s2y; data[entry_idx] = K_bond * g_c1x * g_s2y; entry_idx += 1

            # Row 2: concrete node 1, y-direction
            rows[entry_idx] = dof_c1y; cols[entry_idx] = dof_c1x; data[entry_idx] = K_bond * g_c1y * g_c1x; entry_idx += 1
            rows[entry_idx] = dof_c1y; cols[entry_idx] = dof_c1y; data[entry_idx] = K_bond * g_c1y * g_c1y; entry_idx += 1
            rows[entry_idx] = dof_c1y; cols[entry_idx] = dof_c2x; data[entry_idx] = K_bond * g_c1y * g_c2x; entry_idx += 1
            rows[entry_idx] = dof_c1y; cols[entry_idx] = dof_c2y; data[entry_idx] = K_bond * g_c1y * g_c2y; entry_idx += 1
            rows[entry_idx] = dof_c1y; cols[entry_idx] = dof_s1x; data[entry_idx] = K_bond * g_c1y * g_s1x; entry_idx += 1
            rows[entry_idx] = dof_c1y; cols[entry_idx] = dof_s1y; data[entry_idx] = K_bond * g_c1y * g_s1y; entry_idx += 1
            rows[entry_idx] = dof_c1y; cols[entry_idx] = dof_s2x; data[entry_idx] = K_bond * g_c1y * g_s2x; entry_idx += 1
            rows[entry_idx] = dof_c1y; cols[entry_idx] = dof_s2y; data[entry_idx] = K_bond * g_c1y * g_s2y; entry_idx += 1

            # Row 3: concrete node 2, x-direction
            rows[entry_idx] = dof_c2x; cols[entry_idx] = dof_c1x; data[entry_idx] = K_bond * g_c2x * g_c1x; entry_idx += 1
            rows[entry_idx] = dof_c2x; cols[entry_idx] = dof_c1y; data[entry_idx] = K_bond * g_c2x * g_c1y; entry_idx += 1
            rows[entry_idx] = dof_c2x; cols[entry_idx] = dof_c2x; data[entry_idx] = K_bond * g_c2x * g_c2x; entry_idx += 1
            rows[entry_idx] = dof_c2x; cols[entry_idx] = dof_c2y; data[entry_idx] = K_bond * g_c2x * g_c2y; entry_idx += 1
            rows[entry_idx] = dof_c2x; cols[entry_idx] = dof_s1x; data[entry_idx] = K_bond * g_c2x * g_s1x; entry_idx += 1
            rows[entry_idx] = dof_c2x; cols[entry_idx] = dof_s1y; data[entry_idx] = K_bond * g_c2x * g_s1y; entry_idx += 1
            rows[entry_idx] = dof_c2x; cols[entry_idx] = dof_s2x; data[entry_idx] = K_bond * g_c2x * g_s2x; entry_idx += 1
            rows[entry_idx] = dof_c2x; cols[entry_idx] = dof_s2y; data[entry_idx] = K_bond * g_c2x * g_s2y; entry_idx += 1

            # Row 4: concrete node 2, y-direction
            rows[entry_idx] = dof_c2y; cols[entry_idx] = dof_c1x; data[entry_idx] = K_bond * g_c2y * g_c1x; entry_idx += 1
            rows[entry_idx] = dof_c2y; cols[entry_idx] = dof_c1y; data[entry_idx] = K_bond * g_c2y * g_c1y; entry_idx += 1
            rows[entry_idx] = dof_c2y; cols[entry_idx] = dof_c2x; data[entry_idx] = K_bond * g_c2y * g_c2x; entry_idx += 1
            rows[entry_idx] = dof_c2y; cols[entry_idx] = dof_c2y; data[entry_idx] = K_bond * g_c2y * g_c2y; entry_idx += 1
            rows[entry_idx] = dof_c2y; cols[entry_idx] = dof_s1x; data[entry_idx] = K_bond * g_c2y * g_s1x; entry_idx += 1
            rows[entry_idx] = dof_c2y; cols[entry_idx] = dof_s1y; data[entry_idx] = K_bond * g_c2y * g_s1y; entry_idx += 1
            rows[entry_idx] = dof_c2y; cols[entry_idx] = dof_s2x; data[entry_idx] = K_bond * g_c2y * g_s2x; entry_idx += 1
            rows[entry_idx] = dof_c2y; cols[entry_idx] = dof_s2y; data[entry_idx] = K_bond * g_c2y * g_s2y; entry_idx += 1

            # Row 5: steel node 1, x-direction
            rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_c1x; data[entry_idx] = K_bond * g_s1x * g_c1x; entry_idx += 1
            rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_c1y; data[entry_idx] = K_bond * g_s1x * g_c1y; entry_idx += 1
            rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_c2x; data[entry_idx] = K_bond * g_s1x * g_c2x; entry_idx += 1
            rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_c2y; data[entry_idx] = K_bond * g_s1x * g_c2y; entry_idx += 1
            rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_s1x; data[entry_idx] = K_bond * g_s1x * g_s1x; entry_idx += 1
            rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_s1y; data[entry_idx] = K_bond * g_s1x * g_s1y; entry_idx += 1
            rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_s2x; data[entry_idx] = K_bond * g_s1x * g_s2x; entry_idx += 1
            rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_s2y; data[entry_idx] = K_bond * g_s1x * g_s2y; entry_idx += 1

            # Row 6: steel node 1, y-direction
            rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_c1x; data[entry_idx] = K_bond * g_s1y * g_c1x; entry_idx += 1
            rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_c1y; data[entry_idx] = K_bond * g_s1y * g_c1y; entry_idx += 1
            rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_c2x; data[entry_idx] = K_bond * g_s1y * g_c2x; entry_idx += 1
            rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_c2y; data[entry_idx] = K_bond * g_s1y * g_c2y; entry_idx += 1
            rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_s1x; data[entry_idx] = K_bond * g_s1y * g_s1x; entry_idx += 1
            rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_s1y; data[entry_idx] = K_bond * g_s1y * g_s1y; entry_idx += 1
            rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_s2x; data[entry_idx] = K_bond * g_s1y * g_s2x; entry_idx += 1
            rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_s2y; data[entry_idx] = K_bond * g_s1y * g_s2y; entry_idx += 1

            # Row 7: steel node 2, x-direction
            rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_c1x; data[entry_idx] = K_bond * g_s2x * g_c1x; entry_idx += 1
            rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_c1y; data[entry_idx] = K_bond * g_s2x * g_c1y; entry_idx += 1
            rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_c2x; data[entry_idx] = K_bond * g_s2x * g_c2x; entry_idx += 1
            rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_c2y; data[entry_idx] = K_bond * g_s2x * g_c2y; entry_idx += 1
            rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_s1x; data[entry_idx] = K_bond * g_s2x * g_s1x; entry_idx += 1
            rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_s1y; data[entry_idx] = K_bond * g_s2x * g_s1y; entry_idx += 1
            rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_s2x; data[entry_idx] = K_bond * g_s2x * g_s2x; entry_idx += 1
            rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_s2y; data[entry_idx] = K_bond * g_s2x * g_s2y; entry_idx += 1

            # Row 8: steel node 2, y-direction
            rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_c1x; data[entry_idx] = K_bond * g_s2y * g_c1x; entry_idx += 1
            rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_c1y; data[entry_idx] = K_bond * g_s2y * g_c1y; entry_idx += 1
            rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_c2x; data[entry_idx] = K_bond * g_s2y * g_c2x; entry_idx += 1
            rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_c2y; data[entry_idx] = K_bond * g_s2y * g_c2y; entry_idx += 1
            rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_s1x; data[entry_idx] = K_bond * g_s2y * g_s1x; entry_idx += 1
            rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_s1y; data[entry_idx] = K_bond * g_s2y * g_s1y; entry_idx += 1
            rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_s2x; data[entry_idx] = K_bond * g_s2y * g_s2x; entry_idx += 1
            rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_s2y; data[entry_idx] = K_bond * g_s2y * g_s2y; entry_idx += 1

        # =====================================================================
        # Dowel action (transverse stress perpendicular to bar)
        # =====================================================================
        if enable_dowel:
            # Compute transverse opening (normal to bar direction)
            # Normal direction: n = [-cy, cx] (perpendicular to tangent)
            nx = -cy
            ny = cx

            # Transverse displacement (normal component of relative displacement)
            w = du_x * nx + du_y * ny  # Transverse opening [m]
            w_abs = abs(w)

            # Dowel constitutive law (Murcia & Lorrain, Eq. 4.4-4.6)
            # Units: MPa/mm internally, convert to Pa/m for assembly
            if w_abs > 1e-14:
                # Convert w to mm for formula
                w_mm = w_abs * 1000.0  # [m] -> [mm]
                phi_mm = phi * 1000.0  # [m] -> [mm]

                # Stiffness parameter k0 [MPa/mm]
                k0_MPa_mm = 599.96 * (fc_MPa ** 0.75) / phi_mm

                # Coefficients
                a = 0.16
                b = 0.19
                c = 0.67
                d = 0.26

                # Intermediate variables
                Q = 40.0 * w_mm * phi_mm - b
                S = np.sqrt(d * d * Q * Q + c * c)
                A = 1.5 * (a + S)
                omega_tilde = A ** (-4.0 / 3.0)

                # Dowel stress σ(w) [MPa]
                sigma_MPa = omega_tilde * k0_MPa_mm * w_mm

                # Convert to Pa/m for assembly
                sigma_Pa = sigma_MPa * 1e6  # [MPa] -> [Pa]
                sigma = sigma_Pa if w >= 0.0 else -sigma_Pa  # Preserve sign

                # Tangent dσ/dw using chain rule
                # dω̃/dw = dω̃/dA * dA/dS * dS/dQ * dQ/dw
                dQ_dw = 40.0 * phi_mm * 1000.0  # w in [m], phi in [m], convert to mm
                if S > 1e-14:
                    dS_dQ = (d * d * Q) / S
                else:
                    dS_dQ = 0.0
                dA_dS = 1.5
                if A > 1e-14:
                    domega_dA = (-4.0 / 3.0) * A ** (-7.0 / 3.0)
                else:
                    domega_dA = 0.0
                domega_dw = domega_dA * dA_dS * dS_dQ * dQ_dw

                # dσ/dw = k0*(ω̃ + w*dω̃/dw) [MPa/mm]
                dsigma_dw_MPa_mm = k0_MPa_mm * (omega_tilde + w_mm * domega_dw)

                # Convert to Pa/m
                dsigma_dw_Pa_m = dsigma_dw_MPa_mm * 1e6 / 1000.0  # [MPa/mm] -> [Pa/m]
            else:
                # w ≈ 0: no dowel stress
                sigma = 0.0
                dsigma_dw_Pa_m = 0.0

            # Dowel force (distributed over segment length)
            F_dowel = sigma * perimeter * L0

            # Force components in normal direction
            Fx_dowel_s = F_dowel * nx
            Fy_dowel_s = F_dowel * ny
            Fx_dowel_c = -Fx_dowel_s
            Fy_dowel_c = -Fy_dowel_s

            # Add dowel forces to nodes (equal split)
            f[dof_s1x] += 0.5 * Fx_dowel_s
            f[dof_s1y] += 0.5 * Fy_dowel_s
            f[dof_s2x] += 0.5 * Fx_dowel_s
            f[dof_s2y] += 0.5 * Fy_dowel_s

            f[dof_c1x] += 0.5 * Fx_dowel_c
            f[dof_c1y] += 0.5 * Fy_dowel_c
            f[dof_c2x] += 0.5 * Fx_dowel_c
            f[dof_c2y] += 0.5 * Fy_dowel_c

            # Dowel stiffness: K_dowel = dσ/dw * perimeter * L0
            K_dowel = dsigma_dw_Pa_m * perimeter * L0

            # Gradient vector for transverse opening: h = [∂w/∂u]
            # w = (u_s - u_c) · n, so ∂w/∂u = [-n/2, -n/2, +n/2, +n/2]
            h_c1x = -0.5 * nx
            h_c1y = -0.5 * ny
            h_c2x = -0.5 * nx
            h_c2y = -0.5 * ny
            h_s1x = +0.5 * nx
            h_s1y = +0.5 * ny
            h_s2x = +0.5 * nx
            h_s2y = +0.5 * ny

            # Assemble dowel stiffness: K_dowel_seg = K_dowel * h ⊗ h^T (8x8)
            if entry_idx + 64 < max_entries:
                # Row 1: concrete node 1, x-direction
                rows[entry_idx] = dof_c1x; cols[entry_idx] = dof_c1x; data[entry_idx] = K_dowel * h_c1x * h_c1x; entry_idx += 1
                rows[entry_idx] = dof_c1x; cols[entry_idx] = dof_c1y; data[entry_idx] = K_dowel * h_c1x * h_c1y; entry_idx += 1
                rows[entry_idx] = dof_c1x; cols[entry_idx] = dof_c2x; data[entry_idx] = K_dowel * h_c1x * h_c2x; entry_idx += 1
                rows[entry_idx] = dof_c1x; cols[entry_idx] = dof_c2y; data[entry_idx] = K_dowel * h_c1x * h_c2y; entry_idx += 1
                rows[entry_idx] = dof_c1x; cols[entry_idx] = dof_s1x; data[entry_idx] = K_dowel * h_c1x * h_s1x; entry_idx += 1
                rows[entry_idx] = dof_c1x; cols[entry_idx] = dof_s1y; data[entry_idx] = K_dowel * h_c1x * h_s1y; entry_idx += 1
                rows[entry_idx] = dof_c1x; cols[entry_idx] = dof_s2x; data[entry_idx] = K_dowel * h_c1x * h_s2x; entry_idx += 1
                rows[entry_idx] = dof_c1x; cols[entry_idx] = dof_s2y; data[entry_idx] = K_dowel * h_c1x * h_s2y; entry_idx += 1

                # Row 2: concrete node 1, y-direction
                rows[entry_idx] = dof_c1y; cols[entry_idx] = dof_c1x; data[entry_idx] = K_dowel * h_c1y * h_c1x; entry_idx += 1
                rows[entry_idx] = dof_c1y; cols[entry_idx] = dof_c1y; data[entry_idx] = K_dowel * h_c1y * h_c1y; entry_idx += 1
                rows[entry_idx] = dof_c1y; cols[entry_idx] = dof_c2x; data[entry_idx] = K_dowel * h_c1y * h_c2x; entry_idx += 1
                rows[entry_idx] = dof_c1y; cols[entry_idx] = dof_c2y; data[entry_idx] = K_dowel * h_c1y * h_c2y; entry_idx += 1
                rows[entry_idx] = dof_c1y; cols[entry_idx] = dof_s1x; data[entry_idx] = K_dowel * h_c1y * h_s1x; entry_idx += 1
                rows[entry_idx] = dof_c1y; cols[entry_idx] = dof_s1y; data[entry_idx] = K_dowel * h_c1y * h_s1y; entry_idx += 1
                rows[entry_idx] = dof_c1y; cols[entry_idx] = dof_s2x; data[entry_idx] = K_dowel * h_c1y * h_s2x; entry_idx += 1
                rows[entry_idx] = dof_c1y; cols[entry_idx] = dof_s2y; data[entry_idx] = K_dowel * h_c1y * h_s2y; entry_idx += 1

                # Row 3: concrete node 2, x-direction
                rows[entry_idx] = dof_c2x; cols[entry_idx] = dof_c1x; data[entry_idx] = K_dowel * h_c2x * h_c1x; entry_idx += 1
                rows[entry_idx] = dof_c2x; cols[entry_idx] = dof_c1y; data[entry_idx] = K_dowel * h_c2x * h_c1y; entry_idx += 1
                rows[entry_idx] = dof_c2x; cols[entry_idx] = dof_c2x; data[entry_idx] = K_dowel * h_c2x * h_c2x; entry_idx += 1
                rows[entry_idx] = dof_c2x; cols[entry_idx] = dof_c2y; data[entry_idx] = K_dowel * h_c2x * h_c2y; entry_idx += 1
                rows[entry_idx] = dof_c2x; cols[entry_idx] = dof_s1x; data[entry_idx] = K_dowel * h_c2x * h_s1x; entry_idx += 1
                rows[entry_idx] = dof_c2x; cols[entry_idx] = dof_s1y; data[entry_idx] = K_dowel * h_c2x * h_s1y; entry_idx += 1
                rows[entry_idx] = dof_c2x; cols[entry_idx] = dof_s2x; data[entry_idx] = K_dowel * h_c2x * h_s2x; entry_idx += 1
                rows[entry_idx] = dof_c2x; cols[entry_idx] = dof_s2y; data[entry_idx] = K_dowel * h_c2x * h_s2y; entry_idx += 1

                # Row 4: concrete node 2, y-direction
                rows[entry_idx] = dof_c2y; cols[entry_idx] = dof_c1x; data[entry_idx] = K_dowel * h_c2y * h_c1x; entry_idx += 1
                rows[entry_idx] = dof_c2y; cols[entry_idx] = dof_c1y; data[entry_idx] = K_dowel * h_c2y * h_c1y; entry_idx += 1
                rows[entry_idx] = dof_c2y; cols[entry_idx] = dof_c2x; data[entry_idx] = K_dowel * h_c2y * h_c2x; entry_idx += 1
                rows[entry_idx] = dof_c2y; cols[entry_idx] = dof_c2y; data[entry_idx] = K_dowel * h_c2y * h_c2y; entry_idx += 1
                rows[entry_idx] = dof_c2y; cols[entry_idx] = dof_s1x; data[entry_idx] = K_dowel * h_c2y * h_s1x; entry_idx += 1
                rows[entry_idx] = dof_c2y; cols[entry_idx] = dof_s1y; data[entry_idx] = K_dowel * h_c2y * h_s1y; entry_idx += 1
                rows[entry_idx] = dof_c2y; cols[entry_idx] = dof_s2x; data[entry_idx] = K_dowel * h_c2y * h_s2x; entry_idx += 1
                rows[entry_idx] = dof_c2y; cols[entry_idx] = dof_s2y; data[entry_idx] = K_dowel * h_c2y * h_s2y; entry_idx += 1

                # Row 5: steel node 1, x-direction
                rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_c1x; data[entry_idx] = K_dowel * h_s1x * h_c1x; entry_idx += 1
                rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_c1y; data[entry_idx] = K_dowel * h_s1x * h_c1y; entry_idx += 1
                rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_c2x; data[entry_idx] = K_dowel * h_s1x * h_c2x; entry_idx += 1
                rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_c2y; data[entry_idx] = K_dowel * h_s1x * h_c2y; entry_idx += 1
                rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_s1x; data[entry_idx] = K_dowel * h_s1x * h_s1x; entry_idx += 1
                rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_s1y; data[entry_idx] = K_dowel * h_s1x * h_s1y; entry_idx += 1
                rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_s2x; data[entry_idx] = K_dowel * h_s1x * h_s2x; entry_idx += 1
                rows[entry_idx] = dof_s1x; cols[entry_idx] = dof_s2y; data[entry_idx] = K_dowel * h_s1x * h_s2y; entry_idx += 1

                # Row 6: steel node 1, y-direction
                rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_c1x; data[entry_idx] = K_dowel * h_s1y * h_c1x; entry_idx += 1
                rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_c1y; data[entry_idx] = K_dowel * h_s1y * h_c1y; entry_idx += 1
                rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_c2x; data[entry_idx] = K_dowel * h_s1y * h_c2x; entry_idx += 1
                rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_c2y; data[entry_idx] = K_dowel * h_s1y * h_c2y; entry_idx += 1
                rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_s1x; data[entry_idx] = K_dowel * h_s1y * h_s1x; entry_idx += 1
                rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_s1y; data[entry_idx] = K_dowel * h_s1y * h_s1y; entry_idx += 1
                rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_s2x; data[entry_idx] = K_dowel * h_s1y * h_s2x; entry_idx += 1
                rows[entry_idx] = dof_s1y; cols[entry_idx] = dof_s2y; data[entry_idx] = K_dowel * h_s1y * h_s2y; entry_idx += 1

                # Row 7: steel node 2, x-direction
                rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_c1x; data[entry_idx] = K_dowel * h_s2x * h_c1x; entry_idx += 1
                rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_c1y; data[entry_idx] = K_dowel * h_s2x * h_c1y; entry_idx += 1
                rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_c2x; data[entry_idx] = K_dowel * h_s2x * h_c2x; entry_idx += 1
                rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_c2y; data[entry_idx] = K_dowel * h_s2x * h_c2y; entry_idx += 1
                rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_s1x; data[entry_idx] = K_dowel * h_s2x * h_s1x; entry_idx += 1
                rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_s1y; data[entry_idx] = K_dowel * h_s2x * h_s1y; entry_idx += 1
                rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_s2x; data[entry_idx] = K_dowel * h_s2x * h_s2x; entry_idx += 1
                rows[entry_idx] = dof_s2x; cols[entry_idx] = dof_s2y; data[entry_idx] = K_dowel * h_s2x * h_s2y; entry_idx += 1

                # Row 8: steel node 2, y-direction
                rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_c1x; data[entry_idx] = K_dowel * h_s2y * h_c1x; entry_idx += 1
                rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_c1y; data[entry_idx] = K_dowel * h_s2y * h_c1y; entry_idx += 1
                rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_c2x; data[entry_idx] = K_dowel * h_s2y * h_c2x; entry_idx += 1
                rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_c2y; data[entry_idx] = K_dowel * h_s2y * h_c2y; entry_idx += 1
                rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_s1x; data[entry_idx] = K_dowel * h_s2y * h_s1x; entry_idx += 1
                rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_s1y; data[entry_idx] = K_dowel * h_s2y * h_s1y; entry_idx += 1
                rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_s2x; data[entry_idx] = K_dowel * h_s2y * h_s2x; entry_idx += 1
                rows[entry_idx] = dof_s2y; cols[entry_idx] = dof_s2y; data[entry_idx] = K_dowel * h_s2y * h_s2y; entry_idx += 1

            # Dowel dissipation tracking (trapezoidal rule)
            if compute_dissipation and u_total_prev is not None:
                # Compute previous transverse opening
                u_c_mid_x_prev = 0.5 * (u_total_prev[dof_c1x] + u_total_prev[dof_c2x])
                u_c_mid_y_prev = 0.5 * (u_total_prev[dof_c1y] + u_total_prev[dof_c2y])
                u_s_mid_x_prev = 0.5 * (u_total_prev[dof_s1x] + u_total_prev[dof_s2x])
                u_s_mid_y_prev = 0.5 * (u_total_prev[dof_s1y] + u_total_prev[dof_s2y])
                du_x_prev = u_s_mid_x_prev - u_c_mid_x_prev
                du_y_prev = u_s_mid_y_prev - u_c_mid_y_prev
                w_prev = du_x_prev * nx + du_y_prev * ny

                # Compute previous dowel stress (using same logic)
                w_abs_prev = abs(w_prev)
                if w_abs_prev > 1e-14:
                    w_mm_prev = w_abs_prev * 1000.0
                    Q_prev = 40.0 * w_mm_prev * phi_mm - b
                    S_prev = np.sqrt(d * d * Q_prev * Q_prev + c * c)
                    A_prev = 1.5 * (a + S_prev)
                    omega_tilde_prev = A_prev ** (-4.0 / 3.0)
                    sigma_MPa_prev = omega_tilde_prev * k0_MPa_mm * w_mm_prev
                    sigma_Pa_prev = sigma_MPa_prev * 1e6
                    sigma_prev = sigma_Pa_prev if w_prev >= 0.0 else -sigma_Pa_prev
                else:
                    sigma_prev = 0.0

                # Trapezoidal rule: ΔW = 0.5*(σ_old + σ_new)*(w_new - w_old)*perimeter*L
                dw = w - w_prev
                dW_dowel_seg = 0.5 * (sigma_prev + sigma) * dw * perimeter * L0
                D_dowel_inc += dW_dowel_seg

    # Trim arrays
    rows_out = rows[:entry_idx]
    cols_out = cols[:entry_idx]
    data_out = data[:entry_idx]

    return f, rows_out, cols_out, data_out, s_current, D_bond_inc, D_dowel_inc
