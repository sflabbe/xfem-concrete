"""Case 11: RC balcony cantilever (slab strip) under Eurocode SLS (positive cyclic).

Displacement-controlled (stable) cantilever slab-strip benchmark.
Eurocode service load levels are mapped to equivalent *elastic* tip deflections
for a cantilever under uniformly distributed load:

    delta_tip = w * L^4 / (8 * E * I),   w = q * b

Config units: MPa, mm, N/mm.
"""

from __future__ import annotations

from typing import List

from examples.gutierrez_thesis.case_config import (
    CaseConfig,
    GeometryConfig,
    ConcreteConfig,
    SteelConfig,
    CEBFIPBondLaw,
    RebarLayer,
    CyclicLoading,
    OutputConfig,
)


def _tip_deflection_udl_mm(
    q_kN_m2: float,
    L_mm: float,
    b_mm: float,
    h_mm: float,
    E_MPa: float,
) -> float:
    """Elastic tip deflection (mm) for cantilever under UDL q [kN/m^2] over strip width b."""
    L = L_mm * 1e-3
    b = b_mm * 1e-3
    h = h_mm * 1e-3
    E = E_MPa * 1e6

    # Line load w [N/m]
    w = q_kN_m2 * 1e3 * b

    # Second moment of area I for rectangle b*h^3/12 [m^4]
    I = b * h**3 / 12.0

    # delta_tip [m] for UDL
    delta_m = w * (L**4) / (8.0 * E * I)
    return float(delta_m * 1e3)  # m -> mm


def _ramp_values(end_mm: float, n: int) -> List[float]:
    """Monotone ramp from 0 to end_mm with n equal increments (including end, excluding 0)."""
    if n <= 0:
        return [float(end_mm)]
    return [float(end_mm) * (i / n) for i in range(1, n + 1)]


def create_case_11() -> CaseConfig:
    """Balcony cantilever SLS (positive cyclic) until first crack."""

    # Geometry (slab strip)
    L_mm = 2000.0
    H_mm = 200.0
    b_mm = 1000.0  # 1 m strip width

    geometry = GeometryConfig(
        length=L_mm,
        height=H_mm,
        thickness=b_mm,
        n_elem_x=80,
        n_elem_y=8,
        element_type="Q4",
    )

    # Concrete: C20/25 (approx.)
    concrete = ConcreteConfig(
        E=30000.0,  # MPa
        nu=0.20,
        f_c=20.0,   # MPa
        f_t=2.2,    # MPa
        G_f=0.070,  # N/mm (order-of-magnitude for normal-strength)
        eps_cu=-0.0035,
        model_type="cdp_full",
    )

    # Steel: B500 (approx.)
    steel = SteelConfig(
        E=200000.0,
        nu=0.30,
        f_y=500.0,
        f_u=600.0,
        hardening_modulus=2000.0,
    )

    # Bond law (borrowed from BOSCO beams as a reasonable default for Ø12)
    bond_law_12 = CEBFIPBondLaw(
        s1=1.0,
        s2=2.0,
        s3=12.0,
        tau_max=12.0,
        tau_f=2.0,
        alpha=0.4,
    )

    # Top reinforcement (cantilever: hogging moment -> tension at top)
    cover_mm = 35.0
    dia_mm = 12.0
    y_top_bar = H_mm - (cover_mm + 0.5 * dia_mm)

    rebar_top = RebarLayer(
        diameter=dia_mm,
        y_position=y_top_bar,
        n_bars=7,  # ~Ø12 @ 150 mm over 1 m
        steel=steel,
        bond_law=bond_law_12,
        orientation_deg=0.0,
    )

    # Eurocode-like SLS loads (typical balcony)
    # - self weight: gamma_conc ~ 25 kN/m^3, t=0.20m -> gk ~ 5.0 kN/m^2
    # - imposed: qk ~ 4.0 kN/m^2
    gk = 25.0 * (H_mm * 1e-3)
    qk = 4.0
    psi1 = 0.5  # frequent
    psi2 = 0.3  # quasi-permanent

    q_qp = gk + psi2 * qk
    q_freq = gk + psi1 * qk
    q_char = gk + 1.0 * qk

    u_qp = _tip_deflection_udl_mm(q_qp, L_mm, b_mm, H_mm, concrete.E)
    u_freq = _tip_deflection_udl_mm(q_freq, L_mm, b_mm, H_mm, concrete.E)
    u_char = _tip_deflection_udl_mm(q_char, L_mm, b_mm, H_mm, concrete.E)

    # Positive cyclic trajectory (explicit):
    # 0 -> quasi-permanent (ramp) -> cycles between qp and frequent -> ramp to characteristic -> push
    ramp_steps = 4
    n_cycles = 3

    trajectory: List[float] = [0.0]
    trajectory += _ramp_values(u_qp, ramp_steps)
    for _ in range(n_cycles):
        trajectory.append(u_freq)
        trajectory.append(u_qp)

    # Final ramp beyond characteristic to ensure cracking if not already cracked
    trajectory.append(u_char)
    trajectory.append(1.10 * u_char)
    trajectory.append(1.20 * u_char)

    # Loading patch near free end on top edge
    loading = CyclicLoading(
        targets=trajectory,
        load_x_center=L_mm,
        load_halfwidth=75.0,
        n_cycles_per_target=1,
        targets_are_trajectory=True,
    )

    outputs = OutputConfig(
        output_dir="examples/gutierrez_thesis/outputs/case_11",
        case_name="balcony_cantilever_sls",
        save_load_displacement=True,
        save_crack_data=True,
        save_energy=True,
        save_crack_pattern=True,
        save_damage_field=True,
        save_deformed_shape=True,
        save_metrics=True,
        save_vtk=True,
        vtk_frequency=2,
        compute_crack_widths=True,
        compute_slip_profiles=True,
        compute_bond_profiles=True,
        compute_steel_forces=True,
        compute_base_moment=False,
    )

    case = CaseConfig(
        case_id="11_balcony_cantilever_sls",
        description=(
            "RC balcony cantilever (1m strip) under Eurocode SLS: quasi-permanent/frequent cycles, "
            "then ramp to characteristic (displacement-controlled)"
        ),
        geometry=geometry,
        concrete=concrete,
        loading=loading,
        outputs=outputs,
        rebar_layers=[rebar_top],
        tolerance=1e-6,
        use_line_search=True,
        use_substepping=True,
    )

    return case
