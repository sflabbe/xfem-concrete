"""Direct assembly pullout test (bypasses run_analysis_xfem issues)."""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xfem_clean.fem.mesh import structured_quad_mesh
from xfem_clean.rebar import prepare_rebar_segments
from xfem_clean.xfem.dofs_single import build_xfem_dofs
from xfem_clean.xfem.geometry import XFEMCrack
from xfem_clean.xfem.assembly_single import assemble_xfem_system
from xfem_clean.xfem.material import plane_stress_C
from xfem_clean.cohesive_laws import CohesiveLaw
from xfem_clean.bond_slip import BondSlipModelCode2010, BondSlipStateArrays
from xfem_clean.fem.bcs import apply_dirichlet
from xfem_clean.utils.scaling import diagonal_equilibration, unscale_solution
import scipy.sparse.linalg as spla

print("="*70)
print("PULLOUT TEST - DIRECT ASSEMBLY")
print("="*70)

# Geometry (simple specimen)
L = 0.20  # [m] embedment length
H = 0.05  # [m] height
nx, ny = 20, 5

# Material
E = 33e9  # [Pa]
nu = 0.2
fc = 38e6
ft = 2.9e6
Gf = 120
thickness = 0.10  # [m]

# Steel
d_bar = 0.016
A_bar = np.pi * (d_bar/2)**2
E_steel = 200e9
steel_EA_min = 1e3

# Setup
nodes, elems = structured_quad_mesh(L, H, nx, ny)
rebar_segs = prepare_rebar_segments(nodes, cover=0.025)
crack = XFEMCrack(x0=L/2, y0=0.0, tip_x=L/2, tip_y=0.0, stop_y=H, active=False)
dofs = build_xfem_dofs(
    nodes, elems, crack,
    H_region_ymax=0.0,
    tip_patch=(0,0,0,0),
    rebar_segs=rebar_segs,
    enable_bond_slip=True
)

C = plane_stress_C(E, nu)
law = CohesiveLaw(Kn=0.1*E/0.05, ft=ft, Gf=Gf)
bond_law = BondSlipModelCode2010(f_cm=fc, d_bar=d_bar, condition="good")
bond_states = BondSlipStateArrays.zeros(len(rebar_segs))

print(f"Mesh: {len(nodes)} nodes, {len(elems)} elements")
print(f"DOFs: {dofs.ndof} (steel offset: {dofs.steel_dof_offset})")
print(f"Bond s_reg: {0.1 * bond_law.s1 * 1e6:.1f} μm")

# Boundary conditions: pullout setup
# Fix left end (concrete + steel)
left_nodes = np.where(nodes[:, 0] < 1e-6)[0]
fixed = {}
for n in left_nodes:
    fixed[2*n] = 0.0  # u_x = 0
    fixed[2*n+1] = 0.0  # u_y = 0

# Fix steel DOFs at left end
for seg in rebar_segs:
    n1 = int(seg[0])
    if nodes[n1, 0] < 1e-6:  # Left end
        if dofs.steel[n1, 0] >= 0:
            fixed[dofs.steel[n1, 0]] = 0.0
            fixed[dofs.steel[n1, 1]] = 0.0

# Apply displacement to concrete at right end (pullout)
right_nodes = np.where(nodes[:, 0] > L - 1e-6)[0]
load_dofs_concrete = [2*n for n in right_nodes]  # x-direction

print(f"BCs: {len(fixed)} fixed DOFs, {len(load_dofs_concrete)} load DOFs")

# Load steps
nsteps = 10
u_max = 0.001  # [m] = 1 mm
displacements = []
reactions = []
max_slips = []

q = np.zeros(dofs.ndof)

print(f"\nRunning {nsteps} load steps...")

for step in range(1, nsteps+1):
    u_applied = (step / nsteps) * u_max

    # Update BCs
    fixed_step = fixed.copy()
    for dof in load_dofs_concrete:
        fixed_step[dof] = u_applied

    # Newton iterations
    converged = False
    for it in range(30):
        # Assembly
        K, fint, _, _, _, bond_new, _, _ = assemble_xfem_system(
            nodes=nodes,
            elems=elems,
            dofs=dofs,
            crack=crack,
            C=C,
            thickness=thickness,
            q=q,
            law=law,
            coh_states_comm={},
            tip_enr_radius=0.01,
            rebar_segs=rebar_segs,
            bond_law=bond_law,
            bond_states_comm=bond_states,
            enable_bond_slip=True,
            steel_EA=steel_EA_min,
            use_numba=True,
        )

        # Apply BCs
        r = fint.copy()
        free, K_ff, r_f, _ = apply_dirichlet(K, r, fixed_step, q)
        rhs = -r_f
        norm_r = np.linalg.norm(rhs)

        # Check convergence
        if norm_r < 1e-3:
            converged = True
            if step <= 3 or step == nsteps:
                print(f"  Step {step}, iter {it+1}: Converged (||r||={norm_r:.2e})")
            break

        # Solve with diagonal scaling
        K_ff_scaled, rhs_scaled, D_inv = diagonal_equilibration(K_ff, rhs)
        try:
            du_scaled = spla.spsolve(K_ff_scaled, rhs_scaled)
            du = unscale_solution(du_scaled, D_inv)
        except:
            print(f"  Step {step}, iter {it+1}: Solve failed")
            break

        # Check stagnation
        norm_du = np.linalg.norm(du)
        if norm_du < 1e-9:
            print(f"  Step {step}, iter {it+1}: Stagnated (||du||={norm_du:.2e}, ||r||={norm_r:.2e})")
            break

        # Update
        q[free] += du
        for dof, val in fixed_step.items():
            q[dof] = val

        if step <= 3 and it < 3:
            print(f"  Step {step}, iter {it+1}: ||r||={norm_r:.2e}, ||du||={norm_du:.2e}")

    if not converged:
        print(f"✗ Step {step} failed to converge")
        break

    # Compute reaction
    reaction = -sum(r[d] for d in load_dofs_concrete)

    # Update bond states
    bond_states = bond_new

    displacements.append(u_applied)
    reactions.append(reaction)
    max_slips.append(bond_states.s_max.max())

    print(f"✓ Step {step:2d}: u={u_applied*1e3:.3f}mm, P={reaction/1e3:.2f}kN, s_max={bond_states.s_max.max()*1e6:.2f}μm")

print(f"\n✓ Analysis completed: {len(displacements)} steps")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(np.array(displacements)*1e3, np.array(reactions)/1e3, 'b-o', linewidth=2, markersize=4)
ax1.set_xlabel('Displacement [mm]', fontsize=12)
ax1.set_ylabel('Pullout load [kN]', fontsize=12)
ax1.set_title('Load-Displacement Curve', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2.plot(np.array(displacements)*1e3, np.array(max_slips)*1e6, 'r-o', linewidth=2, markersize=4)
ax2.set_xlabel('Displacement [mm]', fontsize=12)
ax2.set_ylabel('Max slip [μm]', fontsize=12)
ax2.set_title('Slip Evolution', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()

output_dir = Path(__file__).parent / "outputs"
output_dir.mkdir(exist_ok=True)
plot_path = output_dir / "pullout_direct_assembly.png"
plt.savefig(plot_path, dpi=150)
print(f"\n✓ Plot saved: {plot_path}")

plt.show()
print("="*70)
