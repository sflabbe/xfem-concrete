#!/usr/bin/env python3
import math
import sys
from pathlib import Path

# Allow running examples without installing the package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import argparse

def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bulk-material", choices=["elastic", "dp", "cdp", "cdp-lite"], default="elastic",
                    help="Bulk constitutive model: elastic | dp | cdp (default: elastic)")
    ap.add_argument("--dp-phi-deg", type=float, default=30.0, help="Drucker–Prager friction angle phi [deg]")
    ap.add_argument("--dp-cohesion-mpa", type=float, default=2.0, help="Drucker–Prager cohesion [MPa]")
    ap.add_argument("--dp-H-mpa", type=float, default=0.0, help="Drucker–Prager hardening modulus H [MPa]")
    ap.add_argument("--fc-mpa", type=float, default=32.0, help="Concrete compressive strength |fc| [MPa] (CDP)")
    ap.add_argument("--cdp-phi-deg", type=float, default=30.0, help="CDP-lite friction angle phi [deg]")
    ap.add_argument("--cdp-H-mpa", type=float, default=0.0, help="CDP-lite hardening modulus H [MPa]")
    return ap.parse_args()

ARGS = _parse_args()


from xfem_clean.xfem_xfem import XFEMModel, run_analysis_xfem, nodal_average_stress_fields

# Vendored dependency (included in this ZIP)
from cdp_generator import material_properties as mp
from cdp_generator import tension as tn
from cdp_generator import compression as cp

# -------------------------
# Beam + materials (SI units)
# -------------------------
L = 5.0           # m
h = 0.50          # m
b = 0.25          # m

cover = 0.05      # m
nx, ny = 120, 18

# target displacement
umax = 0.010      # m
nsteps = 30

# Concrete C20/25 (Eurocode-ish)
# cdp_generator returns MPa and N/mm based props; we convert as needed.
props = mp.ec2_concrete("C20/25")
E_c = props["E_cm"]  # MPa
f_ctm = props["f_ctm"]  # MPa
Gf = props["G_f"]  # N/mm (note: cdp_generator uses N/mm)
print(f"[cdp_generator] C20/25: f_ctm={f_ctm:.3f} MPa  E_c={E_c:.1f} MPa  Gf={Gf:.4f} N/mm")

# For the cohesive crack (Rankine): take ft = f_ctm (Pa)
ft = f_ctm * 1e6  # Pa

# Reinforcement (simplified)
# Use your earlier 2Ø12 as default
phi = 0.012  # m
As_bar = math.pi*(phi**2)/4.0
n_bars = 2
As_prov = n_bars * As_bar
print(f"[steel] As_prov={As_prov*1e6:.1f} mm² (2Ø12)")

model = XFEMModel(
    L=L, H=h, b=b,
    E=E_c*1e6,
    nu=0.20,
    ft=ft,
    Gf=Gf*1000.0,      # N/mm -> N/m
    steel_A_total=As_prov,
    steel_E=200e9,
    steel_fy=500e6,
    steel_fu=540e6,
    steel_Eh=2.0e9,
    bulk_material=str(ARGS.bulk_material),
    dp_phi_deg=float(ARGS.dp_phi_deg),
    dp_cohesion=float(ARGS.dp_cohesion_mpa) * 1e6,
    dp_H=float(ARGS.dp_H_mpa) * 1e6,
    fc=float(ARGS.fc_mpa) * 1e6,
    cdp_phi_deg=float(ARGS.cdp_phi_deg),
    cdp_H=float(ARGS.cdp_H_mpa) * 1e6,
    cover=cover,
    # solver controls (you can tune)
    newton_maxit=60,
    newton_tol_r=1e-2,
    newton_tol_rel=1e-6,
    line_search=True,
    max_subdiv=14,
    # crack controls (Gutierrez-like)
    crack_margin=0.30,
    crack_rho=0.25,
    crack_tip_stop_y=h,
    # Option-A (cohesive RC): use only Heaviside enrichment (no LEFM tip enrichment)
    tip_enr_radius=0.0,
    # inner cracking loop / adaptivity
    crack_max_inner=8,
    crack_seg_length=None,       # None -> dy per propagation
    dominant_crack=True,
    dominant_window=(0.45, 0.55),
    cand_ymax_factor=2.0,
)

# candidate points: three windows along bottom edge (still one dominant crack for validation)
model.cand_mode = "three"
model.dominant_crack = True
# debug verbosity
model.debug_substeps = True
model.debug_newton = True


# characteristic length for cohesive penalty scaling
dx = L / nx
dy = h / ny
model.lch = (dx*dy)**0.5       # importante: no dejar lch fijo
# Cohesive penalty: for quasi-brittle concrete, too large Kn makes the system
# ill-conditioned and can stall Newton right after crack initiation.
# 0.1 -> delta0 ~ O(10-30) µm for this mesh, which is usually much friendlier.
model.Kn_factor = 0.01
Kn = model.Kn_factor * model.E / model.lch
delta0 = model.ft / Kn
print(f"[cohesive] lch={model.lch:.4f} m  Kn={Kn:.3e} Pa/m  delta0={delta0*1e6:.2f} µm")


print("[solver] XFEM real (Heaviside + cohesive) + adaptive substepping (displacement-control)")
nodes, elems, u, results, crack = run_analysis_xfem(model, nx=nx, ny=ny, nsteps=nsteps, umax=umax)

# -------------------------
# Save results.csv
# -------------------------
with open("results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step","u_m","P_N","M_mid_Nm","kappa_1pm","R_m","tip_x_m","tip_y_m","crack_angle_deg","crack_active"])
    for row in results:
        w.writerow(row.tolist())
print("[io] wrote results.csv")

# -------------------------
# Basic plots (mesh + displacement magnitude + P-u)
# -------------------------
nnode = nodes.shape[0]
# u contains standard + enriched dofs. For mesh deformation/field plots use ONLY standard nodal dofs.
U_std = u[:2*nnode].reshape((nnode, 2))

disp_mag = np.linalg.norm(U_std, axis=1)

def plot_mesh(nodes_xy, conn_quads, title, fname, crack=None, u=None):
    fig, ax = plt.subplots(figsize=(10,3))
    for e,conn in enumerate(conn_quads):
        pts = nodes_xy[conn,:]
        pts2 = np.vstack([pts, pts[0]])
        ax.plot(pts2[:,0], pts2[:,1], linewidth=0.6)
    if crack is not None and crack.active:
        p0 = crack.p0(); pt = crack.pt()
        # "dark gradient" crack edge: multi-stroke
        for lw, a in [(8,0.15),(4,0.35),(2,0.85)]:
            ax.plot([p0[0], pt[0]],[p0[1], pt[1]], linewidth=lw, alpha=a, color="black")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)

# Undeformed mesh
plot_mesh(nodes, elems, "Mesh (undeformed)", "mesh_undeformed.png", crack=crack)

# Deformed mesh (NO scaling)
nodes_def = nodes + U_std
plot_mesh(nodes_def, elems, "Mesh (deformed, scale=1)", "mesh_deformed.png", crack=crack)

# Displacement magnitude field (scatter)
fig, ax = plt.subplots(figsize=(10,3))
sc = ax.tricontourf(nodes[:,0], nodes[:,1], disp_mag, levels=20)
fig.colorbar(sc, ax=ax, label="|u| [m]")
if crack.active:
    p0 = crack.p0(); pt = crack.pt()
    for lw, a in [(8,0.15),(4,0.35),(2,0.85)]:
        ax.plot([p0[0], pt[0]],[p0[1], pt[1]], linewidth=lw, alpha=a, color="black")
ax.set_aspect("equal")
ax.set_title("Displacement magnitude")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
fig.tight_layout()
fig.savefig("disp_mag.png", dpi=200)
plt.close(fig)

# P-u curve
u_m = results[:,1]
P_N = results[:,2]
fig, ax = plt.subplots()
ax.plot(u_m*1e3, P_N/1e3, marker="o")
ax.set_xlabel("u [mm]")
ax.set_ylabel("P [kN]")
ax.set_title("Load-displacement (displacement control)")
fig.tight_layout()
fig.savefig("pu_curve.png", dpi=200)
plt.close(fig)

print("[io] wrote mesh_undeformed.png, mesh_deformed.png, disp_mag.png, pu_curve.png")


# -------------------------
# Stress plots (Abaqus-like: nodal averaged + crack mask)
# -------------------------
def tri_from_quads(conn_quads):
    tris = []
    for conn in conn_quads:
        n0, n1, n2, n3 = [int(x) for x in conn]
        tris.append([n0, n1, n2])
        tris.append([n0, n2, n3])
    return np.asarray(tris, dtype=int)

def crack_mask_triangles(nodes_xy, tris, crack_obj):
    if crack_obj is None or (not crack_obj.active):
        return np.zeros(len(tris), dtype=bool)

    s_tip = float(crack_obj.s_tip())
    mask = np.zeros(len(tris), dtype=bool)

    for i, t in enumerate(tris):
        pts = nodes_xy[t, :]
        # signed distance to crack line
        phi = np.array([crack_obj.phi(float(p[0]), float(p[1])) for p in pts], dtype=float)
        # param along crack tangent from start
        ss = np.array([crack_obj.s(float(p[0]), float(p[1])) for p in pts], dtype=float)

        # only triangles that overlap the segment extent (0 <= s <= s_tip)
        if float(ss.max()) < -1e-12 or float(ss.min()) > s_tip + 1e-12:
            continue

        # if it spans both sides of the crack line -> mask it (gap)
        if float(phi.min()) < 0.0 and float(phi.max()) > 0.0:
            mask[i] = True

    return mask

tris = tri_from_quads(elems)
mask = crack_mask_triangles(nodes, tris, crack)
triang = mtri.Triangulation(nodes[:,0], nodes[:,1], triangles=tris, mask=mask)

sigma1_n, mises_n, tresca_n = nodal_average_stress_fields(nodes, elems, u, model, crack)

def plot_stress_field(field, title, fname, cbar_label):
    fig, ax = plt.subplots(figsize=(10,3))
    levels = 32
    cf = ax.tricontourf(triang, field, levels=levels, cmap="jet")
    fig.colorbar(cf, ax=ax, label=cbar_label)
    if crack.active:
        p0 = crack.p0(); pt = crack.pt()
        for lw, a in [(10,0.12),(6,0.30),(2.5,0.95)]:
            ax.plot([p0[0], pt[0]],[p0[1], pt[1]], linewidth=lw, alpha=a, color="black")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(fname, dpi=220)
    plt.close(fig)

plot_stress_field(sigma1_n/1e6, r"Max principal stress $\sigma_1$ [MPa] (nodal averaged, masked)", "stress_sigma1.png", r"$\sigma_1$ [MPa]")
plot_stress_field(mises_n/1e6, "Von Mises stress [MPa] (nodal averaged, masked)", "stress_von_mises.png", r"$\sigma_{vM}$ [MPa]")
plot_stress_field(tresca_n/1e6, "Tresca stress [MPa] (nodal averaged, masked)", "stress_tresca.png", r"$\sigma_{Tresca}$ [MPa]")

print("[io] wrote stress_sigma1.png, stress_von_mises.png, stress_tresca.png")
