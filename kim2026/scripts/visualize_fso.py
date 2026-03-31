"""Visualization script for FSO propagation simulation results."""

import json
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("output/viz_run")
FIG = Path("output/viz_run/figures")
FIG.mkdir(exist_ok=True)

# ── Load config ──────────────────────────────────────────────
with open(OUT / "config.json") as f:
    cfg = json.load(f)
with open(OUT / "sampling_analysis.json") as f:
    samp = json.load(f)
with open(OUT / "screen_r0.json") as f:
    r0_data = json.load(f)

wvl = cfg["wvl"]
k = cfg["k"]
w0 = cfg["w0"]
Dz = cfg["Dz"]
N = samp["N"]
delta1 = samp["delta1"]
deltan = cfg["delta_n"]
n_reals = cfg["n_reals"]

coords = torch.load(OUT / "coordinates.pt", weights_only=True)
xn = coords["xn"].numpy() if isinstance(coords, dict) else coords[0].numpy()

# Observation-plane coordinates [cm]
x_cm = np.arange(-N//2, N//2) * deltan * 100  # cm

# Source-plane coordinates [mm]
x1_mm = np.arange(-N//2, N//2) * delta1 * 1000  # mm

# ── Load fields ──────────────────────────────────────────────
U_vac = torch.load(OUT / "vacuum/field.pt", weights_only=True)
I_vac = torch.load(OUT / "vacuum/irradiance.pt", weights_only=True)

turb_fields = []
turb_irrad = []
for j in range(n_reals):
    I_j = torch.load(OUT / f"turbulence/irradiance_{j:04d}.pt", weights_only=True)
    turb_irrad.append(I_j.numpy())

I_vac_np = I_vac.numpy()
I_stack = np.stack(turb_irrad)  # [n_reals, N, N]
I_mean = I_stack.mean(axis=0)

print(f"Loaded {n_reals} realizations, N={N}")

# ── Colormap ─────────────────────────────────────────────────
CMAP_IR = "inferno"
CMAP_PH = "twilight_shifted"

# ══════════════════════════════════════════════════════════════
# Figure 1: Source Plane Gaussian Beam
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Source field (Gaussian amplitude)
r1_mm = np.arange(-N//2, N//2) * delta1 * 1000
X1, Y1 = np.meshgrid(r1_mm, r1_mm, indexing="ij")
source = np.exp(-(X1**2 + Y1**2) / (w0*1000)**2)

# Zoomed view around beam
zoom = int(6 * w0 / delta1)  # 6*w0 radius in pixels
c = N // 2
sl = slice(c - zoom, c + zoom)

ax = axes[0]
im = ax.imshow(source[sl, sl], extent=[r1_mm[c-zoom], r1_mm[c+zoom-1],
               r1_mm[c-zoom], r1_mm[c+zoom-1]],
               cmap=CMAP_IR, origin="lower")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_title(f"Source: Collimated Gaussian\n$w_0$ = {w0*1e3:.2f} mm")
plt.colorbar(im, ax=ax, label="Amplitude")

# 1D profile
ax = axes[1]
profile = source[c, :]
ax.plot(r1_mm, profile, "k-", lw=1.5)
ax.axvline(-w0*1000, color="r", ls="--", lw=0.8, label=f"$\\pm w_0$ = {w0*1e3:.2f} mm")
ax.axvline(w0*1000, color="r", ls="--", lw=0.8)
ax.set_xlim(-5*w0*1000, 5*w0*1000)
ax.set_xlabel("x [mm]")
ax.set_ylabel("Amplitude")
ax.set_title("Source 1D Profile")
ax.legend()

plt.suptitle(f"Source Plane ($\\lambda$ = {wvl*1e9:.0f} nm, $\\theta_{{div}}$ = {cfg['theta_div']*1e3:.1f} mrad)",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIG / "fig1_source_plane.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 1: Source plane done")

# ══════════════════════════════════════════════════════════════
# Figure 2: Vacuum vs Turbulence Irradiance (2D images)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(18, 9))

# ROI zoom: central 20 cm
roi_cm = 15  # half-width in cm
roi_pix = int(roi_cm / 100 / deltan)
sl = slice(c - roi_pix, c + roi_pix)
ext = [-roi_cm, roi_cm, -roi_cm, roi_cm]

# Vacuum
vmax_vac = I_vac_np.max()
ax = axes[0, 0]
im = ax.imshow(I_vac_np[sl, sl], extent=ext, cmap=CMAP_IR, origin="lower")
ax.set_title("Vacuum")
ax.set_xlabel("x [cm]")
ax.set_ylabel("y [cm]")
plt.colorbar(im, ax=ax, format="%.1e")

# 6 turbulence samples
vmax_turb = np.percentile(I_stack[:, sl, sl], 99.5)
sample_indices = [0, 5, 10, 15, 20, 25]
for idx, real_j in enumerate(sample_indices):
    row, col = divmod(idx + 1, 4)
    ax = axes[row, col]
    im = ax.imshow(turb_irrad[real_j][sl, sl], extent=ext, cmap=CMAP_IR,
                   origin="lower", vmin=0, vmax=vmax_turb)
    ax.set_title(f"Turb #{real_j}")
    ax.set_xlabel("x [cm]")
    if col == 0:
        ax.set_ylabel("y [cm]")
    plt.colorbar(im, ax=ax, format="%.1e")

# Mean irradiance
ax = axes[1, 3]
im = ax.imshow(I_mean[sl, sl], extent=ext, cmap=CMAP_IR, origin="lower")
ax.set_title(f"Mean (n={n_reals})")
ax.set_xlabel("x [cm]")
plt.colorbar(im, ax=ax, format="%.1e")

plt.suptitle(f"Receiver Irradiance — Dz = {Dz/1e3:.0f} km, $C_n^2$ = {cfg['Cn2']:.0e} m$^{{-2/3}}$",
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(FIG / "fig2_irradiance_gallery.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 2: Irradiance gallery done")

# ══════════════════════════════════════════════════════════════
# Figure 3: Irradiance 1D Profiles (vacuum vs turb)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(x_cm, I_vac_np[c, :], "b-", lw=2, label="Vacuum")
ax.plot(x_cm, I_mean[c, :], "r-", lw=2, label=f"Turb mean (n={n_reals})")
for j in [0, 10, 20]:
    ax.plot(x_cm, turb_irrad[j][c, :], "-", lw=0.4, alpha=0.5)
ax.set_xlim(-roi_cm, roi_cm)
ax.set_xlabel("x [cm]")
ax.set_ylabel("Irradiance")
ax.set_title("1D Profile (y=0)")
ax.legend()

# Analytic Gaussian for comparison
z_R = math.pi * w0**2 / wvl
w_z = w0 * math.sqrt(1 + (Dz / z_R)**2)
I_analytic = (w0/w_z)**2 * np.exp(-2 * (x_cm/100)**2 / w_z**2)
ax.plot(x_cm, I_analytic, "g--", lw=1.5, label=f"Analytic Gaussian\n$w(z)$ = {w_z*100:.0f} cm")
ax.legend(fontsize=9)

# Log scale
ax = axes[1]
ax.semilogy(x_cm, I_vac_np[c, :], "b-", lw=2, label="Vacuum")
ax.semilogy(x_cm, I_mean[c, :], "r-", lw=2, label=f"Turb mean")
ax.semilogy(x_cm, I_analytic, "g--", lw=1.5, label="Analytic")
ax.set_xlim(-25, 25)
ax.set_ylim(1e-9, I_vac_np.max() * 2)
ax.set_xlabel("x [cm]")
ax.set_ylabel("Irradiance (log)")
ax.set_title("1D Profile — Log Scale")
ax.legend()

plt.tight_layout()
plt.savefig(FIG / "fig3_irradiance_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 3: Irradiance profiles done")

# ══════════════════════════════════════════════════════════════
# Figure 4: Structure Function vs Theory
# ══════════════════════════════════════════════════════════════
with open(OUT / "verification/structure_function_report.json") as f:
    sf_report = json.load(f)

# Load a few screens and compute brute-force D for visualization
from kim2026.fso.ft_utils import str_fcn2_bruteforce
from kim2026.fso.phase_screen import ft_sh_phase_screen

r0_vals = r0_data if isinstance(r0_data, list) else r0_data.get("r0_values", r0_data.get("r0", []))
delta_vals = samp["delta_values"]

# Pick a middle plane (plane 3) for visualization
plane_idx = 3
r0_plane = r0_vals[plane_idx]
delta_plane = delta_vals[plane_idx]

D_accum = None
n_vis = 30
for i in range(n_vis):
    phz = ft_sh_phase_screen(r0_plane, N, delta_plane, device="cuda")
    r_t, D_t = str_fcn2_bruteforce(phz, delta_plane)
    if D_accum is None:
        D_accum = D_t.clone()
    else:
        D_accum += D_t
D_avg = (D_accum / n_vis).cpu().numpy()
r_meas = r_t.cpu().numpy()
D_theory = 6.88 * (r_meas / r0_plane) ** (5/3)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.loglog(r_meas * 100, D_avg, "b-", lw=2, label="Measured (30-screen avg)")
ax.loglog(r_meas * 100, D_theory, "r--", lw=2, label=f"Theory: 6.88$(r/r_0)^{{5/3}}$")
ax.axvline(r0_plane * 100, color="gray", ls=":", lw=1, label=f"$r_0$ = {r0_plane*100:.1f} cm")
ax.set_xlabel("Lag r [cm]")
ax.set_ylabel("$D_\\phi(r)$ [rad²]")
ax.set_title(f"Phase Structure Function — Plane {plane_idx}\n$r_0$ = {r0_plane*100:.1f} cm, $\\delta$ = {delta_plane*1e3:.2f} mm")
ax.legend()
ax.grid(True, alpha=0.3)

# Relative error per plane
ax = axes[1]
errors = [p["rel_error"] for p in sf_report["planes"]]
passes = [p["pass"] for p in sf_report["planes"]]
colors = ["green" if p else "red" for p in passes]
bars = ax.bar(range(len(errors)), [e*100 for e in errors], color=colors)
ax.axhline(20, color="orange", ls="--", lw=1.5, label="20% threshold")
ax.set_xlabel("Plane index")
ax.set_ylabel("Avg relative error [%]")
ax.set_title("Structure Function Error per Plane")
ax.legend()

# Annotate r0 values
for i, (e, r0) in enumerate(zip(errors, r0_vals)):
    label = f"r₀={r0:.2f}" if r0 < 10 else "skip"
    ax.text(i, e*100 + 1, label, ha="center", fontsize=7, rotation=45)

plt.tight_layout()
plt.savefig(FIG / "fig4_structure_function.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 4: Structure function done")

# ══════════════════════════════════════════════════════════════
# Figure 5: Coherence Factor vs Theory
# ══════════════════════════════════════════════════════════════
with open(OUT / "verification/coherence_factor_report.json") as f:
    cf_report = json.load(f)

# Recompute coherence for plotting
from kim2026.fso.ft_utils import corr2_ft
from kim2026.fso.verification import make_circular_mask, radial_average

mask = make_circular_mask(N, deltan, cfg["D_roi"], device="cuda")
U_vac_gpu = U_vac.to("cuda").to(torch.complex128)
amp = U_vac_gpu.abs().clamp(min=1e-30)
Uvac_conj = (U_vac_gpu / amp).conj()

Gamma_accum = None
for j in range(n_reals):
    U_j = torch.load(OUT / f"turbulence/field_{j:04d}.pt", weights_only=True)
    U_j = U_j.to("cuda").to(torch.complex128) * Uvac_conj
    Gamma_j = corr2_ft(U_j, U_j, mask, deltan)
    if Gamma_accum is None:
        Gamma_accum = Gamma_j.clone()
    else:
        Gamma_accum += Gamma_j
Gamma_avg = Gamma_accum / n_reals
mu_2d = torch.abs(Gamma_avg) / torch.abs(Gamma_avg[c, c]).clamp(min=1e-30)
r_mu, mu_meas = radial_average(mu_2d, deltan)

# Atmospheric params
from kim2026.fso.atmosphere import compute_atmospheric_params
atm = compute_atmospheric_params(k, cfg["Cn2"], Dz)
r0_sw = atm["r0_sw"]
mu_theory = np.exp(-3.44 * (r_mu / r0_sw) ** (5/3))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(r_mu * 100, mu_meas, "b-", lw=2, label="Measured")
ax.plot(r_mu * 100, mu_theory, "r--", lw=2,
        label=f"Theory: exp$(-3.44(r/r_{{0,sw}})^{{5/3}})$\n$r_{{0,sw}}$ = {r0_sw*100:.1f} cm")
ax.axhline(1/np.e, color="gray", ls=":", lw=1, label="$e^{-1}$")

e_inv_w = cf_report["e_inv_width"] * 100
rho_0 = cf_report["rho_0_theory"] * 100
ax.axvline(e_inv_w, color="blue", ls=":", lw=1, alpha=0.6)
ax.axvline(rho_0, color="red", ls=":", lw=1, alpha=0.6)
ax.annotate(f"Meas: {e_inv_w:.1f} cm", (e_inv_w, 1/np.e), fontsize=9,
            xytext=(e_inv_w+2, 1/np.e+0.05), color="blue")
ax.annotate(f"Theory: {rho_0:.1f} cm", (rho_0, 1/np.e), fontsize=9,
            xytext=(rho_0+2, 1/np.e-0.08), color="red")

ax.set_xlim(0, min(30, r_mu[-1]*100))
ax.set_ylim(0, 1.05)
ax.set_xlabel("Separation $|\\Delta r|$ [cm]")
ax.set_ylabel("$|\\mu(\\Delta r)|$")
ax.set_title(f"Coherence Factor — {cf_report['agreement_level']}\n"
             f"Dz = {Dz/1e3:.0f} km, $C_n^2$ = {cfg['Cn2']:.0e}")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG / "fig5_coherence_factor.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 5: Coherence factor done")

# ══════════════════════════════════════════════════════════════
# Figure 6: Scintillation statistics
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 6a: Histogram of on-axis irradiance
on_axis = I_stack[:, c, c]
ax = axes[0]
ax.hist(on_axis, bins=15, density=True, color="steelblue", edgecolor="k", alpha=0.8)
ax.axvline(I_vac_np[c, c], color="r", ls="--", lw=2, label=f"Vacuum: {I_vac_np[c,c]:.2e}")
ax.axvline(on_axis.mean(), color="orange", ls="-", lw=2, label=f"Turb mean: {on_axis.mean():.2e}")
ax.set_xlabel("Irradiance at center")
ax.set_ylabel("Probability density")
ax.set_title(f"On-axis Irradiance Distribution\n(n={n_reals})")
ax.legend(fontsize=8)

# 6b: Scintillation index map
sigma_I_sq = I_stack.var(axis=0) / (I_mean**2 + 1e-30)
ax = axes[1]
im = ax.imshow(sigma_I_sq[sl, sl], extent=ext, cmap="hot", origin="lower",
               vmin=0, vmax=np.percentile(sigma_I_sq[sl, sl], 95))
ax.set_xlabel("x [cm]")
ax.set_ylabel("y [cm]")
ax.set_title(f"Scintillation Index $\\sigma_I^2$\n$\\sigma_{{\\chi}}^2$ = {atm['sigma2_chi_sw']:.4f}")
plt.colorbar(im, ax=ax)

# 6c: Radial scintillation profile
r_pix = np.sqrt((np.arange(N) - c)**2)
max_r = int(roi_pix)
r_bins = np.arange(0, max_r)
si_radial = np.zeros(max_r)
for b in r_bins:
    ring = (np.abs(r_pix - b) < 0.5)
    if ring.sum() > 0:
        ring_2d = np.outer(ring, np.ones(N)) * np.outer(np.ones(N), ring)
        # Just use 1D for simplicity
        si_radial[b] = sigma_I_sq[c, c-b:c+b+1].mean() if b > 0 else sigma_I_sq[c, c]

ax = axes[2]
r_si_cm = r_bins * deltan * 100
ax.plot(r_si_cm, si_radial, "b-", lw=2)
ax.set_xlabel("Radius [cm]")
ax.set_ylabel("$\\sigma_I^2$")
ax.set_title("Radial Scintillation Profile")
ax.set_xlim(0, roi_cm)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG / "fig6_scintillation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 6: Scintillation done")

# ══════════════════════════════════════════════════════════════
# Figure 7: Sampling & propagation geometry
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

z_km = np.array(samp["z_planes"]) / 1000
deltas_mm = np.array(samp["delta_values"]) * 1000

ax = axes[0]
ax.plot(z_km, deltas_mm, "bo-", lw=2, ms=8)
ax.set_xlabel("z [km]")
ax.set_ylabel("Grid spacing $\\delta$ [mm]")
ax.set_title(f"Grid Spacing Evolution\n$\\delta_1$ = {delta1*1e3:.3f} mm → $\\delta_n$ = {deltan*1e3:.1f} mm")
ax.grid(True, alpha=0.3)

# Physical grid size at each plane
grid_size_m = N * np.array(samp["delta_values"])
ax2 = ax.twinx()
ax2.plot(z_km, grid_size_m, "rs--", lw=1.5, ms=6)
ax2.set_ylabel("Grid extent [m]", color="red")
ax2.tick_params(axis="y", labelcolor="red")

# r0 per screen
ax = axes[1]
r0_plot = np.array(r0_vals)
r0_plot_cm = r0_plot * 100
r0_plot_cm[r0_plot_cm > 500] = np.nan  # skip 50m cap for plotting
ax.bar(range(len(r0_vals)), r0_plot_cm, color="teal", edgecolor="k")
ax.set_xlabel("Screen index")
ax.set_ylabel("$r_0$ [cm]")
ax.set_title(f"Per-Screen Fried Parameter\n$r_{{0,sw}}$ = {r0_sw*100:.1f} cm (bulk)")
ax.axhline(r0_sw*100, color="red", ls="--", lw=1.5, label=f"$r_{{0,sw}}$ = {r0_sw*100:.1f} cm")
ax.legend()

for i, v in enumerate(r0_vals):
    if v < 10:
        ax.text(i, v*100 + 1, f"{v*100:.1f}", ha="center", fontsize=8)
    else:
        ax.text(i, 5, "cap", ha="center", fontsize=8, color="gray")

plt.tight_layout()
plt.savefig(FIG / "fig7_geometry.png", dpi=150, bbox_inches="tight")
plt.close()
print("Fig 7: Geometry done")

print(f"\nAll figures saved to {FIG}/")
