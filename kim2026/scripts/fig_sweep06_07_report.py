#!/usr/bin/env python3
"""Generate figures for Appendix C: Sweep 06 (loss ablation) + 07 (curriculum)."""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "runs" / "figures_sweep_report"
OUT.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def load_fields(path: Path) -> dict[str, np.ndarray]:
    npz = np.load(path)
    return {k: npz[f"{k}_real"] + 1j * npz[f"{k}_imag"]
            for k in ["input", "pred", "target"]}


def load_history(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def compute_metrics(field, target):
    pf, tf = field.ravel(), target.ravel()
    co = abs(np.dot(pf, np.conj(tf))) / max(np.linalg.norm(pf) * np.linalg.norm(tf), 1e-12)
    pi, ti = np.abs(field)**2, np.abs(target)**2
    io = float(np.dot(pi.ravel(), ti.ravel()) / max(np.linalg.norm(pi.ravel()) * np.linalg.norm(ti.ravel()), 1e-12))
    # phase rmse
    inner = np.sum(field * np.conj(target))
    aligned = field * np.exp(-1j * np.angle(inner))
    thr = 0.1 * np.abs(target).max()
    mask = (np.abs(target) > thr) & (np.abs(aligned) > thr)
    pd = np.angle(aligned) - np.angle(target)
    pd = (pd + np.pi) % (2 * np.pi) - np.pi
    pr = float(np.sqrt(np.mean(pd[mask]**2))) if mask.any() else float("nan")
    amp_rmse = float(np.sqrt(np.mean((np.abs(field) - np.abs(target))**2)) / max(np.abs(target).max(), 1e-12))
    return {"co": co, "io": io, "pr": pr, "amp_rmse": amp_rmse}


# ──────────────────────────────────────────────
# Load all data
# ──────────────────────────────────────────────

# Baseline strategies (from earlier sweeps)
baseline_strats = {
    "Complex\n(co+amp)": RUNS / "02_fd2nn_complexloss_roi1024_phase_range_sweep_claude/tanh_2pi",
    "Irradiance\n(io+br+ee)": RUNS / "04_fd2nn_irradianceloss_roi1024_phase_range_sweep_claude/tanh_2pi",
}

# Sweep 06
ablation_strats = {
    "co only\n(no amp)": RUNS / "06_fd2nn_loss_ablation_roi1024_claude/co_only",
    "co+amp:0.1": RUNS / "06_fd2nn_loss_ablation_roi1024_claude/co_amp01",
    "co+io:0.5": RUNS / "06_fd2nn_loss_ablation_roi1024_claude/co_io",
    "co+io+br": RUNS / "06_fd2nn_loss_ablation_roi1024_claude/co_io_br",
}

# Sweep 07
curriculum_strats = {
    "cur 10/20": RUNS / "07_fd2nn_curriculum_roi1024_claude/cur_10_20",
    "cur 15/15": RUNS / "07_fd2nn_curriculum_roi1024_claude/cur_15_15",
    "cur 20/10": RUNS / "07_fd2nn_curriculum_roi1024_claude/cur_20_10",
    "blend": RUNS / "07_fd2nn_curriculum_roi1024_claude/cur_blend",
}

# Compute metrics from sample_fields
all_metrics = {}
target_field = None
turb_field = None

for name, path in {**baseline_strats, **ablation_strats, **curriculum_strats}.items():
    npz_path = path / "sample_fields.npz"
    if not npz_path.exists():
        continue
    data = load_fields(npz_path)
    if target_field is None:
        target_field = data["target"]
        turb_field = data["input"]
    all_metrics[name] = compute_metrics(data["pred"], target_field)

turb_metrics = compute_metrics(turb_field, target_field)
all_metrics["Turbulent\n(no D2NN)"] = turb_metrics

# ──────────────────────────────────────────────
# Figure 11: co↔io Trade-off Scatter
# ──────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 7))

# Group colors
colors_base = {"Complex\n(co+amp)": "#d62728", "Irradiance\n(io+br+ee)": "#2ca02c"}
colors_06 = {"co only\n(no amp)": "#ff7f0e", "co+amp:0.1": "#e377c2", "co+io:0.5": "#17becf", "co+io+br": "#bcbd22"}
colors_07 = {"cur 10/20": "#9467bd", "cur 15/15": "#8c564b", "cur 20/10": "#7f7f7f", "blend": "#1f77b4"}

for name, m in all_metrics.items():
    if name == "Turbulent\n(no D2NN)":
        ax.scatter(m["co"], m["io"], s=200, c="black", marker="*", zorder=10, label=name)
    elif name in colors_base:
        ax.scatter(m["co"], m["io"], s=120, c=colors_base[name], marker="s", zorder=8, label=name, edgecolors="black", linewidths=0.8)
    elif name in colors_06:
        ax.scatter(m["co"], m["io"], s=100, c=colors_06[name], marker="o", zorder=7, label=f"S06: {name}")
    elif name in colors_07:
        ax.scatter(m["co"], m["io"], s=100, c=colors_07[name], marker="^", zorder=7, label=f"S07: {name}")

# Annotate turbulent
ax.annotate("Turbulent\n(no correction)", xy=(turb_metrics["co"], turb_metrics["io"]),
            xytext=(turb_metrics["co"] + 0.02, turb_metrics["io"] - 0.05),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="black"))

# Draw trade-off region
ax.axhline(0.973, color="gray", ls="--", alpha=0.5, label="Turbulent io=0.973")
ax.axvline(0.191, color="gray", ls=":", alpha=0.5, label="Turbulent co=0.191")

# Ideal region
ax.fill_between([0.2, 0.4], 0.9, 1.0, alpha=0.08, color="green")
ax.text(0.30, 0.95, "Ideal region\n(co↑ io↑)", ha="center", va="center", fontsize=10, color="green", alpha=0.7)

ax.set_xlabel("Complex Overlap (co) →  higher = better phase", fontsize=12)
ax.set_ylabel("Intensity Overlap (io) →  higher = better beam", fontsize=12)
ax.set_title("Fig 11. co↔io Trade-off: All 10 Strategies (Sweeps 02-07)", fontsize=13, fontweight="bold")
ax.set_xlim(-0.01, 0.35)
ax.set_ylim(0.3, 1.02)
ax.legend(loc="lower left", fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig11_co_io_tradeoff_all_strategies.png", dpi=200)
plt.close(fig)
print(f"Saved fig11")


# ──────────────────────────────────────────────
# Figure 12: Curriculum Learning Trajectory
# ──────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

cur_names = ["cur_10_20", "cur_15_15", "cur_20_10", "cur_blend"]
cur_labels = ["irr→co (10/20)", "irr→co (15/15)", "irr→co (20/10)", "linear blend"]
cur_colors = ["#9467bd", "#8c564b", "#7f7f7f", "#1f77b4"]
switch_epochs = [10, 15, 20, None]

for idx, (cname, label, col, sw) in enumerate(zip(cur_names, cur_labels, cur_colors, switch_epochs)):
    hist_path = RUNS / f"07_fd2nn_curriculum_roi1024_claude/{cname}/history.json"
    if not hist_path.exists():
        continue
    hist = load_history(hist_path)
    epochs = [h["epoch"] for h in hist]

    cos = [h.get("complex_overlap", h.get("co", None)) for h in hist]
    ios = [h.get("intensity_overlap", h.get("io", None)) for h in hist]

    if cos[0] is None:
        continue

    axes[0].plot(epochs, cos, color=col, label=label, linewidth=1.5)
    axes[1].plot(epochs, ios, color=col, label=label, linewidth=1.5)

    # Mark switch point
    if sw is not None:
        for a in axes[:2]:
            a.axvline(sw, color=col, ls=":", alpha=0.4)

# Add reference lines
for a, metric, turb_val in zip(axes, ["co", "io", "pr"],
                                [turb_metrics["co"], turb_metrics["io"], None]):
    if turb_val is not None:
        a.axhline(turb_val, color="black", ls="--", alpha=0.5, label=f"Turbulent {metric}={turb_val:.3f}")

axes[0].set_ylabel("Complex Overlap (co)")
axes[0].set_title("co trajectory — rises only after switch")
axes[0].set_ylim(-0.02, 0.35)
axes[0].legend(fontsize=7, loc="upper left")
axes[0].grid(True, alpha=0.3)

axes[1].set_ylabel("Intensity Overlap (io)")
axes[1].set_title("io trajectory — collapses on loss switch")
axes[1].set_ylim(0.3, 1.02)
axes[1].legend(fontsize=7, loc="lower left")
axes[1].grid(True, alpha=0.3)

# Panel 3: the key insight — bar chart of final co vs io
strategies_all = ["Turbulent\n(no D2NN)", "Complex\n(co+amp)", "Irradiance\n(io+br+ee)",
                  "co only\n(no amp)", "co+io:0.5", "cur 15/15", "blend"]
co_final = [all_metrics[s]["co"] for s in strategies_all]
io_final = [all_metrics[s]["io"] for s in strategies_all]

x = np.arange(len(strategies_all))
w = 0.35
bars_co = axes[2].bar(x - w/2, co_final, w, label="co", color="#d62728", alpha=0.8)
bars_io = axes[2].bar(x + w/2, io_final, w, label="io", color="#2ca02c", alpha=0.8)
axes[2].set_xticks(x)
axes[2].set_xticklabels([s.replace("\n", " ") for s in strategies_all], rotation=45, ha="right", fontsize=8)
axes[2].set_ylabel("Metric value")
axes[2].set_title("Final co vs io — no strategy achieves both")
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis="y")

for a in axes:
    a.set_xlabel("Epoch")

fig.suptitle("Fig 12. Curriculum Learning Failure — io Collapses on Loss Switch", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig12_curriculum_trajectory_and_tradeoff.png", dpi=200)
plt.close(fig)
print(f"Saved fig12")


# ──────────────────────────────────────────────
# Figure 13: Amplitude RMSE evidence — all strategies
# ──────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: amp RMSE bar chart — all strategies nearly identical
strat_names_short = ["Turb", "Complex", "Irrad", "co only", "co+amp01", "co+io", "co+io+br",
                     "cur10/20", "cur15/15", "cur20/10", "blend"]
strat_keys = ["Turbulent\n(no D2NN)", "Complex\n(co+amp)", "Irradiance\n(io+br+ee)",
              "co only\n(no amp)", "co+amp:0.1", "co+io:0.5", "co+io+br",
              "cur 10/20", "cur 15/15", "cur 20/10", "blend"]

amp_vals = []
io_vals = []
co_vals = []
valid_names = []
for sn, sk in zip(strat_names_short, strat_keys):
    if sk in all_metrics:
        amp_vals.append(all_metrics[sk]["amp_rmse"])
        io_vals.append(all_metrics[sk]["io"])
        co_vals.append(all_metrics[sk]["co"])
        valid_names.append(sn)

x = np.arange(len(valid_names))
bar_colors = ["black"] + ["#d62728", "#2ca02c"] + ["#ff7f0e"]*4 + ["#9467bd"]*4
bar_colors = bar_colors[:len(valid_names)]

axes[0].bar(x, amp_vals, color=bar_colors, alpha=0.8, edgecolor="black", linewidth=0.5)
axes[0].axhline(np.mean(amp_vals[1:]), color="red", ls="--", alpha=0.5, label=f"Mean (D2NN) = {np.mean(amp_vals[1:]):.3f}")
axes[0].set_xticks(x)
axes[0].set_xticklabels(valid_names, rotation=45, ha="right", fontsize=9)
axes[0].set_ylabel("Amplitude RMSE (normalized)")
axes[0].set_title("amp RMSE: 0.159~0.176 across all D2NN strategies\n→ phase-only mask cannot control amplitude")
axes[0].set_ylim(0, 0.22)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3, axis="y")

# Annotate turbulent
axes[0].annotate("~0.02\n(nearly perfect)", xy=(0, amp_vals[0]), xytext=(0.5, 0.05),
                fontsize=9, arrowprops=dict(arrowstyle="->"), ha="center")

# Panel 2: scatter — amp RMSE vs io, showing io varies wildly while amp stays flat
axes[1].scatter(amp_vals[1:], io_vals[1:], s=100, c=bar_colors[1:], edgecolors="black", linewidths=0.8, zorder=5)
for i, name in enumerate(valid_names[1:]):
    axes[1].annotate(name, (amp_vals[i+1], io_vals[i+1]), textcoords="offset points",
                    xytext=(8, 5), fontsize=8)

axes[1].axhline(turb_metrics["io"], color="gray", ls="--", alpha=0.5, label="Turbulent io")
axes[1].set_xlabel("Amplitude RMSE (nearly constant ~0.17)")
axes[1].set_ylabel("Intensity Overlap (io)")
axes[1].set_title("amp RMSE ≈ constant, but io varies 0.37~0.93\n→ difference is energy spatial distribution, not amplitude")
axes[1].set_xlim(0.14, 0.20)
axes[1].set_ylim(0.3, 1.02)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

fig.suptitle("Fig 13. Amplitude RMSE Evidence: Phase-Only Mask Cannot Control Amplitude", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig13_amplitude_rmse_evidence.png", dpi=200)
plt.close(fig)
print(f"Saved fig13")


# ──────────────────────────────────────────────
# Figure 14: Phase difference visualization
# ──────────────────────────────────────────────

# Compare phase error maps for representative strategies
strats_for_phase = {
    "Turbulent (input)": turb_field,
    "Complex (co+amp)": None,
    "Irradiance (io+br+ee)": None,
    "cur 15/15": None,
}

# Load pred fields
for name, path in [
    ("Complex (co+amp)", RUNS / "02_fd2nn_complexloss_roi1024_phase_range_sweep_claude/tanh_2pi"),
    ("Irradiance (io+br+ee)", RUNS / "04_fd2nn_irradianceloss_roi1024_phase_range_sweep_claude/tanh_2pi"),
    ("cur 15/15", RUNS / "07_fd2nn_curriculum_roi1024_claude/cur_15_15"),
]:
    data = load_fields(path / "sample_fields.npz")
    strats_for_phase[name] = data["pred"]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

n = target_field.shape[-1]
center = n // 2
crop = 128  # zoom into center
sl = slice(center - crop, center + crop)

for col, (name, field) in enumerate(strats_for_phase.items()):
    # Align global phase
    inner = np.sum(field * np.conj(target_field))
    aligned = field * np.exp(-1j * np.angle(inner))

    # Phase error
    thr = 0.1 * np.abs(target_field).max()
    mask = (np.abs(target_field) > thr) & (np.abs(aligned) > thr)
    phase_err = np.angle(aligned) - np.angle(target_field)
    phase_err = (phase_err + np.pi) % (2 * np.pi) - np.pi
    phase_err_masked = np.where(mask, phase_err, np.nan)

    # Amplitude error
    amp_err = np.abs(aligned) - np.abs(target_field)
    amp_err_masked = np.where(mask, amp_err, np.nan)

    # Row 0: phase error (zoomed center)
    im0 = axes[0, col].imshow(phase_err_masked[sl, sl], cmap="RdBu_r", vmin=-np.pi, vmax=np.pi,
                               extent=[-crop, crop, -crop, crop])
    axes[0, col].set_title(name, fontsize=10, fontweight="bold")
    if col == 0:
        axes[0, col].set_ylabel("Phase error [rad]", fontsize=11)

    # Row 1: intensity
    pred_int = np.abs(field[sl, sl])**2
    tgt_int = np.abs(target_field[sl, sl])**2
    pred_int_n = pred_int / max(pred_int.max(), 1e-12)
    tgt_int_n = tgt_int / max(tgt_int.max(), 1e-12)

    im1 = axes[1, col].imshow(pred_int_n, cmap="hot", vmin=0, vmax=1,
                               extent=[-crop, crop, -crop, crop])
    axes[1, col].set_title(f"io={compute_metrics(field, target_field)['io']:.3f}", fontsize=10)
    if col == 0:
        axes[1, col].set_ylabel("Intensity (normalized)", fontsize=11)

# Add colorbars
fig.colorbar(im0, ax=axes[0, :], label="Phase error [rad]", shrink=0.8, pad=0.02)
fig.colorbar(im1, ax=axes[1, :], label="Normalized intensity", shrink=0.8, pad=0.02)

fig.suptitle("Fig 14. Phase Error & Intensity: Turbulent vs Complex vs Irradiance vs Curriculum",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig14_phase_intensity_comparison.png", dpi=200)
plt.close(fig)
print(f"Saved fig14")

print(f"\nAll figures saved to {OUT}/")
