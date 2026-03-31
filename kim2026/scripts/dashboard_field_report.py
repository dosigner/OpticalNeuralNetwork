#!/usr/bin/env python
"""FD2NN Sweep Field Report Dashboard.

Generates 4 figures for team-lead-level analysis:
  1. Field intensity comparison (5 runs × 4 cols)
  2. Field phase comparison [0, 2π] (5 runs × 4 cols)
  3. Team lead summary (radial profiles + metrics table + phase error)
  4. Training convergence comparison

Usage:
    cd /root/dj/D2NN/kim2026
    python scripts/dashboard_field_report.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────

PROJ = Path(__file__).resolve().parent.parent
RUNS = PROJ / "runs"
FIG_DIR = PROJ / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Run definitions ──────────────────────────────────────────────────────────

BEST_RUNS = {
    "run01": {
        "dir": RUNS / "01_fd2nn_complexloss_roi1024_spacing_sweep_claude",
        "subdir": "spacing_1mm",
        "label": "01 Complex Loss\nspacing=1mm, tanh π",
        "short": "01 Complex (spacing)",
        "color": "#2196F3",
    },
    "run02": {
        "dir": RUNS / "02_fd2nn_complexloss_roi1024_phase_range_sweep_claude",
        "subdir": "tanh_2pi",
        "label": "02 Complex Loss\ntanh 2π",
        "short": "02 Complex (2π)",
        "color": "#4CAF50",
    },
    "run03": {
        "dir": RUNS / "03_fd2nn_phasorloss_roi1024_phase_range_sweep_claude",
        "subdir": "sig_2pi",
        "label": "03 Phasor Loss\nsig 2π",
        "short": "03 Phasor",
        "color": "#FF9800",
    },
    "run04": {
        "dir": RUNS / "04_fd2nn_irradianceloss_roi1024_phase_range_sweep_claude",
        "subdir": "tanh_2pi",
        "label": "04 Irradiance Loss\ntanh 2π",
        "short": "04 Irradiance",
        "color": "#F44336",
    },
    "run05": {
        "dir": RUNS / "05_fd2nn_hybridloss_roi1024_loss_combo_sweep_claude",
        "subdir": "combo4_sp_leak_io",
        "label": "05 Hybrid Loss\nSP+Leak+IO, tanh 2π",
        "short": "05 Hybrid",
        "color": "#9C27B0",
    },
}

# ── Style ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.size": 10,
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "font.family": "DejaVu Sans",
})


# ── Data loading ─────────────────────────────────────────────────────────────

def load_fields(cfg: dict) -> dict | None:
    path = cfg["dir"] / cfg["subdir"] / "sample_fields.npz"
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return None
    npz = np.load(path)
    return {
        "input":  npz["input_real"]  + 1j * npz["input_imag"],
        "pred":   npz["pred_real"]   + 1j * npz["pred_imag"],
        "target": npz["target_real"] + 1j * npz["target_imag"],
    }


def load_metrics(cfg: dict) -> dict:
    path = cfg["dir"] / cfg["subdir"] / "test_metrics.json"
    with open(path) as f:
        return json.load(f)


def load_history(cfg: dict) -> list:
    path = cfg["dir"] / cfg["subdir"] / "history.json"
    with open(path) as f:
        return json.load(f)


# ── Image helpers ────────────────────────────────────────────────────────────

def center_crop(arr, radius):
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    return arr[cy - radius:cy + radius, cx - radius:cx + radius]


def norm_irr(field, ref):
    return np.abs(field) ** 2 / max(ref, 1e-12)


def phase_0_2pi(field, thr):
    I = np.abs(field) ** 2
    ph = np.angle(field) % (2 * np.pi)
    return np.ma.masked_where(I < thr, ph)


def phase_err(field, target, thr):
    err = np.angle(field * np.conj(target))
    mask = (np.abs(field)**2 < thr) | (np.abs(target)**2 < thr)
    return np.ma.masked_where(mask, err)


def radial_profile(field, ref, max_r=None):
    irr = norm_irr(field, ref)
    cy, cx = irr.shape[0] // 2, irr.shape[1] // 2
    yy, xx = np.mgrid[:irr.shape[0], :irr.shape[1]]
    rr = np.sqrt((xx - cx)**2 + (yy - cy)**2).astype(int)
    if max_r is None:
        max_r = min(rr.max() + 1, irr.shape[0] // 2)
    out = np.zeros(max_r)
    for r in range(max_r):
        m = rr == r
        if m.any():
            out[r] = irr[m].mean()
    return out


# ── Figure 1: Intensity ─────────────────────────────────────────────────────

def fig_intensity(data, crop=200):
    keys = list(data.keys())
    n = len(keys)
    ref = max(float((np.abs(data[k]["target"])**2).max()) for k in keys)

    fig, axes = plt.subplots(n, 4, figsize=(14, 3.0 * n))
    titles = ["Input (turbulent)", "Target (vacuum)", "D2NN Prediction",
              "|Pred − Target| Error"]

    for r, k in enumerate(keys):
        inp = center_crop(data[k]["input"], crop)
        tgt = center_crop(data[k]["target"], crop)
        prd = center_crop(data[k]["pred"], crop)

        imgs = [norm_irr(inp, ref), norm_irr(tgt, ref),
                norm_irr(prd, ref), np.abs(norm_irr(prd, ref) - norm_irr(tgt, ref))]
        vmaxes = [1.0, 1.0, 1.0, 0.3]
        cmaps = ["inferno", "inferno", "inferno", "magma"]

        for c in range(4):
            ax = axes[r, c]
            im = ax.imshow(imgs[c], cmap=cmaps[c], vmin=0, vmax=vmaxes[c],
                           origin="lower", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=10, fontweight="bold")
            if c == 0:
                ax.set_ylabel(BEST_RUNS[k]["label"], fontsize=8, fontweight="bold")
            # Per-panel colorbar for last row
            if r == n - 1:
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        "FD2NN Field Intensity Comparison (center 400×400 px)\n"
        r"$\lambda=1.55\,\mu$m  |  5 layers  |  1024×1024  |  30 epochs",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = FIG_DIR / "report_field_intensity.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


# ── Figure 2: Phase [0, 2π] ─────────────────────────────────────────────────

def fig_phase(data, crop=200):
    keys = list(data.keys())
    n = len(keys)
    ref = max(float((np.abs(data[k]["target"])**2).max()) for k in keys)
    thr = ref * 1e-3

    fig, axes = plt.subplots(n, 4, figsize=(14, 3.0 * n))
    titles = ["Input Phase", "Target Phase", "Prediction Phase",
              "Phase Error (Pred−Target)"]

    for r, k in enumerate(keys):
        inp = center_crop(data[k]["input"], crop)
        tgt = center_crop(data[k]["target"], crop)
        prd = center_crop(data[k]["pred"], crop)

        panels = [
            (phase_0_2pi(inp, thr), "twilight", 0, 2*np.pi),
            (phase_0_2pi(tgt, thr), "twilight", 0, 2*np.pi),
            (phase_0_2pi(prd, thr), "twilight", 0, 2*np.pi),
            (phase_err(prd, tgt, thr), "RdBu_r", -np.pi, np.pi),
        ]

        for c, (img, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
                           origin="lower", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=10, fontweight="bold")
            if c == 0:
                ax.set_ylabel(BEST_RUNS[k]["label"], fontsize=8, fontweight="bold")

            if r == n - 1:
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.ax.tick_params(labelsize=7)
                if c < 3:
                    cb.set_ticks([0, np.pi, 2*np.pi])
                    cb.set_ticklabels(["0", "π", "2π"])
                else:
                    cb.set_ticks([-np.pi, 0, np.pi])
                    cb.set_ticklabels(["-π", "0", "π"])

    fig.suptitle(
        "FD2NN Field Phase Comparison [0, 2π] (center 400×400 px)\n"
        r"$\lambda=1.55\,\mu$m  |  5 layers  |  masked below 0.1% peak",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = FIG_DIR / "report_field_phase.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


# ── Figure 3: Team Lead Summary ─────────────────────────────────────────────

def fig_summary(data, metrics):
    keys = list(data.keys())
    ref = max(float((np.abs(data[k]["target"])**2).max()) for k in keys)
    thr = ref * 1e-3
    n_field = data[keys[0]]["target"].shape[0]
    dx_um = 2048.0 / n_field  # µm/pixel

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

    # ── (a) Radial profiles linear ──
    ax_a = fig.add_subplot(gs[0, 0])
    first = data[keys[0]]
    rad_inp = radial_profile(first["input"], ref, 300)
    rad_tgt = radial_profile(first["target"], ref, 300)
    r_um = np.arange(len(rad_tgt)) * dx_um

    ax_a.plot(r_um, rad_inp, "k--", lw=1.5, alpha=0.5, label="Input (turbulent)")
    ax_a.plot(r_um, rad_tgt, "k-",  lw=2.0, label="Target (vacuum)")
    for k in keys:
        rad = radial_profile(data[k]["pred"], ref, 300)
        ax_a.plot(r_um, rad, color=BEST_RUNS[k]["color"], lw=1.3, alpha=0.9,
                  label=BEST_RUNS[k]["short"])
    ax_a.set_xlabel("Radius (µm)")
    ax_a.set_ylabel("I / I_max")
    ax_a.set_title("(a) Radial Intensity Profile", fontweight="bold")
    ax_a.set_xlim(0, 300)
    ax_a.legend(fontsize=7, loc="upper right")
    ax_a.grid(alpha=0.3)

    # ── (b) Radial profiles log ──
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.semilogy(r_um, np.clip(rad_inp, 1e-7, None), "k--", lw=1.5, alpha=0.5,
                  label="Input (turbulent)")
    ax_b.semilogy(r_um, np.clip(rad_tgt, 1e-7, None), "k-", lw=2.0,
                  label="Target (vacuum)")
    for k in keys:
        rad = radial_profile(data[k]["pred"], ref, 300)
        ax_b.semilogy(r_um, np.clip(rad, 1e-7, None),
                      color=BEST_RUNS[k]["color"], lw=1.3, alpha=0.9,
                      label=BEST_RUNS[k]["short"])
    ax_b.set_xlabel("Radius (µm)")
    ax_b.set_ylabel("I / I_max (log)")
    ax_b.set_title("(b) Radial Profile — Log Scale", fontweight="bold")
    ax_b.set_xlim(0, 500)
    ax_b.set_ylim(1e-6, 2)
    ax_b.legend(fontsize=7, loc="upper right")
    ax_b.grid(alpha=0.3)

    # ── (c) Metrics table ──
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.axis("off")
    ax_c.set_title("(c) Performance Metrics — Best Config per Run",
                    fontweight="bold", pad=15)

    col_labels = ["Run", "Loss", "CO ↑", "IO ↑", "φ RMSE ↓\n(rad)",
                  "Amp RMSE ↓", "Strehl"]
    rows = []
    row_colors = []
    for k in keys:
        m = metrics[k]
        co = m.get("complex_overlap", 0)
        io = m.get("intensity_overlap", 0)
        pr = m.get("phase_rmse_rad", 0)
        ar = m.get("amplitude_rmse", 0)
        sr = m.get("strehl", None)
        sr_str = f"{sr:.2f}" if sr is not None else "—"
        loss = BEST_RUNS[k]["short"].split("(")[0].strip().replace(k[:4] + " ", "")
        rows.append([
            k.replace("run0", "Run 0"),
            loss,
            f"{co:.4f}",
            f"{io:.4f}",
            f"{pr:.3f}",
            f"{ar:.4f}",
            sr_str,
        ])
        row_colors.append([BEST_RUNS[k]["color"] + "18"] * 7)

    tbl = ax_c.table(
        cellText=rows, colLabels=col_labels,
        cellColours=row_colors, colColours=["#e0e0e0"] * 7,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.8)

    # Highlight best CO and best IO
    best_co = max(range(len(keys)), key=lambda i: metrics[keys[i]].get("complex_overlap", 0))
    best_io = max(range(len(keys)), key=lambda i: metrics[keys[i]].get("intensity_overlap", 0))
    tbl[best_co + 1, 2].set_text_props(fontweight="bold", color="#1b7837")
    tbl[best_io + 1, 3].set_text_props(fontweight="bold", color="#1b7837")

    # Baseline row
    bl_co = metrics[keys[0]].get("baseline_complex_overlap",
             metrics[keys[0]].get("baseline_co", 0))
    bl_io = metrics[keys[0]].get("baseline_intensity_overlap",
             metrics[keys[0]].get("baseline_io", 0))
    ax_c.text(0.5, 0.02,
              f"Baseline (no D2NN):  CO={bl_co:.4f}  |  IO={bl_io:.4f}",
              ha="center", fontsize=9, style="italic",
              transform=ax_c.transAxes, color="#666")

    # ── (d) Centerline phase error ──
    ax_d = fig.add_subplot(gs[1, 1])
    target = data[keys[0]]["target"]
    n = target.shape[0]
    x_um = (np.arange(n) - n // 2) * dx_um

    for k in keys:
        pe = phase_err(data[k]["pred"], target, thr)
        cl = pe[n // 2].filled(np.nan)
        ax_d.plot(x_um, cl, color=BEST_RUNS[k]["color"], lw=0.9, alpha=0.8,
                  label=BEST_RUNS[k]["short"])

    ax_d.set_xlabel("Position (µm)")
    ax_d.set_ylabel("Phase error (rad)")
    ax_d.set_title("(d) Centerline Phase Error vs Vacuum", fontweight="bold")
    ax_d.set_xlim(-300, 300)
    ax_d.set_ylim(-np.pi, np.pi)
    ax_d.axhline(0, color="gray", ls="--", lw=0.5)
    ax_d.legend(fontsize=7, loc="upper right")
    ax_d.grid(alpha=0.3)

    fig.suptitle(
        "FD2NN Beam Cleanup — Team Lead Summary\n"
        r"$\lambda=1.55\,\mu$m  |  5 layers  |  Cn²: strong turbulence  |  30 epochs",
        fontsize=13, fontweight="bold",
    )

    out = FIG_DIR / "report_field_summary.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


# ── Figure 4: Convergence ───────────────────────────────────────────────────

def fig_convergence(histories):
    keys = list(histories.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    metrics_keys = [
        ("train_loss",       "Training Loss",           axes[0, 0]),
        ("complex_overlap",  "Complex Overlap (val)",   axes[0, 1]),
        ("phase_rmse_rad",   "Phase RMSE [rad] (val)",  axes[1, 0]),
        ("intensity_overlap","Intensity Overlap (val)",  axes[1, 1]),
    ]

    for k in keys:
        hist = histories[k]
        color = BEST_RUNS[k]["color"]
        label = BEST_RUNS[k]["short"]
        for mkey, title, ax in metrics_keys:
            epochs = [e["epoch"] for e in hist if mkey in e]
            vals   = [e[mkey]   for e in hist if mkey in e]
            if epochs:
                ax.plot(epochs, vals, "o-", color=color, lw=1.4, ms=4,
                        label=label, alpha=0.85)

    for mkey, title, ax in metrics_keys:
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "FD2NN Training Convergence Comparison\n"
        "Best config per loss strategy",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    out = FIG_DIR / "report_convergence.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("FD2NN Field Report Dashboard")
    print(f"Runs dir: {RUNS}")
    print(f"Output:   {FIG_DIR}\n")

    # Load all data
    data = {}
    metrics = {}
    histories = {}
    for k, cfg in BEST_RUNS.items():
        print(f"Loading {k}: {cfg['subdir']} ...")
        fields = load_fields(cfg)
        if fields is None:
            continue
        data[k] = fields
        metrics[k] = load_metrics(cfg)
        histories[k] = load_history(cfg)

    print(f"\nLoaded {len(data)} runs. Generating figures...\n")

    fig_intensity(data)
    fig_phase(data)
    fig_summary(data, metrics)
    fig_convergence(histories)

    print("\nAll done!")


if __name__ == "__main__":
    main()
