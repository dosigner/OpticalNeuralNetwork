#!/usr/bin/env python
"""Generate comprehensive dashboard for FD2NN sweep results (Runs 01-05).

Produces a 6-panel figure summarizing:
  (a) Layer spacing sweep
  (b) Phase range × loss type — complex overlap
  (c) Phase range × loss type — phase RMSE
  (d) Hybrid loss combo metrics
  (e) Training convergence curves
  (f) CO vs IO trade-off scatter

Usage:
    python scripts/dashboard_sweep_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
import torch

# ── Paths ────────────────────────────────────────────────────────────────────
PROJ = Path(__file__).resolve().parent.parent
RUNS = PROJ / "runs"
FIG_DIR = PROJ / "figures"

RUN_DIRS = {
    "run01": RUNS / "01_fd2nn_complexloss_roi1024_spacing_sweep_claude",
    "run02": RUNS / "02_fd2nn_complexloss_roi1024_phase_range_sweep_claude",
    "run03": RUNS / "03_fd2nn_phasorloss_roi1024_phase_range_sweep_claude",
    "run04": RUNS / "04_fd2nn_irradianceloss_roi1024_phase_range_sweep_claude",
    "run05": RUNS / "05_fd2nn_hybridloss_roi1024_loss_combo_sweep_claude",
}

# ── Constants ────────────────────────────────────────────────────────────────
BASELINE_CO = 0.19134825468063354
BASELINE_IO = 0.9725200533866882

SPACING_SUBDIRS = [
    "spacing_0mm", "spacing_0p1mm", "spacing_1mm",
    "spacing_2mm", "spacing_5mm", "spacing_10mm",
]
SPACING_LABELS = ["0", "0.1", "1", "2", "5", "10"]
SPACING_MM = [0.0, 0.1, 1.0, 2.0, 5.0, 10.0]

PHASE_SUBDIRS = ["tanh_pi2", "tanh_pi", "tanh_2pi", "sig_pi", "sig_2pi", "sig_4pi"]
PHASE_LABELS = [
    r"tanh $\pi/2$",
    r"tanh $\pi$",
    r"tanh $2\pi$",
    r"sig $\pi$",
    r"sig $2\pi$",
    r"sig $4\pi$",
]

COMBO_SUBDIRS = ["combo1_io_co", "combo2_io_br_ee", "combo3_co_io_br", "combo4_sp_leak_io"]
COMBO_LABELS = [
    "IO + CO",
    "IO + BR + EE",
    "CO + IO + BR",
    "SP + Leak + IO",
]

# ── Colors (colorblind-safe) ─────────────────────────────────────────────────
C_COMPLEX = "#2166ac"
C_PHASOR = "#4dac26"
C_IRRADIANCE = "#d6604d"

RUN_COLORS = {
    "run01": "#2166ac",
    "run02": "#1b7837",
    "run03": "#762a83",
    "run04": "#e08214",
    "run05": "#c51b7d",
}
RUN_LABELS = {
    "run01": "Spacing sweep (complex)",
    "run02": "Phase range (complex)",
    "run03": "Phase range (phasor)",
    "run04": "Phase range (irradiance)",
    "run05": "Hybrid loss combos",
}

COMBO_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]


# ── Style ────────────────────────────────────────────────────────────────────
def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


# ── Data Loading ─────────────────────────────────────────────────────────────
def load_test_metrics(run_dir: Path, subdirs: list[str]) -> dict[str, dict]:
    """Load test_metrics.json from each subdir, skip missing."""
    results = {}
    for sd in subdirs:
        p = run_dir / sd / "test_metrics.json"
        if p.exists():
            with open(p) as f:
                results[sd] = json.load(f)
    return results


def load_history(run_dir: Path, subdir: str) -> list[dict]:
    """Load history.json from a run subdir."""
    p = run_dir / subdir / "history.json"
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


# ── Panel Functions ──────────────────────────────────────────────────────────

def panel_spacing_sweep(ax: plt.Axes, data: dict[str, dict]):
    """(a) Run 01 — layer spacing sweep: CO bars + phase RMSE line."""
    x = np.arange(len(SPACING_SUBDIRS))
    co_vals = [data[s]["complex_overlap"] for s in SPACING_SUBDIRS]
    rmse_vals = [data[s]["phase_rmse_rad"] for s in SPACING_SUBDIRS]

    bars = ax.bar(x, co_vals, width=0.55, color=C_COMPLEX, alpha=0.85, zorder=2,
                  label="Complex overlap")
    ax.axhline(BASELINE_CO, color="gray", ls="--", lw=0.8, zorder=1)
    ax.text(len(x) - 0.5, BASELINE_CO + 0.003, "no D2NN", fontsize=7,
            color="gray", ha="right", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(SPACING_LABELS)
    ax.set_xlabel("Layer spacing (mm)")
    ax.set_ylabel("Complex overlap", color=C_COMPLEX)
    ax.tick_params(axis="y", labelcolor=C_COMPLEX)
    ax.set_ylim(0.10, 0.25)
    ax.set_title("(a) Layer Spacing Sweep")

    ax2 = ax.twinx()
    ax2.plot(x, rmse_vals, "o-", color="#b2182b", ms=5, lw=1.5, zorder=3,
             label="Phase RMSE")
    ax2.set_ylabel("Phase RMSE (rad)", color="#b2182b")
    ax2.tick_params(axis="y", labelcolor="#b2182b")
    ax2.set_ylim(1.40, 1.75)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)


def panel_co_comparison(ax: plt.Axes, d02: dict, d03: dict, d04: dict):
    """(b) Runs 02/03/04 — complex overlap grouped bars by phase config."""
    x = np.arange(len(PHASE_SUBDIRS))
    w = 0.25

    co_complex = [d02[s]["complex_overlap"] for s in PHASE_SUBDIRS]
    co_phasor = [d03[s]["complex_overlap"] for s in PHASE_SUBDIRS]
    co_irrad = [d04[s]["complex_overlap"] for s in PHASE_SUBDIRS]

    ax.bar(x - w, co_complex, w, color=C_COMPLEX, alpha=0.85, label="Complex loss")
    ax.bar(x, co_phasor, w, color=C_PHASOR, alpha=0.85, label="Phasor loss")
    ax.bar(x + w, co_irrad, w, color=C_IRRADIANCE, alpha=0.85, label="Irradiance loss")

    ax.axhline(BASELINE_CO, color="gray", ls="--", lw=0.8, zorder=1)
    ax.text(len(x) - 0.5, BASELINE_CO + 0.003, "baseline", fontsize=7,
            color="gray", ha="right", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(PHASE_LABELS)
    ax.set_xlabel("Phase constraint range")
    ax.set_ylabel("Complex overlap")
    ax.set_title("(b) Phase Range — Complex Overlap")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_ylim(0.0, 0.32)


def panel_phase_rmse(ax: plt.Axes, d02: dict, d03: dict, d04: dict):
    """(c) Runs 02/03/04 — phase RMSE grouped bars."""
    x = np.arange(len(PHASE_SUBDIRS))
    w = 0.25

    rmse_complex = [d02[s]["phase_rmse_rad"] for s in PHASE_SUBDIRS]
    rmse_phasor = [d03[s]["phase_rmse_rad"] for s in PHASE_SUBDIRS]
    rmse_irrad = [d04[s]["phase_rmse_rad"] for s in PHASE_SUBDIRS]

    ax.bar(x - w, rmse_complex, w, color=C_COMPLEX, alpha=0.85, label="Complex loss")
    ax.bar(x, rmse_phasor, w, color=C_PHASOR, alpha=0.85, label="Phasor loss")
    ax.bar(x + w, rmse_irrad, w, color=C_IRRADIANCE, alpha=0.85, label="Irradiance loss")

    ax.set_xticks(x)
    ax.set_xticklabels(PHASE_LABELS)
    ax.set_xlabel("Phase constraint range")
    ax.set_ylabel("Phase RMSE (rad)")
    ax.set_title("(c) Phase Range — Phase RMSE")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_ylim(0.0, 2.0)

    # Annotate best
    best_val = min(rmse_complex)
    best_idx = rmse_complex.index(best_val)
    ax.annotate(f"{best_val:.2f} rad", xy=(best_idx - w, best_val),
                xytext=(best_idx - w + 0.3, best_val + 0.25),
                fontsize=7, color=C_COMPLEX,
                arrowprops=dict(arrowstyle="->", color=C_COMPLEX, lw=0.8))


def panel_hybrid_metrics(ax: plt.Axes, data: dict[str, dict]):
    """(d) Run 05 — hybrid loss combo normalized metrics."""
    metrics_raw = {}
    for sd in COMBO_SUBDIRS:
        d = data[sd]
        metrics_raw[sd] = {
            "CO": d["complex_overlap"],
            "IO": d["intensity_overlap"],
            "1 - RMSE/pi": 1.0 - d["phase_rmse_rad"] / np.pi,
        }

    metric_names = ["CO", "IO", r"$1 - \phi_{\rm RMSE}/\pi$"]
    y = np.arange(len(COMBO_SUBDIRS))
    h = 0.2

    for j, (mkey, mlabel) in enumerate(zip(["CO", "IO", "1 - RMSE/pi"], metric_names)):
        vals = [metrics_raw[sd][mkey] for sd in COMBO_SUBDIRS]
        ax.barh(y + (j - 1) * h, vals, h, color=["#4575b4", "#d73027", "#fc8d59"][j],
                alpha=0.85, label=mlabel)

    ax.set_yticks(y)
    ax.set_yticklabels(COMBO_LABELS)
    ax.set_xlabel("Metric value (higher = better)")
    ax.set_title("(d) Hybrid Loss Combinations")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlim(0.0, 1.0)
    ax.invert_yaxis()


def panel_convergence(ax: plt.Axes):
    """(e) Training loss convergence for best config per run."""
    best_configs = {
        "run01": "spacing_1mm",
        "run02": "tanh_2pi",
        "run03": "sig_2pi",
        "run04": "tanh_2pi",
        "run05": "combo4_sp_leak_io",
    }
    linestyles = {
        "run01": "-",
        "run02": "-",
        "run03": "--",
        "run04": "-.",
        "run05": ":",
    }

    for rkey, subdir in best_configs.items():
        hist = load_history(RUN_DIRS[rkey], subdir)
        if not hist:
            continue
        epochs = [h["epoch"] for h in hist]
        losses = [h["train_loss"] for h in hist]
        ax.plot(epochs, losses, linestyles[rkey], color=RUN_COLORS[rkey], lw=1.3,
                label=f"{RUN_LABELS[rkey]}", alpha=0.9)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("(e) Training Convergence (best config per run)")
    ax.legend(loc="upper right", fontsize=6.5)
    ax.set_xlim(0, 30)


def panel_co_vs_io(ax: plt.Axes, all_data: dict[str, dict[str, dict]]):
    """(f) CO vs IO trade-off scatter across all 28 configs."""
    marker_map = {
        "run01": "s",   # square — spacing
        "run02": "o",   # circle — phase range
        "run03": "o",
        "run04": "o",
        "run05": "D",   # diamond — hybrid
    }

    for rkey, configs in all_data.items():
        ios = [d["intensity_overlap"] for d in configs.values()]
        cos = [d["complex_overlap"] for d in configs.values()]
        ax.scatter(ios, cos, c=RUN_COLORS[rkey], marker=marker_map[rkey],
                   s=40, alpha=0.8, edgecolors="white", linewidths=0.3,
                   label=RUN_LABELS[rkey], zorder=3)

    # Baseline crosshairs
    ax.axhline(BASELINE_CO, color="gray", ls="--", lw=0.6, alpha=0.6)
    ax.axvline(BASELINE_IO, color="gray", ls="--", lw=0.6, alpha=0.6)
    ax.text(BASELINE_IO - 0.01, 0.01, "baseline IO", fontsize=6, color="gray",
            ha="right", va="bottom", rotation=90)
    ax.text(0.12, BASELINE_CO + 0.004, "baseline CO", fontsize=6, color="gray")

    ax.set_xlabel("Intensity overlap")
    ax.set_ylabel("Complex overlap")
    ax.set_title("(f) CO vs IO Trade-off")
    ax.legend(loc="upper left", fontsize=6.5, markerscale=0.8)
    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(0.0, 0.30)


# ── Phase constraint config for each best run ───────────────────────────────
BEST_RUNS = {
    "run01": {"subdir": "spacing_1mm",       "constraint": "symmetric_tanh", "phase_max": np.pi},
    "run02": {"subdir": "tanh_2pi",          "constraint": "symmetric_tanh", "phase_max": 2*np.pi},
    "run03": {"subdir": "sig_2pi",           "constraint": "sigmoid",        "phase_max": 2*np.pi},
    "run04": {"subdir": "tanh_2pi",          "constraint": "symmetric_tanh", "phase_max": 2*np.pi},
    "run05": {"subdir": "combo4_sp_leak_io", "constraint": "symmetric_tanh", "phase_max": 2*np.pi},
}

BEST_RUN_TITLES = {
    "run01": "01 Complex\nspacing 1mm, tanh π",
    "run02": "02 Complex\ntanh 2π",
    "run03": "03 Phasor\nsig 2π",
    "run04": "04 Irradiance\ntanh 2π",
    "run05": "05 Hybrid\nSP+Leak+IO, tanh 2π",
}


def load_phases_from_checkpoint(run_dir: Path, subdir: str,
                                constraint: str, phase_max: float,
                                downsample: int = 4) -> np.ndarray:
    """Load phase masks from checkpoint, apply constraint, wrap to [0, 2π].

    Returns array of shape (num_layers, n//ds, n//ds) in [0, 2π].
    """
    ckpt_path = run_dir / subdir / "checkpoint.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    layers = []
    for i in range(5):
        raw = sd[f"layers.{i}.raw"].numpy()
        if constraint == "symmetric_tanh":
            phase = phase_max * np.tanh(raw)
        elif constraint == "sigmoid":
            phase = phase_max / (1.0 + np.exp(-raw))
        else:
            phase = raw
        # Wrap to [0, 2π]
        phase_wrapped = phase % (2 * np.pi)
        # Downsample for display
        if downsample > 1:
            phase_wrapped = phase_wrapped[::downsample, ::downsample]
        layers.append(phase_wrapped)

    return np.stack(layers, axis=0)


def generate_phase_layer_figure():
    """Generate 5×5 grid: rows=runs, cols=layers, all phases in [0, 2π]."""
    apply_style()

    fig, axes = plt.subplots(5, 5, figsize=(18, 16))

    # HSV-like cyclic colormap for phase [0, 2π]
    cmap = plt.cm.twilight

    for row_idx, rkey in enumerate(["run01", "run02", "run03", "run04", "run05"]):
        cfg = BEST_RUNS[rkey]
        phases = load_phases_from_checkpoint(
            RUN_DIRS[rkey], cfg["subdir"], cfg["constraint"], cfg["phase_max"]
        )
        print(f"  {rkey}: {cfg['subdir']} — "
              f"phase range [{phases.min():.2f}, {phases.max():.2f}] rad "
              f"(wrapped to [0, 2π])")

        for col_idx in range(5):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(
                phases[col_idx],
                cmap=cmap, vmin=0, vmax=2 * np.pi,
                interpolation="nearest", aspect="equal",
            )
            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(f"Layer {col_idx + 1}", fontsize=10, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(BEST_RUN_TITLES[rkey], fontsize=9, fontweight="bold",
                              labelpad=10)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=mcolors.Normalize(0, 2 * np.pi), cmap=cmap),
        cax=cbar_ax,
    )
    cbar.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    cbar.set_ticklabels(["0", "π/2", "π", "3π/2", "2π"])
    cbar.set_label("Phase (rad)", fontsize=10)

    fig.suptitle(
        "FD2NN Trained Phase Masks — All Wrapped to [0, 2π]\n"
        r"$\lambda=1.55\,\mu$m  |  5 layers  |  1024×1024",
        fontsize=13, fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(left=0.12, right=0.90, top=0.92, bottom=0.03,
                        hspace=0.08, wspace=0.05)

    out_path = FIG_DIR / "dashboard_phase_layers.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── Field Comparison Helpers (reuse from kim2026.viz.fd2nn_sweep) ────────────

def _complex_from_npz(npz, prefix: str) -> np.ndarray:
    return npz[f"{prefix}_real"] + 1j * npz[f"{prefix}_imag"]


def _center_crop(arr: np.ndarray, radius: int) -> np.ndarray:
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    return arr[cy - radius:cy + radius, cx - radius:cx + radius]


def _normalized_irradiance(field: np.ndarray, ref_max: float) -> np.ndarray:
    return np.abs(field) ** 2 / max(ref_max, 1e-12)


def _masked_phase_0_2pi(field: np.ndarray, threshold: float) -> np.ma.MaskedArray:
    """Phase in [0, 2π], masked where intensity < threshold."""
    intensity = np.abs(field) ** 2
    phase = np.angle(field) % (2 * np.pi)
    return np.ma.masked_where(intensity < threshold, phase)


def _phase_error(field: np.ndarray, target: np.ndarray,
                 threshold: float) -> np.ma.MaskedArray:
    """Phase difference ∈ [-π, π], masked where either field is dim."""
    f_i = np.abs(field) ** 2
    t_i = np.abs(target) ** 2
    err = np.angle(field * np.conj(target))
    return np.ma.masked_where((f_i < threshold) | (t_i < threshold), err)


def _radial_profile(field: np.ndarray, ref_max: float) -> np.ndarray:
    irr = _normalized_irradiance(field, ref_max)
    cy, cx = irr.shape[0] // 2, irr.shape[1] // 2
    yy, xx = np.mgrid[:irr.shape[0], :irr.shape[1]]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(int)
    r_max = min(rr.max() + 1, irr.shape[0] // 2)
    radial = np.zeros(r_max)
    for r in range(r_max):
        mask = rr == r
        if mask.any():
            radial[r] = irr[mask].mean()
    return radial


def load_best_run_fields() -> dict[str, dict[str, np.ndarray]]:
    """Load input/pred/target complex fields for each best run."""
    fields = {}
    for rkey, cfg in BEST_RUNS.items():
        npz_path = RUN_DIRS[rkey] / cfg["subdir"] / "sample_fields.npz"
        if not npz_path.exists():
            print(f"  WARNING: {npz_path} not found, skipping")
            continue
        npz = np.load(npz_path)
        fields[rkey] = {
            "input": _complex_from_npz(npz, "input"),
            "pred": _complex_from_npz(npz, "pred"),
            "target": _complex_from_npz(npz, "target"),
        }
    return fields


# ── Figure A: Intensity Comparison ──────────────────────────────────────────

def generate_field_intensity_figure(fields: dict, crop: int = 200):
    """5 runs × 4 cols: Input|Target|Prediction|Error intensity."""
    run_keys = [k for k in BEST_RUNS if k in fields]
    nrows = len(run_keys)

    # Global reference max for consistent colorscale
    ref_max = max(
        float((np.abs(fields[rk]["target"]) ** 2).max())
        for rk in run_keys
    )
    phase_thr = ref_max * 1e-3

    fig, axes = plt.subplots(nrows, 4, figsize=(14, 3.2 * nrows))
    col_titles = ["Input (turbulent)", "Target (vacuum)", "Prediction", "|Pred − Target| error"]

    for row, rk in enumerate(run_keys):
        inp = _center_crop(fields[rk]["input"], crop)
        tgt = _center_crop(fields[rk]["target"], crop)
        prd = _center_crop(fields[rk]["pred"], crop)

        i_inp = _normalized_irradiance(inp, ref_max)
        i_tgt = _normalized_irradiance(tgt, ref_max)
        i_prd = _normalized_irradiance(prd, ref_max)
        i_err = np.abs(i_prd - i_tgt)

        images = [i_inp, i_tgt, i_prd, i_err]
        cmaps = ["inferno", "inferno", "inferno", "magma"]
        vmaxes = [1.0, 1.0, 1.0, 0.3]

        for col, (img, cmap, vmax) in enumerate(zip(images, cmaps, vmaxes)):
            ax = axes[row, col]
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax,
                           origin="lower", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(BEST_RUN_TITLES[rk], fontsize=8, fontweight="bold")

    # Colorbars
    for col_idx, vmax in enumerate([1.0, 1.0, 1.0, 0.3]):
        cbar_ax = fig.add_axes([
            0.06 + col_idx * 0.235,  # left
            0.02,                     # bottom
            0.18,                     # width
            0.012,                    # height
        ])
        sm = plt.cm.ScalarMappable(
            norm=plt.Normalize(0, vmax),
            cmap="inferno" if col_idx < 3 else "magma",
        )
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=7)
        if col_idx < 3:
            cbar.set_label("I / I_max", fontsize=7)
        else:
            cbar.set_label("|ΔI| / I_max", fontsize=7)

    fig.suptitle(
        "FD2NN Field Intensity Comparison (center-cropped)\n"
        r"$\lambda=1.55\,\mu$m  |  5 layers  |  1024×1024",
        fontsize=13, fontweight="bold", y=0.99,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.06,
                        hspace=0.10, wspace=0.05)

    out = FIG_DIR / "dashboard_field_intensity.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


# ── Figure B: Phase Comparison ──────────────────────────────────────────────

def generate_field_phase_figure(fields: dict, crop: int = 200):
    """5 runs × 4 cols: Input|Target|Prediction phase [0,2π] + error [-π,π]."""
    run_keys = [k for k in BEST_RUNS if k in fields]
    nrows = len(run_keys)

    ref_max = max(
        float((np.abs(fields[rk]["target"]) ** 2).max())
        for rk in run_keys
    )
    phase_thr = ref_max * 1e-3

    fig, axes = plt.subplots(nrows, 4, figsize=(14, 3.2 * nrows))
    col_titles = ["Input phase", "Target phase", "Prediction phase", "Phase error (pred−target)"]

    for row, rk in enumerate(run_keys):
        inp = _center_crop(fields[rk]["input"], crop)
        tgt = _center_crop(fields[rk]["target"], crop)
        prd = _center_crop(fields[rk]["pred"], crop)

        p_inp = _masked_phase_0_2pi(inp, phase_thr)
        p_tgt = _masked_phase_0_2pi(tgt, phase_thr)
        p_prd = _masked_phase_0_2pi(prd, phase_thr)
        p_err = _phase_error(prd, tgt, phase_thr)

        for col, (img, cmap, vmin, vmax) in enumerate([
            (p_inp, "twilight", 0, 2 * np.pi),
            (p_tgt, "twilight", 0, 2 * np.pi),
            (p_prd, "twilight", 0, 2 * np.pi),
            (p_err, "RdBu_r", -np.pi, np.pi),
        ]):
            ax = axes[row, col]
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
                           origin="lower", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(BEST_RUN_TITLES[rk], fontsize=8, fontweight="bold")

    # Colorbars
    for col_idx in range(4):
        if col_idx < 3:
            cmap, vmin, vmax = "twilight", 0, 2 * np.pi
            ticks = [0, np.pi, 2 * np.pi]
            labels = ["0", "π", "2π"]
            lbl = "Phase (rad)"
        else:
            cmap, vmin, vmax = "RdBu_r", -np.pi, np.pi
            ticks = [-np.pi, 0, np.pi]
            labels = ["-π", "0", "π"]
            lbl = "Δφ (rad)"

        cbar_ax = fig.add_axes([0.06 + col_idx * 0.235, 0.02, 0.18, 0.012])
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=cmap)
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(labels)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(lbl, fontsize=7)

    fig.suptitle(
        "FD2NN Field Phase Comparison [0, 2π] (center-cropped)\n"
        r"$\lambda=1.55\,\mu$m  |  5 layers  |  1024×1024",
        fontsize=13, fontweight="bold", y=0.99,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.06,
                        hspace=0.10, wspace=0.05)

    out = FIG_DIR / "dashboard_field_phase.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


# ── Figure C: Team Lead Summary ─────────────────────────────────────────────

def generate_field_summary_figure(fields: dict, metrics: dict):
    """3-panel team lead summary: radial profiles + metrics table + phase error."""
    run_keys = [k for k in BEST_RUNS if k in fields]

    ref_max = max(
        float((np.abs(fields[rk]["target"]) ** 2).max())
        for rk in run_keys
    )
    phase_thr = ref_max * 1e-3

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

    # (a) Radial intensity profiles
    ax_rad = fig.add_subplot(gs[0, 0])
    first = fields[run_keys[0]]
    rad_input = _radial_profile(first["input"], ref_max)
    rad_target = _radial_profile(first["target"], ref_max)
    ax_rad.plot(rad_input, "k--", lw=1.5, alpha=0.5, label="Input (turb)")
    ax_rad.plot(rad_target, "k-", lw=2, label="Target (vac)")

    for rk in run_keys:
        rad = _radial_profile(fields[rk]["pred"], ref_max)
        ax_rad.plot(rad, color=RUN_COLORS[rk], lw=1.3, alpha=0.9,
                    label=BEST_RUN_TITLES[rk].replace("\n", " — "))

    ax_rad.set_xlabel("Radius (pixels)")
    ax_rad.set_ylabel("Normalized irradiance")
    ax_rad.set_title("(a) Radial Intensity Profiles", fontweight="bold")
    ax_rad.set_xlim(0, 150)
    ax_rad.legend(fontsize=6.5, loc="upper right")
    ax_rad.grid(alpha=0.3)

    # (b) Log-scale radial profiles
    ax_log = fig.add_subplot(gs[0, 1])
    ax_log.semilogy(rad_input, "k--", lw=1.5, alpha=0.5, label="Input (turb)")
    ax_log.semilogy(rad_target, "k-", lw=2, label="Target (vac)")
    for rk in run_keys:
        rad = _radial_profile(fields[rk]["pred"], ref_max)
        ax_log.semilogy(rad, color=RUN_COLORS[rk], lw=1.3, alpha=0.9,
                        label=BEST_RUN_TITLES[rk].replace("\n", " — "))
    ax_log.set_xlabel("Radius (pixels)")
    ax_log.set_ylabel("Normalized irradiance (log)")
    ax_log.set_title("(b) Radial Profiles (log scale)", fontweight="bold")
    ax_log.set_xlim(0, 300)
    ax_log.set_ylim(1e-6, 1.5)
    ax_log.legend(fontsize=6.5, loc="upper right")
    ax_log.grid(alpha=0.3)

    # (c) Metrics comparison table
    ax_tbl = fig.add_subplot(gs[1, 0])
    ax_tbl.axis("off")
    ax_tbl.set_title("(c) Performance Metrics Summary", fontweight="bold", pad=20)

    col_labels = ["Run", "Loss Type", "CO ↑", "IO ↑", "φ RMSE ↓\n(rad)", "Strehl"]
    cell_data = []
    cell_colors = []

    for rk in run_keys:
        m = metrics[rk]
        co = m.get("complex_overlap", 0)
        io = m.get("intensity_overlap", 0)
        pr = m.get("phase_rmse_rad", 0)
        sr = m.get("strehl", 0)
        loss_type = BEST_RUN_TITLES[rk].split("\n")[0]
        cell_data.append([
            rk.replace("run0", "Run 0"),
            loss_type,
            f"{co:.4f}",
            f"{io:.4f}",
            f"{pr:.3f}",
            f"{sr:.2f}" if sr else "—",
        ])
        cell_colors.append([RUN_COLORS[rk] + "20"] * 6)

    table = ax_tbl.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=["#e0e0e0"] * 6,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    # Highlight best CO
    best_co_idx = max(range(len(run_keys)),
                      key=lambda i: metrics[run_keys[i]].get("complex_overlap", 0))
    table[best_co_idx + 1, 2].set_text_props(fontweight="bold", color="#1b7837")

    # (d) Centerline phase error
    ax_phi = fig.add_subplot(gs[1, 1])
    target_c = first["target"]
    n = target_c.shape[0]
    x_px = np.arange(n) - n // 2
    dx_um = 2.048e3 / n  # µm per pixel

    for rk in run_keys:
        prd = fields[rk]["pred"]
        p_err = _phase_error(prd, target_c, phase_thr)
        centerline = p_err[n // 2].filled(np.nan)
        ax_phi.plot(x_px * dx_um, centerline, color=RUN_COLORS[rk], lw=1.0,
                    alpha=0.8, label=BEST_RUN_TITLES[rk].replace("\n", " — "))

    ax_phi.set_xlabel("Position (µm)")
    ax_phi.set_ylabel("Phase error (rad)")
    ax_phi.set_title("(d) Centerline Phase Error vs Vacuum", fontweight="bold")
    ax_phi.set_xlim(-200, 200)
    ax_phi.set_ylim(-np.pi, np.pi)
    ax_phi.axhline(0, color="gray", ls="--", lw=0.5)
    ax_phi.legend(fontsize=6.5, loc="upper right")
    ax_phi.grid(alpha=0.3)

    fig.suptitle(
        "FD2NN Beam Cleanup — Team Lead Summary\n"
        r"$\lambda=1.55\,\mu$m  |  5 layers  |  Cn²: strong turbulence",
        fontsize=13, fontweight="bold",
    )

    out = FIG_DIR / "dashboard_field_summary.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    apply_style()

    # Load data
    d01 = load_test_metrics(RUN_DIRS["run01"], SPACING_SUBDIRS)
    d02 = load_test_metrics(RUN_DIRS["run02"], PHASE_SUBDIRS)
    d03 = load_test_metrics(RUN_DIRS["run03"], PHASE_SUBDIRS)
    d04 = load_test_metrics(RUN_DIRS["run04"], PHASE_SUBDIRS)
    d05 = load_test_metrics(RUN_DIRS["run05"], COMBO_SUBDIRS)

    print(f"Loaded: run01={len(d01)}, run02={len(d02)}, run03={len(d03)}, "
          f"run04={len(d04)}, run05={len(d05)} configs")

    # Build figure
    fig = plt.figure(figsize=(14, 18))
    gs = gridspec.GridSpec(3, 2, hspace=0.40, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    panel_spacing_sweep(ax1, d01)
    panel_co_comparison(ax2, d02, d03, d04)
    panel_phase_rmse(ax3, d02, d03, d04)
    panel_hybrid_metrics(ax4, d05)
    panel_convergence(ax5)
    panel_co_vs_io(ax6, {"run01": d01, "run02": d02, "run03": d03, "run04": d04, "run05": d05})

    # Suptitle
    fig.suptitle(
        "FD2NN Sweep Results Dashboard\n"
        r"$\lambda=1.55\,\mu$m  |  5 layers  |  1024×1024  |  30 epochs",
        fontsize=13, fontweight="bold", y=0.995,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "dashboard_sweep_results.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.close(fig)

    # Figure 2: Phase layer visualization
    print("\nGenerating phase layer figure...")
    generate_phase_layer_figure()

    # Figures 3-5: Field comparison (requires sample_fields.npz)
    print("\nLoading sample fields...")
    fields = load_best_run_fields()
    if fields:
        # Load metrics for summary table
        best_metrics = {}
        for rk, cfg in BEST_RUNS.items():
            if rk in fields:
                best_metrics[rk] = load_test_metrics(
                    RUN_DIRS[rk], [cfg["subdir"]]
                ).get(cfg["subdir"], {})

        print(f"Loaded fields for {len(fields)} runs: {list(fields.keys())}")

        print("\nGenerating field intensity figure...")
        generate_field_intensity_figure(fields)

        print("Generating field phase figure...")
        generate_field_phase_figure(fields)

        print("Generating team lead summary figure...")
        generate_field_summary_figure(fields, best_metrics)
    else:
        print("No sample_fields.npz found. Run generate_sample_fields.py first.")


if __name__ == "__main__":
    main()
