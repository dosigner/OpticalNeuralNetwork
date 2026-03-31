#!/usr/bin/env python
"""Comprehensive visualization for all FD2NN sweep experiments.

Fig 1: Loss Strategy Comparison (4 strategies × 3 metrics bar chart)
Fig 2: Phase Range Effect Heatmap (loss × phase_range × metric)
Fig 3: Trade-off Scatter (best config per loss, co vs io)
Fig 4: Training Curves (best config per loss)
Fig 5: Hybrid Combo Detail (combo 1-4)
Fig 6: Summary Table
Fig 7: Field Comparison (irradiance + phase images, best per loss)
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

plt.rcParams.update({
    "font.size": 10, "figure.dpi": 150, "figure.facecolor": "white",
    "axes.titlesize": 12, "axes.labelsize": 10,
})

RUNS = Path("/root/dj/D2NN/kim2026/runs")
FIG_DIR = RUNS / "figures_sweep_report"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("/root/dj/D2NN/kim2026/data/kim2026/1km_cn2e-14_w2m_n1024_dx2mm/cache")
MANIFEST = DATA_DIR.parent / "split_manifest.json"

BASELINE_CO = 0.1913
BASELINE_IO = 0.9725

SWEEPS = {
    "02_complex": {
        "dir": "02_fd2nn_complexloss_roi1024_phase_range_sweep_claude",
        "label": "Complex Field Loss\n(complex_overlap + amplitude_mse)",
        "short": "Complex", "color": "#3498db",
    },
    "03_phasor": {
        "dir": "03_fd2nn_phasorloss_roi1024_phase_range_sweep_claude",
        "label": "Phasor MSE Loss\n(unit phasor distance, phase only)",
        "short": "Phasor", "color": "#e74c3c",
    },
    "04_irradiance": {
        "dir": "04_fd2nn_irradianceloss_roi1024_phase_range_sweep_claude",
        "label": "Irradiance Loss\n(intensity_overlap + beam_radius + encircled_energy)",
        "short": "Irradiance", "color": "#2ecc71",
    },
    "05_hybrid": {
        "dir": "05_fd2nn_hybridloss_roi1024_loss_combo_sweep_claude",
        "label": "Hybrid Loss\n(4 combinations of above losses)",
        "short": "Hybrid", "color": "#9b59b6",
    },
}

PHASE_CONFIGS = ["sig_pi", "tanh_pi2", "sig_2pi", "tanh_pi", "sig_4pi", "tanh_2pi"]
PHASE_LABELS = ["[0,pi]\nsigmoid", "[-pi/2,pi/2]\ntanh", "[0,2pi]\nsigmoid",
                "[-pi,pi]\ntanh", "[0,4pi]\nsigmoid", "[-2pi,2pi]\ntanh"]
COMBO_CONFIGS = ["combo1_io_co", "combo2_io_br_ee", "combo3_co_io_br", "combo4_sp_leak_io"]
COMBO_LABELS = ["Combo1\nio:1 + co:0.5", "Combo2\nio:1 + br:0.5 + ee:0.5",
                "Combo3\nco:1 + io:0.5 + br:0.5", "Combo4\nphasor + leak + io"]


def load_metrics(sweep_dir, config_name):
    p = RUNS / sweep_dir / config_name / "test_metrics.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def load_history(sweep_dir, config_name):
    p = RUNS / sweep_dir / config_name / "history.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def get_best(sweep_key, metric="complex_overlap"):
    info = SWEEPS[sweep_key]
    configs = COMBO_CONFIGS if "hybrid" in sweep_key else PHASE_CONFIGS
    best_val, best_name, best_m = -1, None, None
    for c in configs:
        m = load_metrics(info["dir"], c)
        if m and m.get(metric, 0) > best_val:
            best_val = m[metric]
            best_name = c
            best_m = m
    return best_name, best_m


# ═══════════════════════════════════════════════════════════════════
# Fig 1: Loss Strategy Comparison
# ═══════════════════════════════════════════════════════════════════
def fig1():
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    names, co_vals, pr_vals, io_vals, colors, xlabels = [], [], [], [], [], []
    for key, info in SWEEPS.items():
        best_name, m = get_best(key)
        if m is None:
            continue
        names.append(key)
        co_vals.append(m["complex_overlap"])
        pr_vals.append(m["phase_rmse_rad"])
        io_vals.append(m["intensity_overlap"])
        colors.append(info["color"])
        xlabels.append(info["label"])

    x = np.arange(len(names))
    w = 0.55

    # Panel 1: Complex Overlap
    bars = axes[0].bar(x, co_vals, w, color=colors, ec="k", lw=0.8)
    axes[0].axhline(BASELINE_CO, color="k", ls="--", lw=1.5,
                    label=f"Baseline (no D2NN) = {BASELINE_CO:.4f}")
    axes[0].set_xticks(x); axes[0].set_xticklabels(xlabels, fontsize=7)
    axes[0].set_ylabel("Complex Overlap")
    axes[0].set_title("Complex Overlap\n(amplitude + phase fidelity, higher = better)")
    axes[0].legend(fontsize=8, loc="upper right")
    for b, v in zip(bars, co_vals):
        delta = ((v / BASELINE_CO) - 1) * 100
        axes[0].text(b.get_x()+w/2, b.get_height()+0.005,
                    f"{v:.4f}\n({delta:+.0f}%)", ha="center", fontsize=8, fontweight="bold")

    # Panel 2: Phase RMSE
    bars = axes[1].bar(x, pr_vals, w, color=colors, ec="k", lw=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(xlabels, fontsize=7)
    axes[1].set_ylabel("Phase RMSE [rad]")
    axes[1].set_title("Phase RMSE\n(wavefront error, lower = better)")
    for b, v in zip(bars, pr_vals):
        axes[1].text(b.get_x()+w/2, b.get_height()+0.02, f"{v:.3f}", ha="center", fontsize=8)

    # Panel 3: Intensity Overlap
    bars = axes[2].bar(x, io_vals, w, color=colors, ec="k", lw=0.8)
    axes[2].axhline(BASELINE_IO, color="k", ls="--", lw=1.5,
                    label=f"Baseline = {BASELINE_IO:.4f}")
    axes[2].set_xticks(x); axes[2].set_xticklabels(xlabels, fontsize=7)
    axes[2].set_ylabel("Intensity Overlap")
    axes[2].set_title("Intensity Overlap\n(beam shape match, higher = better)")
    axes[2].legend(fontsize=8, loc="lower right")
    for b, v in zip(bars, io_vals):
        axes[2].text(b.get_x()+w/2, b.get_height()+0.01, f"{v:.3f}", ha="center", fontsize=8)

    fig.suptitle("Fig 1: Loss Strategy Comparison — Best Config per Strategy\n"
                 "(FD2NN metalens, dx=2um, spacing=1mm, 5 layers, Cn2=1e-14, 30 epochs)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_loss_strategy_comparison.png", bbox_inches="tight")
    plt.close()
    print("  fig1")


# ═══════════════════════════════════════════════════════════════════
# Fig 2: Phase Range Heatmap
# ═══════════════════════════════════════════════════════════════════
def fig2():
    metrics = ["complex_overlap", "phase_rmse_rad", "intensity_overlap"]
    titles = ["Complex Overlap (higher=better)", "Phase RMSE [rad] (lower=better)", "Intensity Overlap (higher=better)"]
    cmaps = ["Blues", "Reds_r", "Greens"]
    sweep_keys = ["02_complex", "03_phasor", "04_irradiance"]
    sweep_labels = [SWEEPS[k]["short"] for k in sweep_keys]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    for row, (metric, title, cmap) in enumerate(zip(metrics, titles, cmaps)):
        data = np.zeros((len(sweep_keys), len(PHASE_CONFIGS)))
        for i, sk in enumerate(sweep_keys):
            for j, pc in enumerate(PHASE_CONFIGS):
                m = load_metrics(SWEEPS[sk]["dir"], pc)
                data[i, j] = m[metric] if m else 0

        im = axes[row].imshow(data, cmap=cmap, aspect="auto")
        axes[row].set_xticks(range(len(PHASE_LABELS)))
        axes[row].set_xticklabels(PHASE_LABELS, fontsize=8)
        axes[row].set_yticks(range(len(sweep_labels)))
        axes[row].set_yticklabels(sweep_labels, fontsize=10, fontweight="bold")
        axes[row].set_title(title, fontsize=11)
        plt.colorbar(im, ax=axes[row], shrink=0.8)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                axes[row].text(j, i, f"{data[i,j]:.3f}", ha="center", va="center",
                              fontsize=8, fontweight="bold",
                              color="white" if data[i,j] > data.mean() else "black")

    fig.suptitle("Fig 2: Phase Range Effect per Loss Strategy\n"
                 "(each cell = test metric for that loss × phase_range combination)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_phase_range_heatmap.png", bbox_inches="tight")
    plt.close()
    print("  fig2")


# ═══════════════════════════════════════════════════════════════════
# Fig 3: Trade-off Scatter
# ═══════════════════════════════════════════════════════════════════
def fig3():
    fig, ax = plt.subplots(figsize=(10, 8))

    for key, info in SWEEPS.items():
        best_name, m = get_best(key)
        if m is None:
            continue
        ax.scatter(m["intensity_overlap"], m["complex_overlap"],
                  c=info["color"], s=200, label=f"{info['short']} ({best_name})",
                  edgecolors="k", linewidth=1, zorder=3)
        ax.annotate(f"  {info['short']}\n  co={m['complex_overlap']:.3f}\n  io={m['intensity_overlap']:.3f}",
                   (m["intensity_overlap"], m["complex_overlap"]),
                   fontsize=8, fontweight="bold")

    ax.plot(BASELINE_IO, BASELINE_CO, "kx", ms=15, mew=3, label="Baseline (no D2NN)", zorder=5)
    ax.annotate(f"  Baseline\n  co={BASELINE_CO:.3f}\n  io={BASELINE_IO:.3f}",
               (BASELINE_IO, BASELINE_CO), fontsize=8)

    ax.set(xlabel="Intensity Overlap (beam shape match) →",
           ylabel="Complex Overlap (wavefront fidelity) →",
           title="Fig 3: Trade-off — Complex Overlap vs Intensity Overlap\n"
                 "(Ideal: top-right corner)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Ideal region
    ax.fill_between([0.8, 1.0], 0.25, 0.35, alpha=0.1, color="green")
    ax.text(0.9, 0.30, "Ideal\nRegion", ha="center", fontsize=12, color="green", fontweight="bold", alpha=0.5)

    ax.set_xlim(0, 1.05); ax.set_ylim(0, 0.35)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_tradeoff_scatter.png", bbox_inches="tight")
    plt.close()
    print("  fig3")


# ═══════════════════════════════════════════════════════════════════
# Fig 4: Training Curves
# ═══════════════════════════════════════════════════════════════════
def fig4():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for key, info in SWEEPS.items():
        best_name, _ = get_best(key)
        if best_name is None:
            continue
        h = load_history(info["dir"], best_name)
        if h is None:
            continue
        epochs = [e["epoch"] for e in h]
        losses = [e["train_loss"] for e in h]
        co_e = [(e["epoch"], e["complex_overlap"]) for e in h if "complex_overlap" in e]
        io_e = [(e["epoch"], e["intensity_overlap"]) for e in h if "intensity_overlap" in e]

        lbl = f"{info['short']} ({best_name})"
        axes[0].plot(epochs, losses, color=info["color"], lw=1.5, label=lbl)
        if co_e:
            axes[1].plot(*zip(*co_e), "o-", color=info["color"], ms=4, lw=1.5, label=lbl)
        if io_e:
            axes[2].plot(*zip(*io_e), "s-", color=info["color"], ms=4, lw=1.5, label=lbl)

    axes[1].axhline(BASELINE_CO, color="k", ls="--", lw=1, label=f"BL co={BASELINE_CO:.4f}")
    axes[2].axhline(BASELINE_IO, color="k", ls="--", lw=1, label=f"BL io={BASELINE_IO:.4f}")
    axes[0].set(xlabel="Epoch", ylabel="Train Loss", title="Training Loss")
    axes[1].set(xlabel="Epoch", ylabel="Complex Overlap", title="Val Complex Overlap")
    axes[2].set(xlabel="Epoch", ylabel="Intensity Overlap", title="Val Intensity Overlap")
    for ax in axes:
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle("Fig 4: Training Curves — Best Config per Loss Strategy", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_training_curves.png", bbox_inches="tight")
    plt.close()
    print("  fig4")


# ═══════════════════════════════════════════════════════════════════
# Fig 5: Hybrid Combo Detail
# ═══════════════════════════════════════════════════════════════════
def fig5():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]

    co_v, pr_v, io_v = [], [], []
    for c in COMBO_CONFIGS:
        m = load_metrics(SWEEPS["05_hybrid"]["dir"], c)
        co_v.append(m["complex_overlap"] if m else 0)
        pr_v.append(m["phase_rmse_rad"] if m else 0)
        io_v.append(m["intensity_overlap"] if m else 0)

    x = np.arange(len(COMBO_CONFIGS)); w = 0.55

    bars = axes[0].bar(x, co_v, w, color=colors, ec="k", lw=0.8)
    axes[0].axhline(BASELINE_CO, color="k", ls="--", lw=1)
    axes[0].set_xticks(x); axes[0].set_xticklabels(COMBO_LABELS, fontsize=7)
    axes[0].set_title("Complex Overlap"); axes[0].set_ylabel("co")
    for b, v in zip(bars, co_v):
        axes[0].text(b.get_x()+w/2, b.get_height()+0.005, f"{v:.4f}", ha="center", fontsize=8)

    bars = axes[1].bar(x, pr_v, w, color=colors, ec="k", lw=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(COMBO_LABELS, fontsize=7)
    axes[1].set_title("Phase RMSE [rad]")
    for b, v in zip(bars, pr_v):
        axes[1].text(b.get_x()+w/2, b.get_height()+0.02, f"{v:.3f}", ha="center", fontsize=8)

    bars = axes[2].bar(x, io_v, w, color=colors, ec="k", lw=0.8)
    axes[2].axhline(BASELINE_IO, color="k", ls="--", lw=1)
    axes[2].set_xticks(x); axes[2].set_xticklabels(COMBO_LABELS, fontsize=7)
    axes[2].set_title("Intensity Overlap")
    for b, v in zip(bars, io_v):
        axes[2].text(b.get_x()+w/2, b.get_height()+0.01, f"{v:.3f}", ha="center", fontsize=8)

    fig.suptitle("Fig 5: Hybrid Loss Combo Detail (Sweep 05)\n"
                 "Which combination of losses works best?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_hybrid_combo_detail.png", bbox_inches="tight")
    plt.close()
    print("  fig5")


# ═══════════════════════════════════════════════════════════════════
# Fig 6: Summary Table
# ═══════════════════════════════════════════════════════════════════
def fig6():
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.axis("off")

    rows = [["Baseline\n(no D2NN)", "-", f"{BASELINE_CO:.4f}", "-",
             f"{BASELINE_IO:.4f}", "-", "-"]]

    for key, info in SWEEPS.items():
        best_name, m = get_best(key)
        if m is None:
            continue
        co_d = ((m["complex_overlap"] / BASELINE_CO) - 1) * 100
        io_d = ((m["intensity_overlap"] / BASELINE_IO) - 1) * 100
        rows.append([
            info["short"], best_name,
            f"{m['complex_overlap']:.4f}", f"{co_d:+.1f}%",
            f"{m['intensity_overlap']:.4f}", f"{io_d:+.1f}%",
            f"{m['phase_rmse_rad']:.3f}",
        ])

    cols = ["Loss Strategy", "Best Config", "Complex\nOverlap", "co vs BL",
            "Intensity\nOverlap", "io vs BL", "Phase RMSE\n[rad]"]
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 2.0)
    for j in range(len(cols)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Highlight best
    table[2, 2].set_facecolor("#d5f5e3")  # best co
    table[2, 6].set_facecolor("#d5f5e3")  # best pr
    table[4, 4].set_facecolor("#d5f5e3")  # best io

    fig.suptitle("Fig 6: Comprehensive Results Summary", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_summary_table.png", bbox_inches="tight")
    plt.close()
    print("  fig6")


# ═══════════════════════════════════════════════════════════════════
# Fig 7: Field Comparison (irradiance + phase)
# ═══════════════════════════════════════════════════════════════════
def fig7():
    """Load best model per sweep, run inference, compare fields."""
    from kim2026.data.dataset import CachedFieldDataset
    from kim2026.models.fd2nn import BeamCleanupFD2NN
    from kim2026.training.targets import apply_receiver_aperture
    import inspect

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = CachedFieldDataset(cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="test")
    sample = test_ds[0]
    u_turb = sample["u_turb"].unsqueeze(0).to(device)
    u_vac = sample["u_vacuum"].unsqueeze(0).to(device)
    N_grid = u_turb.shape[-1]; C = N_grid // 2; M = 100

    # Common model params (from sweep_phase_range.py COMMON)
    MODEL_KWARGS = dict(
        n=N_grid, wavelength_m=1.55e-6, window_m=0.002048,
        num_layers=5, layer_spacing_m=1e-3,
        dual_2f_f1_m=1e-3, dual_2f_f2_m=1e-3,
        dual_2f_na1=0.16, dual_2f_na2=0.16, dual_2f_apply_scaling=False,
    )
    W_M, AP_M = 0.002048, 0.002

    def crop(f):
        return f[C-M:C+M, C-M:C+M]

    sweep_keys = ["02_complex", "04_irradiance", "05_hybrid"]
    n_rows = len(sweep_keys) + 1  # +1 for phase row
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))

    pred_fields = {}

    for key_idx, key in enumerate(sweep_keys):
        info = SWEEPS[key]
        best_name, best_m = get_best(key)
        ckpt_path = RUNS / info["dir"] / best_name / "checkpoint.pt"
        if not ckpt_path.exists():
            print(f"    Skip {key}: no checkpoint")
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Get phase_max from best_name
        phase_max_map = {"sig_pi": 3.14159, "tanh_pi2": 1.5708, "sig_2pi": 6.2832,
                         "tanh_pi": 3.14159, "sig_4pi": 12.5664, "tanh_2pi": 6.2832,
                         "combo1_io_co": 6.2832, "combo2_io_br_ee": 6.2832,
                         "combo3_co_io_br": 6.2832, "combo4_sp_leak_io": 6.2832}
        constraint_map = {"sig_pi": "sigmoid", "tanh_pi2": "symmetric_tanh",
                         "sig_2pi": "sigmoid", "tanh_pi": "symmetric_tanh",
                         "sig_4pi": "sigmoid", "tanh_2pi": "symmetric_tanh",
                         "combo1_io_co": "symmetric_tanh", "combo2_io_br_ee": "symmetric_tanh",
                         "combo3_co_io_br": "symmetric_tanh", "combo4_sp_leak_io": "symmetric_tanh"}

        try:
            model = BeamCleanupFD2NN(
                **MODEL_KWARGS,
                phase_max=phase_max_map.get(best_name, 6.2832),
                phase_constraint=constraint_map.get(best_name, "symmetric_tanh"),
            ).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
        except Exception as e:
            print(f"    Skip {key} ({best_name}): {e}")
            continue

        with torch.no_grad():
            tgt = apply_receiver_aperture(u_vac, receiver_window_m=W_M, aperture_diameter_m=AP_M)
            inp = apply_receiver_aperture(u_turb, receiver_window_m=W_M, aperture_diameter_m=AP_M)
            pred = model(inp)

        inp_np = inp[0].cpu().numpy()
        pred_np = pred[0].cpu().numpy()
        tgt_np = tgt[0].cpu().numpy()
        pred_fields[key] = pred_np

        row = key_idx + 1
        co = best_m["complex_overlap"]
        io = best_m["intensity_overlap"]

        for col, (title, f) in enumerate([
            ("Turbulent Input\n|E|", inp_np),
            (f"{info['short']} Output\n|E|", pred_np),
            ("Vacuum Target\n|E|", tgt_np),
            ("Residual\n|pred - target|", pred_np - tgt_np),
        ]):
            amp = crop(np.abs(f))
            cmap = "inferno" if col < 3 else "hot"
            im = axes[row, col].imshow(amp, cmap=cmap, origin="lower")
            plt.colorbar(im, ax=axes[row, col], shrink=0.7)
            if row == 1:
                axes[0, col].set_title(title, fontsize=11, fontweight="bold")
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])

        axes[row, 0].set_ylabel(f"{info['short']}\n({best_name})\nco={co:.3f} io={io:.3f}",
                                fontsize=8, fontweight="bold")

    # Row 0: Phase comparison (using complex loss model = best phase)
    best_pred = pred_fields.get("02_complex", pred_np)
    with torch.no_grad():
        tgt = apply_receiver_aperture(u_vac, receiver_window_m=W_M, aperture_diameter_m=AP_M)
        inp = apply_receiver_aperture(u_turb, receiver_window_m=W_M, aperture_diameter_m=AP_M)
    inp_np = inp[0].cpu().numpy()
    tgt_np = tgt[0].cpu().numpy()

    for col, (title, ph) in enumerate([
        ("Input Phase\narg(E_turb)", np.angle(inp_np)),
        ("Complex Loss Output Phase\narg(E_pred)", np.angle(best_pred)),
        ("Target Phase\narg(E_vac)", np.angle(tgt_np)),
        ("Phase Error\narg(pred * conj(tgt))", np.angle(best_pred * np.conj(tgt_np))),
    ]):
        im = axes[0, col].imshow(crop(ph), cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi, origin="lower")
        plt.colorbar(im, ax=axes[0, col], shrink=0.7)
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
    axes[0, 0].set_ylabel("Phase [rad]", fontsize=9, fontweight="bold")

    fig.suptitle("Fig 7: Field Comparison — Phase (row 1) + Irradiance per Loss Strategy (rows 2-4)\n"
                 "(200×200 crop, dx=2um, Cn2=1e-14)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_field_comparison.png", bbox_inches="tight")
    plt.close()
    print("  fig7")


def main():
    print(f"Generating sweep report figures → {FIG_DIR}/\n")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    print("\n  Generating Fig 7 (requires GPU inference)...")
    fig7()
    print(f"\nAll 7 figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()