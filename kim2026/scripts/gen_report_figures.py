#!/usr/bin/env python
"""Generate all figures referenced in fd2nn-loss-sweep-comprehensive.report.md"""

from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "figure.facecolor": "white"})

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
FIG_DIR = RUNS / "figures_sweep_report"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Data loading ───────────────────────────────────────────────

def load_metrics(sweep_dir, configs):
    results = {}
    for c in configs:
        p = sweep_dir / c / "test_metrics.json"
        if p.exists():
            with open(p) as f:
                results[c] = json.load(f)
    return results

def load_history(sweep_dir, config):
    p = sweep_dir / config / "history.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return []

def load_checkpoint_fields(sweep_dir, config, device="cpu"):
    """Load model from checkpoint and generate sample fields."""
    sys.path.insert(0, str(ROOT / "src"))
    from kim2026.data.dataset import CachedFieldDataset
    from kim2026.models.fd2nn import BeamCleanupFD2NN
    from kim2026.training.targets import apply_receiver_aperture

    ckpt_path = sweep_dir / config / "checkpoint.pt"
    if not ckpt_path.exists():
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    ds = CachedFieldDataset(
        cache_dir=str(ROOT / "data/kim2026/1km_cn2e-14_w2m_n1024_dx2mm/cache"),
        manifest_path=str(ROOT / "data/kim2026/1km_cn2e-14_w2m_n1024_dx2mm/split_manifest.json"),
        split="test",
    )
    sample = ds[0]
    u_turb = sample["u_turb"].unsqueeze(0).to(device)
    u_vac = sample["u_vacuum"].unsqueeze(0).to(device)

    W = float(cfg.get("receiver_window_m", 0.002048))
    AP = float(cfg.get("aperture_diameter_m", 0.002))

    model = BeamCleanupFD2NN(
        n=1024, wavelength_m=1.55e-6, window_m=W, num_layers=5,
        layer_spacing_m=float(cfg.get("layer_spacing_m", 1e-3)),
        phase_max=float(cfg.get("phase_max", 6.2832)),
        phase_constraint=str(cfg.get("phase_constraint", "symmetric_tanh")),
        dual_2f_f1_m=float(cfg.get("dual_2f_f1_m", 1e-3)),
        dual_2f_f2_m=float(cfg.get("dual_2f_f2_m", 1e-3)),
        dual_2f_na1=float(cfg.get("dual_2f_na1", 0.16)),
        dual_2f_na2=float(cfg.get("dual_2f_na2", 0.16)),
        dual_2f_apply_scaling=bool(cfg.get("dual_2f_apply_scaling", False)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        inp = apply_receiver_aperture(u_turb, receiver_window_m=W, aperture_diameter_m=AP)
        tgt = apply_receiver_aperture(u_vac, receiver_window_m=W, aperture_diameter_m=AP)
        pred = model(inp)
        phases = [layer.phase().cpu().numpy() for layer in model.layers]

    return {
        "input": inp[0].cpu().numpy(),
        "pred": pred[0].cpu().numpy(),
        "target": tgt[0].cpu().numpy(),
        "phases": phases,
    }

C = 512
def crop(f, m=100):
    return f[C-m:C+m, C-m:C+m]

# ─── Fig 1: Loss Strategy Comparison Bar Chart (§1.1) ──────────

def fig1_summary_bars():
    """4-strategy summary bar chart with baseline reference."""
    strategies = {
        "Baseline": {"co": 0.191, "pr": None, "io": 0.973},
        "Complex": {"co": 0.270, "pr": 0.359, "io": 0.378},
        "Phasor": {"co": 0.098, "pr": 0.874, "io": 0.387},
        "Irradiance": {"co": 0.099, "pr": 1.679, "io": 0.933},
        "Hybrid\n(combo3)": {"co": 0.126, "pr": 1.540, "io": 0.910},
    }
    names = list(strategies.keys())
    colors = ["#555555", "#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # CO
    vals = [s["co"] for s in strategies.values()]
    bars = axes[0].bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].axhline(0.191, color="black", ls="--", lw=1, alpha=0.5)
    axes[0].set_ylabel("Complex Overlap"); axes[0].set_title("Complex Overlap (higher=better)")
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f"{v:.3f}", ha="center", fontsize=8)

    # PR
    vals = [s["pr"] if s["pr"] else 0 for s in strategies.values()]
    bars = axes[1].bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Phase RMSE [rad]"); axes[1].set_title("Phase RMSE (lower=better)")
    for b, v in zip(bars, vals):
        if v > 0:
            axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{v:.3f}", ha="center", fontsize=8)

    # IO
    vals = [s["io"] for s in strategies.values()]
    bars = axes[2].bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)
    axes[2].axhline(0.973, color="black", ls="--", lw=1, alpha=0.5)
    axes[2].set_ylabel("Intensity Overlap"); axes[2].set_title("Intensity Overlap (higher=better)")
    for b, v in zip(bars, vals):
        axes[2].text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{v:.3f}", ha="center", fontsize=8)

    fig.suptitle("Fig 1: Loss Strategy Comparison (best config each, tanh_2pi, spacing=1mm)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_loss_strategy_comparison.png", bbox_inches="tight")
    plt.close()
    print("  fig1_loss_strategy_comparison.png")


# ─── Fig 2: Phase Range Effect (§4) ────────────────────────────

def fig2_phase_range():
    """Phase range sweep results for complex loss."""
    sweep = RUNS / "02_fd2nn_complexloss_roi1024_phase_range_sweep_claude"
    cfgs = ["sig_pi", "tanh_pi2", "sig_2pi", "tanh_pi", "sig_4pi", "tanh_2pi"]
    labels = ["[0,π]", "[-π/2,π/2]", "[0,2π]", "[-π,π]", "[0,4π]", "[-2π,2π]"]
    metrics = load_metrics(sweep, cfgs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    co = [metrics[c]["complex_overlap"] for c in cfgs]
    pr = [metrics[c]["phase_rmse_rad"] for c in cfgs]
    x = range(len(cfgs))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(cfgs)))

    bars = axes[0].bar(x, co, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylabel("Complex Overlap"); axes[0].set_title("CO vs Phase Range")
    axes[0].axhline(0.191, color="black", ls="--", lw=1, label="Baseline")
    axes[0].legend(fontsize=8)
    for b, v in zip(bars, co):
        axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.003, f"{v:.3f}", ha="center", fontsize=7)

    bars = axes[1].bar(x, pr, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_ylabel("Phase RMSE [rad]"); axes[1].set_title("Phase RMSE vs Phase Range")
    for b, v in zip(bars, pr):
        axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{v:.3f}", ha="center", fontsize=7)

    fig.suptitle("Fig 2: Phase Range Effect on Complex Loss (spacing=1mm)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_phase_range_effect.png", bbox_inches="tight")
    plt.close()
    print("  fig2_phase_range_effect.png")


# ─── Fig 3: CO vs IO Trade-off Scatter (§1.2) ──────────────────

def fig3_tradeoff_scatter():
    """Scatter plot: CO vs IO for all experiments."""
    all_points = []

    # Sweep 02: Complex loss
    sweep = RUNS / "02_fd2nn_complexloss_roi1024_phase_range_sweep_claude"
    for c in ["sig_pi", "tanh_pi2", "sig_2pi", "tanh_pi", "sig_4pi", "tanh_2pi"]:
        m = load_metrics(sweep, [c]).get(c)
        if m: all_points.append(("Complex", m["complex_overlap"], m["intensity_overlap"], "#3498db"))

    # Sweep 03: Phasor loss
    sweep = RUNS / "03_fd2nn_phasorloss_roi1024_phase_range_sweep_claude"
    for c in ["sig_pi", "tanh_pi2", "sig_2pi", "tanh_pi", "sig_4pi", "tanh_2pi"]:
        m = load_metrics(sweep, [c]).get(c)
        if m: all_points.append(("Phasor", m["complex_overlap"], m["intensity_overlap"], "#e74c3c"))

    # Sweep 04: Irradiance loss
    sweep = RUNS / "04_fd2nn_irradianceloss_roi1024_phase_range_sweep_claude"
    for c in ["sig_pi", "tanh_pi2", "sig_2pi", "tanh_pi", "sig_4pi", "tanh_2pi"]:
        m = load_metrics(sweep, [c]).get(c)
        if m: all_points.append(("Irradiance", m["complex_overlap"], m["intensity_overlap"], "#2ecc71"))

    # Sweep 05: Hybrid
    sweep = RUNS / "05_fd2nn_hybridloss_roi1024_loss_combo_sweep_claude"
    for c in ["combo1_io_co", "combo2_io_br_ee", "combo3_co_io_br", "combo4_sp_leak_io"]:
        m = load_metrics(sweep, [c]).get(c)
        if m: all_points.append(("Hybrid", m["complex_overlap"], m["intensity_overlap"], "#9b59b6"))

    fig, ax = plt.subplots(figsize=(8, 7))
    for label, co, io, color in all_points:
        ax.scatter(io, co, c=color, s=60, alpha=0.7, edgecolors="black", linewidth=0.5)

    # Legend
    for label, color in [("Complex", "#3498db"), ("Phasor", "#e74c3c"),
                          ("Irradiance", "#2ecc71"), ("Hybrid", "#9b59b6")]:
        ax.scatter([], [], c=color, s=60, label=label, edgecolors="black", linewidth=0.5)

    # Baseline
    ax.scatter(0.973, 0.191, c="black", s=120, marker="*", zorder=5, label="Baseline")

    # Ideal region
    ax.axhspan(0.5, 1.0, xmin=0.5, xmax=1.0, alpha=0.05, color="green")
    ax.text(0.85, 0.55, "Ideal\nRegion", ha="center", fontsize=10, color="green", alpha=0.5)

    ax.set_xlabel("Intensity Overlap (io)", fontsize=12)
    ax.set_ylabel("Complex Overlap (co)", fontsize=12)
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 0.5)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.suptitle("Fig 3: CO vs IO Trade-off — Phase-Only Metalens 한계", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_co_vs_io_tradeoff.png", bbox_inches="tight")
    plt.close()
    print("  fig3_co_vs_io_tradeoff.png")


# ─── Fig 6: Training Curves (§1.1) ─────────────────────────────

def fig6_training_curves():
    """Training curves for best config of each strategy."""
    configs = [
        ("Complex", RUNS / "02_fd2nn_complexloss_roi1024_phase_range_sweep_claude", "tanh_2pi", "#3498db"),
        ("Phasor", RUNS / "03_fd2nn_phasorloss_roi1024_phase_range_sweep_claude", "tanh_2pi", "#e74c3c"),
        ("Irradiance", RUNS / "04_fd2nn_irradianceloss_roi1024_phase_range_sweep_claude", "tanh_2pi", "#2ecc71"),
        ("Hybrid(c3)", RUNS / "05_fd2nn_hybridloss_roi1024_loss_combo_sweep_claude", "combo3_co_io_br", "#9b59b6"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for label, sweep, cfg, color in configs:
        h = load_history(sweep, cfg)
        if not h: continue
        epochs = [e["epoch"] for e in h]
        losses = [e["train_loss"] for e in h]
        axes[0].plot(epochs, losses, color=color, label=label, lw=1.5)

        co_e = [(e["epoch"], e["complex_overlap"]) for e in h if "complex_overlap" in e]
        if co_e: axes[1].plot(*zip(*co_e), "o-", color=color, label=label, ms=4, lw=1.5)

        pr_e = [(e["epoch"], e.get("phase_rmse_rad", e.get("phase_rmse"))) for e in h
                if "complex_overlap" in e and (e.get("phase_rmse_rad") or e.get("phase_rmse"))]
        if pr_e: axes[2].plot(*zip(*pr_e), "s-", color=color, label=label, ms=4, lw=1.5)

    axes[1].axhline(0.191, color="black", ls="--", lw=1, label="Baseline CO")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training Loss")
    axes[1].set(xlabel="Epoch", ylabel="Complex Overlap", title="Val CO (higher=better)")
    axes[2].set(xlabel="Epoch", ylabel="Phase RMSE [rad]", title="Val Phase RMSE (lower=better)")
    for ax in axes: ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle("Fig 6: Training Curves — 4 Loss Strategies", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_training_curves.png", bbox_inches="tight")
    plt.close()
    print("  fig6_training_curves.png")


# ─── Fig 7: Field Comparison 4-strategy (§3) ───────────────────

def fig7_field_comparison():
    """4-row field comparison: Complex / Phasor / Irradiance / Hybrid."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    strategies = [
        ("Complex (co=0.270)", RUNS / "02_fd2nn_complexloss_roi1024_phase_range_sweep_claude", "tanh_2pi"),
        ("Irradiance (io=0.933)", RUNS / "04_fd2nn_irradianceloss_roi1024_phase_range_sweep_claude", "tanh_2pi"),
        ("Hybrid (combo3)", RUNS / "05_fd2nn_hybridloss_roi1024_loss_combo_sweep_claude", "combo3_co_io_br"),
        ("Phasor (failed)", RUNS / "03_fd2nn_phasorloss_roi1024_phase_range_sweep_claude", "tanh_2pi"),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    col_titles = ["|E| Input", "|E| Output", "|E| Target", "|E| Residual"]

    for row, (label, sweep, cfg) in enumerate(strategies):
        fields = load_checkpoint_fields(sweep, cfg, device)
        if fields is None:
            for c in range(4):
                axes[row, c].text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        inp_a = crop(np.abs(fields["input"]))
        pred_a = crop(np.abs(fields["pred"]))
        tgt_a = crop(np.abs(fields["target"]))
        res_a = crop(np.abs(fields["pred"] - fields["target"]))

        vmax = max(inp_a.max(), pred_a.max(), tgt_a.max()) * 0.8
        for c, (data, cmap) in enumerate([(inp_a, "inferno"), (pred_a, "inferno"),
                                            (tgt_a, "inferno"), (res_a, "hot")]):
            vm = vmax if c < 3 else res_a.max()
            im = axes[row, c].imshow(data, cmap=cmap, origin="lower", vmin=0, vmax=vm)
            plt.colorbar(im, ax=axes[row, c], shrink=0.7)
            axes[row, c].set_xticks([]); axes[row, c].set_yticks([])
            if row == 0:
                axes[row, c].set_title(col_titles[c], fontsize=11)
        axes[row, 0].set_ylabel(label, fontsize=10, fontweight="bold")

    fig.suptitle("Fig 7: Field Comparison — 4 Loss Strategies (amplitude, 200×200 crop)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_field_comparison.png", bbox_inches="tight")
    plt.close()
    print("  fig7_field_comparison.png")


# ─── Fig 8: Appendix A evidence (§A.7.4) ───────────────────────

def fig8_amplitude_evidence():
    """3-row comparison: amplitude/intensity/phase for Complex vs Irradiance."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    complex_f = load_checkpoint_fields(
        RUNS / "02_fd2nn_complexloss_roi1024_phase_range_sweep_claude", "tanh_2pi", device)
    irrad_f = load_checkpoint_fields(
        RUNS / "04_fd2nn_irradianceloss_roi1024_phase_range_sweep_claude", "tanh_2pi", device)

    if not complex_f or not irrad_f:
        print("  fig8: missing data, skipped")
        return

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    row_titles = ["Amplitude |E|", "Intensity |E|²", "Phase arg(E)"]
    col_titles = ["Input", "Complex Loss Output", "Irradiance Loss Output", "Target"]

    fields_list = [complex_f["input"], complex_f["pred"], irrad_f["pred"], complex_f["target"]]

    for col, f in enumerate(fields_list):
        amp = crop(np.abs(f))
        inten = crop(np.abs(f)**2)
        phase = crop(np.angle(f))

        im0 = axes[0, col].imshow(amp, cmap="inferno", origin="lower")
        plt.colorbar(im0, ax=axes[0, col], shrink=0.7)

        im1 = axes[1, col].imshow(inten, cmap="inferno", origin="lower")
        plt.colorbar(im1, ax=axes[1, col], shrink=0.7)

        im2 = axes[2, col].imshow(phase, cmap="twilight_shifted", origin="lower", vmin=-np.pi, vmax=np.pi)
        plt.colorbar(im2, ax=axes[2, col], shrink=0.7)

        if col < 4:
            axes[0, col].set_title(col_titles[col], fontsize=11)

    for r, title in enumerate(row_titles):
        axes[r, 0].set_ylabel(title, fontsize=10, fontweight="bold")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Fig 8: Amplitude / Intensity / Phase — Complex vs Irradiance Loss",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_loss_physics_evidence.png", bbox_inches="tight")
    plt.close()
    print("  fig8_loss_physics_evidence.png")


# ─── Fig 9: Spacing Sweep CO/IO vs z/z_R (Appendix B) ────────

def fig9_spacing_sweep():
    """CO and IO vs z/z_R showing plateau behavior (Appendix B.5)."""
    summary_path = RUNS / "중요_01_fd2nn_spacing_sweep_f10mm_claude" / "sweep_summary.json"
    if not summary_path.exists():
        print("  fig9: sweep_summary.json not found, skipped")
        return

    with open(summary_path) as f:
        data = json.load(f)

    configs = ["spacing_0mm", "spacing_1mm", "spacing_3mm", "spacing_6mm",
               "spacing_12mm", "spacing_25mm", "spacing_50mm"]
    spacings_mm = [0, 1, 3, 6, 12, 25, 50]
    z_R = 11.61  # mm, Rayleigh range for 10-pixel feature
    z_zR = [s / z_R for s in spacings_mm]

    co = [data[c]["complex_overlap"] for c in configs]
    io = [data[c]["intensity_overlap"] for c in configs]
    pr = [data[c]["phase_rmse_rad"] for c in configs]
    baseline_co = data[configs[0]]["baseline_complex_overlap"]
    baseline_io = data[configs[0]]["baseline_intensity_overlap"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: CO vs z/z_R
    ax = axes[0]
    ax.plot(z_zR, co, "o-", color="#3498db", lw=2, ms=8, label="CO (complex overlap)")
    ax.axhline(baseline_co, color="black", ls="--", lw=1, alpha=0.6, label=f"Baseline CO={baseline_co:.3f}")
    ax.axvspan(0.4, 1.1, alpha=0.08, color="green", label="Optimal region")
    best_idx = np.argmax(co)
    ax.annotate(f"Best: {co[best_idx]:.3f}\n({spacings_mm[best_idx]}mm)",
                xy=(z_zR[best_idx], co[best_idx]),
                xytext=(z_zR[best_idx]+0.5, co[best_idx]+0.02),
                arrowprops=dict(arrowstyle="->", color="red"), fontsize=9, color="red", fontweight="bold")
    # Plateau annotation
    ax.annotate("plateau\n(0.30–0.32)",
                xy=(2.5, 0.305), fontsize=8, color="#555", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="#ccc"))
    ax.set_xlabel("z / z_R", fontsize=11)
    ax.set_ylabel("Complex Overlap", fontsize=11)
    ax.set_title("CO vs Spacing (higher=better)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Panel 2: IO vs z/z_R
    ax = axes[1]
    ax.plot(z_zR, io, "s-", color="#2ecc71", lw=2, ms=8, label="IO (intensity overlap)")
    ax.axhline(baseline_io, color="black", ls="--", lw=1, alpha=0.6, label=f"Baseline IO={baseline_io:.3f}")
    ax.set_xlabel("z / z_R", fontsize=11)
    ax.set_ylabel("Intensity Overlap", fontsize=11)
    ax.set_title("IO vs Spacing (higher=better)")
    ax.set_ylim(0.85, 1.0)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)
    # Show delta
    for i, (x, y) in enumerate(zip(z_zR, io)):
        delta = (y - baseline_io) / baseline_io * 100
        ax.text(x, y - 0.005, f"{delta:+.1f}%", ha="center", fontsize=7, color="#555")

    # Panel 3: CO vs IO trade-off by spacing
    ax = axes[2]
    cmap = plt.cm.coolwarm(np.linspace(0, 1, len(configs)))
    for i, (c_co, c_io, c_name, c_mm) in enumerate(zip(co, io, configs, spacings_mm)):
        ax.scatter(c_io, c_co, c=cmap[i:i+1], s=100, edgecolors="black", linewidth=0.5, zorder=3)
        ax.annotate(f"{c_mm}mm", (c_io + 0.002, c_co + 0.003), fontsize=8)
    ax.scatter(baseline_io, baseline_co, c="black", s=120, marker="*", zorder=5, label="Baseline")
    ax.set_xlabel("Intensity Overlap", fontsize=11)
    ax.set_ylabel("Complex Overlap", fontsize=11)
    ax.set_title("CO↔IO Trade-off by Spacing")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    # Arrow showing optimal direction
    ax.annotate("", xy=(0.92, 0.33), xytext=(0.88, 0.28),
                arrowprops=dict(arrowstyle="->", color="green", lw=2))
    ax.text(0.915, 0.335, "ideal", fontsize=8, color="green")

    fig.suptitle("Fig 9: f=10mm Spacing Sweep — CO Plateau & IO Trade-off (Appendix B)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig9_spacing_sweep_f10mm.png", bbox_inches="tight")
    plt.close()
    print("  fig9_spacing_sweep_f10mm.png")


# ─── Fig 10: Phase RMSE & Strehl vs Spacing (Appendix B) ─────

def fig10_phase_strehl_spacing():
    """Phase RMSE and Strehl vs spacing — shows monotone trends (Appendix B.8)."""
    summary_path = RUNS / "중요_01_fd2nn_spacing_sweep_f10mm_claude" / "sweep_summary.json"
    if not summary_path.exists():
        print("  fig10: sweep_summary.json not found, skipped")
        return

    with open(summary_path) as f:
        data = json.load(f)

    configs = ["spacing_0mm", "spacing_1mm", "spacing_3mm", "spacing_6mm",
               "spacing_12mm", "spacing_25mm", "spacing_50mm"]
    spacings_mm = [0, 1, 3, 6, 12, 25, 50]
    z_R = 11.61
    z_zR = [s / z_R for s in spacings_mm]

    pr = [data[c]["phase_rmse_rad"] for c in configs]
    strehl = [data[c]["strehl"] for c in configs]
    amp_rmse = [data[c]["amplitude_rmse"] for c in configs]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Phase RMSE — monotone decrease
    ax = axes[0]
    ax.plot(z_zR, pr, "D-", color="#e74c3c", lw=2, ms=8)
    ax.set_xlabel("z / z_R"); ax.set_ylabel("Phase RMSE [rad]")
    ax.set_title("Phase RMSE vs Spacing (lower=better)")
    ax.grid(True, alpha=0.3)
    for x, y in zip(z_zR, pr):
        ax.text(x, y + 0.02, f"{y:.3f}", ha="center", fontsize=7, color="#555")

    # Strehl — monotone increase
    ax = axes[1]
    ax.plot(z_zR, strehl, "^-", color="#9b59b6", lw=2, ms=8)
    ax.set_xlabel("z / z_R"); ax.set_ylabel("Strehl Ratio")
    ax.set_title("Strehl vs Spacing (higher=focus)")
    ax.grid(True, alpha=0.3)
    for x, y in zip(z_zR, strehl):
        ax.text(x, y + 0.03, f"{y:.2f}", ha="center", fontsize=7, color="#555")

    # Amplitude RMSE — flat (phase-only limitation evidence)
    ax = axes[2]
    ax.plot(z_zR, amp_rmse, "o-", color="#f39c12", lw=2, ms=8)
    ax.set_xlabel("z / z_R"); ax.set_ylabel("Amplitude RMSE")
    ax.set_title("Amp RMSE vs Spacing (≈flat → phase-only limit)")
    ax.set_ylim(0.07, 0.11)
    ax.grid(True, alpha=0.3)
    ax.axhspan(min(amp_rmse)-0.005, max(amp_rmse)+0.005, alpha=0.1, color="orange")
    ax.text(2.0, 0.105, "phase-only: no amp control\n→ nearly flat across spacings",
            fontsize=8, color="#555", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="#ccc"))

    fig.suptitle("Fig 10: Phase RMSE / Strehl / Amp RMSE vs Spacing (Appendix B)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig10_phase_strehl_spacing.png", bbox_inches="tight")
    plt.close()
    print("  fig10_phase_strehl_spacing.png")


def main():
    print(f"Generating report figures -> {FIG_DIR}/\n")
    fig1_summary_bars()
    fig2_phase_range()
    fig3_tradeoff_scatter()
    fig6_training_curves()
    fig7_field_comparison()
    fig8_amplitude_evidence()
    fig9_spacing_sweep()
    fig10_phase_strehl_spacing()
    print(f"\nDone. All figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()