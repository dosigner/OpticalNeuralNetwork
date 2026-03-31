#!/usr/bin/env python3
"""Generate theorem verification figures for paper revision.

Figure 1: CO comparison — CO(U_t,U_v) vs CO(HU_t,HU_v) vs CO(HU_t,U_v_det)
Figure 2: L2 distance comparison — ||U_t-U_v|| vs ||HU_t-HU_v||
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kim2026.models.d2nn import BeamCleanupD2NN
from kim2026.data.dataset import CachedFieldDataset
from kim2026.training.metrics import complex_overlap
from kim2026.training.targets import apply_receiver_aperture, make_detector_plane_target

# ── Config ──
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "kim2026" / "1km_cn2_5e-14_tel15cm_n1024_br75"
CKPT_DIR = Path(__file__).resolve().parent / "d2nn_loss_strategy"
STRATEGIES = ["pib_only", "co_pib_hybrid", "strehl_only", "intensity_overlap"]
STRATEGY_LABELS = ["PIB only", "CO+PIB hybrid", "Strehl only", "Intensity overlap"]

WAVELENGTH_M = 1.55e-6
WINDOW_M = 2.048e-3
APERTURE_M = 2.0e-3
NUM_LAYERS = 5
LAYER_SPACING_M = 10e-3
DETECTOR_DISTANCE_M = 10e-3
TOTAL_DISTANCE_M = (NUM_LAYERS - 1) * LAYER_SPACING_M + DETECTOR_DISTANCE_M

OUT_DIR = Path(__file__).resolve().parent / "paper_figures"
OUT_DIR.mkdir(exist_ok=True)


def load_model(ckpt_path, device):
    model = BeamCleanupD2NN(
        n=1024, wavelength_m=WAVELENGTH_M, window_m=WINDOW_M,
        num_layers=NUM_LAYERS, layer_spacing_m=LAYER_SPACING_M,
        detector_distance_m=DETECTOR_DISTANCE_M,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    model.eval()
    return model


def compute_co(a, b):
    if a.ndim == 2: a = a.unsqueeze(0)
    if b.ndim == 2: b = b.unsqueeze(0)
    return complex_overlap(a, b).item()


def compute_l2(a, b):
    return torch.linalg.vector_norm((a - b).flatten()).item()


def gather_data(device):
    dataset = CachedFieldDataset(
        cache_dir=str(DATA_DIR / "cache"),
        manifest_path=str(DATA_DIR / "split_manifest.json"),
        split="test",
    )

    results = {}
    for sname in STRATEGIES:
        ckpt_path = CKPT_DIR / sname / "checkpoint.pt"
        if not ckpt_path.exists():
            continue
        model = load_model(ckpt_path, device)

        co_input, co_both, co_mixed = [], [], []
        l2_input, l2_both = [], []

        with torch.no_grad():
            for i in range(len(dataset)):
                s = dataset[i]
                u_t = s["u_turb"].unsqueeze(0).to(device)
                u_v = s["u_vacuum"].unsqueeze(0).to(device)

                u_t_ap = apply_receiver_aperture(u_t, receiver_window_m=WINDOW_M, aperture_diameter_m=APERTURE_M)
                u_v_ap = apply_receiver_aperture(u_v, receiver_window_m=WINDOW_M, aperture_diameter_m=APERTURE_M)

                hu_t = model(u_t_ap)
                hu_v = model(u_v_ap)

                u_v_det = make_detector_plane_target(
                    u_v, wavelength_m=WAVELENGTH_M, receiver_window_m=WINDOW_M,
                    aperture_diameter_m=APERTURE_M, total_distance_m=TOTAL_DISTANCE_M,
                    complex_mode=True,
                )

                co_input.append(compute_co(u_t_ap, u_v_ap))
                co_both.append(compute_co(hu_t, hu_v))
                co_mixed.append(compute_co(hu_t, u_v_det))
                l2_input.append(compute_l2(u_t_ap, u_v_ap))
                l2_both.append(compute_l2(hu_t, hu_v))

        results[sname] = {
            "co_input": np.array(co_input),
            "co_both": np.array(co_both),
            "co_mixed": np.array(co_mixed),
            "l2_input": np.array(l2_input),
            "l2_both": np.array(l2_both),
        }
    return results


def plot_figure1(results):
    """Main figure: 3-panel comparison of CO metrics across strategies."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    strategies_to_plot = [s for s in STRATEGIES if s in results]
    labels = [STRATEGY_LABELS[STRATEGIES.index(s)] for s in strategies_to_plot]

    # Panel A: CO(U_t,U_v) vs CO(HU_t,HU_v) scatter — Theorem 1 verification
    ax = axes[0]
    # All strategies give same result, use pib_only as representative
    r = results[strategies_to_plot[0]]
    ax.scatter(r["co_input"], r["co_both"], s=30, alpha=0.7, c="#2196F3", edgecolors="k", linewidths=0.3)
    lims = [0, max(r["co_input"].max(), r["co_both"].max()) * 1.1]
    ax.plot(lims, lims, "k--", lw=1, label="y = x (exact preservation)")
    ax.set_xlabel("CO(U_turb, U_vac) — input plane", fontsize=11)
    ax.set_ylabel("CO(HU_turb, HU_vac) — both through D2NN", fontsize=11)
    ax.set_title("(a) Theorem 1 Verification\n(identical for all strategies)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    # Annotate max deviation
    diffs = np.abs(r["co_input"] - r["co_both"])
    ax.text(0.05, 0.92, f"|diff| max = {diffs.max():.1e}", transform=ax.transAxes,
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="honeydew", edgecolor="green"))

    # Panel B: CO(HU_t, U_v_det) — mixed (what paper measured) per strategy
    ax = axes[1]
    x = np.arange(len(strategies_to_plot))
    means = [results[s]["co_mixed"].mean() for s in strategies_to_plot]
    stds = [results[s]["co_mixed"].std() for s in strategies_to_plot]
    baseline_co = results[strategies_to_plot[0]]["co_input"].mean()

    bars = ax.bar(x, means, yerr=stds, width=0.6, color=["#F44336", "#FF9800", "#9C27B0", "#4CAF50"],
                  edgecolor="k", linewidth=0.5, capsize=5, alpha=0.85)
    ax.axhline(baseline_co, color="k", ls="--", lw=1.5, label=f"Baseline CO = {baseline_co:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("CO(HU_turb, U_vac_det) — mixed", fontsize=11)
    ax.set_title("(b) Mixed CO (paper's metric)\nNOT Theorem 1 quantity", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, baseline_co * 1.3)

    # Panel C: Side-by-side comparison of the three CO types for pib_only
    ax = axes[2]
    r = results["pib_only"]
    categories = [
        "CO(U_t, U_v)\nbaseline",
        "CO(HU_t, HU_v)\nThm 1 (both D2NN)",
        "CO(HU_t, U_v_det)\nmixed (paper)"
    ]
    vals = [r["co_input"].mean(), r["co_both"].mean(), r["co_mixed"].mean()]
    errs = [r["co_input"].std(), r["co_both"].std(), r["co_mixed"].std()]
    colors = ["#2196F3", "#4CAF50", "#F44336"]

    bars = ax.bar(range(3), vals, yerr=errs, width=0.6, color=colors,
                  edgecolor="k", linewidth=0.5, capsize=5, alpha=0.85)
    ax.set_xticks(range(3))
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Complex Overlap (CO)", fontsize=11)
    ax.set_title("(c) PIB-only strategy:\nThree CO definitions compared", fontsize=12, fontweight="bold")

    # Annotate values
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(i, v + e + 0.015, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = OUT_DIR / "theorem1_co_verification.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_figure2(results):
    """L2 distance preservation (Theorem 2)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Scatter plot L2 input vs L2 D2NN
    ax = axes[0]
    r = results["pib_only"]
    ax.scatter(r["l2_input"], r["l2_both"], s=30, alpha=0.7, c="#FF9800", edgecolors="k", linewidths=0.3)
    lims = [r["l2_input"].min() * 0.95, r["l2_input"].max() * 1.05]
    ax.plot(lims, lims, "k--", lw=1, label="y = x (exact preservation)")
    ax.set_xlabel("||U_turb - U_vac||_2 — input plane", fontsize=11)
    ax.set_ylabel("||HU_turb - HU_vac||_2 — both through D2NN", fontsize=11)
    ax.set_title("(a) Theorem 2: L2 Distance Preservation\n(PIB-only, identical for all strategies)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")

    diffs = np.abs(r["l2_input"] - r["l2_both"])
    rel_diffs = diffs / r["l2_input"]
    ax.text(0.05, 0.92, f"|diff|/L2 max = {rel_diffs.max():.1e}", transform=ax.transAxes,
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="honeydew", edgecolor="green"))

    # Panel B: Residuals histogram
    ax = axes[1]
    # Collect residuals from all strategies
    all_rel = []
    for sname in STRATEGIES:
        if sname not in results:
            continue
        r = results[sname]
        rd = np.abs(r["l2_input"] - r["l2_both"]) / r["l2_input"]
        all_rel.extend(rd.tolist())

    ax.hist(all_rel, bins=30, color="#FF9800", edgecolor="k", alpha=0.8)
    ax.set_xlabel("Relative L2 deviation |L2_in - L2_D2NN| / L2_in", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("(b) L2 Preservation Error Distribution\n(all 4 strategies, 200 samples)", fontsize=12, fontweight="bold")
    ax.axvline(np.mean(all_rel), color="red", ls="--", lw=1.5, label=f"mean = {np.mean(all_rel):.1e}")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = OUT_DIR / "theorem2_l2_verification.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_summary_table(results):
    """Summary table figure."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")

    col_labels = [
        "Strategy",
        "CO(U_t,U_v)\nbaseline",
        "CO(HU_t,HU_v)\nThm 1",
        "|diff|\nThm 1",
        "CO(HU_t,U_v)\nmixed (paper)",
        "||U_t-U_v||\nbaseline",
        "||HU_t-HU_v||\nThm 2",
        "|diff|\nThm 2",
    ]

    table_data = []
    for sname, slabel in zip(STRATEGIES, STRATEGY_LABELS):
        if sname not in results:
            continue
        r = results[sname]
        table_data.append([
            slabel,
            f"{r['co_input'].mean():.6f}",
            f"{r['co_both'].mean():.6f}",
            f"{np.abs(r['co_input'] - r['co_both']).mean():.1e}",
            f"{r['co_mixed'].mean():.6f}",
            f"{r['l2_input'].mean():.4f}",
            f"{r['l2_both'].mean():.4f}",
            f"{np.abs(r['l2_input'] - r['l2_both']).mean():.1e}",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    # Color header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#E3F2FD")
        table[0, j].set_text_props(fontweight="bold")

    # Color Thm 1 diff column green
    for i in range(1, len(table_data) + 1):
        table[i, 3].set_facecolor("#E8F5E9")  # green = preserved
        table[i, 7].set_facecolor("#E8F5E9")
        table[i, 4].set_facecolor("#FFEBEE")  # red = changed

    ax.set_title(
        "Theorem Verification Summary: CO(HU_t,HU_v) = CO(U_t,U_v) [Thm 1], ||HU_t-HU_v|| = ||U_t-U_v|| [Thm 2]\n"
        "Green = preserved (10^-7 ~ 10^-5 error), Red = mixed metric that changes significantly",
        fontsize=11, fontweight="bold", pad=20,
    )

    out = OUT_DIR / "theorem_verification_table.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Gathering data (4 strategies x 50 test samples)...")

    results = gather_data(device)

    print("\nGenerating figures...")
    plot_figure1(results)
    plot_figure2(results)
    plot_summary_table(results)
    print("Done!")


if __name__ == "__main__":
    main()
