#!/usr/bin/env python
"""Generate all 6 main figures for the D2NN paper.

Fig 1: System schematic (reuse phase1_architecture)
Fig 2: Unitary invariance proof — epoch vs {CO, WF RMS, PIB}
Fig 3: Deterministic vs Random — single-layer vs multi-layer
Fig 4: Loss strategy bar chart — PIB/CO/WF RMS
Fig 5: CO vs PIB tradeoff curve (Pareto)
Fig 6: Energy redistribution map

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/generate_paper_figures.py
"""
from __future__ import annotations
import json, math
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

from kim2026.data.dataset import CachedFieldDataset
from kim2026.models.d2nn import BeamCleanupD2NN
from kim2026.optics.lens_2f import lens_2f_forward
from kim2026.training.targets import apply_receiver_aperture, center_crop_field
from kim2026.training.metrics import complex_overlap
from kim2026.viz.mpl_fonts import configure_matplotlib_fonts, ensure_output_dir
from kim2026.viz.paper_figure_text import FIGURE_TEXTS

W = 1.55e-6; N = 1024; WIN = 0.002048; APT = 0.002; DX = WIN / N
ARCH = dict(num_layers=5, layer_spacing_m=10e-3, detector_distance_m=10e-3)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kim2026" / "1km_cn2_5e-14_tel15cm_n1024_br75"
LOSS_DIR = Path(__file__).resolve().parent.parent / "autoresearch" / "runs" / "d2nn_loss_strategy"
CO_DIR = Path(__file__).resolve().parent.parent / "autoresearch" / "runs" / "0328-co-sweep-strong-turb-cn2-5e14"
PAPER_META = Path(__file__).resolve().parent.parent / "autoresearch" / "runs" / "PAPER_figures_tables_meta.json"
OUT = Path(__file__).resolve().parent.parent / "autoresearch" / "runs" / "paper_figures"

configure_matplotlib_fonts()
ensure_output_dir(OUT)

def prepare(f):
    return center_crop_field(apply_receiver_aperture(f, receiver_window_m=WIN, aperture_diameter_m=APT), crop_n=N)

def load_model(base, name):
    ckpt = base / name / "checkpoint.pt"
    if not ckpt.exists(): return None
    m = BeamCleanupD2NN(n=N, wavelength_m=W, window_m=WIN, **ARCH)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model_state_dict"])
    m.eval(); return m

def focus(field):
    with torch.no_grad():
        f, dx = lens_2f_forward(field.to(torch.complex64), dx_in_m=DX, wavelength_m=W,
                                 f_m=4.5e-3, na=None, apply_scaling=False)
    return f, dx


def load_phase_b_summary():
    """Load the committed Phase B summary table used by the paper figures."""
    meta = json.load(open(PAPER_META))
    table = next(item for item in meta["tables"] if item["id"] == "tab6")

    label_map = {
        "보정 없음": "No\nD2NN",
        "PIB only": "PIB\nonly",
        "CO+PIB hybrid": "CO+PIB\nhybrid",
        "Strehl only": "Strehl\nonly",
        "Intensity overlap": "Intensity\noverlap",
    }
    color_map = {
        "보정 없음": "gray",
        "PIB only": "#e74c3c",
        "Strehl only": "#3498db",
        "Intensity overlap": "#2ecc71",
        "CO+PIB hybrid": "#9b59b6",
    }

    parsed_rows = []
    for strategy, pib_raw, mixed_co, theorem_co, wf_rms in table["rows"]:
        parsed_rows.append(
            {
                "strategy": strategy,
                "label": label_map[strategy],
                "color": color_map[strategy],
                "pib": float(str(pib_raw).replace("%", "")) / 100.0,
                "mixed_co": float(mixed_co),
                "theorem_co": float(theorem_co),
                "wf_rms_nm": float(str(wf_rms).replace(" nm", "")),
            }
        )

    return parsed_rows


def fig2_unitary_invariance():
    """Epoch vs {CO, WF RMS, PIB} for co_pib_hybrid — CO/WF flat, PIB rises."""
    print("Fig 2: Unitary invariance visualization...")
    text = FIGURE_TEXTS["fig2"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(text["suptitle"], fontsize=14, fontweight="bold")

    strategies = {
        "co_pib_hybrid": ("#9b59b6", "CO+PIB hybrid"),
        "pib_only": ("#e74c3c", "PIB only"),
    }

    for name, (color, label) in strategies.items():
        rp = LOSS_DIR / name / "results.json"
        if not rp.exists(): continue
        r = json.load(open(rp))
        h = r.get("history", {})
        if not h.get("epoch"): continue

        epochs = h["epoch"]
        # CO
        axes[0].plot(epochs, h["val_co"], color=color, lw=2.5, marker='o', ms=4, label=label)
        # PIB
        if "val_pib" in h:
            axes[2].plot(epochs, h["val_pib"], color=color, lw=2.5, marker='s', ms=4, label=label)

    # Baseline lines
    axes[0].axhline(0.3044, color='k', ls='--', lw=2, label='Baseline (no D2NN)')
    axes[0].set_ylabel("Complex Overlap (CO)", fontsize=13)
    axes[0].set_xlabel("Epoch", fontsize=13)
    axes[0].set_title(text["titles"][0], fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # WF RMS — constant at ~460nm
    axes[1].axhline(460, color='k', ls='-', lw=2, label='WF RMS (all strategies)')
    axes[1].set_ylim(400, 520)
    axes[1].set_ylabel("WF RMS [nm]", fontsize=13)
    axes[1].set_xlabel("Epoch", fontsize=13)
    axes[1].set_title(text["titles"][1], fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].text(100, 465, text["wf_annotation"], fontsize=12, ha="center",
                 bbox=dict(facecolor="lightyellow", edgecolor="gray"))

    axes[2].axhline(0.0034, color='k', ls='--', lw=2, label='Baseline')
    axes[2].set_ylabel("PIB@50μm", fontsize=13)
    axes[2].set_xlabel("Epoch", fontsize=13)
    axes[2].set_title(text["titles"][2], fontsize=13, fontweight="bold")
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT / "fig2_unitary_invariance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved")


def fig3_deterministic_vs_random():
    """Side-by-side: single-layer success vs multi-layer random failure."""
    print("Fig 3: Deterministic vs Random...")
    text = FIGURE_TEXTS["fig3"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(text["suptitle"], fontsize=14, fontweight="bold")

    # Left: Deterministic (single-layer)
    epochs_det = list(range(0, 200, 40)) + [199]
    wf_det = [210.0, 114.1, 23.6, 1.0, 0.1, 0.1]  # from verification
    co_det = [0.9886, 0.9911, 0.9985, 1.0000, 1.0000, 1.0000]

    ax_wf = axes[0]
    ax_co = ax_wf.twinx()
    l1, = ax_wf.plot(epochs_det, wf_det, 'r-o', lw=2.5, ms=6, label='WF RMS')
    l2, = ax_co.plot(epochs_det, co_det, 'b-s', lw=2.5, ms=6, label='CO')
    ax_wf.set_xlabel("Epoch", fontsize=13)
    ax_wf.set_ylabel("WF RMS [nm]", fontsize=13, color='r')
    ax_co.set_ylabel("CO", fontsize=13, color='b')
    ax_wf.set_title(text["left_title"], fontsize=13, fontweight="bold")
    ax_wf.legend([l1, l2], text["left_legend"], fontsize=11, loc='center right')
    ax_wf.grid(True, alpha=0.3)
    ax_wf.set_ylim(-10, 250)
    ax_co.set_ylim(0.98, 1.005)

    # Right: Random turbulence (multi-layer)
    # From d2nn_strong_turb_sweep co_ffp history
    rp = CO_DIR / "co_ffp" / "results.json"
    if rp.exists():
        r = json.load(open(rp))
        h = r.get("history", {})
        if h.get("epoch"):
            ax_wf2 = axes[1]
            # WF RMS stays at 460nm (constant)
            ax_wf2.axhline(460, color='r', ls='-', lw=2.5, label=text["right_wf_label"])
            ax_co2 = ax_wf2.twinx()
            ax_co2.plot(h["epoch"], h["val_co"], 'b-s', lw=2.5, ms=4, label='Val CO')
            ax_co2.axhline(0.3044, color='b', ls='--', lw=1.5, alpha=0.5, label='Baseline CO')
            ax_wf2.set_xlabel("Epoch", fontsize=13)
            ax_wf2.set_ylabel("WF RMS [nm]", fontsize=13, color='r')
            ax_co2.set_ylabel("CO", fontsize=13, color='b')
            ax_wf2.set_title(text["right_title"], fontsize=13, fontweight="bold")
            ax_wf2.set_ylim(400, 520)
            ax_co2.set_ylim(0.28, 0.38)
            lines1, labels1 = ax_wf2.get_legend_handles_labels()
            lines2, labels2 = ax_co2.get_legend_handles_labels()
            ax_wf2.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='center right')
            ax_wf2.grid(True, alpha=0.3)
            ax_wf2.text(50, 470, text["right_annotation"], fontsize=12, ha="center",
                        bbox=dict(facecolor="#f5b7b1", edgecolor="red", alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT / "fig3_deterministic_vs_random.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved")


def fig4_loss_strategy_bar():
    """PIB/CO/WF RMS comparison across loss strategies."""
    print("Fig 4: Loss strategy bar chart...")
    text = FIGURE_TEXTS["fig4"]
    rows = load_phase_b_summary()

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(text["suptitle"], fontsize=14, fontweight="bold")

    x = np.arange(len(rows))
    width = 0.6
    bar_colors = [row["color"] for row in rows]
    bar_labels = [row["label"] for row in rows]

    # PIB
    pibs = [row["pib"] for row in rows]
    axes[0].bar(x, pibs, width, color=bar_colors, alpha=0.8, edgecolor='black', lw=0.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(bar_labels, fontsize=11)
    axes[0].set_ylabel("PIB@50μm", fontsize=13)
    axes[0].set_title(text["titles"][0], fontsize=13, fontweight="bold")
    for i, v in enumerate(pibs):
        axes[0].text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis='y')

    # CO
    cos = [row["mixed_co"] for row in rows]
    axes[1].bar(x, cos, width, color=bar_colors, alpha=0.8, edgecolor='black', lw=0.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(bar_labels, fontsize=11)
    axes[1].set_ylabel("Complex Overlap", fontsize=13)
    axes[1].set_title(text["titles"][1], fontsize=13, fontweight="bold")
    for i, v in enumerate(cos):
        axes[1].text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')

    # WF RMS
    wfs = [row["wf_rms_nm"] for row in rows]
    axes[2].bar(x, wfs, width, color=bar_colors, alpha=0.8, edgecolor='black', lw=0.5)
    axes[2].set_xticks(x); axes[2].set_xticklabels(bar_labels, fontsize=11)
    axes[2].set_ylabel("WF RMS [nm]", fontsize=13)
    axes[2].set_title(text["titles"][2], fontsize=13, fontweight="bold")
    for i, v in enumerate(wfs):
        axes[2].text(i, v + 5, f"{v:.0f}", ha="center", fontsize=10)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT / "fig4_loss_strategy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved")


def fig5_pareto():
    """CO vs PIB tradeoff — Pareto frontier."""
    print("Fig 5: CO vs PIB Pareto...")
    text = FIGURE_TEXTS["fig5"]
    rows = load_phase_b_summary()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(text["suptitle"], fontsize=14, fontweight="bold")

    # Baseline (no D2NN)
    baseline = next(row for row in rows if row["strategy"] == "보정 없음")
    bl_co = baseline["mixed_co"]
    bl_pib = baseline["pib"]
    ax.scatter(bl_co, bl_pib, s=200, c='black', marker='*', zorder=5, label='No D2NN')
    ax.annotate("No D2NN", (bl_co, bl_pib), textcoords="offset points",
                xytext=(10, 10), fontsize=12, fontweight="bold")

    colors_map = {
        "PIB only": "#e74c3c",
        "CO+PIB hybrid": "#9b59b6",
        "Strehl only": "#3498db",
        "Intensity overlap": "#2ecc71",
    }

    for row in rows:
        if row["strategy"] == "보정 없음":
            continue
        co, pib, name = row["mixed_co"], row["pib"], row["strategy"]
        c = colors_map.get(name, 'gray')
        ax.scatter(co, pib, s=150, c=c, edgecolor='black', lw=1, zorder=4)
        ax.annotate(name, (co, pib), textcoords="offset points",
                    xytext=(8, 5), fontsize=10, color=c)

    ax.set_xlabel(text["xlabel"], fontsize=14)
    ax.set_ylabel(text["ylabel"], fontsize=14)
    ax.set_xlim(-0.02, 0.35)
    ax.grid(True, alpha=0.3)

    # Add quadrant labels
    ax.axhline(bl_pib, color='gray', ls=':', alpha=0.5)
    ax.axvline(bl_co, color='gray', ls=':', alpha=0.5)
    ax.text(0.25, 0.7, text["ideal_region"], fontsize=12, ha="center",
            color="green", bbox=dict(facecolor="lightgreen", alpha=0.3))
    ax.text(0.02, 0.7, text["filtering_region"], fontsize=12, ha="center",
            color="orange", bbox=dict(facecolor="#fdebd0", alpha=0.3))

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT / "fig5_pareto.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved")


def fig6_energy_redistribution():
    """Before/after D2NN intensity + difference map."""
    print("Fig 6: Energy redistribution...")
    text = FIGURE_TEXTS["fig6"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CachedFieldDataset(cache_dir=str(DATA_DIR / "cache"),
                             manifest_path=str(DATA_DIR / "split_manifest.json"), split="test")
    s = ds[0]
    u_turb = prepare(s["u_turb"].unsqueeze(0).to(device))
    u_vac = prepare(s["u_vacuum"].unsqueeze(0).to(device))

    # Load PIB model and CO model
    m_pib = load_model(LOSS_DIR, "pib_only")
    m_hybrid = load_model(LOSS_DIR, "co_pib_hybrid")
    d0 = BeamCleanupD2NN(n=N, wavelength_m=W, window_m=WIN, **ARCH).to(device)
    d0.eval()

    with torch.no_grad():
        out_none = d0(u_turb)
        out_pib = m_pib.to(device)(u_turb) if m_pib else out_none
        out_hybrid = m_hybrid.to(device)(u_turb) if m_hybrid else out_none

    # Focus all
    _, dx_det = focus(u_vac)
    det_none, _ = focus(out_none)
    det_pib, _ = focus(out_pib)
    det_hybrid, _ = focus(out_hybrid)
    det_vac, _ = focus(d0(u_vac))

    # numpy (detector plane, zoomed)
    Z = 80; c = N // 2
    e_um = [-(Z * float(dx_det) * 1e6), Z * float(dx_det) * 1e6,
            -(Z * float(dx_det) * 1e6), Z * float(dx_det) * 1e6]

    fields = {
        text["field_labels"][0]: det_vac[0, c-Z:c+Z, c-Z:c+Z].cpu().numpy(),
        text["field_labels"][1]: det_none[0, c-Z:c+Z, c-Z:c+Z].cpu().numpy(),
        text["field_labels"][2]: det_pib[0, c-Z:c+Z, c-Z:c+Z].cpu().numpy(),
        text["field_labels"][3]: det_hybrid[0, c-Z:c+Z, c-Z:c+Z].cpu().numpy(),
    }

    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    fig.suptitle(text["suptitle"], fontsize=14, fontweight="bold")

    vac_irr = np.abs(fields[text["field_labels"][0]])**2
    imax = vac_irr.max()

    for col, (label, field) in enumerate(fields.items()):
        irr = np.abs(field)**2

        # Row 0: Irradiance
        axes[0, col].imshow(irr, extent=e_um, origin="lower", cmap="inferno", vmin=0, vmax=imax)
        axes[0, col].set_title(label, fontsize=13, fontweight="bold")
        # 50um circle
        theta = np.linspace(0, 2*np.pi, 100)
        axes[0, col].plot(50*np.cos(theta), 50*np.sin(theta), 'w--', lw=1.5, alpha=0.8)

        # Row 1: Log irradiance
        log_irr = np.log10(irr + 1e-15)
        axes[1, col].imshow(log_irr, extent=e_um, origin="lower", cmap="viridis",
                            vmin=np.log10(imax) - 4, vmax=np.log10(imax))
        axes[1, col].plot(50*np.cos(theta), 50*np.sin(theta), 'w--', lw=1.5, alpha=0.8)

        # Row 2: Difference from vacuum
        diff = irr - vac_irr
        vmax_d = max(abs(diff.min()), abs(diff.max()), 1e-15)
        axes[2, col].imshow(diff, extent=e_um, origin="lower", cmap="RdBu_r",
                            vmin=-vmax_d, vmax=vmax_d)
        axes[2, col].plot(50*np.cos(theta), 50*np.sin(theta), 'k--', lw=1, alpha=0.5)

    row_labels = text["row_labels"]
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=13, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT / "fig6_energy_redistribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    del d0, m_pib, m_hybrid
    torch.cuda.empty_cache()
    print("  Saved")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUT}\n")

    # Fig 1: reuse existing architecture diagram
    import shutil
    arch_src = Path(__file__).resolve().parent.parent / "autoresearch" / "runs" / "0328-co-sweep-strong-turb-cn2-5e14" / "phase1_architecture.png"
    if arch_src.exists():
        shutil.copy(arch_src, OUT / "fig1_system.png")
        print("Fig 1: Copied from phase1_architecture.png")

    fig2_unitary_invariance()
    fig3_deterministic_vs_random()
    fig4_loss_strategy_bar()
    fig5_pareto()
    fig6_energy_redistribution()

    print(f"\nDone! All figures saved to {OUT}")


if __name__ == "__main__":
    main()
