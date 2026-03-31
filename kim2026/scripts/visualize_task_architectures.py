#!/usr/bin/env python
"""Architecture diagrams for the 3 active/completed tasks.

Generates:
  single_case_task1/architecture_true_co_only.png
  single_case_task2/architecture_d2nn_sweep_telescope.png
  single_case_task3/architecture_loss_sweep_telescope.png

Usage:
    cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/visualize_task_architectures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

RUNS_ROOT = Path("/root/dj/D2NN/kim2026/autoresearch/runs")

# ══════════════════════════════════════════════════════════════════
# Common drawing utilities
# ══════════════════════════════════════════════════════════════════

C_LENS = "#3498db"
C_MASK_FD = "#e74c3c"
C_MASK_D2 = "#2ecc71"
C_BEAM_FILL = "#f39c1240"
C_BEAM_EDGE = "#e67e22"
C_FOCUS = "#16a085"
C_DET = "#8e44ad"


def draw_fd2nn_layout(ax, title, highlights=None):
    """FD2NN: Input→Lens1(f=25mm)→5 Masks(Fourier plane)→Lens2(f=25mm)→Focus→APD."""
    ax.set_xlim(-2, 85)
    ax.set_ylim(-14, 16)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)

    input_x = 0
    lens1_x = 5
    fourier_start = lens1_x + 25
    mask_spacing = 2.5
    fourier_end = fourier_start + 4 * mask_spacing
    lens2_x = fourier_end + 25
    output_x = lens2_x + 3
    focus_x = output_x + 5
    det_x = focus_x + 5
    bh = 5

    beam_x = [input_x, lens1_x, fourier_start, fourier_end, lens2_x, output_x, focus_x, det_x]
    beam_top = [bh, bh, 0.4, 0.4, bh, bh, bh, 0.3]
    beam_bot = [-bh, -bh, -0.4, -0.4, -bh, -bh, -bh, -0.3]
    ax.fill_between(beam_x, beam_top, beam_bot, color=C_BEAM_FILL, edgecolor=C_BEAM_EDGE, lw=1)

    ax.annotate("Input\n2mm", xy=(input_x, -8), fontsize=12, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ecf0f1", edgecolor="gray"))

    ax.plot([lens1_x, lens1_x], [-8, 8], color=C_LENS, lw=3)
    ax.annotate("Lens1\nf=25mm", xy=(lens1_x, -11), fontsize=11, ha="center", color=C_LENS)

    for i in range(5):
        mx = fourier_start + i * mask_spacing
        ax.add_patch(patches.Rectangle((mx - 0.3, -7), 0.6, 14,
                     facecolor=C_MASK_FD, alpha=0.25, edgecolor=C_MASK_FD, lw=1.5))
        ax.text(mx, 8, f"M{i}", fontsize=11, ha="center", color=C_MASK_FD)

    ax.annotate("Fourier Plane\n19.4mm, dx=18.9\u03bcm", xy=(fourier_start + 2 * mask_spacing, -11),
                fontsize=11, ha="center", color="#9b59b6",
                bbox=dict(boxstyle="round", facecolor="#9b59b6", alpha=0.1))

    ax.plot([lens2_x, lens2_x], [-8, 8], color=C_LENS, lw=3)
    ax.annotate("Lens2\nf=25mm", xy=(lens2_x, -11), fontsize=11, ha="center", color=C_LENS)

    ax.plot([focus_x, focus_x], [-8, 8], color=C_FOCUS, lw=3)
    ax.annotate("Focus\nf=4.5mm", xy=(focus_x, -11), fontsize=11, ha="center", color=C_FOCUS)

    ax.add_patch(patches.Rectangle((det_x - 0.5, -4), 1.0, 8,
                 facecolor=C_DET, alpha=0.4, edgecolor=C_DET, lw=2))
    ax.annotate("APD/MMF", xy=(det_x, -8), fontsize=11, ha="center", color=C_DET, fontweight="bold")

    ax.annotate("", xy=(input_x, 14), xytext=(det_x, 14),
                arrowprops=dict(arrowstyle="<->", color="gray"))
    ax.text((input_x + det_x) / 2, 15, "~80mm total", fontsize=12, ha="center", color="gray")

    ax.annotate("", xy=(lens1_x, 12), xytext=(fourier_start, 12),
                arrowprops=dict(arrowstyle="<->", color=C_LENS))
    ax.text((lens1_x + fourier_start) / 2, 12.8, "25mm", fontsize=11, ha="center", color=C_LENS)

    # Spacing between masks
    ax.annotate("", xy=(fourier_start, 10), xytext=(fourier_start + mask_spacing, 10),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=0.8))
    ax.text(fourier_start + mask_spacing / 2, 10.7, "5mm", fontsize=10, ha="center", color="gray")

    if highlights:
        for h in highlights:
            ax.text(h["x"], h["y"], h["text"], fontsize=h.get("fs", 13),
                    ha="center", color=h.get("color", "red"), fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor=h.get("bg", "lightyellow"),
                              edgecolor=h.get("color", "red"), alpha=0.9))

    ax.set_xlabel("Optical axis [mm]", fontsize=11)
    ax.axhline(0, color="gray", lw=0.3, ls=":")
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=11)


def draw_d2nn_layout(ax, title, highlights=None):
    """D2NN: Input→5 Masks(real-space, 10mm spacing)→Focus→APD."""
    ax.set_xlim(-2, 85)
    ax.set_ylim(-14, 16)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)

    input_x = 0
    mask_spacing = 8
    last_mask_x = 5 + 4 * mask_spacing
    d2nn_out_x = last_mask_x + 8
    focus_x = d2nn_out_x + 5
    det_x = focus_x + 5
    bh = 5

    xs = [input_x]
    tops = [bh]
    bots = [-bh]
    for i in range(5):
        mx = 5 + i * mask_spacing
        xs.extend([mx - 0.3, mx + 0.3])
        spread = 0.3 * i
        tops.extend([bh + spread, bh + spread])
        bots.extend([-bh - spread, -bh - spread])
    xs.extend([d2nn_out_x, focus_x, det_x])
    tops.extend([bh + 1.5, bh + 1.5, 0.3])
    bots.extend([-bh - 1.5, -bh - 1.5, -0.3])
    ax.fill_between(xs, tops, bots, color=C_BEAM_FILL, edgecolor=C_BEAM_EDGE, lw=1)

    ax.annotate("Input\n2mm", xy=(input_x, -8), fontsize=12, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ecf0f1", edgecolor="gray"))

    for i in range(5):
        mx = 5 + i * mask_spacing
        ax.add_patch(patches.Rectangle((mx - 0.3, -7), 0.6, 14,
                     facecolor=C_MASK_D2, alpha=0.25, edgecolor=C_MASK_D2, lw=1.5))
        ax.text(mx, 8, f"M{i}", fontsize=11, ha="center", color=C_MASK_D2)

    ax.annotate("Real-space\n2.048mm, dx=2.0\u03bcm", xy=(5 + 2 * mask_spacing, -11),
                fontsize=11, ha="center", color=C_MASK_D2,
                bbox=dict(boxstyle="round", facecolor=C_MASK_D2, alpha=0.1))

    ax.plot([focus_x, focus_x], [-8, 8], color=C_FOCUS, lw=3)
    ax.annotate("Focus\nf=4.5mm", xy=(focus_x, -11), fontsize=11, ha="center", color=C_FOCUS)

    ax.add_patch(patches.Rectangle((det_x - 0.5, -4), 1.0, 8,
                 facecolor=C_DET, alpha=0.4, edgecolor=C_DET, lw=2))
    ax.annotate("APD/MMF", xy=(det_x, -8), fontsize=11, ha="center", color=C_DET, fontweight="bold")

    ax.annotate("", xy=(5, 12), xytext=(5 + mask_spacing, 12),
                arrowprops=dict(arrowstyle="<->", color="gray"))
    ax.text(5 + mask_spacing / 2, 12.8, "10mm", fontsize=11, ha="center", color="gray")

    ax.annotate("", xy=(input_x, 14), xytext=(det_x, 14),
                arrowprops=dict(arrowstyle="<->", color="gray"))
    ax.text((input_x + det_x) / 2, 15, "~60mm total", fontsize=12, ha="center", color="gray")

    if highlights:
        for h in highlights:
            ax.text(h["x"], h["y"], h["text"], fontsize=h.get("fs", 13),
                    ha="center", color=h.get("color", "green"), fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor=h.get("bg", "lightgreen"),
                              edgecolor=h.get("color", "green"), alpha=0.9))

    ax.set_xlabel("Optical axis [mm]", fontsize=11)
    ax.axhline(0, color="gray", lw=0.3, ls=":")
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=11)


def draw_results_table(ax, title, rows, baseline_co, highlight_best=True):
    """Draw a results table with rank/name/CO/delta columns."""
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)

    header = ["Rank", "Config", "Test CO", "\u0394 vs baseline", "Status"]
    data = [header]

    for i, r in enumerate(rows):
        name = r["name"]
        co = r["co"]
        if co is not None:
            delta = (co - baseline_co) / baseline_co * 100
            delta_str = f"{delta:+.1f}%"
            co_str = f"{co:.4f}"
            status = "done"
        else:
            delta_str = "--"
            co_str = "--"
            status = "running"
        data.append([str(i + 1), name, co_str, delta_str, status])

    table = ax.table(cellText=data, loc="upper center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 2.0)

    for j in range(len(header)):
        table[0, j].set_facecolor("#34495e")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight best
    if highlight_best and len(rows) > 0:
        valid = [(i, r["co"]) for i, r in enumerate(rows) if r["co"] is not None]
        if valid:
            best_idx = max(valid, key=lambda x: x[1])[0]
            for j in range(len(header)):
                table[best_idx + 1, j].set_facecolor("#abebc6")

    # Highlight failures (CO < baseline)
    for i, r in enumerate(rows):
        if r["co"] is not None and r["co"] < baseline_co:
            for j in range(len(header)):
                if table[i + 1, j].get_facecolor() != (0.67, 0.92, 0.78, 1.0):  # not best
                    table[i + 1, j].set_facecolor("#fadbd8")

    # Baseline line text
    ax.text(0.5, -0.02, f"Baseline CO (no correction) = {baseline_co:.4f}",
            transform=ax.transAxes, fontsize=12, ha="center", color="gray", style="italic")


def draw_info_box(ax, lines):
    """Draw a parameter info box."""
    ax.axis("off")
    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#f8f9fa", edgecolor="#dee2e6"))


# ══════════════════════════════════════════════════════════════════
# Task 1: true_co_only — FD2NN, single baseline_co, 100 epochs
# ══════════════════════════════════════════════════════════════════

def draw_task1():
    run_dir = RUNS_ROOT / "loss_sweep_telescope_true_co_only_20260327" / "baseline_co"
    results = None
    if (run_dir / "results.json").exists():
        results = json.loads((run_dir / "results.json").read_text())

    fig = plt.figure(figsize=(22, 24))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 0.5, 0.6], hspace=0.25)

    # Layout
    draw_fd2nn_layout(fig.add_subplot(gs[0]),
        "Task 1: true_co_only — FD2NN Architecture",
        highlights=[
            {"x": 42, "y": 5, "text": "Beam hits only\n0.015% of mask!", "fs": 13,
             "color": "red", "bg": "lightyellow"},
        ])

    # Info
    ax_info = fig.add_subplot(gs[1])
    ax_info.axis("off")

    co_test = results["complex_overlap"] if results else "?"
    co_bl = results["baseline_co"] if results else "?"
    tp = results["throughput"] if results else "?"
    sr = results["strehl"] if results else "?"

    if results:
        co_str = f"{co_test:.4f}"
        bl_str = f"{co_bl:.4f}"
        imp_str = f"{(co_test - co_bl) / co_bl * 100:+.1f}%"
        tp_str = f"{tp:.4f}"
        sr_str = f"{sr:.2f}"
    else:
        co_str = bl_str = imp_str = tp_str = sr_str = "?"

    info = [
        "=" * 65,
        "  Task 1: true_co_only -- FD2NN, CO-only loss, 100 epochs",
        "=" * 65,
        "",
        "  Purpose:  Full 100-epoch training with baseline_co only",
        "            (previous sweep used 30ep -- check convergence)",
        "",
        "  Architecture:  FD2NN (Fourier-space D2NN)",
        "    Lens:        AC254-025-C x2 (f=25mm, NA=0.508)",
        "    Masks:       5 layers, 1024x1024, dx=18.9um",
        "    Spacing:     5mm inter-layer",
        "    Total:       ~70mm (Lens1 25mm + Masks 20mm + Lens2 25mm)",
        "",
        "  Training:      100 epochs, lr=5e-4, batch=2",
        "  Loss:          complex_overlap only (weight=1.0)",
        "  Data:          telescope 15cm, 75:1 BR, 200 realizations",
        "",
        f"  +---------------- RESULTS -----------------+",
        f"  |  Test CO:     {co_str:<10}                  |",
        f"  |  Baseline:    {bl_str:<10}                  |",
        f"  |  Improvement: {imp_str:<10}                  |",
        f"  |  Throughput:  {tp_str:<10}                  |",
        f"  |  Strehl:      {sr_str:<10}  (!) >1.05       |",
        f"  +-------------------------------------------+",
        "",
        "  Status: COMPLETED",
    ]
    ax_info.text(0.05, 0.95, "\n".join(info), transform=ax_info.transAxes, fontsize=12,
                 verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="#eaf2f8", edgecolor="#5dade2"))

    # Single result "table"
    ax_t = fig.add_subplot(gs[2])
    rows = [{"name": "baseline_co (100ep)", "co": results["complex_overlap"] if results else None}]
    draw_results_table(ax_t, "Result: Single Config", rows,
                       baseline_co=results["baseline_co"] if results else 0.6249,
                       highlight_best=True)

    fig.suptitle("Task 1: true_co_only", fontsize=18, fontweight="bold", y=0.98)
    out_dir = RUNS_ROOT / "single_case_task1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "architecture_true_co_only.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Task 2: d2nn_sweep_telescope — D2NN (free-space), 5 configs
# ══════════════════════════════════════════════════════════════════

def draw_task2():
    run_dir = RUNS_ROOT / "d2nn_sweep_telescope"
    configs = ["baseline_co", "co_amp", "co_ffp", "co_phasor", "roi80"]
    rows = []
    for name in configs:
        rpath = run_dir / name / "results.json"
        if rpath.exists():
            r = json.loads(rpath.read_text())
            rows.append({"name": name, "co": r["complex_overlap"]})
        else:
            rows.append({"name": name, "co": None})

    baseline_co = 0.6249
    for r in rows:
        if r["name"] == "baseline_co" and r["co"] is not None:
            # Get actual baseline from results
            rpath = run_dir / "baseline_co" / "results.json"
            if rpath.exists():
                res = json.loads(rpath.read_text())
                baseline_co = res.get("baseline_co", 0.6249)
            break

    fig = plt.figure(figsize=(22, 28))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 0.5, 0.7], hspace=0.25)

    draw_d2nn_layout(fig.add_subplot(gs[0]),
        "Task 2: d2nn_sweep_telescope — D2NN (Free-space) Architecture",
        highlights=[
            {"x": 21, "y": 5, "text": "Beam covers 100%\nof mask area", "fs": 13,
             "color": "green", "bg": "lightgreen"},
            {"x": 5 + 0.5 * 8, "y": -7, "text": "Spread: 124\u03bcm = 62px",
             "fs": 11, "color": "gray", "bg": "white"},
        ])

    # Info
    ax_info = fig.add_subplot(gs[1])
    ax_info.axis("off")
    n_done = sum(1 for r in rows if r["co"] is not None)

    info = [
        "=" * 65,
        "  Task 2: d2nn_sweep_telescope -- Plain D2NN, 5 configs",
        "=" * 65,
        "",
        "  Purpose:  Avoid FD2NN Fourier under-resolution problem",
        "            by operating directly in real-space",
        "",
        "  Architecture:  D2NN (Free-space, Angular Spectrum)",
        "    No lens:     phase masks operate directly in real-space",
        "    Masks:       5 layers, 1024x1024, dx=2.0um",
        "    Spacing:     10mm inter-layer (AS propagation)",
        "    Total:       ~60mm (5x10mm + 10mm detector)",
        "",
        "  Training:      100 epochs, lr=5e-4, batch=2",
        "  Data:          telescope 15cm, 75:1 BR, 200 realizations",
        "",
        "  Key advantage: ALL 5.2M parameters are active",
        "    (FD2NN uses only ~800 px near beam center)",
        "",
        f"  Status: {n_done}/{len(configs)} completed",
    ]
    ax_info.text(0.05, 0.95, "\n".join(info), transform=ax_info.transAxes, fontsize=12,
                 verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="#eafaf1", edgecolor="#27ae60"))

    # Results
    ax_t = fig.add_subplot(gs[2])
    draw_results_table(ax_t, "D2NN Sweep Results (Telescope Data)", rows,
                       baseline_co=baseline_co)

    fig.suptitle("Task 2: d2nn_sweep_telescope", fontsize=18, fontweight="bold", y=0.98)
    out_dir = RUNS_ROOT / "single_case_task2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "architecture_d2nn_sweep_telescope.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Task 3: loss_sweep_telescope — FD2NN, 10 configs
# ══════════════════════════════════════════════════════════════════

def draw_task3():
    run_dir = RUNS_ROOT / "loss_sweep_telescope"
    configs = [
        "baseline_co", "co_phasor", "co_ffp", "roi80", "co_soft_phasor_g2",
        "baseline_co_amp", "phasor_only", "co_phasor_strong", "co_ffp_strong", "roi80_ph1",
    ]
    rows = []
    for name in configs:
        rpath = run_dir / name / "results.json"
        if rpath.exists():
            r = json.loads(rpath.read_text())
            rows.append({"name": name, "co": r["complex_overlap"]})
        else:
            rows.append({"name": name, "co": None})

    baseline_co = 0.6249
    rpath = run_dir / "baseline_co" / "results.json"
    if rpath.exists():
        res = json.loads(rpath.read_text())
        baseline_co = res.get("baseline_co", 0.6249)

    fig = plt.figure(figsize=(22, 32))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 0.5, 1.0], hspace=0.25)

    draw_fd2nn_layout(fig.add_subplot(gs[0]),
        "Task 3: loss_sweep_telescope — FD2NN Architecture",
        highlights=[
            {"x": 42, "y": 5, "text": "Beam hits only\n0.015% of mask!", "fs": 13,
             "color": "red", "bg": "lightyellow"},
        ])

    # Info
    ax_info = fig.add_subplot(gs[1])
    ax_info.axis("off")
    n_done = sum(1 for r in rows if r["co"] is not None)

    info = [
        "=" * 65,
        "  Task 3: loss_sweep_telescope -- FD2NN, 10 loss configs",
        "=" * 65,
        "",
        "  Purpose:  Sweep loss functions to find which works best",
        "            for FD2NN beam cleanup on telescope data",
        "",
        "  Architecture:  FD2NN (Fourier-space D2NN)",
        "    Lens:        AC254-025-C x2 (f=25mm, NA=0.508)",
        "    Masks:       5 layers, 1024x1024, dx=18.9um",
        "    Spacing:     5mm inter-layer",
        "    Total:       ~70mm",
        "",
        "  Training:      100 epochs, lr=5e-4, batch=2",
        "  Data:          telescope 15cm, 75:1 BR, 200 realizations",
        "",
        "  Known issue:   Fourier beam spot (12um) < pixel (18.9um)",
        "    -> ROI-based losses likely degenerate",
        "    -> Phase correction losses may help",
        "",
        f"  Status: {n_done}/{len(configs)} completed",
    ]
    ax_info.text(0.05, 0.95, "\n".join(info), transform=ax_info.transAxes, fontsize=12,
                 verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="#eaf2f8", edgecolor="#5dade2"))

    # Results — sort by CO descending
    rows_sorted = sorted(rows, key=lambda r: r["co"] if r["co"] is not None else -1, reverse=True)
    ax_t = fig.add_subplot(gs[2])
    draw_results_table(ax_t, "FD2NN Loss Sweep Results (Telescope Data)", rows_sorted,
                       baseline_co=baseline_co)

    fig.suptitle("Task 3: loss_sweep_telescope", fontsize=18, fontweight="bold", y=0.98)
    out_dir = RUNS_ROOT / "single_case_task3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "architecture_loss_sweep_telescope.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    draw_task1()
    draw_task2()
    draw_task3()
    print("\nAll 3 task architecture diagrams generated.")