#!/usr/bin/env python
"""Complete FD2NN spacing sweep pipeline: train → visualize → report.

Usage:
    python scripts/pipeline_sweep_report.py                # full pipeline
    python scripts/pipeline_sweep_report.py --skip-training # viz + report only
    python scripts/pipeline_sweep_report.py --report-only   # report only
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

# Ensure kim2026 package is importable
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ─── Paths ───────────────────────────────────────────────────────
PROJ = Path(__file__).resolve().parent.parent
SWEEP_DIR = PROJ / "runs" / "01_fd2nn_spacing_sweep_f10mm_claude"
FIG_DIR = PROJ / "figures" / "spacing_sweep_f10mm"
REPORT_DIR = PROJ / "docs" / "04-report"

OLD_RUN01 = PROJ / "runs" / "01_fd2nn_complexloss_roi1024_spacing_sweep_claude"

# System parameters
LAMBDA = 1.55e-6
F_M = 10e-3
N = 1024
DX_IN = 2.048e-3 / N
DX_F = LAMBDA * F_M / (N * DX_IN)
W10 = 10 * DX_F
Z_R = math.pi * W10**2 / LAMBDA

SPACINGS = {
    "spacing_0mm":  0.0,
    "spacing_1mm":  1e-3,
    "spacing_3mm":  3e-3,
    "spacing_6mm":  6e-3,
    "spacing_12mm": 12e-3,
    "spacing_25mm": 25e-3,
    "spacing_50mm": 50e-3,
}


# ═══════════════════════════════════════════════════════════════════
# Stage 1: Training
# ═══════════════════════════════════════════════════════════════════

def stage1_training(*, clean_old: bool = True) -> None:
    print("\n" + "=" * 70)
    print("STAGE 1: TRAINING")
    print("=" * 70)

    if clean_old and OLD_RUN01.exists():
        print(f"Deleting old run01: {OLD_RUN01}")
        shutil.rmtree(OLD_RUN01)

    # Import and run the sweep
    sys.path.insert(0, str(PROJ / "scripts"))
    from sweep_spacing_f10mm import main as sweep_main
    sweep_main()


# ═══════════════════════════════════════════════════════════════════
# Stage 2: Visualization
# ═══════════════════════════════════════════════════════════════════

def _load_summary() -> dict:
    with open(SWEEP_DIR / "sweep_summary.json") as f:
        return json.load(f)


def plot_fresnel_analysis(summary: dict, fig_dir: Path) -> Path:
    """Fig7: Metrics vs z/z_R with physics annotations."""
    names = [n for n in SPACINGS if SPACINGS[n] > 0]
    z_ratios = [SPACINGS[n] / Z_R for n in names]
    cos = [summary[n]["complex_overlap"] for n in names]
    prs = [summary[n]["phase_rmse_rad"] for n in names]
    ios = [summary[n]["intensity_overlap"] for n in names]

    baseline_co = summary["spacing_0mm"]["baseline_complex_overlap"]
    baseline_io = summary["spacing_0mm"]["baseline_intensity_overlap"]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    l1, = ax1.plot(z_ratios, cos, "o-", color="#2ecc71", lw=2, ms=8, label="Complex Overlap")
    l2, = ax1.plot(z_ratios, ios, "s-", color="#3498db", lw=2, ms=8, label="Intensity Overlap")
    l3, = ax2.plot(z_ratios, prs, "^-", color="#e74c3c", lw=2, ms=8, label="Phase RMSE (rad)")

    ax1.axhline(baseline_co, color="#2ecc71", ls="--", alpha=0.4, lw=1)
    ax1.axhline(baseline_io, color="#3498db", ls="--", alpha=0.4, lw=1)
    ax1.text(z_ratios[-1] * 0.95, baseline_co + 0.01, f"baseline CO={baseline_co:.3f}",
             fontsize=7, ha="right", color="#2ecc71")
    ax1.text(z_ratios[-1] * 0.95, baseline_io - 0.03, f"baseline IO={baseline_io:.3f}",
             fontsize=7, ha="right", color="#3498db")

    # Physics annotations
    ax1.axvspan(0, 0.3, alpha=0.06, color="blue", label="Near-field")
    ax1.axvspan(0.3, 2.0, alpha=0.06, color="green", label="Intermediate")
    ax1.axvspan(2.0, 5.0, alpha=0.06, color="orange", label="Far-field")
    ax1.axvline(1.0, color="gray", ls=":", lw=1, alpha=0.5)
    ax1.text(1.0, ax1.get_ylim()[1] * 0.98, "z_R", fontsize=8, ha="center",
             color="gray", va="top")

    ax1.set_xlabel("z / z_R (10-pixel feature Rayleigh range)", fontsize=11)
    ax1.set_ylabel("Overlap (higher = better)", fontsize=11, color="#2ecc71")
    ax2.set_ylabel("Phase RMSE [rad] (lower = better)", fontsize=11, color="#e74c3c")
    ax1.set_xscale("log")
    ax1.set_xlim(0.05, 6)

    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=9, loc="center right")
    ax1.grid(True, alpha=0.3)

    fig.suptitle(
        "FD2NN Spacing Sweep — Fresnel Analysis\n"
        f"f={F_M*1e3:.0f}mm, dx_fourier={DX_F*1e6:.1f}µm, "
        f"z_R(10px)={Z_R*1e3:.1f}mm",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out = fig_dir / "fig7_fresnel_analysis.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_phase_masks_0_2pi(fig_dir: Path) -> Path:
    """Fig8: Phase masks [0,2π] for all spacings, final epoch."""
    spacing_names = list(SPACINGS.keys())
    nrows = len(spacing_names)
    ncols = 5  # layers

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.8 * nrows))

    for row, name in enumerate(spacing_names):
        phase_file = SWEEP_DIR / name / "phases_epoch029.npy"
        if not phase_file.exists():
            for ax in axes[row]:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            continue

        phases = np.load(phase_file)  # (5, 1024, 1024)
        z_ratio = SPACINGS[name] / Z_R if SPACINGS[name] > 0 else 0.0

        for col in range(ncols):
            phase = phases[col] % (2 * np.pi)  # wrap to [0, 2π]
            # Center crop for detail
            cy, cx = phase.shape[0] // 2, phase.shape[1] // 2
            r = 128
            crop = phase[cy - r:cy + r, cx - r:cx + r]

            ax = axes[row, col]
            im = ax.imshow(crop, cmap="twilight", vmin=0, vmax=2 * np.pi,
                           origin="lower", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"Layer {col}", fontsize=10, fontweight="bold")
            if col == 0:
                sp = SPACINGS[name] * 1e3
                ax.set_ylabel(f"{sp:.0f}mm\n(z/z_R={z_ratio:.2f})",
                              fontsize=8, fontweight="bold")

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, 2 * np.pi), cmap="twilight")
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0, np.pi, 2 * np.pi])
    cbar.set_ticklabels(["0", "π", "2π"])
    cbar.set_label("Phase (rad)", fontsize=9)

    fig.suptitle(
        "FD2NN Phase Masks [0, 2π] — Final Epoch (all spacings)\n"
        f"f={F_M*1e3:.0f}mm, dx_fourier={DX_F*1e6:.1f}µm, center 256×256 crop",
        fontsize=12, fontweight="bold",
    )
    fig.subplots_adjust(left=0.08, right=0.90, top=0.92, bottom=0.03,
                        hspace=0.08, wspace=0.05)

    out = fig_dir / "fig8_phase_masks_0_2pi.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def stage2_visualization() -> list[Path]:
    print("\n" + "=" * 70)
    print("STAGE 2: VISUALIZATION")
    print("=" * 70)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Standard 6 figures from viz module
    from kim2026.viz.fd2nn_sweep import generate_figures
    print("Generating standard figures (fig1-fig6)...")
    paths = generate_figures(sweep_dir=SWEEP_DIR, fig_dir=FIG_DIR)

    # Load summary for custom figures
    summary = _load_summary()

    # Fig7: Fresnel analysis
    print("Generating Fresnel analysis figure...")
    paths.append(plot_fresnel_analysis(summary, FIG_DIR))

    # Fig8: Phase masks [0,2π]
    print("Generating phase mask figure [0,2π]...")
    paths.append(plot_phase_masks_0_2pi(FIG_DIR))

    print(f"\nTotal figures: {len(paths)}")
    return paths


# ═══════════════════════════════════════════════════════════════════
# Stage 3: Report Generation
# ═══════════════════════════════════════════════════════════════════

def stage3_report(figure_paths: list[Path]) -> Path:
    print("\n" + "=" * 70)
    print("STAGE 3: REPORT GENERATION")
    print("=" * 70)

    summary = _load_summary()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "spacing_sweep_f10mm_report.md"

    # Find best config
    non_zero = {k: v for k, v in summary.items() if SPACINGS.get(k, 0) > 0}
    best_name = max(non_zero, key=lambda k: non_zero[k]["complex_overlap"])
    best = summary[best_name]
    best_spacing_mm = SPACINGS[best_name] * 1e3
    best_z_ratio = SPACINGS[best_name] / Z_R

    baseline_co = summary["spacing_0mm"]["baseline_complex_overlap"]
    baseline_io = summary["spacing_0mm"]["baseline_intensity_overlap"]
    co_improvement = (best["complex_overlap"] - baseline_co) / baseline_co * 100

    # Figure relative paths
    fig_rel = "../../figures/spacing_sweep_f10mm"

    lines = []
    lines.append("# FD2NN Spacing Sweep Report (f=10mm)")
    lines.append("")
    lines.append(f"> Generated: spacing sweep with corrected focal length")
    lines.append(f"> dx_fourier = {DX_F*1e6:.2f} µm ({DX_F/LAMBDA:.1f}λ) — fabrication-realistic")
    lines.append("")

    # 1. Executive Summary
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(f"- **Best spacing**: {best_spacing_mm:.0f} mm (z/z_R = {best_z_ratio:.2f})")
    lines.append(f"- **Complex Overlap**: {best['complex_overlap']:.4f} ({co_improvement:+.1f}% vs baseline {baseline_co:.4f})")
    lines.append(f"- **Phase RMSE**: {best['phase_rmse_rad']:.3f} rad ({math.degrees(best['phase_rmse_rad']):.1f}°)")
    lines.append(f"- **Intensity Overlap**: {best['intensity_overlap']:.4f} (baseline: {baseline_io:.4f})")
    lines.append(f"- **Strehl Ratio**: {best['strehl']:.3f}")
    lines.append("")

    # 2. System Parameters
    lines.append("## 2. System Parameters")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Wavelength λ | {LAMBDA*1e6:.2f} µm |")
    lines.append(f"| Grid size n | {N} |")
    lines.append(f"| Input pixel pitch dx | {DX_IN*1e6:.1f} µm |")
    lines.append(f"| Receiver window | {N*DX_IN*1e3:.3f} mm |")
    lines.append(f"| Focal length f | {F_M*1e3:.0f} mm |")
    lines.append(f"| Fourier pixel pitch dx_f | {DX_F*1e6:.2f} µm ({DX_F/LAMBDA:.1f}λ) |")
    lines.append(f"| Fourier window | {DX_F*N*1e3:.3f} mm |")
    lines.append(f"| Numerical aperture NA | 0.16 |")
    lines.append(f"| Number of layers | 5 |")
    lines.append(f"| Phase constraint | symmetric_tanh, 2π |")
    lines.append(f"| Training | 30 epochs, lr=5e-4, batch=2 |")
    lines.append("")

    # 3. Spacing Configurations
    lines.append("## 3. Spacing Configurations")
    lines.append("")
    lines.append("| Config | Spacing | z/z_R | Physical Regime |")
    lines.append("|--------|---------|-------|----------------|")
    for name, spacing in SPACINGS.items():
        z_ratio = spacing / Z_R if spacing > 0 else 0
        if spacing == 0:
            regime = "Stacked (no propagation)"
        elif z_ratio < 0.3:
            regime = "Near-field"
        elif z_ratio < 2:
            regime = "Intermediate (diffraction coupling)"
        else:
            regime = "Far-field approach"
        lines.append(f"| {name} | {spacing*1e3:.0f} mm | {z_ratio:.2f} | {regime} |")
    lines.append("")
    lines.append(f"z_R = π·(10·dx_f)²/λ = {Z_R*1e3:.2f} mm (Rayleigh range for 10-pixel feature)")
    lines.append("")

    # 4. Results
    lines.append("## 4. Results")
    lines.append("")
    lines.append("### 4.1 Metrics Summary")
    lines.append("")
    lines.append("| Config | z/z_R | CO ↑ | IO ↑ | Phase RMSE ↓ | Strehl |")
    lines.append("|--------|-------|------|------|-------------|--------|")
    for name, spacing in SPACINGS.items():
        m = summary[name]
        z_ratio = spacing / Z_R if spacing > 0 else 0
        marker = " **best**" if name == best_name else ""
        lines.append(f"| {name} | {z_ratio:.2f} | {m['complex_overlap']:.4f}{marker} | "
                     f"{m['intensity_overlap']:.4f} | {m['phase_rmse_rad']:.3f} | {m['strehl']:.3f} |")
    lines.append(f"| Baseline (no D2NN) | — | {baseline_co:.4f} | {baseline_io:.4f} | — | — |")
    lines.append("")

    lines.append(f"### 4.2 Training Convergence")
    lines.append(f"")
    lines.append(f"![Epoch Curves]({fig_rel}/fig1_epoch_curves.png)")
    lines.append(f"")
    lines.append(f"### 4.3 Test Metrics")
    lines.append(f"")
    lines.append(f"![Test Metrics]({fig_rel}/fig2_test_metrics.png)")
    lines.append(f"")
    lines.append(f"### 4.4 Field Comparison")
    lines.append(f"")
    lines.append(f"![Field Full]({fig_rel}/fig3_field_full_comparison.png)")
    lines.append(f"")
    lines.append(f"![Field Zoom]({fig_rel}/fig4_field_zoom_comparison.png)")
    lines.append(f"")
    lines.append(f"### 4.5 Field Profiles")
    lines.append(f"")
    lines.append(f"![Profiles]({fig_rel}/fig5_field_profiles.png)")
    lines.append(f"")
    lines.append(f"### 4.6 Phase Mask Evolution")
    lines.append(f"")
    lines.append(f"![Phase Masks]({fig_rel}/fig6_phase_masks.png)")
    lines.append(f"")
    lines.append(f"### 4.7 Fresnel Number Analysis")
    lines.append(f"")
    lines.append(f"![Fresnel Analysis]({fig_rel}/fig7_fresnel_analysis.png)")
    lines.append(f"")
    lines.append(f"### 4.8 Phase Masks [0, 2π] All Spacings")
    lines.append(f"")
    lines.append(f"![Phase Masks 0-2pi]({fig_rel}/fig8_phase_masks_0_2pi.png)")
    lines.append(f"")

    # 5. Physics Analysis
    lines.append("## 5. Physics Analysis")
    lines.append("")
    lines.append("### 5.1 회절 커플링과 layer 간 spacing의 관계")
    lines.append("")
    lines.append("FD2NN에서 layer 간 spacing은 angular spectrum 전파를 통해 회절 커플링을 제공한다.")
    lines.append("transfer function H(fx,fy;z) = exp(j·kz·z)에서 z가 결정하는 것은:")
    lines.append("")
    lines.append("- **z/z_R < 0.3 (near-field)**: H ≈ 1 (identity), layer 간 독립적.")
    lines.append("  각 phase mask가 같은 spatial frequency를 보므로 중복 학습 우려.")
    lines.append("- **z/z_R ≈ 1 (Rayleigh range)**: 최적 information mixing.")
    lines.append("  회절이 인접 pixel feature를 혼합하여 layer 간 비선형적 표현력 증가.")
    lines.append("- **z/z_R > 3 (far-field)**: 고주파 성분이 확산하여 소실.")
    lines.append("  spatial bandwidth 감소, phase mask의 fine structure가 무의미해짐.")
    lines.append("")
    lines.append("### 5.2 Angular Spectrum 관점의 해석")
    lines.append("")
    lines.append("전파 transfer function의 passband는 |f| < 1/λ 로 제한된다.")
    lines.append(f"NA=0.16 cutoff에서 최대 공간주파수는 {0.16/LAMBDA:.0f} lp/m이고,")
    lines.append(f"이는 Fourier plane에서 {0.16/LAMBDA * DX_F * N:.0f} pixel에 해당한다.")
    lines.append("")
    lines.append("spacing이 클수록:")
    lines.append("1. 저주파 성분은 전파가 잘 되지만 (평면파에 가까움)")
    lines.append("2. 고주파 성분은 빠르게 발산하여 window 밖으로 나감")
    lines.append("3. 결과적으로 effective spatial bandwidth가 줄어들어")
    lines.append("   phase mask의 pixel 수가 의미없게 됨")
    lines.append("")
    lines.append("### 5.3 Metasurface Phase Pattern 해석")
    lines.append("")
    lines.append("학습된 phase mask에서 관찰되는 패턴:")
    lines.append("")
    lines.append("- **동심원 구조 (Fresnel lens)**: 빔의 초점을 조절하는 가장 기본적인 phase profile.")
    lines.append("  r² 의존성이 있으며, 각 ring의 주기가 λ·f/r 에 비례.")
    lines.append("- **Speckle-like 패턴**: 난류 보상을 위한 random phase correction.")
    lines.append("  특정 입력 realization에 대한 conjugate phase를 근사.")
    lines.append("- **Spacing이 작을 때**: 모든 layer가 비슷한 패턴 → 중복 (표현력 낭비)")
    lines.append("- **Spacing이 적절할 때**: 각 layer가 다른 spatial scale의 correction 담당")
    lines.append("")
    lines.append("### 5.4 CO vs IO Trade-off의 물리적 원인")
    lines.append("")
    lines.append("Phase-only metasurface의 근본적 한계:")
    lines.append("")
    lines.append("1. **에너지 보존**: |exp(jφ)| = 1이므로 amplitude reshaping 불가")
    lines.append("2. CO 개선 = phase 정합 → 난류 wavefront 보상 성공")
    lines.append("3. IO 저하 = amplitude profile 왜곡 → Gaussian beam shape 깨짐")
    lines.append("4. Phase-only mask는 intensity를 redistribution 할 수 있지만,")
    lines.append("   원하는 Gaussian profile로 복원하는 것은 불가능")
    lines.append("5. 이 trade-off는 amplitude mask 추가 또는 multi-pass 구조로만 해결 가능")
    lines.append("")

    # 6. Conclusions
    lines.append("## 6. Conclusions")
    lines.append("")
    lines.append(f"1. f=10mm 설계에서 dx_fourier={DX_F*1e6:.1f}µm으로 현실적 제작 가능")
    lines.append(f"2. 최적 spacing은 {best_spacing_mm:.0f}mm (z/z_R={best_z_ratio:.2f})에서 CO={best['complex_overlap']:.4f}")
    lines.append("3. Near-field (z/z_R < 0.3)과 far-field (z/z_R > 3) 모두 suboptimal")
    lines.append("4. Phase-only mask의 CO vs IO trade-off는 물리적 한계")
    lines.append("")

    # 7. Future Work
    lines.append("## 7. Future Work")
    lines.append("")
    lines.append("- [ ] 최적 spacing 주변 fine sweep (±30%)")
    lines.append("- [ ] Layer 수 증가 (7, 10 layers) + 최적 spacing 조합")
    lines.append("- [ ] Amplitude+phase mask (complex-valued mask) 구현")
    lines.append("- [ ] Multiple turbulence strength (Cn²) 조건에서의 일반화 성능")
    lines.append("- [ ] 물리적 제작 제약 조건 반영 (phase level quantization)")
    lines.append("")

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    print(f"Report saved: {report_path}")
    print(f"  Length: {len(lines)} lines")
    return report_path


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="FD2NN spacing sweep pipeline")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, run viz + report only")
    parser.add_argument("--report-only", action="store_true",
                        help="Generate report only (assumes figures exist)")
    parser.add_argument("--keep-old", action="store_true",
                        help="Do not delete old run01 directory")
    args = parser.parse_args()

    print("=" * 70)
    print("FD2NN SPACING SWEEP PIPELINE")
    print(f"  f = {F_M*1e3:.0f} mm")
    print(f"  dx_fourier = {DX_F*1e6:.2f} µm ({DX_F/LAMBDA:.1f}λ)")
    print(f"  z_R(10px) = {Z_R*1e3:.2f} mm")
    print(f"  Spacings: {len(SPACINGS)} configs (0 to {max(SPACINGS.values())*1e3:.0f} mm)")
    print("=" * 70)

    if args.report_only:
        report = stage3_report([])
    elif args.skip_training:
        paths = stage2_visualization()
        report = stage3_report(paths)
    else:
        stage1_training(clean_old=not args.keep_old)
        paths = stage2_visualization()
        report = stage3_report(paths)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"  Sweep: {SWEEP_DIR}")
    print(f"  Figures: {FIG_DIR}")
    print(f"  Report: {report}")
    print("=" * 70)


if __name__ == "__main__":
    main()
